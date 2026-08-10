"""Local training script mirroring the Colab notebook.

Training data = controlled digit sources:
  * curated print-font synthetic digits (1-9) - no deps
  * MNIST handwriting (1-9) - optional, needs torchvision
  * EMNIST handwriting (1-9) - optional, needs torchvision
  * synthetic empty cells (class 0) - blank noisy cells through preprocess_cell
  * REAL PHOTOS (--include-photo): Lexski/sudoku-image-recognition, split
    70/15/15 by photo_data.py into a SEALED photo_splits.json. The train and
    val partitions feed the CNN (definite cells only - candidate-only cells
    are skipped, exactly like the benchmark metrics); the TEST partition is
    never loaded here - evaluate it with `benchmark.py --split-file`.

    python train_local.py --epochs 40 --empty-per-class 15000 --include-synth
        --include-mnist --include-emnist --include-photo
        --benchmark-data-dir path/to/sudoku_dataset

Model = double-convolution DigitCNN (32->64->128, 2 convs per stage, GAP
head); optimizer = AdamW(lr=3e-4, weight_decay=1e-4) + ReduceLROnPlateau;
conservative augmentation (rotation +/-8 deg, shift +/-2 px, weak noise,
occasional blur, brightness/contrast). The BEST epoch (by mean of per-source
validation accuracies) is saved to --out (default digit_cnn.pth) - never the
last epoch. A TEMPERATURE-scaling factor is then fitted on the validation
split (photo val when present - the distribution the solver's confidence
gate cares about) and saved next to the weights as
<out>.temperature.json. Use --device auto|cpu|cuda and --batch-size 256
(1024 was the old default; 256 works on both CPU and GPU).
"""
import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn

import digit_cnn
from digit_cnn import DigitCNN, make_empty_cells

SEED = 0


def load_mnist(root="./mnist", target=48, n_per_class=None, seed=0):
    """MNIST digits 1-9, inverted and passed through preprocess_cell (like
    the notebook). Digit 0 is dropped: class 0 means EMPTY cell.
    n_per_class caps per-class (seeded shuffle) for smoke runs.

    The OFFICIAL TEST SET IS KEPT SEPARATE: returns (X, y, Xt, yt) so the
    training pool never sees test pixels (the old concatenation leaked them
    into the random split).
    """
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    def pack(ds):
        X, y = [], []
        for img, label in ds:
            if int(label) == 0:
                continue
            arr = img.numpy()[0]
            arr = 1.0 - arr
            arr = cv2.resize(arr, (56, 56), interpolation=cv2.INTER_CUBIC)
            arr = np.clip(arr, 0.0, 1.0)
            X.append(digit_cnn.preprocess_cell((arr * 255.0).astype(np.uint8),
                                               target=target))
            y.append(int(label))
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)

    def cap(X, y):
        if n_per_class:
            rng = np.random.default_rng(seed)
            keep = []
            for k in np.unique(y):
                idx = np.where(y == k)[0]
                rng.shuffle(idx)
                keep.append(idx[:n_per_class])
            sel = np.concatenate(keep)
            return X[sel], y[sel]
        return X, y

    train = datasets.MNIST(root=root, train=True, download=True, transform=ToTensor())
    test = datasets.MNIST(root=root, train=False, download=True, transform=ToTensor())
    X, y = cap(*pack(train))
    Xt, yt = cap(*pack(test))
    return X, y, Xt, yt


def split_stratified(y, train_frac=0.85, val_frac=0.15, seed=0):
    """Stratified train/validation split (the OFFICIAL test set is never
    split here - it stays untouched for final evaluation)."""
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * train_frac))
        n_va = int(round(len(idx) * val_frac))
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
    return np.concatenate(tr), np.concatenate(va)


def augment_cell(img, np_rng):
    """CONSERVATIVE augmentation (real puzzle cells are fronto-parallel after
    the grid warp): rotation +/-8 deg, shift +/-2 px, small scale variation,
    weak Gaussian noise, occasional light blur, brightness/contrast lighting.
    """
    im = img[..., 0].copy()
    if np_rng.random() < 0.7:
        im = digit_cnn.rotate(im, np_rng.uniform(-8, 8))
        im = digit_cnn.center_crop_or_pad(im, 48)
    if np_rng.random() < 0.4:                     # small scale variation
        s = np_rng.uniform(0.9, 1.0)
        im = cv2.resize(im, (int(48 * s), int(48 * s)), interpolation=cv2.INTER_AREA)
        im = digit_cnn.center_crop_or_pad(im, 48)
    dx, dy = int(np_rng.uniform(-2, 3)), int(np_rng.uniform(-2, 3))
    im = digit_cnn.translate(im, dx, dy)
    if np_rng.random() < 0.6:                     # lighting: brightness/contrast
        gain = np_rng.uniform(0.75, 1.25)
        bias = np_rng.uniform(-0.08, 0.08)
        im = np.clip(im * gain + bias, 0.0, 1.0)
    if np_rng.random() < 0.25:                    # weak sensor noise
        im = im + np_rng.normal(0, np_rng.uniform(0.004, 0.012), im.shape).astype(np.float32)
    if np_rng.random() < 0.2:                     # occasional light blur
        im = cv2.GaussianBlur(im, (3, 3), 0)
    return np.clip(im, 0, 1).astype(np.float32)[..., None]


class AugmentedDataset(torch.utils.data.Dataset):
    """Wraps (X, y) and applies `augment_cell` ON THE FLY per access.

    Replaces the old `np.stack([augment_cell(x, rng) for x in X_tr])` eager
    copy (~2.2 GB for a 250k-sample run); augmentation is also now applied
    FRESH every epoch instead of once before training. Not safe for
    num_workers > 0 (shared RNG is fine single-threaded)."""

    def __init__(self, X, y, seed=0):
        self.X, self.y = X, y
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return augment_cell(self.X[i], self.rng), int(self.y[i])


def train(X_tr, y_tr, src_tr, X_va, y_va, src_va, epochs, batch_size, lr,
          device, out_path, ckpt_path=None, resume=False, init_state=None,
          early_stop_patience=5, min_epochs=12):
    """Train with the same semantics as the notebook's train_digit_model:
    lazy per-epoch augmentation, source-balanced sampling, per-epoch
    checkpoints, best-epoch save, and early stopping (never before
    `min_epochs` - the first ReduceLROnPlateau drops need a few flat epochs).

    `init_state` warm-starts from a pretrained state dict (fine-tuning a
    synthetic checkpoint on real photos) - pass a low --lr (e.g. 1e-4) so the
    clean digit knowledge is adapted, not overwritten.
    """
    torch.manual_seed(SEED)
    model = DigitCNN().to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
        print(f"warm-started from {len(init_state)} tensors (fine-tune, lr={lr})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=4, min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    sources = sorted(set(src_va))

    best_score, best_state = -1.0, None
    start_ep = 0
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state"])
        start_ep = ck["epoch"] + 1
        best_score = ck["best_score"]
        best_state = ck["best_state"]
        print(f"resumed from {ckpt_path} (epoch {ck['epoch'] + 1})")

    # Source-balanced sampling: EMNIST (~69% of the pool) no longer dominates
    # the gradient; each source contributes equally per epoch.
    src_counts = {s: int((src_tr == s).sum()) for s in np.unique(src_tr)}
    weights = np.array([1.0 / src_counts[s] for s in src_tr], dtype=np.float64)
    dataset = AugmentedDataset(X_tr, y_tr)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(X_tr), replacement=True)

    epochs_no_improve = 0
    for ep in range(start_ep, epochs):
        model.train()
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=sampler, drop_last=False)
        tot = cor = 0
        loss_sum = 0.0
        for xb_np, yb_np in dl:
            xb = xb_np.permute(0, 3, 1, 2).to(device)
            yb = yb_np.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            tot += len(xb)
            cor += (out.argmax(1) == yb).sum().item()
        tl, ta = loss_sum / tot, cor / tot

        model.eval()
        with torch.no_grad():
            src_cor = {s: 0 for s in sources}
            src_n = {s: 0 for s in sources}
            for i in range(0, len(X_va), batch_size):
                xb = torch.from_numpy(X_va[i:i + batch_size]).permute(0, 3, 1, 2).to(device)
                yb = torch.from_numpy(y_va[i:i + batch_size]).to(device)
                out = model(xb)
                preds = out.argmax(1)
                for j in range(len(preds)):
                    s = src_va[i + j]
                    src_cor[s] += int(preds[j] == yb[j])
                    src_n[s] += 1
            va_agg = sum(src_cor.values()) / len(X_va)
            src_acc = {s: src_cor[s] / max(src_n[s], 1) for s in sources}
            va = float(np.mean(list(src_acc.values())))
        sched.step(va)
        print(f"epoch {ep + 1}/{epochs}  train acc {ta:.4f} | val agg {va_agg:.4f} "
              f"| score {va:.4f} | " + " ".join(f"{s}={src_acc[s]:.3f}" for s in sources))
        if va > best_score:
            best_score = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if ep + 1 >= min_epochs and epochs_no_improve >= early_stop_patience:
                print(f"early stopping at epoch {ep + 1}: score has not improved "
                      f"for {early_stop_patience} epochs (best {best_score:.4f})")
                break
        if ckpt_path:
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save({"epoch": ep,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state": sched.state_dict(),
                        "best_score": best_score,
                        "best_state": best_state},
                       ckpt_path)

    model.load_state_dict(best_state)
    model.eval()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved BEST epoch (score {best_score:.4f}) to {out_path}")
    return model


def main():
    ap = argparse.ArgumentParser(description="Local CNN training (well-known datasets only)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--empty-per-class", type=int, default=15000,
                    help="synthetic EMPTY-cell (class 0) samples")
    ap.add_argument("--include-mnist", action="store_true",
                    help="add MNIST 1-9 (needs torchvision; skipped if missing)")
    ap.add_argument("--mnist-per-class", type=int, default=0,
                    help="per-class cap for MNIST (0 = all ~7.7k/class)")
    ap.add_argument("--include-emnist", action="store_true",
                    help="add EMNIST handwritten digits 1-9 (needs torchvision)")
    ap.add_argument("--emnist-per-class", type=int, default=0,
                    help="per-class cap for EMNIST (0 = uncapped; auto-capped "
                         "to 10000 on CPU so local runs stay feasible)")
    ap.add_argument("--include-synth", action="store_true",
                    help="add curated print-font synthetic digits 1-9 (no deps)")
    ap.add_argument("--synth-per-class", type=int, default=4000)
    ap.add_argument("--include-photo", action="store_true",
                    help="add REAL photos (Lexski, sealed 70/15/15 split): the "
                         "train+val partitions of --photo-split-file; the test "
                         "partition is SEALED and never loaded here")
    ap.add_argument("--photo-dir", default="data",
                    help="root folder with Lexski data/<split> subfolders "
                         "(images/ + metadata.jsonl)")
    ap.add_argument("--photo-split-file", default="photo_splits.json",
                    help="sealed partition file from photo_data.py "
                         "(build it once with `python -c \"import photo_data; "
                         "photo_data.build_splits()\"`)")
    ap.add_argument("--warm-start", default=None,
                    help="pretrained .pth to fine-tune from (e.g. the synthetic "
                         "checkpoint); pass a low --lr like 1e-4")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="models/digit_cnn.pth")
    ap.add_argument("--ckpt", default="models/train_ckpt.pt",
                    help="per-epoch checkpoint path (model+optimizer+best state)")
    ap.add_argument("--resume", action="store_true",
                    help="resume training from the --ckpt file")
    ap.add_argument("--early-stop-patience", type=int, default=5,
                    help="stop once the validation score has not improved for "
                         "this many epochs (notebook parity)")
    ap.add_argument("--min-epochs", type=int, default=12,
                    help="never early-stop before this epoch (notebook parity)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print dataset sizes + estimated memory, then exit")
    ap.add_argument("--benchmark-data-dir", default=None,
                    help="run the labeled benchmark on this dataset after training")
    ap.add_argument("--benchmark-frac", type=float, default=0.5)
    ap.add_argument("--benchmark-out", default="results/figures/benchmark.png")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("device:", device)
    if device == "cpu":
        print("note: torch.cuda.is_available() is False in this torch build - "
              "install the CUDA wheels for your RTX 3060 Ti to train on GPU")

    print("generating empty cells (class 0)...")
    X_empty, y_empty = make_empty_cells(n_per_class=args.empty_per_class, seed=SEED)
    X_synth = np.zeros((0, 48, 48, 1), dtype=np.float32)
    y_synth = np.zeros((0,), dtype=np.int64)
    if args.include_synth:
        X_synth, y_synth = digit_cnn.make_synthetic_digits(
            n_per_class=args.synth_per_class, seed=SEED)
    X_mnist = np.zeros((0, 48, 48, 1), dtype=np.float32)
    y_mnist = np.zeros((0,), dtype=np.int64)
    X_mnist_te = np.zeros((0, 48, 48, 1), dtype=np.float32)
    y_mnist_te = np.zeros((0,), dtype=np.int64)
    if args.include_mnist:
        try:
            X_mnist, y_mnist, X_mnist_te, y_mnist_te = load_mnist(
                n_per_class=args.mnist_per_class, seed=SEED)
        except Exception as exc:
            print(f"MNIST unavailable ({exc}); continuing without it.")
    X_emnist = np.zeros((0, 48, 48, 1), dtype=np.float32)
    y_emnist = np.zeros((0,), dtype=np.int64)
    X_emnist_te = np.zeros((0, 48, 48, 1), dtype=np.float32)
    y_emnist_te = np.zeros((0,), dtype=np.int64)
    if args.include_emnist:
        cap = args.emnist_per_class
        if cap == 0 and device == "cpu":
            cap = 10000
            print("CPU run: EMNIST auto-capped to 10000/class (--emnist-per-class)")
        try:
            X_emnist, y_emnist, X_emnist_te, y_emnist_te = digit_cnn.make_emnist_digits(
                n_per_class=cap, seed=SEED, return_test=True)
        except Exception as exc:
            print(f"EMNIST unavailable ({exc}); continuing without it.")
    X = np.concatenate([X_empty, X_synth, X_mnist, X_emnist]).astype(np.float32)
    y = np.concatenate([y_empty, y_synth, y_mnist, y_emnist]).astype(np.int64)
    src = np.array(["empty"] * len(y_empty) + ["synth"] * len(y_synth)
                   + ["mnist"] * len(y_mnist) + ["emnist"] * len(y_emnist))
    # official test set = MNIST/EMNIST test splits ONLY (never trained on;
    # the old code concatenated them into the pool and split randomly).
    X_te = np.concatenate([X_mnist_te, X_emnist_te]).astype(np.float32)
    y_te = np.concatenate([y_mnist_te, y_emnist_te]).astype(np.int64)
    src_te = np.array(["mnist"] * len(y_mnist_te) + ["emnist"] * len(y_emnist_te))
    print(f"total {len(y)} | empty {len(y_empty)} | synth {len(y_synth)} "
          f"| mnist {len(y_mnist)} | emnist {len(y_emnist)}")
    print(f"official test set: mnist {len(y_mnist_te)} | emnist {len(y_emnist_te)}")

    X_photo_tr = np.zeros((0, 48, 48, 1), dtype=np.float32)
    y_photo_tr = np.zeros((0,), dtype=np.int64)
    X_photo_va = np.zeros((0, 48, 48, 1), dtype=np.float32)
    y_photo_va = np.zeros((0,), dtype=np.int64)
    if args.include_photo:
        import photo_data
        print("loading photo cells (train + val partitions of the sealed split)...")
        X_photo_tr, y_photo_tr, *_ = photo_data.build_photo_cells(
            "train", hf_dir=args.photo_dir, split_file=args.photo_split_file)
        X_photo_va, y_photo_va, *_ = photo_data.build_photo_cells(
            "val", hf_dir=args.photo_dir, split_file=args.photo_split_file)
        print(f"photo cells: train {len(y_photo_tr)} | val {len(y_photo_va)} "
              f"(test partition sealed - never loaded here)")

    if args.dry_run:
        n_bytes = X.nbytes + X_te.nbytes + X_photo_tr.nbytes + X_photo_va.nbytes
        print(f"DRY RUN: {len(y) + len(y_photo_tr) + len(y_photo_va)} "
              f"train+val samples, {len(y_te)} official test samples")
        print(f"  raw X payloads: {n_bytes / 2 ** 30:.2f} GiB (48x48x1 float32, uncopied)")
        print("  eager per-epoch augmentation copy is GONE (lazy DataLoader now);")
        print("  remaining copies: concat, train/val split indexing (~0.7x each).")
        print("  full uncapped EMNIST: ~366k samples -> ~3.2 GiB X + ~2.2 GiB split"
              " copies; if RAM is tight use --emnist-per-class 5000-10000.")
        return

    # Non-photo sources: stratified 85/15 split as before. Photo cells keep
    # their SEALED partition membership (train->train, val->val); the photo
    # val partition drives model selection AND temperature scaling.
    tr_idx, va_idx = split_stratified(y, train_frac=0.85, val_frac=0.15, seed=SEED)
    X_tr = np.concatenate([X[tr_idx], X_photo_tr]).astype(np.float32)
    y_tr = np.concatenate([y[tr_idx], y_photo_tr]).astype(np.int64)
    src_tr = np.concatenate([src[tr_idx],
                             np.array(["photo"] * len(y_photo_tr))])
    X_va = np.concatenate([X[va_idx], X_photo_va]).astype(np.float32)
    y_va = np.concatenate([y[va_idx], y_photo_va]).astype(np.int64)
    src_va = np.concatenate([src[va_idx],
                             np.array(["photo"] * len(y_photo_va))])
    print(f"split -> train {len(y_tr)} | val {len(y_va)} "
          f"(photo val {len(y_photo_va)})")

    init_state = None
    if args.warm_start:
        init_state = digit_cnn.load_digit_model(args.warm_start, device=device) \
            .state_dict()
    model = train(X_tr, y_tr, src_tr, X_va, y_va, src_va,
                  args.epochs, args.batch_size, args.lr, device, args.out,
                  ckpt_path=args.ckpt, resume=args.resume, init_state=init_state,
                  early_stop_patience=args.early_stop_patience,
                  min_epochs=args.min_epochs)

    # Temperature scaling: calibrate on the photo val split when present (the
    # distribution the solver's confidence gate actually faces); otherwise on
    # the whole validation split.
    cal_y, cal_X = y_va, X_va
    ph = src_va == "photo"
    if ph.any():
        cal_X, cal_y = X_va[ph], y_va[ph]
    if len(cal_y) > 0:
        T = digit_cnn.fit_temperature(model, cal_X, cal_y, device)
        digit_cnn.save_temperature(T, args.out)

    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()          # release training's cached blocks
    preds_all = []
    with torch.no_grad():
        # batched - the whole test set in one forward pass would OOM (the
        # first conv layer's activations are ~6 GiB for 21k x 48x48)
        for i in range(0, len(X_te), args.batch_size):
            xb = torch.from_numpy(X_te[i:i + args.batch_size]).permute(0, 3, 1, 2).to(device)
            preds_all.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds_all)
    print(f"\nofficial test accuracy: {float((pred == y_te).mean()):.4f}"
          f" (MNIST/EMNIST test splits - independent of training)")
    for s in np.unique(src_te):
        m = src_te == s
        print(f"  {s:9s}: {(pred[m] == y_te[m]).mean():.4f} ({m.sum()} samples)")

    if args.benchmark_data_dir:
        import benchmark
        benchmark.run(args.benchmark_data_dir, model_path=args.out,
                      frac=args.benchmark_frac, out_png=args.benchmark_out)


if __name__ == "__main__":
    main()
