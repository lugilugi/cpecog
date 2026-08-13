"""Generate the intro-ML "reference figures" (R1-R7) for the capstone write-up.

These are teaching-style figures for an introductory machine-learning course:
they explain the classical techniques and the machine-learning concepts the
project relies on (thresholding, calibration, training dynamics, architecture,
augmentation, data balancing, synthetic-vs-real data) with REAL project data
wherever possible.

Figures (all written to narrative_figures/):
  ref_r1_thresholding_compare.png - global Otsu vs adaptive thresholding on
        the SAME cell: a synthetic shaded cell and a real photo cell, with the
        measured ink fractions (the statistics behind the degeneracy gate).
  ref_r2_calibration_reliability.png - temperature scaling: the fitted NLL(T)
        curve (sidecar T marked, on the photo val pack T was fitted on) and a
        reliability diagram (pre vs post calibration) on the SEALED TEST
        partition cells - the deployment distribution where overconfidence
        actually lives.
  ref_r3_training_history.png - train/val loss + validation accuracy curves
        from the final checkpoint's histories, with the best epoch and the
        early-stopping patience window marked.
  ref_r4_cnn_architecture.png - schematic of the double-conv GAP DigitCNN
        (48x48 -> 32->64->128 -> GAP -> 10-way softmax), with real parameter
        counts from the loaded weights.
  ref_r5_augmentation_grid.png - one rendered digit under the documented
        augmentation set (shift, rotation, blur, brightness/contrast, noise,
        erode/dilate, zoom), each pushed through preprocess_cell - the same
        path training and inference share.
  ref_r6_data_balance.png - per-source training volumes and the naive vs
        balanced (WeightedRandomSampler) source shares.
  ref_r7_empty_cells_synth_vs_real.png - real photo empty cells (as the CNN
        sees them) vs synthetic empty cells, with measured purity statistics
        over ALL validation empties and fragment survivors shown in the grid.

Run as `python tools/make_reference_figures.py` from the repo root. Needs the
final weights + temperature sidecar (models/digit_cnn.pth*), the photo val
pack (emnist/photo_packed_val.npz) and the final checkpoint
(models/best_model.pt) for R3. R1 needs the real-photo test images (data/test).
"""
import json
import os
import sys

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "narrative_figures")
os.makedirs(OUT, exist_ok=True)

DATA = os.path.join(REPO, "data")
TEST_IMG = os.path.join(DATA, "test", "images")
TEST_META = os.path.join(DATA, "test", "metadata.jsonl")
WEIGHTS = os.path.join(REPO, "models", "digit_cnn.pth")
TEMP_SIDECAR = os.path.join(REPO, "models", "digit_cnn.pth.temperature.json")
VAL_PACK = os.path.join(REPO, "emnist", "photo_packed_val.npz")
BEST_CKPT = os.path.join(REPO, "models", "best_model.pt")

DPI = 150
BEST = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def device_auto():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def imread_any(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))[:, :, ::-1].copy()


# --------------------------------------------------------------------------- R1

def _shaded_digit_cell():
    """A rendered digit under a left-dark -> right-bright lighting gradient."""
    from digit_cnn import find_print_fonts, render_digit
    fonts = find_print_fonts()
    ink = render_digit("7", fonts[0], size=64)
    digit = np.clip(255.0 - ink, 0, 255).astype(np.uint8)  # dark-on-light
    canvas = np.full((96, 96), 255, dtype=np.uint8)
    ys, xs = np.nonzero(digit < 128)
    ys, xs = ys + 16, xs + 16
    canvas[ys, xs] = digit[np.nonzero(digit < 128)]
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    xf = np.linspace(0.0, 1.0, 96)
    canvas = np.clip(canvas.astype(np.float64) * (0.30 + 0.70 * xf[None, :]), 0, 255).astype(np.uint8)
    return canvas


def _real_cells():
    """One definite digit cell + one definite empty cell from data/test."""
    from sudoku_core import four_point_transform, extract_cells
    have = set(os.listdir(TEST_IMG))
    for line in open(TEST_META, encoding="utf-8"):
        row = json.loads(line)
        name = os.path.basename(row["file_name"])
        if name not in have:
            continue
        flags = row["cells"]
        grid = np.full((9, 9), -1, dtype=int)
        for r in range(9):
            for c in range(9):
                f = flags[r][c]
                dig = [k for k in range(1, 10) if f[k] == 1]
                if len(dig) == 1:
                    grid[r, c] = dig[0]
                elif len(dig) == 0 and f[0] == 0:
                    grid[r, c] = 0
        img = imread_any(os.path.join(TEST_IMG, name))
        kp = np.array(row["keypoints"], dtype=np.float32).reshape(4, 2)
        pts = kp[[0, 3, 2, 1]]  # TL,BL,BR,TR -> TL,TR,BR,BL
        cells = extract_cells(four_point_transform(img, pts, size=600), size=600)
        if len(cells) != 81:
            continue
        digit_pos = [(r, c) for r in range(9) for c in range(9) if 1 <= grid[r, c] <= 9]
        empty_pos = [(r, c) for r in range(9) for c in range(9) if grid[r, c] == 0]
        if digit_pos and empty_pos:
            dr, dc = digit_pos[0]
            er, ec = empty_pos[0]
            return cells[dr * 9 + dc], cells[er * 9 + ec]
    raise RuntimeError("no usable puzzle found in data/test")


def _to_gray(cell):
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    return cell


def _thresholds(cell):
    """(blur, otsu, adaptive) + ink fractions for the caption."""
    blur = cv2.GaussianBlur(_to_gray(cell), (3, 3), 0)
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 7)
    return (blur, otsu, adaptive,
            float((otsu > 0).mean()), float((adaptive > 0).mean()))


def _panel_ax(ax, img, title, ink=None):
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if ink is not None:
        ax.set_xlabel(f"ink {ink * 100:.1f}%", fontsize=8, color="#555555")


def fig_r1():
    shaded = _shaded_digit_cell()
    real_digit, real_empty = _real_cells()

    fig, axes = plt.subplots(3, 4, figsize=(12.5, 8))
    rows = [("Synthetic digit under a shading gradient", shaded),
            ("Real photo digit cell", real_digit),
            ("Real photo EMPTY cell", real_empty)]
    for ri, (label, cell) in enumerate(rows):
        blur, otsu, adaptive, ink_o, ink_a = _thresholds(cell)
        # Otsu on the shaded cell: one threshold cannot follow the gradient.
        panels = [(blur, "input (blurred)", None),
                  (otsu, "global Otsu (one threshold)", ink_o),
                  (adaptive, "adaptive Gaussian (15, C=7)", ink_a),
                  (otsu, "gate: is Otsu trusted?", None)]
        for ci, (im, title, ink) in enumerate(panels):
            if ci == 3:
                deg = ink_a < 0.005
                spread = float(blur.std()) > 18
                ax = axes[ri][ci]
                ax.axis("off")
                ax.imshow(np.full((96, 96, 3), 255, dtype=np.uint8))
                verdict = "Otsu used" if (deg and spread) else "adaptive kept"
                ax.set_title(title, fontsize=9)
                ax.text(0.06, 0.62, verdict, transform=ax.transAxes,
                        ha="left", va="center",
                        fontsize=11, fontweight="bold",
                        color=RED if (deg and spread) else BEST)
                ax.text(0.06, 0.32,
                        "degeneracy gate:\n"
                        f"ink fraction < 0.5%: {'yes' if deg else 'no'}\n"
                        f"blur std > 18: {'yes' if spread else 'no'}",
                        transform=ax.transAxes,
                        ha="left", va="center", fontsize=7.5,
                        color="#333333")
                continue
            _panel_ax(axes[ri][ci], im, title, ink)
        axes[ri][0].set_ylabel(label, fontsize=9, rotation=90, labelpad=20)
    fig.suptitle("R1 — Global vs adaptive thresholding: one threshold cannot "
                 "follow shading", fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "ref_r1_thresholding_compare.png")


# --------------------------------------------------------------------------- R2

def _forward_logits(X, model, device):
    import torch
    dev = torch.device(device)
    model.eval()
    logits = []
    for i in range(0, len(X), 1024):
        xb = torch.from_numpy(np.asarray(X[i:i + 1024], dtype=np.float32)) \
            .permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            logits.append(model(xb).detach().cpu().numpy())
    return np.concatenate(logits).astype(np.float64)


def _val_logits():
    """Logits on the photo VALIDATION pack (the set T was fitted on)."""
    from digit_cnn import load_digit_model
    d = np.load(VAL_PACK, allow_pickle=True)
    model = load_digit_model(WEIGHTS, device=device_auto())
    return _forward_logits(d["X"], model, device_auto()), np.asarray(d["y"], dtype=np.int64)


def _test_logits():
    """Logits over the SEALED TEST partition cells (the deployment
    distribution - where overconfidence actually lives), built through the
    same versioned photo_data path as the benchmark."""
    from digit_cnn import load_digit_model
    import photo_data

    if not os.path.exists(os.path.join(REPO, "emnist", "photo_packed_test.npz")):
        print("building emnist/photo_packed_test.npz (one-time, ~2-4 min)...")
    X, y, _names, _rows, _cols = photo_data.build_photo_cells("test")
    model = load_digit_model(WEIGHTS, device=device_auto())
    return _forward_logits(X, model, device_auto()), np.asarray(y, dtype=np.int64)


def _nll(logits, y, t):
    z = logits / max(float(t), 1e-3)
    zm = z - z.max(1, keepdims=True)
    logsum = np.log(np.exp(zm).sum(1)) + z.max(1)
    return float(-np.mean(z[np.arange(len(y)), y] - logsum))


def _reliability(probs, y, nbins=10):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0.0, 1.0, nbins + 1)
    accs, means, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        counts.append(int(m.sum()))
        means.append(float(conf[m].mean()) if m.any() else float("nan"))
        accs.append(float(correct[m].mean()) if m.any() else float("nan"))
    means = np.array(means)
    accs = np.array(accs)
    ece = float(np.nansum(np.where(np.isnan(accs), 0, counts) *
                          np.abs(np.where(np.isnan(accs), 0, accs) -
                                 np.where(np.isnan(means), 0, means))) / len(y))
    return conf, correct, means, accs, ece


def fig_r2():
    val_logits, val_y = _val_logits()
    test_logits, test_y = _test_logits()
    n_test = len(test_y)
    T_side = float(json.load(open(TEMP_SIDECAR, encoding="utf-8"))["temperature"])

    ts = np.linspace(0.4, 4.0, 61)
    nlls = np.array([_nll(val_logits, val_y, t) for t in ts])
    t_best = float(ts[np.argmin(nlls)])

    probs_pre = np.exp(test_logits - test_logits.max(1, keepdims=True))
    probs_pre /= probs_pre.sum(1, keepdims=True)
    probs_post = np.exp((test_logits - test_logits.max(1, keepdims=True)) / T_side)
    probs_post /= probs_post.sum(1, keepdims=True)

    _, _, m_pre, a_pre, ece_pre = _reliability(probs_pre, test_y)
    _, _, m_post, a_post, ece_post = _reliability(probs_post, test_y)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.plot(ts, nlls, color=BEST, lw=1.8)
    ax.axvline(t_best, color=GREEN, ls="--", lw=1.2,
               label=f"argmin T = {t_best:.2f}")
    ax.axvline(T_side, color=ORANGE, ls=":", lw=1.6,
               label=f"sidecar T = {T_side:.3f}")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("softmax NLL on photo val (11,735 cells)")
    ax.set_title("R2a — Temperature scaling: the fitted NLL(T) curve",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1,
            label="perfect calibration")
    ax.plot(m_pre, a_pre, "o-", color=RED, lw=1.5, ms=4,
            label=f"raw softmax (ECE {ece_pre * 100:.2f}%)")
    ax.plot(m_post, a_post, "s-", color=BEST, lw=1.5, ms=4,
            label=f"temperature-scaled (ECE {ece_post * 100:.2f}%)")
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("observed accuracy in bin")
    ax.set_title(f"R2b — Reliability diagram (sealed test cells, {n_test:,})",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.text(0.03, 0.30,
            "points ABOVE the diagonal = overconfident:\nthe model believes it "
            "is right more often than it is.\nA wrong cell with conf > 0.9 is a "
            "structured failure -\nthe correction search will never revisit it.",
            fontsize=7.5, color="#555555", va="top")

    fig.tight_layout()
    savefig(fig, "ref_r2_calibration_reliability.png")


# --------------------------------------------------------------------------- R3

def fig_r3():
    import torch
    ck = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
    h = ck["histories"]
    epochs = np.arange(1, len(h["train_loss"]) + 1)
    train_loss = np.array(h["train_loss"])
    val_loss = np.array(h["val_loss"])
    val_acc = np.array(h["val_acc"])
    val_score = np.array(h["val_score"])
    val_photo = np.array(h["val_photo_acc"])
    best_ep = int(np.argmax(val_score))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.plot(epochs, train_loss, "o-", color=BEST, lw=1.6, ms=3, label="train loss")
    ax.plot(epochs, val_loss, "s-", color=ORANGE, lw=1.6, ms=3, label="val loss")
    ax.axvline(best_ep + 1, color=GREEN, ls="--", lw=1.2,
               label=f"best epoch {best_ep + 1} (weights saved)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("R3a — Loss curves and early stopping", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, np.array(val_acc) * 100, "o-", color=BEST, lw=1.6, ms=3,
            label="val accuracy (all sources)")
    ax.plot(epochs, val_photo * 100, "s-", color=ORANGE, lw=1.6, ms=3,
            label="val photo accuracy")
    ax.plot(epochs, val_score * 100, "^-", color=GREEN, lw=1.4, ms=3,
            label="selection score (mean per-source)")
    ax.axvspan(best_ep + 2, min(len(epochs), best_ep + 6), color=RED, alpha=0.12,
               label="patience window (5 epochs)")
    ax.axvline(best_ep + 1, color=GREEN, ls="--", lw=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("percent")
    ax.set_title("R3b — Validation curves (final 21-epoch run)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, "ref_r3_training_history.png")


# --------------------------------------------------------------------------- R4

def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head.parameters()) if hasattr(model, "head") else 0
    return total, head


def fig_r4():
    from digit_cnn import load_digit_model
    model = load_digit_model(WEIGHTS, device="cpu")
    total, head = _count_params(model)

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    blocks = [
        (2, "Input\n48×48×1", "#f0f0f0", 8.0),
        (12, "Conv3×3 32\nBN + ReLU\n×2 stages", "#dbe9f6", 11.0),
        (25, "MaxPool2", "#f0f0f0", 6.0),
        (33, "Conv3×3 64\nBN + ReLU\n×2 stages", "#dbe9f6", 11.0),
        (46, "MaxPool2", "#f0f0f0", 6.0),
        (54, "Conv3×3 128\nBN + ReLU\n×2 stages", "#dbe9f6", 11.0),
        (67, "MaxPool2", "#f0f0f0", 6.0),
        (76, "GAP 1×1\n(no position lock)", "#fde9d9", 10.0),
        (89, "Dropout + Linear\n128→10", "#fde9d9", 11.0),
    ]
    for x, label, color, w in blocks:
        box = FancyBboxPatch((x, 36), w, 28, boxstyle="round,pad=0.5",
                             linewidth=1.3, edgecolor="#333333",
                             facecolor=color)
        ax.add_patch(box)
        ax.text(x + w / 2, 50, label, ha="center", va="center", fontsize=7.5)
    for x0, x1 in [(10.0, 12), (23.0, 25), (31.0, 33), (44.0, 46), (52.0, 54),
                   (65.0, 67), (73.0, 76), (86.0, 89)]:
        ax.add_patch(FancyArrowPatch((x0, 50), (x1, 50), arrowstyle="-|>",
                                     mutation_scale=13, color="#333333",
                                     lw=1.1))
    ax.text(50, 92,
            "R4 — DigitCNN: double-convolution stages with a "
            "global-average-pooling head",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(50, 84,
            "fully convolutional up to the head — no positional bias, any input size",
            ha="center", fontsize=9, color="#555555")
    ax.text(50, 14,
            f"total parameters {total:,}  |  head parameters {head:,}  "
            f"(the old Linear(4608,10) flatten head had ~46k — "
            f"{46080 // max(head, 1)}× the GAP head)",
            ha="center", fontsize=9, color="#555555")
    ax.text(50, 7,
            "81 cells → 81×10 softmax → argmax per cell → 9×9 recognized grid",
            ha="center", fontsize=9, color="#555555")

    savefig(fig, "ref_r4_cnn_architecture.png")


# --------------------------------------------------------------------------- R5

def fig_r5():
    from digit_cnn import (blur, find_print_fonts, preprocess_cell, render_digit,
                           rotate, translate)
    rng = np.random.default_rng(11)
    font = find_print_fonts()[0]
    base = np.clip(255.0 - render_digit("7", font, size=64), 0, 255).astype(np.uint8)
    variants = [
        ("original", base, False),
        ("shift +2 px", translate(base, 2, 2), False),
        ("shift −2 px", translate(base, -2, -2), False),
        ("rotate +8°", rotate(base, 8), False),
        ("rotate −8°", rotate(base, -8), False),
        ("light blur", blur(base), False),
        ("brightness ×0.6", np.clip(base * 0.6, 0, 255).astype(np.uint8), False),
        ("noise N(0,6)", np.clip(base.astype(np.float64) +
                                 rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8), False),
        ("erode (print weight)", cv2.erode(base, np.ones((2, 2), np.uint8)), False),
        ("zoom 0.8", cv2.resize(base, (51, 51),
                                interpolation=cv2.INTER_AREA), False),
    ]
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(1.15 * n + 2, 3.4))
    for ax, (label, raw, _) in zip(axes, variants):
        proc = preprocess_cell(raw, target=48)
        ax.imshow(proc[..., 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("R5 — Augmentation on the RAW cell before preprocessing "
                 "(each variant is what the CNN sees after preprocess_cell)",
                 fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    savefig(fig, "ref_r5_augmentation_grid.png")


# --------------------------------------------------------------------------- R6

def fig_r6():
    sources = ["Curated print\nfonts (9 digits)", "MNIST", "EMNIST",
               "Real photo\ncells (train)", "Synthetic\nempties"]
    volumes = np.array([4000 * 9, 60000, 252000, 52685, 15000])
    naive = volumes / volumes.sum()
    balanced = np.full(5, 0.20)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    bars = ax.bar(sources, volumes, color=[BEST, ORANGE, GREEN, RED, "#9467bd"])
    for b, v in zip(bars, volumes):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02,
                f"{v / 1000:.0f}k" if v >= 1000 else f"{v:,}",
                ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("samples (log scale)")
    ax.set_title("R6a — Raw training-source volumes", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1]
    x = np.arange(5)
    w = 0.36
    ax.bar(x - w / 2, naive * 100, w, label="naive share (sampling by volume)",
           color="#999999")
    ax.bar(x + w / 2, balanced * 100, w,
           label="WeightedRandomSampler (equal source weight)",
           color=BEST)
    ax.set_xticks(x)
    ax.set_xticklabels(sources, fontsize=7)
    ax.set_ylabel("share of each mini-batch (%)")
    ax.set_ylim(0, 75)
    ax.set_title("R6b — Source share in the gradient (naive vs balanced)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.08,
            "EMNIST is ~61% of the pool — without balancing it dominates every "
            "batch and the photo domain is under-learned",
            transform=ax.transAxes, fontsize=7, color="#555555", va="bottom")

    fig.tight_layout()
    savefig(fig, "ref_r6_data_balance.png")


# --------------------------------------------------------------------------- R7

def fig_r7():
    from digit_cnn import make_empty_cells
    d = np.load(VAL_PACK, allow_pickle=True)
    X, y = d["X"], d["y"]
    empties = X[y == 0]

    def stats(cells):
        black = float((cells.reshape(len(cells), -1).max(axis=1) == 0).mean())
        ink = [float(c.max()) for c in cells if float(c.max()) > 0]
        return black, (float(np.mean(ink)) if ink else 0.0)

    rb, rink = stats(empties)
    synth, _ = make_empty_cells(n_per_class=1200, seed=7)
    sb, sink = stats(synth)

    # Montage: prefer real fragment survivors (the ~3% that keep thin lines)
    # so the figure shows BOTH looks; pad with pure-black cells.
    frag_idx = [i for i, c in enumerate(empties) if float(c.max()) > 0]
    show_real = np.array([empties[i] for i in frag_idx[:8]])
    black_idx = [i for i, c in enumerate(empties) if float(c.max()) == 0]
    need = 12 - len(show_real)
    if need > 0:
        pad = np.array([empties[i] for i in black_idx[:need]])
        show_real = np.concatenate([show_real, pad]) if len(show_real) else pad
    show_real = show_real[:12]
    show_synth = synth[:12]

    def row(axs, cells):
        for ax, c in zip(axs, cells):
            ax.imshow(c[..., 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

    fig, axes = plt.subplots(2, 12, figsize=(13.5, 3.6))
    row(axes[0], show_real)
    row(axes[1], show_synth)
    axes[0][0].set_ylabel("real photo\nempties\n(as CNN sees it)",
                          fontsize=8, rotation=0, labelpad=52, va="center")
    axes[1][0].set_ylabel("synthetic\nempties", fontsize=8, rotation=0,
                          labelpad=52, va="center")
    fig.suptitle(
        f"R7 — Class-0 (empty) training data: synthetic calibrated to measured "
        f"real appearance\n(all {len(empties):,} val empties: "
        f"{rb * 100:.1f}% pure black, survivors avg {rink * 100:.1f}% ink  |  "
        f"synthetic 1200: {sb * 100:.1f}% pure black, "
        f"{sink * 100:.1f}% ink)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    savefig(fig, "ref_r7_empty_cells_synth_vs_real.png")


# --------------------------------------------------------------------------- main

def main():
    print("device:", device_auto())
    fig_r1()
    fig_r2()
    fig_r3()
    fig_r4()
    fig_r5()
    fig_r6()
    fig_r7()
    print("all reference figures written to", OUT)


if __name__ == "__main__":
    main()
