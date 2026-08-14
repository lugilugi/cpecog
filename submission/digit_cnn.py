import os
import random

import numpy as np
import torch
import torch.nn as nn

# Bump whenever the preprocessing pipeline changes shape/semantics - cached
# packed arrays (EMNIST) and any model metadata validate against it.
PREPROCESS_VERSION = "v5"


def render_digit(d, font_path, size=64):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, int(size * 0.8))
    bbox = draw.textbbox((0, 0), d, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), d, fill=255, font=font)
    return np.asarray(img, dtype=np.float32)


def find_print_fonts(extra_dirs=None):
    """Curated COMMON PRINT fonts only (DejaVu, Arial, Times, Courier, Verdana,
    Georgia, Tahoma, Corbel, Segoe UI, Liberation, Free*). No decorative or
    handwriting fonts - those produced the 'weird' synthetic digits."""
    wanted = ("dejavu", "arial", "times", "cour", "verdana", "georgia", "tahoma",
              "corbel", "segoeui", "liberation", "freesans", "freeserif")
    dirs = list(extra_dirs or [])
    if os.name == "nt":
        dirs.append(r"C:\Windows\Fonts")
    for d in ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
              "/usr/share/fonts/truetype/freefont", "/usr/share/fonts"]:
        if os.path.isdir(d):
            dirs.append(d)
    found = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            low = name.lower()
            if low.endswith((".ttf", ".ttc")) and any(k in low for k in wanted):
                found.append(os.path.join(d, name))
    return sorted(set(found))


def make_synthetic_digits(n_per_class=4000, target=48, seed=0, sizes=(52, 56, 64, 72)):
    """Curated print-font synthetic digits (classes 1-9), rendered dark-on-light
    and passed through the SAME `preprocess_cell` as everything else.

    CONSERVATIVE augmentation on the RAW rendered cell BEFORE preprocessing
    (deliberately mild - real puzzle cells are fronto-parallel after the grid
    warp): zoom 0.8-1.0, shift +/-2 px, rotation +/-8 deg, occasional
    print-weight erode/dilate, WEAK sensor noise, occasional light blur, plus
    a brightness/contrast (lighting) variation. The thresholding pipeline sees
    the same camera/paper effects it sees in real photos. Trains to ~99%
    validation. Returns (X, y); empty (0,0,0,1) arrays if no print fonts.
    """
    import cv2
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    fonts = find_print_fonts()
    if not fonts:
        return np.zeros((0, 48, 48, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X, y = [], []
    for k in range(1, 10):
        for _ in range(n_per_class):
            font = rng.choice(fonts)
            size = rng.choice(sizes)
            img = render_digit(str(k), font, size)     # white on black
            # --- augment the RAW cell BEFORE preprocessing --------------------
            scale = np_rng.uniform(0.8, 1.0)           # small zoom in/out
            img = cv2.resize(img, (int(target / scale), int(target / scale)),
                             interpolation=cv2.INTER_AREA)
            img = center_crop_or_pad(img, target)
            dx, dy = int(np_rng.uniform(-2, 3)), int(np_rng.uniform(-2, 3))
            img = translate(img, dx, dy)                 # shift +/-2 px (black border)
            if rng.random() < 0.8:                     # camera tilt (mild)
                img = rotate(img, np_rng.uniform(-8, 8))
                img = center_crop_or_pad(img, target)
            if rng.random() < 0.4:                     # print weight: thin/heavy
                ker = np.ones((3, 3), np.uint8)
                if rng.random() < 0.5:
                    img = cv2.erode(img, ker, iterations=1)
                else:
                    img = cv2.dilate(img, ker, iterations=1)
            if rng.random() < 0.6:                     # lighting: brightness/contrast
                gain = np_rng.uniform(0.75, 1.25)
                bias = np_rng.uniform(-25, 25)
                img = np.clip(img * gain + bias, 0, 255)
            if rng.random() < 0.6:                     # weak sensor/paper noise
                img = img + np_rng.normal(0, np_rng.uniform(2, 6), img.shape)
            if rng.random() < 0.2:                     # occasional light blur
                img = cv2.GaussianBlur(img, (3, 3), 0)
            if rng.random() < 0.05:                    # rare strong blur
                img = cv2.GaussianBlur(img, (5, 5), 0)
            raw = np.clip(255 - img, 0, 255).astype(np.uint8)   # dark on light
            X.append(preprocess_cell(raw, target=target))
            y.append(k)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def _letterbox_fragment(img, target):
    """Tight-crop + 12% pad + aspect-preserving letterbox of a fragment.

    The SAME geometry `_finish_preprocess` applies to surviving components, so
    generated class-0 artifacts look like real post-preprocessing empties
    instead of staying at native size/location. Pure-black input passes
    through unchanged.
    """
    import cv2
    ys, xs = np.nonzero(img)
    if len(xs) == 0:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(max(x1 - x0, y1 - y0) * 0.12) + 2
    crop = img[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1]
    ch, cw = crop.shape
    scale = target / max(ch, cw)
    nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target), dtype=np.float32)
    y0c, x0c = (target - nh) // 2, (target - nw) // 2
    canvas[y0c:y0c + nh, x0c:x0c + nw] = resized
    return canvas


def make_empty_cells(n_per_class=15000, target=48, seed=0):
    """Class-0 (EMPTY cell) samples matching the post-preprocessing appearance
    of real empty photo cells under the CURRENT pipeline (measured on the
    labeled benchmark photos): ~97% pure black - the Phase 1 preprocessing
    strips grid fragments and tiny specks - and ~3% carrying a surviving thin
    line fragment or small blob (5-15% ink). Digits 1-9 come from MNIST +
    curated print-font synthetic - no font rendering.

    Fragments are drawn at full intensity (1.0 - the thresholded pipeline
    output is binary white), then passed through `_letterbox_fragment` so
    their size/location match real post-cleanup geometry.
    """
    import cv2
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_per_class):
        img = np.zeros((target, target), dtype=np.float32)
        if rng.random() < 0.97:
            pass                                    # clean empty (~97%)
        else:
            kind = np_rng.choice(["blob", "line", "big"], p=[0.5, 0.35, 0.15])
            if kind == "blob":                      # small dirt blob (5-10%)
                cx = np_rng.uniform(14, target - 14)
                cy = np_rng.uniform(14, target - 14)
                rx, ry = np_rng.uniform(4, 7), np_rng.uniform(4, 7)
                cv2.ellipse(img, (int(cx), int(cy)), (int(rx), int(ry)),
                            np_rng.uniform(0, 180), 0, 360, 1.0, -1)
            elif kind == "line":                    # thin fragment (3-6%)
                if rng.random() < 0.5:
                    y0 = int(np_rng.uniform(4, target - 6))
                    x0 = int(np_rng.uniform(2, target - 2))
                    img[y0:y0 + int(np_rng.uniform(1, 3)),
                        x0:x0 + int(np_rng.uniform(20, target - 4))] = 1.0
                else:
                    x0 = int(np_rng.uniform(4, target - 6))
                    y0 = int(np_rng.uniform(2, target - 2))
                    img[y0:y0 + int(np_rng.uniform(20, target - 4)),
                        x0:x0 + int(np_rng.uniform(1, 3))] = 1.0
            else:                                   # big blob (10-15%)
                cx = np_rng.uniform(16, target - 16)
                cy = np_rng.uniform(16, target - 16)
                rx, ry = np_rng.uniform(7, 10), np_rng.uniform(7, 10)
                cv2.ellipse(img, (int(cx), int(cy)), (int(rx), int(ry)),
                            np_rng.uniform(0, 180), 0, 360, 1.0, -1)
            img = _letterbox_fragment(img, target)
        img = cv2.GaussianBlur(img, (3, 3), 0)      # soften (anti-alias)
        X.append(np.clip(img, 0, 1)[..., None].astype(np.float32))
        y.append(0)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def cv2_resize(img, size):
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def center_crop_or_pad(img, target):
    import cv2
    h, w = img.shape
    if h >= target and w >= target:
        y0, x0 = (h - target) // 2, (w - target) // 2
        return img[y0:y0 + target, x0:x0 + target]
    canvas = np.zeros((target, target), dtype=np.float32)
    y0, x0 = (target - h) // 2, (target - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def rotate(img, angle):
    import cv2
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def translate(img, dx, dy):
    """Shift the image by (dx, dy) filling the void with black (border).

    Realistic camera movement: pixels pushed out of frame are lost, not
    wrapped around to the opposite edge (unlike np.roll).
    """
    import cv2
    h, w = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def blur(img):
    import cv2
    return cv2.GaussianBlur(img, (3, 3), 0)


class DigitCNN(nn.Module):
    """Double-convolution CNN: 32 -> 64 -> 128 (2x Conv3x3-BN-ReLU per stage,
    MaxPool 2 per stage), then a GLOBAL-AVERAGE-POOLING head.

    GAP replaces the old Flatten + Linear(4608, 10): the head is now
    AdaptiveAvgPool2d(1,1) -> Flatten -> Dropout -> Linear(128, num_classes).
    That removes the spatial position-lock of the old 4608-vector classifier
    (synthetic fonts always sat at fixed spots; real digits drift) and drops
    the head from ~46k params to ~1.3k. Inputs no longer have to be 48x48
    (the head adapts to any spatial size), though the pipeline keeps 48x48.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_emnist_digits(n_per_class=None, target=48, seed=0, root="emnist",
                       return_test=False):
    """EMNIST handwritten digits 1-9 (split='digits', ~280k samples, 28x28).

    Same treatment as MNIST: upscaled to 56 then through the SAME
    `preprocess_cell` (adaptive-threshold route). Digit 0 is dropped: class 0
    means EMPTY cell. n_per_class=None uses everything; a cap keeps CPU runs
    feasible (seeded per-class shuffle).

    The OFFICIAL TEST SET IS KEPT SEPARATE - it is never concatenated into
    the training pool (with return_test=True you get (X, y, Xt, yt); the
    default returns the training set only). Both are cached to
    <root>/emnist_packed.npz with a PREPROCESS_VERSION marker; a cache from a
    different preprocessing version is regenerated (~5-15 min), never reused
    silently. Raises if torchvision / the download is unavailable.
    """
    import cv2
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    cache_path = os.path.join(root, "emnist_packed.npz")

    def pack(ds):
        X, y = [], []
        for img, label in ds:
            if int(label) == 0:
                continue
            arr = img.numpy()[0]                   # 28x28, WHITE on black
            arr = 1.0 - arr                        # invert: dark ink on paper
            arr = cv2.resize(arr, (56, 56), interpolation=cv2.INTER_CUBIC)
            arr = np.clip(arr, 0.0, 1.0)           # cubic resize can overshoot
            X.append(preprocess_cell((arr * 255.0).astype(np.uint8), target=target))
            y.append(int(label))
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)

    X = y = Xt = yt = None
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            if (data.get("preprocess_version") is not None
                    and str(data["preprocess_version"].item()) == PREPROCESS_VERSION
                    and all(k in data for k in ("X_tr", "y_tr", "X_te", "y_te"))):
                X, y = data["X_tr"].astype(np.float32), data["y_tr"].astype(np.int64)
                Xt, yt = data["X_te"].astype(np.float32), data["y_te"].astype(np.int64)
        except Exception:
            X = y = Xt = yt = None
    if X is None:
        train = datasets.EMNIST(root=root, split="digits", train=True,
                                download=True, transform=ToTensor())
        test = datasets.EMNIST(root=root, split="digits", train=False,
                               download=True, transform=ToTensor())
        X, y = pack(train)
        Xt, yt = pack(test)
        os.makedirs(root, exist_ok=True)
        np.savez(cache_path, X_tr=X, y_tr=y, X_te=Xt, y_te=yt,
                 preprocess_version=PREPROCESS_VERSION)
    if n_per_class:
        rng = np.random.default_rng(seed)
        keep = []
        for k in np.unique(y):
            idx = np.where(y == k)[0]
            rng.shuffle(idx)
            keep.append(idx[:n_per_class])
        sel = np.concatenate(keep)
        X, y = X[sel], y[sel]
        keep = []
        for k in np.unique(yt):
            idx = np.where(yt == k)[0]
            rng.shuffle(idx)
            keep.append(idx[:max(n_per_class // 4, 1)])
        sel = np.concatenate(keep)
        Xt, yt = Xt[sel], yt[sel]
    if return_test:
        return X, y, Xt, yt
    return X, y


def train_digit_model(X, y, epochs=8, batch_size=256, lr=1e-3, val_frac=0.1, device="cpu",
                      weight_path="models/digit_cnn.pth", verbose=True):
    torch.manual_seed(0)
    device = torch.device(device)
    n = len(X)
    perm = np.random.RandomState(0).permutation(n)
    n_val = int(n * val_frac)
    X, y = X[perm], y[perm]
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    model = DigitCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=4, min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(X_, y_, train=True):
        model.train() if train else model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(train):
            for i in range(0, len(X_), batch_size):
                xb = torch.from_numpy(X_[i:i + batch_size]).permute(0, 3, 1, 2).to(device)
                yb = torch.from_numpy(y_[i:i + batch_size]).to(device)
                out = model(xb)
                loss = loss_fn(out, yb)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                loss_sum += loss.item() * len(xb)
                total += len(xb)
                correct += (out.argmax(1) == yb).sum().item()
        return loss_sum / total, correct / total

    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        tl, ta = run_epoch(X_tr, y_tr, train=True)
        vl, va = run_epoch(X_val, y_val, train=False)
        sched.step(va)
        if verbose:
            print(f"epoch {ep + 1}/{epochs}  train acc {ta:.4f}  val acc {va:.4f}")
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    # save the BEST epoch's weights, not the last epoch's
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), weight_path)
    return model


def load_digit_model(path="models/digit_cnn.pth", device="cpu"):
    """Load the CURRENT double-conv DigitCNN weights (class 0 = empty).

    Validates the state-dict key set against the current architecture: the
    OLD single-conv `digit_cnn.pth` (class 0 = digit 0) fails here with an
    explicit message instead of loading garbage. Full notebook-style
    checkpoints ({'model_state_dict': ...}) are unwrapped.
    """
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    expected = DigitCNN().state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(
            f"model weights '{path}' do not match the current DigitCNN "
            f"architecture (double-conv 32->64->128, class 0 = empty).\n"
            f"  missing keys: {missing[:4]}{'...' if len(missing) > 4 else ''}\n"
            f"  extra keys:   {extra[:4]}{'...' if len(extra) > 4 else ''}\n"
            f"Old single-conv weights (class 0 = digit 0) are incompatible - "
            f"retrain with the current pipeline.")
    model = DigitCNN()
    model.load_state_dict(state)
    model.eval()
    return model.to(torch.device(device))


@torch.no_grad()
def predict_cells(cells, model, device="cpu", temperature=1.0, **preprocess_kwargs):
    """cells: list of raw cell images -> (predicted class per cell, confidences).

    Class 0 = empty cell, classes 1-9 = digits. `preprocess_kwargs` are
    forwarded to `preprocess_cell` (margin_frac, empty_frac, ... for sweeps).
    `temperature` divides the logits before softmax (temperature scaling;
    argmax is unchanged - only confidences are calibrated).
    """
    model.eval()
    dev = torch.device(device)
    batch = np.stack([preprocess_cell(c, target=48, **preprocess_kwargs) for c in cells])
    xb = torch.from_numpy(batch).permute(0, 3, 1, 2).to(dev)
    probs = torch.softmax(model(xb) / temperature, dim=1)
    preds = probs.argmax(1).cpu().numpy()
    confs = probs.max(1).values.cpu().numpy()
    return preds, confs


@torch.no_grad()
def predict_cells_probs(cells, model, device="cpu", temperature=1.0,
                        **preprocess_kwargs):
    """cells: list of raw cell images -> (81, 10) softmax matrix (class 0 = empty).

    `preprocess_kwargs` are forwarded to `preprocess_cell` for sweep runs.
    `temperature` divides the logits before softmax (see `fit_temperature`).
    """
    model.eval()
    batch = np.stack([preprocess_cell(c, target=48, **preprocess_kwargs) for c in cells])
    return classify_preprocessed(batch, model, device, temperature=temperature)


@torch.no_grad()
def classify_preprocessed(inputs, model, device="cpu", temperature=1.0):
    """inputs: (N,48,48,1) already-preprocessed float32 -> (N,10) softmax.

    Used by the re-sensing solver to classify preprocessing variants of a cell.
    `temperature` divides the logits before softmax (calibration; see
    `fit_temperature`).
    """
    model.eval()
    dev = torch.device(device)
    xb = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).permute(0, 3, 1, 2).to(dev)
    return torch.softmax(model(xb) / temperature, dim=1).cpu().numpy()


def fit_temperature(model, X_val, y_val, device="cpu"):
    """Temperature-scaling calibration (Guo et al. 2017): fit ONE scalar T
    minimizing the NLL of softmax(logits / T) on a held-out validation set.

    Fit T on the REAL-PHOTO validation split - fitting on synthetic
    validation calibrates the wrong distribution (the whole point is that
    photo confidences are inflated). T > 1 flattens the softmax: wrong
    cells that used to clear conf_thresh=0.9 no longer do, so the solver's
    correction search actually revisits them.
    """
    model.eval()
    dev = torch.device(device)
    logits_all = []
    for i in range(0, len(y_val), 1024):
        xb = torch.from_numpy(np.asarray(X_val[i:i + 1024], dtype=np.float32)) \
            .permute(0, 3, 1, 2).to(dev)
        logits_all.append(model(xb).detach().cpu().numpy())
    logits = np.concatenate(logits_all).astype(np.float64)
    y = np.asarray(y_val, dtype=np.int64)

    def nll(t):
        t = max(float(t), 1e-3)
        z = logits / t
        zm = z - z.max(1, keepdims=True)
        logsum = np.log(np.exp(zm).sum(1)) + z.max(1)
        return float(-np.mean(z[np.arange(len(y)), y] - logsum))

    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        T = float(res.x)
    except ImportError:
        T, v = 1.0, nll(1.0)
        for t in np.linspace(0.1, 5.0, 50):
            if nll(t) < v:
                T, v = t, nll(t)
        for t in np.linspace(max(0.1, T - 0.5), T + 0.5, 200):
            if nll(t) < v:
                T, v = t, nll(t)
    print(f"temperature scaling: T = {T:.3f} "
          f"(val NLL {nll(1.0):.4f} -> {nll(T):.4f})")
    return T


def save_temperature(T, weight_path):
    """Persist the fitted temperature next to the weights
    (<weight_path>.temperature.json); load it back with `load_temperature`."""
    import json
    with open(f"{weight_path}.temperature.json", "w", encoding="utf-8") as f:
        json.dump({"temperature": float(T)}, f)
    print(f"saved calibration temperature {float(T):.3f} to "
          f"{weight_path}.temperature.json")


def load_temperature(weight_path, default=1.0):
    """Fitted calibration temperature for a weight file; `default` if absent."""
    import json
    import os
    p = f"{weight_path}.temperature.json"
    if os.path.exists(p):
        try:
            return float(json.load(open(p, encoding="utf-8"))["temperature"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return default


def _preprocess_legacy(cell, target=48, return_stats=False):
    """The EXACT pre-study preprocessing pipeline (frozen baseline).

    Adaptive threshold (block 15, constant 7, inverted) with an Otsu fallback
    for washed-out/low-contrast cells, then: a small 4% margin strip (no
    dilation - strokes keep their natural print weight), largest component
    only (tiny specks < 0.5% -> empty), tight crop with 12% padding,
    aspect-ratio-preserving letterboxed resize (digits never stretched).

    FROZEN for the preprocessing study: `preprocess_cell_legacy` and the
    Phase 0 `preprocess_cell` are both this function. The new (Phase 1)
    pipeline replaces `preprocess_cell`'s body; this stays untouched as the
    A/B baseline and regression fallback.

    With return_stats=True returns (final, stats): stats carries per-cell
    diagnostics (threshold path taken, thresholded foreground fraction,
    component count, largest-component fraction) and the intermediate stage
    images (original, blur, thresh, margin, comps, final) - all from the SAME
    code path the CNN sees.
    """
    import cv2
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    if cell.dtype != np.uint8:
        cell = np.clip(cell, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 7)
    threshold_used = "adaptive"
    if (th > 0).mean() < 0.005 and blur.std() > 18:
        otsu = cv2.threshold(blur, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        if (otsu > 0).mean() > (th > 0).mean():
            th = otsu
            threshold_used = "otsu"
    th_raw = th.copy()
    h, w = th.shape
    m = max(1, int(min(h, w) * 0.04))
    th[:m, :] = 0
    th[-m:, :] = 0
    th[:, :m] = 0
    th[:, -m:] = 0
    th_margin = th.copy()
    n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(th, 8)
    comp_count = int(n - 1)                    # foreground components
    largest_frac = 0.0
    if n > 1:
        areas = cc_stats[1:, 4]
        keep = int(np.argmax(areas)) + 1
        largest_frac = cc_stats[keep, 4] / (h * w)
        th[labels != keep] = 0
        if cc_stats[keep, 4] < h * w * 0.005:
            th[:] = 0
    th_comp = th.copy()
    ys, xs = np.nonzero(th)
    if len(xs) == 0:
        final = np.zeros((target, target, 1), dtype=np.float32)   # empty cell
    else:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = int(max(x1 - x0, y1 - y0) * 0.12) + 2
        crop = th[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1]
        ch, cw = crop.shape
        scale = target / max(ch, cw)
        nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target, target), dtype=np.float32)
        y0c, x0c = (target - nh) // 2, (target - nw) // 2
        canvas[y0c:y0c + nh, x0c:x0c + nw] = resized
        final = (canvas / 255.0)[..., None]
    if return_stats:
        return final, {
            "threshold_used": threshold_used,
            "th_ink_frac": float((th_raw > 0).mean()),
            "comp_count": comp_count,
            "largest_comp_frac": largest_frac,
            "stages": {"original": cell, "blur": blur, "thresh": th_raw,
                       "margin": th_margin, "comps": th_comp, "input": final},
        }
    return final


def preprocess_cell_legacy(cell, target=48):
    """FROZEN legacy preprocessing -> (48,48,1) float32.

    The exact pre-study pipeline, kept for A/B testing and as a regression
    fallback: legacy preprocessing + existing model vs new preprocessing +
    the same model. Do not modify - the study compares against this.
    """
    return _preprocess_legacy(cell, target)


# --- Phase 1 pipeline (v5) ---------------------------------------------------
# The benchmark montage showed the dominant failure mode: in empty cells the
# thresholded grid-LINE / border fragments survive the "keep the largest
# connected component" step and the CNN reads them as digits. Measured on the
# wrong empty->digit cells, the fragments are thin (median 1.7 px) elongated
# lines running from a cell border (often spanning edge-to-edge) or L-shaped
# corner chunks. The pipeline: a margin strip (kills near-border clutter),
# shape-based fragment removal on the stripped region (edge-to-edge lines,
# one-border thin lines, L-shaped corner chunks), tiny-component drop, a
# conditional rejoin of split stroke pieces via a THIN BRIDGE (not a rect
# fill - a fill gives a broken stroke the look of a filled blob), then a
# conservative empty test (largest SURVIVING component below a threshold ->
# the cell goes all-black). All surviving components are KEPT - there is no
# largest-component-only step - so a digit is never truncated and a digit +
# stray fragment is passed to the CNN whole (the training data contains the
# same real-photo look). All fractions use the FULL cell area.

_DEFAULT_MARGIN_FRAC = 0.10     # strip: kills border clutter, ~7px on a 66px cell
_DEFAULT_MIN_AREA_FRAC = 0.005  # tiny components below this (of the FULL cell) are removed
_DEFAULT_EMPTY_FRAC = 0.01      # largest surviving component below this -> forced EMPTY
_DEFAULT_GRID_THICKNESS = 6.0   # px: opposite-border lines up to this area/length are fragments
_DEFAULT_GRID_THIN = 2.5        # px: one-border lines below this area/length are fragments
_DEFAULT_GRID_ASPECT = 4.0      # bbox aspect above which an elongated comp is a line
_DEFAULT_GRID_SPAN = 0.7        # one-border lines must span >= this of the inner region
_DEFAULT_CORNER_AREA = 0.06     # corner fragments: area below this of the full cell
_DEFAULT_CORNER_SPAN = 0.4      # corner fragments: bbox >= this of the inner region in both dims
_DEFAULT_MERGE_GAP = 5          # px between split stroke pieces that may merge
_DEFAULT_MERGE_OVERLAP = 0.5    # min projection overlap (of the narrower piece) to merge
_MAX_MERGES = 4                 # greedy merge iterations


def _merge_split_pieces(th, h, w, min_area, gap=_DEFAULT_MERGE_GAP,
                        overlap=_DEFAULT_MERGE_OVERLAP):
    """Greedily rejoin split stroke pieces (e.g. a 7's stem and bar).

    A pair merges when BOTH pieces are non-tiny (>= min_area of the full
    cell - noise specks never fuse into a phantom digit), their bounding
    boxes are within `gap` px along the stack axis, their projections
    overlap by >= `overlap`, and the union stays digit-like (bbox
    aspect <= 3.5, area >= min_area). The pieces are joined by a THIN
    BRIDGE - a 1-2 px line through the gap drawn at the overlap center -
    so they become ONE connected component. A solid rect fill is not
    used: it would give a broken stroke the look of a filled blob.
    Returns (th, merged_bool).
    """
    import cv2
    merged = False
    for _ in range(_MAX_MERGES):
        n, labels, cc, _ = cv2.connectedComponentsWithStats(th, 8)
        if n <= 2:
            break
        comps = [(i, cc[i, 0], cc[i, 1], cc[i, 2], cc[i, 3], cc[i, 4])
                 for i in range(1, n)]
        best = None
        for ai in range(len(comps)):
            for bi in range(ai + 1, len(comps)):
                ia, xa, ya, wa, ha, aa = comps[ai]
                ib, xb, yb, wb, hb, ab = comps[bi]
                if aa < min_area or ab < min_area:
                    continue
                if ya + ha <= yb:                    # a above b: vertical bridge
                    gap_px, ov = yb - (ya + ha), min(xa + wa, xb + wb) - max(xa, xb)
                    ov_ref = min(wa, wb)
                    bx = (max(xa, xb) + min(xa + wa, xb + wb)) // 2
                    (x1, y1), (x2, y2) = (bx, ya + ha), (bx, yb)
                elif yb + hb <= ya:                  # b above a: vertical bridge
                    gap_px, ov = ya - (yb + hb), min(xa + wa, xb + wb) - max(xa, xb)
                    ov_ref = min(wa, wb)
                    bx = (max(xa, xb) + min(xa + wa, xb + wb)) // 2
                    (x1, y1), (x2, y2) = (bx, yb + hb), (bx, ya)
                elif xa + wa <= xb:                  # a left of b: horizontal bridge
                    gap_px, ov = xb - (xa + wa), min(ya + ha, yb + hb) - max(ya, yb)
                    ov_ref = min(ha, hb)
                    by = (max(ya, yb) + min(ya + ha, yb + hb)) // 2
                    (x1, y1), (x2, y2) = (xa + wa, by), (xb, by)
                elif xb + wb <= xa:                  # b left of a: horizontal bridge
                    gap_px, ov = xa - (xb + wb), min(ya + ha, yb + hb) - max(ya, yb)
                    ov_ref = min(ha, hb)
                    by = (max(ya, yb) + min(ya + ha, yb + hb)) // 2
                    (x1, y1), (x2, y2) = (xb + wb, by), (xa, by)
                else:
                    continue
                if gap_px < 0 or gap_px > gap or ov < overlap * ov_ref:
                    continue
                bx0, bx1 = min(xa, xb), max(xa + wa, xb + wb)
                by0, by1 = min(ya, yb), max(ya + ha, yb + hb)
                aspect = max(bx1 - bx0, by1 - by0) / max(min(bx1 - bx0, by1 - by0), 1)
                if aspect <= 3.5 and aa + ab >= min_area:
                    if best is None or aa + ab > best[0]:
                        best = (aa + ab, ia, ib, x1, y1, x2, y2)
        if best is None:
            break
        _, _, _, x1, y1, x2, y2 = best
        cv2.line(th, (x1, y1), (x2, y2), 255, 2)
        merged = True
    return th, merged


def _finish_preprocess(th, target=48, return_stats=False,
                       margin_frac=_DEFAULT_MARGIN_FRAC,
                       min_area_frac=_DEFAULT_MIN_AREA_FRAC,
                       empty_frac=_DEFAULT_EMPTY_FRAC,
                       grid_thickness=_DEFAULT_GRID_THICKNESS,
                       grid_thin=_DEFAULT_GRID_THIN,
                       grid_aspect=_DEFAULT_GRID_ASPECT,
                       grid_span=_DEFAULT_GRID_SPAN,
                       corner_area=_DEFAULT_CORNER_AREA,
                       corner_span=_DEFAULT_CORNER_SPAN,
                       merge_gap=_DEFAULT_MERGE_GAP,
                       merge_overlap=_DEFAULT_MERGE_OVERLAP):
    """Phase 1 component cleanup + letterboxed resize -> (48,48) float32.

    1. margin strip (default 10% - kills near-border clutter; all fractions
       below use the FULL cell area so the empty semantics stay stable)
    2. drop TINY components (< min_area_frac of the full cell - the measured
       0.5-1% largest-component bucket was 13/13 false digits)
    3. drop LINE fragments:
       - touching two OPPOSITE borders: thin (area/length < grid_thickness)
         AND elongated (bbox aspect > grid_aspect) - lines run edge-to-edge
       - touching ANY border: thin (area/length < grid_thin) AND elongated
         AND spanning >= grid_span of the region - one-sided remnants
       - digits are thicker than grid_thin and touch at most one border
    4. drop CORNER fragments: touching two adjacent borders, spanning >=
       corner_span of the inner region in both dims, area < corner_area
       (L-shaped remnants)
    5. conditionally rejoin split stroke pieces (7 stem+bar, ...) with a
       THIN BRIDGE - only pieces >= min_area_frac may merge, so noise never
       fuses into a phantom digit
    6. keep ALL surviving components (no largest-component-only step):
       digits are never truncated, a digit + stray fragment goes to the CNN
       whole, and the CNN sees the same real-photo look it was trained on
    7. conservative EMPTY: the largest surviving component below empty_frac
       of the full cell -> all-black (measured 0.5-1% bucket was 13/13
       false digits); strokes above the threshold are never rejected
    8. tight crop with 12% padding, aspect-preserving letterboxed resize

    With return_stats=True returns (final, stats): component diagnostics +
    the comps/input stage images.
    """
    import cv2
    h, w = th.shape
    m = max(1, int(round(min(h, w) * margin_frac)))
    th[:m, :] = 0
    th[-m:, :] = 0
    th[:, :m] = 0
    th[:, -m:] = 0
    inner = th[m:h - m, m:w - m]           # component work on the stripped
    hh, ww = inner.shape                   # region (its border = the strip edge)
    min_area = h * w * min_area_frac
    removed_tiny = removed_grid = removed_corner = 0
    comp_count = 0
    if (inner > 0).any():
        n, labels, cc, _ = cv2.connectedComponentsWithStats(inner, 8)
        comp_count = int(n - 1)
        for i in range(1, n):
            x, y, cw, chh, area = cc[i, 0], cc[i, 1], cc[i, 2], cc[i, 3], cc[i, 4]
            if area < min_area:
                removed_tiny += 1
                inner[labels == i] = 0
                continue
            length = max(cw, chh)
            thin = area / max(length, 1) < grid_thickness
            elongated = length / max(min(cw, chh), 1) > grid_aspect
            horizontal_line = x == 0 and x + cw == ww     # spans edge to edge
            vertical_line = y == 0 and y + chh == hh
            if (horizontal_line or vertical_line) and thin and elongated:
                removed_grid += 1
                inner[labels == i] = 0
                continue
            touch = (x == 0 or x + cw == ww or y == 0 or y + chh == hh)
            thin2 = area / max(length, 1) < grid_thin
            if touch and thin2 and elongated and length >= grid_span * max(hh, ww):
                removed_grid += 1
                inner[labels == i] = 0
                continue
            corner = (((x == 0 or x + cw == ww) and (y == 0 or y + chh == hh)) and
                      cw >= corner_span * ww and chh >= corner_span * hh and
                      area < corner_area * h * w)
            if corner:
                removed_corner += 1
                inner[labels == i] = 0
        inner, merged = _merge_split_pieces(inner, hh, ww, min_area,
                                            gap=merge_gap, overlap=merge_overlap)
    else:
        merged = False
    th_comps = inner.copy()
    n, labels, cc, _ = cv2.connectedComponentsWithStats(inner, 8)
    largest_frac = 0.0
    if n > 1:
        areas = cc[1:, 4]
        largest_frac = float(areas.max()) / (h * w)
        if areas.max() < h * w * empty_frac:   # conservative empty: nothing
            inner[:] = 0                       # credible survived
    ys, xs = np.nonzero(inner)
    if len(xs) == 0:
        final = np.zeros((target, target), dtype=np.float32)
    else:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = int(max(x1 - x0, y1 - y0) * 0.12) + 2
        crop = inner[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1]
        ch, cw = crop.shape
        scale = target / max(ch, cw)
        nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target, target), dtype=np.float32)
        y0c, x0c = (target - nh) // 2, (target - nw) // 2
        canvas[y0c:y0c + nh, x0c:x0c + nw] = resized
        final = canvas / 255.0
    if return_stats:
        surviving_ink_frac = float((inner > 0).sum()) / (h * w)
        return final, {"comp_count": comp_count,
                       "largest_comp_frac": largest_frac,
                       "surviving_ink_frac": surviving_ink_frac,
                       "removed_tiny": removed_tiny, "removed_grid": removed_grid,
                       "removed_corner": removed_corner,
                       "merged": merged,
                       "stages": {"comps": th_comps, "input": final[..., None]}}
    return final


def _preprocess_new(cell, target=48, return_stats=False, **kwargs):
    """Phase 1 preprocessing: threshold then _finish_preprocess.

    Thresholding stays lenient and simple: adaptive (block 15, constant 7,
    inverted) primary; a global Otsu pass ONLY when adaptive is clearly
    degenerate (washed-out/low-contrast cell: < 0.5% ink AND std > 18).
    The fallback is a pure degeneracy gate - the two thresholds are never
    compared on ink fraction. All component decisions live in
    `_finish_preprocess` (see its docstring); `**kwargs` are forwarded
    there (margin_frac, min_area_frac, empty_frac, corner_span, ... for
    A/B sweeps).
    """
    import cv2
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    if cell.dtype != np.uint8:
        cell = np.clip(cell, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 7)
    threshold_used = "adaptive"
    if (th > 0).mean() < 0.005 and blur.std() > 18:
        th = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        threshold_used = "otsu"
    th_raw = th.copy()
    final, fin = _finish_preprocess(th, target, return_stats=True, **kwargs)
    if not return_stats:
        return final[..., None]
    stats = {"threshold_used": threshold_used,
             "th_ink_frac": float((th_raw > 0).mean()),
             "comp_count": fin["comp_count"],
             "largest_comp_frac": fin["largest_comp_frac"],
             "surviving_ink_frac": fin["surviving_ink_frac"],
             "removed_tiny": fin["removed_tiny"],
             "removed_grid": fin["removed_grid"],
             "removed_corner": fin["removed_corner"],
             "merged": fin["merged"],
             "stages": {"original": cell, "blur": blur, "thresh": th_raw,
                        "comps": fin["stages"]["comps"],
                        "input": fin["stages"]["input"]}}
    return final[..., None], stats


def preprocess_cell(cell, target=48, margin_frac=_DEFAULT_MARGIN_FRAC,
                    min_area_frac=_DEFAULT_MIN_AREA_FRAC,
                    empty_frac=_DEFAULT_EMPTY_FRAC,
                    corner_span=_DEFAULT_CORNER_SPAN,
                    grid_thin=_DEFAULT_GRID_THIN, **kwargs):
    """Phase 1 lenient preprocessing -> (48,48,1) float32.

    Adaptive threshold (block 15, constant 7, inverted) with an Otsu
    fallback ONLY when adaptive is clearly degenerate, then the component
    cleanup in `_finish_preprocess`: margin strip, line/corner/tiny
    fragment removal by SHAPE, split strokes conditionally re-joined with
    a thin bridge, all surviving components kept, conservative empty,
    letterboxed resize. The sweep-relevant parameters (margin_frac,
    min_area_frac, empty_frac, corner_span, grid_thin) are exposed;
    the rest of `_finish_preprocess`'s tunables pass through **kwargs.
    """
    return _preprocess_new(cell, target, margin_frac=margin_frac,
                           min_area_frac=min_area_frac, empty_frac=empty_frac,
                           corner_span=corner_span, grid_thin=grid_thin,
                           **kwargs)


def preprocess_cell_stats(cell, target=48, **kwargs):
    """(final, stats) - the processed cell plus per-cell diagnostics.

    Diagnostic companion for benchmark --cell-dump / --cell-montage: the
    threshold path taken, thresholded foreground fraction, component counts,
    removed-fragment counts, merge flag, and the stage images - all produced
    by the CURRENT pipeline (same code path the CNN classifies). The same
    tunables as `preprocess_cell` are accepted for sweep runs.
    """
    return _preprocess_new(cell, target, return_stats=True, **kwargs)
