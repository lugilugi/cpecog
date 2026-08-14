"""Generate the visual pipeline walkthrough figure (CAPSTONE_NARRATIVE.md §3.2).

Runs ONE real photo through the ACTUAL pipeline functions (no
reimplementation) and renders a labeled montage showing what the image looks
like at every stage:

  fig_walkthrough_pipeline.png (narrative_figures/)
    A  Input photo + detected quad (detect_grid_contour) + detection map
    B  Perspective warp (four_point_transform) + 9x9 cell mosaic (extract_cells)
    C  Per-cell preprocessing strips (preprocess_cell_stats): raw -> blur ->
       adaptive threshold -> shape cleanup -> letterboxed 48x48 input, with
       the per-cell diagnostics (threshold path, ink fractions, removals)
    D  Recognition (predict_cells_probs + temperature): recognized 9x9 +
       per-cell confidence heatmap
    E  Solution (solve_with_resensing) + the deployed live-app overlay

Run as `python tools/make_pipeline_walkthrough.py [--image <path>] [--out <name>]`
from the repo root. Needs the final weights + temperature sidecar
(models/digit_cnn.pth*). Exits NONZERO if the grid is not detected (no GT
fallback - the walkthrough shows the honest deployed path).
"""
import argparse
import os
import sys

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "narrative_figures")
os.makedirs(OUT, exist_ok=True)

WEIGHTS = os.path.join(REPO, "models", "digit_cnn.pth")
DEFAULT_IMAGE = os.path.join(REPO, "benchmark_data", "hf_test_sample",
                             "images", "0t1g4k7u4lec1.jpeg")
SIZE = 600
DPI = 150
GIVEN = "#1a237e"
FILL = "#00695c"
BEST = "#1f77b4"


def imread_any(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))[:, :, ::-1].copy()


def caption_bar(text, width=160, height=26, scale=0.45):
    """A white bar with dark caption text, placed under a panel."""
    bar = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (3, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (20, 20, 20), 1, cv2.LINE_AA)
    return bar


def show(img, size=160, nearest=False):
    """Gray -> BGR, upscaled, for display."""
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    s = size / max(h, w)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=interp)


def panel(img, text, size=160, nearest=False):
    im = show(img, size=size, nearest=nearest)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im
    bar = caption_bar(text, width=im.shape[1])
    return np.vstack([im, bar])


def stage_strip(cell, label):
    """F3-style strip: the 5 exposed preprocessing stages + diagnostics bar."""
    from digit_cnn import preprocess_cell_stats
    _, stats = preprocess_cell_stats(cell)
    st = stats["stages"]
    panels = [
        panel(st["original"], "1 raw", 160, nearest=False),
        panel(st["blur"], "2 blur 3x3", 160, nearest=False),
        panel(st["thresh"], "3 adaptive thr.", 160, nearest=True),
        panel(st["comps"], "4 cleanup", 160, nearest=True),
        panel(st["input"], "5 48x48 input", 160, nearest=True),
    ]
    hmax = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.full((hmax - p.shape[0], p.shape[1], 3),
                                    255, np.uint8)]) for p in panels]
    strip = np.hstack(panels)
    diag = (f"{label} | {stats['threshold_used']} | "
            f"th_ink={stats['th_ink_frac']:.3f} | "
            f"comps={stats['comp_count']} | "
            f"largest={stats['largest_comp_frac']:.4f} | "
            f"tiny/grid/corner={stats['removed_tiny']}/"
            f"{stats['removed_grid']}/{stats['removed_corner']} | "
            f"merged={stats['merged']}")
    bar = np.full((28, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, diag, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([strip, bar])


def pick_cells(cells, grid):
    """Up to 3 representative cells, chosen from the REAL pipeline diagnostics:
    a digit (ink survives cleanup), an empty the CNN read as 0 whose raw
    threshold still had fragments (the cleanup story), and a split-stroke
    merge case when one exists."""
    from digit_cnn import preprocess_cell_stats
    info = []
    for i, c in enumerate(cells):
        _, st = preprocess_cell_stats(c)
        info.append((i, st["th_ink_frac"], st["largest_comp_frac"], st["merged"]))
    digits = [t for t in info if t[2] >= 0.05]
    digit = max(digits, key=lambda t: t[2]) if digits else max(info, key=lambda t: t[2])
    empties = [t for t in info
               if t[0] != digit[0]
               and grid[t[0] // 9, t[0] % 9] == 0 and t[2] < 0.01]
    empty = max(empties, key=lambda t: t[1]) if empties else None
    merged = [t for t in info
              if t[3] and t[0] != digit[0]
              and (empty is None or t[0] != empty[0])]
    third = max(merged, key=lambda t: t[2]) if merged else None
    return [t[0] for t in (digit, empty, third) if t is not None]


def grid_ax(ax, grid, title, fills=None, marks=None):
    """9x9 grid text panel: givens dark; `fills` cells teal (solution)."""
    ax.set_title(title, fontsize=10)
    ax.imshow(np.zeros((9, 9, 3), dtype=np.uint8))
    for i in range(10):
        lw = 1.6 if i % 3 == 0 else 0.4
        ax.axhline(i - 0.5, color="k", lw=lw)
        ax.axvline(i - 0.5, color="k", lw=lw)
    for r in range(9):
        for c in range(9):
            v = int(grid[r, c])
            if v == 0:
                continue
            color = FILL if (fills is not None and fills[r, c]) else GIVEN
            ax.text(c, r, str(v), ha="center", va="center", fontsize=12,
                    fontweight="bold", color=color)
            if marks is not None and marks[r, c]:
                ax.text(c, r, "*", ha="left", va="top", fontsize=9, color="red")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description="Visual pipeline walkthrough")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="input photo")
    parser.add_argument("--out", default="fig_walkthrough_pipeline",
                        help="output name (narrative_figures/<out>.png)")
    args = parser.parse_args()

    import torch
    import sudoku_core as sc
    from digit_cnn import (classify_preprocessed, load_digit_model,
                           load_temperature, predict_cells_probs)
    from live_solver import annotate_frame

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_digit_model(WEIGHTS, device=device)
    temperature = load_temperature(WEIGHTS)
    print(f"model {os.path.basename(WEIGHTS)} on {device}, T={temperature:.3f}")

    frame = imread_any(args.image)
    if frame is None:
        print(f"could not read {args.image}", file=sys.stderr)
        return 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"image {args.image}  {gray.shape[1]}x{gray.shape[0]}")

    # 1. detection
    quad, detmap = sc.detect_grid_contour(gray)
    if quad is None:
        print(f"ERROR: grid not detected in {args.image} "
              f"(no contour, no GT fallback). Try another photo.",
              file=sys.stderr)
        return 1
    quad_pts = quad.reshape(4, 2).astype(np.float64)
    print("detection: contour quad", quad_pts.round(1).tolist())

    # 2. warp + 3. cells
    warped = sc.four_point_transform(gray, quad, SIZE)
    cells = sc.extract_cells(warped, SIZE)
    mosaic = np.vstack([np.hstack(cells[r * 9:(r + 1) * 9]) for r in range(9)])
    print(f"warp: {warped.shape[0]}x{warped.shape[1]}, 81 cells extracted")

    # 5. recognition
    probs = predict_cells_probs(cells, model, device=device,
                                temperature=temperature)
    grid = probs.argmax(axis=1).reshape(9, 9)
    conf = probs.max(axis=1).reshape(9, 9)
    print("recognized grid:\n", grid)

    # 4. preprocessing strips on representative cells (after recognition so
    # the empty picker can use what the CNN actually read)
    chosen = pick_cells(cells, grid)
    strips = [stage_strip(cells[i], f"cell ({i // 9},{i % 9})") for i in chosen]
    strips_stack = np.vstack(strips) if len(strips) > 1 else strips[0]

    # 6. solve
    stats = {}
    solved, ok, resensed = sc.solve_with_resensing(
        grid, probs, cells,
        lambda views: classify_preprocessed(views, model, device,
                                            temperature=temperature),
        stats=stats)
    print(f"solve: ok={ok} nodes={stats.get('nodes', 0)} "
          f"resensed={resensed}")
    if not ok:
        print("WARNING: no solution found for this photo (recognition errors)",
              file=sys.stderr)

    # 7. deployed overlay (the live-app look)
    overlay = annotate_frame(frame, quad_pts, grid, solved, ok)

    # ------------------------------------------------------------- figure
    fills = np.zeros((9, 9), dtype=bool) if not ok else ((solved != 0) & (grid == 0))
    fig = plt.figure(figsize=(15, 17))
    gs = fig.add_gridspec(12, 2, height_ratios=[1.1, 0.5, 0.9, 0.5, 1.0, 1.0,
                                                 1.0, 1.2, 0.7, 1.2, 0.7, 0.7],
                          hspace=0.6, wspace=0.15)
    fig.suptitle(f"Visual pipeline walkthrough — {os.path.basename(args.image)}\n"
                 "each panel shows the image under the code that produced it "
                 f"(T={temperature:.2f})",
                 fontsize=13)

    ax = fig.add_subplot(gs[0, :])
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    q = np.vstack([quad_pts, quad_pts[0]])
    ax.plot(q[:, 0], q[:, 1], color="lime", lw=2.5)
    ax.set_title("1  Input photo — green quad = detect_grid_contour() "
                 "(adaptive threshold → largest 4-point contour)",
                 fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(detmap, cmap="gray")
    ax.set_title("detection map (thresholded)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(warped, cmap="gray")
    ax.set_title("2  Perspective warp — four_point_transform() → "
                 f"{SIZE}×{SIZE}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[2, :])
    ax.imshow(mosaic, cmap="gray")
    ax.set_title("3  81 row-major cells — extract_cells() (fractional "
                 "boundaries, cell 0 = top-left)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[3:6, :])
    ax.imshow(strips_stack, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("4  Per-cell preprocessing — preprocess_cell_stats(): "
                 "raw → blur → adaptive threshold → shape cleanup → "
                 "letterboxed 48×48 input (diagnostics bar below each strip)",
                 fontsize=10, loc="left")

    ax = fig.add_subplot(gs[7, 0])
    grid_ax(ax, grid, "5a  Recognized 9×9 — predict_cells_probs() + argmax")
    ax = fig.add_subplot(gs[7, 1])
    im = ax.imshow(conf, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("5b  Per-cell confidence (max softmax, T-scaled)",
                 fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[9, 0])
    grid_ax(ax, solved, f"6  Solution — solve_with_resensing()  "
            f"(nodes={stats.get('nodes', 0):,}, resensed={resensed})",
            fills=fills)
    ax = fig.add_subplot(gs[9, 1])
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.set_title("7  Deployed overlay — the live app look: white = recognized "
                 "given, yellow = solver-filled", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # status footer
    ax = fig.add_subplot(gs[11, :])
    ax.axis("off")
    status = (f"solve ok={ok}  nodes={stats.get('nodes', 0):,}  "
              f"resensed={resensed}  temperature={temperature:.2f}")
    ax.text(0.5, 0.5, status, ha="center", va="center", fontsize=10,
            color="#2e7d32" if ok else "#c62828")

    out = os.path.join(OUT, f"{args.out}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
