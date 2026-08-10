"""Generate the narrative figures for CAPSTONE_NARRATIVE.md.

Figures:
  fig_f2_pipeline_generations.png - the SAME two cells (one digit, one empty)
        through ALL THREE preprocessing generations: P0 (legacy, frozen),
        P1-style (RECONSTRUCTED from the study-era notes - the exact P1 code
        was superseded by P2), and P2 (current): raw cell -> what each
        generation sends to the CNN.
  fig_f3_stage_strip.png          - one cell through every stage of the
        current (P2) preprocessing, with the per-cell diagnostics.
  fig_f4_fragment_anatomy.png     - the EMPTY-cell fragment failure: the
        thresholded cell with every removed component boxed in red and
        labeled by the shape rule that removed it, vs the cleaned output.
  fig_f6a_per_digit_accuracy.png  - per-digit accuracy of the 24-sample
        evolution runs, parsed from the run logs.
  fig_f6b_wrong_cell_confidence.png - wrong-cell confidence cases
        (Case A conf<0.5, Case B conf>0.9) of the same runs.
  fig_f7_solver_nodes.png         - solver node counts (mean/max) + node-limit
        hits per run: the "node explosion = CNN-corrupted puzzle" diagnostic.
  fig_f8_solver_case.png          - one puzzle, 3 panels: ground-truth grid,
        recognized grid with errors highlighted, solution.
  fig_f9_margin_sweep.png         - the P2 parameter sweep (margin/empty/
        corner_span): definite-cell accuracy + exact grids.

F10 (confusion matrix) and F11 (position heatmaps) + the complete
`final_metrics.json` bundle come from `tools/final_metrics_figures.py`
(they are final-run-only figures, not log-history figures).

Refresh protocol after the final 40-epoch run: re-run the benchmark, then
re-run this script - all chart values and the T1 metric JSON are parsed
fresh from the logs (update RUNS below if a log name changes).
"""
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "narrative_figures")
os.makedirs(OUT, exist_ok=True)

RESULTS = os.path.join(REPO, "results")
RUNS_DIR = os.path.join(RESULTS, "runs")
CELLS_DIR = os.path.join(RESULTS, "cells")
FIG_DIR = os.path.join(RESULTS, "figures")

DATA = os.path.join(REPO, "data")
TEST_IMG = os.path.join(DATA, "test", "images")
TEST_META = os.path.join(DATA, "test", "metadata.jsonl")

# --------------------------------------------------------------------------- helpers

def imread_any(path):
    """cv2 first, Pillow fallback (cv2 cannot decode some webp files)."""
    img = cv2.imread(path)
    if img is not None:
        return img
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))[:, :, ::-1].copy()


def parse_flags(flags):
    """(9,9,10) GT flags -> (9,9) ints: 0 empty, 1-9 digit, -1 candidate-only.

    flag[0] == 1 marks a cell holding a digit; the digit is the one-hot
    position among flags[1:10]. Empty cells are all-zero. Cells with more
    than one digit flag are candidate-only (non-definite).
    """
    grid = np.full((9, 9), -1, dtype=int)
    for r in range(9):
        for c in range(9):
            f = flags[r][c]
            dig = [k for k in range(1, 10) if f[k] == 1]
            if len(dig) == 1:
                grid[r, c] = dig[0]
            elif len(dig) == 0 and f[0] == 0:
                grid[r, c] = 0
    return grid


def load_meta():
    rows = []
    with open(TEST_META, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def first_useful_cells():
    """A real puzzle: one definite digit cell and one definite empty cell.

    Prefers an EMPTY cell where the legacy pipeline leaves ink (the fragment
    failure the P2 cleanup removes) so the F2 strip shows the contrast.
    """
    sys.path.insert(0, REPO)
    from sudoku_core import four_point_transform, extract_cells
    from digit_cnn import preprocess_cell_legacy, preprocess_cell

    have = set(os.listdir(TEST_IMG))
    for row in load_meta():
        name = os.path.basename(row["file_name"])
        if name not in have:
            continue
        grid = parse_flags(row["cells"])
        img = imread_any(os.path.join(TEST_IMG, name))
        kp = np.array(row["keypoints"], dtype=np.float32).reshape(4, 2)
        pts = kp[[0, 3, 2, 1]]  # TL,BL,BR,TR -> TL,TR,BR,BL
        warped = four_point_transform(img, pts, size=600)
        cells = extract_cells(warped, size=600)
        if len(cells) != 81:
            continue
        digit_cells = [(r, c, grid[r, c]) for r in range(9) for c in range(9)
                       if 1 <= grid[r, c] <= 9]
        empty_cells = [(r, c) for r in range(9) for c in range(9)
                       if grid[r, c] == 0]
        if not digit_cells or not empty_cells:
            continue
        er, ec = empty_cells[0]
        legacy = preprocess_cell_legacy(cells[er * 9 + ec])
        p2 = preprocess_cell(cells[er * 9 + ec])
        if float((legacy > 0.5).mean()) > 0.005 and float((p2 > 0.5).mean()) < 0.001:
            return cells, grid, (digit_cells[0], (er, ec))
    for row in load_meta():
        name = os.path.basename(row["file_name"])
        if name not in have:
            continue
        grid = parse_flags(row["cells"])
        img = imread_any(os.path.join(TEST_IMG, name))
        kp = np.array(row["keypoints"], dtype=np.float32).reshape(4, 2)
        pts = kp[[0, 3, 2, 1]]
        warped = four_point_transform(img, pts, size=600)
        cells = extract_cells(warped, size=600)
        digit_cells = [(r, c, grid[r, c]) for r in range(9) for c in range(9)
                       if 1 <= grid[r, c] <= 9]
        empty_cells = [(r, c) for r in range(9) for c in range(9)
                       if grid[r, c] == 0]
        if digit_cells and empty_cells:
            return cells, grid, (digit_cells[0], (empty_cells[0][0], empty_cells[0][1]))
    raise RuntimeError("no usable puzzle found in data/test")


def show(img, size=160, nearest=False):
    """Gray -> BGR, upscaled, for display."""
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    s = size / max(h, w)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=interp)


def caption_bar(text, width=160, height=26, scale=0.45):
    """A white bar with dark caption text, placed under a panel."""
    bar = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (3, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (20, 20, 20), 1, cv2.LINE_AA)
    return bar


def panel(img, text, size=160, nearest=False):
    im = show(img, size=size, nearest=nearest)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im
    bar = caption_bar(text, width=im.shape[1])
    return np.vstack([im, bar])


def make_row(label, cols, label_width=150):
    """A labeled row: caption bar + white filler on the left, panels to the
    right, all columns padded to one common height."""
    h = max(c.shape[0] for c in cols)
    cols = [np.vstack([c, np.full((h - c.shape[0], c.shape[1], 3), 255, np.uint8)])
            for c in cols]
    lab = np.vstack([caption_bar(label, width=label_width, height=26),
                     np.full((h - 26, label_width, 3), 255, np.uint8)])
    return np.hstack([lab, *cols])


# --------------------------------------------------------------------------- F2

def preprocess_p1_style(cell, target=48):
    """P1-era preprocessing, RECONSTRUCTED from the study-era notes (the
    exact P1 code was superseded by P2; only P0 is frozen and P2 current).

    The study-era pipeline (README "Preprocessing" of that period):
    Gaussian blur -> adaptive threshold (block 15, constant 7, inverted) ->
    strip an 8% border margin -> keep the LARGEST connected component ->
    tight crop with 10% padding -> resize to 48x48.
    """
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    if cell.dtype != np.uint8:
        cell = np.clip(cell, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 7)
    h, w = th.shape
    m = max(1, int(min(h, w) * 0.08))
    th[:m, :] = 0
    th[-m:, :] = 0
    th[:, :m] = 0
    th[:, -m:] = 0
    n, labels, cc, _ = cv2.connectedComponentsWithStats(th, 8)
    if n > 1:
        areas = cc[1:, 4]
        keep = int(np.argmax(areas)) + 1
        th[labels != keep] = 0
        if cc[keep, 4] < h * w * 0.005:
            th[:] = 0
    ys, xs = np.nonzero(th)
    if len(xs) == 0:
        return np.zeros((target, target, 1), dtype=np.float32)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int((x1 - x0) * 0.1) + 2
    crop = th[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1]
    resized = cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[..., None]


def fig_f2():
    from digit_cnn import preprocess_cell_legacy, preprocess_cell
    cells, grid, (digit, empty) = first_useful_cells()
    dr, dc, dval = digit
    er, ec = empty

    dcell = cells[dr * 9 + dc]
    ecell = cells[er * 9 + ec]
    p0d = preprocess_cell_legacy(dcell)
    p1d = preprocess_p1_style(dcell)
    p2d = preprocess_cell(dcell)
    p0e = preprocess_cell_legacy(ecell)
    p1e = preprocess_p1_style(ecell)
    p2e = preprocess_cell(ecell)

    row_digit = make_row(f"Digit cell (GT {dval})", [
        panel(dcell, "Raw cell", 150, nearest=True),
        panel(p0d, "P0 - legacy (frozen)", 150, nearest=True),
        panel(p1d, "P1 - reconstructed", 150, nearest=True),
        panel(p2d, "P2 - current", 150, nearest=True),
    ])
    row_empty = make_row("Empty cell (GT 0)", [
        panel(ecell, "Raw cell", 150, nearest=True),
        panel(p0e, "P0 - legacy (frozen)", 150, nearest=True),
        panel(p1e, "P1 - reconstructed", 150, nearest=True),
        panel(p2e, "P2 - current", 150, nearest=True),
    ])

    w = max(row_digit.shape[1], row_empty.shape[1])
    if row_digit.shape[1] < w:
        row_digit = np.hstack([row_digit, np.full((row_digit.shape[0], w - row_digit.shape[1], 3), 255, np.uint8)])
    if row_empty.shape[1] < w:
        row_empty = np.hstack([row_empty, np.full((row_empty.shape[0], w - row_empty.shape[1], 3), 255, np.uint8)])
    canvas = np.vstack([row_digit, row_empty])
    out = os.path.join(OUT, "fig_f2_pipeline_generations.png")
    cv2.imwrite(out, canvas)
    print("F2 ->", out)
    print("     empty cell (r,c) =", (er, ec),
          "| ink: P0", round(float((p0e > 0.5).mean()), 4),
          "P1", round(float((p1e > 0.5).mean()), 4),
          "P2", round(float((p2e > 0.5).mean()), 4))


# --------------------------------------------------------------------------- F3

def fig_f3():
    from digit_cnn import preprocess_cell_stats
    cells, grid, (digit, empty) = first_useful_cells()
    dr, dc, dval = digit
    _, stats = preprocess_cell_stats(cells[dr * 9 + dc])
    stages = stats["stages"]
    panels = [
        panel(stages["original"], "1  Raw cell", 140, nearest=False),
        panel(stages["blur"], "2  Gaussian blur 3x3", 140, nearest=False),
        panel(stages["thresh"], "3  Adaptive threshold", 140, nearest=True),
        panel(stages["comps"], "4  Shape-based cleanup", 140, nearest=True),
        panel(stages["input"], "5  Letterboxed input 48x48", 140, nearest=True),
    ]
    hmax = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.full((hmax - p.shape[0], p.shape[1], 3), 255, np.uint8)]) for p in panels]
    canvas = np.hstack(panels)

    txt = (f"threshold={stats['threshold_used']}  th_ink={stats['th_ink_frac']:.3f}  "
           f"comps={stats['comp_count']}  largest={stats['largest_comp_frac']:.4f}  "
           f"removed tiny/grid/corner = {stats['removed_tiny']}/{stats['removed_grid']}/"
           f"{stats['removed_corner']}  merged={stats['merged']}")
    bar = np.full((30, canvas.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, txt, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    canvas = np.vstack([canvas, bar])

    out = os.path.join(OUT, "fig_f3_stage_strip.png")
    cv2.imwrite(out, canvas)
    print("F3 ->", out)
    print("     diagnostics:", txt)


# --------------------------------------------------------------------------- F4a

def _classify_removed(inner, comps, h, w):
    """Classify the components of the margin-stripped threshold `inner` that
    vanish in the cleaned stage `comps`, using the SAME geometry rules as the
    pipeline (digit_cnn._finish_preprocess): tiny, edge-to-edge line,
    1-border line, L-shaped corner chunk, other."""
    n, labels, cc, _ = cv2.connectedComponentsWithStats(inner, 8)
    hh, ww = inner.shape
    min_area = h * w * 0.005
    out = []
    if n <= 1:
        return out
    for i in range(1, n):
        x, y, cw, chh, area = cc[i]
        if area == 0:
            continue
        mask = labels == i
        if (comps[mask] > 0).any():          # survived the cleanup
            continue
        length = max(cw, chh)
        thin = area / max(length, 1) < 6.0
        elongated = length / max(min(cw, chh), 1) > 4.0
        if area < min_area:
            label = "tiny"
        elif thin and elongated and ((x == 0 and x + cw == ww) or (y == 0 and y + chh == hh)):
            label = "edge-to-edge"
        elif (x == 0 or x + cw == ww or y == 0 or y + chh == hh) and \
             area / max(length, 1) < 2.5 and elongated and length >= 0.7 * max(hh, ww):
            label = "1-border line"
        elif ((x == 0 or x + cw == ww) and (y == 0 or y + chh == hh)) and \
             cw >= 0.4 * ww and chh >= 0.4 * hh and area < 0.06 * h * w:
            label = "corner chunk"
        else:
            label = "other"
        out.append((x, y, cw, chh, label))
    return out


def find_fragment_empty(max_puzzles=60):
    """An empty cell whose fragments SURVIVE the 10% margin strip and are then
    removed by the P2 shape rules - the case the anatomy figure must show.
    Scans the first `max_puzzles` test images; returns None if not found."""
    sys.path.insert(0, REPO)
    from sudoku_core import four_point_transform, extract_cells
    from digit_cnn import preprocess_cell_stats
    have = set(os.listdir(TEST_IMG))
    for n_puz, row in enumerate(load_meta()):
        if n_puz >= max_puzzles:
            break
        name = os.path.basename(row["file_name"])
        if name not in have:
            continue
        grid = parse_flags(row["cells"])
        img = imread_any(os.path.join(TEST_IMG, name))
        kp = np.array(row["keypoints"], dtype=np.float32).reshape(4, 2)
        pts = kp[[0, 3, 2, 1]]
        warped = four_point_transform(img, pts, size=600)
        cells = extract_cells(warped, size=600)
        digit_cells = [(r, c, grid[r, c]) for r in range(9) for c in range(9)
                       if 1 <= grid[r, c] <= 9]
        empty_cells = [(r, c) for r in range(9) for c in range(9)
                       if grid[r, c] == 0]
        if not digit_cells or not empty_cells:
            continue
        for er, ec in empty_cells:
            _, stats = preprocess_cell_stats(cells[er * 9 + ec])
            stages = stats["stages"]
            h, w = stages["thresh"].shape
            m = max(1, int(round(min(h, w) * 0.10)))
            inner = stages["thresh"][m:h - m, m:w - m]
            removed = _classify_removed(inner, stages["comps"], h, w)
            if removed:
                return cells, grid, digit_cells[0], (er, ec), removed
    return None


def fig_f4a():
    from digit_cnn import preprocess_cell_stats
    found = find_fragment_empty()
    if found is None:
        print("F4a -> skipped (no fragment-surviving empty cell found)")
        return
    cells, grid, digit, empty, removed = found
    er, ec = empty
    ecell = cells[er * 9 + ec]
    _, stats = preprocess_cell_stats(ecell)
    stages = stats["stages"]
    thresh = stages["thresh"]
    comps = stages["comps"]
    h, w = thresh.shape
    m = max(1, int(round(min(h, w) * 0.10)))
    inner = thresh[m:h - m, m:w - m]

    annotated = show(inner, size=150, nearest=True)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    s = 150 / max(inner.shape)
    for x, y, cw, chh, label in removed:
        p1 = (int(round(x * s)), int(round(y * s)))
        p2 = (int(round((x + cw) * s)), int(round((y + chh) * s)))
        cv2.rectangle(annotated, p1, p2, (0, 0, 255), 1)
        ty = max(10, p1[1] - 4)
        cv2.putText(annotated, label, (p1[0] + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1, cv2.LINE_AA)

    row = make_row("Empty cell (GT 0)", [
        panel(ecell, "Raw cell", 150, nearest=True),
        panel(annotated, "Thresholded, margin stripped", 150, nearest=False),
        panel(comps, "After shape-based cleanup", 150, nearest=True),
        panel(stages["input"], "CNN input (letterboxed)", 150, nearest=True),
    ])
    txt = (f"P2 removed {len(removed)} component(s): "
           f"{[l for _, _, _, _, l in removed]} | "
           f"tiny={stats['removed_tiny']} grid={stats['removed_grid']} "
           f"corner={stats['removed_corner']} | "
           f"largest surviving {stats['largest_comp_frac']:.4f} < 1% -> "
           f"forced EMPTY ({stats['surviving_ink_frac']:.4f} ink survives)")
    bar = np.full((30, row.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, txt, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    canvas = np.vstack([row, bar])

    out = os.path.join(OUT, "fig_f4_fragment_anatomy.png")
    cv2.imwrite(out, canvas)
    print("F4a ->", out)
    print("     empty cell (r,c) =", (er, ec),
          "| removed:", [l for _, _, _, _, l in removed])


# --------------------------------------------------------------------------- metrics from logs

RE = {
    "definite": re.compile(r"definite-cell acc: (\d+)/(\d+) \(([\d.]+)\)"),
    "digit": re.compile(r"DIGIT acc \(solved\): (\d+)/(\d+) \(([\d.]+)\)"),
    "empty": re.compile(r"empty acc:\s+(\d+)/(\d+) \(([\d.]+)\)"),
    "exact": re.compile(r"exact grids:\s+(\d+)/(\d+) \(([\d.]+)%\)"),
    "solve": re.compile(r"solve rate:\s+(\d+)/(\d+) \(([\d.]+)%\)"),
    "nodes": re.compile(r"solver nodes:\s+mean (\d+), max (\d+)\s+\(node-limit hits: (\d+)\)"),
    "perdigit": re.compile(r"per-digit accuracy on solved cells: \{([^}]*)\}"),
    "conf": re.compile(r"wrong-cell confidence: n=(\d+)\s+Case A \(conf<0\.5\): (\d+)\s+Case B \(conf>0\.9\): (\d+)"),
}

RUNS = [
    ("P0 legacy baseline", "baseline_run.log", "low-epoch model, P0 (legacy) preprocessing"),
    ("P1 first redesign", "ab_run.log", "low-epoch model, P1 preprocessing"),
    ("P2 current pipeline", "v5_run.log", "low-epoch model, P2 preprocessing"),
    ("P2 + stale smoke", "v5smoke_run.log", "P1-trained smoke model under P2 (train/inference mismatch)"),
    ("P2 + P2 smoke", "v5smoke2_run.log", "P2-retrained smoke model"),
    ("P3 engineering fixes", "fix_run_recognition.log", "P2-retrained smoke + P3 fixes"),
    ("GAP synth smoke (sealed 210)", "gap_smoke_synth_recognition.log", "3-epoch GAP, synthetic-only, sealed test"),
    ("GAP photo smoke (sealed 210)", "gap_smoke_photo_recognition.log", "3-epoch GAP, photo fine-tune, sealed test"),
    ("GAP photo FINAL 21-epoch (sealed 210)", "final_run.log", "FINAL 21-epoch GAP, photo fine-tune, sealed test"),
]
RUNS = [(n, os.path.join(RUNS_DIR, f), m) for n, f, m in RUNS]

F6_RUNS = ["P0 legacy baseline", "P1 first redesign", "P2 + P2 smoke", "P3 engineering fixes"]


def parse_run(name, log):
    text = open(log, encoding="utf-8", errors="replace").read()
    m = {k: (re.search(rx, text) if re.search(rx, text) else None) for k, rx in RE.items()}
    frac = lambda mm: (int(mm.group(1)), int(mm.group(2)), float(mm.group(3))) if mm else None
    pd = {}
    if m["perdigit"]:
        for pair in m["perdigit"].group(1).split(","):
            d, v = pair.split(":")
            pd[int(d)] = float(v)
    conf = tuple(int(x) for x in m["conf"].groups()) if m["conf"] else None
    nodes = tuple(int(x) for x in m["nodes"].groups()) if m["nodes"] else None
    return {
        "label": name,
        "log": log,
        "model": {r[0]: r[2] for r in RUNS}[name],
        "definite": frac(m["definite"]),
        "digit": frac(m["digit"]),
        "empty": frac(m["empty"]),
        "exact": frac(m["exact"]),
        "solve": frac(m["solve"]),
        "nodes": nodes,
        "per_digit": pd,
        "wrong_conf": conf,
    }


def parse_all():
    runs = {}
    for name, log, _model in RUNS:
        if not os.path.exists(log):
            print("!! missing log:", log)
            continue
        runs[name] = parse_run(name, log)
    return runs


# --------------------------------------------------------------------------- F6

def fig_f6a(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    digits = list(range(1, 10))
    labels = [name.replace("P2 + ", "P2 ").replace("P3 ", "P3 ").replace("P0 legacy baseline", "P0 baseline")
              .replace("P1 first redesign", "P1 redesign") for name in F6_RUNS]
    data = [runs[n]["per_digit"] for n in F6_RUNS]
    x = np.arange(len(digits))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (lab, d) in enumerate(zip(labels, data)):
        vals = [d.get(k, float("nan")) * 100 for k in digits]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in digits])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Accuracy on solved cells (%)")
    ax.set_ylim(55, 100)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-digit recognition accuracy across pipeline generations (24-puzzle HF sample)")
    fig.tight_layout()
    out = os.path.join(OUT, "fig_f6a_per_digit_accuracy.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("F6a ->", out)


def fig_f6b(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [n.replace("P2 + ", "P2 ").replace("P0 legacy baseline", "P0 baseline")
              .replace("P1 first redesign", "P1 redesign") for n in F6_RUNS]
    n_wrong = [runs[n]["wrong_conf"][0] for n in F6_RUNS]
    case_a = [runs[n]["wrong_conf"][1] for n in F6_RUNS]
    case_b = [runs[n]["wrong_conf"][2] for n in F6_RUNS]
    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - w, n_wrong, w, label="wrong cells (total)")
    ax.bar(x, case_a, w, label="Case A (conf < 0.5)")
    ax.bar(x + w, case_b, w, label="Case B (conf > 0.9)")
    for i, v in enumerate(n_wrong):
        ax.text(x[i] - w, v + 3, str(v), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Wrong cells (24-puzzle HF sample)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Wrong-cell confidence cases across pipeline generations")
    fig.tight_layout()
    out = os.path.join(OUT, "fig_f6b_wrong_cell_confidence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("F6b ->", out)


# --------------------------------------------------------------------------- F7

def fig_f7(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"].replace("P2 + ", "P2 ").replace("P0 legacy baseline", "P0 baseline")
              .replace("P1 first redesign", "P1 redesign") for r in runs.values()]
    means = [r["nodes"][0] for r in runs.values()]
    maxes = [r["nodes"][1] for r in runs.values()]
    hits = [r["nodes"][2] for r in runs.values()]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, means, 0.6, label="mean nodes", color="#4c72b0")
    ax.scatter(x, maxes, marker="^", color="#c44e52", zorder=3, label="max nodes")
    ax.set_yscale("log")
    ax.set_ylim(1, 10 ** 6)
    for i, (me, ma, hi) in enumerate(zip(means, maxes, hits)):
        ax.text(i, me * 1.3, f"{me:,}", ha="center", fontsize=7)
        ax.text(i, ma * 1.3, f"max {ma:,}", ha="center", fontsize=6.5, color="#c44e52")
        ax.text(i, 0.6, f"limit hits: {hi}", ha="center", fontsize=6.5, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("solver nodes (log scale)")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_title("Correction-search node counts per run: a node explosion (max / limit hits) "
                 "means the CNN corrupted the puzzle")
    fig.tight_layout()
    out = os.path.join(OUT, "fig_f7_solver_nodes.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("F7 ->", out)


# --------------------------------------------------------------------------- F8

def _csv_has_photo(csv_path, photo):
    import csv as _csv
    with open(csv_path, encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if os.path.basename(r["photo"]) == photo:
                return True
    return False


def fig_f8():
    """One puzzle, three 9x9 grids: ground truth, recognized (errors in red),
    solution. Source: a run's cell dump CSV + the sealed metadata + the
    solver. Uses 3bor2sty6q0d1.webp - the documented regression puzzle.
    Uses the FINAL run's dump when present (final_cells_recognition.csv),
    falling back to the P3-era fix run dump."""
    import csv
    from matplotlib.patches import Rectangle

    photo = "3bor2sty6q0d1.webp"
    csv_path = os.path.join(CELLS_DIR, "final_cells_recognition.csv")
    era = "final"
    if not (os.path.exists(csv_path)
            and _csv_has_photo(csv_path, photo)):
        csv_path = os.path.join(CELLS_DIR, "fix_cells_recognition_recognition.csv")
        era = "P3"
    if not os.path.exists(csv_path):
        print("F8 -> skipped (missing cell dump csv)")
        return
    gt = None
    for row in load_meta():
        if os.path.basename(row["file_name"]) == photo:
            gt = parse_flags(row["cells"])
            break
    if gt is None:
        print("F8 -> skipped (photo not in sealed metadata)")
        return

    rec = np.full((9, 9), -2, dtype=int)          # -2 = no row in the CSV
    with open(csv_path, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            if os.path.basename(r["photo"]) != photo:
                continue
            rec[int(r["row"]) - 1, int(r["col"]) - 1] = int(r["pred"])

    wrong = (rec != -2) & (gt >= 0) & (rec != gt)
    n_def = int(((gt >= 0) & (rec != -2)).sum())
    n_wrong = int(wrong.sum())

    work = np.where(gt == -1, 0, gt)
    sys.path.insert(0, REPO)
    from sudoku_core import solve_sudoku
    sol, ok = solve_sudoku(work)

    def draw_grid(ax, g, mark=None, title=""):
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.axis("off")
        for i in range(10):
            lw = 2.2 if i % 3 == 0 else 0.5
            ax.plot([i, i], [0, 9], color="k", lw=lw)
            ax.plot([0, 9], [i, i], color="k", lw=lw)
        for r in range(9):
            for c in range(9):
                v = g[r, c]
                if v == -1 or v == -2:
                    ax.text(c + 0.5, 8.5 - r, "?", ha="center", va="center",
                            fontsize=9, color="gray")
                elif v == 0:
                    continue
                else:
                    red = mark is not None and mark[r, c]
                    if red:
                        ax.add_patch(Rectangle((c + 0.02, 8.5 - r - 0.48), 0.96, 0.96,
                                               fc="#ffd5d5", ec="none", zorder=0))
                    ax.text(c + 0.5, 8.5 - r, str(v), ha="center", va="center",
                            fontsize=13, color="red" if red else "black", zorder=1)
        ax.set_title(title, fontsize=10)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    draw_grid(axes[0], gt, title="Ground truth (definite cells)")
    draw_grid(axes[1], rec, mark=wrong,
              title=f"Recognized ({era}): {n_wrong} wrong of {n_def} definite cells")
    draw_grid(axes[2], sol if ok else np.full((9, 9), 0),
              title="Solution (from GT givens)"
                    if ok else "No solution from definite givens")
    fig.suptitle(f"{photo} - what the correction search has to work with", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(OUT, "fig_f8_solver_case.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("F8 ->", out, f"| wrong {n_wrong}/{n_def} | solved={ok}")


# --------------------------------------------------------------------------- F9

def fig_f9():
    """The P2 parameter sweep (low-epoch model, 24-puzzle HF sample), from the
    v5_run sweep records: definite-cell accuracy + exact grids per setting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = ["margin 0.04", "margin 0.08", "margin 0.10\n(default)",
                "margin 0.12", "empty 0.005", "corner_span 0.3"]
    definite = [94.07, 96.12, 96.34, 96.20, 95.90, 96.34]
    exact = [12, 18, 20, 20, 19, 20]
    x = np.arange(len(settings))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(x, definite, 0.55, color="#4c72b0")
    bars[2].set_color("#dd8452")                 # the chosen default
    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=8)
    ax.set_ylabel("definite-cell accuracy (%)")
    ax.set_ylim(90, 98)
    for i, v in enumerate(definite):
        ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, exact, "o-", color="#c44e52", label="exact grids")
    ax2.set_ylabel("exact grids (of 24)", color="#c44e52")
    ax2.set_ylim(0, 24)
    for i, v in enumerate(exact):
        ax2.text(i, v + 0.7, str(v), ha="center", fontsize=8, color="#c44e52")
    ax.set_title("P2 parameter sweep (low-epoch model, 24-puzzle HF sample): "
                 "margin 0.10 is the optimum")
    fig.tight_layout()
    out = os.path.join(OUT, "fig_f9_margin_sweep.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("F9 ->", out)


# --------------------------------------------------------------------------- T1 JSON

def dump_metrics(runs):
    out = os.path.join(OUT, "t1_metrics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(runs, fh, indent=2, default=list)
    print("T1 metrics ->", out)


# --------------------------------------------------------------------------- main

def main():
    print("=== narrative figures ===")
    fig_f2()
    fig_f3()
    fig_f4a()
    runs = parse_all()
    fig_f6a(runs)
    fig_f6b(runs)
    fig_f7(runs)
    fig_f8()
    fig_f9()
    dump_metrics(runs)
    print("done.")


if __name__ == "__main__":
    main()
