"""Labeled benchmark: run the pipeline on real sudoku photos with ground
truth and report detection / recognition / solve performance.

Every run reports TWO separately-labeled benchmark modes (--mode both):

- RECOGNITION/SOLVER: the ground-truth grid corners are warped directly, so
  the numbers isolate CNN recognition + solving from grid detection.
- END-TO-END: corners are ignored and the grid is found automatically
  (contour -> lines). Detection misses are counted honestly in a
  "grid detection" rate and excluded from the recognition metrics.

Two benchmark sources:

1. wichtounet/sudoku_dataset (GitHub clone, folder mode)
       python benchmark.py --data-dir path/to/sudoku_dataset --model digit_cnn.pth
   Uses outlines_sorted.csv + images/ (.dat ground truth). VERSIONS:
   - V2 (images/, givens in .dat) - the default and only version used here;
     the version is detected automatically and printed.
   - mixed (mixed/, complete 81-digit grids) - NOT mixed with V2: pass
     --data-dir pointing at a folder whose outlines csv lists mixed images,
     and the report switches to full-grid metrics.

2. Lexski/sudoku-image-recognition (HuggingFace, 1400 images, explicit
   train/val/test splits; cells are (9,9,10) flags + 4 grid keypoints)
       python benchmark.py --hf --hf-split test --model digit_cnn.pth
       # or, without the datasets library, download data/<split> (images/
       # + metadata.jsonl) and run:
       python benchmark.py --hf-dir path/to/data/test --model digit_cnn.pth

   Evaluation semantics (per the dataset README): flag 0 = solved,
   flags 1-9 = digit presence. A cell is DEFINITE when it is solved with
   exactly one digit, or unsolved with no candidates (empty). Cells holding
   candidates are skipped. The headline recognition metric is the SOLVED-CELL
   DIGIT accuracy; keypoints are reordered (TL,BL,BR,TR -> TL,TR,BR,BL).

Prints per-puzzle grids with wrong cells marked '*' and saves a figure with
example photos + recognized vs. ground-truth grids.
"""
import argparse
import csv
import glob
import json
import os
import tempfile
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np

import sudoku_core as sc
from digit_cnn import load_digit_model, load_temperature, predict_cells_probs, \
    classify_preprocessed, preprocess_cell_stats

HF_BASE = ("https://huggingface.co/datasets/Lexski/sudoku-image-recognition/"
           "resolve/main/data")


def imread_any(path):
    """cv2.imread with a Pillow fallback; None if the file is missing/undecodable.

    A sample folder (--hf-dir) legitimately lacks most split images - the
    caller skips missing ones. The Pillow fallback covers genuine cv2 decode
    failures without turning a missing file into a hard error.
    """
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
    except (OSError, ValueError):
        return None
    return arr[:, :, ::-1].copy()               # RGB -> BGR


def _pick_device(device):
    """auto -> cuda when available; explicit `cuda` without a GPU errors out
    instead of silently falling back to CPU (which used to distort solver
    node stats and look like a regression)."""
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda but torch.cuda.is_available() is False")
    return device


# ---------------------------------------------------------------- folder mode
def parse_dat(path):
    """Read a .dat puzzle file -> 9x9 int array (0 = empty cell)."""
    lines = open(path, encoding="utf-8", errors="ignore").read().strip().splitlines()
    rows = []
    for line in lines:
        toks = line.split()
        if len(toks) >= 9 and all(t.isdigit() and int(t) <= 9 for t in toks[:9]):
            rows.append([int(t) for t in toks[:9]])
        elif len(line.strip()) == 9 and all(c.isdigit() for c in line.strip()):
            rows.append([int(c) for c in line.strip()])
        if len(rows) == 9:
            break
    assert len(rows) == 9, f"could not parse 9 rows from {path}"
    return np.array(rows, dtype=int)


def load_outlines(csv_path):
    """outlines_sorted.csv -> {filename: (absolute_image_path, 4x2 corners)}."""
    outlines = {}
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            rel = parts[0].replace("\\", "/").lstrip("./")
            img_path = os.path.normpath(os.path.join(base_dir, rel))
            try:
                pts = np.array([float(x) for x in parts[1:9]]).reshape(4, 2)
            except ValueError:
                continue
            outlines[os.path.basename(img_path)] = (img_path, pts)
    return outlines


def find_outlines_csv(data_dir):
    for cand in glob.glob(os.path.join(data_dir, "**", "outlines_sorted.csv"),
                          recursive=True):
        return cand
    return None


def detect_dataset_version(outlines, first_name):
    """Detect the wichtounet dataset version from the .dat file of the FIRST
    image the outlines csv references: mixed (complete grids, no zeros) vs
    V2 (givens with empties). V2 and mixed are never mixed in one run."""
    img_path, _ = outlines[first_name]
    dat_path = os.path.join(os.path.dirname(img_path),
                            os.path.splitext(first_name)[0] + ".dat")
    if not os.path.exists(dat_path):
        return "unknown"
    grid = parse_dat(dat_path)
    if int((grid == 0).sum()) == 0:
        return "mixed (complete 81-digit grids)"
    return "V2 (givens, empties present)"


# ---------------------------------------------------------------- HF mode
def hf_cells_to_gt(cells):
    """(9,9,10) flags -> (gt_grid, definite_mask).

    flag 0 = solved; flags 1-9 = digit presence. Definite cells are solved
    cells with exactly one digit, and unsolved cells with no candidates
    (empty). Cells with candidates are NOT definite (skipped in accuracy).
    """
    cells = np.asarray(cells).reshape(9, 9, 10)
    gt = np.zeros((9, 9), dtype=int)
    definite = np.zeros((9, 9), dtype=bool)
    for r in range(9):
        for c in range(9):
            flags = cells[r, c]
            solved = int(flags[0]) == 1
            digits = [d for d in range(1, 10) if int(flags[d]) == 1]
            if solved and len(digits) == 1:
                gt[r, c] = digits[0]
                definite[r, c] = True
            elif not solved and len(digits) == 0:
                gt[r, c] = 0
                definite[r, c] = True
    return gt, definite


def load_hf_folder(hf_dir, frac=1.0, seed=0):
    """Load data/<split> downloaded from the HF repo (images/ + metadata.jsonl)."""
    meta = os.path.join(hf_dir, "metadata.jsonl")
    img_dir = os.path.join(hf_dir, "images")
    if not os.path.exists(meta):
        raise FileNotFoundError(meta)
    rows = [json.loads(l) for l in open(meta, encoding="utf-8") if l.strip()]
    rng = np.random.RandomState(seed)
    n = max(1, int(round(len(rows) * frac)))
    rows = [rows[i] for i in rng.permutation(len(rows))[:n]]
    puzzles = []
    for row in rows:
        name = os.path.basename(row["file_name"].replace("\\", "/"))
        img = imread_any(os.path.join(img_dir, name))
        if img is None:
            print("skip unreadable image:", name)
            continue
        kp = np.array(row["keypoints"], dtype=np.float32).reshape(4, 2)
        kp = kp[[0, 3, 2, 1]]                      # TL,BL,BR,TR -> TL,TR,BR,BL
        gt, definite = hf_cells_to_gt(row["cells"])
        puzzles.append({"name": name, "img": img, "kp": kp,
                        "gt": gt, "definite": definite})
    return puzzles


def load_hf_dataset(split, frac=1.0, seed=0):
    """Load via the datasets library (preinstalled on Colab)."""
    from datasets import load_dataset

    ds = load_dataset("Lexski/sudoku-image-recognition", split=split)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(ds))[:max(1, int(round(len(ds) * frac)))]
    puzzles = []
    for i in idx:
        row = ds[i]
        img = np.array(row["image"])               # RGB PIL -> BGR ndarray
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        kp = np.array(row["keypoints"], dtype=np.float32).reshape(4, 2)
        kp = kp[[0, 3, 2, 1]]                      # TL,BL,BR,TR -> TL,TR,BR,BL
        gt, definite = hf_cells_to_gt(row["cells"])
        puzzles.append({"name": f"hf[{i}]", "img": img, "kp": kp,
                        "gt": gt, "definite": definite})
    return puzzles


def download_hf_sample(split="test", n=24, seed=0, dest=None):
    """Lightweight HF sample: download metadata.jsonl + n seeded images with
    plain urllib (no datasets library, fast - a few MB instead of the whole
    dataset). Returns the folder path for load_hf_folder."""
    if dest is None:
        dest = os.path.join(tempfile.gettempdir(), f"hf_{split}_sample")
    img_dir = os.path.join(dest, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta_path = os.path.join(dest, "metadata.jsonl")
    if not os.path.exists(meta_path):
        urllib.request.urlretrieve(f"{HF_BASE}/{split}/metadata.jsonl", meta_path)
    rows = [json.loads(l) for l in open(meta_path, encoding="utf-8") if l.strip()]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(rows))[:min(n, len(rows))]
    got = 0
    for i in idx:
        name = os.path.basename(rows[i]["file_name"].replace("\\", "/"))
        out = os.path.join(img_dir, name)
        if not os.path.exists(out):
            urllib.request.urlretrieve(f"{HF_BASE}/{split}/images/{name}", out)
        got += 1
    print(f"downloaded {got} of {len(rows)} images of split '{split}' -> {dest}")
    return dest


# ---------------------------------------------------------------- evaluation
def valid_solution(g):
    for r in range(9):
        if sorted(g[r, :]) != list(range(1, 10)):
            return False
    for c in range(9):
        if sorted(g[:, c]) != list(range(1, 10)):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if sorted(g[br:br + 3, bc:bc + 3].flat) != list(range(1, 10)):
                return False
    return True


def detect_and_warp(gray, size=600):
    """-> (warped_grid, method) with method 'contour' / 'lines' / None.

    No blind center-crop fallback: when neither detector finds a grid the
    result is (None, None) so callers can count an honest detection miss.
    """
    quad, _ = sc.detect_grid_contour(gray)
    if quad is not None:
        return sc.four_point_transform(gray, quad, size), "contour"
    warped = sc.line_grid_quad(gray, size)
    if warped is not None:
        return warped, "lines"
    return None, None


def draw_grid_text(ax, grid, title, marks=None):
    ax.imshow(np.zeros((9, 9, 3), dtype=np.uint8))
    for i in range(10):
        lw = 3 if i % 3 == 0 else 1
        ax.axhline(i - 0.5, color="k", lw=lw)
        ax.axvline(i - 0.5, color="k", lw=lw)
    for r in range(9):
        for c in range(9):
            v = grid[r, c]
            wrong = marks is not None and marks[r, c]
            ax.text(c, r, str(v) if v else "", ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="red" if wrong else "black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


STAGE_TITLES = ["original", "blur", "thresh", "comps", "input"]
GROUP_TITLES = {
    "empty_to_digit": "TRUE EMPTY -> predicted DIGIT",
    "1_to_7": "TRUE 1 -> predicted 7",
    "7_to_1": "TRUE 7 -> predicted 1",
    "8_ok": "TRUE 8 -> CORRECT 8 (reference)",
    "other": "OTHER ERRORS",
}


def write_montages(groups, base_path, max_per_group=12):
    """Save one per-group stage montage PNG. Each row is a cell: info label +
    the preprocessing stages (original, blur, thresh, comps, input)."""
    written = 0
    for key, items in groups.items():
        if not items:
            continue
        n_shown = min(len(items), max_per_group)
        fig, axes = plt.subplots(n_shown, len(STAGE_TITLES) + 1,
                                 figsize=(15, 2.2 * n_shown + 1.2))
        axes = np.atleast_2d(axes)
        for j, title in enumerate(STAGE_TITLES):
            axes[0][j + 1].set_title(title, fontsize=10)
        for i in range(n_shown):
            row, stages = items[i]
            axes[i][0].axis("off")
            axes[i][0].text(0, 0.5, f"{row['photo']}\n({row['row']},{row['col']})\n"
                                    f"gt {row['gt']} -> pred {row['pred']}\n"
                                    f"conf {row['conf']:.2f}\n{row['threshold']}",
                            va="center", ha="left", fontsize=8,
                            transform=axes[i][0].transAxes)
            for j, st in enumerate(STAGE_TITLES):
                img = stages[st]
                vmax = 255.0 if np.issubdtype(np.asarray(img).dtype, np.uint8) else 1.0
                axes[i][j + 1].imshow(img, cmap="gray", vmin=0, vmax=vmax)
                axes[i][j + 1].set_xticks([])
                axes[i][j + 1].set_yticks([])
        fig.suptitle(f"{GROUP_TITLES[key]}  (showing {n_shown} of {len(items)} cells)",
                     fontsize=12)
        fig.tight_layout()
        out = base_path.replace(".png", f"_{key}.png")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved montage: {out} ({n_shown} cells)")
        written += 1
    return written


def evaluate_puzzles(puzzles, model, device, max_correction_cells=12,
                     max_examples=2, out_png=None, title="benchmark",
                     use_detection=False, cell_dump=None, cell_montage=None,
                     preprocess_kwargs=None, temperature=1.0):
    """Run the pipeline on a list of puzzle dicts and report metrics.

    Each puzzle: {name, img (BGR), kp (4x2 TL,TR,BR,BL) or None,
                  gt (9x9), definite (9x9 bool) or None (all definite)}.

    Two modes:
    - use_detection=False: RECOGNITION/SOLVER benchmark - the ground-truth
      grid corners (pz['kp']) are warped directly, isolating CNN recognition
      + solving from grid detection.
    - use_detection=True: END-TO-END benchmark - the corners are ignored and
      the grid is found automatically (contour -> lines). Puzzles where
      detection fails are counted as misses: they stay in the detection-rate
      denominator but are excluded from recognition metrics.

    `preprocess_kwargs` (margin_frac, empty_frac, corner_span, ...) are
    forwarded to every preprocessing call - preprocessing A/B sweeps share
    the same model and only change this dict.
    """
    methods = {"contour": 0, "lines": 0, "keypoints": 0, "failed": 0}
    n_detected = n_exact = n_solved = 0
    n_no_dup = n_gt_solved = 0
    n_oracle_tried = n_oracle_solved = 0
    cand_cells = cand_false_clues = 0
    total_resensed = 0
    cor_def = tot_def = 0
    cor_digit = tot_digit = 0
    cor_empty = tot_empty = 0
    per_digit_cor = {k: [0, 0] for k in range(1, 10)}
    examples = []
    node_stats = []
    n_total = len(puzzles)
    pos_cor = np.zeros((9, 9), dtype=int)
    pos_tot = np.zeros((9, 9), dtype=int)
    wrong_confs = []
    diag_rows = []
    diag_groups = {k: [] for k in GROUP_TITLES}
    for pz in puzzles:
        gt = pz["gt"]
        definite = pz.get("definite")
        if definite is None:
            definite = np.ones((9, 9), dtype=bool)
        gray = cv2.cvtColor(pz["img"], cv2.COLOR_BGR2GRAY)
        if use_detection:
            warped, method = detect_and_warp(gray)
            if warped is None:
                methods["failed"] += 1
                continue
        else:
            warped = sc.four_point_transform(gray, pz["kp"])
            method = "keypoints"
        methods[method] += 1
        n_detected += 1
        cells = sc.extract_cells(warped)
        pre = preprocess_kwargs or {}
        probs = predict_cells_probs(cells, model, device, temperature=temperature, **pre)
        grid = probs.argmax(1).reshape(9, 9)
        d = definite
        tot_def += int(d.sum())
        cor_def += int(((grid == gt) & d).sum())
        dig = d & (gt != 0)
        emp = d & (gt == 0)
        tot_digit += int(dig.sum())
        cor_digit += int((grid[dig] == gt[dig]).sum())
        tot_empty += int(emp.sum())
        cor_empty += int((grid[emp] == 0).sum())
        for k in range(1, 10):
            m = dig & (gt == k)
            per_digit_cor[k][1] += int(m.sum())
            per_digit_cor[k][0] += int((grid[m] == k).sum())
        pos_tot += d.astype(int)
        pos_cor += ((grid == gt) & d).astype(int)
        for i in range(81):
            if not d.flat[i]:
                continue
            r, c = divmod(i, 9)
            g, p = int(gt.flat[i]), int(grid.flat[i])
            conf = float(probs[i].max())
            if g != p:
                wrong_confs.append(conf)
            if cell_dump or cell_montage:
                _, ps = preprocess_cell_stats(cells[i], **pre)
                row = dict(photo=pz["name"], row=r + 1, col=c + 1, gt=g, pred=p,
                           conf=round(conf, 4), threshold=ps["threshold_used"],
                           fg_frac=round(ps["th_ink_frac"], 4),
                           post_ink_frac=round(ps.get("surviving_ink_frac", 0.0), 4),
                           comps=ps["comp_count"],
                           largest_frac=round(ps["largest_comp_frac"], 4),
                           tiny_removed=ps["removed_tiny"],
                           grid_removed=ps["removed_grid"],
                           corner_removed=ps["removed_corner"],
                           merged=int(ps["merged"]))
                diag_rows.append(row)
                if cell_montage and g != p:
                    key = ("empty_to_digit" if g == 0 else
                           "1_to_7" if (g, p) == (1, 7) else
                           "7_to_1" if (g, p) == (7, 1) else "other")
                    diag_groups[key].append((row, ps["stages"]))
                elif cell_montage and (g, p) == (8, 8):
                    diag_groups["8_ok"].append((row, ps["stages"]))
        if ((grid == gt) | ~d).all():
            n_exact += 1
        suspects, _ = sc.find_violated_cells(grid)
        if not suspects:
            n_no_dup += 1
        if (~d).any():
            cand_cells += int((~d).sum())
            cand_false_clues += int(((grid != 0) & ~d).sum())
        st = {}
        solved, ok, n_resensed = sc.solve_with_resensing(
            grid, probs, cells,
            lambda views: classify_preprocessed(views, model, device),
            max_correction_cells=max_correction_cells, stats=st,
            preprocess_kwargs=pre)
        node_stats.append((st.get("nodes", 0), st.get("limit_hit", False)))
        total_resensed += n_resensed
        gt_ok = ok and valid_solution(solved) and ((solved == gt) | ~d).all()
        if ok and valid_solution(solved):
            n_solved += 1
        if gt_ok:
            n_gt_solved += 1
        elif (~d).any():
            # oracle diagnostic: candidate-only cells emptied (their argmax
            # values are ignored by the metrics but reach the solver as
            # possible false clues). Shows how many failed puzzles solve when
            # those cells carry no false givens - NOT a production metric.
            oracle_grid = grid.copy()
            oracle_grid[~d] = 0
            ost = {}
            osol, ook, _ = sc.solve_with_resensing(
                oracle_grid, probs, cells,
                lambda views: classify_preprocessed(views, model, device),
                max_correction_cells=max_correction_cells, stats=ost)
            n_oracle_tried += 1
            if ook and valid_solution(osol) and ((osol == gt) | ~d).all():
                n_oracle_solved += 1
        if len(examples) < max_examples:
            examples.append((pz["name"], pz["img"], gt, definite, grid, solved, ok))

    mode_label = "END-TO-END (auto grid detection)" if use_detection else \
        "RECOGNITION/SOLVER (ground-truth corners)"
    print(f"benchmark mode:   {mode_label}")
    print(f"evaluated:        {n_detected}/{n_total} puzzles"
          + (f"  [{', '.join(f'{m} {c}' for m, c in methods.items() if c)}]" if n_total else ""))
    if use_detection and n_total:
        print(f"grid detection:   {n_detected}/{n_total} ({n_detected / n_total:.1%})")
    n = max(n_detected, 1)
    print(f"cells re-sensed:  {total_resensed} (constraint-guided re-sensing)")
    if node_stats:
        ns = [n for n, _ in node_stats]
        print(f"solver nodes:     mean {np.mean(ns):.0f}, max {max(ns)}  "
              f"(node-limit hits: {sum(h for _, h in node_stats)})")
    if tot_def:
        print(f"definite-cell acc: {cor_def}/{tot_def} ({cor_def / tot_def:.4f})")
    if tot_digit:
        print(f"DIGIT acc (solved): {cor_digit}/{tot_digit} ({cor_digit / tot_digit:.4f})"
              f"  <-- number-recognition headline")
    if tot_empty:
        print(f"empty acc:         {cor_empty}/{tot_empty} ({cor_empty / tot_empty:.4f})")
    print(f"exact grids:       {n_exact}/{n_detected} ({n_exact / n:.1%})  <-- definite cells only")
    print(f"no-duplicate grids:{n_no_dup}/{n_detected} ({n_no_dup / n:.1%})  (recognized full grid)")
    print(f"solve rate:        {n_solved}/{n_detected} ({n_solved / n:.1%})  (valid sudoku, bounded correction)")
    print(f"gt-preserving:     {n_gt_solved}/{n_detected} ({n_gt_solved / n:.1%})  (valid AND keeps all GT definite clues)")
    if cand_cells:
        print(f"candidate-only cells read as digits: {cand_false_clues}/{cand_cells}"
              f" (potential false clues, ignored by exact)")
    if n_oracle_tried:
        print(f"oracle (candidate cells emptied): {n_oracle_solved}/{n_oracle_tried}"
              f" of the failed puzzles solve GT-preserving (diagnostic, not a metric)")
    if tot_digit:
        print("per-digit accuracy on solved cells:",
              {k: (round(per_digit_cor[k][0] / per_digit_cor[k][1], 3)
                   if per_digit_cor[k][1] else None) for k in range(1, 10)})
    if wrong_confs:
        case_a = sum(1 for c in wrong_confs if c < 0.5)
        case_b = sum(1 for c in wrong_confs if c > 0.9)
        print(f"wrong-cell confidence: n={len(wrong_confs)}  "
              f"Case A (conf<0.5): {case_a}  Case B (conf>0.9): {case_b}")
    if pos_tot.any():
        hdr = "      " + "".join(f"{f'c{c + 1}':>7}" for c in range(9))
        print("position accuracy (%):")
        print(hdr)
        for r in range(9):
            line = f"r{r + 1:<3} "
            for c in range(9):
                line += f"{100 * pos_cor[r, c] / pos_tot[r, c]:6.0f}%" if pos_tot[r, c] else "     -"
            print(line)
        print("position sample count:")
        print(hdr)
        for r in range(9):
            print(f"r{r + 1:<3} " + "".join(f"{pos_tot[r, c]:7d}" for c in range(9)))

    for name, img, gt, definite, grid, solved, ok in examples:
        marks = (grid != gt) & definite
        print(f"\n{name} | definite-cell acc {(grid[definite] == gt[definite]).mean() if definite.any() else float('nan'):.4f}"
              f" | solved: {ok}")
        print("ground truth (candidate-only cells shown as .):")
        for r in range(9):
            print(" ".join(str(gt[r, c]) if definite[r, c] else "." for c in range(9)))
        print("recognized (wrong cells marked *):")
        for r in range(9):
            print(" ".join(f"{grid[r, c]}{'*' if marks[r, c] else ' '}" for c in range(9)))

    if out_png and examples:
        fig, axes = plt.subplots(len(examples), 3, figsize=(14, 5 * len(examples)))
        axes = np.atleast_2d(axes)
        for i, (name, img, gt, definite, grid, solved, ok) in enumerate(examples):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i][0].imshow(img_rgb)
            axes[i][0].axis("off")
            axes[i][0].set_title(f"{name}\nsolved: {ok}")
            draw_grid_text(axes[i][1], grid, "recognized (red = wrong)",
                           marks=(grid != gt) & definite)
            gt_mark = gt.copy()
            gt_mark[~definite] = -1
            draw_grid_text(axes[i][2], gt_mark, "ground truth (. = candidates)")
        acc = (cor_def / tot_def) if tot_def else float("nan")
        fig.suptitle(f"{title} [{mode_label}]: definite-cell acc {acc:.3f} | "
                     f"digit {(cor_digit / max(tot_digit, 1)):.3f} | "
                     f"exact {(n_exact / n):.1%} | solved {(n_solved / n):.1%}",
                     fontsize=14)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"\nsaved figure to {out_png}")
    if cell_dump and diag_rows:
        os.makedirs(os.path.dirname(cell_dump) or ".", exist_ok=True)
        with open(cell_dump, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["photo", "row", "col", "gt", "pred",
                                              "conf", "threshold", "fg_frac",
                                              "post_ink_frac", "comps", "largest_frac",
                                              "tiny_removed", "grid_removed",
                                              "corner_removed", "merged"])
            w.writeheader()
            w.writerows(diag_rows)
        print(f"wrote per-cell diagnostics to {cell_dump} "
              f"({len(diag_rows)} definite cells)")
    if cell_montage:
        write_montages(diag_groups, cell_montage)
    return {"detected": n_detected, "total": n_total, "definite_acc": cor_def / max(tot_def, 1),
            "digit_acc": cor_digit / max(tot_digit, 1),
            "empty_acc": cor_empty / max(tot_empty, 1),
            "exact": n_exact / n, "solved": n_solved / n,
            "solved_gt": n_gt_solved / n, "no_dup": n_no_dup / n}


def run_wichtounet(data_dir, model_path="models/digit_cnn.pth", frac=0.5, seed=0,
                   out_png=None, device="auto", max_correction_cells=12,
                   mode="both", cell_dump=None, cell_montage=None,
                   preprocess_kwargs=None):
    csv_path = find_outlines_csv(data_dir)
    if csv_path is None:
        print(f"error: no outlines_sorted.csv found under {data_dir}")
        return None
    outlines = load_outlines(csv_path)
    fnames = sorted(outlines)
    version = detect_dataset_version(outlines, fnames[0])
    print(f"wichtounet dataset version detected: {version}")

    device = _pick_device(device)
    model = load_digit_model(model_path, device)
    outlines = load_outlines(csv_path)
    fnames = sorted(outlines)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(fnames))
    n_eval = max(1, int(round(len(fnames) * frac)))
    eval_names = [fnames[i] for i in perm[:n_eval]]
    print(f"benchmark: {len(eval_names)} of {len(fnames)} photos (seeded {frac:.0%}) "
          f"| model {model_path} | device {device} | mode {mode}")

    puzzles = []
    for fname in eval_names:
        img_path, pts = outlines[fname]
        dat_path = os.path.join(os.path.dirname(img_path),
                                os.path.splitext(fname)[0] + ".dat")
        if not os.path.exists(dat_path):
            continue
        gt = parse_dat(dat_path)
        img = imread_any(img_path)
        if img is None:
            continue
        puzzles.append({"name": fname, "img": img, "kp": pts,
                        "gt": gt, "definite": None})
    if not puzzles:
        print("no readable puzzles found")
        return None
    for use_det in (False, True) if mode == "both" else \
            ((False,) if mode == "recognition" else (True,)):
        tag = "e2e" if use_det else "recognition"
        title = f"wichtounet ({version})"
        out = out_png.replace(".png", f"_{tag}.png") if out_png else None
        dump = cell_dump.replace(".csv", f"_{tag}.csv") if cell_dump else None
        mont = cell_montage.replace(".png", f"_{tag}.png") if cell_montage else None
        evaluate_puzzles(puzzles, model, device, max_correction_cells,
                         out_png=out, title=title, use_detection=use_det,
                         cell_dump=dump, cell_montage=mont,
                         preprocess_kwargs=preprocess_kwargs,
                         temperature=load_temperature(model_path))


def run_hf(model_path="models/digit_cnn.pth", split="test", frac=1.0, seed=0,
           hf_dir=None, hf_sample=None, out_png=None, device="auto",
           max_correction_cells=12, mode="both", cell_dump=None,
           cell_montage=None, preprocess_kwargs=None, split_file=None,
           split_name="test"):
    device = _pick_device(device)
    model = load_digit_model(model_path, device)
    if split_file:
        import photo_data
        puzzles = photo_data.load_photo_puzzles(split_name, hf_dir=hf_dir or "data",
                                                split_file=split_file)
        source = f"sealed partition '{split_name}' of {split_file}"
    elif hf_sample:
        hf_dir = download_hf_sample(split, n=hf_sample, seed=seed)
        puzzles = load_hf_folder(hf_dir, frac=1.0, seed=seed)
        source = f"HF sample of {hf_sample} (split '{split}')"
    elif hf_dir:
        puzzles = load_hf_folder(hf_dir, frac, seed)
        source = f"HF folder {hf_dir}"
    else:
        puzzles = load_hf_dataset(split, frac, seed)
        source = f"HF split '{split}'"
    print(f"benchmark: {len(puzzles)} puzzles from {source} "
          f"| model {model_path} | device {device} | mode {mode}")
    if not puzzles:
        print("no puzzles loaded")
        return None
    for use_det in (False, True) if mode == "both" else \
            ((False,) if mode == "recognition" else (True,)):
        tag = "e2e" if use_det else "recognition"
        title = f"Lexski/sudoku-image-recognition ({split})"
        out = out_png.replace(".png", f"_{tag}.png") if out_png else None
        dump = cell_dump.replace(".csv", f"_{tag}.csv") if cell_dump else None
        mont = cell_montage.replace(".png", f"_{tag}.png") if cell_montage else None
        evaluate_puzzles(puzzles, model, device, max_correction_cells,
                         out_png=out, title=title, use_detection=use_det,
                         cell_dump=dump, cell_montage=mont,
                         preprocess_kwargs=preprocess_kwargs,
                         temperature=load_temperature(model_path))


def run(data_dir, model_path="models/digit_cnn.pth", frac=0.5, seed=0, out_png=None,
        device="auto", max_correction_cells=12, mode="both"):
    """Folder-mode benchmark entry point - called by train_local.py's
    --benchmark-data-dir (the missing `benchmark.run` used to crash training
    after the final epoch)."""
    return run_wichtounet(data_dir, model_path=model_path, frac=frac, seed=seed,
                          out_png=out_png, device=device,
                          max_correction_cells=max_correction_cells, mode=mode)


def main():
    ap = argparse.ArgumentParser(description="Labeled pipeline benchmark")
    ap.add_argument("--data-dir", default=None,
                    help="wichtounet folder containing outlines_sorted.csv + photos + .dat")
    ap.add_argument("--hf", action="store_true",
                    help="benchmark on Lexski/sudoku-image-recognition (needs datasets lib)")
    ap.add_argument("--hf-dir", default=None,
                    help="local copy of data/<split> (images/ + metadata.jsonl)")
    ap.add_argument("--hf-sample", type=int, default=None,
                    help="download only N seeded images (lightweight, no datasets lib)")
    ap.add_argument("--hf-split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-file", default=None,
                    help="sealed photo_splits.json (photo_data.py): evaluate on "
                         "the exact partition's puzzle names, not a seeded frac "
                         "or a raw HF split")
    ap.add_argument("--split-name", default="test", choices=["train", "val", "test"],
                    help="partition of --split-file to evaluate (default test)")
    ap.add_argument("--model", default="models/digit_cnn.pth")
    ap.add_argument("--frac", type=float, default=0.5,
                    help="fraction of the dataset to benchmark (seeded)")
    ap.add_argument("--out", default="results/figures/benchmark.png", help="output figure path")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-correction-cells", type=int, default=12)
    ap.add_argument("--mode", default="both", choices=["recognition", "e2e", "both"],
                    help="recognition = ground-truth corners, e2e = auto grid "
                         "detection, both = run and report both (default)")
    ap.add_argument("--cell-dump", default=None,
                    help="write per-definite-cell diagnostics (gt, pred, conf, "
                         "preprocess stats) to a CSV")
    ap.add_argument("--cell-montage", default=None,
                    help="write grouped preprocessing-stage montages "
                         "(empty->digit, 1->7, 7->1, 8 reference, other errors)")
    ap.add_argument("--pre-margin", type=float, default=None,
                    help="preprocessing A/B: margin_frac (default 0.10)")
    ap.add_argument("--pre-empty", type=float, default=None,
                    help="preprocessing A/B: empty_frac (default 0.01)")
    ap.add_argument("--pre-corner-span", type=float, default=None,
                    help="preprocessing A/B: corner_span (default 0.4)")
    args = ap.parse_args()

    pre_kwargs = {}
    if args.pre_margin is not None:
        pre_kwargs["margin_frac"] = args.pre_margin
    if args.pre_empty is not None:
        pre_kwargs["empty_frac"] = args.pre_empty
    if args.pre_corner_span is not None:
        pre_kwargs["corner_span"] = args.pre_corner_span

    if args.data_dir:
        run_wichtounet(args.data_dir, args.model, args.frac, out_png=args.out,
                       device=args.device, mode=args.mode,
                       max_correction_cells=args.max_correction_cells,
                       cell_dump=args.cell_dump, cell_montage=args.cell_montage,
                       preprocess_kwargs=pre_kwargs)
    elif args.hf or args.hf_dir or args.hf_sample or args.split_file:
        run_hf(args.model, args.hf_split, args.frac, hf_dir=args.hf_dir,
               hf_sample=args.hf_sample, split_file=args.split_file,
               split_name=args.split_name, out_png=args.out, device=args.device,
               mode=args.mode, max_correction_cells=args.max_correction_cells,
               cell_dump=args.cell_dump, cell_montage=args.cell_montage,
               preprocess_kwargs=pre_kwargs)
    else:
        ap.error("pass --data-dir, --hf, --hf-dir, --hf-sample, or --split-file")


if __name__ == "__main__":
    main()
