"""Why do exact-definite grids fail to solve? Per-puzzle: definite accuracy,
exact flag, solve status, and whether bigger correction budgets fix it."""
import json
import os

import cv2
import numpy as np

import sudoku_core as sc
import benchmark
from digit_cnn import load_digit_model, predict_cells_probs, classify_preprocessed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
HF = os.path.join(REPO, "benchmark_data", "hf_test_sample")
device = "cuda"
model = load_digit_model(os.path.join(REPO, "models", "digit_cnn_smoke.pth"), device)
meta = [json.loads(l) for l in open(os.path.join(HF, "metadata.jsonl"), encoding="utf-8")]

solved_default = 0
exact_def = 0
for row in meta:
    name = os.path.basename(row["file_name"].replace("\\", "/"))
    img = cv2.imread(os.path.join(HF, "images", name))
    if img is None:
        continue
    kp = np.array(row["keypoints"], dtype=np.float32).reshape(4, 2)[[0, 3, 2, 1]]
    warped = sc.four_point_transform(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), kp)
    cells = sc.extract_cells(warped)
    probs = predict_cells_probs(cells, model, device)
    grid = probs.argmax(1).reshape(9, 9)
    gt, definite = benchmark.hf_cells_to_gt(row["cells"])
    dig = definite & (gt != 0)
    acc = float(((grid == gt) & definite).sum() / max(definite.sum(), 1))
    exact = bool(((grid == gt) | ~definite).all())
    if exact:
        exact_def += 1

    def _solve(mcc, max_nodes):
        st = {}
        sol, ok, _ = sc.solve_with_resensing(
            grid, probs, cells,
            lambda v: classify_preprocessed(v, model, device),
            max_correction_cells=mcc, max_rounds=2, max_nodes=max_nodes, stats=st)
        return ok and benchmark.valid_solution(sol) if ok else False

    ok_def = _solve(12, 30_000)
    if ok_def:
        solved_default += 1
    n_def_wrong = int(((grid != gt) & definite).sum())
    cand = ~definite
    print(f"{name[:28]:30s} def-acc {acc:.3f} wrong-def {n_def_wrong:2d} "
          f"cand-cells {int(cand.sum()):2d} | exact {int(exact)} solve {int(ok_def)}")

print(f"\nPuzzles: {24} | exact-definite: {exact_def} | solved: {solved_default}")
