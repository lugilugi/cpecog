# End-to-End Image Sudoku Solver (CPECOG1)

Academic capstone (CPECOG1, Term 9): an end-to-end system that detects a Sudoku grid in a
photo, recognizes the printed digits with a CNN, reconstructs the puzzle, and solves it.

## How it works

```
photo -> detect grid -> perspective warp -> 81 cells -> adaptive thresholding
      -> CNN digit recognition -> 9x9 puzzle -> MRV backtracking solver -> solution
```

* **`sudoku_core.py`** - OpenCV pipeline: grid detection/warping, cell extraction,
  digit-presence test, solver. `full_pipeline()` is the top-level entry;
  `solve_sudoku()` is the MRV backtracking solver; `solve_with_correction()` is the
  solver-guided error-correction path.
* **`run_sudoku.py`** - command-line driver: detects and solves a puzzle photo with
  the CNN (see [Usage](#usage-local-after-training)).
* **`digit_cnn.py`** - PyTorch `DigitCNN` (48x48 input), synthetic digit generation,
  training, `load_digit_model()`, `predict_cells()`.
* **`models/`** - all trained weights (`digit_cnn.pth`, smoke models) + their
  `.temperature.json` sidecars + the `infra_ckpt.pt` training checkpoint. The default
  `--model` path of every CLI points here.
* **`tools/`** - `make_narrative_figures.py` (regenerates the narrative figures +
  metric JSON by parsing the run logs) and `diagnose_solve.py` (per-puzzle solver
  diagnostic on the HF sample).
* **`results/`** - benchmark artifacts: `results/runs/` (logs), `results/cells/`
  (per-cell CSV dumps), `results/figures/` (benchmark figures + error montages).
* **`sudoku_cnn_colab.ipynb`** - Google Colab notebook to **retrain the CNN on real
  sudoku-photo digits** (see [Training](#training) below).
* **`cpecog_extracted/aug/`** - 2620 Sudoku-grid JPEGs (Kaggle "Sudoku Box Detection",
  extracted from `cpecog_extracted/cpecog.zip`, ~124 MB) used for end-to-end
  verification. Large data artifacts: never re-extract, never commit.
* **`CPECOG1 _Group 4 Project.md`** - the paper draft (not code docs).
* **`AGENTS.md`** - agent guidance for AI tooling (not required to use the project).

## Environment

Python 3.14.5, no `requirements.txt`. Dependencies: `opencv-python` (cv2 4.13.0),
`torch` 2.13.0+cpu, `numpy`, `Pillow`.

Sanity check:

```powershell
python -c "import sudoku_core, digit_cnn"
```

## Training

The CNN is retrained end-to-end in **`sudoku_cnn_colab.ipynb`** on Google Colab
(`Runtime > Change runtime type > T4 GPU`, then `Runtime > Run all`). Note: cell 4/3b
(the optional Kaggle-token upload) blocks "Run all" waiting for file input - either
upload the token or set `CONFIG["data_mode"] = "clone"` first.

### Why retrain?

The shipped `digit_cnn.pth` was trained on *synthetic rendered fonts* only. Real photos
of printed puzzles are a different distribution (camera noise, shadows, print bleed),
so the CNN overfits the synthetic domain: wrong digits arrive at the solver with
confidence > 0.9 (the benchmark's "Case B"), and the correction search stalls. The fix
pipeline (2026-08-10): **fine-tune on real photo cells** from the Lexski
[Sudoku Image Recognition](https://huggingface.co/datasets/Lexski/sudoku-image-recognition)
dataset under a **sealed 70/15/15 split** (the TEST partition is never trained on),
warm-started from a synthetic checkpoint at lr 1e-4, on a **GAP-head DigitCNN**
(no more 48x48-flatten position lock), followed by **temperature scaling** fit on the
photo validation split so the solver's `conf_thresh=0.9` gate becomes meaningful.
On the sealed 210-puzzle test split a 3-epoch smoke went 1393 -> 235 wrong cells
(Case B 511 -> 59); the full run is expected to be much better still.

### What the notebook does

| Section | Purpose |
|---------|---------|
| 1-3 | Setup, config, dataset provisioning. `auto` mode uses the Kaggle API if `kaggle.json` is uploaded, otherwise falls back to a credentials-free `git clone` of the public mirror. |
| 4 | Explores the dataset: `.dat` files carry the ground-truth puzzle (0 = empty, 1-9 = digits); `outlines_sorted.csv` carries each photo's grid corners. |
| 5 | Extracts labeled digit crops: perspective warp from the outlines -> 81 row-major cells -> **adaptive-threshold preprocessing** (details below) -> keep cells with digits 1-9, label from the `.dat`. |
| 6 | Generates balanced synthetic digits (classes 0-9) with rotation/shift/noise/blur corruption, using fonts found on the machine (DejaVu ships with Colab). |
| 6b | **Optional** MNIST handwritten-digit supplement (60k samples, upscaled 28->48) so the classifier also reads handwriting (hand-solved and hand-drawn puzzles). Set `CONFIG["include_mnist"] = False` to skip. |
| 7 | Pools photos + synthetic (+ MNIST) with a per-sample `source` label, splits **70 / 15 / 15 (train / val / test) stratified by digit, seeded (SEED=0)**, so every class is represented in every split. |
| 8 | Augments the **train split only** (rotation, shift, noise, blur); val and test stay untouched. |
| 9-10 | `DigitCNN` + training (15 epochs, Adam, LR halved every 10 epochs via `StepLR`) with per-epoch checkpoints (`checkpoints/ckpt_epochNN.pt`, plus `best_model.pt` = best validation epoch). |
| 10c | Optional resume: sets `RESUME_PATH` to the latest checkpoint; re-running the training cell continues where it stopped (LR schedule state is reconstructed from the epoch). |
| 11 | Final evaluation on the held-out test: overall/per-class accuracy, confusion matrix, and accuracy broken down by source (photo vs synthetic vs mnist). |
| 12 | Exports `digit_cnn.pth` (plain `state_dict`, compatible with `load_digit_model()`) and downloads it. |
| 13 | **Optional** end-to-end verification on your own sudoku photos (e.g. `cpecog.zip`): detection rate + solve-validity on a seeded 80/20 split. Skipped automatically if `CONFIG["cpecog_dir"]` is not set. Every labeled set is evaluated twice: **recognition/solver** (ground-truth corners) and **end-to-end** (auto grid detection, honest detection rate). |

### Preprocessing (adaptive thresholding)

Every cell passes through the same pipeline at training *and* inference time
(identical to `digit_cnn.preprocess_cell`):

1. Gaussian blur (3x3) - smooths sensor noise.
2. **Adaptive thresholding** (`ADAPTIVE_THRESH_GAUSSIAN_C`, block 15, constant 7,
   inverted) - a per-pixel local threshold, so uneven lighting and shadows don't
   destroy the digit the way a fixed global threshold would.
3. Strip an 8% border margin - removes grid-line fragments at cell edges.
4. Keep the largest connected component - drops specks.
5. Tight crop around the digit's bounding box with 10% padding.
6. Resize to 48x48, normalize to float32 [0,1] + channel dim -> `(48,48,1)`.

### Partitioning (best practice)

* **Stratified** - each split keeps the same per-class proportions (critical because
  class 0 exists only in the synthetic data).
* **Seeded** - `SEED = 0` makes the partition reproducible across runs.
* **Test held out** - evaluated exactly once, at the very end; it never influences
  checkpoints or hyperparameters. Val picks the best checkpoint; train alone is
  augmented.
* Grid-level verification (section 13) additionally uses a seeded 80/20 split so the
  reported detection/solve rates come from a held-out 20% of the photos.

### Expected runtime

15 epochs over ~116k samples (5.8k photo crops + 50k synthetic + 60k MNIST):
roughly 5-10 minutes on a Colab T4 GPU; much slower on CPU.

## Usage (local, after training)

### End-to-end solver CLI (`run_sudoku.py`)

```powershell
python run_sudoku.py puzzle.jpg            # prints the recognized grid + solution
python run_sudoku.py puzzle.jpg --show     # pops a 2x2 matplotlib figure
python run_sudoku.py puzzle.jpg --save out.png   # saves the figure instead
python run_sudoku.py puzzle.jpg --model other.pth --size 900
```

Flags: `image` (required), `--model` (default `models/digit_cnn.pth`), `--size` (warp
resolution, default 600), `--show`, `--save <path>`. Console output is always printed;
`--show`/`--save` add the 2x2 figure (original + grid quad, warped grid, recognized
puzzle, solution). The solution is verified with `valid_solution` before reporting.

### Library use

```python
import cv2
import numpy as np
from digit_cnn import load_digit_model, predict_cells
from sudoku_core import full_pipeline, solve_sudoku

model = load_digit_model("models/digit_cnn.pth")   # the weights downloaded from the notebook
grid, warped, cells, img = full_pipeline("puzzle.jpg", lambda c: predict_cells([c], model)[0][0])
print(grid)                                 # 9x9 numpy array, 0 = empty cell
solved, ok = solve_sudoku(grid)
print(solved if ok else "no solution (digit recognition errors?)")
```

## Evaluation / verification

* **Digit classifier** - reported by the notebook (section 11): test accuracy, per-class,
  photo-vs-synthetic breakdown, confusion matrix.
* **Labeled benchmark** (`benchmark.py` / notebook section 13) - every puzzle set is
  evaluated in **two separately-labeled modes**:
  * **RECOGNITION/SOLVER** - the dataset's ground-truth grid corners are warped
    directly, isolating CNN recognition + solving from grid detection (solved-cell
    digit accuracy, empty accuracy, exact grids, solve rate).
  * **END-TO-END** - corners are ignored and the grid is found automatically
    (contour -> lines). There is **no center-crop fallback**: detection misses are
    counted honestly in a "grid detection" rate and excluded from recognition
    metrics. This is the real `photo -> detect -> solve` number.
  `benchmark.py` runs both by default (`--mode both`; `recognition`/`e2e` select one).
* **Unlabeled grid-detection check** - notebook section 13 on `cpecog.zip` images
  (upload or set `CONFIG["cpecog_dir"]`): grid-detection rate, counting only real
  detections (contour/lines) - failures are counted as failures.
* **Baseline** (measured 2026-08-04, synthetic-only weights, 26-image sample of
  `cpecog_extracted/aug/`): grid detected 53.8%, puzzles solved validly 0%.
* **After first retraining** (2026-08-04, `digit_cnn.pth` retrained on real photo
  digits): digit accuracy on real photos 80.2%, but end-to-end still 0 solves
  (0.8^27 ~ 0.2% solve rate at 80% digit accuracy) - digit accuracy must reach ~99%
  for reliable solving. Track progress in the work log below.

## Work log

| Date | Task | Status | Result / notes |
|------|------|--------|----------------|
| 2026-08-04 | Colab training notebook (`sudoku_cnn_colab.ipynb`) built; parsing/extraction/training/checkpoint/resume/CV code validated locally against the wichtounet dataset and cpecog images | Done | 202 photos -> 5,872 labeled digit crops; split/checkpoint/resume logic verified |
| 2026-08-04 | Baseline end-to-end verification with existing weights on cpecog images | Done | Detection 53.8%, solve 0% (26-image sample) |
| 2026-08-04 | Fixed `solve_with_correction()` NameError: `solve_sudoku` (MRV backtracking, `max_nodes=100_000`, upfront `valid_givens` check) moved to module level; orphaned dead code removed | Done | `solve_sudoku` verified on a real puzzle; `solve_with_correction` returns valid solutions |
| 2026-08-04 | `run_sudoku.py` CLI added (console default, `--show`/`--save` figure, `--model`/`--size`) | Done | Smoke-tested on cpecog + wichtounet images; honest "no solution found" on unsolvable recognitions |
| 2026-08-04 | Notebook upgrade: 15 epochs + StepLR, 5,000 synthetic/class, optional MNIST supplement (28->48 upscale) | Done | Rebuilt (34 cells); logic validated locally with `include_mnist=False`; transform + LR-resume parity unit-tested |
| 2026-08-04 | Retrained `digit_cnn.pth` in Colab with real photo digits (+ synthetic) | Done | Real-photo digit accuracy **80.2%** (1,721 labeled crops, 59 photos), per-class 72-86%, mean confidence 0.85; file 566,872 bytes |
| 2026-08-04 | End-to-end re-verification after retraining | Done | Detection 7/40 detected; 0/7 solved (0.8^27 ~ 0.24% solve rate at 80% digit accuracy) - digit accuracy must reach ~99% for reliable solving |
| 2026-08-08 | Review fixes: augmentation shift now uses OpenCV translation with black borders (no more np.roll wrap-around); grid detection is honest (center-crop fallback removed everywhere - misses are counted as misses); benchmark split into two modes: recognition/solver (GT corners) vs end-to-end (auto detection) | Done | `--mode both` default; notebook regenerated (37 cells) |
| | Retrain in Colab with the upgraded notebook (15 epochs + MNIST) | Pending | target: >=99% per-digit on real photos |
| | Copy `digit_cnn.pth` back into the repo | Pending | |
| 2026-08-10 | Repo reorganized for readability: weights -> `models/`, benchmark logs/CSVs/figures -> `results/{runs,cells,figures}/`, tools -> `tools/`, `cpecog.zip` -> `cpecog_extracted/`. All CLI defaults + docs updated; nothing renamed or deleted | Done | `python benchmark.py --out` defaults to `results/figures/benchmark.png`; `--model` defaults to `models/digit_cnn.pth` |

## Known bugs / gotchas

* **`preprocess_cell` exists in both modules with different return shapes** -
  `digit_cnn.py` returns `(48,48,1)` float32 (required by `predict_cells`);
  `sudoku_core.py` returns `(48,48)`. Don't mix them.
* **48x48 size lock** - `DigitCNN`'s classifier is `Linear(128*6*6, 10)`; inputs must
  stay 48x48 (3 max-pools halve it three times).
* **Cell order** is row-major: 81 cells, index `r*9+c`; `sudoku_from_cells` reshapes to
  9x9.
* **`solve_sudoku` gives up at `max_nodes=100_000`** - fine for correctly recognized
  puzzles (solves in ms), but recognition errors can make the puzzle unsolvable; that
  surfaces as `ok=False` or "no solution found".
* **Digit accuracy drives solve rate** - a puzzle with ~27 givens needs ~99% per-digit
  accuracy to solve reliably (0.8^27 ~ 0.2% at 80% accuracy). Run the notebook's
  retraining (with MNIST) before expecting solves.
* **`train_digit_model()` overwrites `models/digit_cnn.pth`** by default - pass a different
  `weight_path` to keep the current weights.
* **Notebook cell 4/3b (optional Kaggle-token upload) blocks "Run all"** while waiting
  for file input - upload `kaggle.json`, or use `CONFIG["data_mode"] = "clone"`.

## References

* T. Yato and T. Seta, "Complexity and Completeness of Finding Another Solution and Its
  Application to Puzzles," *IEICE Trans. Fundamentals*, 2003.
* Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," *Proc.
  IEEE*, 1998.
* A. Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural
  Networks," *NeurIPS*, 2012.
* S. S. Magaji, "Sudoku Box Detection," Kaggle - the 2620-image dataset in
  `cpecog_extracted/aug/`.
* M. Ehrminger, "Sudoku Image Dataset," Kaggle (mirror of
  github.com/wichtounet/sudoku_dataset) - real photos with ground-truth puzzles and
  grid outlines, used by the training notebook.
