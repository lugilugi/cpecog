"""Real-photo training data from Lexski/sudoku-image-recognition.

Builds per-cell crops through EXACTLY the same path the RECOGNITION
benchmark mode sees: ground-truth corners -> `four_point_transform` ->
`extract_cells` (FRACTIONAL boundaries) -> `preprocess_cell` (v5) ->
(48, 48, 1) float32. The CNN is therefore trained on the same geometry it
gets at inference.

Labels come from the (9,9,10) cell flags (the benchmark's `hf_cells_to_gt`):
definite cells only - solved with exactly one digit -> classes 1-9, unsolved
with no candidates -> class 0 (empty). Candidate-only cells are SKIPPED
(the same rule the benchmark metrics use); the model never trains on
ambiguous labels.

SPLITS: the full dataset (all three metadata.jsonl files, ~1400 images) is
shuffled ONCE with a fixed seed and partitioned 70/15/15 into
photo_splits.json (per-partition name lists). The TEST partition is SEALED:
it never enters the training or validation pools, and
`benchmark.py --split-file photo_splits.json` evaluates on it.

CACHE: preprocessed cells are cached to emnist/photo_packed.npz with a
preprocess_version marker + the exact name list - a change to the
preprocessing or the split regenerates the cache, never silently reuses it.
"""
import hashlib
import json
import os

import cv2
import numpy as np

import sudoku_core as sc
from digit_cnn import PREPROCESS_VERSION, preprocess_cell as _pp  # noqa: E402

SPLIT_FILE = "photo_splits.json"
SPLIT_SEED = 42
HF_SPLITS = ("train", "val", "test")

# benchmark.* helpers (imread_any, hf_cells_to_gt, HF_BASE) are imported
# LAZILY inside functions: benchmark.py imports photo_data at runtime too,
# and a top-level import here would create an import cycle.


def hf_dir_path(hf_dir, split):
    return os.path.join(hf_dir, split)


def ensure_hf_metadata(hf_dir="data"):
    """Download metadata.jsonl for every HF split if missing (metadata only -
    images are fetched by `ensure_hf_images`). Returns
    {split: [row, ...]}."""
    from benchmark import HF_BASE
    import urllib.request

    out = {}
    for split in HF_SPLITS:
        d = hf_dir_path(hf_dir, split)
        os.makedirs(d, exist_ok=True)
        meta = os.path.join(d, "metadata.jsonl")
        if not os.path.exists(meta):
            url = f"{HF_BASE}/{split}/metadata.jsonl"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, meta)
        rows = [json.loads(l) for l in open(meta, encoding="utf-8") if l.strip()]
        out[split] = rows
    return out


def ensure_hf_images(hf_dir="data", splits=HF_SPLITS, workers=8):
    """Download every image of the given HF splits (parallel, ~100 MB total
    for the whole dataset). Idempotent: existing files are skipped. Returns
    the total image count available."""
    from benchmark import HF_BASE
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    meta = ensure_hf_metadata(hf_dir)
    jobs = []
    for split in splits:
        img_dir = os.path.join(hf_dir_path(hf_dir, split), "images")
        os.makedirs(img_dir, exist_ok=True)
        for row in meta[split]:
            name = os.path.basename(row["file_name"].replace("\\", "/"))
            out = os.path.join(img_dir, name)
            if not os.path.exists(out):
                jobs.append((f"{HF_BASE}/{split}/images/{name}", out))
    done = 0

    def fetch(job):
        url, out = job
        try:
            urllib.request.urlretrieve(url, out)
            return 1
        except Exception as exc:
            print(f"download failed ({url}): {exc}")
            return 0

    if jobs:
        print(f"downloading {len(jobs)} images to {hf_dir} (workers={workers})...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for n in ex.map(fetch, jobs):
                done += n
        print(f"downloaded {done}/{len(jobs)} images")
    else:
        print("all images already present")
    return sum(len(os.listdir(os.path.join(hf_dir_path(hf_dir, s), "images")))
               for s in splits if os.path.isdir(os.path.join(
                   hf_dir_path(hf_dir, s), "images")))


def all_record_names(hf_dir="data"):
    """Every image name in the dataset (across all three HF splits), with the
    split it lives in. Names are the metadata file_name basenames."""
    meta = ensure_hf_metadata(hf_dir)
    names = {}
    for split, rows in meta.items():
        for row in rows:
            name = os.path.basename(row["file_name"].replace("\\", "/"))
            names[name] = split
    return names


def build_splits(hf_dir="data", seed=SPLIT_SEED, split_file=SPLIT_FILE):
    """Seeded 70/15/15 partition of the FULL dataset -> photo_splits.json.

    The shuffle is fixed (seed) so the test partition is sealed and
    reproducible: it never changes between runs, so models can be compared
    on exactly the same photos.
    """
    names = sorted(all_record_names(hf_dir))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(names))
    n_tr = int(round(len(names) * 0.70))
    n_va = int(round(len(names) * 0.15))
    parts = {
        "train": [names[i] for i in perm[:n_tr]],
        "val": [names[i] for i in perm[n_tr:n_tr + n_va]],
        "test": [names[i] for i in perm[n_tr + n_va:]],
    }
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "created": __import__("datetime")
                   .datetime.now().isoformat(),
                   **parts}, f, indent=1)
    for k, v in parts.items():
        print(f"split '{k}': {len(v)} puzzles")
    print(f"wrote {split_file}")
    return parts


def load_splits(split_file=SPLIT_FILE):
    with open(split_file, encoding="utf-8") as f:
        data = json.load(f)
    return {k: data[k] for k in HF_SPLITS}


def load_photo_puzzles(split_name, hf_dir="data", split_file=SPLIT_FILE):
    """Puzzles (name/img/kp/gt/definite) for exactly the names in the
    requested partition of the split file. Images are looked up in whichever
    HF split dir they live in and read through the Pillow fallback."""
    splits = load_splits(split_file)
    names = set(splits[split_name])
    if not names:
        return []
    from benchmark import imread_any, hf_cells_to_gt
    meta = ensure_hf_metadata(hf_dir)
    name_to_split = {}
    for s, rows in meta.items():
        for row in rows:
            name = os.path.basename(row["file_name"].replace("\\", "/"))
            name_to_split[name] = s
    puzzles = []
    for name in sorted(names):
        s = name_to_split.get(name)
        if s is None:
            print("skip (not in metadata):", name)
            continue
        img = imread_any(os.path.join(hf_dir_path(hf_dir, s), "images", name))
        if img is None:
            print("skip unreadable image:", name)
            continue
        row = next(r for r in meta[s]
                   if os.path.basename(r["file_name"].replace("\\", "/")) == name)
        kp = np.array(row["keypoints"], dtype=np.float32).reshape(4, 2)
        kp = kp[[0, 3, 2, 1]]                      # TL,BL,BR,TR -> TL,TR,BR,BL
        gt, definite = hf_cells_to_gt(row["cells"])
        puzzles.append({"name": name, "img": img, "kp": kp,
                        "gt": gt, "definite": definite})
    print(f"loaded {len(puzzles)}/{len(names)} puzzles of partition '{split_name}'")
    return puzzles


def extract_photo_cells(puzzles, preprocess_kwargs=None):
    """Per-puzzle: GT-corner warp -> fractional extract_cells -> v5
    preprocess_cell on every DEFINITE cell.

    Returns (X, y, names, rows, cols): X is (N,48,48,1) float32 in exactly
    the representation predict_cells_probs consumes; y are the definite
    labels (0 = empty, 1-9 = digits); rows/cols are the grid positions.
    """
    pre = preprocess_kwargs or {}
    X, y, names, rows, cols = [], [], [], [], []
    for pz in puzzles:
        gray = cv2.cvtColor(pz["img"], cv2.COLOR_BGR2GRAY)
        warped = sc.four_point_transform(gray, pz["kp"])
        cells = sc.extract_cells(warped)
        d = pz["definite"]
        gt = pz["gt"]
        for r in range(9):
            for c in range(9):
                if not d[r, c]:
                    continue
                X.append(_pp(cells[r * 9 + c], target=48, **pre))
                y.append(int(gt[r, c]))
                names.append(pz["name"])
                rows.append(r)
                cols.append(c)
    return (np.stack(X).astype(np.float32), np.array(y, dtype=np.int64),
            np.array(names, dtype=object),
            np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))


def _cache_key(names):
    h = hashlib.sha256()
    for n in sorted(names):
        h.update(n.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def build_photo_cells(split_name, hf_dir="data", split_file=SPLIT_FILE,
                      cache=None, preprocess_kwargs=None):
    """Preprocessed definite cells of a partition, cached to npz.

    Cache validity = preprocess_version marker + exact name list + seed; any
    mismatch regenerates (old-format caches are never silently reused).
    The cache is PER-PARTITION (emnist/photo_packed_<split>.npz) so the
    train and val pools never overwrite each other.
    Returns (X, y, names, rows, cols).
    """
    if cache is None:
        cache = os.path.join("emnist", f"photo_packed_{split_name}.npz")
    puzzles = load_photo_puzzles(split_name, hf_dir=hf_dir, split_file=split_file)
    names_req = sorted(pz["name"] for pz in puzzles)
    if cache and os.path.exists(cache):
        try:
            z = np.load(cache, allow_pickle=True)
            if (str(z["preprocess_version"]) == PREPROCESS_VERSION
                    and int(z["seed"]) == SPLIT_SEED
                    and set(z["names"].tolist()) == set(names_req)):
                print(f"photo cache hit ({len(names_req)} puzzles): {cache}")
                return (z["X"].astype(np.float32),
                        z["y"].astype(np.int64),
                        z["names"].astype(object),
                        z["rows"].astype(np.int64),
                        z["cols"].astype(np.int64))
        except Exception as exc:
            print(f"photo cache unreadable ({exc}); regenerating")
    print(f"building photo cells for partition '{split_name}' "
          f"({len(puzzles)} puzzles)...")
    X, y, names, rows, cols = extract_photo_cells(puzzles, preprocess_kwargs)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, X=X, y=y, names=np.asarray(names, dtype=object),
                 rows=rows, cols=cols,
                 preprocess_version=PREPROCESS_VERSION, seed=SPLIT_SEED,
                 cache_key=_cache_key(names))
        print(f"cached {len(y)} cells to {cache}")
    return X, y, names, rows, cols
