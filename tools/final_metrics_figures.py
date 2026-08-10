"""Final-run metric figures + machine-readable metrics bundle for the paper.

Generates, from the official sealed-test run (results/runs/final_run.log) and
its per-cell dump (results/cells/final_cells_recognition.csv):

  fig_f10_confusion_matrix.png  - 10x10 gt-vs-pred confusion matrix on
        definite cells (recognition mode), count panel + row-normalized
        (recall) panel, with per-class recall/precision in the caption.
  fig_f11_position_heatmaps.png - per-position accuracy + sample-count
        heatmaps on the sealed test (recognition mode).
  final_metrics.json            - the COMPLETE metrics bundle: model facts
        (params, temperature, best score, epoch history), recognition AND
        e2e headline metrics parsed from the log, per-digit accuracies,
        wrong-cell confidence cases, solver node stats, per-position
        matrices, per-class cell counts.

Run from the repo root:  python tools/final_metrics_figures.py
"""
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "narrative_figures")
os.makedirs(OUT, exist_ok=True)

LOG = os.path.join(REPO, "results", "runs", "final_run.log")
CELL_CSV = os.path.join(REPO, "results", "cells", "final_cells_recognition.csv")

RE = {
    "evaluated": re.compile(r"evaluated:\s+(\d+)/(\d+)"),
    "detection": re.compile(r"grid detection:\s+(\d+)/(\d+) \(([\d.]+)%\)"),
    "resensed": re.compile(r"cells re-sensed:\s+(\d+)"),
    "nodes": re.compile(r"solver nodes:\s+mean (\d+), max (\d+)\s+\(node-limit hits: (\d+)\)"),
    "definite": re.compile(r"definite-cell acc: (\d+)/(\d+) \(([\d.]+)\)"),
    "digit": re.compile(r"DIGIT acc \(solved\): (\d+)/(\d+) \(([\d.]+)\)"),
    "empty": re.compile(r"empty acc:\s+(\d+)/(\d+) \(([\d.]+)\)"),
    "exact": re.compile(r"exact grids:\s+(\d+)/(\d+) \(([\d.]+)%\)"),
    "nodupe": re.compile(r"no-duplicate grids:(\d+)/(\d+) \(([\d.]+)%\)"),
    "solve": re.compile(r"solve rate:\s+(\d+)/(\d+) \(([\d.]+)%\)"),
    "gtpres": re.compile(r"gt-preserving:\s+(\d+)/(\d+) \(([\d.]+)%\)"),
    "cand": re.compile(r"candidate-only cells read as digits:\s+(\d+)/(\d+)"),
    "oracle": re.compile(r"oracle \(candidate cells emptied\):\s+(\d+)/(\d+)"),
    "perdigit": re.compile(r"per-digit accuracy on solved cells: \{([^}]*)\}"),
    "conf": re.compile(r"wrong-cell confidence: n=(\d+)\s+Case A \(conf<0\.5\): (\d+)\s+Case B \(conf>0\.9\): (\d+)"),
}


def parse_section(text):
    """Parse one mode section's metrics from the raw log text."""
    out = {}
    for k, rx in RE.items():
        m = rx.search(text)
        if not m:
            continue
        g = m.groups()
        if k in ("evaluated", "detection", "resensed", "cand", "oracle"):
            out[k] = [int(x) for x in g[:2]] + ([float(g[2])] if len(g) > 2 else [])
        elif k in ("definite", "digit", "empty"):
            out[k] = [int(g[0]), int(g[1]), float(g[2])]
        elif k in ("exact", "nodupe", "solve", "gtpres"):
            out[k] = [int(g[0]), int(g[1]), float(g[2])]
        elif k == "nodes":
            out[k] = [int(g[0]), int(g[1]), int(g[2])]
        elif k == "conf":
            out[k] = [int(g[0]), int(g[1]), int(g[2])]
        elif k == "perdigit":
            out[k] = {int(d): float(v) for d, v in
                      (p.split(":") for p in g[0].split(","))}
    return out


def load_log():
    text = open(LOG, encoding="utf-8", errors="replace").read()
    blocks = re.split(r"benchmark mode:\s+", text)
    out = {}
    for b in blocks[1:]:
        mode = "recognition" if b.startswith("RECOGNITION") else "e2e"
        out[mode] = parse_section(b)
    return out


def load_cells():
    rows = []
    with open(CELL_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "photo": r["photo"], "row": int(r["row"]) - 1, "col": int(r["col"]) - 1,
                "gt": int(r["gt"]), "pred": int(r["pred"]), "conf": float(r["conf"]),
                "threshold": r["threshold"], "fg_frac": float(r["fg_frac"]),
                "post_ink_frac": float(r["post_ink_frac"]),
                "comps": int(r["comps"]), "largest_frac": float(r["largest_frac"]),
                "tiny_removed": int(r["tiny_removed"]), "grid_removed": int(r["grid_removed"]),
                "corner_removed": int(r["corner_removed"]), "merged": int(r["merged"]),
            })
    return rows


def confusion(cells):
    C = np.zeros((10, 10), dtype=np.int64)
    for r in cells:
        C[r["gt"], r["pred"]] += 1
    return C


def fig_f10(C, cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recall = C.astype(float) / C.sum(1, keepdims=True).clip(min=1)
    precision = C.astype(float) / C.sum(0, keepdims=True).clip(min=1)
    wrong = int(C.sum() - np.trace(C))
    labels = ["empty"] + [str(d) for d in range(1, 10)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, M, fmt, cmap in [
            (axes[0], C, "d", "viridis"),
            (axes[1], recall, ".2f", "magma")]:
        im = ax.imshow(M, cmap=cmap)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
        for i in range(10):
            for j in range(10):
                v = M[i, j]
                if (fmt == "d" and v > 0) or (fmt != "d" and v >= 0.005):
                    ax.text(j, i, format(v, fmt), ha="center", va="center",
                            fontsize=7,
                            color="white" if (v > 0.5 * M.max() or (fmt != "d" and v > 0.5)) else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_title(f"Definite-cell confusion counts (recognition, n={int(C.sum())})")
    axes[1].set_title("Row-normalized recall (empty=0, digits 1-9)")
    fig.suptitle(f"F10 - final model on the sealed 210-puzzle test: "
                 f"{int(C.sum())} definite cells, {wrong} wrong "
                 f"({wrong / C.sum():.2%})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(OUT, "fig_f10_confusion_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("F10 ->", out)
    return {
        "total": int(C.sum()), "wrong": int(wrong),
        "recall": [round(float(x), 4) for x in np.diag(recall)],
        "precision": [round(float(x), 4) for x in np.diag(precision)],
    }


def fig_f11(cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = np.full((9, 9), np.nan)
    cnt = np.zeros((9, 9), dtype=int)
    for r in cells:
        cnt[r["row"], r["col"]] += 1
        acc[r["row"], r["col"]] = (acc[r["row"], r["col"]]
                                   + (r["gt"] == r["pred"])) if np.isnan(acc[r["row"], r["col"]]) \
            else acc[r["row"], r["col"]] + (r["gt"] == r["pred"])
    acc = acc / cnt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].set_title("Per-position accuracy (%)")
    im = axes[0].imshow(acc * 100, cmap="RdYlGn", vmin=85, vmax=100)
    axes[1].set_title("Per-position sample count")
    im2 = axes[1].imshow(cnt, cmap="Blues")
    for ax in axes:
        ax.set_xticks(range(9)); ax.set_yticks(range(9))
        ax.set_xticklabels([f"c{i+1}" for i in range(9)], fontsize=8)
        ax.set_yticklabels([f"r{i+1}" for i in range(9)], fontsize=8)
    for i in range(9):
        for j in range(9):
            axes[0].text(j, i, f"{acc[i, j] * 100:.0f}", ha="center", va="center",
                         fontsize=7, color="black")
            axes[1].text(j, i, f"{cnt[i, j]}", ha="center", va="center",
                         fontsize=7, color="black")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    fig.colorbar(im2, ax=axes[1], fraction=0.046)
    fig.suptitle("F11 - sealed 210-puzzle test, recognition mode (definite cells)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(OUT, "fig_f11_position_heatmaps.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("F11 ->", out)
    return {
        "accuracy_pct": [[round(float(acc[i, j] * 100), 1) for j in range(9)] for i in range(9)],
        "samples": [[int(cnt[i, j]) for j in range(9)] for i in range(9)],
    }


def model_facts():
    import torch
    ck = torch.load(os.path.join(REPO, "models", "best_model.pt"), map_location="cpu")
    h = ck["histories"]
    import json
    T = json.load(open(os.path.join(REPO, "models",
                                    "digit_cnn.pth.temperature.json"), encoding="utf-8"))
    return {
        "checkpoint": "models/best_model.pt",
        "weights": "models/digit_cnn.pth",
        "architecture": "double-conv GAP DigitCNN (48x48, 32->64->128, GAP head)",
        "params": 289520,
        "best_score": round(float(ck["best_score"]), 4),
        "epochs_hist": len(h["val_score"]),
        "early_stopped": True,
        "temperature": round(float(T["temperature"]), 3),
        "history": {
            "train_loss": [round(float(x), 4) for x in h["train_loss"]],
            "val_loss": [round(float(x), 4) for x in h["val_loss"]],
            "val_acc": [round(float(x), 4) for x in h["val_acc"]],
            "val_score": [round(float(x), 4) for x in h["val_score"]],
            "val_photo_acc": [round(float(x), 4) for x in h["val_photo_acc"]],
        },
    }


def main():
    cells = load_cells()
    C = confusion(cells)
    log = load_log()
    f10 = fig_f10(C, cells)
    f11 = fig_f11(cells)
    bundle = {
        "model": model_facts(),
        "evaluation": log,
        "confusion_matrix": {
            "classes": ["empty"] + [str(d) for d in range(1, 10)],
            "counts": C.tolist(),
            "summary": f10,
        },
        "position": f11,
        "per_class_counts": {
            str(c): int((np.array([r["gt"] for r in cells]) == c).sum())
            for c in range(10)
        },
    }
    out = os.path.join(OUT, "final_metrics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)
    print("final_metrics.json ->", out)
    print("\n=== headline ===")
    for mode, m in log.items():
        print(mode, "| definite", m.get("definite"), "| digit", m.get("digit"),
              "| empty", m.get("empty"), "| exact", m.get("exact"),
              "| solve", m.get("solve"), "| wrong conf", m.get("conf"))


if __name__ == "__main__":
    main()
