"""Run the end-to-end Sudoku solver on a single image.

Usage:
    python run_sudoku.py puzzle.jpg
    python run_sudoku.py puzzle.jpg --show
    python run_sudoku.py puzzle.jpg --save result.png
    python run_sudoku.py puzzle.jpg --model digit_cnn.pth --size 600

Pipeline: detect grid -> perspective warp -> 81 cells -> adaptive-threshold
preprocessing -> CNN digit recognition -> MRV backtracking solver.
"""
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

import sudoku_core as sc
from digit_cnn import load_digit_model, load_temperature, predict_cells_probs, \
    classify_preprocessed


def imread_any(path):
    """cv2.imread with a Pillow fallback (webp/jpeg cv2 cannot decode)."""
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


def detect_and_warp(gray, size=600):
    """Return (warped_grid, quad) using contour detection with line fallback.

    Returns (None, None) when no grid is detected - no center-crop fallback.
    """
    quad, _ = sc.detect_grid_contour(gray)
    if quad is not None:
        return sc.four_point_transform(gray, quad, size), quad
    warped = sc.line_grid_quad(gray, size)
    if warped is not None:
        return warped, None
    return None, None


def print_grid(grid, title):
    print(title)
    for r in range(9):
        row = " | ".join(
            " ".join(str(v) for v in grid[r, c:c + 3]) for c in range(0, 9, 3))
        print(row)
        if r in (2, 5):
            print("-" * len(row))


def draw_grid(ax, grid, title):
    ax.imshow(np.zeros((9, 9, 3), dtype=np.uint8))
    for i in range(10):
        lw = 3 if i % 3 == 0 else 1
        ax.axhline(i - 0.5, color="k", lw=lw)
        ax.axvline(i - 0.5, color="k", lw=lw)
    for r in range(9):
        for c in range(9):
            if grid[r, c]:
                ax.text(c, r, str(int(grid[r, c])), ha="center", va="center",
                        fontsize=18, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def build_figure(original, quad, warped, grid, solved, ok):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0][0].imshow(img_rgb)
    if quad is not None:
        pts = np.vstack([quad.reshape(4, 2), quad.reshape(4, 2)[0]])
        axes[0][0].plot(pts[:, 0], pts[:, 1], "r-", lw=2)
        axes[0][0].plot(pts[:-1, 0], pts[:-1, 1], "ro", ms=6)
    axes[0][0].axis("off")
    axes[0][0].set_title("original photo (red = detected grid)")
    axes[0][1].imshow(warped, cmap="gray")
    axes[0][1].axis("off")
    axes[0][1].set_title("warped grid (600x600)")
    draw_grid(axes[1][0], grid, "recognized puzzle (0 = empty)")
    draw_grid(axes[1][1], solved if ok else grid,
              "solution" if ok else "no solution found")
    fig.tight_layout()
    return fig


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


def main():
    parser = argparse.ArgumentParser(description="Solve a sudoku from an image.")
    parser.add_argument("image", help="path to the sudoku photo")
    parser.add_argument("--model", default="models/digit_cnn.pth", help="CNN weights")
    parser.add_argument("--size", type=int, default=600, help="warped grid size")
    parser.add_argument("--show", action="store_true", help="show matplotlib figure")
    parser.add_argument("--save", help="save annotated figure to this path")
    args = parser.parse_args()

    img = imread_any(args.image)
    if img is None:
        print(f"error: could not read image '{args.image}'")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_digit_model(args.model, device=device)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    temperature = load_temperature(args.model)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    warped, quad = detect_and_warp(gray, args.size)
    if warped is None:
        print(f"grid detected:    no (contour and line-grid detection failed)")
        print("status: no grid detected in this image")
        return 1
    cells = sc.extract_cells(warped, args.size)
    probs = predict_cells_probs(cells, model, device=device, temperature=temperature)
    grid = probs.argmax(1).reshape(9, 9)

    print(f"image:            {args.image}")
    print(f"grid detected:    yes ({'contour' if quad is not None else 'line-grid'})")
    print_grid(grid, "recognized puzzle:")

    probe_stats = {}
    _, probe_ok = sc.solve_sudoku(grid, max_nodes=10_000, stats=probe_stats)
    print(f"solver probe:     {'solved cleanly' if probe_ok else 'no solution / node limit'} "
          f"({probe_stats['nodes']} nodes, {probe_stats['propagated']} propagated fills)")
    if not probe_ok:
        suspects, _ = sc.find_violated_cells(grid)
        if suspects:
            print("warning: recognized grid does not solve - duplicate-digit "
                  "suspects at " + ", ".join(f"({i // 9},{i % 9})"
                  for i in sorted(suspects)))
        else:
            low = sorted(range(81), key=lambda i: probs[i].max())[:5]
            print("warning: recognized grid does not solve cleanly - least "
                  "confident cells: " + ", ".join(f"({i // 9},{i % 9})" for i in low))

    corr_stats = {}
    solved, ok, n_resensed = sc.solve_with_resensing(
        grid, probs, cells,
        lambda views: classify_preprocessed(views, model, device),
        stats=corr_stats)
    print(f"solver:           correction {corr_stats.get('nodes', 0)} nodes "
          f"({corr_stats.get('propagated', 0)} propagated, "
          f"node limit hit: {corr_stats.get('limit_hit', False)}), "
          f"re-sensed {n_resensed} cells")
    if ok and valid_solution(solved):
        print_grid(solved, "solution:")
        print("status: solved (valid solution)")
    else:
        print("status: no solution found (digit recognition errors?)")
        if args.show or args.save:
            fig = build_figure(img, quad, warped, grid, solved, ok)
            if args.save:
                fig.savefig(args.save, dpi=150)
                print(f"saved figure to {args.save}")
            if args.show:
                plt.show()
        return 1

    if args.show or args.save:
        fig = build_figure(img, quad, warped, grid, solved, ok)
        if args.save:
            fig.savefig(args.save, dpi=150)
            print(f"saved figure to {args.save}")
        if args.show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
