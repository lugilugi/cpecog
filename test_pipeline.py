"""Regression tests for the CPECOG1 pipeline (no test framework - run directly:

    python test_pipeline.py

Covers the fixed-cell-boundary extraction, scale-correct line-grid detection,
preprocessing semantics, solver correctness (duplicate rejection, correction
budget, release branching, re-sensing kwargs), and model-loading validation.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import digit_cnn
import sudoku_core as sc

PUZZLE = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
], dtype=int)


def is_valid_sudoku(g):
    if g.shape != (9, 9):
        return False
    for i in range(9):
        if sorted(g[i, :].tolist()) != list(range(1, 10)):
            return False
        if sorted(g[:, i].tolist()) != list(range(1, 10)):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if sorted(g[br:br + 3, bc:bc + 3].flat) != list(range(1, 10)):
                return False
    return True


class TestExtraction(unittest.TestCase):
    def test_fractional_tiling_covers_full_warp(self):
        for size in (594, 600, 900):
            warped = np.zeros((size, size), dtype=np.uint8)
            cells = sc.extract_cells(warped)
            self.assertEqual(len(cells), 81)
            rows = [c.shape[0] for c in cells[0::9]]
            cols = [c.shape[1] for c in cells[:9]]
            self.assertEqual(sum(rows), size, f"rows must tile {size}")
            self.assertEqual(sum(cols), size, f"cols must tile {size}")
            self.assertEqual(cells[0].shape, cells[-1].shape)

    def test_non_square_warp(self):
        warped = np.zeros((450, 600), dtype=np.uint8)
        cells = sc.extract_cells(warped)
        self.assertEqual(len(cells), 81)
        self.assertEqual(sum(c.shape[0] for c in cells[0::9]), 450)
        self.assertEqual(sum(c.shape[1] for c in cells[:9]), 600)

    def test_all_pixels_covered_exactly_once(self):
        warped = (np.arange(600 * 600).reshape(600, 600) % 251).astype(np.uint8)
        cells = sc.extract_cells(warped)
        rows = [int(round(i * 600 / 9)) for i in range(10)]
        cols = [int(round(i * 600 / 9)) for i in range(10)]
        rebuilt = np.zeros_like(warped)
        for i in range(81):
            r, c = divmod(i, 9)
            rebuilt[rows[r]:rows[r + 1], cols[c]:cols[c + 1]] = cells[i]
        np.testing.assert_array_equal(
            rebuilt, warped,
            "cells must tile the warp exactly (no gaps, no overlaps, no loss)")


def _grid_img(size, spacing, thickness=2):
    img = np.full((size, size), 255, dtype=np.uint8)
    for i in range(0, size, spacing):
        img[:, i:i + thickness] = 60
        img[i:i + thickness, :] = 60
    return img


class TestGridDetection(unittest.TestCase):
    def test_line_spacing_scales_with_source(self):
        # 1000px image, ~100px spacing: old size-based gate (36..84) rejected
        # this; the source-relative gate must accept it.
        warped = sc.line_grid_quad(_grid_img(1000, 100))
        self.assertIsNotNone(warped)
        self.assertEqual(warped.shape[:2], (600, 600))
        # small image, ~30px spacing: old gate (36..84) rejected this too.
        warped = sc.line_grid_quad(_grid_img(300, 30))
        self.assertIsNotNone(warped)

    def test_sparse_lines_rejected(self):
        self.assertIsNone(sc.line_grid_quad(_grid_img(2000, 400)))


class TestPreprocessing(unittest.TestCase):
    def test_shape_dtype_range(self):
        out = digit_cnn.preprocess_cell(np.full((66, 66), 255, np.uint8))
        self.assertEqual(out.shape, (48, 48, 1))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_empty_cell_is_black(self):
        out = digit_cnn.preprocess_cell(np.full((66, 66), 255, np.uint8))
        self.assertEqual(float(out.max()), 0.0)

    def test_digit_survives(self):
        img = np.full((66, 66), 255, np.uint8)
        cv2.rectangle(img, (20, 20), (40, 40), 0, -1)
        out = digit_cnn.preprocess_cell(img)
        self.assertGreater(float(out.max()), 0.0)

    def test_make_empty_cells_distribution(self):
        X, y = digit_cnn.make_empty_cells(n_per_class=400, seed=3)
        self.assertEqual(X.shape, (400, 48, 48, 1))
        self.assertTrue((y == 0).all())
        black = float((X.reshape(400, -1).max(axis=1) == 0).mean())
        self.assertGreaterEqual(black, 0.90, "~97% clean empties expected")
        # fragments that survive must be letterboxed to a reasonable size
        frags = X.reshape(400, -1).max(axis=1) > 0
        self.assertTrue(frags.any())
        ink = X[frags].reshape(len(frags[frags]), -1).mean(axis=1)
        self.assertTrue((ink > 0.001).all())


class TestSolver(unittest.TestCase):
    def test_plain_solver(self):
        sol, ok = sc.solve_sudoku(PUZZLE.copy())
        self.assertTrue(ok)
        self.assertTrue(is_valid_sudoku(sol))
        for r in range(9):
            for c in range(9):
                if PUZZLE[r, c]:
                    self.assertEqual(sol[r, c], PUZZLE[r, c])

    def test_duplicate_givens_rejected(self):
        dup = PUZZLE.copy()
        dup[0, 1] = 5                      # duplicate 5 in row 0
        _, ok = sc.solve_sudoku(dup)
        self.assertFalse(ok)

    def test_correction_fills_missing_clue(self):
        sol, _ = sc.solve_sudoku(PUZZLE.copy())
        idx = 0
        recognized = sol.copy()
        recognized.flat[idx] = 0           # CNN guessed empty on a digit cell
        candidates = {idx: [int(sol.flat[idx])]}
        fixed, ok = sc.solve_with_correction(recognized, candidates, [idx],
                                             change_budget=0)
        self.assertTrue(ok)
        self.assertTrue(is_valid_sudoku(fixed))
        self.assertEqual(int(fixed.flat[idx]), int(sol.flat[idx]))

    def test_correction_budget_respected(self):
        sol, _ = sc.solve_sudoku(PUZZLE.copy())
        idx = 5
        true_v = int(sol.flat[idx])
        r, c = divmod(idx, 9)
        wrong = None
        for d in range(1, 10):
            if d != true_v and d in sol[r, :]:
                wrong = d
                break
        self.assertIsNotNone(wrong, "need a digit duplicated by the swap")
        recognized = sol.copy()
        recognized.flat[idx] = wrong
        candidates = {idx: [wrong, true_v]}
        fixed, ok = sc.solve_with_correction(recognized, candidates, [idx],
                                             change_budget=0)
        # budget 0: the wrong digit cannot be placed (duplicate), the true one
        # costs 1 change -> must fail
        self.assertFalse(ok)
        fixed, ok = sc.solve_with_correction(recognized, candidates, [idx],
                                             change_budget=1)
        self.assertTrue(ok)
        self.assertTrue(is_valid_sudoku(fixed))
        self.assertEqual(int(fixed.flat[idx]), true_v)

    def test_release_branch_when_zero_not_first(self):
        """Regression: restricted domains used to dead-end the search forever
        when the releasable cell's candidate list did not START with 0.

        x's domain is restricted to {t} (t is already fixed at y in the same
        row), so forcing it stranding-wise contradicts; the release branch
        (0 in the candidate list, NOT first) must be tried within a tiny
        node budget.
        """
        sol, _ = sc.solve_sudoku(PUZZLE.copy())
        x, y = 0, 1                        # (0,0) and (0,1): same row
        tx, t = int(sol.flat[x]), int(sol.flat[y])
        self.assertEqual(t, sol[0, 1])
        self.assertEqual(tx, sol[0, 0])
        recognized = sol.copy()
        recognized.flat[x] = t             # wrong guess duplicating y
        candidates = {x: [t, 0, tx]}       # release NOT first (suspect order)
        fixed, ok = sc.solve_with_correction(recognized, candidates, [x],
                                             max_nodes=500)
        self.assertTrue(ok, "release branch must repair the dead-end")
        self.assertTrue(is_valid_sudoku(fixed))
        self.assertEqual(int(fixed.flat[x]), tx)
        self.assertEqual(int(fixed.flat[y]), t)

    def test_violated_cells_found(self):
        bad = PUZZLE.copy()
        bad[0, 1] = 5
        suspects, violated = sc.find_violated_cells(bad)
        self.assertIn(0, suspects)
        self.assertIn(1, suspects)
        self.assertIn(0 * 9 + 2, violated)  # same row is a violated unit

    def test_resensing_receives_preprocess_kwargs(self):
        sol, _ = sc.solve_sudoku(PUZZLE.copy())
        grid = sol.copy()
        grid[0, 1] = 5                     # duplicate -> suspects -> re-sense
        cells = [np.full((66, 66), 255, np.uint8) for _ in range(81)]
        probs = np.full((81, 10), 0.05)
        probs[np.arange(81), grid.reshape(-1)] = 0.6
        seen = {}

        def fake_variants(cell, **kwargs):
            seen.update(kwargs)
            return [np.zeros((48, 48), np.float32)] * 4

        def fake_classify(views):
            return np.tile(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                           (4, 1))

        with mock.patch.object(sc, "preprocess_variants", side_effect=fake_variants):
            sc.solve_with_resensing(grid, probs, cells, fake_classify,
                                    max_rounds=1,
                                    preprocess_kwargs={"margin_frac": 0.05})
        self.assertEqual(seen.get("margin_frac"), 0.05,
                         "preprocess kwargs must reach the re-sense variants")


class TestModelLoading(unittest.TestCase):
    def test_incompatible_weights_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "old.pth")
            import torch
            torch.save({"foo": 1, "bar": 2}, p)
            with self.assertRaises(ValueError):
                digit_cnn.load_digit_model(p, device="cpu")

    def test_inference_helpers_force_eval_mode(self):
        import torch
        model = digit_cnn.DigitCNN()
        model.train()
        self.assertTrue(model.training)
        inputs = np.zeros((2, 48, 48, 1), dtype=np.float32)
        digit_cnn.classify_preprocessed(inputs, model, device="cpu")
        self.assertFalse(model.training, "inference helpers must call model.eval()")


if __name__ == "__main__":
    unittest.main(verbosity=2)
