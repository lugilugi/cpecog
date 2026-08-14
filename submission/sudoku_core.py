import os

import cv2
import numpy as np


def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(img, pts, size=600):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (size, size))


def detect_grid_contour(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    biggest = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
    if len(approx) == 4:
        return approx, thresh
    return None, thresh


def detect_grid_lines(gray, line_gap=12, min_cluster=3):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # catch both light-gray and dark lines: pixels notably darker than image mean
    bg = float(np.percentile(blur, 90))
    line_mask = ((blur < max(90, bg - 40)) & (blur < 235)).astype(np.uint8) * 255

    # horizontal lines: long thin structures
    klen = max(w // 12, 1)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    hlines = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, hk)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
    vlines = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, vk)

    def locate(binmap, axis):
        dens = binmap.mean(axis=axis)
        idxs = np.where(dens > 0.05)[0]
        clusters = []
        for i in idxs:
            if clusters and i - clusters[-1][-1] <= line_gap:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        return [int(np.mean(c)) for c in clusters if len(c) >= min_cluster]

    h = locate(hlines, axis=1)   # rows where horizontal lines live
    v = locate(vlines, axis=0)
    return (h, v)


def line_grid_quad(gray, size=600):
    res = detect_grid_lines(gray)
    if res is None:
        return None
    h, v = res
    if len(h) < 9 or len(v) < 9:
        return None
    h = sorted(h)
    v = sorted(v)
    dh = np.median(np.diff(h))
    dv = np.median(np.diff(v))
    # Line spacing is measured in SOURCE pixels: the grid occupies roughly the
    # full source image, so the interval scales with the source min dimension,
    # NOT with the requested output `size` (a 1200px image has ~110px spacing,
    # which old size-based bounds rejected; a 300px one fell below them).
    ref = max(min(gray.shape), 1)
    if not (ref * 0.06 < dh < ref * 0.14 and ref * 0.06 < dv < ref * 0.14):
        return None
    # outer boundary = one grid interval beyond first/last line
    y0 = max(0, int(h[0] - dh / 2))
    y1 = min(gray.shape[0], int(h[-1] + dh / 2))
    x0 = max(0, int(v[0] - dv / 2))
    x1 = min(gray.shape[1], int(v[-1] + dv / 2))
    src = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32")
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (size, size))


def extract_cells(warped, size=600):
    """81 row-major cells from a warped grid, using FRACTIONAL boundaries.

    size // 9 truncates (600 -> 66 px), misaligning every later row/column and
    discarding the final 6 px of the warp. Boundaries are instead placed at
    rounded i*h/9 positions, so the cells tile the FULL warp exactly (600 ->
    0, 67, 133, ..., 600). The `size` argument is kept for callers but the
    actual warp shape wins, so non-square warps are handled too.
    """
    h, w = warped.shape[:2]
    rows = [int(round(i * h / 9)) for i in range(10)]
    cols = [int(round(i * w / 9)) for i in range(10)]
    cells = []
    for r in range(9):
        for c in range(9):
            cell = warped[rows[r]:rows[r + 1], cols[c]:cols[c + 1]]
            cells.append(cell)
    return cells


def preprocess_cell_legacy(cell, target=48):
    """FROZEN legacy preprocessing -> (48,48) float32.

    The exact pre-study pipeline (same algorithm as
    digit_cnn.preprocess_cell_legacy, which adds the channel dim): adaptive
    threshold (block 15, constant 7, inverted) with an Otsu fallback for
    washed-out/low-contrast cells, 4% margin strip, no dilation (strokes keep
    their natural print weight), largest component only (tiny specks ->
    empty), aspect-ratio-preserving letterboxed resize (digits are never
    stretched). Kept for A/B testing and as a regression fallback.
    """
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    if cell.dtype != np.uint8:
        cell = np.clip(cell, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 7)
    if (th > 0).mean() < 0.005 and blur.std() > 18:
        otsu = cv2.threshold(blur, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        if (otsu > 0).mean() > (th > 0).mean():
            th = otsu
    return _finish_preprocess_legacy(th, target)


def preprocess_cell(cell, target=48, **kwargs):
    """Phase 1 lenient preprocessing -> (48,48) float32.

    Same algorithm as digit_cnn.preprocess_cell (which adds the channel dim):
    adaptive threshold (block 15, constant 7, inverted) with an Otsu fallback
    only when adaptive is clearly degenerate, then digit_cnn's
    `_finish_preprocess`: margin strip, line/corner/tiny fragment removal by
    shape, split strokes conditionally re-joined with a thin bridge, all
    surviving components kept, conservative empty, letterboxed resize.
    `**kwargs` are forwarded to `_preprocess_new` (margin_frac,
    min_area_frac, empty_frac, corner_span, ... for A/B sweeps).
    """
    from digit_cnn import _preprocess_new
    return _preprocess_new(cell, target, **kwargs)[..., 0]


def _finish_preprocess_legacy(th, target=48):
    """FROZEN legacy finishing steps after thresholding: light margin strip,
    largest component (specks -> empty), letterboxed resize -> (48,48).

    Deliberately LIGHT: no dilation (which thickened every stroke), small 4%
    margin, so the preprocessed digits keep their natural print weight.
    """
    h, w = th.shape
    m = max(1, int(min(h, w) * 0.04))
    th[:m, :] = 0
    th[-m:, :] = 0
    th[:, :m] = 0
    th[:, -m:] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if n > 1:
        areas = stats[1:, 4]
        keep = int(np.argmax(areas)) + 1
        th[labels != keep] = 0
        if stats[keep, 4] < h * w * 0.005:
            th[:] = 0
    ys, xs = np.nonzero(th)
    if len(xs) == 0:
        return np.zeros((target, target), dtype=np.float32)
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
    return canvas / 255.0


def preprocess_variants(cell, target=48, **kwargs):
    """Multiple lenient views of the same raw cell -> list of (48,48) arrays.

    Re-sensing variants used when the recognized puzzle does not solve: the
    adaptive threshold is re-run with constants 5/7/9 and a global Otsu pass
    (polarity auto-corrected), each finished by the CURRENT
    `digit_cnn._finish_preprocess` (margin strip, fragment removal, thin-
    bridge merge, letterbox - the same cleanup the main path uses). A
    low-contrast digit the default threshold lost often survives one of
    these views. `**kwargs` (margin_frac, empty_frac, ...) are forwarded to
    the finish step so A/B sweeps cover the re-sense path too.
    """
    from digit_cnn import _finish_preprocess
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    if cell.dtype != np.uint8:
        cell = np.clip(cell, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    views = []
    for c_val in (5, 7, 9):
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, c_val)
        views.append(_finish_preprocess(th, target, **kwargs))
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if (otsu > 0).mean() > 0.5:                # digit came out black -> flip
        otsu = 255 - otsu
    views.append(_finish_preprocess(otsu, target, **kwargs))
    return views


def sudoku_from_cells(cells, predict_fn):
    """81 row-major cells -> 9x9 grid.

    Class 0 = empty cell, classes 1-9 = digits (the CNN is trained on real
    empty cells, so no separate ink-ratio gate is needed).
    """
    grid = []
    for cell in cells:
        grid.append(int(predict_fn(cell)))
    return np.array(grid).reshape(9, 9)


def candidates_from_probs(probs, max_alts=3, conf_thresh=0.9, order_cap=12,
                          priority=None):
    """Build (candidates, order) for solve_with_correction from CNN softmax.

    probs: (81, 10) softmax rows; class 0 = empty cell.
    candidates: {cell_index: [alternatives, best first]}; the current argmax
        guess is always first, the rest are the next-best classes (including
        class 0, so correction may also *remove* a wrongly recognized digit).
    order: the order_cap cell indices to reconsider, least confident first.
        `priority` is a {cell_index: rank} map (lower = earlier) - e.g. the
        suspects/violated cells of re-sensing - so a confident-but-wrong digit
        in a violated unit is not starved of reconsideration by the many
        low-confidence empties.
    """
    candidates = {}
    order = []
    for idx, p in enumerate(probs):
        guess = int(p.argmax())
        ranked = [int(k) for k in p.argsort()[::-1]]
        if priority and priority.get(idx, 2) == 0:
            # suspect (rank 0): its current value participates in a duplicate.
            # Try the digit ALTERNATIVES first, then removal (empty), and keep
            # the current value LAST - so the search prefers a real correction
            # over weakening the puzzle by deleting givens (which can make the
            # solution non-unique and drift from the photo)
            others = [k for k in ranked if k != guess and k != 0][:2]
            alts = (others + [0, guess]) if guess != 0 else ([0] + others)
        else:
            alts = [guess] + [k for k in ranked if k != guess][:max_alts - 1]
        candidates[idx] = alts
        if p.max() < conf_thresh or (priority and idx in priority):
            order.append(idx)
    order.sort(key=lambda idx: (priority.get(idx, 2) if priority else 1,
                                probs[idx].max()))
    if order_cap:
        order = order[:order_cap]
    return candidates, order


_DIGIT_BITS = 0x3FE        # candidate bits 1..9; bit 0 = "release" marker


def _search(grid, domains, branch_order, max_nodes, stats=None, accept_fn=None):
    """Shared backtracking search engine with constraint propagation.

    grid: 9x9 int working copy (0 = empty); filled with the solution on success.
    domains: 9x9 int bitmasks - bit d set means digit d (1..9) is still allowed
        in that cell; bit 0 marks a cell the correction path may 'release'
        (treat as a plain empty and fill it freely).
    branch_order: {cell_index: [digit, ...]} - per-cell candidate lists tried
        best-first (CNN softmax order). Cells absent from it branch over their
        remaining digits in ascending order.
    max_nodes: total branching-node budget; when hit the whole search aborts.
    accept_fn: optional validator called on every complete grid; a complete
        grid rejected by it backtracks and the search continues.
    stats: optional dict updated with nodes / propagated / limit_hit / ok.

    Assignment legality is checked against 9-bit row/col/box used-masks, so a
    digit duplicating any already-placed digit (given or guessed) is rejected
    on the spot. Propagation (naked + hidden singles) runs at every node; only
    when no cell is forced does the search branch (MRV).
    """
    DIGITS = 0x3FE
    grid = grid.copy()
    domains = [[int(domains[r, c]) for c in range(9)] for r in range(9)]
    row = [0] * 9
    col = [0] * 9
    box = [0] * 9
    for r in range(9):
        for c in range(9):
            v = int(grid[r, c])
            if v:
                bit = 1 << v
                row[r] |= bit
                col[c] |= bit
                box[3 * (r // 3) + c // 3] |= bit
    units = []
    for r in range(9):
        units.append(([(r, c) for c in range(9)], row, r))
    for c in range(9):
        units.append(([(r, c) for r in range(9)], col, c))
    for br in range(3):
        for bc in range(3):
            units.append(([(br * 3 + i, bc * 3 + j) for i in range(3) for j in range(3)],
                          box, 3 * br + bc))
    trail = []
    nodes = 0
    propagated = 0
    aborted = False

    def cand(r, c):
        return domains[r][c] & ~(row[r] | col[c] | box[3 * (r // 3) + c // 3])

    def fill(r, c, v):
        nonlocal propagated
        grid[r, c] = v
        bit = 1 << v
        row[r] |= bit
        col[c] |= bit
        box[3 * (r // 3) + c // 3] |= bit
        trail.append(("f", r, c))
        propagated += 1

    def undo(mark):
        while len(trail) > mark:
            t = trail.pop()
            if t[0] == "f":
                _, r, c = t
                v = int(grid[r, c])
                grid[r, c] = 0
                bit = 1 << v
                row[r] &= ~bit
                col[c] &= ~bit
                box[3 * (r // 3) + c // 3] &= ~bit
            else:
                _, r, c, old = t
                domains[r][c] = old

    def propagate():
        """Fill forced cells (naked + hidden singles) to a fixpoint.
        Returns True at fixpoint, False on dead end."""
        while True:
            filled_any = False
            for r in range(9):
                for c in range(9):
                    if grid[r, c]:
                        continue
                    dbits = cand(r, c) & DIGITS
                    if not dbits:
                        return False
                    if dbits & (dbits - 1) == 0:
                        fill(r, c, dbits.bit_length() - 1)
                        filled_any = True
            if filled_any:
                continue
            for cells, mask_list, mi in units:
                umask = mask_list[mi]
                for d in range(1, 10):
                    bit = 1 << d
                    if umask & bit:
                        continue
                    where = None
                    cnt = 0
                    for r, c in cells:
                        if not grid[r, c] and cand(r, c) & bit:
                            cnt += 1
                            where = (r, c)
                            if cnt > 1:
                                break
                    if cnt == 0:
                        return False
                    if cnt == 1:
                        fill(where[0], where[1], d)
                        filled_any = True
            if not filled_any:
                return True

    def dfs():
        nonlocal nodes, aborted
        if aborted:
            return False
        nodes += 1
        if nodes > max_nodes:
            aborted = True
            return False
        mark0 = len(trail)
        prop_ok = propagate()
        if not prop_ok:
            # Drop the contradictory partial fills: propagation just forced a
            # restricted correction domain's digit (or stranded a digit that
            # only a restricted cell could hold) - the state cannot be
            # repaired by MRV branching on the zero-candidate cells.
            undo(mark0)
        best = None
        best_cnt = 10
        if not prop_ok:
            # Branch on a releasable order cell first (release = its first
            # action when the CNN guessed empty): expanding its domain to the
            # full DIGITS set is what repairs the contradiction. A cell whose
            # candidate list merely CONTAINS 0 (suspects keep 0 last so the
            # search prefers a real correction) can also repair it - fall
            # back to those when no release-first cell is free.
            for idx, alts in branch_order.items():
                if alts and alts[0] == 0:
                    r, c = divmod(idx, 9)
                    if grid[r, c] == 0:
                        best, best_cnt = (r, c), -1
                        break
            if best is None:
                for idx, alts in branch_order.items():
                    if 0 in alts:
                        r, c = divmod(idx, 9)
                        if grid[r, c] == 0:
                            best, best_cnt = (r, c), -1
                            break
        if best is None:
            for r in range(9):
                for c in range(9):
                    if grid[r, c]:
                        continue
                    cnt = (cand(r, c) & DIGITS).bit_count()
                    if cnt < best_cnt:
                        best_cnt = cnt
                        best = (r, c)
                        if cnt == 1:
                            break
                if best_cnt == 1:
                    break
        if best is None:
            return accept_fn is None or accept_fn(grid)
        r, c = best
        cnd = cand(r, c)
        ordered = branch_order.get(r * 9 + c)
        if ordered is not None:
            for v in ordered:
                mark = len(trail)
                if v == 0:
                    if not (cnd & 1):
                        continue
                    old = domains[r][c]
                    domains[r][c] = DIGITS
                    trail.append(("d", r, c, old))
                else:
                    if not (cnd & (1 << v)):
                        continue
                    fill(r, c, v)
                if dfs():
                    return True
                undo(mark)
        else:
            for d in range(1, 10):
                if cnd & (1 << d):
                    mark = len(trail)
                    fill(r, c, d)
                    if dfs():
                        return True
                    undo(mark)
        return False

    ok = dfs()
    if not ok:
        undo(0)
    if stats is not None:
        stats["nodes"] = nodes
        stats["propagated"] = propagated
        stats["limit_hit"] = aborted
        stats["ok"] = ok
    return grid, ok


def solve_sudoku(grid, max_nodes=100_000, stats=None):
    """Constraint-propagation MRV solver; returns (solution_grid, ok).

    Bitmask candidate domains, naked/hidden-single propagation at every node,
    most-constrained-cell branching. Duplicate givens are rejected upfront.
    Optional stats dict: {nodes, propagated, limit_hit, ok}.
    """
    grid = grid.copy().astype(int)

    def valid_givens(g):
        for r in range(9):
            seen = [v for v in g[r, :] if v]
            if len(set(seen)) != len(seen):
                return False
        for c in range(9):
            seen = [v for v in g[:, c] if v]
            if len(set(seen)) != len(seen):
                return False
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                seen = [v for v in g[br:br + 3, bc:bc + 3].flat if v]
                if len(set(seen)) != len(seen):
                    return False
        return True

    if not valid_givens(grid):
        if stats is not None:
            stats.update(nodes=0, propagated=0, limit_hit=False, ok=False)
        return grid, False
    domains = np.where(grid != 0, 1 << grid, _DIGIT_BITS).astype(int)
    return _search(grid, domains, {}, max_nodes, stats)


def solve_with_correction(grid, candidates, order, max_nodes=200_000,
                          change_budget=4, stats=None, budget_ref=None):
    """Confidence-aware error correction as ONE integrated search.

    grid: best-guess puzzle (0 = empty cell).
    candidates: dict cell_index -> list of digit alternatives, best first
        (from candidates_from_probs; class 0 = release the cell, letting the
        solver fill it freely - a wrongly recognized digit gets replaced).
    order: cell indices to reconsider, least confident first.
    budget_ref: the FIRST-PASS recognized grid used for the change budget.
        Defaults to `grid`. Re-sensing updates `grid` between rounds while the
        budget keeps counting edits against what the CNN first saw.

    Non-order cells are FIXED to the recognized values in `grid` (duplicate
    digits among them are rejected upfront); order cells branch over their CNN
    candidate lists inside the same propagation/MRV search - no per-leaf
    re-solve. A solution is accepted only if at most change_budget RECOGNIZED
    DIGITS of `budget_ref` differ - empty cells being filled by the solver do
    not count; budget-violating complete grids are rejected and the search
    continues. This anchors the solution to the photo: a correction that
    deletes or alters many digits would 'solve' an unrelated valid sudoku.
    Returns (solution_grid, ok).
    """
    base = grid.copy().astype(int)
    ref = (budget_ref if budget_ref is not None else base).copy()
    order_set = set(order)
    for idx in order:
        r, c = divmod(idx, 9)
        base[r, c] = 0                    # variable cells start empty; the
                                          # search assigns them from `candidates`

    def fixed_units_ok():
        def unit_ok(cells):
            seen = set()
            for idx in cells:
                if idx in order_set:
                    continue
                v = base.flat[idx]
                if v:
                    if v in seen:
                        return False
                    seen.add(v)
            return True

        for r in range(9):
            if not unit_ok([r * 9 + c for c in range(9)]):
                return False
        for c in range(9):
            if not unit_ok([r * 9 + c for r in range(9)]):
                return False
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                if not unit_ok([(br + i) * 9 + (bc + j) for i in range(3) for j in range(3)]):
                    return False
        return True

    if not fixed_units_ok():
        if stats is not None:
            stats.update(nodes=0, propagated=0, limit_hit=False, ok=False)
        return base, False

    domains = np.full((9, 9), _DIGIT_BITS, dtype=int)
    branch_order = {}
    for idx in order:
        r, c = divmod(idx, 9)
        alts = candidates[idx]
        mask = 0
        for v in alts:
            mask |= 1 << v
        if not mask & _DIGIT_BITS:
            mask = _DIGIT_BITS          # release-only cell: fill freely
        domains[r, c] = mask
        branch_order[idx] = alts
    for idx in range(81):
        if idx not in order_set:
            v = base.flat[idx]
            if v:
                r, c = divmod(idx, 9)
                domains[r, c] = 1 << v

    def accept(g):
        return sum(1 for idx in order
                   if ref.flat[idx] != 0 and g.flat[idx] != ref.flat[idx]) <= change_budget

    return _search(base, domains, branch_order, max_nodes, stats, accept)


def find_violated_cells(grid):
    """-> (suspects, all_violated).

    suspects: cells holding a digit that appears more than once in a
    row/column/box (the prime candidates for a misread);
    all_violated: every cell of any violated unit. These are the 'chunks' the
    re-sensing solver re-examines first: a puzzle that does not solve usually
    shows its mistakes as duplicate givens within a unit.
    """
    suspects = set()
    bad = set()

    def check_unit(idx_list):
        vals = {}
        for idx in idx_list:
            v = grid[idx // 9, idx % 9]
            if v:
                vals.setdefault(v, []).append(idx)
        if any(len(l) > 1 for l in vals.values()):
            bad.update(idx_list)
            for l in vals.values():
                if len(l) > 1:
                    suspects.update(l)

    for r in range(9):
        check_unit([r * 9 + c for c in range(9)])
    for c in range(9):
        check_unit([r * 9 + c for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            check_unit([(br + i) * 9 + (bc + j) for i in range(3) for j in range(3)])
    return suspects, bad


def solve_with_resensing(grid, probs, cells, classify_fn, max_correction_cells=12,
                         max_rounds=2, conf_thresh=0.9, max_nodes=200_000,
                         stats=None, preprocess_kwargs=None):
    """Constraint-guided re-sensing solve.

    grid: best-guess puzzle (0 = empty cell); probs: (81,10) softmax from the
        first pass; cells: the 81 raw cell images (for re-preprocessing);
        classify_fn: (N,48,48,1) preprocessed inputs -> (N,10) softmax.
        preprocess_kwargs: preprocessing overrides (margin_frac, ...) for the
        re-sense variants - kept in sync with the first-pass pipeline.

    Strategy (chunk by chunk):
    1. try the plain bounded correction first;
    2. if the puzzle still does not solve, find the VIOLATED units (rows/
       columns/boxes with duplicate digits) and RE-SENSE those cells first
       (re-run preprocessing variants: adaptive C=5/7/9 + Otsu, classify each,
       average the opinions and blend with the first pass), then the
       least-confident cells;
    3. retry the bounded correction with the updated probs, up to max_rounds.

    The working puzzle `base` is UPDATED to each re-sensed cell's new argmax
    before the next round, so a changed opinion actually reaches the search;
    the FIRST-PASS grid stays the change-budget reference, so the budget keeps
    counting edits against what the CNN first recognized.

    Returns (solution_grid, ok, n_resensed). `stats` accumulates the
    {nodes, propagated, limit_hit} counters of every correction search.
    """
    base = grid.copy()
    first_pass = grid.copy()
    cur_probs = probs.copy()
    resensed = set()
    total_resensed = 0

    for round_no in range(max_rounds + 1):
        cap = max_correction_cells * (round_no + 1)
        suspects, violated = find_violated_cells(base)
        # re-sense only when there is something to disambiguate: violations
        # exist (round 0) or the earlier rounds failed (round_no > 0)
        if round_no > 0 or suspects:
            focus = list(suspects)
            for i in range(81):
                if len(focus) >= cap:
                    break
                if i not in focus and cur_probs[i].max() < conf_thresh:
                    focus.append(i)
            focus.sort(key=lambda i: (0 if i in suspects else 1, cur_probs[i].max()))
            focus = focus[:cap]
            for idx in focus:
                if idx in resensed:
                    continue
                views = np.stack(preprocess_variants(cells[idx], **preprocess_kwargs or {}))[..., None].astype(np.float32)
                vp = np.asarray(classify_fn(views), dtype=np.float64).reshape(-1, 10)
                # conservative blend: the first-pass opinion stays dominant so a
                # noisy re-sense cannot corrupt a confident correct cell
                cur_probs[idx] = 0.7 * cur_probs[idx] + 0.3 * vp.mean(axis=0)
                base.flat[idx] = int(cur_probs[idx].argmax())   # update the puzzle
                resensed.add(idx)
                total_resensed += 1

        rank = {i: 0 for i in suspects}
        rank.update({i: 1 for i in violated if i not in rank})
        candidates, order = candidates_from_probs(cur_probs, max_alts=3,
                                                  conf_thresh=conf_thresh,
                                                  order_cap=cap,
                                                  priority=rank)
        st = {}
        solved, ok = solve_with_correction(base, candidates, order,
                                           max_nodes=max_nodes, stats=st,
                                           budget_ref=first_pass)
        if stats is not None:
            for k in ("nodes", "propagated", "limit_hit"):
                stats[k] = stats.get(k, 0) + st.get(k, 0)
            stats["ok"] = ok
        if ok:
            return solved, True, total_resensed
        if round_no == max_rounds:
            break

    return base, False, total_resensed


def extract_puzzle(image, size=600):
    """Detect the Sudoku grid in an image and return (warped_grid, cells, original).

    Returns (None, None, img) when no grid is detected - no blind center-crop
    fallback, so failures are honest and reported as failures.
    """
    if isinstance(image, (str, os.PathLike)):
        img = cv2.imread(str(image))
    else:
        img = image
    if img is None:
        return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad, thresh = detect_grid_contour(gray)
    if quad is not None:
        warped = four_point_transform(gray, quad, size)
    else:
        warped = line_grid_quad(gray, size)
    if warped is None:
        return None, None, img
    cells = extract_cells(warped, size)
    return warped, cells, img


def full_pipeline(image_path, predict_fn, size=600):
    warped, cells, img = extract_puzzle(image_path, size)
    if warped is None:
        return None, None, None, None
    grid = sudoku_from_cells(cells, predict_fn)
    return grid, warped, cells, img
