"""
ROI detection utilities for XRF coarse scans.

Strategy (simple → can be swapped for more complex methods later):
    1. Threshold the intensity map to separate signal from noise
    2. Label connected regions (scipy.ndimage)
    3. Return bounding boxes in both pixel indices and real mm coordinates
"""

import numpy as np
from scipy import ndimage


# ── individual threshold methods ──────────────────────────────────────────────

def _thresh_mad(x, k):
    """Median + k * (MAD / 0.6745)  — most robust, insensitive to outliers."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med + k * (mad / 0.6745)


def _thresh_trimmed(x, k, alpha=0.05):
    """Trimmed mean + k * trimmed std — discard top alpha fraction first."""
    cutoff = np.quantile(x, 1.0 - alpha)
    bg = x[x <= cutoff]
    return bg.mean() + k * bg.std()


def _thresh_iqr(x, k):
    """Q75 + k * IQR  — boxplot whisker rule."""
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return q75 + k * (q75 - q25)


def _thresh_otsu(x, nbins=256):
    """
    Otsu's method (vectorized): find the threshold that maximizes the
    inter-class variance between background (x <= t) and signal (x > t).
    Useful for splitting 'empty space' vs 'sample matrix' in coarse XRF maps.
    """
    hist, bin_edges = np.histogram(x, bins=nbins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(float)

    w_bg = np.cumsum(hist)
    w_fg = hist.sum() - w_bg
    sum_bg = np.cumsum(hist * centers)
    sum_total = sum_bg[-1]
    sum_fg = sum_total - sum_bg

    valid = (w_bg > 0) & (w_fg > 0)
    mean_bg = np.where(w_bg > 0, sum_bg / np.maximum(w_bg, 1), 0)
    mean_fg = np.where(w_fg > 0, sum_fg / np.maximum(w_fg, 1), 0)
    sigma_b = w_bg * w_fg * (mean_fg - mean_bg) ** 2
    sigma_b = np.where(valid, sigma_b, -np.inf)
    return float(centers[int(np.argmax(sigma_b))])


# ── Fisher inter-class variance (selection criterion) ─────────────────────────

def _fisher_score(x, t):
    """
    Variance between background (x <= t) and signal (x > t) classes.
    Higher = better separation between background and signal.

        score = w_bg * w_fg * (mu_fg - mu_bg)^2
    """
    bg = x[x <= t]
    fg = x[x > t]
    if len(bg) == 0 or len(fg) == 0:
        return -np.inf
    w_bg = len(bg) / len(x)
    w_fg = len(fg) / len(x)
    return w_bg * w_fg * (fg.mean() - bg.mean()) ** 2


# ── public interface ──────────────────────────────────────────────────────────

def threshold_map(channel_map, method="auto", k=2.0, mask=None):
    """
    Compute a binary mask of high-intensity (ROI) pixels.

    Parameters
    ----------
    channel_map : 2D array
    method : str
        'auto'    — compute MAD, trimmed and IQR candidates, keep the best
                    (highest Fisher inter-class variance score)
        'mad'     — median + k * (MAD / 0.6745)
        'trimmed' — trimmed mean + k * trimmed std  (top 5% discarded)
        'iqr'     — Q75 + k * IQR
        'sigma'   — mean + k * std  (non-robust, kept for comparison)
    k : float
        sensitivity multiplier (default 2.0, lower = more ROI pixels)
    mask : bool 2D array, optional
        If given, the threshold statistics are computed over channel_map[mask]
        only. Essential when the map was zero-filled outside a scan footprint:
        including the injected zeros collapses the robust statistics (median/
        MAD/IQR -> 0) and the threshold separates footprint-from-void instead
        of particle-from-matrix. The returned mask still covers the full map;
        intersect it with the footprint if needed.

    Returns
    -------
    mask        : bool 2D array  (True = ROI pixel)
    thresh      : float          threshold value used
    method_used : str            which method was selected
    """
    if mask is not None:
        x = channel_map[mask].ravel().astype(float)
    else:
        x = channel_map.ravel().astype(float)

    candidates = {
        "mad":     _thresh_mad(x, k),
        "trimmed": _thresh_trimmed(x, k),
        "iqr":     _thresh_iqr(x, k),
    }

    if method == "sigma":
        thresh = x.mean() + k * x.std()
        method_used = "sigma"
    elif method in candidates:
        thresh = candidates[method]
        method_used = method
    elif method == "auto":
        scores = {m: _fisher_score(x, t) for m, t in candidates.items()}
        method_used = max(scores, key=scores.get)
        thresh = candidates[method_used]
        # print detail for transparency
        print("  Threshold candidates:")
        for m, t in candidates.items():
            marker = " <-- selected" if m == method_used else ""
            print(f"    {m:10s}: {t:8.1f}  (Fisher score: {scores[m]:.3e}){marker}")
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Use 'auto', 'mad', 'trimmed', 'iqr', or 'sigma'."
        )

    roi_mask = channel_map > thresh
    return roi_mask, thresh, method_used


def _sample_threshold(values, mode="otsu_lower", level=50.0):
    """Core threshold-only helper used by sample_mask. Returns the scalar threshold."""
    x = np.asarray(values).ravel().astype(float)
    if len(x) < 20:
        return float(np.percentile(x, 5.0)) if len(x) else 0.0
    if mode == "otsu_lower":
        limit = np.percentile(x, level)
        x_low = x[x <= limit]
        return _thresh_otsu(x_low) if len(x_low) >= 20 else float(np.percentile(x, 5.0))
    if mode == "percentile":
        return float(np.percentile(x, level))
    if mode == "otsu_clip":
        clip_val = np.percentile(x, level)
        return _thresh_otsu(np.minimum(x, clip_val))
    raise ValueError(f"Unknown mode '{mode}'.")


def sample_mask(composite, kernel_px=2, mode="otsu_lower", level=50.0,
                verbose=False, restrict_to=None,
                closing_px=None, opening_px=None):
    """
    Detect the sample footprint (vs empty surroundings) on a composite XRF map.

    Strategy (different mission than ROI detection):
      - We want to separate 'void' (noise floor, off-sample) from
        'sample' (matrix + particles). Direct Otsu on the full histogram
        fails because bright particles dominate the inter-class variance
        and Otsu ends up separating matrix from particles instead.

      - Pragmatic fixes:
        * mode='otsu_lower' : apply Otsu only on the lower portion of the
          histogram (pixels with composite <= percentile `level`). This finds
          the natural bimodal cut between void and matrix-low if it exists.
        * mode='percentile' : threshold = percentile `level` of the composite.
          Use when you know roughly what fraction is void.
        * mode='otsu_clip'  : clip composite at percentile `level` before
          full-histogram Otsu. Same idea as otsu_lower but softer.

      Then: morphological closing fills small holes, opening removes islands.

    The returned mask answers "is there material here?", not "is there a
    particle". Use it to RESTRICT a finer scan to the sample area, then run
    the real ROI detection (threshold_map) on the higher-resolution data.

    Parameters
    ----------
    composite : 2D array
    kernel_px : int            morphological kernel size [pixels]
    mode      : str            'otsu_lower' | 'percentile' | 'otsu_clip'
    level     : float          percentile parameter for the chosen mode
    verbose   : bool           print histogram statistics

    Returns
    -------
    mask    : bool 2D array  (True = sample)
    thresh  : float          threshold value used
    """
    if restrict_to is not None:
        values = composite[restrict_to].ravel().astype(float)
    else:
        values = composite.ravel().astype(float)

    if verbose:
        qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        vals = np.percentile(values, qs)
        print("  Composite histogram quantiles"
              + (" (restricted)" if restrict_to is not None else "")
              + ":")
        for q, v in zip(qs, vals):
            print(f"    p{q:>2}: {v:8.1f}")
        print(f"    min: {values.min():8.1f}   max: {values.max():8.1f}")

    thresh = _sample_threshold(values, mode=mode, level=level)

    mask = composite > thresh
    if restrict_to is not None:
        mask = mask & restrict_to

    # closing fills small dark gaps INSIDE the sample (good for coverage)
    # opening removes isolated bright pixels (BAD for capturing rare particles)
    # → default: closing only. Set opening_px=N to enable noise removal.
    c_px = closing_px if closing_px is not None else kernel_px
    o_px = opening_px if opening_px is not None else 0
    if c_px > 0:
        structure = np.ones((c_px, c_px), dtype=bool)
        mask = ndimage.binary_closing(mask, structure=structure)
    if o_px > 0:
        structure = np.ones((o_px, o_px), dtype=bool)
        mask = ndimage.binary_opening(mask, structure=structure)
    if restrict_to is not None:
        mask = mask & restrict_to

    return mask, thresh


def remove_border_pixels(mask, border=1):
    """Zero out pixels on the edge of the array (frequent scan artifacts)."""
    clean = mask.copy()
    if border <= 0:
        # clean[-0:] would select the WHOLE array and blank everything.
        return clean
    clean[:border, :]  = False
    clean[-border:, :] = False
    clean[:, :border]  = False
    clean[:, -border:] = False
    return clean


def label_rois(mask, min_pixels=2, remove_border=False):
    """
    Label connected regions in a binary mask.

    Parameters
    ----------
    mask          : bool 2D array
    min_pixels    : int   minimum region size (smaller regions discarded as noise)
    remove_border : bool  strip edge pixels before labeling (avoids scan artifacts)

    Returns
    -------
    labeled : int 2D array  (0 = background, 1..N = ROI labels)
    n_rois  : int
    """
    clean = remove_border_pixels(mask) if remove_border else mask
    labeled, n_rois = ndimage.label(clean)

    # Discard regions smaller than min_pixels. Vectorized: np.bincount counts
    # every label's size in a single O(P) pass; `remove[labeled]` then maps the
    # per-label keep/drop decision back onto every pixel via fancy indexing.
    # (Replaces an O(R*P) Python loop — critical at fine resolution, ~9000 ROIs.)
    if n_rois > 0 and min_pixels > 1:
        counts = np.bincount(labeled.ravel())
        remove = counts < min_pixels
        remove[0] = False                      # never remove the background label
        labeled[remove[labeled]] = 0

    # re-label after removal
    labeled, n_rois = ndimage.label(labeled > 0)
    return labeled, n_rois


def get_bounding_boxes(labeled, xdata, ydata):
    """
    Compute bounding boxes for each labeled ROI in pixel and mm coordinates.

    Vectorized: uses ndimage.find_objects (returns all slices in one pass) and
    ndimage.sum / ndimage.center_of_mass for sizes and centers.

    Returns a list of dicts, one per ROI:
        {
          'id'    : int,
          'pixels': (row_min, row_max, col_min, col_max),   # inclusive
          'mm'    : (x_min, x_max, y_min, y_max),           # real coords [mm]
          'size'  : int,
          'center_mm': (x_center, y_center),
        }
    """
    n_rois = int(labeled.max())
    if n_rois == 0:
        return []

    slices = ndimage.find_objects(labeled)            # list of (slice_r, slice_c)
    sizes  = np.bincount(labeled.ravel(), minlength=n_rois + 1)[1:]
    coms   = np.asarray(ndimage.center_of_mass(
        np.ones_like(labeled, dtype=np.float32),
        labels=labeled, index=np.arange(1, n_rois + 1))).reshape(-1, 2)
    cx_all = np.interp(coms[:, 1], np.arange(len(xdata)), xdata)
    cy_all = np.interp(coms[:, 0], np.arange(len(ydata)), ydata)

    boxes = []
    for roi_id, sl in enumerate(slices, 1):
        if sl is None:                                # labels may be sparse
            continue
        sr, sc = sl
        r0, r1 = sr.start, sr.stop - 1
        c0, c1 = sc.start, sc.stop - 1
        boxes.append({
            "id":        roi_id,
            "pixels":    (r0, r1, c0, c1),
            "mm":        (float(xdata[c0]), float(xdata[c1]),
                          float(ydata[r0]), float(ydata[r1])),
            "size":      int(sizes[roi_id - 1]),
            "center_mm": (float(cx_all[roi_id - 1]), float(cy_all[roi_id - 1])),
        })
    return boxes


# ── scanner cost model (from Paper 1 & Paper 2) ───────────────────────────────
# Paper 1: v_max=200 mm/s, a_max=500 mm/s²
# Paper 2: Newport XPS motors, line-by-line trajectories
#
# Single source for the kinematics constants: every cost computation (here and
# in validation_utils / the scripts) defaults to these. Change them HERE when
# moving to different motors.

V_MAX_MM_S  = 200.0   # scanner max velocity [mm/s]      (Paper 1)
A_MAX_MM_S2 = 500.0   # scanner max acceleration [mm/s²] (Paper 1)
SETUP_MS    = 500.0   # per-region overhead (positioning, triggering) [ms]


def travel_time(distance_mm, v_max=V_MAX_MM_S, a_max=A_MAX_MM_S2):
    """
    Minimum time [s] for the scanner to travel distance_mm [mm].
    Uses trapezoidal velocity profile (Paper 1 kinematics).

        d_crit = v_max²/a_max  (distance to reach full speed)
        d <= d_crit → triangular profile: T = 2*sqrt(d/a_max)
        d >  d_crit → trapezoidal profile: T = v_max/a_max + d/v_max
    """
    if distance_mm <= 0:
        return 0.0
    d_crit = v_max ** 2 / a_max
    if distance_mm <= d_crit:
        return 2.0 * np.sqrt(distance_mm / a_max)
    return v_max / a_max + distance_mm / v_max


def group_rois(boxes, dwell_ms, dx, dy,
               v_max=V_MAX_MM_S, a_max=A_MAX_MM_S2, setup_ms=SETUP_MS,
               size_px_mm=None):
    """
    Greedily merge ROI pairs where merging saves time.
    Iterates until no more beneficial merges exist.

    Decision rule (T_sep vs T_merge): merge two boxes when the time wasted
    scanning the gap pixels of the merged bbox at fine resolution is smaller
    than the travel + setup cost of visiting them separately:

        merge if:  n_gap * dwell  <  T_travel(d) + T_setup

    Vectorized: each iteration computes all pairwise saving matrices via numpy
    broadcasting (O(N^2) numpy ops vs O(N^2) Python ops). For N~500 ROIs this
    is ~100x faster than the pure-Python version.

    Parameters
    ----------
    boxes      : ROI dicts from get_bounding_boxes()
    dwell_ms   : dwell time per pixel at FINE resolution [ms]
    dx, dy     : fine pixel size [mm] (the resolution the merged bbox will be
                 scanned at)
    size_px_mm : (dx_src, dy_src) pixel size [mm] that box['size'] is counted
                 in. get_bounding_boxes counts sizes on the CURRENT (coarse)
                 grid, so pass the coarse pixel size here; sizes are then
                 converted to fine-pixel units so the gap count
                 n_gap = n_merged_fine - sizes is unit-consistent. If None,
                 sizes are assumed to already be in (dx, dy) units — with
                 coarse-grid sizes that overestimates n_gap by
                 ~(coarse_px/fine_px)^2 and suppresses all merging.

    Returns a new list of (possibly merged) ROI boxes (one dict per group).
    """
    if not boxes:
        return []

    # Initialize per-group arrays (one row per active group)
    x0 = np.array([min(b["mm"][0], b["mm"][1]) for b in boxes], dtype=float)
    x1 = np.array([max(b["mm"][0], b["mm"][1]) for b in boxes], dtype=float)
    y0 = np.array([min(b["mm"][2], b["mm"][3]) for b in boxes], dtype=float)
    y1 = np.array([max(b["mm"][2], b["mm"][3]) for b in boxes], dtype=float)
    size = np.array([b["size"] for b in boxes], dtype=int)
    if size_px_mm is not None:
        factor = (size_px_mm[0] / dx) * (size_px_mm[1] / dy)
        size = np.maximum(1, np.round(size * factor)).astype(int)
    n_orig = np.ones(len(boxes), dtype=int)   # original ROI count per group

    d_crit = v_max ** 2 / a_max

    while len(x0) >= 2:
        # Pairwise merged bbox (i = rows, j = cols)
        mx0 = np.minimum(x0[:, None], x0[None, :])
        mx1 = np.maximum(x1[:, None], x1[None, :])
        my0 = np.minimum(y0[:, None], y0[None, :])
        my1 = np.maximum(y1[:, None], y1[None, :])

        # pixels in merged bbox at fine resolution
        merged_nx = np.maximum(1, np.round(np.abs(mx1 - mx0) / dx).astype(int) + 1)
        merged_ny = np.maximum(1, np.round(np.abs(my1 - my0) / dy).astype(int) + 1)
        n_merged  = merged_nx * merged_ny
        n_gap     = np.maximum(0, n_merged - size[:, None] - size[None, :])

        # Edge-to-edge distance
        dxv = np.maximum(0.0, np.maximum(x0[:, None], x0[None, :]) -
                              np.minimum(x1[:, None], x1[None, :]))
        dyv = np.maximum(0.0, np.maximum(y0[:, None], y0[None, :]) -
                              np.minimum(y1[:, None], y1[None, :]))
        dist = np.sqrt(dxv ** 2 + dyv ** 2)

        # Travel time (trapezoidal): triangular if d<=d_crit else trapezoid
        t_travel = np.where(
            dist <= 0,
            0.0,
            np.where(
                dist <= d_crit,
                2.0 * np.sqrt(np.maximum(dist, 0) / a_max),
                v_max / a_max + dist / v_max,
            ),
        ) * 1000.0  # → ms

        saving = (t_travel + setup_ms) - (n_gap * dwell_ms)

        # Mask diagonal and lower triangle (only consider i<j pairs)
        mask = np.triu(np.ones_like(saving, dtype=bool), k=1)
        saving_masked = np.where(mask, saving, -np.inf)

        flat_idx = np.argmax(saving_masked)
        best_save = saving_masked.flat[flat_idx]
        if best_save <= 0:
            break

        i, j = np.unravel_index(flat_idx, saving_masked.shape)

        # Merge j into i, then drop j
        x0[i] = mx0[i, j]; x1[i] = mx1[i, j]
        y0[i] = my0[i, j]; y1[i] = my1[i, j]
        size[i] = size[i] + size[j]
        n_orig[i] = n_orig[i] + n_orig[j]

        x0 = np.delete(x0, j); x1 = np.delete(x1, j)
        y0 = np.delete(y0, j); y1 = np.delete(y1, j)
        size = np.delete(size, j); n_orig = np.delete(n_orig, j)

    merged_boxes = []
    for gid in range(len(x0)):
        cx = (x0[gid] + x1[gid]) / 2
        cy = (y0[gid] + y1[gid]) / 2
        merged_boxes.append({
            "id":              gid + 1,
            "mm":              (float(x0[gid]), float(x1[gid]),
                                float(y0[gid]), float(y1[gid])),
            "size":            int(size[gid]),
            "center_mm":       (float(cx), float(cy)),
            "pixels":          None,
            "n_original_rois": int(n_orig[gid]),
        })
    return merged_boxes


def add_margin(boxes, coarse_px_mm, fine_px_mm, n_fine=5):
    """
    Expand each ROI bounding box by a safety margin on all four sides.

    Margin = half coarse pixel + n_fine * fine pixel
    (covers ±half-pixel centering uncertainty + positioning error buffer)

    Parameters
    ----------
    boxes        : list of ROI dicts (mm field will be expanded in-place copy)
    coarse_px_mm : coarse scan pixel size [mm]  (e.g. 0.25 for 250 um)
    fine_px_mm   : fine scan pixel size [mm]    (e.g. 0.025 for 25 um)
    n_fine       : number of extra fine pixels to add (default 5 = 125 um)

    Returns
    -------
    list of new ROI dicts with expanded 'mm' and added 'margin_mm' key
    """
    margin = coarse_px_mm / 2.0 + n_fine * fine_px_mm
    result = []
    for b in boxes:
        x0, x1, y0, y1 = b["mm"]
        x0m, x1m = min(x0, x1) - margin, max(x0, x1) + margin
        y0m, y1m = min(y0, y1) - margin, max(y0, y1) + margin
        new_b = dict(b)
        new_b["mm"]        = (x0m, x1m, y0m, y1m)
        new_b["margin_mm"] = margin
        cx = (x0m + x1m) / 2
        cy = (y0m + y1m) / 2
        new_b["center_mm"] = (cx, cy)
        result.append(new_b)
    return result


def nearest_neighbor_tour(centers, start_idx=0):
    """
    Build a tour by always visiting the nearest unvisited ROI next.
    Vectorized: O(N) distance evaluations per step using numpy.
    Returns a list of indices in visit order.
    """
    centers = np.asarray(centers, dtype=float)
    n       = len(centers)
    if n == 0:
        return []
    visited = np.zeros(n, dtype=bool)
    tour    = [int(start_idx)]
    visited[start_idx] = True
    for _ in range(n - 1):
        current = tour[-1]
        dists   = np.linalg.norm(centers - centers[current], axis=1)
        dists[visited] = np.inf
        next_idx = int(np.argmin(dists))
        tour.append(next_idx)
        visited[next_idx] = True
    return tour


def two_opt_improve(centers, tour):
    """
    2-opt for an OPEN path: try reversing every sub-segment [i+1..j], keep if
    the path gets shorter. Runs until no improving swap exists.

    The scan path is open (the scanner does not return to the start), so the
    cost model must NOT include a closing edge tour[-1] -> tour[0]: optimizing
    the closed cycle can accept swaps that shorten the phantom closing edge
    while lengthening the path actually traveled. We model this by treating
    the successor of the last node as a virtual depot at zero distance from
    everything (d = 0), which makes cycle length == open-path length.

    Vectorized: each iteration builds an (N, N) matrix of swap savings using
    numpy broadcasting, then applies the single best improving swap. ~50-100x
    faster than the pure-Python double loop for N > 30.
    """
    centers = np.asarray(centers, dtype=float)
    best    = list(tour)
    n       = len(best)
    if n < 3:
        return best

    while True:
        idx = np.array(best, dtype=int)
        pts = centers[idx]                      # (n, 2) along the tour
        # edges a=tour[i], b=tour[i+1], c=tour[j], d=tour[j+1]
        a = pts                                  # (n, 2)
        b = np.roll(pts, -1, axis=0)             # (n, 2), b[i] = tour[(i+1)%n]
        # pairwise distances |a_i - a_j| and |b_i - b_j|
        diff_aa = a[:, None, :] - a[None, :, :]
        diff_bb = b[:, None, :] - b[None, :, :]
        d_ac = np.linalg.norm(diff_aa, axis=2)   # (n, n) — distance(a_i, a_j)
        d_bd = np.linalg.norm(diff_bb, axis=2)   # (n, n) — distance(b_i, b_j)
        # old length contributed by edges i and j
        d_ab = np.linalg.norm(a - b, axis=1)     # (n,)
        # open path: the "edge" after the last node goes to the virtual depot
        # (zero length), so reversing a tail segment is charged correctly.
        d_ab[-1] = 0.0
        d_bd[:, -1] = 0.0
        d_bd[-1, :] = 0.0
        old = d_ab[:, None] + d_ab[None, :]      # (n, n)
        new = d_ac + d_bd
        delta = new - old                        # negative = improvement

        # only consider j >= i + 2 (so segment reverse is non-trivial)
        ii, jj = np.indices(delta.shape)
        valid = jj >= ii + 2
        delta_masked = np.where(valid, delta, np.inf)

        flat = int(np.argmin(delta_masked))
        i, j = int(flat // n), int(flat % n)
        if delta_masked[i, j] >= -1e-10:
            break
        best[i + 1:j + 1] = best[i + 1:j + 1][::-1]

    return best


def or_opt_improve(centers, tour, block_sizes=(1, 2, 3)):
    """
    Or-opt improvement for an OPEN path: for each block of k consecutive nodes,
    try every possible reinsertion position (forward and reversed) and keep
    moves that reduce total Euclidean path length.
    Runs passes over all block sizes until no improvement is found.

    Complement to 2-opt: finds improvements that do not require segment
    reversal (e.g. relocating a node to a better spot without uncrossing).

    Open-path semantics: no wrap-around edge exists between tour[-1] and
    tour[0], and position 0 is never moved — callers (script 02) anchor the
    start on the ROI nearest the scanner's starting position, and the cost of
    the first hop is charged from that fixed start.

    Parameters
    ----------
    centers     : (N, 2) array of (x, y) centers [mm]
    tour        : list of node indices (visit order)
    block_sizes : tuple of block lengths to try (default 1, 2, 3)

    Returns
    -------
    improved tour as a list of indices
    """
    centers = np.asarray(centers, dtype=float)
    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    def dist(a, b):
        return dmat[a, b]

    best = list(tour)
    n    = len(best)
    if n < 3:
        return best

    for k in block_sizes:
        if n - k < 2:                 # need at least the anchor + one other node
            continue
        improved = True
        while improved:
            improved = False
            # position 0 is the anchored start: blocks begin at i >= 1
            for i in range(1, n - k + 1):
                block  = best[i:i + k]
                prev_i = best[i - 1]
                next_i = best[i + k] if i + k < n else None

                # gain from removing the block (block at the tail: no next edge)
                if next_i is None:
                    remove_gain = dist(prev_i, block[0])
                else:
                    remove_gain = (dist(prev_i, block[0])
                                   + dist(block[-1], next_i)
                                   - dist(prev_i, next_i))

                rest = best[:i] + best[i + k:]
                m = len(rest)
                # insert after rest[j] (j = m-1 appends at the open end)
                for j in range(m):
                    if rest[j] == prev_i:
                        continue      # reinserting at the original spot: no-op
                    node_j  = rest[j]
                    node_j1 = rest[j + 1] if j + 1 < m else None

                    for seg in (block, block[::-1]):
                        if node_j1 is None:
                            insert_cost = dist(node_j, seg[0])
                        else:
                            insert_cost = (dist(node_j, seg[0])
                                           + dist(seg[-1], node_j1)
                                           - dist(node_j, node_j1))
                        if remove_gain - insert_cost > 1e-10:
                            best     = rest[:j + 1] + seg + rest[j + 1:]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return best


def summarize_rois(boxes, dx, dy):
    """Print a readable summary of detected ROIs."""
    print(f"\n{'='*55}")
    print(f"  {len(boxes)} ROI(s) detected")
    print(f"{'='*55}")
    for b in boxes:
        x0, x1, y0, y1 = b["mm"]
        w = abs(x1 - x0) + dx
        h = abs(y1 - y0) + dy
        print(f"  ROI {b['id']:2d} | {b['size']:5d} px | "
              f"x=[{x0:.3f}, {x1:.3f}] y=[{y0:.3f}, {y1:.3f}] mm | "
              f"{w:.3f} x {h:.3f} mm")
    print(f"{'='*55}\n")
