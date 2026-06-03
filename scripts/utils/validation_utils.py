"""
Validation helpers: project a coarse-grid mask onto a fine-scan grid and
compute signal-capture / speedup metrics.

Used by scripts 03, 04, 06 (Pareto sweep), 07 (temporal adaptive).
"""

import numpy as np
from scipy import ndimage

from .hdf5_reader import load_xrf, get_composite_map
from .roi_utils import travel_time, nearest_neighbor_tour, two_opt_improve


def _nearest_idx(arr, vals):
    """Index of nearest entry in `arr` for each value in `vals`.
    Handles ascending or descending `arr` order."""
    order = np.argsort(arr)
    arr_s = arr[order]
    pos   = np.clip(np.searchsorted(arr_s, vals), 0, len(arr_s) - 1)
    left  = np.clip(pos - 1, 0, len(arr_s) - 1)
    choose_left = np.abs(arr_s[left] - vals) < np.abs(arr_s[pos] - vals)
    return order[np.where(choose_left, left, pos)]


def project_mask(mask_src, x_src, y_src, x_dst, y_dst):
    """Project a binary mask from one rectilinear grid to another (nearest)."""
    ix = _nearest_idx(x_src, x_dst)
    iy = _nearest_idx(y_src, y_dst)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    return mask_src[IY, IX]


def project_coarse_to_fine(coarse_comp, coarse_data, fine_data):
    """Replicate each coarse pixel value to its corresponding fine pixels."""
    ix = _nearest_idx(coarse_data["xdata"], fine_data["xdata"])
    iy = _nearest_idx(coarse_data["ydata"], fine_data["ydata"])
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    return coarse_comp[IY, IX]


def coarse_blockmean(fine_comp, coarse_data, fine_data):
    """
    Block-average the FINE composite onto the coarse grid, then expand back to
    the fine grid. The result is the idealized coarse-resolution view of the
    truth: for each fine pixel, the mean fine value over the coarse cell that
    contains it.

    Why not the raw coarse scan? A coarse pixel integrates more signal than a
    fine pixel, so the two maps live on different intensity scales (~3x here);
    comparing them directly inflates errors. The block-mean is same-scale by
    construction, so (blockmean - fine)^2 isolates exactly the sub-coarse-pixel
    detail lost by NOT scanning finely — the correct fill for the MSE bias term.
    """
    ix = _nearest_idx(coarse_data["xdata"], fine_data["xdata"])   # fine col -> coarse col
    iy = _nearest_idx(coarse_data["ydata"], fine_data["ydata"])   # fine row -> coarse row
    nx_c, ny_c = len(coarse_data["xdata"]), len(coarse_data["ydata"])
    cell = (iy[:, None] * nx_c + ix[None, :]).ravel()             # coarse-cell id per fine px
    n_cells = nx_c * ny_c
    sums = np.bincount(cell, weights=fine_comp.ravel(), minlength=n_cells)
    cnts = np.bincount(cell, minlength=n_cells)
    means = sums / np.maximum(cnts, 1)
    return means[cell].reshape(fine_comp.shape)


def compute_validation_metrics(mask_fine, fine_comp, fine_dwell_ms=10.0,
                                coarse_overhead_ms=0.0,
                                travel_overhead_ms=0.0):
    """
    Compute the standard set of metrics for a fine-grid binary mask.

    Parameters
    ----------
    mask_fine          : bool array, fine-grid mask of pixels we WOULD scan
    fine_comp          : float array, fine-scan composite (ground truth)
    fine_dwell_ms      : per-pixel dwell time at fine resolution [ms]
    coarse_overhead_ms : time already spent at coarser levels [ms]

    Returns
    -------
    dict with keys
        n_in, n_tot, area_frac, signal_total, signal_in, signal_captured,
        raster_ms, adaptive_ms, speedup, efficiency
    """
    n_in  = int(mask_fine.sum())
    n_tot = mask_fine.size
    area_frac = n_in / n_tot if n_tot else 0.0

    signal_total    = float(fine_comp.sum())
    signal_in       = float(fine_comp[mask_fine].sum())
    signal_captured = signal_in / signal_total if signal_total > 0 else 0.0

    raster_ms     = n_tot * fine_dwell_ms
    adaptive_dwell_ms = n_in * fine_dwell_ms + coarse_overhead_ms
    adaptive_ms   = adaptive_dwell_ms + travel_overhead_ms
    speedup       = raster_ms / adaptive_ms if adaptive_ms > 0 else np.inf
    speedup_no_travel = raster_ms / adaptive_dwell_ms if adaptive_dwell_ms > 0 else np.inf
    efficiency    = signal_captured / area_frac if area_frac > 0 else np.inf

    return {
        "n_in":               n_in,
        "n_tot":              n_tot,
        "area_frac":          area_frac,
        "signal_total":       signal_total,
        "signal_in":          signal_in,
        "signal_captured":    signal_captured,
        "raster_ms":          raster_ms,
        "adaptive_dwell_ms":  adaptive_dwell_ms,
        "travel_ms":          travel_overhead_ms,
        "adaptive_ms":        adaptive_ms,
        "speedup":            speedup,
        "speedup_no_travel":  speedup_no_travel,
        "efficiency":         efficiency,
    }


def estimate_travel_overhead(mask, x_grid, y_grid, setup_ms=500.0):
    """
    Estimate the inter-region travel + setup overhead for an adaptive scan.

    A binary mask is split into connected components (each is a contiguous
    scan region). Within a region the scanner does serpentine raster (the
    pixel-to-pixel travel is dominated by the dwell time, so we count it as
    zero overhead). Between regions, the scanner makes a long jump using
    the trapezoidal velocity profile, plus a fixed `setup_ms` overhead per
    region (positioning, triggering, etc.).

    Parameters
    ----------
    mask     : 2D bool array (fine-grid mask of pixels to scan)
    x_grid   : 1D x-coordinates [mm], length = mask.shape[1]
    y_grid   : 1D y-coordinates [mm], length = mask.shape[0]
    setup_ms : per-region overhead [ms]

    Returns
    -------
    dict with: n_regions, travel_ms, setup_total_ms, overhead_ms (= travel+setup)
    """
    labeled, n = ndimage.label(mask)
    if n == 0:
        return {"n_regions": 0, "travel_ms": 0.0, "setup_total_ms": 0.0,
                "overhead_ms": 0.0}
    if n == 1:
        return {"n_regions": 1, "travel_ms": 0.0, "setup_total_ms": setup_ms,
                "overhead_ms": setup_ms}

    centers = np.zeros((n, 2))
    for i in range(1, n + 1):
        rows, cols = np.where(labeled == i)
        centers[i - 1] = [x_grid[cols].mean(), y_grid[rows].mean()]

    tour = nearest_neighbor_tour(centers, start_idx=0)
    if len(tour) > 2:
        tour = two_opt_improve(centers, tour)

    travel_ms = 0.0
    for i in range(len(tour) - 1):
        d = np.linalg.norm(centers[tour[i]] - centers[tour[i + 1]])
        travel_ms += travel_time(d) * 1000.0

    setup_total = n * setup_ms
    return {"n_regions": n, "travel_ms": travel_ms,
            "setup_total_ms": setup_total,
            "overhead_ms": travel_ms + setup_total}


def load_and_compose(path, channels=None):
    """Convenience wrapper: load + get_composite_map."""
    data = load_xrf(path)
    comp, used, excluded = get_composite_map(data, channels)
    return data, comp, used, excluded
