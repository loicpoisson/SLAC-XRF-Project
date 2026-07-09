"""Regression tests for the bugs fixed in the project code review."""

import numpy as np
import pytest

import utils.paths as paths
from utils.roi_utils import (group_rois, remove_border_pixels, threshold_map,
                             two_opt_improve, or_opt_improve,
                             nearest_neighbor_tour)
from utils.cascade import refine_step
from utils.quality import poisson_mse
from utils.validation_utils import estimate_travel_overhead


def _path_length(centers, tour):
    """OPEN path length (no closing edge) — what the scanner actually travels."""
    return sum(np.linalg.norm(centers[tour[i]] - centers[tour[i + 1]])
               for i in range(len(tour) - 1))


# ── group_rois unit consistency ────────────────────────────────────────────────

def test_group_rois_merges_adjacent_boxes_with_coarse_sizes():
    # Two adjacent 2x2-coarse-pixel ROIs on a 0.25mm grid, fine px 0.025mm.
    # The true gap between them is tiny, so merging must win over a 500ms setup.
    # Before the fix, coarse sizes were subtracted from fine-pixel bbox counts,
    # inflating the gap ~100x and suppressing the merge.
    boxes = [
        {"id": 1, "mm": (0.0, 0.25, 0.0, 0.25), "size": 4,
         "center_mm": (0.125, 0.125), "pixels": None},
        {"id": 2, "mm": (0.75, 1.0, 0.0, 0.25), "size": 4,
         "center_mm": (0.875, 0.125), "pixels": None},
    ]
    merged = group_rois(boxes, dwell_ms=10.0, dx=0.025, dy=0.025,
                        size_px_mm=(0.25, 0.25))
    assert len(merged) == 1
    assert merged[0]["n_original_rois"] == 2

    # Without the conversion the same input must NOT merge (documents why
    # size_px_mm is required when sizes come from the coarse grid).
    unmerged = group_rois(boxes, dwell_ms=10.0, dx=0.025, dy=0.025)
    assert len(unmerged) == 2


# ── threshold_map mask parameter ───────────────────────────────────────────────

def test_threshold_map_mask_ignores_zero_filled_background():
    # Footprint covers 25% of the frame: matrix ~100, particles ~5000,
    # everything outside the footprint zero-filled. Statistics must come from
    # the in-mask pixels only, otherwise median/MAD collapse to 0 and the
    # whole footprint is flagged as ROI.
    comp = np.zeros((20, 20))
    mask = np.zeros((20, 20), dtype=bool)
    mask[:10, :10] = True
    comp[mask] = 100.0
    comp[2:4, 2:4] = 5000.0
    roi, thr, _ = threshold_map(comp, method="auto", k=1.0, mask=mask)
    roi &= mask
    assert roi[2:4, 2:4].all()
    assert roi.sum() == 4          # only the particles, not the whole footprint
    assert thr >= 100.0            # matrix level, not the zero-filled void


# ── remove_border_pixels(border=0) ─────────────────────────────────────────────

def test_remove_border_zero_is_noop():
    mask = np.ones((5, 5), dtype=bool)
    out = remove_border_pixels(mask, border=0)
    assert out.all()


# ── open-path tour optimizers ──────────────────────────────────────────────────

def test_two_opt_never_lengthens_open_path():
    rng = np.random.default_rng(7)
    for _ in range(20):
        centers = rng.random((12, 2)) * 10
        tour = nearest_neighbor_tour(centers, start_idx=0)
        improved = two_opt_improve(centers, tour)
        assert sorted(improved) == list(range(len(centers)))
        assert (_path_length(centers, improved)
                <= _path_length(centers, tour) + 1e-9)


def test_two_opt_improves_tail_reversal():
    # Reversing the whole tail [1..8] turns this into the optimal straight
    # path (length 8 instead of 15). The move touches the last position, so
    # the old closed-cycle cost model scored it delta = 0 (the gain was hidden
    # in the phantom closing edge) and refused it; open-path 2-opt must take it.
    centers = np.array([[float(i), 0.0] for i in range(9)])
    tour = [0, 8, 7, 6, 5, 4, 3, 2, 1]
    improved = two_opt_improve(centers, tour)
    assert np.isclose(_path_length(centers, improved), 8.0)


def test_or_opt_keeps_start_anchor_and_never_lengthens():
    rng = np.random.default_rng(11)
    for _ in range(20):
        centers = rng.random((10, 2)) * 10
        tour = nearest_neighbor_tour(centers, start_idx=0)
        improved = or_opt_improve(centers, tour)
        assert improved[0] == tour[0]        # anchored start never relocated
        assert sorted(improved) == list(range(len(centers)))
        assert (_path_length(centers, improved)
                <= _path_length(centers, tour) + 1e-9)


# ── anchored resolution globs ──────────────────────────────────────────────────

def test_find_fine_50um_does_not_match_250um(tmp_path, monkeypatch):
    (tmp_path / "SMW_UA1_P1_250um_10ms_12000_0_001.hdf5").touch()
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    assert paths.find_fine(res_um=50) is None      # '50um' must not match '250um'
    assert paths.find_coarse(res_um=250) is not None

    (tmp_path / "SMW_UA1_P1_50um_10ms_12000_0_001.hdf5").touch()
    found = paths.find_fine(res_um=50)
    assert found is not None and "_50um_" in found.name


# ── cascade time accounting ────────────────────────────────────────────────────

def test_refine_step_reports_scanned_pixels():
    comp = np.full((10, 10), 100.0)
    comp[:5, :] = 0.0
    grid = {"xdata": np.arange(10.0), "ydata": np.arange(10.0),
            "dx": 1.0, "dy": 1.0}
    # First level: the full frame is acquired regardless of the refined mask.
    mask, _, st = refine_step(comp, grid, kernel_px=1)
    assert st["n_scanned"] == comp.size
    assert st["n_in"] <= st["n_scanned"]
    # Later level: only the projection of the previous mask is acquired.
    prev = np.zeros((10, 10), dtype=bool)
    prev[:, :4] = True
    _, _, st2 = refine_step(comp, grid, prev_mask=prev, prev_data=grid,
                            kernel_px=1)
    assert st2["n_scanned"] == int(prev.sum())
    assert st2["n_in"] <= st2["n_scanned"]


# ── poisson_mse input sanitation ───────────────────────────────────────────────

def test_poisson_mse_tolerates_nan_and_negative_pixels():
    truth = np.full((8, 8), 50.0)
    truth[0, 0] = np.nan          # fit artifact inside the mask
    truth[1, 1] = -3.0            # slightly negative fit residual
    dwell = np.full((8, 8), 10.0)
    mask = np.ones((8, 8), dtype=bool)
    res = poisson_mse(truth, dwell, mask, t_ref=10.0, predict="zero")
    assert np.isfinite(res["mse_adaptive"])
    assert np.isfinite(res["ratio"])
    # Negative pixels are clamped to rate 0, never SUBTRACTED from the variance.
    clean = np.full((8, 8), 50.0)
    clean[0, 0] = 0.0
    clean[1, 1] = 0.0
    ref = poisson_mse(clean, dwell, mask, t_ref=10.0, predict="zero")
    assert np.isclose(res["mse_adaptive"], ref["mse_adaptive"])


# ── vectorized travel-overhead centers ─────────────────────────────────────────

def test_estimate_travel_overhead_centers_match_reference_loop():
    rng = np.random.default_rng(5)
    mask = rng.random((40, 40)) > 0.8
    x = np.linspace(0.0, 10.0, 40)
    y = np.linspace(-5.0, 5.0, 40)
    from scipy import ndimage
    labeled, n = ndimage.label(mask)
    if n == 0:
        pytest.skip("random mask came out empty")
    ref = np.zeros((n, 2))
    for i in range(1, n + 1):
        rows, cols = np.where(labeled == i)
        ref[i - 1] = [x[cols].mean(), y[rows].mean()]
    fast = estimate_travel_overhead(mask, x, y, setup_ms=500.0)
    slow = __import__("utils.validation_utils", fromlist=["x"]) \
        .travel_overhead_from_centers(ref, setup_ms=500.0)
    assert fast["n_regions"] == slow["n_regions"] == n
    assert np.isclose(fast["travel_ms"], slow["travel_ms"])
