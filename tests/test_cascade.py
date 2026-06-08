"""Tests for the cascade primitives (utils.cascade) and grid helpers."""

import numpy as np
import pytest

from utils.cascade import dwell_ms_from_path, refine_step, available_levels
from utils.validation_utils import (coarse_blockmean, project_mask,
                                     travel_overhead_from_centers,
                                     estimate_travel_overhead)


def test_dwell_ms_from_path():
    assert dwell_ms_from_path("SMW_UA1_P1_250um_10ms_12000_0_001.hdf5") == 10.0
    assert dwell_ms_from_path("SMW_X_25um_25ms_x.hdf5") == 25.0
    assert dwell_ms_from_path("no_token_here.hdf5", default=7.0) == 7.0


def test_refine_step_first_level_shapes():
    comp = np.full((10, 10), 100.0)
    comp[:3, :] = 0.0
    data = {"xdata": np.arange(10.0), "ydata": np.arange(10.0), "dx": 1.0, "dy": 1.0}
    mask, thr, st = refine_step(comp, data, kernel_px=1)
    assert mask.shape == comp.shape
    assert 0.0 <= st["area_frac"] <= 1.0
    assert 0.0 <= st["signal_in_frac"] <= 1.0


def test_refine_step_restricts_to_previous():
    # A refinement step must never select pixels outside the projected prev mask.
    comp = np.full((8, 8), 100.0)
    grid = {"xdata": np.arange(8.0), "ydata": np.arange(8.0), "dx": 1.0, "dy": 1.0}
    prev_mask = np.zeros((8, 8), dtype=bool)
    prev_mask[:, :4] = True                       # only left half allowed
    mask, _, _ = refine_step(comp, grid, prev_mask=prev_mask, prev_data=grid,
                             kernel_px=1)
    assert not mask[:, 4:].any()


def test_coarse_blockmean_same_scale():
    # Block-mean of a fine map onto a coarse grid then back: same intensity scale,
    # and the mean over each coarse cell is preserved.
    fine = np.arange(16.0).reshape(4, 4)
    fine_data = {"xdata": np.arange(4.0), "ydata": np.arange(4.0)}
    coarse_data = {"xdata": np.array([0.5, 2.5]), "ydata": np.array([0.5, 2.5])}  # 2x2 cells
    bm = coarse_blockmean(fine, coarse_data, fine_data)
    assert bm.shape == fine.shape
    # top-left 2x2 block mean of [[0,1],[4,5]] = 2.5
    assert np.isclose(bm[0, 0], 2.5)
    assert np.isclose(bm.mean(), fine.mean())     # global mean preserved


def test_project_mask_roundtrip_same_grid():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    x = np.arange(5.0)
    out = project_mask(m, x, x, x, x)             # same grid -> identity
    assert np.array_equal(out, m)


def test_available_levels_ua1():
    lv = available_levels("UA1_P1")
    if not lv:
        pytest.skip("UA1_P1 data not available")
    assert lv == [250, 100, 50, 25]


def test_travel_overhead_edge_cases():
    assert travel_overhead_from_centers(np.empty((0, 2)))["overhead_ms"] == 0.0
    one = travel_overhead_from_centers(np.array([[0.0, 0.0]]), setup_ms=500.0)
    assert one["travel_ms"] == 0.0 and one["overhead_ms"] == 500.0
    two = travel_overhead_from_centers(np.array([[0.0, 0.0], [10.0, 0.0]]),
                                       setup_ms=500.0)
    assert two["n_regions"] == 2
    assert two["travel_ms"] > 0.0            # a real jump costs time
    assert two["overhead_ms"] > 1000.0       # travel + 2 * setup


def test_estimate_travel_overhead_delegates_to_centers():
    # mask-based and centers-based paths must agree on the same two regions.
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1] = True
    mask[8, 8] = True
    x = np.arange(10.0)
    em = estimate_travel_overhead(mask, x, x, setup_ms=500.0)
    fc = travel_overhead_from_centers(np.array([[1.0, 1.0], [8.0, 8.0]]),
                                      setup_ms=500.0)
    assert em["n_regions"] == fc["n_regions"] == 2
    assert np.isclose(em["travel_ms"], fc["travel_ms"])
