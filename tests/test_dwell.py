"""Tests for dwell allocation (utils.dwell)."""

import numpy as np

from utils.dwell import allocate_dwell


def test_binary_has_two_levels():
    x = np.array([1., 2, 3, 4, 5, 100])
    d, thr = allocate_dwell(x, "binary", 1.0, 50.0)
    assert set(np.unique(d)).issubset({1.0, 50.0})
    assert thr is not None


def test_sqrt_is_monotonic_and_bounded():
    x = np.linspace(0, 1000, 50)
    d, _ = allocate_dwell(x, "sqrt", 1.0, 50.0)
    assert np.all(np.diff(d) >= -1e-9)             # non-decreasing in signal
    assert d.min() >= 1.0 - 1e-9 and d.max() <= 50.0 + 1e-9


def test_sqrt_proportional_to_sqrt_signal():
    # Below the cap, dwell should track sqrt(signal): dwell(4x) ~= 2 * dwell(x).
    x = np.array([0., 25., 100.])                  # sqrt -> 0, 5, 10
    d, _ = allocate_dwell(x, "sqrt", 0.0, 50.0)
    # ratio d[2]/d[1] should be ~ sqrt(100)/sqrt(25) = 2
    assert np.isclose(d[2] / d[1], 2.0, rtol=0.05)


def test_mask_pooling_lowers_binary_threshold():
    x = np.array([1., 1, 1, 1, 100, 100])
    _, thr_full = allocate_dwell(x, "binary", 1, 50)
    mask = np.array([True, True, True, True, False, False])   # exclude the bright pair
    _, thr_masked = allocate_dwell(x, "binary", 1, 50, mask=mask)
    assert thr_masked < thr_full


def test_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        allocate_dwell(np.arange(5.0), "nope", 1, 50)


def test_empty_mask_does_not_crash():
    # An all-False mask (e.g. a footprint that collapsed to zero pixels) must
    # degrade gracefully instead of raising on an empty quantile pool.
    x = np.arange(12.0).reshape(3, 4)
    empty = np.zeros_like(x, dtype=bool)
    for strat in ("binary", "linear", "log", "sqrt"):
        d, thr = allocate_dwell(x, strat, 1.0, 50.0, mask=empty)
        assert d.shape == x.shape
        assert np.all(d == 1.0)
