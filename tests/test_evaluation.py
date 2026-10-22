"""Tests for utils.evaluation (step-1 robust evaluation)."""

import numpy as np
import pytest

from utils.evaluation import (budget_dwell, sqrt_weight, pixel_errors,
                              raster_errors, block_bootstrap_ratio, hot_spots,
                              simulate_tasks)


def test_budget_dwell_spends_exact_budget_within_bounds():
    rng = np.random.default_rng(0)
    w = rng.exponential(1.0, (50, 60)) ** 3          # heavy tail -> clipping
    mask = rng.random((50, 60)) > 0.3
    d = budget_dwell(w, mask, budget_ms=10.0 * mask.sum(), lo=1.0, hi=50.0)
    assert d.sum() == pytest.approx(10.0 * mask.sum(), rel=1e-5)
    assert d[mask].min() >= 1.0 and d[mask].max() <= 50.0
    assert np.all(d[~mask] == 0)


def test_budget_dwell_uniform_weight_is_raster():
    mask = np.ones((4, 5), bool)
    d = budget_dwell(np.ones((4, 5)), mask, budget_ms=200.0)
    np.testing.assert_allclose(d, 10.0)


def test_sqrt_weight_regularises_zero_estimate():
    est = np.array([[0.0, 4.0]])
    w = sqrt_weight(est, np.ones_like(est, bool), reg=1.0)
    np.testing.assert_allclose(w, np.sqrt([[2.0, 6.0]]))


def test_pixel_errors_debias_truth_noise():
    # Truth rate r measured over t_truth; unscanned pixels predicted exactly.
    # E[(r - r_hat)^2] = Var(r_hat) -> debiased error averages to 0.
    rng = np.random.default_rng(1)
    r, t_truth = np.full((200, 200), 3.0), 25.0
    r_hat = rng.poisson(r * t_truth) / t_truth
    mask = np.zeros_like(r, bool)
    e = pixel_errors(r_hat, r_hat / t_truth, np.zeros_like(r), mask, pred_rate=r)
    assert abs(e.mean()) < 0.01 * (r / t_truth).mean() * 10


def test_full_frame_uniform_dwell_matches_raster():
    r = np.random.default_rng(2).exponential(2.0, (30, 30))
    full = np.ones_like(r, bool)
    e_ad = pixel_errors(r, r / 10.0, np.full_like(r, 10.0), full, r)
    np.testing.assert_allclose(e_ad, raster_errors(r, 10.0))


def test_block_bootstrap_contains_point_ratio():
    rng = np.random.default_rng(3)
    er = rng.exponential(1.0, (120, 120))
    ea = er / 1.3 * rng.uniform(0.9, 1.1, er.shape)
    lo, hi = block_bootstrap_ratio(er, ea, np.ones_like(er, bool), block=20, n_boot=200)
    point = er.sum() / ea.sum()
    assert lo < point < hi and lo > 1.2 and hi < 1.4


def test_hot_spots_and_perfect_reconstruction():
    r = np.full((60, 60), 1.0)
    r[10:15, 10:15] = 50.0
    r[40:44, 30:36] = 80.0
    valid = np.ones_like(r, bool)
    lab, n = hot_spots(r, valid, k=3.0)
    assert n == 2
    # exact fill everywhere, nothing scanned -> no quantification error,
    # every hot spot re-detected
    res = simulate_tasks(r, np.zeros_like(r), np.zeros_like(r, bool), r, valid,
                         lab, n, n_draws=2)
    assert res["quant_err"] == pytest.approx(0.0)
    assert res["recall"] == 1.0 and res["precision"] == 1.0


def test_register_rows_recovers_drifting_shift():
    from utils.evaluation import register_rows
    rng = np.random.default_rng(4)
    ref = rng.exponential(1.0, (160, 200))
    ref = np.maximum(ref, 0) + 20 * (rng.random(ref.shape) > 0.98)
    moving = np.empty_like(ref)
    true = np.repeat([0, 2, 4, -3], 40)                 # drift by row band
    for y, s in enumerate(true):
        moving[y] = np.roll(ref[y], -s)
    out, valid, shifts = register_rows(moving, ref, band=40)
    np.testing.assert_array_equal(shifts[[20, 60, 100, 140]], [0, 2, 4, -3])
    assert np.allclose(out[valid & (np.arange(160)[:, None] % 40 == 20)],
                       ref[valid & (np.arange(160)[:, None] % 40 == 20)])


def test_sqrt_errors_uniform_dwell_is_optimal_in_sqrt_domain():
    from utils.evaluation import sqrt_errors
    r = np.random.default_rng(5).exponential(3.0, (40, 40))
    full = np.ones_like(r, bool)
    uni = sqrt_errors(r, 25.0, np.full_like(r, 10.0), full, r).sum()
    w = np.sqrt(r); d = 10.0 * w / w.mean()             # same total time
    assert sqrt_errors(r, 25.0, d, full, r).sum() > uni


def test_top_share():
    from utils.evaluation import top_share
    e = np.ones(1000); e[0] = 1000.0
    assert top_share(e, np.ones(1000, bool), 0.001) == pytest.approx(1000 / 1999)


def test_sqrt_poisson_var_exact_limits():
    from utils.evaluation import sqrt_poisson_var
    v = sqrt_poisson_var(np.array([0.0, 0.5, 1000.0]))
    assert v[0] == 0.0 and v[2] == 0.25
    # direct Monte-Carlo check at low counts, where 1/4 is wrong
    n = np.random.default_rng(6).poisson(0.5, 2_000_000)
    assert v[1] == pytest.approx(np.sqrt(n).var(), rel=0.01)
    assert abs(v[1] - 0.25) > 0.05          # the 1/(4t) shortcut is off here
