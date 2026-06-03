"""Tests for the Poisson MSE / SNR metric (utils.quality)."""

import numpy as np

from utils.quality import poisson_mse, poisson_mse_montecarlo, snr_map, mean_snr


def test_uniform_dwell_gives_ratio_one():
    # Full mask, dwell == t_ref everywhere -> adaptive IS the raster -> ratio 1.
    rng = np.random.default_rng(0)
    sig = rng.uniform(1, 100, size=(20, 20))
    t_ref = 10.0
    dwell = np.full_like(sig, t_ref)
    mask = np.ones_like(sig, dtype=bool)
    res = poisson_mse(sig, dwell, mask, t_ref=t_ref, predict="zero")
    assert res["bias_term"] == 0.0
    assert abs(res["ratio"] - 1.0) < 1e-9


def test_sqrt_is_variance_optimal():
    # Cauchy-Schwarz: t_i proportional to sqrt(r_i) minimizes sum r_i/t_i at a
    # fixed time budget. This is the theoretical basis of the 'sqrt' strategy.
    rng = np.random.default_rng(1)
    r = rng.uniform(1, 1000, size=500)
    T = 500 * 10.0
    t_uniform = np.full_like(r, T / len(r))
    t_sqrt = T * np.sqrt(r) / np.sqrt(r).sum()
    var_uniform = (r / t_uniform).sum()
    var_sqrt = (r / t_sqrt).sum()
    assert var_sqrt < var_uniform                 # strictly better for non-constant r
    # and equals the closed form (sum sqrt(r))^2 / T
    assert np.isclose(var_sqrt, np.sqrt(r).sum() ** 2 / T)


def test_analytic_matches_montecarlo():
    rng = np.random.default_rng(2)
    sig = rng.uniform(5, 500, size=(40, 40))
    t_ref = 10.0
    dwell = np.clip(sig / sig.max() * 50, 1, 50)
    mask = np.ones_like(sig, dtype=bool)
    a = poisson_mse(sig, dwell, mask, t_ref=t_ref, predict="zero")
    m = poisson_mse_montecarlo(sig, dwell, mask, t_ref=t_ref, predict="zero",
                               n_draws=400, seed=0)
    assert abs(a["ratio"] - m["ratio"]) / a["ratio"] < 0.10   # within MC noise


def test_bias_term_grows_when_bright_pixels_unscanned():
    sig = np.zeros((10, 10))
    sig[5, 5] = 1000.0                            # one bright pixel
    t_ref = 10.0
    dwell = np.full_like(sig, t_ref)
    scan_it = np.ones_like(sig, dtype=bool)
    skip_it = scan_it.copy(); skip_it[5, 5] = False
    mse_scanned = poisson_mse(sig, dwell, scan_it, t_ref=t_ref, predict="zero")
    mse_skipped = poisson_mse(sig, dwell, skip_it, t_ref=t_ref, predict="zero")
    assert mse_skipped["bias_term"] > mse_scanned["bias_term"]
    assert mse_skipped["mse_adaptive"] > mse_scanned["mse_adaptive"]


def test_snr_map_formula():
    sig = np.array([[0.0, 4.0], [9.0, 16.0]])
    dwell = np.full_like(sig, 10.0)
    s = snr_map(sig, dwell, t_ref=10.0)           # sqrt(sig * dwell / t_ref) = sqrt(sig)
    assert np.allclose(s, np.sqrt(sig))


def test_mean_snr_increases_with_dwell():
    sig = np.full((8, 8), 100.0)
    lo = mean_snr(sig, np.full_like(sig, 10.0))
    hi = mean_snr(sig, np.full_like(sig, 40.0))
    assert hi > lo
