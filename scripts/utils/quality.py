"""
Image-quality metrics under Poisson photon-counting noise (paper-aligned).

XRF detection counts photons: a pixel with true rate r [counts/ms] observed for
a dwell t [ms] yields N ~ Poisson(r*t). The rate estimate rhat = N/t is unbiased
with variance r/t, so:

    SNR_pixel   = sqrt(r * t)
    MSE_scanned = Var(rhat) = r / t                  (unbiased -> MSE = variance)
    MSE_unscanned = (pred - r)^2                     (pure bias: we predict `pred`)

Total reconstruction MSE for an adaptive scan (mask = scanned pixels):

    MSE_adaptive = (1/P) [ sum_{i in mask} r_i / t_i  +  sum_{i not in mask} (pred_i - r_i)^2 ]
    MSE_raster   = (1/P)   sum_i r_i / t_ref          (everything scanned at t_ref)

We report the ratio MSE_raster / MSE_adaptive at equal total time (>1 = adaptive
wins) — the same figure of merit as Boltz-Webb (2026).

Optimal allocation: minimizing sum r_i/t_i under sum t_i = T (Lagrange / water-
filling) gives t_i proportional to sqrt(r_i). See the 'sqrt' strategy in
utils.dwell.allocate_dwell.

Ground-truth convention: the finest available scan F (counts at t_ref) is taken
as truth, with rate r_i = F_i / t_ref. The ratio is dimensionless, so absolute
detector calibration does not matter.
"""

import numpy as np

_EPS = 1e-9


def _pred_rate(true_signal, t_ref, predict, predict_signal):
    """Predicted rate for UNSCANNED pixels."""
    if predict == "zero":
        return np.zeros_like(true_signal, dtype=float)
    if predict == "coarse":
        if predict_signal is None:
            raise ValueError("predict='coarse' requires predict_signal "
                             "(coarse composite projected onto the fine grid)")
        return predict_signal.astype(float) / t_ref
    raise ValueError(f"Unknown predict mode '{predict}' (use 'coarse' or 'zero')")


def poisson_mse(true_signal, dwell_map, mask, t_ref=10.0,
                predict="coarse", predict_signal=None):
    """
    Analytic (expected) reconstruction MSE under Poisson noise.

    Parameters
    ----------
    true_signal   : 2D array  ground-truth counts at the reference dwell t_ref
    dwell_map     : 2D array  per-pixel dwell [ms] (only used inside the mask)
    mask          : bool 2D   scanned pixels (True)
    t_ref         : float     reference/raster dwell [ms]
    predict       : 'coarse' | 'zero'  how unscanned pixels are filled in
    predict_signal: 2D array  (for 'coarse') fill counts at t_ref on the fine grid

    Returns
    -------
    dict: mse_raster, mse_adaptive, ratio, var_term, bias_term, area_frac
    """
    r = true_signal.astype(float) / t_ref
    P = r.size

    var_term = np.where(mask, r / np.maximum(dwell_map, _EPS), 0.0).sum()
    pred = _pred_rate(true_signal, t_ref, predict, predict_signal)
    bias_term = np.where(~mask, (pred - r) ** 2, 0.0).sum()

    mse_adaptive = (var_term + bias_term) / P
    mse_raster = (r / t_ref).sum() / P
    return {
        "mse_raster":   mse_raster,
        "mse_adaptive": mse_adaptive,
        "ratio":        mse_raster / mse_adaptive if mse_adaptive > 0 else np.inf,
        "var_term":     var_term / P,
        "bias_term":    bias_term / P,
        "area_frac":    float(mask.mean()),
    }


def poisson_mse_montecarlo(true_signal, dwell_map, mask, t_ref=10.0,
                           predict="coarse", predict_signal=None,
                           n_draws=200, seed=0):
    """
    Empirical reconstruction MSE by simulating Poisson photon counts.
    Matches the papers' Monte-Carlo methodology and gives error bars.

    Returns dict: mse_raster (mean,std), mse_adaptive (mean,std), ratio.
    """
    rng = np.random.default_rng(seed)
    r = true_signal.astype(float) / t_ref
    pred = _pred_rate(true_signal, t_ref, predict, predict_signal)
    t_safe = np.maximum(dwell_map, _EPS)

    adaptive, raster = np.empty(n_draws), np.empty(n_draws)
    for d in range(n_draws):
        # adaptive: scanned pixels measured at t_i, unscanned filled with pred
        N = rng.poisson(np.where(mask, r * t_safe, 0.0))
        rhat = np.where(mask, N / t_safe, pred)
        adaptive[d] = np.mean((rhat - r) ** 2)
        # raster: everything at t_ref
        N_r = rng.poisson(r * t_ref)
        raster[d] = np.mean((N_r / t_ref - r) ** 2)

    ma, mr = adaptive.mean(), raster.mean()
    return {
        "mse_raster":   (mr, raster.std()),
        "mse_adaptive": (ma, adaptive.std()),
        "ratio":        mr / ma if ma > 0 else np.inf,
        "n_draws":      n_draws,
    }


def snr_map(true_signal, dwell_map, t_ref=10.0):
    """Per-pixel SNR = sqrt(r * t) = sqrt(true_signal * dwell / t_ref)."""
    return np.sqrt(np.maximum(true_signal.astype(float) * dwell_map / t_ref, 0.0))


def mean_snr(true_signal, dwell_map, t_ref=10.0, mask=None, weight_by_signal=True):
    """
    Scalar SNR summary. By default signal-weighted (bright pixels matter more),
    which is what the papers care about for quantification.
    """
    snr = snr_map(true_signal, dwell_map, t_ref)
    w = true_signal.astype(float) if weight_by_signal else np.ones_like(snr)
    if mask is not None:
        snr, w = snr[mask], w[mask]
    return float(np.average(snr, weights=w)) if w.sum() > 0 else 0.0
