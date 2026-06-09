"""
Dwell allocation — single source of truth (extracted from script 07).

Given a signal map (the coarse interest estimate) and a strategy, compute a
per-pixel dwell time. Shared by:
  - script 07 (temporal): called with mask=None  -> quantiles over the whole map
  - script 10 (production): called with a spatial mask -> quantiles pooled over
    the in-mask pixels only (so the dynamic range used for scaling reflects the
    sample area, not the empty surroundings).
"""

import numpy as np


def allocate_dwell(signal, strategy, dwell_low, dwell_high,
                   threshold=None, mask=None):
    """
    Compute a per-pixel dwell array (ms) for a given allocation strategy.

    Parameters
    ----------
    signal      : float array      interest estimate (e.g. coarse composite)
    strategy    : str              'binary' | 'linear' | 'log' | 'sqrt'
    dwell_low   : float            min dwell [ms] (background)
    dwell_high  : float            max dwell [ms] (bright pixels)
    threshold   : float, optional  binary cut on `signal` (default = Q75 of pool)
    mask        : bool array, opt  if given, quantiles are computed over
                                   signal[mask] only; values are still returned
                                   for every pixel (caller masks afterwards).

    Returns
    -------
    (dwell_map, thr) : dwell array (same shape as `signal`) and the binary
                       threshold used (None for linear/log).
    """
    x = signal.astype(float)
    pool = x[mask] if mask is not None else x.ravel()

    if pool.size == 0:
        # Empty mask (nothing to scan) -> degrade gracefully instead of letting
        # np.quantile raise on an empty array.
        return np.full_like(x, dwell_low), None

    if strategy == "binary":
        thr = threshold if threshold is not None else float(np.quantile(pool, 0.75))
        return np.where(x > thr, dwell_high, dwell_low), thr

    if strategy == "linear":
        lo, hi = np.quantile(pool, [0.05, 0.95])
        if hi <= lo:
            return np.full_like(x, (dwell_low + dwell_high) / 2), None
        normed = np.clip((x - lo) / (hi - lo), 0, 1)
        return dwell_low + normed * (dwell_high - dwell_low), None

    if strategy == "log":
        eps = 1.0
        logx = np.log(np.maximum(x, eps))
        pool_log = logx[mask] if mask is not None else logx.ravel()
        lo, hi = np.quantile(pool_log, [0.05, 0.95])
        if hi <= lo:
            return np.full_like(x, (dwell_low + dwell_high) / 2), None
        normed = np.clip((logx - lo) / (hi - lo), 0, 1)
        return dwell_low + normed * (dwell_high - dwell_low), None

    if strategy == "sqrt":
        # MSE-optimal allocation: minimizing sum r_i/t_i under a time budget gives
        # t_i PROPORTIONAL to sqrt(r_i) (water-filling — see utils.quality).
        # We realize the proportionality directly: t_i = dwell_high * sqrt(x/x_ref),
        # capped at dwell_high and floored at dwell_low. x_ref = 95th pct of the
        # pool (robust to outliers). Pair with --match-budget for equal-time scale;
        # the proportional shape is what makes this the optimum, not an affine remap.
        sq = np.sqrt(np.maximum(x, 0.0))
        x_ref = np.quantile(np.maximum(pool, 0.0), 0.95)
        if x_ref <= 0:
            return np.full_like(x, (dwell_low + dwell_high) / 2), None
        dwell = dwell_high * sq / np.sqrt(x_ref)
        return np.clip(dwell, dwell_low, dwell_high), None

    raise ValueError(f"Unknown strategy '{strategy}'")
