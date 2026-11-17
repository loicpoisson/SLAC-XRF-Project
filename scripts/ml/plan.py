"""
From a predicted fine-rate map to a scan plan the benchmark can score.

A plan is (mask, weight, pred) on the fine grid - see utils/bench.py. The
learned and the non-learned planners below differ ONLY in the rate map they
are given, so a comparison isolates what the model's prediction is worth:

    where to scan : the `area` fraction of pixels with the highest rate
    how long      : dwell ~ sqrt(rate + mean)
    the rest      : the rate map itself, or the graph-harmonic interpolation
"""

import numpy as np
from scipy import ndimage

from utils.bench import T_REF
from utils.evaluation import sqrt_weight
from utils.reconstruction import coarse_rate, fill_bilinear, fill_harmonic


def top_area_mask(score, area, valid=None):
    """The `area` fraction of pixels with the highest score."""
    pool = score[valid] if valid is not None else score.ravel()
    return score > np.quantile(pool, 1.0 - area)


def plan_from_rate(case, rate, area, fill="rate", level=100):
    """(mask, weight, pred) from a fine rate map [counts/ms].

    fill='rate'     : unscanned pixels take the rate map's own value
    fill='harmonic' : graph-harmonic interpolation of the coarse points lying
                      outside the scanned area (the current pipeline's fill)
    """
    mask = top_area_mask(rate, area, case.edge_ok & case.full)
    weight = sqrt_weight(rate, mask)
    if fill == "rate":
        pred = rate
    else:
        key = f"c{level}"
        crate = coarse_rate(case.coarse(level), case.loo_scale(level), case.dwell[key])
        pred = fill_harmonic(crate, case.data[key], case.F, mask,
                             np.zeros(case.shape), w_fine=0.0)
    return mask, weight, pred


# ── rate maps ──────────────────────────────────────────────────────────────────

def rate_bilinear(case, level=100):
    """Non-learned control: bilinear interpolation of the coarse target map."""
    key = f"c{level}"
    crate = coarse_rate(case.coarse(level), case.loo_scale(level), case.dwell[key])
    return fill_bilinear(crate, case.data[key], case.F)


def rate_smoothed(case, sigma=1.0, level=100):
    """Non-learned control: the coarse target map smoothed by a Gaussian of
    `sigma` coarse pixels, then interpolated. What 'just denoise the coarse
    scan' buys - a learned model has to beat this, not the raw threshold."""
    key = f"c{level}"
    crate = coarse_rate(case.coarse(level), case.loo_scale(level), case.dwell[key])
    return fill_bilinear(ndimage.gaussian_filter(crate, sigma), case.data[key], case.F)


def rate_learned(case, ex, model):
    """The RateGNN's prediction, on the count scale measured on OTHER samples
    (the model itself never saw this sample either)."""
    from .train import predict
    return predict(model, ex) * case.loo_scale(100) / T_REF
