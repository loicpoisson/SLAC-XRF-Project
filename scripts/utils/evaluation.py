"""
Robust evaluation of adaptive scan strategies (2026-10, "step 1").

What changes vs utils.quality / script 10 --simulate:

1. Independent truth. The score is computed against a SECOND fine scan of the
   same area (UA1_P1 / UB1_P1 have 25 um scans at 10 ms and 25 ms), never the
   map the decisions or the fill could have seen. Its own Poisson noise is
   removed without bias:  E[(p - r_hat)^2] = (p - r)^2 + Var(r_hat), and
   Var(r_hat) = r / t_truth is estimated by r_hat / t_truth.
2. Real coarse fill. Unscanned fine pixels are filled with the MEASURED coarse
   scan (100 or 250 um), rescaled to the fine intensity scale by a factor
   calibrated on OTHER samples (leave-one-sample-out) - not with the block
   mean of the fine truth, which leaks the answer.
3. Exact budget. Dwell maps are scaled + clipped iteratively until they spend
   exactly (fine raster time - time of the earlier levels).
4. Task metrics next to the Poisson MSE: per-hot-spot quantification error and
   hot-spot detection recall / precision, by Monte-Carlo simulation.
5. Uncertainty: spatial block bootstrap of the MSE ratio (structure sampling)
   and Monte-Carlo spread (photon noise).

Rates are counts per ms per fine pixel throughout.
"""

import numpy as np
from scipy import ndimage

from .roi_utils import threshold_map


# ── budget ─────────────────────────────────────────────────────────────────────

def budget_dwell(weight, mask, budget_ms, lo=1.0, hi=50.0, n_iter=60):
    """Dwell [ms] proportional to `weight` inside `mask`, clipped to [lo, hi],
    summing to `budget_ms` (as closely as the bounds allow). 0 outside."""
    w = np.where(mask, np.maximum(weight, 0.0), 0.0)
    n = int(mask.sum())
    if n == 0 or budget_ms <= 0:
        return np.zeros_like(w, dtype=float)
    budget_ms = float(np.clip(budget_ms, lo * n, hi * n))
    if w.sum() <= 0:
        return np.where(mask, budget_ms / n, 0.0)
    s = budget_ms / w.sum()
    d = w
    for _ in range(n_iter):
        d = np.where(mask, np.clip(w * s, lo, hi), 0.0)
        tot = d.sum()
        if abs(tot - budget_ms) <= 1e-6 * budget_ms:
            break
        s *= budget_ms / tot
    return d


def sqrt_weight(estimate, mask, reg=1.0):
    """sqrt(estimate + reg * mean) - the regularised Poisson-MSE optimum."""
    e = np.maximum(estimate, 0.0)
    m = e[mask].mean() if mask.any() else 0.0
    return np.sqrt(e + reg * m)


# ── coarse -> fine calibration ─────────────────────────────────────────────────

def coarse_to_fine_scale(coarse_on_fine, fine, valid=None):
    """Scalar s with mean(fine) = s * mean(coarse projected on the fine grid)."""
    v = np.ones(fine.shape, bool) if valid is None else valid
    c = float(coarse_on_fine[v].mean())
    return float(fine[v].mean()) / c if c > 0 else 1.0


# ── scoring against an independent truth ───────────────────────────────────────

def pixel_errors(truth_rate, truth_var, dwell, mask, pred_rate, pred_corr=0.0):
    """Expected squared error per pixel [rate^2], debiased for truth noise.
    Scanned: Var(N/t) = r / t (r estimated by the unbiased truth). This is an
    idealised Poisson measurement of the truth, for raster and adaptive alike.
    Unscanned: (pred - r)^2, estimated by (pred - r_hat)^2 - Var(r_hat).
    pred_corr: if pred is itself an average of n truth pixels (block mean),
    pass 1/n - its noise is then correlated with r_hat and only
    Var(r_hat) * (1 - 1/n) must be removed."""
    r = truth_rate
    scanned = np.where(mask, np.maximum(r, 0.0) / np.maximum(dwell, 1e-9), 0.0)
    unscanned = np.where(mask, 0.0,
                         (pred_rate - r) ** 2 - truth_var * (1.0 - pred_corr))
    return scanned + unscanned


def raster_errors(truth_rate, t_ref):
    return np.maximum(truth_rate, 0.0) / t_ref


def block_bootstrap_ratio(err_raster, err_adapt, valid, block=40, n_boot=300,
                          seed=0):
    """95% interval of sum(err_raster)/sum(err_adapt) resampling square blocks."""
    ny, nx = err_raster.shape
    by, bx = np.arange(ny) // block, np.arange(nx) // block
    bid = (by[:, None] * (bx.max() + 1) + bx[None, :])[valid]
    nb = bid.max() + 1
    sr = np.bincount(bid, weights=err_raster[valid], minlength=nb)
    sa = np.bincount(bid, weights=err_adapt[valid], minlength=nb)
    present = np.bincount(bid, minlength=nb) > 0
    sr, sa = sr[present], sa[present]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sr), size=(n_boot, len(sr)))
    ratios = sr[idx].sum(1) / np.maximum(sa[idx].sum(1), 1e-300)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


# ── hot spots: quantification + detection ──────────────────────────────────────

def spot_threshold(rate_map, valid, k=3.0, smooth=3):
    """MAD threshold (median + k MAD) of the smoothed map inside `valid`."""
    sm = ndimage.uniform_filter(rate_map, smooth)
    return float(threshold_map(sm, method="mad", k=k, mask=valid)[1])


def hot_spots(rate_map, valid, k=3.0, smooth=3, min_px=4, thresh=None):
    """Labelled hot spots on a smoothed map, inside `valid`: above `thresh`,
    or above the map's own MAD threshold when `thresh` is None."""
    sm = ndimage.uniform_filter(rate_map, smooth)
    if thresh is None:
        thresh = spot_threshold(rate_map, valid, k, smooth)
    m = (sm > thresh) & valid
    lab, n = ndimage.label(m)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_px
        keep[0] = False
        lab = np.where(keep[lab], lab, 0)
        lab, n = ndimage.label(lab > 0)
    return lab, n


def _quant_error(rec, truth, lab, n):
    """Median |sum(rec) - sum(truth)| / sum(truth) over hot spots."""
    if n == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    s_rec = ndimage.sum(rec, lab, idx)
    s_tru = ndimage.sum(truth, lab, idx)
    ok = s_tru > 0
    return float(np.median(np.abs(s_rec[ok] - s_tru[ok]) / s_tru[ok])) if ok.any() else np.nan


def _detection(rec, lab_true, n_true, valid, thresh):
    """Recall / precision of hot spots re-detected on a reconstructed map at
    the TRUTH's threshold. (Re-thresholding every reconstruction on its own
    MAD moved the threshold with the fill and inflated recall at the cost of
    precision - a metric artefact, not recovered spots.)"""
    lab_rec, n_rec = hot_spots(rec, valid, thresh=thresh)
    if n_true == 0:
        return np.nan, np.nan
    hit_true = np.unique(lab_true[(lab_true > 0) & (lab_rec > 0)])
    recall = len(hit_true) / n_true
    if n_rec == 0:
        return recall, np.nan
    hit_rec = np.unique(lab_rec[(lab_rec > 0) & (lab_true > 0)])
    return recall, len(hit_rec) / n_rec


def simulate_tasks(truth_rate, dwell, mask, pred_rate, valid, lab, n_spots,
                   k=3.0, n_draws=10, seed=0, thresh=None):
    """Monte-Carlo: draw counts N ~ Poisson(r t) on scanned pixels, fill the
    rest with pred, then score quantification and detection on the result.
    Also returns the spread of the realised MSE (photon-noise error bar)."""
    rng = np.random.default_rng(seed)
    r = np.maximum(truth_rate, 0.0)
    t = np.maximum(dwell, 1e-9)
    if thresh is None:
        thresh = spot_threshold(r, valid, k)
    # true spots lying ENTIRELY outside the scanned area, with a margin equal
    # to the smoothing radius so that no measured pixel enters their window:
    # can the fill alone find them?
    skipped = np.zeros(n_spots + 1, bool)
    if n_spots and not mask.all():
        near = ndimage.binary_dilation(mask, iterations=1)
        frac = ndimage.mean((~near).astype(float), lab, np.arange(1, n_spots + 1))
        skipped[1:] = np.asarray(frac) >= 1.0
    q, rec_, prec_, mse, rsk = [], [], [], [], []
    for _ in range(n_draws):
        N = rng.poisson(np.where(mask, r * t, 0.0))
        rec = np.where(mask, N / t, pred_rate)
        q.append(_quant_error(rec, r, lab, n_spots))
        a, b = _detection(rec, lab, n_spots, valid, thresh)
        rec_.append(a); prec_.append(b)
        if skipped.any():
            lab_rec, _ = hot_spots(rec, valid, thresh=thresh)
            hit = np.unique(lab[(lab > 0) & (lab_rec > 0)])
            rsk.append(float(skipped[hit].sum() / skipped.sum()))
        mse.append(float(np.mean(((rec - r) ** 2)[valid])))
    return {"quant_err": float(np.nanmean(q)), "recall": float(np.nanmean(rec_)),
            "precision": float(np.nanmean(prec_)),
            "recall_skipped": float(np.mean(rsk)) if rsk else np.nan,
            "mse_mc_mean": float(np.mean(mse)), "mse_mc_std": float(np.std(mse))}


# ── registration of a repeated scan (review F1) ────────────────────────────────

def register_rows(moving, reference, band=40, max_shift=8, smooth=5):
    """Align `moving` to `reference` with an integer x-shift per row.

    Two fly scans of the same area drift by up to ~5 fine pixels, smoothly
    across the frame. The shift is estimated per band of rows (correlation of
    smoothed band images), interpolated to every row, and applied with the
    pixels that wrap around marked invalid.
    Returns (registered, valid, shifts_per_row).
    """
    ny, nx = moving.shape
    ref = ndimage.uniform_filter(reference.astype(float), smooth)
    mov = ndimage.uniform_filter(moving.astype(float), smooth)
    centres, shifts = [], []
    for y0 in range(0, ny, band):
        a, b = ref[y0:y0 + band], mov[y0:y0 + band]
        if b.sum() <= 0:
            continue
        best, best_c = 0, -np.inf
        for s in range(-max_shift, max_shift + 1):
            sl = slice(max_shift, nx - max_shift)
            c = np.corrcoef(a[:, sl].ravel(), np.roll(b, s, axis=1)[:, sl].ravel())[0, 1]
            if c > best_c:
                best, best_c = s, c
        centres.append(y0 + min(band, ny - y0) / 2.0)
        shifts.append(best)
    if not shifts:
        return moving.copy(), np.zeros(moving.shape, bool), np.zeros(ny, int)
    per_row = np.rint(np.interp(np.arange(ny), centres, shifts)).astype(int)
    out = np.empty_like(moving, dtype=float)
    valid = np.ones(moving.shape, bool)
    for y, s in enumerate(per_row):
        out[y] = np.roll(moving[y], s)
        if s > 0:
            valid[y, :s] = False
        elif s < 0:
            valid[y, s:] = False
    return out, valid, per_row


# ── variance-stabilised criterion (review F3) ──────────────────────────────────

_LAM = np.concatenate([[0.0], np.logspace(-4, np.log10(400.0), 600)])


def _sqrt_var_table():
    from scipy.stats import poisson
    out = np.empty_like(_LAM)
    for i, lam in enumerate(_LAM):
        n = np.arange(0, int(lam + 12 * np.sqrt(lam) + 30))
        p = poisson.pmf(n, lam)
        m = (np.sqrt(n) * p).sum()
        out[i] = max((n * p).sum() - m * m, 0.0)
    return out


_SQRT_VAR = _sqrt_var_table()


def sqrt_poisson_var(lam):
    """Exact Var(sqrt(N)), N ~ Poisson(lam) (-> 1/4 for large lam). The 1/(4 t)
    approximation is badly off below a few counts (review F3)."""
    lam = np.maximum(np.asarray(lam, dtype=float), 0.0)
    v = np.interp(lam, _LAM, _SQRT_VAR)
    return np.where(lam > _LAM[-1], 0.25, v)


def sqrt_errors(truth_rate, t_truth, dwell, mask, pred_rate):
    """Expected squared error in the sqrt-rate domain.

    The Poisson MSE is dominated by a few very bright pixels (a missed
    particle costs r^2, a scanned one r/t). On sqrt(rate) the noise variance is
    ~1/(4 t) whatever the rate, so every pixel weighs alike. Uses the exact
    Var(sqrt(N/t)) = Var(sqrt N) / t, for the scanned pixels and for the
    truth-noise debias of the unscanned ones.
    """
    r = np.maximum(truth_rate, 0.0)
    t = np.maximum(dwell, 1e-9)
    scanned = np.where(mask, sqrt_poisson_var(r * t) / t, 0.0)
    unscanned = np.where(mask, 0.0,
                         (np.sqrt(np.maximum(pred_rate, 0.0)) - np.sqrt(r)) ** 2
                         - sqrt_poisson_var(r * t_truth) / t_truth)
    return scanned + unscanned


def top_share(err, valid, frac=0.01):
    """Share of the total error carried by the `frac` worst pixels."""
    e = np.sort(err[valid])[::-1]
    k = max(1, int(frac * e.size))
    return float(e[:k].sum() / e.sum()) if e.sum() > 0 else np.nan
