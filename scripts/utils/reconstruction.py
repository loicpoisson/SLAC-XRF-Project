"""
Reconstruction of unscanned fine pixels (step 2).

An adaptive scan leaves two kinds of measurements on the fine grid:
  - coarse samples (250 / 100 um scans), which behave like POINT samples at
    their pixel centres: the step-1 review showed bright particles a few fine
    pixels from a coarse centre leave almost no counts in that coarse pixel,
    so a coarse value is not the average of its cell;
  - fine measurements inside the scanned mask (noisy, dwell t_i).
Every reconstructor below sees only those measurements, never the truth.

  fill_cell      : replicate each coarse value over its cell (step-1 baseline)
  fill_bilinear  : bilinear interpolation between coarse sample points
  fill_harmonic  : graph Laplacian interpolation (label propagation) on the
                   fine 4-neighbour grid graph, anchored on coarse samples and
                   fine measurements, solved as (W + lam L) f = W y
  fill_kriging   : simple kriging of the residual to the bilinear trend, from
                   the K nearest anchors, with per-anchor Poisson nugget.
                   Experimental: the residual semivariogram of these scans is
                   dominated by a few cut particles, so the fitted covariance
                   is unstable (step-2 review F2) and it gains nothing over
                   bilinear; kept behind --kriging only.

Harmonic with fine anchors propagates the values of particles cut by the ROI
border into the (by construction darker) pixels just outside: on FP1 Fe that
multiplies the unscanned error by ~10. With coarse anchors only (w_fine=0) it
is a stable interpolator of the coarse point samples.

Rates are counts per ms per fine pixel.
"""

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree

from .validation_utils import project_coarse_to_fine, _nearest_idx


# ── coarse samples on the fine grid ────────────────────────────────────────────

def coarse_rate(cmap, scale, t_coarse):
    """Coarse counts -> fine-scale rate (calibrated)."""
    return np.maximum(cmap, 0.0) * scale / t_coarse


def fill_cell(c_rate, cdata, fdata):
    return project_coarse_to_fine(c_rate, cdata, fdata)


def fill_bilinear(c_rate, cdata, fdata):
    """Bilinear interpolation between coarse pixel centres (point samples)."""
    xc, yc = np.asarray(cdata["xdata"]), np.asarray(cdata["ydata"])
    ox, oy = np.argsort(xc), np.argsort(yc)
    f = RegularGridInterpolator((yc[oy], xc[ox]), c_rate[np.ix_(oy, ox)],
                                method="linear", bounds_error=False, fill_value=None)
    xf = np.clip(fdata["xdata"], xc.min(), xc.max())
    yf = np.clip(fdata["ydata"], yc.min(), yc.max())
    YY, XX = np.meshgrid(yf, xf, indexing="ij")
    return np.maximum(f(np.stack([YY.ravel(), XX.ravel()], 1)).reshape(YY.shape), 0.0)


def coarse_anchor_index(cdata, fdata):
    """Fine (row, col) of the pixel nearest each coarse centre, and the coarse
    (row, col) it comes from. Assumes the coarse and fine frames coincide (true
    for these SMAK scans); a coarse centre outside the fine frame would map onto
    the nearest edge pixel."""
    ix = _nearest_idx(np.asarray(fdata["xdata"]), np.asarray(cdata["xdata"]))
    iy = _nearest_idx(np.asarray(fdata["ydata"]), np.asarray(cdata["ydata"]))
    CY, CX = np.meshgrid(np.arange(len(iy)), np.arange(len(ix)), indexing="ij")
    return iy[CY].ravel(), ix[CX].ravel(), CY.ravel(), CX.ravel()


# ── graph Laplacian interpolation ──────────────────────────────────────────────

@lru_cache(maxsize=4)
def _grid_laplacian(ny, nx):
    """Combinatorial Laplacian of the 4-neighbour grid graph (ny*nx nodes).
    Cached per shape: it is the same for every fill of a given scan."""
    idx = np.arange(ny * nx).reshape(ny, nx)
    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    n = ny * nx
    A = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    A = (A + A.T).tocsr()
    return sparse.diags(np.asarray(A.sum(1)).ravel()) - A


def fill_harmonic(c_rate, cdata, fdata, mask, meas, lam=1.0, w_coarse=1.0,
                  w_fine=1.0, x0=None, maxiter=20000):
    """Label propagation on the fine grid graph.

    Minimises  sum_i w_i (f_i - y_i)^2 + lam * sum_(i~j) (f_i - f_j)^2
    over all fine pixels, with anchors y = coarse samples (at the fine pixel
    nearest each coarse centre) and fine measurements inside `mask`.
    Inside `mask` the fine measurement replaces the coarse anchor, so with
    w_fine=0 only the coarse samples OUTSIDE the scanned area anchor the fill
    (no value from inside the ROIs leaks across their border); the result
    then depends on the mask but not on the measurements.
    """
    ny, nx = mask.shape
    y = np.zeros(ny * nx)
    w = np.zeros(ny * nx)
    fr, fc, cr, cc = coarse_anchor_index(cdata, fdata)
    k = fr * nx + fc
    y[k] = c_rate[cr, cc]
    w[k] = w_coarse
    m = mask.ravel()
    y[m] = np.maximum(meas.ravel()[m], 0.0)
    w[m] = w_fine
    M = sparse.diags(w) + lam * _grid_laplacian(ny, nx)
    if x0 is None:
        x0 = fill_bilinear(c_rate, cdata, fdata)
    pre = sparse.diags(1.0 / M.diagonal())
    b = w * y
    f, info = cg(M, b, x0=x0.ravel(), M=pre, maxiter=maxiter, rtol=1e-6)
    if info != 0:
        # sparse anchors (250um -> 25um: 1 per 100 px) converge slowly; accept
        # a solution that is accurate to 1e-4, refuse anything worse
        rel = np.linalg.norm(M @ f - b) / max(np.linalg.norm(b), 1e-300)
        if rel > 1e-4:
            raise RuntimeError(f"harmonic CG did not converge (rel. residual {rel:.1e})")
    return np.maximum(f.reshape(ny, nx), 0.0)


# ── kriging of the residual to the bilinear trend ──────────────────────────────

def _exp_cov(h, sill, rng):
    return sill * np.exp(-h / rng)


def fit_residual_cov(res, mask, noise_var, max_lag=24, n_pairs=200000, seed=0):
    """Exponential covariance (sill, range in fine px) of the residual field,
    from an empirical semivariogram of fine anchors, nugget removed."""
    rng_ = np.random.default_rng(seed)
    yy, xx = np.nonzero(mask)
    if len(yy) < 100:
        return 0.0, 1.0
    i = rng_.integers(0, len(yy), n_pairs)
    dy = rng_.integers(-max_lag, max_lag + 1, n_pairs)
    dx = rng_.integers(-max_lag, max_lag + 1, n_pairs)
    y2, x2 = yy[i] + dy, xx[i] + dx
    ok = (y2 >= 0) & (y2 < mask.shape[0]) & (x2 >= 0) & (x2 < mask.shape[1])
    i, y2, x2, dy, dx = i[ok], y2[ok], x2[ok], dy[ok], dx[ok]
    ok = mask[y2, x2]
    i, y2, x2, dy, dx = i[ok], y2[ok], x2[ok], dy[ok], dx[ok]
    h = np.hypot(dy, dx)
    g = 0.5 * (res[yy[i], xx[i]] - res[y2, x2]) ** 2 \
        - 0.5 * (noise_var[yy[i], xx[i]] + noise_var[y2, x2])
    bins = np.arange(1, max_lag + 1)
    gb = np.array([g[(h >= b - 0.5) & (h < b + 0.5)].mean() if np.any((h >= b - 0.5) & (h < b + 0.5)) else np.nan
                   for b in bins])
    sill = max(float(np.nanmax(gb)), 0.0)
    if sill <= 0:
        return 0.0, 1.0
    # range: lag where the semivariogram reaches (1 - 1/e) of the sill
    reach = np.nonzero(gb >= (1 - np.exp(-1)) * sill)[0]
    rng_px = float(bins[reach[0]]) if len(reach) else float(max_lag)
    return sill, max(rng_px, 0.5)


def fill_kriging(c_rate, cdata, fdata, mask, meas, dwell, coarse_var,
                 k=8, chunk=50000, reach=3.0):
    """trend (bilinear) + simple kriging of the residual from the k nearest
    anchors (coarse samples with residual 0 and their own noise, fine
    measurements with Poisson nugget meas/dwell). Only pixels within
    `reach` covariance ranges of a fine measurement are kriged: farther away
    the residual correction is ~0 and the prediction is the trend."""
    trend = fill_bilinear(c_rate, cdata, fdata)
    res = np.where(mask, meas - trend, 0.0)
    noise = np.where(mask, np.maximum(meas, 0.0) / np.maximum(dwell, 1e-9), 0.0)
    sill, rng_px = fit_residual_cov(res, mask, noise)
    out = trend.copy()
    if sill <= 0:
        return out
    fr, fc, cr, cc = coarse_anchor_index(cdata, fdata)
    keep = ~mask[fr, fc]                 # a fine measurement supersedes it
    fr, fc, cr, cc = fr[keep], fc[keep], cr[keep], cc[keep]
    ay = np.concatenate([np.nonzero(mask)[0], fr])
    ax = np.concatenate([np.nonzero(mask)[1], fc])
    av = np.concatenate([res[mask], np.zeros(len(fr))])
    an = np.concatenate([noise[mask], coarse_var[cr, cc]])
    tree = cKDTree(np.stack([ay, ax], 1))
    from scipy.ndimage import distance_transform_edt
    near = (~mask) & (distance_transform_edt(~mask) <= reach * rng_px)
    qy, qx = np.nonzero(near)
    for s in range(0, len(qy), chunk):
        q = np.stack([qy[s:s + chunk], qx[s:s + chunk]], 1)
        _, nb = tree.query(q, k=k)
        P = np.stack([ay[nb], ax[nb]], -1).astype(np.float32)       # (n, k, 2)
        d_aa = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
        C = _exp_cov(d_aa, sill, rng_px)
        C[:, np.arange(k), np.arange(k)] += an[nb] + 1e-6 * sill
        c0 = _exp_cov(np.linalg.norm(P - q[:, None, :], axis=-1), sill, rng_px)
        wts = np.linalg.solve(C, c0[..., None])[..., 0]
        out[qy[s:s + chunk], qx[s:s + chunk]] += np.einsum("nk,nk->n", wts, av[nb])
    return np.maximum(out, 0.0)


# ── coarse -> fine intensity calibration (leave-one-sample-out) ────────────────

def calibrate_scale(sample, channels, coarse_px, fine_px, candidates=None,
                    data_dir=None, dwell_ms=10):
    """Coarse->fine count scale for `sample`, measured on OTHER samples.

    A 100 um pixel does not hold the same counts as a 25 um pixel at the same
    dwell (~0.7x at 100->25, ~1.0x at 100->50 on the May 2026 scans), so the
    coarse map must be rescaled before it fills fine pixels. The factor is
    mean(fine) / mean(coarse projected) on every other sample that has both a
    `coarse_px` and a `fine_px` scan at `dwell_ms`.
    Returns (scale, n_samples_used); (1.0, 0) when no other sample qualifies.
    """
    from pathlib import Path
    from .hdf5_reader import load_xrf, get_composite_map
    from .cascade import find_level_file
    from .paths import DATA_DIR, sample_hint_from_path
    base = Path(data_dir) if data_dir else DATA_DIR
    if candidates is None:
        candidates = sorted({sample_hint_from_path(p) for p in
                             base.rglob(f"*_{coarse_px}um_*.hdf5")} - {None})
    scales = []
    for other in candidates:
        if other == sample:
            continue
        pc = find_level_file(other, coarse_px, data_dir=base)
        pf = next(iter(sorted(base.rglob(f"*{other}_{fine_px}um_{dwell_ms}ms_*.hdf5"))), None)
        if pc is None or pf is None:
            continue
        dc, df = load_xrf(pc), load_xrf(pf)
        try:
            c, used, _ = get_composite_map(dc, channels)
            f, _, _ = get_composite_map(df, used)
        except ValueError:
            continue
        cf = project_coarse_to_fine(c, dc, df)
        if cf.mean() > 0:
            scales.append(float(f.mean()) / float(cf.mean()))
    return (float(np.mean(scales)), len(scales)) if scales else (1.0, 0)


def reconstruct(coarse_counts, cdata, fdata, scanned, fine_counts, scale,
                t_coarse, t_fine):
    """Final fine map of an adaptive scan, in counts at the fine dwell `t_fine`:
    the fine measurement where `scanned`, elsewhere the graph-harmonic
    interpolation of the coarse point samples lying outside the scanned area
    (see fill_harmonic and the module docstring).
    Returns (reconstructed, fill)."""
    if scanned.all():                       # nothing skipped, nothing to fill
        return fine_counts.copy(), fine_counts.copy()
    c_rate = coarse_rate(coarse_counts, scale, t_coarse)
    fill = fill_harmonic(c_rate, cdata, fdata, scanned,
                         np.zeros(scanned.shape), w_fine=0.0) * t_fine
    return np.where(scanned, fine_counts, fill), fill
