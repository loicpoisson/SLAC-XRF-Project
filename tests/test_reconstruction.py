"""Tests for utils.reconstruction (step 2)."""

import numpy as np
import pytest

from utils.reconstruction import (fill_cell, fill_bilinear, fill_harmonic,
                                  fill_kriging, coarse_anchor_index)


def _grids(nc=10, ratio=4):
    xc = np.arange(nc) * 0.1
    xf = np.arange((nc - 1) * ratio + 1) * 0.1 / ratio
    return ({"xdata": xc, "ydata": xc}, {"xdata": xf, "ydata": xf})


def test_bilinear_reproduces_a_plane_and_cell_does_not():
    cd, fd = _grids()
    YY, XX = np.meshgrid(cd["ydata"], cd["xdata"], indexing="ij")
    plane_c = 1.0 + 3.0 * XX + 2.0 * YY
    Yf, Xf = np.meshgrid(fd["ydata"], fd["xdata"], indexing="ij")
    plane_f = 1.0 + 3.0 * Xf + 2.0 * Yf
    np.testing.assert_allclose(fill_bilinear(plane_c, cd, fd), plane_f, atol=1e-9)
    assert np.abs(fill_cell(plane_c, cd, fd) - plane_f).max() > 0.1


def test_coarse_anchors_land_on_coarse_centres():
    cd, fd = _grids()
    fr, fc, cr, cc = coarse_anchor_index(cd, fd)
    np.testing.assert_allclose(fd["xdata"][fc], cd["xdata"][cc])
    np.testing.assert_allclose(fd["ydata"][fr], cd["ydata"][cr])


def test_harmonic_honours_anchors_and_stays_between_them():
    cd, fd = _grids()
    c = np.full((10, 10), 2.0)
    n = len(fd["xdata"])
    mask = np.zeros((n, n), bool)
    mask[10:14, 10:14] = True
    meas = np.where(mask, 10.0, 0.0)
    f = fill_harmonic(c, cd, fd, mask, meas, lam=0.05)
    assert f[mask].mean() > 8.0                       # close to the measurements
    assert f[0, 0] == pytest.approx(2.0, abs=0.2)     # far away: coarse level
    assert f.min() >= 2.0 - 1e-6 and f.max() <= 10.0 + 1e-6   # maximum principle


def test_kriging_equals_trend_without_fine_structure():
    cd, fd = _grids()
    YY, XX = np.meshgrid(cd["ydata"], cd["xdata"], indexing="ij")
    c = 1.0 + XX
    trend = fill_bilinear(c, cd, fd)
    n = len(fd["xdata"])
    mask = np.zeros((n, n), bool)
    mask[:, :8] = True
    out = fill_kriging(c, cd, fd, mask, np.where(mask, trend, 0.0),
                       np.where(mask, 10.0, 0.0), np.full((10, 10), 1e-3))
    np.testing.assert_allclose(out[~mask], trend[~mask], atol=1e-6)


def test_reconstruct_keeps_measurements_and_fills_the_rest():
    from utils.reconstruction import reconstruct
    cd, fd = _grids()
    n = len(fd["xdata"])
    coarse = np.full((10, 10), 20.0)                  # counts at 10 ms
    fine = np.full((n, n), 7.0)
    scanned = np.zeros((n, n), bool); scanned[:, : n // 2] = True
    recon, fill = reconstruct(coarse, cd, fd, scanned, fine, scale=0.5,
                              t_coarse=10.0, t_fine=10.0)
    np.testing.assert_array_equal(recon[scanned], 7.0)
    np.testing.assert_allclose(recon[~scanned], 20.0 * 0.5, rtol=1e-4)
    full, _ = reconstruct(coarse, cd, fd, np.ones((n, n), bool), fine, 0.5, 10.0, 10.0)
    np.testing.assert_array_equal(full, fine)


def test_calibrate_scale_without_other_samples(tmp_path):
    from utils.reconstruction import calibrate_scale
    assert calibrate_scale("XX", None, 100, 25, data_dir=tmp_path) == (1.0, 0)
