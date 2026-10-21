"""Regression tests for the 2026-10 review: HDF5 channel layout, TIME
normalisation, mask borders and the production-pipeline outputs."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from utils.hdf5_reader import load_xrf, get_channel, get_composite_map
from utils.roi_utils import sample_mask, threshold_map

LABELS = ["I0", "Fe.Ka", "Zn.Ka", "OCR", "TIME"]


def _write_smak(path, fe, zn, time, ny=4, nx=5):
    """Minimal SMAK-style file: planes 0/1 = stage X/Y, plane i+2 = labels[i]."""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(2.0, 3.0, ny)
    xx, yy = np.meshgrid(x, y)
    planes = [xx, yy, np.ones((ny, nx)), fe, zn, fe + zn, time]
    with h5py.File(path, "w") as f:
        f["main/mapdata"] = np.stack(planes, axis=2)
        f["main/xdata"] = x
        f["main/ydata"] = y
        f["main"].attrs["labels"] = [s.encode() for s in LABELS]
        f["main"].attrs["energy"] = 12000.0


def test_load_xrf_skips_stage_position_planes(tmp_path):
    fe = np.full((4, 5), 7.0)
    zn = np.arange(20, dtype=float).reshape(4, 5)
    _write_smak(tmp_path / "s.hdf5", fe, zn, np.ones((4, 5)))
    d = load_xrf(tmp_path / "s.hdf5")
    assert d["mapdata"].shape == (4, 5, len(LABELS))
    np.testing.assert_array_equal(get_channel(d, "Fe.Ka"), fe)
    np.testing.assert_array_equal(get_channel(d, "Zn.Ka"), zn)
    comp, used, excluded = get_composite_map(d)
    assert used == ["Zn.Ka"] and excluded == ["Fe.Ka"]   # Fe constant here


def test_load_xrf_rescales_counts_by_time(tmp_path):
    # Column 0 integrated 60x longer: counts 60x higher, same true rate.
    time = np.ones((4, 5)); time[:, 0] = 60.0
    zn = np.full((4, 5), 10.0) * time
    _write_smak(tmp_path / "s.hdf5", np.ones((4, 5)) * time, zn, time)
    d = load_xrf(tmp_path / "s.hdf5")
    np.testing.assert_allclose(get_channel(d, "Zn.Ka"), 10.0)
    np.testing.assert_allclose(get_channel(d, "I0"), 1.0)      # not a count
    raw = load_xrf(tmp_path / "s.hdf5", normalize_time=False)
    assert get_channel(raw, "Zn.Ka")[0, 0] == 600.0


def test_sample_mask_closing_keeps_border_pixels():
    comp = np.full((10, 12), 5.0)
    comp[:, :] += np.random.default_rng(0).random((10, 12))
    comp[0, :] = 500.0
    comp[:, 0] = 500.0
    mask, thresh = sample_mask(comp, kernel_px=2, mode="percentile", level=50)
    assert np.all(mask[comp > thresh])        # closing only ever adds pixels


def test_threshold_map_empty_mask():
    m, t, _ = threshold_map(np.ones((5, 5)), mask=np.zeros((5, 5), bool))
    assert not m.any() and np.isnan(t)


def _load_pipeline():
    path = Path(__file__).parent.parent / "scripts" / "10_run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_json_safe_handles_masks_and_inf():
    rp = _load_pipeline()
    out = rp._json_safe({"a": np.float64(np.inf), "b": np.int64(3),
                         "c": [np.zeros((2, 2), bool)], "d": (1.5, np.nan)})
    text = json.dumps(out, allow_nan=False)        # must be strict JSON
    assert json.loads(text) == {"a": None, "b": 3,
                                "c": [[[False, False], [False, False]]],
                                "d": [1.5, None]}


def test_raster_baseline_matches_full_frame_region():
    rp = _load_pipeline()
    grid = {"xdata": np.arange(93) * 0.25, "ydata": np.arange(66) * 0.25,
            "dx": 0.25, "dy": 0.25}
    # The real UA1_P1 100um scan of this frame is 231 x 164 pixels.
    assert rp.raster_time_ms(grid, 0.1, 10.0) == 231 * 164 * 10.0


def test_match_budget_respects_dwell_bounds():
    rp = _load_pipeline()
    grid = {"xdata": np.arange(10) * 0.25, "ydata": np.arange(10) * 0.25,
            "dx": 0.25, "dy": 0.25}
    mask = np.zeros((10, 10), bool); mask[0, 0] = True    # tiny footprint
    dwell = np.where(mask, 10.0, 0.0)
    out = rp.match_budget(dwell, mask, grid, 0.025, 10.0,
                          dwell_low=1.0, dwell_high=50.0)
    assert out[mask].max() <= 50.0


def test_sqrt_reg_lifts_noise_floored_pixels():
    from utils.dwell import allocate_dwell
    sig = np.array([0.0, 0.0, 1.0, 100.0])
    plain, _ = allocate_dwell(sig, "sqrt", 1.0, 50.0)
    reg, _ = allocate_dwell(sig, "sqrt", 1.0, 50.0, reg=1.0)
    assert plain[0] == 1.0 and reg[0] > plain[0]     # zero estimate not starved
    assert reg[3] >= reg[2] >= reg[0]                  # still monotone


def test_element_ladder_skips_costly_intermediate_level():
    rp = _load_pipeline()
    assert rp.element_ladder([250, 100, 50, 25]) == [250, 100, 25]
    assert rp.element_ladder([250, 100, 50]) == [250, 100, 50]
    assert rp.element_ladder([250, 25]) == [250, 25]


def test_match_budget_deducts_earlier_levels():
    rp = _load_pipeline()
    grid = {"xdata": np.arange(10) * 0.25, "ydata": np.arange(10) * 0.25,
            "dx": 0.25, "dy": 0.25}
    full = np.ones((10, 10), bool)
    raster = rp.raster_time_ms(grid, 0.25, 10.0)
    out = rp.match_budget(np.full((10, 10), 10.0), full, grid, 0.25, 10.0,
                          spent_ms=0.25 * raster)
    assert out.sum() == pytest.approx(0.75 * raster)
