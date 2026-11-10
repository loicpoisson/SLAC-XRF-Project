"""Tests for utils.beamline (scanner parameters) and the self-calibration."""

import json

import numpy as np
import pytest

from utils.beamline import load_beamline, DEFAULTS
from utils.reconstruction import self_scale


def test_defaults_without_file():
    assert load_beamline(None) == DEFAULTS


def test_partial_file_overrides_only_its_keys(tmp_path):
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"_comment": "7-2", "v_max_mm_s": 50, "target_ladder_um": [200, 50]}))
    cfg = load_beamline(f)
    assert cfg["v_max_mm_s"] == 50.0 and cfg["a_max_mm_s2"] == DEFAULTS["a_max_mm_s2"]
    assert cfg["target_ladder_um"] == [200, 50]


@pytest.mark.parametrize("bad", [
    {"vmax": 1}, {"a_max_mm_s2": 0}, {"dwell_low_ms": 60, "dwell_high_ms": 50},
    {"v_max_mm_s": "200"}, {"v_max_mm_s": True}, {"v_max_mm_s": None},
    {"dwell_low_ms": 20},                       # above the 10 ms reference dwell
    {"target_ladder_um": []}, {"target_ladder_um": [25, 100, 250]},
    {"target_ladder_um": [250, -5]}, {"target_ladder_um": [250, 12.5]},
    {"target_ladder_um": 250}])
def test_bad_files_are_refused(tmp_path, bad):
    f = tmp_path / "b.json"
    f.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_beamline(f)


def test_repo_beamline_file_is_valid():
    from pathlib import Path
    load_beamline(Path(__file__).parent.parent / "beamline.json")


def test_self_scale_is_the_count_ratio_on_scanned_pixels():
    xc = np.arange(10) * 0.1
    xf = np.arange(37) * 0.025
    cd, fd = {"xdata": xc, "ydata": xc}, {"xdata": xf, "ydata": xf}
    coarse = np.full((10, 10), 40.0)
    fine = np.full((37, 37), 28.0)
    scanned = np.zeros((37, 37), bool); scanned[:20] = True
    assert self_scale(coarse, cd, fd, scanned, fine) == pytest.approx(0.7)
    few = np.zeros((37, 37), bool); few[0, :5] = True
    assert self_scale(coarse, cd, fd, few, fine) is None          # too few pixels


def test_nan_and_missing_file_are_refused(tmp_path):
    f = tmp_path / "b.json"
    f.write_text('{"v_max_mm_s": NaN}')
    with pytest.raises(ValueError):
        load_beamline(f)
    with pytest.raises(ValueError, match="not readable"):
        load_beamline(tmp_path / "missing.json")
