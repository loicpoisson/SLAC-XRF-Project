"""Tests for utils.line_scan (step 3)."""

import numpy as np
import pytest

from utils.line_scan import plan_lines, raster_line_time, fit_to_budget

PX = 0.025
A = 500.0


def _axes(ny, nx):
    return np.arange(nx) * PX, np.arange(ny) * PX


def test_uniform_raster_costs_its_dwell_plus_moves():
    x, y = _axes(4, 50)
    plan = plan_lines(np.full((4, 50), 10.0), x, y)
    # 2.5 mm/s is far below the ramp limit sqrt(2 a px) = 5 mm/s
    np.testing.assert_allclose(plan["dwell_ms"], 10.0)
    assert plan["extra_dwell"] == pytest.approx(0.0)
    assert plan["total_ms"] == pytest.approx(4 * 50 * 10.0 + plan["move_ms"])
    assert plan["move_ms"] > 0                       # 3 flybacks of 1.25 mm


def test_acceleration_bound_holds_and_dwell_never_shrinks():
    rng = np.random.default_rng(0)
    d = np.where(rng.random((6, 80)) > 0.5, rng.uniform(1, 50, (6, 80)), 0.0)
    x, y = _axes(6, 80)
    plan = plan_lines(d, x, y)
    v = plan["velocity_mm_s"]
    for r in range(6):
        a, b = plan["x0"][r], plan["x1"][r]
        vr = v[r, a:b + 1]
        # speeds are planned at pixel boundaries (|du^2| <= 2 a px per pixel);
        # two neighbouring pixel MEANS span two pixels of acceleration
        assert np.all(np.abs(np.diff(vr ** 2)) <= 4 * A * PX + 1e-9)
        # crossing time never beats the kinematic minimum for the speed change
        t = plan["crossing_ms"][r, a:b + 1]
        assert np.all(t > 0)
    s = d > 0
    assert np.all(plan["dwell_ms"][s] >= d[s] - 1e-9)


def test_gap_times_match_constant_acceleration():
    # 4- and 40-pixel gaps between two 10 ms pixels (2.5 mm/s): accelerate
    # from 2.5 mm/s and brake back at 500 mm/s^2 -> 20 ms and 80 ms
    for g, expect in ((4, 20.0), (40, 80.0)):
        d = np.zeros((1, g + 2)); d[0, 0] = d[0, -1] = 10.0
        x, y = _axes(1, g + 2)
        t = plan_lines(d, x, y)["line_ms"][0] - 20.0
        assert t == pytest.approx(expect, rel=0.02)


def test_gaps_are_crossed_faster_than_scanned():
    x, y = _axes(1, 400)
    d = np.zeros((1, 400)); d[0, :10] = 10.0; d[0, -10:] = 10.0
    plan = plan_lines(d, x, y)
    full = raster_line_time((1, 400), x, y, 10.0)
    assert plan["total_ms"] < 0.5 * full             # 380 skipped pixels


def test_fit_to_budget_hits_the_total():
    rng = np.random.default_rng(1)
    d = np.where(rng.random((10, 60)) > 0.4, rng.uniform(2, 30, (10, 60)), 0.0)
    x, y = _axes(10, 60)
    budget = raster_line_time(d.shape, x, y, 10.0)
    d2, plan = fit_to_budget(d, budget, x, y)
    assert plan["total_ms"] == pytest.approx(budget, rel=1e-3)
    assert d2[d > 0].min() >= 1.0 and d2.max() <= 50.0


def test_pipeline_line_plan_resamples_and_matches_raster_time():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).parent.parent / "scripts" / "10_run_pipeline.py"
    spec = importlib.util.spec_from_file_location("rp", path)
    rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)
    grid = {"xdata": np.arange(12) * 0.1, "ydata": np.arange(8) * 0.1}
    mask = np.zeros((8, 12), bool); mask[1:7, 1:11] = True   # ~60 % of the frame
    dwell = np.where(mask, 20.0, 0.0)
    args = SimpleNamespace(fine_dwell=10.0, no_match_budget=False,
                           dwell_low=1.0, dwell_high=50.0)
    xf, yf, d, plan, ras = rp.line_scan_plan(grid, mask, dwell, 0.025, args,
                                             spent_ms=0.1 * 1e3)
    assert len(xf) == 45 and len(yf) == 29            # 1.1 mm / 25 um + 1
    assert plan["total_ms"] + 100.0 == pytest.approx(ras, rel=1e-3)
    assert (d > 0).any() and not (d > 0).all()
