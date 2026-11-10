"""End-to-end test of the stepwise pipeline on synthetic SMAK files:
250 um -> full 100 um detection scan -> PARTIAL 25 um scan (only the planned
pixels, variable dwell, split over two files) -> reconstructed map.
No project data needed, so it also runs in CI."""

import json

import h5py
import numpy as np
import pytest

import xrf_pipeline

LABELS = ["I0", "Zn.Ka", "Fe.Ka", "TIME"]
TICKS_PER_MS = 1000.0
W_MM, H_MM = 6.0, 5.0
BACKGROUND, SPOT = 0.3, 30.0          # Zn rate [counts / ms]


def _truth_rate(x, y):
    """Zn rate at stage position (x, y): flat background + 3 hot spots."""
    r = np.full(np.broadcast(x, y).shape, BACKGROUND)
    for cx, cy, rad in ((1.5, 1.2, 0.30), (4.2, 3.4, 0.40), (3.0, 2.0, 0.25)):
        r = np.where((x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2, SPOT, r)
    return r


def _write(path, x, y, zn_counts, time_ms):
    """SMAK layout: planes 0/1 = stage X/Y, then one plane per label."""
    xx, yy = np.meshgrid(x, y)
    planes = [xx, yy, np.ones_like(xx), zn_counts, zn_counts * 0.5 + 1.0,
              time_ms * TICKS_PER_MS]
    with h5py.File(path, "w") as f:
        f["main/mapdata"] = np.stack(planes, axis=2).astype(float)
        f["main/xdata"] = x
        f["main/ydata"] = y
        f["main"].attrs["labels"] = [s.encode() for s in LABELS]
        f["main"].attrs["energy"] = 12000.0


def _full_scan(path, px_mm, rng, dwell=10.0):
    x = np.arange(int(round(W_MM / px_mm)) + 1) * px_mm
    y = np.arange(int(round(H_MM / px_mm)) + 1) * px_mm
    xx, yy = np.meshgrid(x, y)
    counts = rng.poisson(_truth_rate(xx, yy) * dwell).astype(float)
    _write(path, x, y, counts, np.full(xx.shape, dwell))


@pytest.fixture
def cascade(tmp_path):
    rng = np.random.default_rng(0)
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    # --roi-k 2: on this flat-noise background the default k = 0.5 selects ~90 %
    # of the frame (a fixed share of pure noise exceeds a low threshold, and the
    # safety ring spreads it); this test is about the mechanics, not the rule
    common = ["--target", "Zn.Ka", "--roi-k", "2", "--out-dir", str(out),
              "--data-dir", str(data), "--no-show"]
    state = out / "SYN_P1_cascade_state.pkl"

    # step 0: only the coarse scan exists -> ladder from beamline.json
    f250 = data / "SMW_SYN_P1_250um_10ms_12000_0_001.hdf5"
    _full_scan(f250, 0.25, rng)
    xrf_pipeline.main(["--coarse", str(f250)] + common)

    # step 1: the full-frame 100 um detection scan -> plan of the 25 um scan
    f100 = data / "SMW_SYN_P1_100um_10ms_12000_0_001.hdf5"
    _full_scan(f100, 0.1, rng)
    xrf_pipeline.main(["--scan", str(f100), "--resume", str(state)] + common)

    # "acquire" the plan: only the planned pixels, at the planned dwell
    plan = np.load(out / "SYN_P1_step1_100um_line_plan.npz")
    x, y, dwell = plan["xdata_mm"], plan["ydata_mm"], plan["dwell_ms"].astype(float)
    xx, yy = np.meshgrid(x, y)
    rate = _truth_rate(xx, yy)
    counts = rng.poisson(rate * dwell).astype(float)        # dwell 0 -> 0 counts
    half = len(y) // 2
    fa = data / "SMW_SYN_P1_25um_10ms_12000_0_001.hdf5"
    fb = data / "SMW_SYN_P1_25um_10ms_12000_0_002.hdf5"
    _write(fa, x, y[:half], counts[:half], dwell[:half])
    _write(fb, x, y[half:], counts[half:], dwell[half:])

    # step 2: final step with the partial scan in two files
    xrf_pipeline.main(["--scan", str(fa), str(fb), "--resume", str(state)] + common)
    cascade.files = {"f250": f250, "f100": f100, "fa": fa, "fb": fb, "state": state,
                     "io": ["--out-dir", str(out), "--data-dir", str(data), "--no-show"]}
    return out, dwell, rate, counts


def test_cascade_plans_and_reconstructs_from_a_partial_scan(cascade):
    out, dwell, rate, counts = cascade
    planned = dwell > 0

    # the plan skips area and concentrates the dwell on the hot spots
    assert 0.02 < planned.mean() < 0.8
    assert planned[rate == SPOT].mean() > 0.95
    est = json.loads((out / "SYN_P1_step1_100um_scan_plan.json").read_text())["estimate"]
    assert est["line_scan_budget_matched"]
    assert est["line_scan_total_s"] == pytest.approx(est["line_scan_raster_s"], rel=2e-3)

    recon = np.load(out / "SYN_P1_step2_25um_reconstruction.npy")
    scanned = np.load(out / "SYN_P1_step2_25um_scanned_mask.npy")
    np.testing.assert_array_equal(scanned, planned)          # exactly what was measured

    # scanned pixels: the measurement itself, brought to one common dwell
    t_ref = np.median(dwell[planned])
    np.testing.assert_allclose(recon[scanned], (counts / np.where(planned, dwell, 1.0)
                                                * t_ref)[scanned], rtol=1e-9)
    # skipped pixels: interpolated from the 100 um scan, on the same scale -
    # close to the true background, far below a hot spot
    fill_rate = recon[~scanned] / t_ref
    assert np.all(np.isfinite(fill_rate)) and fill_rate.min() >= 0
    assert fill_rate.mean() == pytest.approx(rate[~scanned].mean(), rel=0.35)
    assert np.median(fill_rate) < 0.1 * SPOT


def test_final_step_refuses_a_scan_with_no_planned_pixel(cascade, tmp_path):
    out, dwell, rate, counts = cascade
    # an (empty) scan far away from the planned area
    far = tmp_path / "SMW_SYN_P1_25um_10ms_12000_0_009.hdf5"
    x = 100.0 + np.arange(20) * 0.025
    _write(far, x, x, np.zeros((20, 20)), np.ones((20, 20)))
    # the cascade state was consumed by the fixture's final step but is kept
    with pytest.raises(SystemExit, match="No planned pixel"):
        xrf_pipeline.main(["--scan", str(far), "--resume",
                           str(out / "SYN_P1_cascade_state.pkl"), "--target", "Zn.Ka",
                           "--out-dir", str(out), "--no-show"])


# ── regressions for the review of the operational pipeline ────────────────────

def _state(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def test_final_step_refuses_the_scan_of_another_level(cascade):
    f = cascade_files()
    with pytest.raises(SystemExit, match="another level"):
        xrf_pipeline.main(["--scan", str(f["f100"]), "--resume", str(f["state"])] + f["io"])


def test_resume_keeps_the_target_and_refuses_a_different_one(cascade):
    f = cascade_files()
    st = _state(f["state"])
    assert st["target"] == "Zn.Ka" and st["roi_k"] == 2.0
    # no --target / --roi-k on the command line: the stored ones are used
    xrf_pipeline.main(["--scan", str(f["fa"]), str(f["fb"]), "--resume", str(f["state"])]
                      + f["io"])
    with pytest.raises(SystemExit, match="contradicts"):
        xrf_pipeline.main(["--scan", str(f["fa"]), str(f["fb"]), "--resume",
                           str(f["state"]), "--target", "Fe.Ka"] + f["io"])


def test_rerunning_step0_keeps_the_full_ladder(cascade):
    # the 100 um detection scan is on disk now: the ladder must still reach 25 um
    f = cascade_files()
    xrf_pipeline.main(["--coarse", str(f["f250"]), "--target", "Zn.Ka", "--roi-k", "2"]
                      + f["io"])
    assert _state(f["state"])["ladder"] == [250, 100, 25]


def test_explicit_equal_dwell_scale_accounts_for_the_longer_fine_dwell(cascade):
    # coarse and fine sample the same rate, so the equal-dwell scale is 1; the
    # fine scan dwells longer than 10 ms on average, which the fill must follow
    out, dwell, rate, counts = cascade
    f = cascade_files()
    xrf_pipeline.main(["--scan", str(f["fa"]), str(f["fb"]), "--resume", str(f["state"]),
                       "--coarse-scale", "1.0"] + f["io"])
    recon = np.load(out / "SYN_P1_step2_25um_reconstruction.npy")
    planned = dwell > 0
    t_ref = np.median(dwell[planned])
    assert t_ref > 12.0                                   # not the nominal 10 ms
    fill_rate = recon[~planned] / t_ref
    assert fill_rate.mean() == pytest.approx(rate[~planned].mean(), rel=0.35)


def cascade_files():
    return cascade.files
