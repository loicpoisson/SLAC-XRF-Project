"""Tests for utils.smak_export: the SMAK dataqueue (.dpm) region file."""

import importlib
import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.smak_export import (HEADER, SMAK_DEFAULTS, SMAK_REGION_KEYS,
                               read_smak_regions, write_smak_regions)


def _region(rid=0, x0=-285.9268, x1=-284.9268, y0=204.7499, y1=205.4999,
            px=0.025, dwell=12.5):
    """Same keys/types as 10_run_pipeline._region_dict."""
    return {"region_id": rid, "x_start_mm": x0, "x_end_mm": x1,
            "y_start_mm": y0, "y_end_mm": y1, "fine_px_mm": px,
            "n_pixels_x": int(np.ceil(abs(x1 - x0) / px)) + 1,
            "n_pixels_y": int(np.ceil(abs(y1 - y0) / px)) + 1,
            "dwell_mean_ms": dwell, "dwell_median_ms": dwell,
            "dwell_min_ms": dwell, "dwell_max_ms": dwell,
            "dwell_uniform_ms": dwell}


PLAN = [_region(0), _region(1, -280.0, -279.5, 210.0, 211.25, 0.025, 30.0)]


# ── format ────────────────────────────────────────────────────────────────────

def test_header_bytes_and_ascii_payload(tmp_path):
    p = tmp_path / "plan.dpm"
    write_smak_regions(p, PLAN, 12000.0, "UA1_P1_step1_25um")
    raw = p.read_bytes()
    assert raw.startswith(b"DATASERVER REGIONS\n") and HEADER == raw[:19]
    assert max(raw) < 128                       # protocol 0 -> pure ASCII
    # SMAK's own existence check: text mode, locale codec (cp1252 on Windows)
    with open(p, "r", encoding="cp1252") as f:
        assert f.readline() == "DATASERVER REGIONS\n"
    # the rest is exactly one pickle
    assert pickle.loads(raw[len(HEADER):]) == read_smak_regions(p)


def test_round_trip_values_and_key_order(tmp_path):
    p = tmp_path / "plan.dpm"
    written = write_smak_regions(p, PLAN, 12000.0, "S")
    q = read_smak_regions(p)
    assert q == written
    assert list(q) == [1, 2]
    for (k, (kind, reg)), r in zip(sorted(q.items()), PLAN):
        assert kind == "image"
        assert tuple(reg) == SMAK_REGION_KEYS           # same keys, same order
        assert reg["yrange"] == [r["x_start_mm"], r["x_end_mm"], r["fine_px_mm"]]
        assert reg["zrange"] == [r["y_start_mm"], r["y_end_mm"], -r["fine_px_mm"]]
        assert reg["dwell"] == r["dwell_uniform_ms"]
        assert reg["energy"] == [12000.0]
        assert reg["file"] == f"S_r{r['region_id']}"
        for f, v in SMAK_DEFAULTS.items():
            assert reg[f] == v
        assert all(type(v) is float for v in reg["yrange"] + reg["zrange"])


def test_empty_plan(tmp_path):
    p = tmp_path / "empty.dpm"
    assert write_smak_regions(p, [], 12000.0, "S") == {}
    assert p.read_bytes().startswith(HEADER)
    assert read_smak_regions(p) == {}


def test_append_numbers_like_smak(tmp_path):
    p = tmp_path / "plan.dpm"
    write_smak_regions(p, PLAN, 12000.0, "A")
    q = write_smak_regions(p, PLAN[:1], 11000.0, "B", append=True)
    assert list(q) == [1, 2, 3]
    assert q[3][1]["file"] == "B_r0" and q[3][1]["energy"] == [11000.0]
    assert read_smak_regions(p) == q


def test_unsorted_box_is_written_low_to_high(tmp_path):
    p = tmp_path / "plan.dpm"
    r = _region(0, x0=-284.0, x1=-285.0, y0=206.0, y1=205.0)
    reg = write_smak_regions(p, [r], 12000.0, "S")[1][1]
    assert reg["yrange"][:2] == [-285.0, -284.0]
    assert reg["zrange"][:2] == [205.0, 206.0]


def test_field_overrides_and_validation(tmp_path):
    p = tmp_path / "plan.dpm"
    reg = write_smak_regions(p, PLAN[:1], [12000, 12100], "S", wheel=3,
                             comment="adaptive\n")[1][1]
    assert reg["wheel"] == 3 and reg["comment"] == "adaptive\n"
    assert reg["energy"] == [12000.0, 12100.0]
    with pytest.raises(TypeError):
        write_smak_regions(p, PLAN, 12000.0, "S", dwell=5)
    with pytest.raises(ValueError):
        write_smak_regions(p, [{**_region(), "fine_px_mm": 0.0}], 12000.0, "S")
    with pytest.raises(ValueError):
        write_smak_regions(p, [_region(dwell=float("nan"))], 12000.0, "S")
    with pytest.raises(ValueError):
        write_smak_regions(p, PLAN, 0.0, "S")
    with pytest.raises(KeyError):
        write_smak_regions(p, [{"x_start_mm": 0.0}], 12000.0, "S")


def test_reader_rejects_bad_header(tmp_path):
    p = tmp_path / "bad.dpm"
    p.write_bytes(b"NOT REGIONS\n" + pickle.dumps({}))
    with pytest.raises(ValueError):
        read_smak_regions(p)


def test_accepts_real_region_dict_from_pipeline(tmp_path):
    path = Path(__file__).parent.parent / "scripts" / "10_run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", path)
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    r = rp._region_dict(7, -285.9, -285.0, 204.8, 205.3,
                        np.array([5.0, 15.0]), 0.025, 10.0)
    reg = write_smak_regions(tmp_path / "p.dpm", [r], 12000.0, "S")[1][1]
    assert reg["dwell"] == 10.0 and reg["file"] == "S_r7"
    assert reg["yrange"] == [-285.9, -285.0, 0.025]
    assert reg["zrange"] == [204.8, 205.3, -0.025]


# ── against SMAK's own code (smak.py Main.exportScanAreaDQ) ──────────────────

def _smak_main():
    try:
        mod = importlib.import_module("smak")
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"SMAK not importable: {e}")
    if not hasattr(getattr(mod, "Main", None), "exportScanAreaDQ"):
        pytest.skip("SMAK Main.exportScanAreaDQ not found")
    return mod


def _run_smak_export(monkeypatch, target, xsc, ysc, zmxyi, dwell, energy,
                     wheel, fname):
    """Run SMAK's real exportScanAreaDQ with its Tk dialogs stubbed out.
    smak.py only imports tkinter/pickle/globalfuncs under __main__, so they
    are injected as module globals."""
    smak = _smak_main()
    msgs = []
    floats = {"dwell": dwell, "energy": energy}
    simpledialog = SimpleNamespace(
        askfloat=lambda title, prompt, initialvalue=None:
            floats["dwell" if "dwell" in prompt else "energy"],
        askinteger=lambda title, prompt, initialvalue=None: wheel,
        askstring=lambda title, prompt, initialvalue=None: fname)
    gf = SimpleNamespace(trimdirext=lambda s: s,
                         ask_save_file=lambda fn, d: str(target),
                         setstatus=lambda status, msg: msgs.append(msg))
    monkeypatch.setattr(smak, "tkinter",
                        SimpleNamespace(simpledialog=simpledialog), raising=False)
    monkeypatch.setattr(smak, "globalfuncs", gf, raising=False)
    monkeypatch.setattr(smak, "pickle", pickle, raising=False)
    fake = SimpleNamespace(
        hasdata=True, dataFileBuffer={0: {"fname": "map"}}, activeFileBuffer=0,
        filedir=SimpleNamespace(get=lambda: ""), status=None,
        maindisp=SimpleNamespace(xsc=xsc, ysc=ysc, zmxyi=zmxyi),
        mapdata=SimpleNamespace(energy=energy))
    smak.Main.exportScanAreaDQ(fake)
    return msgs


def _grid():
    # Real FP1 25 um grid layout (ascending xdata/ydata as in the HDF5 files).
    xdata = np.round(-285.9268 + 0.025 * np.arange(1001), 4)
    ydata = np.round(204.7499 + 0.025 * np.arange(671), 4)
    return xdata, ydata


def test_smak_writes_same_region_for_same_box(tmp_path, monkeypatch):
    xdata, ydata = _grid()
    i0, i1, j0, j1 = 40, 80, 100, 130
    ny = len(ydata)
    ysc = ydata[::-1]                            # Display.placeData: yax[::-1]
    zmxyi = [i0, ny - 1 - j1, i1, ny - 1 - j0, 0, 0]   # sorted zoom box
    smak_file = tmp_path / "smak.dpm"
    msgs = _run_smak_export(monkeypatch, smak_file, xdata, ysc, zmxyi,
                            dwell=12.5, energy=12000.0, wheel=0, fname="S_r0")
    assert msgs[-1] == "Export complete"
    smak_q = read_smak_regions(smak_file)
    ours = write_smak_regions(
        tmp_path / "ours.dpm",
        [_region(0, xdata[i0], xdata[i1], ydata[j0], ydata[j1], 0.025, 12.5)],
        12000.0, "S")
    assert list(smak_q) == list(ours) == [1]
    s_kind, s = smak_q[1]
    o_kind, o = ours[1]
    assert s_kind == o_kind == "image"
    assert list(s) == list(o) == list(SMAK_REGION_KEYS)
    for k in ("comment", "ooptics", "scandir", "scantype", "slave", "wheel",
              "file", "dwell", "energy"):
        assert s[k] == o[k], k
    np.testing.assert_allclose(s["yrange"], o["yrange"], atol=1e-9)
    np.testing.assert_allclose(s["zrange"], o["zrange"], atol=1e-9)
    assert s["zrange"][2] < 0 < s["yrange"][2]   # SMAK's sign convention
    assert s["zrange"][0] < s["zrange"][1]       # ...with start < stop


@pytest.mark.parametrize("n_regions", [0, 3])
def test_smak_appends_to_our_file(tmp_path, monkeypatch, n_regions):
    """SMAK's reader side: header check + pickle.loads + len()+1 append."""
    xdata, ydata = _grid()
    p = tmp_path / "plan.dpm"
    plan = [_region(i, dwell=5.0 + i) for i in range(n_regions)]
    ours = write_smak_regions(p, plan, 12000.0, "S")
    msgs = _run_smak_export(monkeypatch, p, xdata, ydata[::-1],
                            [0, 0, -1, -1, 0, 0], dwell=25.0, energy=12000.0,
                            wheel=0, fname="")
    assert msgs[-1] == "Export complete", msgs   # not 'Save cancelled'
    q = read_smak_regions(p)
    assert list(q) == list(range(1, n_regions + 2))
    for k in ours:
        assert q[k] == ours[k]                   # our entries survive intact
    assert tuple(q[n_regions + 1][1]) == SMAK_REGION_KEYS
