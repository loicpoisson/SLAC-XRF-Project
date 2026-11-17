"""Tests for the learned planner (scripts/ml). The numpy parts always run;
the model / training tests need PyTorch and skip without it (smak env, CI)."""

from types import SimpleNamespace

import numpy as np
import pytest

from ml.data import grid_edges, queries, N_FEATURES
from ml.plan import top_area_mask


def _fake_case(ny=12, nx=14, ratio=4, seed=0):
    """Minimal stand-in for utils.bench.Case: a coarse grid and a fine grid
    `ratio` times finer over the same frame."""
    rng = np.random.default_rng(seed)
    xc, yc = np.arange(nx) * 0.1, np.arange(ny) * 0.1
    xf = np.arange((nx - 1) * ratio + 1) * 0.1 / ratio
    yf = np.arange((ny - 1) * ratio + 1) * 0.1 / ratio
    coarse = rng.poisson(20.0, (ny, nx)).astype(float)
    return SimpleNamespace(data={"c100": {"xdata": xc, "ydata": yc}},
                           F={"xdata": xf, "ydata": yf},
                           coarse=lambda level=100: coarse), coarse


def test_grid_edges_are_the_8_neighbourhood():
    src, dst, rel = grid_edges(5, 7)
    assert len(src) == 2 * (4 * 7 + 5 * 6 + 2 * 4 * 6)      # h, v and both diagonals
    assert set(map(tuple, rel.astype(int))) == {(a, b) for a in (-1, 0, 1)
                                                for b in (-1, 0, 1)} - {(0, 0)}
    np.testing.assert_array_equal((dst - src), rel[:, 0].astype(int) * 7
                                  + rel[:, 1].astype(int))
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert all((d, s) in pairs for s, d in pairs)            # symmetric


def test_queries_bracket_each_fine_pixel():
    case, coarse = _fake_case()
    nbr, qrel, base = queries(case)
    ny, nx = coarse.shape
    assert nbr.min() >= 0 and nbr.max() < ny * nx
    # offsets to the 4 corners lie within one coarse pixel
    assert np.all(np.abs(qrel) <= 1.0 + 1e-6)
    # a fine pixel sitting on a coarse node reproduces it
    shape = (len(case.F["ydata"]), len(case.F["xdata"]))
    np.testing.assert_allclose(base.reshape(shape)[::4, ::4], coarse, rtol=1e-5)


def test_top_area_mask_fraction():
    score = np.random.default_rng(1).random((80, 90))
    assert top_area_mask(score, 0.3).mean() == pytest.approx(0.3, abs=0.01)


# ── PyTorch part ───────────────────────────────────────────────────────────────

def _example(seed=0, ny=16, nx=18, ratio=4):
    """A synthetic example where the label depends on a SECOND, clean channel:
    the noisy target alone (the bilinear baseline) cannot predict it well."""
    rng = np.random.default_rng(seed)
    case, _ = _fake_case(ny, nx, ratio, seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    clean = 5.0 + 40.0 * np.exp(-((yy - ny / 2) ** 2 + (xx - nx / 3) ** 2) / 8.0)
    noisy = rng.poisson(clean / 4.0).astype(float) * 4.0     # target: very noisy
    case.coarse = lambda level=100: noisy
    nbr, qrel, base = queries(case)
    feat = np.zeros((ny * nx, N_FEATURES), np.float32)
    feat[:, 0] = np.log1p(clean.ravel() / np.median(clean))  # informative channel
    feat[:, -3] = np.log1p(noisy.ravel())
    case.coarse = lambda level=100: clean
    truth = queries(case)[2]                                  # smooth fine truth
    y = rng.poisson(truth).astype(np.float32)
    src, dst, rel = grid_edges(ny, nx)
    shape = (len(case.F["ydata"]), len(case.F["xdata"]))
    return {"sample": "SYN", "target": "X", "fine_px": 25, "x": feat, "src": src,
            "dst": dst, "erel": rel, "nbr": nbr, "qrel": qrel, "base": base,
            "rho": np.float32(0.25), "shape": shape, "cshape": (ny, nx), "y": y,
            "valid": np.ones(len(y), bool), "t": np.float32(1.0),
            "scale": np.float32(y.mean() / base.mean())}


def test_untrained_model_is_the_bilinear_baseline():
    torch = pytest.importorskip("torch")
    from ml.model import RateGNN
    from ml.train import predict
    ex = _example()
    torch.manual_seed(0)
    pred = predict(RateGNN(N_FEATURES), ex)
    np.testing.assert_allclose(pred.ravel(), ex["base"], rtol=1e-5, atol=1e-5)


def test_message_passing_ignores_node_order():
    torch = pytest.importorskip("torch")
    from ml.model import RateGNN
    ex = _example()
    torch.manual_seed(0)
    m = RateGNN(N_FEATURES).eval()
    x = torch.from_numpy(ex["x"]); rel = torch.from_numpy(ex["erel"])
    src = torch.from_numpy(ex["src"]).long(); dst = torch.from_numpy(ex["dst"]).long()
    h = m.encode(x, src, dst, rel)
    perm = torch.randperm(x.shape[0]); inv = torch.argsort(perm)
    h2 = m.encode(x[perm], inv[src], inv[dst], rel)          # relabelled graph
    assert torch.allclose(h, h2[inv], atol=1e-5)


def test_training_beats_the_baseline_and_checkpoint_round_trips(tmp_path):
    torch = pytest.importorskip("torch")
    from ml.train import train, predict, save, load
    ex = _example()
    model, hist = train([ex], steps=150, batch=2048, eval_every=50, device="cpu",
                        log=lambda *_: None, block=8)
    assert hist["best_nll"] < hist["base_nll"] - 0.02        # learned the clean channel
    save(model, tmp_path / "m.pt", {"hidden": 32, "layers": 3, "target": "X",
                                    "heldout": "none"})
    again, meta = load(tmp_path / "m.pt")
    np.testing.assert_allclose(predict(again, ex), predict(model, ex), rtol=1e-6)
    assert meta["target"] == "X"


def test_a_diverging_run_returns_the_baseline_not_its_last_step():
    torch = pytest.importorskip("torch")
    from ml.train import train, predict
    ex = _example()
    model, hist = train([ex], steps=40, batch=2048, eval_every=10, lr=0.5,
                        device="cpu", log=lambda *_: None, block=8)
    if not hist["improved"]:                    # lr = 0.5 normally diverges
        np.testing.assert_allclose(predict(model, ex).ravel(), ex["base"],
                                   rtol=1e-5, atol=1e-5)
    assert hist["best_nll"] <= hist["base_nll"]


def test_validation_blocks_cover_the_same_area_at_both_resolutions():
    from ml.data import val_split
    held = {}
    for ratio in (4, 2):                        # 25 um and 50 um over one coarse scan
        case, coarse = _fake_case(ny=40, nx=44, ratio=ratio)
        nbr, _, base = queries(case)
        shape = (len(case.F["ydata"]), len(case.F["xdata"]))
        ex = {"nbr": nbr, "cshape": coarse.shape, "valid": np.ones(len(base), bool)}
        va = val_split(ex, block=8)[1].reshape(shape)
        assert 0.05 < va.mean() < 0.5
        held[ratio] = va[::ratio, ::ratio]      # sampled on the coarse nodes
    np.testing.assert_array_equal(held[4], held[2])


def test_folds_never_contain_the_held_out_sample():
    from ml.data import SCANS
    for test_sample in ("UA1_P1", "UB1_P1", "FP1_P1"):
        train_scans = [s for s, _ in SCANS if s != test_sample]
        assert test_sample not in train_scans and len(set(train_scans)) == 2


def test_bench_reproduces_the_reference_evaluation():
    # utils.bench must stay the protocol of 13_evaluate_strategies.py: the
    # rule plan scored here equals the row that script wrote (tracked CSV)
    import csv
    from pathlib import Path
    from utils.cascade import find_level_file
    if find_level_file("FP1_P1", 100) is None:
        pytest.skip("data not available")
    from utils.bench import Case
    ref = Path(__file__).parent.parent / "outputs" / "evaluation_step2_lines.csv"
    want = next(r for r in csv.DictReader(open(ref))
                if (r["sample"], r["target"], r["strategy"]) ==
                ("FP1_P1", "Zn.Ka", "roi100+harmonic_c"))
    case = Case("FP1_P1", "Zn.Ka")
    got = case.score(*case.rule_plan(1.0), draws=1, boot=20)[0]
    assert got["mse_ratio"] == pytest.approx(float(want["mse_ratio"]), rel=1e-6)
    assert got["area"] == pytest.approx(float(want["area"]), rel=1e-6)
