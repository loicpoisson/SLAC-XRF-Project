"""
Golden-value tests on real data (UA1_P1, FP1_P1). These make the manual
'non-regression' check automatic: if the descriptors or the recommender rules
drift, these fail. Skipped when the data files are not present (e.g. CI/Colab).

Values regenerated 2026-10 after the loader fixes (channel planes offset by 2,
counts rescaled by the TIME channel); the old values came from the wrong
planes plus the ~65x-dwell column 0.
"""

import pytest

from utils.strategy import describe_sample, choose_strategy, describe_composite
from utils.cascade import find_level_file


def _coarse(sample):
    p = find_level_file(sample, 250)
    if p is None:
        pytest.skip(f"{sample} 250um data not available")
    return p


def test_ua1_descriptors_golden():
    d = describe_sample(_coarse("UA1_P1"))
    assert d["sample_frac"] == pytest.approx(0.762, abs=0.01)
    assert d["concentration"] == pytest.approx(0.283, abs=0.01)
    assert d["dynamic_range"] == pytest.approx(1.63, abs=0.05)


def test_ua1_strategy_is_temporal_linear():
    s = choose_strategy(describe_sample(_coarse("UA1_P1")))
    assert s["method"] == "temporal"
    assert s["dwell"] == "linear"


def test_fp1_strategy_is_temporal_log():
    s = choose_strategy(describe_sample(_coarse("FP1_P1")))
    assert s["method"] == "temporal"
    assert s["dwell"] == "log"


def test_describe_sample_matches_describe_composite():
    # describe_sample must be a thin wrapper over describe_composite.
    from utils.hdf5_reader import load_xrf, get_composite_map
    path = _coarse("UA1_P1")
    comp, _, _ = get_composite_map(load_xrf(path))
    a = describe_composite(comp)
    b = describe_sample(path)
    for k in ("sample_frac", "concentration", "dynamic_range", "sparsity"):
        assert a[k] == b[k]          # thin wrapper -> byte-identical, not just approx
