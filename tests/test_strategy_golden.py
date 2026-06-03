"""
Golden-value tests on real data (UA1_P1, FP1_P1). These make the manual
'non-regression' check automatic: if the descriptors or the recommender rules
drift, these fail. Skipped when the data files are not present (e.g. CI/Colab).
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
    assert d["sample_frac"] == pytest.approx(0.774, abs=0.01)
    assert d["concentration"] == pytest.approx(0.595, abs=0.01)
    assert d["dynamic_range"] == pytest.approx(2.84, abs=0.05)


def test_ua1_strategy_is_temporal_log():
    s = choose_strategy(describe_sample(_coarse("UA1_P1")))
    assert s["method"] == "temporal"
    assert s["dwell"] == "log"


def test_fp1_strategy_is_combined():
    s = choose_strategy(describe_sample(_coarse("FP1_P1")))
    assert s["method"] == "combined"


def test_describe_sample_matches_describe_composite():
    # describe_sample must be a thin wrapper over describe_composite.
    from utils.hdf5_reader import load_xrf, get_composite_map
    path = _coarse("UA1_P1")
    comp, _, _ = get_composite_map(load_xrf(path))
    a = describe_composite(comp)
    b = describe_sample(path)
    for k in ("sample_frac", "concentration", "dynamic_range", "sparsity"):
        assert a[k] == pytest.approx(b[k])
