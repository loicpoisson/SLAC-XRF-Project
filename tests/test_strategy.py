"""
Synthetic (data-free) tests for the recommender rules in utils.strategy.

These exercise every branch of choose_strategy on constructed descriptor dicts,
so the threshold logic is covered even on CI/Colab where the HDF5 data (and the
golden tests in test_strategy_golden.py) are absent.
"""

from utils.strategy import choose_strategy


def _desc(sample_frac, concentration, dynamic_range):
    return {"sample_frac": sample_frac,
            "concentration": concentration,
            "dynamic_range": dynamic_range}


# ── method rule (based on sample_frac, with the concentration override) ────────

def test_sparse_is_roi():
    assert choose_strategy(_desc(0.2, 0.5, 2.0))["method"] == "roi"


def test_medium_density_is_combined():
    assert choose_strategy(_desc(0.5, 0.5, 2.0))["method"] == "combined"


def test_dense_diffuse_is_temporal():
    # dense AND not concentrated -> spatial capped -> temporal
    assert choose_strategy(_desc(0.85, 0.40, 2.0))["method"] == "temporal"


def test_dense_concentrated_is_combined():
    # the c>0.70 override: dense but concentrated -> combined still helps
    assert choose_strategy(_desc(0.85, 0.75, 2.0))["method"] == "combined"


# ── dwell rule ─────────────────────────────────────────────────────────────────

def test_high_concentration_gives_binary():
    assert choose_strategy(_desc(0.5, 0.65, 2.0))["dwell"] == "binary"


def test_wide_dynamic_range_gives_log():
    assert choose_strategy(_desc(0.5, 0.50, 3.0))["dwell"] == "log"


def test_smooth_gives_linear():
    assert choose_strategy(_desc(0.5, 0.50, 1.5))["dwell"] == "linear"


def test_thresholds_are_boundaries():
    # exactly at 0.30 is NOT < 0.30 -> not roi; exactly 0.70 is not < 0.70
    assert choose_strategy(_desc(0.30, 0.5, 2.0))["method"] != "roi"
    assert choose_strategy(_desc(0.70, 0.5, 2.0))["method"] == "temporal"
