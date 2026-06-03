"""
Sample descriptors + strategy recommendation — single source of truth.

Extracted from script 09 so that 09 (report) and 10 (production pipeline)
share ONE definition of the descriptors and ONE definition of the decision
rules. Nothing here does plotting or argument parsing.

Descriptors (computed on the coarse composite):
  - sparsity      : fraction of pixels above the Otsu threshold (clipped)
  - sample_frac   : fraction of pixels above the void/matter floor
  - concentration : signal_top10pct / signal_total (how concentrated the signal is)
  - dynamic_range : log10(max/median + 1)
"""

from pathlib import Path

import numpy as np

from .hdf5_reader import load_xrf, get_composite_map
from .roi_utils import _thresh_otsu


# ── Descriptors ───────────────────────────────────────────────────────────────

def describe_composite(comp):
    """
    Compute the descriptor dict from a composite 2D map (no I/O).

    This is the math core shared by script 09 (`describe_sample`) and the
    production pipeline (script 10), which already holds the composite in memory.
    """
    x = comp.ravel().astype(float)

    # Sparsity: fraction above Otsu (on clipped data so bright outliers don't
    # dominate the inter-class variance).
    clip = np.percentile(x, 95)
    otsu_low = _thresh_otsu(np.minimum(x, clip))
    sparsity_strict = float(np.mean(x > otsu_low))

    # sample_frac: fraction above the void/matter floor (Otsu on lower half).
    limit = np.percentile(x, 50)
    x_low = x[x <= limit]
    otsu_floor = _thresh_otsu(x_low) if len(x_low) > 20 else float(np.percentile(x, 5))
    sample_frac = float(np.mean(x > otsu_floor))

    # Concentration: signal in top 10% pixels / total signal.
    top_thr = np.quantile(x, 0.90)
    sig_top = x[x > top_thr].sum()
    concentration = float(sig_top / x.sum()) if x.sum() > 0 else 0.0

    # Dynamic range.
    med = np.median(x)
    dyn_range = float(np.log10(x.max() / max(med, 1) + 1))

    return {
        "shape":         comp.shape,
        "sparsity":      sparsity_strict,
        "sample_frac":   sample_frac,
        "concentration": concentration,
        "dynamic_range": dyn_range,
        "median":        float(med),
        "max":           float(x.max()),
        "otsu_low":      otsu_low,
        "otsu_floor":    otsu_floor,
    }


def describe_sample(coarse_path, channels=None):
    """Load a coarse HDF5 + compute descriptors. Adds 'path' to the dict."""
    data = load_xrf(coarse_path)
    comp, used, _ = get_composite_map(data, channels)
    desc = describe_composite(comp)
    desc["path"] = str(coarse_path)
    return desc


# ── Decision rules (single source) ──────────────────────────────────────────────

def choose_strategy(desc):
    """
    Apply the recommender rules. Returns a dict with machine-readable tokens
    plus human explanations:
        method      : 'roi' | 'combined' | 'temporal'
        dwell       : 'binary' | 'log' | 'linear'
        why_method  : str
        why_dwell   : str

    Primary recommendation is based on sample_frac (occupied area), NOT sparsity:
    sparsity says where the bright pixels are but not whether they are CLUSTERED
    (ROI works) or DISPERSED (subpixel averaging → ROI fails on coarse).
    sample_frac is the fraction of zones containing matter — a direct measure of
    spatial speedup potential.
    """
    f = desc["sample_frac"]
    c = desc["concentration"]
    d = desc["dynamic_range"]

    if f < 0.30:
        method = "roi"
        why_m = f"Sample sparse ({f*100:.0f}% area occupied) -> big spatial gain possible"
    elif f < 0.70:
        method = "combined"
        why_m = (f"Medium density ({f*100:.0f}% area) -> combine spatial mask "
                 f"(gain {(1-f)*100:.0f}% area) and temporal dwell")
    elif c > 0.70:
        method = "combined"
        why_m = (f"Dense ({f*100:.0f}% area) but signal concentrated ({c*100:.0f}%) "
                 f"-> combined still helps")
    else:
        method = "temporal"
        why_m = (f"Dense ({f*100:.0f}% area, conc {c*100:.0f}%) -> spatial capped "
                 f"at ~{1/f:.2f}x. Use dwell allocation for SNR gain at equal time")

    if c > 0.60:
        dwell = "binary"
        why_d = f"Signal concentrated ({c*100:.0f}% in top-10% pixels) -> binary high/low"
    elif d > 2.5:
        dwell = "log"
        why_d = f"Wide dynamic range ({d:.1f} decades) -> log scaling avoids saturation"
    else:
        dwell = "linear"
        why_d = (f"Smooth distribution (conc {c*100:.0f}%, range {d:.1f}) -> "
                 f"linear allocation")

    return {"method": method, "dwell": dwell, "why_method": why_m, "why_dwell": why_d}


# Maps the machine tokens to the verbose, script-referencing labels used in the
# script-09 report.
_METHOD_LABEL = {
    "roi":      "ROI spatial (script 03/06 ROI)",
    "combined": "Morpho + temporal combined (script 08)",
    "temporal": "Temporal adaptive (script 07)",
}
_DWELL_LABEL = {
    "binary": "dwell binary (high vs low)",
    "log":    "dwell log",
    "linear": "dwell linear",
}


def recommend(desc):
    """
    Human-readable recommendation list `[(name, why), ...]`, built ON TOP of
    `choose_strategy` so the threshold rules live in exactly one place.
    """
    s = choose_strategy(desc)
    return [
        (_METHOD_LABEL[s["method"]], s["why_method"]),
        (_DWELL_LABEL[s["dwell"]],   s["why_dwell"]),
    ]
