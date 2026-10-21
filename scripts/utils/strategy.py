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
        dwell       : 'sqrt' (binary | log | linear only via overrides)
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

    # Dwell: sqrt is the equal-time Poisson-MSE optimum (utils.quality). The
    # old concentration/dynamic-range heuristics (binary/linear/log) all score
    # < 1x vs raster on every sample (2026-10 review), so they are only
    # available as explicit overrides.
    dwell = "sqrt"
    why_d = (f"t ~ sqrt(signal) minimises Poisson MSE at equal time "
             f"(conc {c*100:.0f}%, range {d:.1f})")

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
    "sqrt":   "dwell sqrt (MSE-optimal)",
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


# ── Element-targeted spatial rule ─────────────────────────────────────────────

# Measured 2026-10 on UA1/UB1/FP1 (Fe, Zn, Mn, Cu; equal total time including
# the earlier levels): element ROIs detected at 250um never pay off for a 25um
# scan (MSE ratio 0.01-0.5x: sub-pixel structure lost in the skipped cells),
# while ROIs detected at 100um do in most cases (Fe UB1 1.4x, Mn 1.3-2.1x).
MAX_DETECT_RATIO = 4.0


def spatial_allowed(detect_px_um, final_px_um, max_ratio=MAX_DETECT_RATIO):
    """True if ROIs found at detect_px_um are fine enough to skip area at final_px_um."""
    return detect_px_um <= max_ratio * final_px_um


def choose_element_strategy(detect_px_um, final_px_um, max_ratio=MAX_DETECT_RATIO):
    """Strategy for a single target element (--target in script 10)."""
    if spatial_allowed(detect_px_um, final_px_um, max_ratio):
        return {"method": "roi", "dwell": "sqrt",
                "why_method": (f"Element ROIs at {detect_px_um}um are <= "
                               f"{max_ratio:g}x the {final_px_um}um target -> "
                               f"scan only the element's hot spots"),
                "why_dwell": "sqrt inside the ROIs (Poisson-MSE optimum)"}
    return {"method": "temporal", "dwell": "sqrt",
            "why_method": (f"{detect_px_um}um is > {max_ratio:g}x the "
                           f"{final_px_um}um target: coarse ROIs miss sub-pixel "
                           f"hot spots -> full frame, sqrt dwell"),
            "why_dwell": "sqrt over the full frame (Poisson-MSE optimum)"}
