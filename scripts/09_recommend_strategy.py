"""
Script 09 - Sample descriptor + strategy recommender
=====================================================================
From a coarse scan, compute 3 simple descriptors and recommend which family
of methods is likely to work best.

Descriptors:
  - sparsity      : fraction of pixels above the Otsu threshold (sample/total)
  - concentration : signal_top10pct / signal_total (how concentrated the signal is)
  - dynamic_range : log10(max/median + 1)         (width of the distribution)

Heuristic recommendation rules:
  - sample_frac < 0.30              -> Spatial ROI (sparse sample, high possible gain)
  - 0.30 < sample_frac < 0.70       -> Combined morpho + temporal
  - sample_frac > 0.70              -> Pure temporal (spatial capped)
  - concentration > 0.60            -> + binary dwell high/low
  - dynamic_range > 2.5             -> + log strategy when linear would saturate

Usage:
    python scripts/09_recommend_strategy.py
    python scripts/09_recommend_strategy.py --coarse data/...250um...hdf5
    python scripts/09_recommend_strategy.py --batch UA1_P1 UB1_P1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map
from utils.roi_utils import _thresh_otsu
from utils.paths import find_coarse, find_first, require, PROJECT_ROOT, DATA_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Sample descriptor + strategy recommender")
    dc = find_coarse()
    p.add_argument("--coarse", default=str(dc) if dc else None,
                   help="Coarse HDF5 (auto-detected from data/ if omitted)")
    p.add_argument("--batch", nargs="+", default=None,
                   help="Run on multiple sample stems (e.g. UA1_P1 UB1_P1 FP1_P1)")
    p.add_argument("--res-um", default=250, type=int, help="Coarse resolution [um]")
    return p.parse_args()


def coarse_path_for(sample, res_um):
    """Find the coarse HDF5 for a named sample stem (UA1_P1 etc.)."""
    return find_first(f"*{sample}*{res_um}um*.hdf5")


def describe_sample(coarse_path):
    """Compute 3 descriptors on the coarse scan."""
    data = load_xrf(coarse_path)
    comp, used, _ = get_composite_map(data)
    x = comp.ravel().astype(float)

    # Sparsity: fraction above Otsu (on clipped data to avoid bright outliers dominating)
    clip = np.percentile(x, 95)
    otsu_low = _thresh_otsu(np.minimum(x, clip))
    sparsity_strict = float(np.mean(x > otsu_low))   # above the matrix threshold

    # An alternative: Otsu_lower (where vide/matrix split would be)
    limit = np.percentile(x, 50)
    x_low = x[x <= limit]
    otsu_floor = _thresh_otsu(x_low) if len(x_low) > 20 else float(np.percentile(x, 5))
    sample_frac = float(np.mean(x > otsu_floor))   # any matter at all

    # Concentration: signal in top 10% / total signal
    top_thr = np.quantile(x, 0.90)
    sig_top = x[x > top_thr].sum()
    concentration = float(sig_top / x.sum()) if x.sum() > 0 else 0

    # Dynamic range
    med = np.median(x)
    dyn_range = float(np.log10(x.max() / max(med, 1) + 1))

    return {
        "path":            str(coarse_path),
        "shape":           comp.shape,
        "sparsity":        sparsity_strict,
        "sample_frac":     sample_frac,
        "concentration":   concentration,
        "dynamic_range":   dyn_range,
        "median":          float(med),
        "max":             float(x.max()),
        "otsu_low":        otsu_low,
        "otsu_floor":      otsu_floor,
    }


def recommend(desc):
    """Return a list of recommended strategies given a descriptor dict."""
    reco = []
    s = desc["sparsity"]
    f = desc["sample_frac"]
    c = desc["concentration"]
    d = desc["dynamic_range"]

    # Primary recommendation — based on sample_frac (occupied area), NOT sparsity:
    # sparsity says "where are the bright pixels" but doesn't say if they are
    # CLUSTERED (ROI works) or DISPERSED (subpixel averaging → ROI fails on coarse).
    # sample_frac is the fraction of zones that contain matter — direct measure
    # of spatial speedup potential.
    if f < 0.30:
        primary = ("ROI spatial (script 03/06 ROI)",
                   f"Sample is sparse ({f*100:.0f}% area occupied) -> "
                   f"big spatial gain possible")
    elif f < 0.70:
        primary = ("Morpho + temporal combined (script 08)",
                   f"Medium density ({f*100:.0f}% area occupied) -> combine "
                   f"spatial mask (gain {(1-f)*100:.0f}% area) and temporal dwell")
    else:
        primary = ("Temporal adaptive (script 07)",
                   f"Sample is dense ({f*100:.0f}% area occupied) -> spatial gain "
                   f"capped at ~{1/f:.2f}x. Use dwell allocation for SNR gain "
                   f"at equal time")

    reco.append(primary)

    # Secondary: dwell strategy
    if c > 0.60:
        reco.append(("dwell binary (high vs low)",
                     f"Signal concentrated ({c*100:.0f}% in top-10% pixels) -> "
                     "binary allocation favored"))
    elif d > 2.5:
        reco.append(("dwell log",
                     f"Wide dynamic range ({d:.1f} decades) -> log scaling avoids saturation"))
    else:
        reco.append(("dwell linear",
                     f"Smooth distribution (concentration {c*100:.0f}%, range {d:.1f}) -> "
                     "linear allocation"))

    return reco


def print_report(desc):
    print(f"\n=== SAMPLE DESCRIPTOR ===")
    print(f"File           : {Path(desc['path']).name}")
    print(f"Shape          : {desc['shape']}")
    print(f"")
    print(f"  median           = {desc['median']:>8.1f}")
    print(f"  max              = {desc['max']:>8.1f}")
    print(f"  sample_frac      = {desc['sample_frac']*100:>7.1f}%  "
          f"(area where there's matter, threshold={desc['otsu_floor']:.1f})")
    print(f"  sparsity         = {desc['sparsity']*100:>7.1f}%  "
          f"(area above matrix threshold {desc['otsu_low']:.1f})")
    print(f"  concentration    = {desc['concentration']*100:>7.1f}%  "
          f"(signal in top-10% pixels)")
    print(f"  dynamic_range    = {desc['dynamic_range']:>8.2f}  (log10 of max/median)")

    print(f"\n=== RECOMMENDATION ===")
    for name, why in recommend(desc):
        print(f"  -> {name}")
        print(f"     because: {why}")


def main():
    args = parse_args()

    if args.batch:
        for sample in args.batch:
            path = coarse_path_for(sample, args.res_um)
            if path is None or not path.exists():
                print(f"\n[skip] {sample}: no coarse HDF5 found in data/")
                continue
            desc = describe_sample(path)
            print_report(desc)
    else:
        args.coarse = str(require(args.coarse, "coarse HDF5"))
        desc = describe_sample(args.coarse)
        print_report(desc)


if __name__ == "__main__":
    main()
