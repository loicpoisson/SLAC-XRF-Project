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
from utils.strategy import describe_sample, recommend
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
