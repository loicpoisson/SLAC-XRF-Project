"""
Script 09 - Sample descriptor + strategy recommender
=====================================================================
From a coarse scan, compute 3 simple descriptors and recommend which family
of methods is likely to work best.

Descriptors:
  - sparsity      : fraction of pixels above the Otsu threshold (sample/total)
  - concentration : signal_top10pct / signal_total (how concentrated the signal is)
  - dynamic_range : log10(max/median + 1)         (width of the distribution)

Heuristic recommendation rules (single source: utils.strategy.choose_strategy):
  - sample_frac < 0.30                       -> Spatial ROI (sparse, high gain)
  - 0.30 < sample_frac < 0.70                -> Combined morpho + temporal
  - sample_frac > 0.70 AND concentration>0.70 -> Combined (still helps when signal
                                                is concentrated despite density)
  - sample_frac > 0.70 (otherwise)           -> Pure temporal (spatial capped)
  - concentration > 0.60                     -> + binary dwell high/low
  - dynamic_range > 2.5                      -> + log strategy when linear saturates
  - otherwise                                -> + linear dwell

Usage:
    python scripts/09_recommend_strategy.py
    python scripts/09_recommend_strategy.py --coarse data/...250um...hdf5
    python scripts/09_recommend_strategy.py --batch UA1_P1 UB1_P1
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.strategy import describe_sample, recommend, choose_strategy
from utils.cascade import find_level_file
from utils.paths import find_coarse, require, DATA_DIR, PROJECT_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="Sample descriptor + strategy recommender")
    dc = find_coarse()
    p.add_argument("--coarse", default=str(dc) if dc else None,
                   help="Coarse HDF5 (auto-detected from data/ if omitted)")
    p.add_argument("--batch", nargs="+", default=None,
                   help="Run on multiple sample stems (e.g. UA1_P1 UB1_P1 FP1_P1)")
    p.add_argument("--res-um", default=250, type=int, help="Coarse resolution [um]")
    p.add_argument("--plot", action="store_true",
                   help="Save the strategy map (sample_frac x concentration, "
                        "coloured by recommended method) over all/--batch samples")
    return p.parse_args()


def _discover_stems():
    """All sample stems present under data/ (e.g. UA1_P1, FP1_1x1, ...)."""
    stems = set()
    for f in DATA_DIR.rglob("*um_*.hdf5"):
        m = re.match(r"SMW_(.+?)_\d+um_", f.name)
        if m:
            stems.add(m.group(1))
    return sorted(stems)


def plot_strategy_map(samples, res_um=250):
    """Scatter sample_frac x concentration, coloured by recommended method,
    with the recommender's decision regions. Shows 'which strategy when'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"roi": "#2ca02c", "combined": "#ff7f0e", "temporal": "#1f77b4"}
    pts = []
    for s in samples:
        path = coarse_path_for(s, res_um)
        if path is None or not path.exists():
            continue
        d = describe_sample(path)
        pts.append((s, d["sample_frac"], d["concentration"], choose_strategy(d)["method"]))
    if not pts:
        raise SystemExit("No samples found to plot.")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axvspan(0, 0.30, color="#2ca02c", alpha=0.06)
    ax.axvspan(0.30, 0.70, color="#ff7f0e", alpha=0.06)
    ax.axvspan(0.70, 1.0, color="#1f77b4", alpha=0.06)
    ax.axvline(0.30, color="gray", ls=":", lw=1)
    ax.axvline(0.70, color="gray", ls=":", lw=1)
    ax.axhline(0.70, color="gray", ls="--", lw=1)
    seen = set()
    for s, sf, c, meth in pts:
        ax.scatter(sf, c, s=70, color=colors.get(meth, "k"), edgecolor="k", zorder=3,
                   label=meth if meth not in seen else None)
        seen.add(meth)
        ax.annotate(s, (sf, c), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("sample_frac  (occupied area)")
    ax.set_ylabel("concentration  (signal in top-10% pixels)")
    ax.set_title("Recommended strategy per sample\n"
                 "(green<30% ROI | 30-70% combined | >70% temporal, "
                 "or combined if concentration>70%)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(title="method", fontsize=8)
    fig.tight_layout()
    out = PROJECT_ROOT / "outputs" / "report_strategy_map.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}  ({len(pts)} samples)")


def coarse_path_for(sample, res_um):
    """Find the coarse HDF5 for a named sample stem (UA1_P1 etc.)."""
    # Delegate to find_level_file: its `_<px>um_` anchor avoids 50um matching 250um.
    return find_level_file(sample, res_um)


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

    if args.plot:
        plot_strategy_map(args.batch or _discover_stems(), args.res_um)
        return

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
