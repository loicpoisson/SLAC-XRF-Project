"""
Script 12 - Generate the (current) report figures
=====================================================================
Produces the two figures the v2 progress report relies on, from the verified
analysis (so they cannot drift from the text):

  1. report_mse_dwell_<sample>.png  — reconstruction MSE ratio (raster/adaptive)
     per dwell strategy, temporal mode at equal time. Shows sqrt (t proportional
     to sqrt(signal)) is the only strategy above the raster line; binary is worst.
  2. report_per_element_<sample>.png — bright-pixel fraction per element channel,
     with the composite's sample_frac for contrast. Shows the composite is dense
     while individual elements are sparse hot spots.

Usage:
    python scripts/12_report_figures.py                 # sample UA1_P1
    python scripts/12_report_figures.py --sample UB1_P1
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless: this script only writes files
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map, list_element_channels
from utils.cascade import find_level_file
from utils.strategy import describe_composite
from utils.dwell import allocate_dwell
from utils.quality import poisson_mse
from utils.validation_utils import project_coarse_to_fine
from utils.paths import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"
T_REF = 10.0
STRATEGIES = ["sqrt", "linear", "log", "binary"]


def fig_mse_by_dwell(sample, fine_um=25, prev_um=50):
    """MSE ratio per dwell strategy (temporal, full frame, equal time)."""
    fp_fine, fp_prev = find_level_file(sample, fine_um), find_level_file(sample, prev_um)
    if fp_fine is None or fp_prev is None:
        print(f"[skip MSE fig] need {fine_um}um and {prev_um}um for {sample}")
        return None
    fine = load_xrf(fp_fine)
    fine_comp, used, _ = get_composite_map(fine)
    prev = load_xrf(fp_prev)
    prev_comp, _, _ = get_composite_map(prev, used)
    # dwell decided from the previous (coarser) level, as the pipeline does
    dwell_signal = project_coarse_to_fine(prev_comp, prev, fine)
    full = np.ones_like(fine_comp, dtype=bool)

    ratios = []
    for strat in STRATEGIES:
        dwell, _ = allocate_dwell(dwell_signal, strat, 1.0, 50.0, mask=full)
        dwell = dwell * (fine_comp.size * T_REF / dwell.sum())   # equal-time budget
        ratios.append(poisson_mse(fine_comp, dwell, full, t_ref=T_REF,
                                  predict="zero")["ratio"])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = ["#2ca02c" if r >= 1 else "#d62728" for r in ratios]
    ax.bar(STRATEGIES, ratios, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="raster (equal time)")
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.02, f"{r:.2f}x", ha="center", fontsize=9)
    ax.set_ylabel("MSE ratio  (raster / adaptive)")
    ax.set_title(f"{sample}: dwell strategy vs reconstruction MSE at equal time\n"
                 f"(>1 = better than raster; sqrt = t ∝ √signal optimum)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / f"report_mse_dwell_{sample}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}  ({dict(zip(STRATEGIES, [round(r,2) for r in ratios]))})")
    return path


def fig_per_element(sample, px=250):
    """Bright-pixel fraction per element vs the composite sample_frac."""
    fp = find_level_file(sample, px)
    if fp is None:
        print(f"[skip per-element fig] no {px}um scan for {sample}")
        return None
    d = load_xrf(fp)
    comp, used, _ = get_composite_map(d)
    comp_frac = describe_composite(comp)["sample_frac"]

    els, fracs = [], []
    for ch in list_element_channels(d):
        arr = d["mapdata"][:, :, d["labels"].index(ch)].astype(float)
        if arr.min() == arr.max():
            continue
        els.append(ch.replace(".Ka", "").replace(".La", ""))
        fracs.append(describe_composite(arr)["sparsity"])
    order = np.argsort(fracs)
    els = [els[i] for i in order]
    fracs = [fracs[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(els)), 4.2))
    ax.bar(els, np.array(fracs) * 100, color="#1f77b4")
    ax.axhline(comp_frac * 100, color="#d62728", ls="--", lw=1.5,
               label=f"composite presence (sample_frac) = {comp_frac*100:.0f}%")
    ax.axhline(30, color="gray", ls=":", lw=1, label="sparse threshold (30%)")
    ax.set_ylabel("bright-pixel fraction  [%]")
    ax.set_title(f"{sample} ({px}um): each element is sparse (hot spots ~10-20%)\n"
                 f"while the composite is dense ({comp_frac*100:.0f}% occupied)")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / f"report_per_element_{sample}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")
    return path


def main():
    p = argparse.ArgumentParser(description="Generate the v2 report figures")
    p.add_argument("--sample", default="UA1_P1")
    args = p.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Report figures for {args.sample}:")
    fig_mse_by_dwell(args.sample)
    fig_per_element(args.sample)


if __name__ == "__main__":
    main()
