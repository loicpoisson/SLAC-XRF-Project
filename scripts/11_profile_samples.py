"""
Script 11 - Profile every sample by sparsity
=====================================================================
Adaptive scanning only pays off when the sample is SPARSE (small occupied
area): then scanning just the footprint is much faster, or — at equal time —
concentrates far more dwell on the ROI. On dense samples a plain fine raster
is competitive or better.

This script computes the descriptors (utils.strategy) on the coarsest available
scan of every sample found under data/, and ranks them sparsest-first, so you
can see at a glance which samples are good candidates for the adaptive pipeline.

Usage:
    python scripts/11_profile_samples.py
"""

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map, list_element_channels
from utils.strategy import describe_composite, choose_strategy
from utils.cascade import find_level_file
from utils.paths import DATA_DIR, PROJECT_ROOT


def discover_samples():
    """Group HDF5 files by sample stem -> {resolution_um: path}."""
    samples = collections.defaultdict(dict)
    for f in sorted(DATA_DIR.rglob("*.hdf5")):
        m = re.match(r"SMW_(.+?)_(\d+)um_", f.name)
        if not m:
            continue
        stem, px = m.group(1), int(m.group(2))
        samples[stem].setdefault(px, f)          # keep first (10ms sorts first)
    if not samples:
        raise SystemExit(f"No SMW_*um_*.hdf5 samples found under {DATA_DIR}")
    return samples


def profile_per_element(samples):
    """
    For each sample's coarse scan, profile EACH element channel separately.
    The composite (all elements summed) can look dense while a single trace
    element is sparse and localized -> that element is where adaptive wins.
    """
    # For per-element ADAPTIVE potential the right measure is the bright-pixel
    # fraction ('sparsity' = fraction above the matrix threshold), not sample_frac
    # ('any presence'): adaptive targets the hot spots, not the diffuse presence.
    hits = []   # (sample, element, sparsity, concentration)
    for stem, byres in samples.items():
        px = 250 if 250 in byres else max(byres)
        d = load_xrf(byres[px])
        per_el = []
        for ch in list_element_channels(d):
            arr = d["mapdata"][:, :, d["labels"].index(ch)].astype(float)
            if arr.min() == arr.max():
                continue                          # constant channel, no info
            de = describe_composite(arr)
            per_el.append((ch, de["sparsity"], de["concentration"]))
            if de["sparsity"] < 0.30:
                hits.append((stem, ch, de["sparsity"], de["concentration"]))
        per_el.sort(key=lambda t: t[1])           # most concentrated (low sparsity) first
        spk = ", ".join(f"{c}={sp*100:.0f}%" for c, sp, _ in per_el[:4])
        print(f"  {stem:<14} ({px}um): bright-pixel fraction per element | "
              f"lowest: {spk}")

    print("\n=== (sample, element) with bright-pixel fraction < 30% "
          "[hot spots concentrated -> adaptive could target them] ===")
    if not hits:
        print("  none.")
    else:
        hits.sort(key=lambda t: t[2])
        print(f"  {'sample':<14}{'element':<10}{'bright_frac':>12}{'concentr':>10}")
        for stem, ch, sp, c in hits:
            print(f"  {stem:<14}{ch:<10}{sp*100:>10.1f}%{c*100:>9.1f}%")


def profile_composite(samples):
    rows = []
    for stem, byres in samples.items():
        px = 250 if 250 in byres else max(byres)   # prefer a 250um coarse scan
        try:
            d = load_xrf(byres[px])
            comp, _, _ = get_composite_map(d)
            de = describe_composite(comp)
            st = choose_strategy(de)
            rows.append([stem, px, de["sample_frac"], de["sparsity"],
                         de["concentration"], de["dynamic_range"],
                         st["method"], comp.shape])
        except Exception as e:                       # noqa: BLE001 - report, don't abort
            rows.append([stem, px, None, None, None, None, f"ERR {e}"[:40], None])

    rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 9))

    print(f"\nSamples under {DATA_DIR} (coarsest scan, sparsest first):\n")
    print(f"{'sample':<15}{'coarse':<7}{'sample_frac':>12}{'sparsity':>10}"
          f"{'concentr':>10}{'dyn_rng':>9}  {'reco':<9}shape")
    print("-" * 96)
    for stem, px, sf, sp, c, dr, meth, shape in rows:
        if sf is None:
            print(f"{stem:<15}{str(px)+'um':<7}  {meth}")
            continue
        flag = ("  <-- SPARSE (adaptive wins)" if sf < 0.30
                else "  <- medium" if sf < 0.70 else "")
        print(f"{stem:<15}{str(px)+'um':<7}{sf*100:>10.1f}%{sp*100:>9.1f}%"
              f"{c*100:>9.1f}%{dr:>9.2f}  {meth:<9}{shape}{flag}")
    print("\nsample_frac = fraction of area containing matter "
          "(<30% sparse, 30-70% medium, >70% dense).")


def plot_per_element(sample, px=250):
    """Bar chart of bright-pixel fraction per element vs the composite presence.
    The figure the v2 report uses to show 'composite dense, elements sparse'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fp = find_level_file(sample, px)
    if fp is None:
        raise SystemExit(f"No {px}um scan for {sample} under {DATA_DIR}")
    d = load_xrf(fp)
    comp, _, _ = get_composite_map(d)
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
    fracs = [fracs[i] * 100 for i in order]

    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(els)), 4.2))
    ax.bar(els, fracs, color="#1f77b4")
    ax.axhline(comp_frac * 100, color="#d62728", ls="--", lw=1.5,
               label=f"composite presence (sample_frac) = {comp_frac*100:.0f}%")
    ax.axhline(30, color="gray", ls=":", lw=1, label="sparse threshold (30%)")
    ax.set_ylabel("bright-pixel fraction  [%]")
    ax.set_title(f"{sample} ({px}um): each element is sparse (hot spots ~10-20%)\n"
                 f"while the composite is dense ({comp_frac*100:.0f}% occupied)")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    out = PROJECT_ROOT / "outputs" / f"report_per_element_{sample}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


def main():
    p = argparse.ArgumentParser(description="Profile samples by sparsity")
    p.add_argument("--per-element", action="store_true",
                   help="Break each sample down per element channel (the composite "
                        "can be dense while a single element is sparse)")
    p.add_argument("--plot", action="store_true",
                   help="Save the per-element bright-fraction figure for --sample")
    p.add_argument("--sample", default="UA1_P1",
                   help="Sample stem for --plot (default UA1_P1)")
    args = p.parse_args()
    if args.plot:
        plot_per_element(args.sample)
        return
    samples = discover_samples()
    if args.per_element:
        print(f"\nPer-element sparsity under {DATA_DIR} (coarsest scan):\n")
        profile_per_element(samples)
    else:
        profile_composite(samples)


if __name__ == "__main__":
    main()
