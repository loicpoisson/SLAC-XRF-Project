"""
Script 06 - Pareto sweep: scan several methods and parameters
====================================================================
For each (method, parameter), compute signal_captured and speedup.
Plot the Pareto signal vs speedup curve => the main figure of the report.

Methods tested:
  - ROI detection (threshold_map + bounding boxes + margin) : k = 0.3, 0.5, 1.0, 1.5
  - Morpho sample mask (Otsu_lower)                          : level = 10, 25, 40, 60, 75

Outputs:
  - CSV     : outputs/<sample>_pareto.csv
  - Figure  : outputs/<sample>_pareto.png

Usage:
    python scripts/06_pareto_sweep.py
    python scripts/06_pareto_sweep.py --sample UB1_P1
    python scripts/06_pareto_sweep.py --sample UA1_P1 UB1_P1 FP1_P1
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map
from utils.roi_utils import (sample_mask, threshold_map, label_rois,
                              get_bounding_boxes, group_rois, add_margin)
from utils.validation_utils import (project_mask, compute_validation_metrics,
                                     load_and_compose, estimate_travel_overhead)
from utils.paths import find_first, PROJECT_ROOT, DATA_DIR

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Default sweep parameters
ROI_K_VALUES     = [0.3, 0.5, 1.0, 1.5]
MORPHO_LEVELS    = [10, 25, 40, 60, 75]

# Coarse and fine resolution for the sweep
COARSE_PX_UM = 250
FINE_PX_UM   = 25


def parse_args():
    p = argparse.ArgumentParser(description="Pareto sweep over ROI/morpho methods")
    p.add_argument("--samples", nargs="+", default=["UA1_P1"],
                   help="Sample stems, e.g. UA1_P1 UB1_P1 FP1_P1")
    p.add_argument("--dwell",   default=10.0, type=float)
    p.add_argument("--fine-px", default=0.025, type=float,
                   help="Fine pixel size [mm] used for ROI margin")
    p.add_argument("--setup-ms", default=500.0, type=float,
                   help="Per-region overhead (positioning, triggering) [ms]")
    p.add_argument("--fine-um",  default=25, type=int,
                   help="Fine resolution to use as ground truth [um] (25 or 50)")
    return p.parse_args()


def coarse_path(sample, px_um=COARSE_PX_UM):
    """Find a coarse HDF5 for the given sample stem and resolution."""
    return find_first(f"*{sample}*{px_um}um*.hdf5") or (
        DATA_DIR / f"SMW_{sample}_{px_um}um_10ms_12000_0_001.hdf5"
    )


def fine_path(sample, px_um=FINE_PX_UM):
    """Find a fine HDF5 for the given sample stem and resolution."""
    return find_first(f"*{sample}*{px_um}um*.hdf5") or (
        DATA_DIR / f"SMW_{sample}_{px_um}um_10ms_12000_0_001.hdf5"
    )


def fine_path_with_fallback(sample, px_um):
    """Glob-based fine path lookup (returns whatever is found first)."""
    m = find_first(f"*{sample}*{px_um}um*.hdf5")
    if m is not None:
        return m
    # last-resort guesses, in case the glob couldn't match
    candidates = [
        DATA_DIR / f"SMW_{sample}_{px_um}um_10ms_12000_0_001.hdf5",
        DATA_DIR / f"SMW_{sample}_{px_um}um_25ms_12000_0_001.hdf5",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # let downstream skip handle missing


def mask_from_roi_detection(coarse_comp, coarse_data, k, fine_px_mm,
                             dwell_ms, n_fine=5):
    """Build a coarse-grid mask from ROI detection (threshold + boxes + margin)."""
    bin_mask, _, _ = threshold_map(coarse_comp, method="auto", k=k)
    labeled, _ = label_rois(bin_mask, min_pixels=1)
    boxes = get_bounding_boxes(labeled, coarse_data["xdata"], coarse_data["ydata"])
    if len(boxes) == 0:
        return np.zeros_like(coarse_comp, dtype=bool)
    boxes = group_rois(boxes, dwell_ms=dwell_ms, dx=fine_px_mm, dy=fine_px_mm)
    boxes_m = add_margin(boxes, coarse_px_mm=coarse_data["dx"],
                         fine_px_mm=fine_px_mm, n_fine=n_fine)

    # Project boxes back into a coarse-grid binary mask
    mask = np.zeros_like(coarse_comp, dtype=bool)
    xc, yc = coarse_data["xdata"], coarse_data["ydata"]
    for b in boxes_m:
        x0, x1, y0, y1 = b["mm"]
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        j0 = np.searchsorted(np.sort(xc), xmin)
        j1 = np.searchsorted(np.sort(xc), xmax)
        i0 = np.searchsorted(np.sort(yc), ymin)
        i1 = np.searchsorted(np.sort(yc), ymax)
        # use range of indices (assume monotonic)
        ix_lo, ix_hi = min(j0, j1), max(j0, j1)
        iy_lo, iy_hi = min(i0, i1), max(i0, i1)
        mask[iy_lo:iy_hi+1, ix_lo:ix_hi+1] = True
    return mask


def sweep_sample(sample, dwell_ms, fine_px_mm, setup_ms, fine_um=25):
    """Run the sweep on one sample, return a list of dicts."""
    cpath = coarse_path(sample)
    fpath = fine_path_with_fallback(sample, fine_um)
    if not cpath.exists() or not fpath.exists():
        print(f"  [skip] missing files for {sample}: {cpath.name} or {fpath.name}")
        return []

    print(f"\n=== SAMPLE {sample} ===")
    print(f"Coarse: {cpath.name}")
    coarse, coarse_comp, coarse_used, _ = load_and_compose(cpath)
    print(f"  shape {coarse['mapdata'].shape}, pixel {coarse['dx']*1000:.0f}um")

    print(f"Fine  : {fpath.name}")
    fine, fine_comp, _, _ = load_and_compose(fpath, channels=coarse_used)
    print(f"  shape {fine['mapdata'].shape}, pixel {fine['dx']*1000:.0f}um")

    results = []

    def _measure(mask_f, label):
        tov = estimate_travel_overhead(mask_f, fine["xdata"], fine["ydata"],
                                       setup_ms=setup_ms)
        m = compute_validation_metrics(
            mask_f, fine_comp, fine_dwell_ms=dwell_ms,
            travel_overhead_ms=tov["overhead_ms"],
        )
        m["n_regions"] = tov["n_regions"]
        m["travel_overhead_ms"] = tov["overhead_ms"]
        print(f"  {label}: area={m['area_frac']*100:>5.1f}%  "
              f"signal={m['signal_captured']*100:>5.1f}%  "
              f"regions={tov['n_regions']:>4}  "
              f"speedup_naive={m['speedup_no_travel']:>5.2f}x  "
              f"speedup_real={m['speedup']:>5.2f}x")
        return m

    # ── ROI detection sweep ───────────────────────────────────────────────
    for k in ROI_K_VALUES:
        mask_c = mask_from_roi_detection(coarse_comp, coarse,
                                         k=k, fine_px_mm=fine_px_mm,
                                         dwell_ms=dwell_ms)
        mask_f = project_mask(mask_c, coarse["xdata"], coarse["ydata"],
                              fine["xdata"], fine["ydata"])
        m = _measure(mask_f, f"ROI k={k:>4}")
        m.update({"sample": sample, "method": "ROI",
                  "param": f"k={k}", "param_value": k})
        results.append(m)

    # ── Morpho sample mask sweep ──────────────────────────────────────────
    for lvl in MORPHO_LEVELS:
        mask_c, _ = sample_mask(coarse_comp, kernel_px=2,
                                mode="otsu_lower", level=float(lvl))
        mask_f = project_mask(mask_c, coarse["xdata"], coarse["ydata"],
                              fine["xdata"], fine["ydata"])
        m = _measure(mask_f, f"Morpho L={lvl:>2}")
        m.update({"sample": sample, "method": "Morpho",
                  "param": f"level={lvl}", "param_value": float(lvl)})
        results.append(m)

    return results


def save_csv(rows, outpath):
    if not rows:
        return
    fields = ["sample", "method", "param", "param_value",
              "area_frac", "signal_captured",
              "speedup_no_travel", "speedup", "n_regions",
              "travel_overhead_ms", "efficiency",
              "n_in", "n_tot"]
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nCSV saved -> {outpath}")


def plot_pareto(rows, outpath):
    if not rows:
        return
    samples = sorted({r["sample"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = {"ROI": "o", "Morpho": "s"}
    colors  = plt.cm.tab10.colors

    for i, sample in enumerate(samples):
        for method, mk in markers.items():
            pts = [(r["speedup"], r["signal_captured"] * 100, r["param"])
                   for r in rows if r["sample"] == sample and r["method"] == method]
            if not pts:
                continue
            xs, ys, labels = zip(*pts)
            ax.scatter(xs, ys, marker=mk, s=70, color=colors[i % 10],
                       label=f"{sample} {method}", alpha=0.8, edgecolor="k")
            for x, y, lbl in pts:
                ax.annotate(lbl, (x, y), fontsize=7, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")

    ax.axhline(90, color="gray", linestyle=":", alpha=0.5, label="signal=90%")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Speedup (raster / adaptive)")
    ax.set_ylabel("Signal captured [%]")
    ax.set_title("Pareto sweep: signal capture vs speedup\n"
                 "(top-right = good)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Figure saved -> {outpath}")
    plt.show()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_results = []
    for sample in args.samples:
        all_results.extend(sweep_sample(sample, args.dwell, args.fine_px,
                                         args.setup_ms, args.fine_um))

    # Combined output (also useful for single sample)
    tag = "_".join(args.samples)
    save_csv(all_results, OUTPUT_DIR / f"pareto_{tag}.csv")
    plot_pareto(all_results, OUTPUT_DIR / f"pareto_{tag}.png")


if __name__ == "__main__":
    main()
