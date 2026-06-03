"""
Script 05 - Validation Option 4: hierarchical multi-resolution cascade
========================================================================
Paper-2 style workflow (Boltz-Webb 2026):
  Intermediate levels (e.g. 250, 100, 50 um): progressive refinement of the
    sample footprint via Otsu + morphology. NO ROI detection here.
  Final level (e.g. 25 um): scan inside the refined mask, then fine ROI
    detection (k=1.0 on IQR) on the actually scanned pixels.

At each intermediate level we simulate "scan only inside the previous mask"
by projecting the mask onto the current grid, then re-applying Otsu on the
composite restricted to that mask.

Typical configs:
  --levels 250 25            (1 step: equivalent to 04_validate_morphological)
  --levels 250 100 25        (1 refinement)
  --levels 250 100 50 25     (2 refinements - config C, main)
  --levels 250 50 25         (skip 100)

Usage:
    python scripts/05_validate_hierarchical.py --levels 250 100 50 25
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map
from utils.roi_utils import (sample_mask, threshold_map, label_rois,
                              get_bounding_boxes, group_rois, add_margin)
from utils.validation_utils import project_mask
from utils.cascade import find_level_file, refine_cascade
from utils.plotting import save_and_show

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data/Data_May2026"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

RES_TO_DWELL = {250: 10, 100: 10, 50: 10, 25: 10}   # all 10ms for UA1


def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical multi-resolution validation")
    p.add_argument("--sample",   default="UA1_P1", help="Sample stem (e.g. UA1_P1)")
    p.add_argument("--levels",   nargs="+", type=int, default=[250, 100, 50, 25],
                   help="Resolution cascade in um, finest last (default: 250 100 50 25)")
    p.add_argument("--kernel",   default=2, type=int,
                   help="Morphological kernel size in pixels at each level")
    p.add_argument("--mode",     default="otsu_lower",
                   choices=["otsu_lower", "percentile", "otsu_clip"],
                   help="Sample-mask thresholding mode (default otsu_lower)")
    p.add_argument("--level",    default=50.0, type=float,
                   help="Percentile parameter for the chosen mode (default 50)")
    p.add_argument("--k",        default=1.0, type=float,
                   help="ROI detection sensitivity at FINAL level (default 1.0)")
    p.add_argument("--min-px",   default=1, type=int)
    p.add_argument("--channels", default=None)
    p.add_argument("--no-show",  action="store_true",
                   help="Save figures without opening a window")
    return p.parse_args()


def main():
    args = parse_args()
    levels = list(args.levels)
    if len(levels) < 2:
        raise ValueError("Need at least 2 levels (coarse + fine)")

    print(f"\n========== HIERARCHICAL CASCADE ==========")
    print(f"Sample: {args.sample}")
    print(f"Levels: {' -> '.join(f'{p}um' for p in levels)}")

    # ── Load all levels up front (coarse -> fine ordered list) ────────────
    scans = {}
    scans_list = []
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    ref_channels = channels  # will fix to coarse's channels after level 0

    for i, px in enumerate(levels):
        path = find_level_file(args.sample, px, data_dir=DATA_DIR)
        if path is None:
            raise FileNotFoundError(f"No HDF5 found for sample={args.sample} "
                                     f"resolution={px}um under {DATA_DIR}")
        print(f"\nLevel {i} ({px}um): loading {path.name}")
        d = load_xrf(path)
        comp, used, excluded = get_composite_map(d, ref_channels)
        if i == 0:
            ref_channels = used     # lock channel set on coarse, reuse for all
            print(f"  Channels used: {used}")
            if excluded:
                print(f"  Excluded (constant): {excluded}")
        scans[px] = {"data": d, "comp": comp}
        scans_list.append({"data": d, "comp": comp, "px": px})
        print(f"  Shape: {d['mapdata'].shape}  pixel: {d['dx']*1000:.0f}um")

    fine_px      = levels[-1]
    fine         = scans[fine_px]["data"]
    fine_comp    = scans[fine_px]["comp"]
    signal_total = float(fine_comp.sum())

    # ── Iterate cascade (shared utils.cascade.refine_cascade) ─────────────
    # detect_final=False reproduces the original behavior: the finest level only
    # projects the previous mask (ROI detection happens below, not a re-detection).
    print(f"\n--- refining footprint across levels ---")
    mask, per_level_stats = refine_cascade(
        scans_list, kernel_px=args.kernel, mode=args.mode, level=args.level,
        detect_final=False, verbose=True,
    )

    # Augment generic stats with 05-specific dwell/time + percentage aliases.
    for s in per_level_stats:
        s["area_pct"]      = s["area_frac"] * 100
        s["signal_in_pct"] = s["signal_in_frac"] * 100
        s["dwell"]         = RES_TO_DWELL.get(s["px"], 10)
        s["time_ms"]       = s["n_in"] * s["dwell"]

    # ── Final-level ROI detection (k=1.0) inside the mask ─────────────────
    print(f"\n=== FINAL ROI DETECTION (level {fine_px}um, k={args.k}) ===")
    comp_in_mask = np.where(mask, fine_comp, 0.0)
    # apply threshold_map on full composite but restrict mask intersection
    roi_mask_raw, thresh_roi, method_used = threshold_map(
        comp_in_mask, method="auto", k=args.k,
    )
    roi_mask_final = roi_mask_raw & mask
    labeled, _ = label_rois(roi_mask_final, min_pixels=args.min_px)
    boxes = get_bounding_boxes(labeled, fine["xdata"], fine["ydata"])
    print(f"  ROIs detected at fine level: {len(boxes)}")

    # ── Top-level metrics ─────────────────────────────────────────────────
    n_pix_in        = int(mask.sum())
    n_pix_total     = mask.size
    area_frac       = n_pix_in / n_pix_total
    signal_captured = float(fine_comp[mask].sum()) / signal_total if signal_total else 0
    signal_missed   = 1.0 - signal_captured

    raster_ms       = n_pix_total * RES_TO_DWELL[fine_px]
    total_cascade_ms = sum(s["time_ms"] for s in per_level_stats)
    speedup         = raster_ms / total_cascade_ms if total_cascade_ms else np.inf

    print(f"\n=== CASCADE SUMMARY ===")
    print(f"  {'level':>6}  {'shape':>16}  {'in mask':>10}  {'%area':>7}  {'%signal':>8}  {'time (s)':>9}")
    for s in per_level_stats:
        print(f"  {s['px']:>4}um  {str(s['shape']):>16}  "
              f"{s['n_in']:>10,}  {s['area_pct']:>6.1f}%  {s['signal_in_pct']:>7.1f}%  "
              f"{s['time_ms']/1000:>8.1f}")
    print(f"")
    print(f"  Final signal captured (at {fine_px}um) : {signal_captured*100:.1f}%")
    print(f"  Final pixels scanned                : {area_frac*100:.1f}%")
    print(f"  Raster baseline ({fine_px}um full)        : {raster_ms/1000:.1f} s")
    print(f"  Cascade total time                  : {total_cascade_ms/1000:.1f} s")
    print(f"  REAL SPEEDUP (raster / cascade)     : {speedup:.2f}x")

    # ── Figure: 1 panel per level + final ROI overlay ─────────────────────
    n_panels = len(levels) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    cascade_label = " -> ".join(f"{p}um" for p in levels)
    fig.suptitle(
        f"Option 4 hierarchical: {cascade_label}  |  "
        f"signal captured: {signal_captured*100:.1f}%  |  "
        f"area scanned: {area_frac*100:.1f}%  |  speedup: {speedup:.2f}x",
        fontsize=10,
    )

    for i, px in enumerate(levels):
        ax = axes[i]
        d    = scans[px]["data"]
        comp = scans[px]["comp"]
        s    = per_level_stats[i]
        extent = [d["xdata"].min(), d["xdata"].max(),
                  d["ydata"].max(), d["ydata"].min()]
        vmax = np.percentile(comp, 99.5 if px <= 50 else 99)
        ax.imshow(comp, extent=extent, origin="upper", aspect="equal",
                  cmap="inferno", vmin=0, vmax=vmax)

        # contour of the mask at this level
        if i == 0:
            m_show = sample_mask(comp, kernel_px=args.kernel)[0]
        else:
            # reconstitute the mask projected onto this level for display
            prev_d = scans[levels[i - 1]]["data"]
            # rebuild from per_level_stats - we lost intermediate masks;
            # for display, just use intersection of comp>thresh with prev projected
            # (the mask variable now holds the FINEST level mask)
            # Simpler: project final mask back onto this level
            m_show = project_mask(mask, fine["xdata"], fine["ydata"],
                                  d["xdata"], d["ydata"])

        ax.contour(m_show.astype(float), levels=[0.5], colors="cyan",
                   linewidths=0.8, extent=extent, origin="upper")
        ax.set_title(f"Level {i}: {px}um\n{s['n_in']:,} pix ({s['area_pct']:.1f}%)  |  "
                     f"{s['time_ms']/1000:.1f}s")
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    # last panel: final ROI overlay on fine
    ax = axes[-1]
    extent = [fine["xdata"].min(), fine["xdata"].max(),
              fine["ydata"].max(), fine["ydata"].min()]
    vmax = np.percentile(fine_comp, 99.5)
    ax.imshow(fine_comp, extent=extent, origin="upper", aspect="equal",
              cmap="inferno", vmin=0, vmax=vmax)
    for b in boxes:
        x0, x1, y0, y1 = b["mm"]
        rect = mpatches.Rectangle(
            (min(x0, x1), min(y0, y1)),
            abs(x1 - x0), abs(y1 - y0),
            linewidth=0.6, edgecolor="lime", facecolor="none", alpha=0.8,
        )
        ax.add_patch(rect)
    ax.set_title(f"ROIs detected (k={args.k}): {len(boxes)} boxes")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    label = "_".join(str(p) for p in levels)
    outpath = OUTPUT_DIR / f"{args.sample}_hierarchical_{label}.png"
    save_and_show(fig, outpath, show=not args.no_show)


if __name__ == "__main__":
    main()
