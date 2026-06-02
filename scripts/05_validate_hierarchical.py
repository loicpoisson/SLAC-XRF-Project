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

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data/Data_May2026"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

# Map "resolution in um" -> file path stem (UA1_P1 series)
FILE_TEMPLATE = "SMW_UA1_P1_{px}um_{dw}ms_12000_0_001.hdf5"
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
    return p.parse_args()


def _nearest_idx(arr, vals):
    order = np.argsort(arr)
    arr_s = arr[order]
    pos   = np.clip(np.searchsorted(arr_s, vals), 0, len(arr_s) - 1)
    left  = np.clip(pos - 1, 0, len(arr_s) - 1)
    choose_left = np.abs(arr_s[left] - vals) < np.abs(arr_s[pos] - vals)
    return order[np.where(choose_left, left, pos)]


def project_mask(mask_src, x_src, y_src, x_dst, y_dst):
    """Project a binary mask from one rectilinear grid to another (nearest)."""
    ix = _nearest_idx(x_src, x_dst)
    iy = _nearest_idx(y_src, y_dst)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    return mask_src[IY, IX]


def find_data_file(sample, px_um):
    """Locate the HDF5 file for a given resolution. Tries several naming conventions."""
    candidates = [
        DATA_DIR / FILE_TEMPLATE.format(px=px_um, dw=RES_TO_DWELL.get(px_um, 10)),
        DATA_DIR / f"SMW_{sample}_{px_um}um_10ms_12000_0_001.hdf5",
        DATA_DIR / f"SMW_{sample}_{px_um}um_25ms_12000_0_001.hdf5",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No HDF5 found for sample={sample} resolution={px_um}um. "
                             f"Tried: {[str(c) for c in candidates]}")


def main():
    args = parse_args()
    levels = list(args.levels)
    if len(levels) < 2:
        raise ValueError("Need at least 2 levels (coarse + fine)")

    print(f"\n========== HIERARCHICAL CASCADE ==========")
    print(f"Sample: {args.sample}")
    print(f"Levels: {' -> '.join(f'{p}um' for p in levels)}")

    # ── Load all levels up front ──────────────────────────────────────────
    scans = {}
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    ref_channels = channels  # will fix to coarse's channels after level 0

    for i, px in enumerate(levels):
        path = find_data_file(args.sample, px)
        print(f"\nLevel {i} ({px}um): loading {path.name}")
        d = load_xrf(path)
        comp, used, excluded = get_composite_map(d, ref_channels)
        if i == 0:
            ref_channels = used     # lock channel set on coarse, reuse for all
            print(f"  Channels used: {used}")
            if excluded:
                print(f"  Excluded (constant): {excluded}")
        scans[px] = {"data": d, "comp": comp}
        print(f"  Shape: {d['mapdata'].shape}  pixel: {d['dx']*1000:.0f}um")

    fine_px      = levels[-1]
    fine         = scans[fine_px]["data"]
    fine_comp    = scans[fine_px]["comp"]
    signal_total = float(fine_comp.sum())

    # ── Iterate cascade ───────────────────────────────────────────────────
    mask = None
    per_level_stats = []
    cumul_pixels    = 0
    cumul_signal    = 0.0   # signal contributed at this level (not used downstream)

    for i, px in enumerate(levels):
        d    = scans[px]["data"]
        comp = scans[px]["comp"]

        if i == 0:
            # First level: sample_mask on full composite
            m, thresh = sample_mask(comp, kernel_px=args.kernel,
                                    mode=args.mode, level=args.level,
                                    verbose=True)
            print(f"\nLevel 0 ({px}um): {args.mode} threshold = {thresh:.1f}")
        elif i < len(levels) - 1:
            # Intermediate: project previous mask -> sample_mask restricted to it
            prev_px = levels[i - 1]
            prev_d  = scans[prev_px]["data"]
            mask_here = project_mask(mask, prev_d["xdata"], prev_d["ydata"],
                                     d["xdata"], d["ydata"])
            m, thresh = sample_mask(comp, kernel_px=args.kernel,
                                    mode=args.mode, level=args.level,
                                    restrict_to=mask_here, verbose=True)
            print(f"\nLevel {i} ({px}um): {args.mode} threshold = {thresh:.1f}")
        else:
            # FINAL level: project previous mask onto fine grid; ROI detection
            # happens on the fine composite below, restricted to this mask.
            prev_px = levels[i - 1]
            prev_d  = scans[prev_px]["data"]
            m = project_mask(mask, prev_d["xdata"], prev_d["ydata"],
                             d["xdata"], d["ydata"])
            thresh = None
            print(f"\nLevel {i} ({px}um, FINAL): projected previous mask onto fine grid")

        n_in     = int(m.sum())
        n_tot    = m.size
        sig_in   = float(comp[m].sum())
        sig_pct  = sig_in / float(comp.sum()) * 100 if comp.sum() > 0 else 0
        print(f"  Pixels in mask : {n_in}/{n_tot}  ({n_in/n_tot*100:.1f}%)")
        print(f"  Signal in mask : {sig_pct:.1f}% of this level's total")

        # Time at this level = n_in pixels * level dwell
        dwell = RES_TO_DWELL.get(px, 10)
        level_time_ms = n_in * dwell
        per_level_stats.append({
            "px":            px,
            "shape":         d["mapdata"].shape,
            "n_in":          n_in,
            "n_tot":         n_tot,
            "area_pct":      n_in / n_tot * 100,
            "signal_in_pct": sig_pct,
            "dwell":         dwell,
            "time_ms":       level_time_ms,
            "threshold":     thresh if i < len(levels) - 1 else None,
        })

        mask = m

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
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {outpath}")
    plt.show()


if __name__ == "__main__":
    main()
