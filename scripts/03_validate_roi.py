"""
Script 03 - Validation: coarse ROIs vs fine scan (ground truth)
===================================================================
Input  : coarse HDF5 (e.g. 250 um) + fine HDF5 (e.g. 25 um) of the same sample
Output : validation metrics + comparison figure saved into outputs/

Scientific question:
  If we had only scanned the ROIs detected from the coarse 250 um, how much
  of the total XRF signal would we have captured? And how much time saved
  compared to a full raster fine scan?

Main metrics:
  1. signal_captured = sum(fine[in_ROI]) / sum(fine_total)   (target >90%)
  2. area_fraction   = ROI_area / total_area                  (target <30%)
  3. real_speedup    = 1 / area_fraction                      (time ratio)

Usage:
    python scripts/03_validate_roi.py
    python scripts/03_validate_roi.py --coarse data/...250um... --fine data/...25um...
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map, list_element_channels
from utils.roi_utils import (threshold_map, label_rois, get_bounding_boxes,
                              group_rois, add_margin)
from utils.validation_utils import (compute_validation_metrics,
                                     estimate_travel_overhead)
from utils.cli import add_io_args
from utils.plotting import save_and_show
from utils.paths import require, PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Validate coarse ROIs against fine ground-truth scan")
    add_io_args(p)
    p.add_argument("--method",   default="auto",
                   choices=["auto", "mad", "trimmed", "iqr", "sigma"])
    p.add_argument("--k",        default=1.0,  type=float)
    p.add_argument("--min-px",   default=1,    type=int)
    p.add_argument("--dwell",    default=10.0, type=float,
                   help="Fine scan dwell per pixel [ms] (default 10, same "
                        "baseline as scripts 04/06/07/08/10)")
    p.add_argument("--fine-px",  default=0.025, type=float)
    p.add_argument("--setup-ms", default=500.0, type=float)
    p.add_argument("--n-fine",   default=5,    type=int,
                   help="Fine-pixel margin added on each side of ROIs")
    return p.parse_args()


def build_roi_mask(fine_data, boxes_margin):
    """
    For each pixel of the fine scan, mark True if it falls inside any
    ROI bounding box (in mm coordinates).

    Returns
    -------
    mask : (ny_fine, nx_fine) bool array
    """
    xf, yf = fine_data["xdata"], fine_data["ydata"]
    XF, YF = np.meshgrid(xf, yf)
    mask = np.zeros(XF.shape, dtype=bool)

    for b in boxes_margin:
        x0, x1, y0, y1 = b["mm"]
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        inside = (XF >= xmin) & (XF <= xmax) & (YF >= ymin) & (YF <= ymax)
        mask |= inside

    return mask


def main():
    args = parse_args()
    args.coarse = str(require(args.coarse, "coarse HDF5"))
    args.fine   = str(require(args.fine,   "fine HDF5"))

    # ── 1. Coarse scan: ROI detection ─────────────────────────────────────
    print(f"\n=== COARSE SCAN ===")
    print(f"Loading {args.coarse}")
    coarse = load_xrf(args.coarse)
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    coarse_comp, coarse_used, _ = get_composite_map(coarse, channels)
    print(f"  Shape  : {coarse['mapdata'].shape}")
    print(f"  Pixel  : {coarse['dx']*1000:.0f} x {coarse['dy']*1000:.0f} um")

    mask_thr, thresh, method = threshold_map(coarse_comp, method=args.method, k=args.k)
    print(f"  Threshold: {thresh:.1f}  (method={method}, k={args.k})")

    labeled, _ = label_rois(mask_thr, min_pixels=args.min_px)
    boxes = get_bounding_boxes(labeled, coarse["xdata"], coarse["ydata"])
    boxes = group_rois(boxes, dwell_ms=args.dwell, dx=args.fine_px,
                       dy=args.fine_px, setup_ms=args.setup_ms,
                       size_px_mm=(coarse["dx"], coarse["dy"]))
    boxes_margin = add_margin(boxes, coarse_px_mm=coarse["dx"],
                              fine_px_mm=args.fine_px, n_fine=args.n_fine)
    print(f"  ROIs detected: {len(boxes_margin)}")

    # ── 2. Fine scan: ground truth ───────────────────────────────────────
    print(f"\n=== FINE SCAN (ground truth) ===")
    print(f"Loading {args.fine}")
    fine = load_xrf(args.fine)
    # Use SAME channels as coarse for fair comparison
    fine_comp, fine_used, _ = get_composite_map(fine, coarse_used)
    print(f"  Shape  : {fine['mapdata'].shape}")
    print(f"  Pixel  : {fine['dx']*1000:.0f} x {fine['dy']*1000:.0f} um")
    print(f"  Channels used (same as coarse): {fine_used}")

    # ── 3. Build ROI mask on fine grid ───────────────────────────────────
    roi_mask = build_roi_mask(fine, boxes_margin)

    # ── 4. Validation metrics (shared implementation: same definitions and
    #      travel model as scripts 04/06/07/08/10, so speedups are comparable)
    tov = estimate_travel_overhead(roi_mask, fine["xdata"], fine["ydata"],
                                   setup_ms=args.setup_ms)
    m = compute_validation_metrics(roi_mask, fine_comp, fine_dwell_ms=args.dwell,
                                   travel_overhead_ms=tov["overhead_ms"])
    area_frac       = m["area_frac"]
    signal_captured = m["signal_captured"]
    signal_missed   = 1.0 - signal_captured
    speedup         = m["speedup"]

    # Per-pixel: where is the missed signal concentrated? (03-specific diagnostic)
    missed_map = fine_comp.copy()
    missed_map[roi_mask] = 0
    top_missed = float(missed_map.max())
    median_missed = float(np.median(missed_map[missed_map > 0])) if (missed_map > 0).any() else 0.0
    median_in_roi = float(np.median(fine_comp[roi_mask])) if m["n_in"] > 0 else 0.0

    print(f"\n=== VALIDATION METRICS ===")
    print(f"  Fine scan pixels       : {m['n_tot']:,}")
    print(f"  Pixels inside ROIs     : {m['n_in']:,}  ({area_frac*100:.1f}%)")
    print(f"")
    print(f"  Total XRF signal       : {m['signal_total']:.3e}")
    print(f"  Signal captured by ROIs: {m['signal_in']:.3e}  ({signal_captured*100:.1f}%)")
    print(f"  Signal MISSED          : {m['signal_total'] - m['signal_in']:.3e}  ({signal_missed*100:.1f}%)")
    print(f"")
    print(f"  Median intensity in ROI : {median_in_roi:.1f}")
    print(f"  Median missed intensity : {median_missed:.1f}   (low = nothing important missed)")
    print(f"  Max missed intensity    : {top_missed:.1f}     (high = at least one real particle missed)")
    print(f"")
    print(f"  Raster fine time   : {m['raster_ms']/1000:.0f} s  ({m['raster_ms']/60000:.1f} min)")
    print(f"  Adaptive time      : {m['adaptive_ms']/1000:.0f} s  ({m['adaptive_ms']/60000:.1f} min)")
    print(f"    travel + setup   : {tov['overhead_ms']/1000:.1f} s  over {tov['n_regions']} regions")
    print(f"  REAL SPEEDUP       : {speedup:.1f}x  (no-travel: {m['speedup_no_travel']:.1f}x)")
    print(f"  Signal efficiency  : {m['efficiency']:.1f}x  (ratio signal_captured / area_scanned)")

    # ── 5. Figure: ground truth + ROIs + what we'd see ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Validation: coarse ROIs ({coarse['dx']*1000:.0f} um) vs fine ground truth ({fine['dx']*1000:.0f} um)\n"
        f"{Path(args.fine).name}  |  {len(boxes_margin)} ROIs  |  "
        f"signal captured: {signal_captured*100:.1f}%  |  area scanned: {area_frac*100:.1f}%  |  "
        f"speedup: {speedup:.1f}x",
        fontsize=10
    )

    extent_fine = [fine["xdata"].min(), fine["xdata"].max(),
                   fine["ydata"].max(), fine["ydata"].min()]
    vmax = np.percentile(fine_comp, 99.5)

    # Panel 1: ground truth + ROI overlays
    ax = axes[0]
    ax.imshow(fine_comp, extent=extent_fine, origin="upper",
              aspect="equal", cmap="inferno", vmin=0, vmax=vmax)
    for b in boxes_margin:
        x0, x1, y0, y1 = b["mm"]
        rect = mpatches.Rectangle(
            (min(x0, x1), min(y0, y1)),
            abs(x1 - x0), abs(y1 - y0),
            linewidth=0.6, edgecolor="cyan", facecolor="none", alpha=0.7,
        )
        ax.add_patch(rect)
    ax.set_title(f"Fine scan + {len(boxes_margin)} ROI boxes (cyan)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")

    # Panel 2: what we'd see with adaptive scan (only ROI pixels)
    ax = axes[1]
    visible = np.where(roi_mask, fine_comp, np.nan)
    ax.imshow(visible, extent=extent_fine, origin="upper",
              aspect="equal", cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title(f"Adaptive view ({area_frac*100:.1f}% of pixels scanned)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_facecolor("#101010")

    # Panel 3: missed signal
    ax = axes[2]
    missed_view = np.where(~roi_mask, fine_comp, 0)
    ax.imshow(missed_view, extent=extent_fine, origin="upper",
              aspect="equal", cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(f"Missed signal ({signal_missed*100:.1f}% of total)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / f"{Path(args.fine).stem}_validation.png"
    save_and_show(fig, outpath, show=not args.no_show)


if __name__ == "__main__":
    main()
