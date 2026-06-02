"""
Script 01 - Automatic ROI detection from a coarse XRF scan
===========================================================
Input  : coarse HDF5 file (e.g. 250 um pixel size)
Output : ROI bounding boxes printed to console + figure saved to outputs/

Usage:
    python scripts/01_roi_detection.py
    python scripts/01_roi_detection.py --file data/Data_May2026/SMW_UA1_P1_100um_10ms_12000_0_001.hdf5
    python scripts/01_roi_detection.py --channels Fe.Ka
    python scripts/01_roi_detection.py --channels Fe.Ka,Au.La
    python scripts/01_roi_detection.py --channels Fe.Ka --k 3.0
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map, list_element_channels
from utils.roi_utils   import (threshold_map, label_rois, get_bounding_boxes,
                                group_rois, summarize_rois)
from utils.paths       import find_coarse, require, PROJECT_ROOT

# ── defaults ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="XRF coarse scan ROI detector")
    default_file = find_coarse()
    p.add_argument("--file", default=str(default_file) if default_file else None,
                   help="Path to HDF5 file (auto-detects a 250um file in data/ if omitted)")
    p.add_argument("--channels", default=None,
                   help="Comma-separated element channels to sum (default: all elements). "
                        "E.g. Fe.Ka  or  Fe.Ka,Au.La")
    p.add_argument("--method",   default="auto",
                   choices=["auto", "mad", "trimmed", "iqr", "sigma"],
                   help="Threshold method (default: auto — selects best via Fisher score)")
    p.add_argument("--k",        default=1.0, type=float,
                   help="Sigma multiplier for 'sigma' method (default: 2.0)")
    p.add_argument("--min-px",   default=1, type=int,
                   help="Minimum ROI size in pixels (default: 2)")
    p.add_argument("--dwell",    default=25.0, type=float,
                   help="Fine scan dwell time per pixel [ms] (default: 25)")
    p.add_argument("--fine-px",  default=0.025, type=float,
                   help="Fine scan pixel size [mm] (default: 0.025 = 25 um)")
    p.add_argument("--setup-ms", default=500.0, type=float,
                   help="Per-region scanner setup overhead [ms] (default: 500)")
    return p.parse_args()


def build_channel_label(used_channels, all_elements):
    """Human-readable label for the composite map."""
    if set(used_channels) == set(all_elements):
        return "all elements (sum)"
    if len(used_channels) == 1:
        return used_channels[0]
    return "sum: " + " + ".join(used_channels)


def plot_results(data, channel_label, composite, mask, thresh, labeled, boxes, args, method_used):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{Path(args.file).name}\n"
        f"channel: {channel_label}   |   "
        f"threshold: {method_used} (k={args.k}) = {thresh:.1f}",
        fontsize=10
    )

    extent = [data["xdata"].min(), data["xdata"].max(),
              data["ydata"].max(), data["ydata"].min()]

    # clip colorscale to 99th percentile so bright outliers don't crush the map
    vmax = np.percentile(composite, 99)

    # 1 - raw composite map
    ax = axes[0]
    im = ax.imshow(composite, extent=extent, origin="upper",
                   aspect="equal", cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title("Composite intensity map  (clip 99%)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    plt.colorbar(im, ax=ax, fraction=0.046, label="counts")

    # 2 - threshold mask
    ax = axes[1]
    ax.imshow(mask, extent=extent, origin="upper", aspect="equal", cmap="Reds")
    ax.set_title(f"ROI mask  (>{thresh:.1f} cts)")
    ax.set_xlabel("X [mm]")

    # 3 - labeled ROIs with bounding boxes overlaid on map
    ax = axes[2]
    ax.imshow(composite, extent=extent, origin="upper",
              aspect="equal", cmap="inferno", alpha=0.6)

    colors = plt.cm.tab10.colors
    for b in boxes:
        x0, x1, y0, y1 = b["mm"]
        dx, dy = data["dx"], data["dy"]
        rect = mpatches.Rectangle(
            (min(x0, x1) - dx / 2, min(y0, y1) - dy / 2),
            abs(x1 - x0) + dx, abs(y1 - y0) + dy,
            linewidth=1.5,
            edgecolor=colors[(b["id"] - 1) % 10],
            facecolor="none",
            label=f"ROI {b['id']} ({b['size']} px)"
        )
        ax.add_patch(rect)
        cx, cy = b["center_mm"]
        ax.text(cx, cy, str(b["id"]), color="white", fontsize=9,
                ha="center", va="center", fontweight="bold")

    ax.set_title(f"{len(boxes)} ROI(s) detected")
    ax.set_xlabel("X [mm]")
    if boxes:
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    tag = "all" if "all elements" in channel_label else channel_label.replace(".", "_").replace(" ", "")
    outpath = OUTPUT_DIR / f"{Path(args.file).stem}_roi_{tag}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Figure saved -> {outpath}")
    plt.show()


def main():
    args = parse_args()
    args.file = str(require(args.file, "coarse HDF5"))

    # ── load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading  {args.file}")
    data = load_xrf(args.file)
    all_elements = list_element_channels(data)
    print(f"  Shape    : {data['mapdata'].shape}  "
          f"(ny={data['mapdata'].shape[0]}, nx={data['mapdata'].shape[1]}, "
          f"channels={data['mapdata'].shape[2]})")
    print(f"  Energy   : {data['energy']:.0f} eV")
    print(f"  Pixel    : dx={data['dx']*1000:.0f} um  dy={data['dy']*1000:.0f} um")
    print(f"  Elements : {all_elements}")

    # ── channel selection ─────────────────────────────────────────────────────
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    composite, used, excluded = get_composite_map(data, channels)
    channel_label = build_channel_label(used, all_elements)
    print(f"\nComposite map : {channel_label}")
    if excluded:
        print(f"  Excluded (constant/floor): {excluded}")
    print(f"  Intensity range : {composite.min():.1f} - {composite.max():.1f} counts")

    # ── threshold ─────────────────────────────────────────────────────────────
    mask, thresh, method_used = threshold_map(composite, method=args.method, k=args.k)
    coverage = 100.0 * mask.sum() / mask.size
    print(f"\nThreshold : {thresh:.1f} counts  (method={method_used}, k={args.k})")
    print(f"ROI pixels: {mask.sum()} / {mask.size}  ({coverage:.1f}%)")

    # ── label regions ─────────────────────────────────────────────────────────
    labeled, _ = label_rois(mask, min_pixels=args.min_px)
    boxes = get_bounding_boxes(labeled, data["xdata"], data["ydata"])
    print(f"\nBefore grouping: {len(boxes)} raw ROI(s)")

    # ── group nearby ROIs using scanner cost model ────────────────────────────
    boxes = group_rois(
        boxes,
        dwell_ms  = args.dwell,
        dx        = args.fine_px,
        dy        = args.fine_px,
        setup_ms  = args.setup_ms,
    )
    n_after = len(boxes)
    n_merges = sum(b.get("n_original_rois", 1) - 1 for b in boxes)
    if n_merges > 0:
        print(f"After  grouping: {n_after} ROI(s)  ({n_merges} merge(s) — travel > gap cost)")
    else:
        print(f"After  grouping: {n_after} ROI(s)  (no merges — travel cheaper than scanning gap)")
    summarize_rois(boxes, data["dx"], data["dy"])

    # ── plot ──────────────────────────────────────────────────────────────────
    if not boxes:
        print("No ROIs detected. Try lowering --k (e.g. --k 1.5) or --min-px 1")
    else:
        plot_results(data, channel_label, composite, mask, thresh, labeled, boxes, args, method_used)


if __name__ == "__main__":
    main()
