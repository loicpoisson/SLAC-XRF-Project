"""
Script 02 - Trajectory planning for fine-scan ROI visits
=========================================================
Input  : coarse HDF5 file (same as 01)
Output : ordered list of ROI scan areas + time estimate + figure saved to outputs/

Pipeline:
  1. Run the same ROI detection as Script 01
  2. Add a safety margin around each bounding box
  3. Order the visits with a nearest-neighbor TSP heuristic (O(N^2))
  4. Optionally improve with a 2-opt pass (exchanges pairs of edges)
  5. Print a scan table + estimated total time
  6. Save the trajectory figure

Usage:
    python scripts/02_trajectory.py
    python scripts/02_trajectory.py --file data/Data_May2026/SMW_UA1_P1_100um_10ms_12000_0_001.hdf5
    python scripts/02_trajectory.py --n-fine 10 --no-2opt
    python scripts/02_trajectory.py --start-xy -400 0   (starting position in mm)
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
                                group_rois, add_margin, travel_time, summarize_rois,
                                nearest_neighbor_tour, two_opt_improve, or_opt_improve)
from utils.paths       import find_coarse, require, PROJECT_ROOT

# ── defaults ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="XRF fine-scan trajectory planner")
    default_file = find_coarse()
    p.add_argument("--file", default=str(default_file) if default_file else None,
                   help="Path to coarse HDF5 (auto-detected from data/ if omitted)")
    p.add_argument("--channels",  default=None,
                   help="Comma-separated element channels (default: all)")
    p.add_argument("--method",    default="auto",
                   choices=["auto", "mad", "trimmed", "iqr", "sigma"])
    p.add_argument("--k",         default=1.0,  type=float)
    p.add_argument("--min-px",    default=1,    type=int)
    p.add_argument("--dwell",     default=25.0, type=float,
                   help="Fine scan dwell time per pixel [ms]")
    p.add_argument("--fine-px",   default=0.025, type=float,
                   help="Fine scan pixel size [mm]")
    p.add_argument("--setup-ms",  default=500.0, type=float,
                   help="Per-ROI scanner setup overhead [ms]")
    p.add_argument("--n-fine",    default=5, type=int,
                   help="Extra fine pixels added as margin on each side (default 5)")
    p.add_argument("--no-2opt",   action="store_true",
                   help="Skip 2-opt improvement (use pure nearest-neighbor only)")
    p.add_argument("--no-oropt", action="store_true",
                   help="Skip Or-opt improvement (applied after 2-opt by default)")
    p.add_argument("--start-xy",  default=None, nargs=2, type=float,
                   metavar=("X", "Y"),
                   help="Starting XY position of the scanner [mm] (default: first ROI center)")
    return p.parse_args()


def pick_start(boxes, start_xy):
    """Return index of ROI whose center is closest to start_xy."""
    centers = np.array([b["center_mm"] for b in boxes])
    start   = np.array(start_xy)
    dists   = np.linalg.norm(centers - start, axis=1)
    return int(np.argmin(dists))


# ── scan time estimator ───────────────────────────────────────────────────────

def estimate_scan_time(box, fine_px_mm, dwell_ms, setup_ms):
    """
    Estimate total time to scan one ROI box [ms].

    Scan time = setup + n_pixels * dwell
    n_pixels  = ceil(width/fine_px) * ceil(height/fine_px)
    """
    x0, x1, y0, y1 = box["mm"]
    w  = abs(x1 - x0)
    h  = abs(y1 - y0)
    nx = max(1, int(np.ceil(w / fine_px_mm)) + 1)
    ny = max(1, int(np.ceil(h / fine_px_mm)) + 1)
    return setup_ms + nx * ny * dwell_ms, nx, ny


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_trajectory(data, composite, boxes_raw, boxes_margin, tour,
                    start_pos, args, method_label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{Path(args.file).name}\n"
        f"Trajectory: {len(boxes_margin)} ROI(s), {method_label}",
        fontsize=10
    )

    extent = [data["xdata"].min(), data["xdata"].max(),
              data["ydata"].max(), data["ydata"].min()]
    vmax = np.percentile(composite, 99)
    colors = plt.cm.tab10.colors

    for ax_idx, (ax, use_margin) in enumerate(zip(axes, [False, True])):
        boxes = boxes_raw if not use_margin else boxes_margin
        title = "Raw ROI boxes" if not use_margin else f"With margin ({boxes_margin[0]['margin_mm']*1000:.0f} um)"

        ax.imshow(composite, extent=extent, origin="upper",
                  aspect="equal", cmap="inferno", alpha=0.55, vmin=0, vmax=vmax)

        # draw ROI boxes
        for b in boxes:
            x0, x1, y0, y1 = b["mm"]
            color = colors[(b["id"] - 1) % 10]
            rect = mpatches.Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0), abs(y1 - y0),
                linewidth=1.5, edgecolor=color, facecolor="none",
                label=f"ROI {b['id']}"
            )
            ax.add_patch(rect)

        # draw trajectory arrows on margin panel
        if use_margin:
            ordered = [boxes_margin[i] for i in tour]
            xs = [start_pos[0]] + [b["center_mm"][0] for b in ordered]
            ys = [start_pos[1]] + [b["center_mm"][1] for b in ordered]
            ax.plot(xs, ys, "w--", linewidth=0.8, alpha=0.7)
            for step, (x, y) in enumerate(zip(xs[1:], ys[1:]), 1):
                ax.annotate(
                    str(step),
                    xy=(x, y), fontsize=7, color="white",
                    ha="center", va="center", fontweight="bold"
                )
            # mark start
            ax.plot(start_pos[0], start_pos[1], "g^", markersize=8,
                    label="start", zorder=5)

        ax.set_title(title)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / f"{Path(args.file).stem}_trajectory.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Figure saved -> {outpath}")
    plt.show()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.file = str(require(args.file, "coarse HDF5"))

    # ── step 1: ROI detection (same as script 01) ─────────────────────────
    print(f"\nLoading  {args.file}")
    data        = load_xrf(args.file)
    all_elements = list_element_channels(data)
    channels    = [c.strip() for c in args.channels.split(",")] if args.channels else None
    composite, used, excluded = get_composite_map(data, channels)

    print(f"  Shape    : {data['mapdata'].shape}")
    print(f"  Elements : {all_elements}")
    if excluded:
        print(f"  Excluded (constant): {excluded}")

    mask, thresh, method_used = threshold_map(composite, method=args.method, k=args.k)
    print(f"\nThreshold : {thresh:.1f}  (method={method_used}, k={args.k})")

    labeled, _ = label_rois(mask, min_pixels=args.min_px)
    boxes = get_bounding_boxes(labeled, data["xdata"], data["ydata"])
    print(f"Raw ROIs  : {len(boxes)}")

    boxes = group_rois(
        boxes,
        dwell_ms  = args.dwell,
        dx        = args.fine_px,
        dy        = args.fine_px,
        setup_ms  = args.setup_ms,
    )
    print(f"After grouping: {len(boxes)} ROI(s)")
    boxes_raw = boxes   # keep un-margined copy for plotting

    if not boxes:
        print("No ROIs detected. Try lowering --k or --min-px.")
        return

    # ── step 2: add safety margin ─────────────────────────────────────────
    boxes_margin = add_margin(
        boxes,
        coarse_px_mm = data["dx"],
        fine_px_mm   = args.fine_px,
        n_fine       = args.n_fine,
    )
    margin_um = boxes_margin[0]["margin_mm"] * 1000
    print(f"\nMargin    : {margin_um:.0f} um per side  "
          f"(= {data['dx']*1000/2:.0f} um coarse/2 + {args.n_fine}x{args.fine_px*1000:.0f} um fine)")

    # ── step 3: starting position ─────────────────────────────────────────
    centers = np.array([b["center_mm"] for b in boxes_margin])

    if args.start_xy is not None:
        start_pos = tuple(args.start_xy)
        start_idx = pick_start(boxes_margin, start_pos)
    else:
        start_idx = 0
        start_pos = boxes_margin[0]["center_mm"]

    # ── step 4: nearest-neighbor TSP ──────────────────────────────────────
    tour_nn = nearest_neighbor_tour(centers, start_idx=start_idx)

    tour = tour_nn
    method_label = "nearest-neighbor"

    if not args.no_2opt and len(boxes_margin) > 2:
        tour = two_opt_improve(centers, tour)
        method_label += " + 2-opt"

    if not args.no_oropt and len(boxes_margin) > 2:
        print("  Running Or-opt...")
        tour = or_opt_improve(centers, tour)
        method_label += " + Or-opt"

    # ── step 5: time estimate & summary table ─────────────────────────────
    print(f"\nTour method : {method_label}")
    print(f"\n{'='*75}")
    print(f"  {'Step':>4}  {'ROI':>4}  {'nx':>5}  {'ny':>5}  "
          f"{'Scan ms':>9}  {'Travel ms':>10}  {'Cumul s':>8}")
    print(f"{'='*75}")

    cumul_ms   = 0.0
    prev_center = np.array(start_pos)

    for step, idx in enumerate(tour, 1):
        b = boxes_margin[idx]
        scan_ms, nx, ny = estimate_scan_time(b, args.fine_px, args.dwell, args.setup_ms)
        cx, cy = b["center_mm"]
        dist      = np.linalg.norm(np.array([cx, cy]) - prev_center)
        trav_ms   = travel_time(dist) * 1000.0
        cumul_ms += scan_ms + trav_ms
        print(f"  {step:>4}  {b['id']:>4}  {nx:>5}  {ny:>5}  "
              f"{scan_ms:>9.0f}  {trav_ms:>10.1f}  {cumul_ms/1000:>8.1f}")
        prev_center = np.array([cx, cy])

    print(f"{'='*75}")
    total_s   = cumul_ms / 1000.0
    total_min = total_s  / 60.0
    print(f"\n  Total estimated scan time: {total_s:.1f} s  ({total_min:.1f} min)")

    # raster baseline: scan every pixel in the full coarse-scan area
    full_w = abs(data["xdata"].max() - data["xdata"].min()) + data["dx"]
    full_h = abs(data["ydata"].max() - data["ydata"].min()) + data["dy"]
    nx_full = int(np.ceil(full_w / args.fine_px)) + 1
    ny_full = int(np.ceil(full_h / args.fine_px)) + 1
    raster_ms = nx_full * ny_full * args.dwell
    print(f"  Raster baseline (full area at {args.fine_px*1000:.0f} um): "
          f"{raster_ms/1000:.0f} s  ({raster_ms/60000:.1f} min)")
    print(f"  Speedup: {raster_ms / cumul_ms:.1f}x\n")

    # ── step 6: plot ──────────────────────────────────────────────────────
    plot_trajectory(data, composite, boxes_raw, boxes_margin, tour,
                    start_pos, args, method_label)


if __name__ == "__main__":
    main()
