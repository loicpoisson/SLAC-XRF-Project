"""
Script 04 - Validation Option 3: morphological mask (Otsu + closing/opening)
==============================================================================
Mission of the coarse scan: find the SAMPLE FOOTPRINT (material vs void),
NOT detect particles.

Pipeline:
  1. Load coarse scan, compute XRF composite
  2. Otsu threshold -> separates background (void) from material
  3. Morpho closing (fill holes) + opening (remove isolated noise)
  4. Project the mask onto the 25 um fine grid
  5. Measure: signal captured, area scanned, speedup

Usage:
    python scripts/04_validate_morphological.py
    python scripts/04_validate_morphological.py --coarse data/...100um... --kernel 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map
from utils.roi_utils import sample_mask
from utils.validation_utils import project_mask
from utils.plotting import save_and_show
from utils.paths import find_coarse, find_fine, require, PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Validate morphological sample mask")
    dc, df = find_coarse(), find_fine()
    p.add_argument("--coarse", default=str(dc) if dc else None,
                   help="Coarse HDF5 (auto-detected from data/ if omitted)")
    p.add_argument("--fine",   default=str(df) if df else None,
                   help="Fine HDF5 ground truth (auto-detected if omitted)")
    p.add_argument("--channels", default=None)
    p.add_argument("--kernel",   default=2, type=int,
                   help="Morphological kernel size [coarse pixels] (default 2)")
    p.add_argument("--mode",     default="otsu_lower",
                   choices=["otsu_lower", "percentile", "otsu_clip"],
                   help="Sample-mask thresholding mode (default otsu_lower)")
    p.add_argument("--level",    default=50.0, type=float,
                   help="Percentile parameter for the chosen mode (default 50)")
    p.add_argument("--dwell",    default=10.0, type=float)
    p.add_argument("--no-show",  action="store_true",
                   help="Save figures without opening a window")
    return p.parse_args()


def main():
    args = parse_args()
    args.coarse = str(require(args.coarse, "coarse HDF5"))
    args.fine   = str(require(args.fine,   "fine HDF5"))

    # ── 1. Coarse scan + Otsu sample mask ────────────────────────────────
    print(f"\n=== COARSE SCAN ===")
    print(f"Loading {args.coarse}")
    coarse = load_xrf(args.coarse)
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    coarse_comp, coarse_used, _ = get_composite_map(coarse, channels)
    print(f"  Shape  : {coarse['mapdata'].shape}")
    print(f"  Pixel  : {coarse['dx']*1000:.0f} x {coarse['dy']*1000:.0f} um")

    mask_coarse, otsu_thresh = sample_mask(coarse_comp, kernel_px=args.kernel,
                                           mode=args.mode, level=args.level,
                                           verbose=True)
    n_in  = int(mask_coarse.sum())
    n_tot = mask_coarse.size
    print(f"  Mode: {args.mode}  level={args.level}  threshold = {otsu_thresh:.1f}")
    print(f"  Sample pixels : {n_in}/{n_tot}  ({n_in/n_tot*100:.1f}%)")

    # ── 2. Fine scan (ground truth) ──────────────────────────────────────
    print(f"\n=== FINE SCAN (ground truth) ===")
    print(f"Loading {args.fine}")
    fine = load_xrf(args.fine)
    fine_comp, fine_used, _ = get_composite_map(fine, coarse_used)
    print(f"  Shape  : {fine['mapdata'].shape}")

    # ── 3. Project coarse mask onto fine grid ────────────────────────────
    mask_fine = project_mask(
        mask_coarse,
        coarse["xdata"], coarse["ydata"],
        fine["xdata"],   fine["ydata"],
    )

    # ── 4. Metrics ───────────────────────────────────────────────────────
    n_pix_total = mask_fine.size
    n_pix_in    = int(mask_fine.sum())
    area_frac   = n_pix_in / n_pix_total

    signal_total    = float(fine_comp.sum())
    signal_in_mask  = float(fine_comp[mask_fine].sum())
    signal_captured = signal_in_mask / signal_total if signal_total > 0 else 0.0
    signal_missed   = 1.0 - signal_captured

    # time speedup (fine scan with same dwell everywhere)
    raster_ms   = n_pix_total * args.dwell
    adaptive_ms = n_pix_in    * args.dwell
    speedup     = raster_ms / adaptive_ms if adaptive_ms > 0 else np.inf

    print(f"\n=== VALIDATION METRICS (morphological) ===")
    print(f"  Fine scan pixels       : {n_pix_total:,}")
    print(f"  Pixels inside mask     : {n_pix_in:,}  ({area_frac*100:.1f}%)")
    print(f"  Total XRF signal       : {signal_total:.3e}")
    print(f"  Signal captured        : {signal_in_mask:.3e}  ({signal_captured*100:.1f}%)")
    print(f"  Signal MISSED          : {signal_total - signal_in_mask:.3e}  ({signal_missed*100:.1f}%)")
    print(f"  Raster fine time       : {raster_ms/1000:.0f} s")
    print(f"  Adaptive time          : {adaptive_ms/1000:.0f} s")
    print(f"  REAL SPEEDUP           : {speedup:.2f}x")
    eff_str = f"{signal_captured/area_frac:.2f}x" if area_frac > 0 else "n/a"
    print(f"  Signal efficiency      : {eff_str}")

    # ── 5. Figure: 3 panels (mask coarse | ground truth + overlay | missed) ─
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Option 3 (morphological mask): {Path(args.coarse).stem}  ->  {Path(args.fine).stem}\n"
        f"kernel={args.kernel}px  |  area scanned: {area_frac*100:.1f}%  |  "
        f"signal captured: {signal_captured*100:.1f}%  |  speedup: {speedup:.2f}x",
        fontsize=10,
    )

    extent_c = [coarse["xdata"].min(), coarse["xdata"].max(),
                coarse["ydata"].max(), coarse["ydata"].min()]
    extent_f = [fine["xdata"].min(),   fine["xdata"].max(),
                fine["ydata"].max(),   fine["ydata"].min()]
    vmax_c   = np.percentile(coarse_comp, 99)
    vmax_f   = np.percentile(fine_comp,   99.5)

    # Panel 1: coarse composite + Otsu mask outline
    ax = axes[0]
    ax.imshow(coarse_comp, extent=extent_c, origin="upper", aspect="equal",
              cmap="inferno", vmin=0, vmax=vmax_c)
    ax.contour(mask_coarse.astype(float),
               levels=[0.5], colors="cyan", linewidths=0.8,
               extent=extent_c, origin="upper")
    ax.set_title(f"Coarse {coarse['dx']*1000:.0f}um + Otsu mask (cyan)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    # Panel 2: fine scan + projected mask outline
    ax = axes[1]
    ax.imshow(fine_comp, extent=extent_f, origin="upper", aspect="equal",
              cmap="inferno", vmin=0, vmax=vmax_f)
    ax.contour(mask_fine.astype(float),
               levels=[0.5], colors="cyan", linewidths=0.6,
               extent=extent_f, origin="upper")
    ax.set_title(f"Fine {fine['dx']*1000:.0f}um + projected mask")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    # Panel 3: missed signal (outside the mask)
    ax = axes[2]
    missed = np.where(~mask_fine, fine_comp, 0)
    ax.imshow(missed, extent=extent_f, origin="upper", aspect="equal",
              cmap="magma", vmin=0, vmax=vmax_f)
    ax.set_title(f"Missed signal ({signal_missed*100:.1f}% of total)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / f"{Path(args.coarse).stem}_morphological_k{args.kernel}.png"
    save_and_show(fig, outpath, show=not args.no_show)


if __name__ == "__main__":
    main()
