"""
Script 07 - TEMPORAL adaptive: variable dwell from signal (Paper 2 style)
=============================================================================
Instead of excluding pixels spatially, keep every pixel but adapt the DWELL
(measurement time) to the locally estimated signal.

Mechanism:
  1. Coarse scan 250um (fast, low dwell)
  2. For each coarse pixel, estimate the "interest level" (composite value)
  3. Compute the dwell allocated to the fine scan for each fine pixel based
     on the coarse value of its corresponding region:
       - high signal -> long dwell  (typically 50-100 ms)
       - medium      -> normal dwell (10 ms)
       - low         -> short dwell (1-2 ms)
  4. Compare:
       - signal capture (weighted by dwell, like Paper 2 does for MSE)
       - total time
       - "quality": mean SNR weighted by signal

Strategy comparison:
  - 'binary'  : dwell = (high if coarse > threshold else low)  [2 levels]
  - 'linear'  : dwell = base + scale * coarse_signal            [continuous]
  - 'log'     : dwell = base * log(coarse_signal + eps)         [soft]

Usage:
    python scripts/07_validate_temporal.py
    python scripts/07_validate_temporal.py --strategy linear --dwell-low 1 --dwell-high 50
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map
from utils.validation_utils import project_coarse_to_fine
from utils.dwell import allocate_dwell
from utils.plotting import save_and_show
from utils.paths import find_coarse, find_fine, require, PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Temporal adaptive (dwell-variable) validation")
    dc, df = find_coarse(), find_fine()
    p.add_argument("--coarse", default=str(dc) if dc else None,
                   help="Coarse HDF5 (auto-detected from data/ if omitted)")
    p.add_argument("--fine",   default=str(df) if df else None,
                   help="Fine HDF5 ground truth (auto-detected if omitted)")
    p.add_argument("--channels",   default=None)
    p.add_argument("--strategy",   default="linear",
                   choices=["binary", "linear", "log", "sqrt"],
                   help="Dwell allocation strategy ('sqrt' = MSE-optimal shape)")
    p.add_argument("--dwell-low",  default=1.0,   type=float,
                   help="Minimum dwell [ms]  (background pixels)")
    p.add_argument("--dwell-high", default=100.0, type=float,
                   help="Maximum dwell [ms]  (bright pixels)")
    p.add_argument("--fine-dwell", default=10.0,  type=float,
                   help="Reference uniform dwell [ms] used by the raster baseline")
    p.add_argument("--threshold",  default=None,  type=float,
                   help="Binary strategy: threshold on coarse composite "
                        "(default = Q75 of coarse)")
    p.add_argument("--match-budget", action="store_true",
                   help="Scale the dwell map so the total time matches the "
                        "raster baseline (fair comparison at equal time)")
    p.add_argument("--no-show", action="store_true",
                   help="Save figures without opening a window")
    return p.parse_args()


def main():
    args = parse_args()
    args.coarse = str(require(args.coarse, "coarse HDF5"))
    args.fine   = str(require(args.fine,   "fine HDF5"))

    # ── 1. Coarse + fine scans ────────────────────────────────────────────
    print(f"\n=== TEMPORAL ADAPTIVE VALIDATION ===")
    print(f"Strategy : {args.strategy}")
    print(f"Dwell    : low={args.dwell_low}ms, high={args.dwell_high}ms, "
          f"reference raster={args.fine_dwell}ms")

    coarse = load_xrf(args.coarse)
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    coarse_comp, used, _ = get_composite_map(coarse, channels)
    print(f"\nCoarse {Path(args.coarse).name}")
    print(f"  shape {coarse['mapdata'].shape}, pixel {coarse['dx']*1000:.0f}um")

    fine = load_xrf(args.fine)
    fine_comp, _, _ = get_composite_map(fine, used)
    print(f"Fine   {Path(args.fine).name}")
    print(f"  shape {fine['mapdata'].shape}, pixel {fine['dx']*1000:.0f}um")

    # ── 2. Project coarse signal onto fine grid ───────────────────────────
    coarse_at_fine = project_coarse_to_fine(coarse_comp, coarse, fine)
    print(f"\nProjected coarse signal range: "
          f"[{coarse_at_fine.min():.0f}, {coarse_at_fine.max():.0f}]")

    # ── 3. Allocate dwell per fine pixel ──────────────────────────────────
    dwell_map_ms, thr = allocate_dwell(
        coarse_at_fine, args.strategy, args.dwell_low, args.dwell_high,
        threshold=args.threshold,
    )
    if thr is not None:
        print(f"Binary threshold on coarse: {thr:.1f}")

    if args.match_budget:
        target_total_ms = dwell_map_ms.size * args.fine_dwell
        scale = target_total_ms / dwell_map_ms.sum()
        dwell_map_ms = dwell_map_ms * scale
        print(f"Budget matched: scaled dwells by {scale:.3f} so total = {target_total_ms/1000:.1f}s")

    print(f"\nDwell map stats:")
    print(f"  min  = {dwell_map_ms.min():.1f} ms")
    print(f"  p25  = {np.percentile(dwell_map_ms, 25):.1f} ms")
    print(f"  med  = {np.percentile(dwell_map_ms, 50):.1f} ms")
    print(f"  p75  = {np.percentile(dwell_map_ms, 75):.1f} ms")
    print(f"  max  = {dwell_map_ms.max():.1f} ms")

    # ── 4. Compute metrics ────────────────────────────────────────────────
    # Total time
    raster_total_ms   = dwell_map_ms.size * args.fine_dwell
    adaptive_total_ms = float(dwell_map_ms.sum())
    speedup           = raster_total_ms / adaptive_total_ms if adaptive_total_ms > 0 else np.inf

    # "Quality" weighted by signal: SNR is proportional to sqrt(dwell * signal).
    # We compute the total photons-on-signal (signal * dwell) for both schemes.
    # Higher photons-on-signal => better quantification of bright pixels.
    signal_photons_adaptive = float((fine_comp * dwell_map_ms).sum())
    signal_photons_raster   = float((fine_comp * args.fine_dwell).sum())

    # Where most photons go (allocation efficiency)
    bright_mask = fine_comp > np.percentile(fine_comp, 90)
    bright_time_adaptive = float(dwell_map_ms[bright_mask].sum())
    bright_time_raster   = bright_mask.sum() * args.fine_dwell
    bright_time_frac_adaptive = bright_time_adaptive / adaptive_total_ms
    bright_time_frac_raster   = bright_time_raster   / raster_total_ms

    # Information gain: ratio of dwell allocated to bright pixels
    bright_dwell_boost = bright_time_frac_adaptive / bright_time_frac_raster

    print(f"\n=== METRICS ===")
    print(f"  Raster total time            : {raster_total_ms/1000:>8.1f} s "
          f"({raster_total_ms/60000:.1f} min)")
    print(f"  Adaptive total time          : {adaptive_total_ms/1000:>8.1f} s "
          f"({adaptive_total_ms/60000:.1f} min)")
    print(f"  TIME SPEEDUP                 : {speedup:>8.2f}x")
    print(f"")
    print(f"  Signal*dwell raster (photons): {signal_photons_raster:.3e}")
    print(f"  Signal*dwell adaptive        : {signal_photons_adaptive:.3e}")
    print(f"  Adaptive/raster (info ratio) : "
          f"{signal_photons_adaptive/signal_photons_raster:>6.2f}x")
    print(f"")
    print(f"  Time on top-10% brightest pixels:")
    print(f"    Raster   : {bright_time_frac_raster*100:>5.1f}%  ({bright_time_raster/1000:.1f}s)")
    print(f"    Adaptive : {bright_time_frac_adaptive*100:>5.1f}%  ({bright_time_adaptive/1000:.1f}s)")
    print(f"    BOOST    : {bright_dwell_boost:.2f}x more time on bright pixels")

    # ── 5. Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Temporal adaptive ({args.strategy}): low={args.dwell_low}ms, "
        f"high={args.dwell_high}ms  |  time speedup: {speedup:.2f}x  |  "
        f"info ratio: {signal_photons_adaptive/signal_photons_raster:.2f}x  |  "
        f"bright-pixel time: {bright_dwell_boost:.2f}x boost",
        fontsize=10,
    )

    extent = [fine["xdata"].min(), fine["xdata"].max(),
              fine["ydata"].max(), fine["ydata"].min()]
    vmax_signal = np.percentile(fine_comp, 99.5)

    # Panel 1: fine scan (ground truth)
    ax = axes[0]
    ax.imshow(fine_comp, extent=extent, origin="upper", aspect="equal",
              cmap="inferno", vmin=0, vmax=vmax_signal)
    ax.set_title(f"Fine signal (ground truth)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    # Panel 2: allocated dwell map
    ax = axes[1]
    im = ax.imshow(dwell_map_ms, extent=extent, origin="upper", aspect="equal",
                   cmap="viridis", vmin=args.dwell_low, vmax=args.dwell_high)
    plt.colorbar(im, ax=ax, label="Dwell [ms]", fraction=0.04)
    ax.set_title(f"Allocated dwell (strategy={args.strategy})")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    # Panel 3: "info" = signal * dwell  (what Paper 2 effectively maximizes)
    ax = axes[2]
    info = fine_comp * dwell_map_ms
    vmax_info = np.percentile(info, 99.5)
    ax.imshow(info, extent=extent, origin="upper", aspect="equal",
              cmap="magma", vmin=0, vmax=vmax_info)
    ax.set_title(f"Photons collected = signal x dwell")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = OUTPUT_DIR / f"{Path(args.coarse).stem}_temporal_{args.strategy}.png"
    save_and_show(fig, outpath, show=not args.no_show)


if __name__ == "__main__":
    main()
