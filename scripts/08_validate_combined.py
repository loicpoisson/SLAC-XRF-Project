"""
Script 08 - Combined: SPATIAL morpho + TEMPORAL adaptive dwell
================================================================
Combines both strategies:
  1. SPATIAL : morpho sample_mask (Otsu_lower) on the coarse -> filters out
     off-sample void zones. No scan on those pixels.
  2. TEMPORAL: on the in-mask pixels, allocate a variable dwell based on the
     projected coarse signal (binary by default: short matrix / long ROI).

Expected Pareto result:
  - Spatial speedup : we skip scanning the void zones
  - Temporal quality: better SNR on bright particles
  - Combined        : trade-off at equal time OR faster

Comparison:
  - Raster baseline      : every pixel at uniform dwell
  - Spatial only (04)    : scan only inside the mask, uniform dwell
  - Temporal only (07)   : all pixels but variable dwell
  - Combined (this script): in-mask pixels only + variable dwell inside

Usage:
    python scripts/08_validate_combined.py
    python scripts/08_validate_combined.py --strategy binary --dwell-high 50
    python scripts/08_validate_combined.py --match-budget
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map
from utils.roi_utils import sample_mask
from utils.validation_utils import (project_mask, _nearest_idx,
                                     estimate_travel_overhead,
                                     compute_validation_metrics)
from utils.paths import find_coarse, find_fine, require, PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Combined spatial+temporal adaptive scan")
    dc, df = find_coarse(), find_fine()
    p.add_argument("--coarse", default=str(dc) if dc else None,
                   help="Coarse HDF5 (auto-detected from data/ if omitted)")
    p.add_argument("--fine",   default=str(df) if df else None,
                   help="Fine HDF5 ground truth (auto-detected if omitted)")
    p.add_argument("--channels",   default=None)
    # Spatial parameters (morpho)
    p.add_argument("--mode",       default="otsu_lower",
                   choices=["otsu_lower", "percentile", "otsu_clip"])
    p.add_argument("--level",      default=60.0, type=float,
                   help="Level for sample_mask (default 60 = sweet spot from sweep)")
    p.add_argument("--kernel",     default=2, type=int)
    # Temporal parameters
    p.add_argument("--strategy",   default="binary",
                   choices=["binary", "linear", "log"])
    p.add_argument("--dwell-low",  default=1.0,   type=float)
    p.add_argument("--dwell-high", default=50.0,  type=float)
    p.add_argument("--threshold",  default=None,  type=float)
    p.add_argument("--match-budget", action="store_true",
                   help="Scale temporal dwells so total time = raster baseline")
    p.add_argument("--fine-dwell", default=10.0,  type=float,
                   help="Reference uniform dwell [ms] used by the raster baseline")
    p.add_argument("--setup-ms",   default=500.0, type=float)
    return p.parse_args()


def project_coarse_to_fine(coarse_comp, coarse_data, fine_data):
    ix = _nearest_idx(coarse_data["xdata"], fine_data["xdata"])
    iy = _nearest_idx(coarse_data["ydata"], fine_data["ydata"])
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    return coarse_comp[IY, IX]


def allocate_dwell(coarse_at_fine, strategy, dwell_low, dwell_high,
                    threshold=None, mask=None):
    """
    Compute dwell for each fine pixel. Only meaningful where mask is True;
    elsewhere the value is 0 (these pixels are not scanned).
    Threshold computations use the in-mask distribution.
    """
    x = coarse_at_fine.astype(float)
    in_mask_vals = x[mask] if mask is not None else x.ravel()

    if strategy == "binary":
        thr = threshold if threshold is not None else float(np.quantile(in_mask_vals, 0.75))
        d = np.where(x > thr, dwell_high, dwell_low)
        return d, thr

    if strategy == "linear":
        lo, hi = np.quantile(in_mask_vals, [0.05, 0.95])
        if hi <= lo:
            return np.full_like(x, (dwell_low + dwell_high) / 2), None
        normed = np.clip((x - lo) / (hi - lo), 0, 1)
        return dwell_low + normed * (dwell_high - dwell_low), None

    if strategy == "log":
        eps = 1.0
        logx = np.log(np.maximum(x, eps))
        in_mask_logs = logx[mask] if mask is not None else logx.ravel()
        lo, hi = np.quantile(in_mask_logs, [0.05, 0.95])
        if hi <= lo:
            return np.full_like(x, (dwell_low + dwell_high) / 2), None
        normed = np.clip((logx - lo) / (hi - lo), 0, 1)
        return dwell_low + normed * (dwell_high - dwell_low), None

    raise ValueError(f"Unknown strategy '{strategy}'")


def main():
    args = parse_args()
    args.coarse = str(require(args.coarse, "coarse HDF5"))
    args.fine   = str(require(args.fine,   "fine HDF5"))

    print(f"\n=== COMBINED SPATIAL + TEMPORAL ===")
    print(f"Spatial : sample_mask({args.mode} L={args.level} kernel={args.kernel})")
    print(f"Temporal: strategy={args.strategy} dwell=[{args.dwell_low}, {args.dwell_high}]ms"
          + (" [budget-matched]" if args.match_budget else ""))

    # ── Load data ─────────────────────────────────────────────────────────
    coarse = load_xrf(args.coarse)
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    coarse_comp, used, _ = get_composite_map(coarse, channels)
    print(f"\nCoarse  {Path(args.coarse).name}")
    print(f"  shape {coarse['mapdata'].shape}, pixel {coarse['dx']*1000:.0f}um")

    fine = load_xrf(args.fine)
    fine_comp, _, _ = get_composite_map(fine, used)
    print(f"Fine    {Path(args.fine).name}")
    print(f"  shape {fine['mapdata'].shape}, pixel {fine['dx']*1000:.0f}um")

    # ── Step 1: SPATIAL mask via morpho ───────────────────────────────────
    mask_coarse, thresh_sp = sample_mask(coarse_comp, kernel_px=args.kernel,
                                          mode=args.mode, level=args.level)
    mask_fine = project_mask(mask_coarse,
                              coarse["xdata"], coarse["ydata"],
                              fine["xdata"],   fine["ydata"])
    n_in     = int(mask_fine.sum())
    n_tot    = mask_fine.size
    area_frac = n_in / n_tot
    print(f"\nSpatial mask: threshold={thresh_sp:.1f}, "
          f"{n_in}/{n_tot} pixels in-mask ({area_frac*100:.1f}%)")

    # ── Step 2: TEMPORAL dwell within the mask ────────────────────────────
    coarse_at_fine = project_coarse_to_fine(coarse_comp, coarse, fine)
    dwell_map, thr = allocate_dwell(
        coarse_at_fine, args.strategy, args.dwell_low, args.dwell_high,
        threshold=args.threshold, mask=mask_fine,
    )
    if thr is not None:
        print(f"  Binary threshold (in-mask Q75): {thr:.1f}")

    # zero outside mask (we don't scan there)
    dwell_map_in = np.where(mask_fine, dwell_map, 0.0)

    # ── Budget matching: scale in-mask dwells so total = raster total ─────
    raster_total_ms   = n_tot * args.fine_dwell
    if args.match_budget:
        current_total = dwell_map_in.sum()
        if current_total > 0:
            scale = raster_total_ms / current_total
            dwell_map_in = dwell_map_in * scale
            print(f"  Budget matched: scaled in-mask dwells by {scale:.3f}")

    # ── Step 3: travel overhead (between mask regions) ────────────────────
    travel = estimate_travel_overhead(mask_fine, fine["xdata"], fine["ydata"],
                                      setup_ms=args.setup_ms)

    # ── Metrics ───────────────────────────────────────────────────────────
    adaptive_dwell_ms  = float(dwell_map_in.sum())
    adaptive_total_ms  = adaptive_dwell_ms + travel["overhead_ms"]
    speedup            = raster_total_ms / adaptive_total_ms if adaptive_total_ms else np.inf

    # Info: photons collected weighted by signal
    info_adaptive = float((fine_comp * dwell_map_in).sum())
    info_raster   = float((fine_comp * args.fine_dwell).sum())
    info_ratio    = info_adaptive / info_raster

    # Signal captured (sum of fine signal in scanned pixels) — note: temporal scans
    # all in-mask pixels, just for varying durations, so 'signal captured' is just
    # the in-mask signal.
    signal_total = float(fine_comp.sum())
    signal_in    = float(fine_comp[mask_fine].sum())
    signal_captured = signal_in / signal_total if signal_total else 0

    # Boost on bright pixels
    bright_mask = fine_comp > np.percentile(fine_comp, 90)
    bright_time_adapt = float(dwell_map_in[bright_mask].sum())
    bright_time_rast  = bright_mask.sum() * args.fine_dwell
    bright_boost = bright_time_adapt / bright_time_rast if bright_time_rast else 0

    print(f"\n=== METRICS ===")
    print(f"  Raster total          : {raster_total_ms/1000:>8.1f} s ({raster_total_ms/60000:.1f} min)")
    print(f"  Adaptive dwell sum    : {adaptive_dwell_ms/1000:>8.1f} s")
    print(f"  Travel + setup        : {travel['overhead_ms']/1000:>8.1f} s "
          f"({travel['n_regions']} regions)")
    print(f"  Adaptive total        : {adaptive_total_ms/1000:>8.1f} s ({adaptive_total_ms/60000:.1f} min)")
    print(f"")
    print(f"  TIME SPEEDUP          : {speedup:>8.2f}x")
    print(f"  Signal captured       : {signal_captured*100:>7.1f}%")
    print(f"  Info ratio (vs raster): {info_ratio:>8.2f}x")
    print(f"  Bright-pixel boost    : {bright_boost:>8.2f}x")

    # ── Figure: 4 panels ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        f"Combined spatial({args.mode} L={args.level}) + temporal({args.strategy})  |  "
        f"speedup={speedup:.2f}x  |  signal_in_mask={signal_captured*100:.1f}%  |  "
        f"info_ratio={info_ratio:.2f}x  |  bright_boost={bright_boost:.2f}x",
        fontsize=10,
    )
    extent = [fine["xdata"].min(), fine["xdata"].max(),
              fine["ydata"].max(), fine["ydata"].min()]
    vmax_signal = np.percentile(fine_comp, 99.5)

    # 1: fine signal
    ax = axes[0]
    ax.imshow(fine_comp, extent=extent, origin="upper", aspect="equal",
              cmap="inferno", vmin=0, vmax=vmax_signal)
    ax.contour(mask_fine.astype(float), levels=[0.5], colors="cyan",
               linewidths=0.6, extent=extent, origin="upper")
    ax.set_title("Fine signal + spatial mask (cyan)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    # 2: dwell map (in mask only)
    ax = axes[1]
    vmax_d = float(dwell_map_in[mask_fine].max()) if mask_fine.any() else 1.0
    dwell_view = np.where(mask_fine, dwell_map_in, np.nan)
    im = ax.imshow(dwell_view, extent=extent, origin="upper", aspect="equal",
                   cmap="viridis", vmin=0, vmax=vmax_d)
    plt.colorbar(im, ax=ax, label="Dwell [ms]", fraction=0.04)
    ax.set_title(f"Allocated dwell (out-of-mask = not scanned)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
    ax.set_facecolor("#101010")

    # 3: photons collected = signal × dwell (only in mask)
    ax = axes[2]
    info_map = fine_comp * dwell_map_in
    vmax_i = np.percentile(info_map[info_map > 0], 99.5) if (info_map > 0).any() else 1.0
    ax.imshow(info_map, extent=extent, origin="upper", aspect="equal",
              cmap="magma", vmin=0, vmax=vmax_i)
    ax.set_title(f"Photons collected (= signal × dwell)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    # 4: missed signal (out of mask)
    ax = axes[3]
    missed = np.where(~mask_fine, fine_comp, 0)
    ax.imshow(missed, extent=extent, origin="upper", aspect="equal",
              cmap="magma", vmin=0, vmax=vmax_signal)
    ax.set_title(f"Missed signal ({(1-signal_captured)*100:.1f}% of total)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    tag = f"{args.mode}_L{int(args.level)}_{args.strategy}_low{args.dwell_low}_high{args.dwell_high}"
    if args.match_budget:
        tag += "_matched"
    outpath = OUTPUT_DIR / f"{Path(args.coarse).stem}_combined_{tag}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {outpath}")
    plt.show()


if __name__ == "__main__":
    main()
