"""
Script 10 - PRODUCTION pipeline: coarse only -> adaptive scan plan
=========================================================================
The only script that does NOT need ground truth.
Workflow:
  1. Load a coarse HDF5
  2. Compute descriptors (script 09 logic)
  3. Automatically pick the strategy (ROI / Combined / Temporal)
  4. Build the spatial mask + dwell map
  5. Outputs:
     - Text report (chosen strategy, parameters, estimated time)
     - JSON scan plan (regions + dwells)
     - Figure showing the plan
     - SMAK-compatible pickle (.dpm-style)

Usage:
    python scripts/10_run_pipeline.py
    python scripts/10_run_pipeline.py --coarse data/.../foo_250um.hdf5
    python scripts/10_run_pipeline.py --strategy force_combined  --level 60
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map
from utils.roi_utils import (sample_mask, threshold_map, label_rois,
                              get_bounding_boxes, group_rois, add_margin,
                              _thresh_otsu)
from utils.paths import find_coarse, require, PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(
        description="Production XRF adaptive scan planner",
        epilog="Pass the coarse HDF5 as a positional arg, or as --coarse, "
               "or omit it to auto-detect a 250um file from data/.")
    p.add_argument("coarse_pos", nargs="?", default=None,
                   help="Path to coarse HDF5 (positional, optional)")
    default_coarse = find_coarse()
    p.add_argument("--coarse", default=str(default_coarse) if default_coarse else None,
                   help="Path to coarse HDF5 (named form, same effect as positional)")
    p.add_argument("--channels",   default=None,
                   help="Comma-separated element channels to use (default: all)")
    p.add_argument("--strategy",   default="auto",
                   choices=["auto", "force_roi", "force_morpho",
                            "force_combined", "force_temporal"],
                   help="Force a strategy or let the recommender decide (default)")
    p.add_argument("--fine-px",    default=0.025, type=float,
                   help="Target fine-scan pixel size [mm] (default 25 um)")
    p.add_argument("--fine-dwell", default=10.0,  type=float,
                   help="Reference fine dwell [ms] (default 10)")
    p.add_argument("--dwell-low",  default=1.0,   type=float,
                   help="Min dwell for background pixels [ms]")
    p.add_argument("--dwell-high", default=50.0,  type=float,
                   help="Max dwell for bright pixels [ms]")
    p.add_argument("--level",      default=60.0,  type=float,
                   help="sample_mask level (default 60)")
    p.add_argument("--k",          default=1.0,   type=float,
                   help="ROI detection sensitivity (default 1.0)")
    p.add_argument("--setup-ms",   default=500.0, type=float)
    p.add_argument("--no-match-budget", action="store_true",
                   help="Don't normalize dwells to raster total time. By default "
                        "we scale so total time <= raster (guarantees speedup >= 1).")
    return p.parse_args()


# ── 1. Descriptor + strategy choice (from script 09) ──────────────────────────

def describe(composite):
    """Return descriptor dict for the coarse composite."""
    x = composite.ravel().astype(float)
    clip = np.percentile(x, 95)
    otsu_low = _thresh_otsu(np.minimum(x, clip))
    limit = np.percentile(x, 50)
    x_low = x[x <= limit]
    otsu_floor = _thresh_otsu(x_low) if len(x_low) > 20 else float(np.percentile(x, 5))
    top_thr = np.quantile(x, 0.90)
    sig_top = x[x > top_thr].sum()
    med = np.median(x)
    return {
        "sparsity":      float(np.mean(x > otsu_low)),
        "sample_frac":   float(np.mean(x > otsu_floor)),
        "concentration": float(sig_top / x.sum()) if x.sum() > 0 else 0,
        "dynamic_range": float(np.log10(x.max() / max(med, 1) + 1)),
        "median":        float(med),
        "max":           float(x.max()),
        "otsu_low":      otsu_low,
        "otsu_floor":    otsu_floor,
    }


def choose_strategy(desc):
    """Apply recommender rules; return (method_name, dwell_strategy, why)."""
    s, f = desc["sparsity"], desc["sample_frac"]
    c, d = desc["concentration"], desc["dynamic_range"]

    if f < 0.30:
        method, why_m = "ROI", f"Sample sparse ({f*100:.0f}% area) -> big spatial gain"
    elif f < 0.70:
        method, why_m = "combined", f"Medium density ({f*100:.0f}%) -> spatial mask + temporal"
    elif c > 0.70:
        method, why_m = "combined", (f"Dense ({f*100:.0f}%) but signal concentrated ({c*100:.0f}%) "
                                     "-> combined still helps")
    else:
        method, why_m = "temporal", (f"Dense ({f*100:.0f}%, conc {c*100:.0f}%) -> temporal only "
                                     "(spatial would cap at " + f"{1/f:.2f}x)")

    if c > 0.60:
        dwell, why_d = "binary", f"concentration {c*100:.0f}% > 60% -> binary tout-ou-rien"
    elif d > 2.5:
        dwell, why_d = "log", f"dynamic range {d:.1f} > 2.5 -> log scaling"
    else:
        dwell, why_d = "linear", f"smooth distribution -> linear scaling"
    return method, dwell, why_m, why_d


# ── 2. Mask construction (from scripts 03/04/08) ─────────────────────────────

def build_roi_mask_coarse(composite, coarse_data, k, fine_px, dwell_ms,
                           setup_ms, n_fine=5):
    """Build coarse-grid mask from ROI detection. Returns (mask, boxes_margin)."""
    bin_mask, _, _ = threshold_map(composite, method="auto", k=k)
    labeled, _ = label_rois(bin_mask, min_pixels=1)
    boxes = get_bounding_boxes(labeled, coarse_data["xdata"], coarse_data["ydata"])
    if not boxes:
        return np.zeros_like(composite, dtype=bool), []
    boxes = group_rois(boxes, dwell_ms=dwell_ms, dx=fine_px, dy=fine_px,
                       setup_ms=setup_ms)
    boxes_m = add_margin(boxes, coarse_px_mm=coarse_data["dx"],
                         fine_px_mm=fine_px, n_fine=n_fine)
    mask = np.zeros_like(composite, dtype=bool)
    xc, yc = coarse_data["xdata"], coarse_data["ydata"]
    for b in boxes_m:
        x0, x1, y0, y1 = b["mm"]
        ix = np.where((xc >= min(x0, x1)) & (xc <= max(x0, x1)))[0]
        iy = np.where((yc >= min(y0, y1)) & (yc <= max(y0, y1)))[0]
        if len(ix) > 0 and len(iy) > 0:
            mask[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1] = True
    return mask, boxes_m


def build_morpho_mask(composite, level):
    """Sample mask via otsu_lower morpho (closing only)."""
    mask, thresh = sample_mask(composite, kernel_px=2, mode="otsu_lower",
                                level=level)
    return mask, thresh


# ── 3. Dwell allocation (from script 07) ─────────────────────────────────────

def allocate_dwell(coarse_signal, strategy, dwell_low, dwell_high, mask=None):
    """Compute per-pixel dwell on the coarse grid (will be replicated to fine)."""
    x = coarse_signal.astype(float)
    pool = x[mask] if mask is not None else x.ravel()
    if strategy == "binary":
        thr = float(np.quantile(pool, 0.75))
        return np.where(x > thr, dwell_high, dwell_low), thr
    if strategy == "linear":
        lo, hi = np.quantile(pool, [0.05, 0.95])
        if hi <= lo:
            return np.full_like(x, (dwell_low + dwell_high) / 2), None
        normed = np.clip((x - lo) / (hi - lo), 0, 1)
        return dwell_low + normed * (dwell_high - dwell_low), None
    if strategy == "log":
        eps = 1.0
        logx = np.log(np.maximum(x, eps))
        pool_log = logx[mask] if mask is not None else logx.ravel()
        lo, hi = np.quantile(pool_log, [0.05, 0.95])
        if hi <= lo:
            return np.full_like(x, (dwell_low + dwell_high) / 2), None
        normed = np.clip((logx - lo) / (hi - lo), 0, 1)
        return dwell_low + normed * (dwell_high - dwell_low), None
    raise ValueError(f"Unknown strategy '{strategy}'")


# ── 4. Build scan plan (regions + dwells) ─────────────────────────────────────

def build_scan_plan(mask_coarse, dwell_map_coarse, coarse, fine_px, fine_dwell):
    """
    Convert (mask, dwell_map) on coarse grid into a scan plan: list of
    rectangular regions with uniform (or majority) dwell to send to the scanner.
    Each region = connected component in the mask.
    """
    labeled, n = ndimage.label(mask_coarse)
    plan = []
    for i in range(1, n + 1):
        rows, cols = np.where(labeled == i)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        x0_mm = float(coarse["xdata"][c0])
        x1_mm = float(coarse["xdata"][c1])
        y0_mm = float(coarse["ydata"][r0])
        y1_mm = float(coarse["ydata"][r1])
        region_dwell = dwell_map_coarse[labeled == i]
        median_dwell = float(np.median(region_dwell))
        max_dwell    = float(region_dwell.max())
        min_dwell    = float(region_dwell.min())

        nx_fine = int(np.ceil(abs(x1_mm - x0_mm) / fine_px)) + 1
        ny_fine = int(np.ceil(abs(y1_mm - y0_mm) / fine_px)) + 1
        plan.append({
            "region_id":    i,
            "x_start_mm":   min(x0_mm, x1_mm),
            "x_end_mm":     max(x0_mm, x1_mm),
            "y_start_mm":   min(y0_mm, y1_mm),
            "y_end_mm":     max(y0_mm, y1_mm),
            "fine_px_mm":   fine_px,
            "n_pixels_x":   nx_fine,
            "n_pixels_y":   ny_fine,
            "dwell_median_ms": median_dwell,
            "dwell_min_ms":    min_dwell,
            "dwell_max_ms":    max_dwell,
            "dwell_uniform_ms": median_dwell,   # what the scanner will actually use
        })
    return plan


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    # Positional wins over --coarse if both are provided; both fall back to autodetect.
    coarse_path = args.coarse_pos or args.coarse
    args.coarse = str(require(coarse_path, "coarse HDF5"))

    print(f"\n{'='*70}")
    print(f"  XRF ADAPTIVE SCAN PLANNER (production)")
    print(f"{'='*70}")
    print(f"  Input: {args.coarse}")

    # 1. Load coarse
    coarse = load_xrf(args.coarse)
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    composite, used, excluded = get_composite_map(coarse, channels)
    print(f"  Shape: {coarse['mapdata'].shape}, pixel {coarse['dx']*1000:.0f}um")
    print(f"  Channels: {used}")

    # 2. Descriptor + strategy
    desc = describe(composite)
    print(f"\n  === SAMPLE DESCRIPTORS ===")
    for k in ["sample_frac", "sparsity", "concentration", "dynamic_range"]:
        print(f"    {k:>15}: {desc[k]*100 if 'frac' in k or k in ('sparsity','concentration') else desc[k]:.2f}"
              + ("%" if 'frac' in k or k in ('sparsity','concentration') else ""))

    auto_method, auto_dwell, why_m, why_d = choose_strategy(desc)
    if args.strategy == "auto":
        method = auto_method
    else:
        method = args.strategy.replace("force_", "")
    dwell_strat = auto_dwell

    print(f"\n  === CHOSEN STRATEGY ===")
    print(f"    method  : {method}      ({'AUTO' if args.strategy=='auto' else 'FORCED'})")
    print(f"    dwell   : {dwell_strat}")
    print(f"    why_m   : {why_m}")
    print(f"    why_d   : {why_d}")

    # 3. Build mask + dwell
    if method == "ROI":
        mask_coarse, boxes = build_roi_mask_coarse(
            composite, coarse, args.k, args.fine_px, args.fine_dwell,
            args.setup_ms,
        )
        dwell_map, _ = allocate_dwell(composite, dwell_strat,
                                       args.dwell_low, args.dwell_high,
                                       mask=mask_coarse)
    elif method == "morpho":
        mask_coarse, thresh = build_morpho_mask(composite, args.level)
        dwell_map = np.full_like(composite, args.fine_dwell, dtype=float)
    elif method == "combined":
        mask_coarse, thresh = build_morpho_mask(composite, args.level)
        dwell_map, _ = allocate_dwell(composite, dwell_strat,
                                       args.dwell_low, args.dwell_high,
                                       mask=mask_coarse)
    elif method == "temporal":
        mask_coarse = np.ones_like(composite, dtype=bool)
        dwell_map, _ = allocate_dwell(composite, dwell_strat,
                                       args.dwell_low, args.dwell_high)
    else:
        raise ValueError(f"Unknown method '{method}'")

    # 3b. Budget matching: scale dwells so total time <= raster baseline
    # (default ON; pass --no-match-budget to disable and let dwell_high apply as-is)
    if not args.no_match_budget and method != "morpho":
        # Estimate raster total at fine resolution
        full_w = abs(coarse["xdata"].max() - coarse["xdata"].min()) + coarse["dx"]
        full_h = abs(coarse["ydata"].max() - coarse["ydata"].min()) + coarse["dy"]
        nx_full = int(np.ceil(full_w / args.fine_px)) + 1
        ny_full = int(np.ceil(full_h / args.fine_px)) + 1
        raster_total_ms = nx_full * ny_full * args.fine_dwell

        # In-mask fine pixels
        n_fine_in_mask = float((mask_coarse.astype(float).sum() *
                                ((coarse["dx"] / args.fine_px) *
                                 (coarse["dy"] / args.fine_px))))
        # Current sum of dwells over scanned area
        # (we approximate: each coarse pixel covers (dx/fine_px)^2 fine pixels with same dwell)
        replicas_per_coarse = (coarse["dx"] / args.fine_px) * (coarse["dy"] / args.fine_px)
        current_total_ms = float((dwell_map * mask_coarse).sum()) * replicas_per_coarse
        if current_total_ms > 0:
            scale = raster_total_ms / current_total_ms
            dwell_map = np.where(mask_coarse, dwell_map * scale, dwell_map)
            print(f"\n  Budget matched: dwells scaled by {scale:.3f} so total time = raster")

    # 4. Build scan plan
    plan = build_scan_plan(mask_coarse, dwell_map, coarse, args.fine_px,
                            args.fine_dwell)

    # Time estimate (sum over regions)
    total_time_s = sum(
        r["dwell_uniform_ms"] * r["n_pixels_x"] * r["n_pixels_y"] / 1000.0
        for r in plan
    ) + len(plan) * args.setup_ms / 1000.0
    # Raster baseline (full fine area, uniform dwell)
    full_w = abs(coarse["xdata"].max() - coarse["xdata"].min()) + coarse["dx"]
    full_h = abs(coarse["ydata"].max() - coarse["ydata"].min()) + coarse["dy"]
    nx_full = int(np.ceil(full_w / args.fine_px)) + 1
    ny_full = int(np.ceil(full_h / args.fine_px)) + 1
    raster_time_s = nx_full * ny_full * args.fine_dwell / 1000.0
    speedup = raster_time_s / total_time_s if total_time_s > 0 else np.inf

    print(f"\n  === SCAN PLAN ===")
    print(f"    Regions to scan : {len(plan)}")
    print(f"    Total fine pixels : ~{sum(r['n_pixels_x']*r['n_pixels_y'] for r in plan):,}")
    print(f"    Estimated time : {total_time_s:.0f} s ({total_time_s/60:.1f} min)")
    print(f"    Raster baseline: {raster_time_s:.0f} s ({raster_time_s/60:.1f} min)")
    print(f"    SPEEDUP        : {speedup:.2f}x")

    # 5. Save outputs
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = Path(args.coarse).stem
    json_path = OUTPUT_DIR / f"{stem}_scan_plan.json"
    summary = {
        "input_coarse": str(args.coarse),
        "descriptor":   desc,
        "strategy":     {"method": method, "dwell": dwell_strat,
                         "why_method": why_m, "why_dwell": why_d},
        "estimate":     {"total_time_s": total_time_s,
                         "raster_time_s": raster_time_s,
                         "speedup": speedup,
                         "n_regions": len(plan)},
        "regions":      plan,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  JSON plan saved -> {json_path}")

    # Pickle for SSRL/SMAK compatibility (just regions in their expected format)
    pkl_path = OUTPUT_DIR / f"{stem}_scan_plan.pkl"
    smak_compat = {
        "energy":  float(coarse["energy"]),
        "regions": [{
            "yrange": [r["x_start_mm"], r["x_end_mm"], r["fine_px_mm"]],
            "zrange": [r["y_start_mm"], r["y_end_mm"], r["fine_px_mm"]],
            "dwell":  r["dwell_uniform_ms"],
        } for r in plan],
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(smak_compat, f)
    print(f"  Pickle plan saved -> {pkl_path}")

    # 6. Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Scan plan: {stem}\n"
        f"Strategy = {method} / {dwell_strat}  |  {len(plan)} regions  |  "
        f"{total_time_s/60:.1f} min  ({speedup:.2f}x speedup)",
        fontsize=10,
    )
    extent = [coarse["xdata"].min(), coarse["xdata"].max(),
              coarse["ydata"].max(), coarse["ydata"].min()]
    vmax = np.percentile(composite, 99)

    # Left: coarse + mask outline + regions
    ax = axes[0]
    ax.imshow(composite, extent=extent, origin="upper", aspect="equal",
              cmap="inferno", vmin=0, vmax=vmax)
    ax.contour(mask_coarse.astype(float), levels=[0.5], colors="cyan",
               linewidths=1.0, extent=extent, origin="upper")
    for r in plan:
        rect = mpatches.Rectangle(
            (r["x_start_mm"], r["y_start_mm"]),
            r["x_end_mm"] - r["x_start_mm"],
            r["y_end_mm"] - r["y_start_mm"],
            linewidth=0.8, edgecolor="lime", facecolor="none", alpha=0.7,
        )
        ax.add_patch(rect)
    ax.set_title(f"Coarse + mask (cyan) + {len(plan)} regions (lime)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    # Right: dwell map masked
    ax = axes[1]
    dwell_view = np.where(mask_coarse, dwell_map, np.nan)
    im = ax.imshow(dwell_view, extent=extent, origin="upper", aspect="equal",
                   cmap="viridis", vmin=args.dwell_low, vmax=args.dwell_high)
    plt.colorbar(im, ax=ax, label="Dwell [ms]", fraction=0.04)
    ax.set_title(f"Allocated dwell map (out-of-mask = not scanned)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
    ax.set_facecolor("#101010")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{stem}_scan_plan.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved -> {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
