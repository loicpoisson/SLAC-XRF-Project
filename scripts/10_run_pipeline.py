"""
Script 10 - PRODUCTION pipeline: ITERATIVE adaptive scan planner
=========================================================================
The only script that does NOT need ground truth. It now mirrors the real
beamline workflow: a multi-resolution cascade where each level is acquired
AFTER analysing the previous one.

  coarse scan -> analyse -> plan next (finer) scan in the retained zones
              -> (acquire it) -> analyse -> plan the next finer scan -> ...

Logic is NOT re-implemented here: it is composed from the shared utils modules
  - utils.strategy : describe_composite + choose_strategy  (was script 09)
  - utils.dwell    : allocate_dwell                        (was script 07)
  - utils.cascade  : refine_step / refine_cascade          (was script 05)

Two modes
---------
A) STEPWISE (production, default) - one invocation = one beamline step, resumable
   between real acquisitions via an on-disk state file.
     # step 0 : analyse the coarse, plan the next level
     python scripts/10_run_pipeline.py --coarse data/.../UA1_P1_250um....hdf5
     # step N : feed the level you just acquired, plan the next one
     python scripts/10_run_pipeline.py --scan data/.../UA1_P1_100um....hdf5 \
            --resume outputs/UA1_P1_cascade_state.pkl

B) SIMULATE (offline validation, when every level already exists on disk)
     python scripts/10_run_pipeline.py --simulate --sample UA1_P1
     python scripts/10_run_pipeline.py --simulate --sample UA1_P1 --levels 250 100 50 25

Degenerate: `--coarse FILE` alone, with no cascade ladder, falls back to the
single-shot plan at --fine-px (the original script-10 behavior).

Outputs (per step): a text report, a JSON scan plan, a SMAK-compatible pickle,
and a figure.
"""

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map
from utils.strategy import describe_composite, choose_strategy
from utils.dwell import allocate_dwell
from utils.cascade import find_level_file, available_levels, refine_step, refine_cascade
from utils.roi_utils import label_rois, get_bounding_boxes, group_rois, add_margin
from utils.validation_utils import coarse_blockmean
from utils.quality import poisson_mse, poisson_mse_montecarlo, mean_snr
from utils.plotting import save_and_show
from utils.paths import find_coarse, require, PROJECT_ROOT, DATA_DIR

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(
        description="Iterative production XRF adaptive scan planner",
        epilog="Default = stepwise. Use --simulate to run the whole cascade "
               "offline when all resolution levels already exist on disk.")
    # Inputs
    p.add_argument("coarse_pos", nargs="?", default=None,
                   help="Path to coarse HDF5 (positional, optional)")
    p.add_argument("--coarse", default=None,
                   help="Coarse HDF5 for step 0 (auto-detected if omitted)")
    p.add_argument("--scan", default=None,
                   help="Stepwise resume: the level you JUST acquired")
    p.add_argument("--resume", default=None,
                   help="Stepwise resume: path to the cascade state .pkl")
    p.add_argument("--force-level", action="store_true",
                   help="Resume even if --scan resolution != the expected ladder level")
    p.add_argument("--simulate", action="store_true",
                   help="Run the full cascade offline using files on disk")
    p.add_argument("--sample", default=None,
                   help="Sample stem (e.g. UA1_P1) for level auto-detection")
    p.add_argument("--levels", nargs="+", type=int, default=None,
                   help="Resolution ladder in um, coarse first (e.g. 250 100 50 25)")
    p.add_argument("--channels", default=None,
                   help="Comma-separated element channels (default: all)")
    # Strategy
    p.add_argument("--strategy", default="auto",
                   choices=["auto", "force_roi", "force_morpho",
                            "force_combined", "force_temporal"],
                   help="Force a strategy or let the recommender decide (default)")
    p.add_argument("--dwell-strategy", default="auto",
                   choices=["auto", "binary", "linear", "log", "sqrt"],
                   help="Override the dwell allocation ('auto' = recommender; "
                        "'sqrt' = MSE-optimal shape)")
    # sample_mask (footprint) parameters
    p.add_argument("--kernel", default=2, type=int, help="Morphological kernel [px]")
    p.add_argument("--mode", default="otsu_lower",
                   choices=["otsu_lower", "percentile", "otsu_clip"])
    p.add_argument("--level", default=50.0, type=float,
                   help="sample_mask percentile parameter (default 50)")
    # Dwell parameters
    p.add_argument("--fine-px", default=0.025, type=float,
                   help="Target fine pixel [mm] for the single-shot fallback")
    p.add_argument("--fine-dwell", default=10.0, type=float,
                   help="Reference fine dwell [ms] (raster baseline)")
    p.add_argument("--dwell-low", default=1.0, type=float)
    p.add_argument("--dwell-high", default=50.0, type=float)
    p.add_argument("--setup-ms", default=500.0, type=float)
    p.add_argument("--no-match-budget", action="store_true",
                   help="Don't normalize dwells to the raster total time")
    p.add_argument("--margin-fine", default=5, type=int,
                   help="Safety margin added around grouped ROIs [fine pixels]")
    p.add_argument("--mc-draws", default=0, type=int,
                   help="Poisson Monte-Carlo draws for the MSE metric (0 = analytic only)")
    p.add_argument("--no-show", action="store_true",
                   help="Save figures without opening a window")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def sample_hint_from_path(path):
    """Extract a sample stem like 'UA1_P1' from a SMAK filename, else None."""
    m = re.search(r"SMW_(.+?)_\d+um_", Path(path).name)
    return m.group(1) if m else None


def px_um_of(data):
    """Pixel size of a scan in microns (int)."""
    return int(round(data["dx"] * 1000))


def resolve_strategy(desc, forced, forced_dwell="auto"):
    """Pick the strategy dict, honoring --strategy and --dwell-strategy overrides."""
    strat = dict(choose_strategy(desc))
    if forced != "auto":
        strat["method"] = forced.replace("force_", "")
        strat["why_method"] = f"FORCED ({forced})"
    if forced_dwell != "auto":
        strat["dwell"] = forced_dwell
        strat["why_dwell"] = f"FORCED ({forced_dwell})"
    return strat


def _region_dict(rid, x0, x1, y0, y1, dwell_vals, next_px_mm, fine_dwell):
    """Assemble one scan-plan region (uniform = median dwell of its signal)."""
    if dwell_vals.size:
        med, lo, hi = (float(np.median(dwell_vals)),
                       float(dwell_vals.min()), float(dwell_vals.max()))
    else:
        med = lo = hi = fine_dwell
    return {
        "region_id":        rid,
        "x_start_mm":       float(min(x0, x1)),
        "x_end_mm":         float(max(x0, x1)),
        "y_start_mm":       float(min(y0, y1)),
        "y_end_mm":         float(max(y0, y1)),
        "fine_px_mm":       next_px_mm,
        "n_pixels_x":       int(np.ceil(abs(x1 - x0) / next_px_mm)) + 1,
        "n_pixels_y":       int(np.ceil(abs(y1 - y0) / next_px_mm)) + 1,
        "dwell_median_ms":  med,
        "dwell_min_ms":     lo,
        "dwell_max_ms":     hi,
        "dwell_uniform_ms": med,
    }


def _box_dwell_vals(mask, dwell_map, grid, x0, x1, y0, y1):
    """Dwell values of the in-mask pixels inside a clipped mm box."""
    gx, gy = grid["xdata"], grid["ydata"]
    x0c, x1c = max(min(x0, x1), gx.min()), min(max(x0, x1), gx.max())
    y0c, y1c = max(min(y0, y1), gy.min()), min(max(y0, y1), gy.max())
    ix = np.where((gx >= x0c) & (gx <= x1c))[0]
    iy = np.where((gy >= y0c) & (gy <= y1c))[0]
    if ix.size == 0 or iy.size == 0:
        return np.array([]), (x0c, x1c, y0c, y1c)
    sub = np.ix_(iy, ix)
    return dwell_map[sub][mask[sub]], (x0c, x1c, y0c, y1c)


def build_scan_plan(mask, dwell_map, grid, next_px_mm, *, group=True,
                    setup_ms=500.0, margin_fine=5, fine_dwell=10.0):
    """
    Convert (mask, dwell_map) on the current grid into a scan plan.
    `next_px_mm` is the pixel size of the NEXT (finer) scan the plan describes.

    group=True (spatial methods): footprint -> ROI boxes -> travel-aware merge
    (group_rois) -> safety margin (add_margin). group=False (temporal / full
    frame): one region per connected component, no margin.
    """
    if not group:
        labeled, n = ndimage.label(mask)
        plan = []
        for i in range(1, n + 1):
            rows, cols = np.where(labeled == i)
            x0 = float(grid["xdata"][int(cols.min())])
            x1 = float(grid["xdata"][int(cols.max())])
            y0 = float(grid["ydata"][int(rows.min())])
            y1 = float(grid["ydata"][int(rows.max())])
            plan.append(_region_dict(i, x0, x1, y0, y1,
                                     dwell_map[labeled == i], next_px_mm, fine_dwell))
        return plan

    labeled, n = label_rois(mask, min_pixels=1)
    if n == 0:
        return []
    raw = get_bounding_boxes(labeled, grid["xdata"], grid["ydata"])
    med_dwell = float(np.median(dwell_map[mask])) if mask.any() else fine_dwell
    grouped = group_rois(raw, dwell_ms=med_dwell, dx=next_px_mm, dy=next_px_mm,
                         setup_ms=setup_ms)
    boxed = add_margin(grouped, coarse_px_mm=grid["dx"], fine_px_mm=next_px_mm,
                       n_fine=margin_fine)
    plan = []
    for i, b in enumerate(boxed, 1):
        x0, x1, y0, y1 = b["mm"]
        vals, (x0c, x1c, y0c, y1c) = _box_dwell_vals(mask, dwell_map, grid,
                                                     x0, x1, y0, y1)
        plan.append(_region_dict(i, x0c, x1c, y0c, y1c, vals, next_px_mm, fine_dwell))
    return plan


def full_area_mm(grid):
    """(width, height) of the scanned area in mm, including the edge pixels."""
    w = abs(grid["xdata"].max() - grid["xdata"].min()) + grid["dx"]
    h = abs(grid["ydata"].max() - grid["ydata"].min()) + grid["dy"]
    return w, h


def make_dwell_map(comp, grid, mask, strat, args):
    """
    Build the per-pixel dwell map for the chosen method.
      temporal -> scan everything, variable dwell (no spatial mask)
      morpho   -> footprint mask, uniform dwell
      roi/comb -> footprint mask, variable dwell
    Returns (mask_eff, dwell_map).
    """
    method = strat["method"]
    if method == "temporal":
        mask_eff = np.ones_like(comp, dtype=bool)
        dwell_map, _ = allocate_dwell(comp, strat["dwell"], args.dwell_low,
                                      args.dwell_high, mask=None)
    elif method == "morpho":
        mask_eff = mask
        dwell_map = np.full_like(comp, args.fine_dwell, dtype=float)
    else:  # roi / combined : footprint + variable dwell
        mask_eff = mask
        dwell_map, _ = allocate_dwell(comp, strat["dwell"], args.dwell_low,
                                      args.dwell_high, mask=mask_eff)
    return mask_eff, dwell_map


def match_budget(dwell_map, mask_eff, grid, next_px_mm, fine_dwell):
    """
    Scale in-mask dwells so the total adaptive time equals the raster baseline
    at next_px_mm (guarantees speedup >= 1). Returns the scaled dwell_map.
    """
    w, h = full_area_mm(grid)
    nx_full = int(np.ceil(w / next_px_mm)) + 1
    ny_full = int(np.ceil(h / next_px_mm)) + 1
    raster_total_ms = nx_full * ny_full * fine_dwell

    replicas = (grid["dx"] / next_px_mm) * (grid["dy"] / next_px_mm)
    current_total_ms = float((dwell_map * mask_eff).sum()) * replicas
    if current_total_ms > 0:
        scale = raster_total_ms / current_total_ms
        dwell_map = np.where(mask_eff, dwell_map * scale, dwell_map)
        print(f"  Budget matched: dwells scaled by {scale:.3f} so total = raster")
    return dwell_map


def emit_outputs(stem, comp, grid, mask, dwell_map, plan, strat, desc,
                 next_px_mm, args, ground_truth_comp=None, per_level_stats=None,
                 coarse_fill=None):
    """Print the report, save JSON + SMAK pickle + figure for one step.
    Returns the summary dict (so callers can log a cascade manifest)."""
    # Time estimate
    total_time_s = sum(r["dwell_uniform_ms"] * r["n_pixels_x"] * r["n_pixels_y"]
                       for r in plan) / 1000.0 + len(plan) * args.setup_ms / 1000.0
    w, h = full_area_mm(grid)
    nx_full = int(np.ceil(w / next_px_mm)) + 1
    ny_full = int(np.ceil(h / next_px_mm)) + 1
    raster_time_s = nx_full * ny_full * args.fine_dwell / 1000.0
    speedup = raster_time_s / total_time_s if total_time_s > 0 else np.inf

    print(f"\n  === SCAN PLAN (next level @ {next_px_mm*1000:.0f}um) ===")
    print(f"    Regions to scan   : {len(plan)}")
    print(f"    Total fine pixels : ~{sum(r['n_pixels_x']*r['n_pixels_y'] for r in plan):,}")
    print(f"    Estimated time    : {total_time_s:.0f} s ({total_time_s/60:.1f} min)")
    print(f"    Raster baseline   : {raster_time_s:.0f} s ({raster_time_s/60:.1f} min)")
    print(f"    SPEEDUP           : {speedup:.2f}x")

    # Validation metrics when ground truth is available (simulate mode)
    quality = None
    if ground_truth_comp is not None:
        total = float(ground_truth_comp.sum())
        captured = float(ground_truth_comp[mask].sum()) / total if total else 0.0
        predict = "coarse" if coarse_fill is not None else "zero"
        mse = poisson_mse(ground_truth_comp, dwell_map, mask,
                          t_ref=args.fine_dwell, predict=predict,
                          predict_signal=coarse_fill)
        snr_ad = mean_snr(ground_truth_comp, dwell_map, t_ref=args.fine_dwell)
        snr_ras = mean_snr(ground_truth_comp,
                           np.full_like(dwell_map, args.fine_dwell),
                           t_ref=args.fine_dwell)
        quality = {
            "signal_captured": captured,
            "mse_raster":      mse["mse_raster"],
            "mse_adaptive":    mse["mse_adaptive"],
            "mse_ratio":       mse["ratio"],
            "mse_var_term":    mse["var_term"],
            "mse_bias_term":   mse["bias_term"],
            "snr_adaptive":    snr_ad,
            "snr_raster":      snr_ras,
            "snr_gain":        snr_ad / snr_ras if snr_ras else 0.0,
            "predict":         predict,
        }
        print(f"\n  === VALIDATION (ground truth, Poisson MSE) ===")
        print(f"    Signal captured in footprint : {captured*100:.1f}%")
        print(f"    MSE ratio (raster/adaptive)  : {mse['ratio']:.2f}x  "
              f"(>1 = adaptive better)")
        print(f"      variance term : {mse['var_term']:.3e}   "
              f"bias term : {mse['bias_term']:.3e}  (fill={predict})")
        print(f"    Signal-weighted SNR gain     : {quality['snr_gain']:.2f}x")
        if args.mc_draws > 0:
            mc = poisson_mse_montecarlo(ground_truth_comp, dwell_map, mask,
                                        t_ref=args.fine_dwell, predict=predict,
                                        predict_signal=coarse_fill,
                                        n_draws=args.mc_draws)
            quality["mse_ratio_mc"] = mc["ratio"]
            print(f"    MSE ratio (Monte-Carlo, n={mc['n_draws']}) : "
                  f"{mc['ratio']:.2f}x")

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = {
        "input": stem,
        "descriptor": desc,
        "strategy": strat,
        "next_px_mm": next_px_mm,
        "estimate": {"total_time_s": total_time_s, "raster_time_s": raster_time_s,
                     "speedup": speedup, "n_regions": len(plan)},
        "quality": quality,
        "cascade_levels": per_level_stats,
        "regions": plan,
    }
    json_path = OUTPUT_DIR / f"{stem}_scan_plan.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  JSON plan saved   -> {json_path}")

    pkl_path = OUTPUT_DIR / f"{stem}_scan_plan.pkl"
    smak_compat = {"regions": [{
        "yrange": [r["x_start_mm"], r["x_end_mm"], r["fine_px_mm"]],
        "zrange": [r["y_start_mm"], r["y_end_mm"], r["fine_px_mm"]],
        "dwell":  r["dwell_uniform_ms"],
    } for r in plan]}
    with open(pkl_path, "wb") as f:
        pickle.dump(smak_compat, f)
    print(f"  Pickle plan saved -> {pkl_path}")

    _save_figure(stem, comp, grid, mask, dwell_map, plan, strat, speedup,
                 total_time_s, args, show=not args.no_show)
    return summary


def _save_figure(stem, comp, grid, mask, dwell_map, plan, strat, speedup,
                 total_time_s, args, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Scan plan: {stem}  |  {strat['method']} / {strat['dwell']}  |  "
        f"{len(plan)} regions  |  {total_time_s/60:.1f} min  ({speedup:.2f}x)",
        fontsize=10)
    extent = [grid["xdata"].min(), grid["xdata"].max(),
              grid["ydata"].max(), grid["ydata"].min()]
    vmax = np.percentile(comp, 99)

    ax = axes[0]
    ax.imshow(comp, extent=extent, origin="upper", aspect="equal",
              cmap="inferno", vmin=0, vmax=vmax)
    ax.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.0,
               extent=extent, origin="upper")
    for r in plan:
        ax.add_patch(mpatches.Rectangle(
            (r["x_start_mm"], r["y_start_mm"]),
            r["x_end_mm"] - r["x_start_mm"], r["y_end_mm"] - r["y_start_mm"],
            linewidth=0.8, edgecolor="lime", facecolor="none", alpha=0.7))
    ax.set_title(f"Composite + footprint (cyan) + {len(plan)} regions (lime)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

    ax = axes[1]
    dwell_view = np.where(mask, dwell_map, np.nan)
    im = ax.imshow(dwell_view, extent=extent, origin="upper", aspect="equal",
                   cmap="viridis", vmin=args.dwell_low, vmax=args.dwell_high)
    plt.colorbar(im, ax=ax, label="Dwell [ms]", fraction=0.04)
    ax.set_title("Allocated dwell (out-of-mask = not scanned)")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")
    ax.set_facecolor("#101010")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{stem}_scan_plan.png"
    save_and_show(fig, fig_path, show=show)


def resolve_ladder(args, sample_hint, current_px):
    """Determine the resolution ladder (coarse->fine, in um)."""
    if args.levels:
        return list(args.levels)
    if sample_hint:
        lv = available_levels(sample_hint, data_dir=DATA_DIR)
        if lv:
            return lv
    return [current_px]   # single-shot fallback


def next_px_mm_for(ladder, idx, args, current_px):
    """Pixel size [mm] of the next finer level, or the single-shot target."""
    if idx + 1 < len(ladder):
        return ladder[idx + 1] / 1000.0
    if len(ladder) == 1:
        return args.fine_px           # single-shot fallback (original behavior)
    return current_px / 1000.0        # cascade finished: plan at current res


def _append_manifest(sample, idx, px_um, strat, summary, fresh=False):
    """Append one row per cascade step to outputs/<sample>_cascade_manifest.csv."""
    path = OUTPUT_DIR / f"{sample}_cascade_manifest.csv"
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_header = fresh or not path.exists()
    e = summary["estimate"]
    with open(path, "w" if fresh else "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["step", "px_um", "method", "dwell", "n_regions",
                        "speedup", "est_time_s"])
        w.writerow([idx, px_um, strat["method"], strat["dwell"], e["n_regions"],
                    f"{e['speedup']:.3f}", f"{e['total_time_s']:.1f}"])
    print(f"  Manifest updated  -> {path}")


# ── modes ─────────────────────────────────────────────────────────────────────

def run_simulate(args):
    sample = args.sample or sample_hint_from_path(args.coarse_pos or args.coarse or "")
    if not sample:
        raise SystemExit("--simulate needs --sample (or a coarse path to infer it from)")
    ladder = args.levels or available_levels(sample, data_dir=DATA_DIR)
    if len(ladder) < 2:
        raise SystemExit(f"Need >=2 levels for a cascade; found {ladder} for {sample}")

    print(f"\n{'='*70}\n  XRF ITERATIVE PLANNER - SIMULATE  ({sample})\n{'='*70}")
    print(f"  Ladder: {' -> '.join(f'{p}um' for p in ladder)}")

    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    scans_list, ref_channels = [], channels
    for i, px in enumerate(ladder):
        path = find_level_file(sample, px, data_dir=DATA_DIR)
        if path is None:
            raise SystemExit(f"No file for {sample} @ {px}um under {DATA_DIR}")
        d = load_xrf(path)
        comp, used, _ = get_composite_map(d, ref_channels)
        if i == 0:
            ref_channels = used
        scans_list.append({"data": d, "comp": comp, "px": px})
        print(f"  {px}um: {path.name}  {d['mapdata'].shape}")

    desc = describe_composite(scans_list[0]["comp"])
    strat = resolve_strategy(desc, args.strategy, args.dwell_strategy)
    print(f"\n  Strategy: {strat['method']} / {strat['dwell']}  ({strat['why_method']})")

    mask, per_level_stats = refine_cascade(
        scans_list, kernel_px=args.kernel, mode=args.mode, level=args.level,
        detect_final=False, verbose=True)

    print(f"\n  === CASCADE FOOTPRINT ===")
    for s in per_level_stats:
        print(f"    {s['px']:>4}um : in-mask {s['area_frac']*100:5.1f}%  "
              f"signal {s['signal_in_frac']*100:5.1f}%")

    fine = scans_list[-1]
    coarse0 = scans_list[0]
    mask_eff, dwell_map = make_dwell_map(fine["comp"], fine["data"], mask, strat, args)
    next_px = fine["data"]["dx"]   # finest level scans at its own resolution
    if not args.no_match_budget and strat["method"] != "morpho":
        dwell_map = match_budget(dwell_map, mask_eff, fine["data"], next_px, args.fine_dwell)
    plan = build_scan_plan(mask_eff, dwell_map, fine["data"], next_px,
                           group=(strat["method"] != "temporal"),
                           setup_ms=args.setup_ms, margin_fine=args.margin_fine,
                           fine_dwell=args.fine_dwell)
    # Fill for unscanned pixels in the MSE bias term = the fine truth block-
    # averaged to coarse resolution (same intensity scale; the raw coarse scan
    # sits on a different scale, see utils.validation_utils.coarse_blockmean).
    coarse_fill = coarse_blockmean(fine["comp"], coarse0["data"], fine["data"])
    emit_outputs(f"{sample}_simulate", fine["comp"], fine["data"], mask_eff,
                 dwell_map, plan, strat, desc, next_px, args,
                 ground_truth_comp=fine["comp"], per_level_stats=per_level_stats,
                 coarse_fill=coarse_fill)


def run_step(args):
    """Stepwise mode: step 0 (from --coarse) or resume (from --scan + --resume)."""
    if args.resume:
        state = pickle.load(open(args.resume, "rb"))
        scan_path = require(args.scan, "acquired scan HDF5 (--scan)")
        data = load_xrf(scan_path)
        comp, _, _ = get_composite_map(data, state["channels"])
        prev_data = {"xdata": state["xdata"], "ydata": state["ydata"]}
        prev_mask = state["mask"]
        idx = state["done_idx"] + 1
        ladder = state["ladder"]
        desc, strat = state["desc"], state["strategy"]
        sample = state["sample"]
        cur_px = px_um_of(data)
        # Guard: the acquired scan must match the level this step expects, else the
        # cascade silently mis-steps (project a mask onto the wrong grid).
        if idx >= len(ladder):
            raise SystemExit(f"Cascade already complete ({len(ladder)} levels). "
                             f"Nothing left to scan.")
        if cur_px != ladder[idx] and not args.force_level:
            raise SystemExit(
                f"--scan is {cur_px}um but step {idx} expects {ladder[idx]}um "
                f"(ladder {ladder}). Provide the {ladder[idx]}um scan, or pass "
                f"--force-level to override.")
        print(f"\n{'='*70}\n  XRF ITERATIVE PLANNER - STEP {idx} @ {cur_px}um  ({sample})"
              f"\n{'='*70}")
        print(f"  Resumed from {Path(args.resume).name}")
        mask, _, st = refine_step(comp, data, prev_mask=prev_mask,
                                  prev_data=prev_data, kernel_px=args.kernel,
                                  mode=args.mode, level=args.level)
    else:
        coarse_path = require(args.coarse_pos or args.coarse or find_coarse(),
                              "coarse HDF5")
        data = load_xrf(coarse_path)
        channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
        comp, used, _ = get_composite_map(data, channels)
        sample = args.sample or sample_hint_from_path(coarse_path) or Path(coarse_path).stem
        cur_px = px_um_of(data)
        ladder = resolve_ladder(args, sample, cur_px)
        idx = ladder.index(cur_px) if cur_px in ladder else 0
        desc = describe_composite(comp)
        strat = resolve_strategy(desc, args.strategy, args.dwell_strategy)
        state = {"channels": used}
        print(f"\n{'='*70}\n  XRF ITERATIVE PLANNER - STEP 0 @ {cur_px}um  ({sample})"
              f"\n{'='*70}")
        print(f"  Input: {coarse_path}")
        print(f"  Ladder: {' -> '.join(f'{p}um' for p in ladder)}")
        print(f"  Strategy: {strat['method']} / {strat['dwell']}")
        print(f"    why method: {strat['why_method']}")
        print(f"    why dwell : {strat['why_dwell']}")
        mask, _, st = refine_step(comp, data, prev_mask=None, kernel_px=args.kernel,
                                  mode=args.mode, level=args.level)

    print(f"  Footprint: in-mask {st['area_frac']*100:.1f}%  "
          f"signal {st['signal_in_frac']*100:.1f}%")

    next_px = next_px_mm_for(ladder, idx, args, cur_px)
    final_level = idx + 1 >= len(ladder) and len(ladder) > 1
    mask_eff, dwell_map = make_dwell_map(comp, data, mask, strat, args)
    if not args.no_match_budget and strat["method"] != "morpho":
        dwell_map = match_budget(dwell_map, mask_eff, data, next_px, args.fine_dwell)
    plan = build_scan_plan(mask_eff, dwell_map, data, next_px,
                           group=(strat["method"] != "temporal"),
                           setup_ms=args.setup_ms, margin_fine=args.margin_fine,
                           fine_dwell=args.fine_dwell)
    # Per-step stem so each level's plan is archived (not overwritten).
    stem = f"{sample}_step{idx}_{cur_px}um"
    summary = emit_outputs(stem, comp, data, mask_eff, dwell_map, plan, strat,
                           desc, next_px, args)
    _append_manifest(sample, idx, cur_px, strat, summary, fresh=not args.resume)

    # Persist state so the next acquired level can resume this cascade.
    if not final_level:
        state.update({
            "sample": sample, "ladder": ladder, "done_idx": idx,
            "channels": state["channels"], "desc": desc, "strategy": strat,
            "mask": mask, "xdata": data["xdata"], "ydata": data["ydata"],
            "px_um": cur_px,
        })
        state_path = OUTPUT_DIR / f"{sample}_cascade_state.pkl"
        OUTPUT_DIR.mkdir(exist_ok=True)
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Cascade state     -> {state_path}  (next: acquire @ "
              f"{next_px*1000:.0f}um, then --scan it with --resume)")
    else:
        print(f"  Cascade complete (finest level reached).")


def main():
    args = parse_args()
    if args.simulate:
        run_simulate(args)
    else:
        run_step(args)


if __name__ == "__main__":
    main()
