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

Outputs (per step): a text report, a JSON scan plan, a SMAK dataqueue region file (.dpm)
(rectangles at a uniform dwell = MEAN of the desired per-pixel dwells), the
exact per-pixel dwell map (<stem>_dwell_map.npy), and a figure.
"""

import argparse
import csv
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
from utils.strategy import (describe_composite, choose_strategy,
                            choose_element_strategy, MAX_DETECT_RATIO)
from utils.dwell import allocate_dwell
from utils.cascade import (find_level_file, available_levels, refine_step,
                           refine_cascade, dwell_ms_from_path)
from utils.roi_utils import (label_rois, get_bounding_boxes, group_rois, add_margin,
                             element_roi_mask)
from utils.validation_utils import (project_coarse_to_fine,
                                    project_mask,
                                     travel_overhead_from_centers)
from utils.quality import poisson_mse, poisson_mse_montecarlo, mean_snr
from utils.plotting import save_and_show
from utils.smak_export import write_smak_regions
from utils.reconstruction import calibrate_scale, reconstruct
from utils.line_scan import (plan_lines, raster_line_time, fit_to_budget,
                             export_line_plan)
from utils.paths import (find_coarse, require, sample_hint_from_path,
                         PROJECT_ROOT, DATA_DIR)

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
    p.add_argument("--target", default=None,
                   help="Single target element (e.g. Fe.Ka): footprint = that "
                        "element's hot spots, ladder coarse -> detect -> fine "
                        "(see utils.strategy.choose_element_strategy)")
    p.add_argument("--roi-k", default=0.5, type=float,
                   help="--target: MAD multiplier of the element ROI threshold. "
                        "0.5 (default) keeps hot-spot recall within ~3 points "
                        "of a raster; 1.0 maximises Poisson MSE but misses up "
                        "to 16 %% of the weak-element hot spots (report sec. 16)")
    p.add_argument("--max-detect-ratio", default=MAX_DETECT_RATIO, type=float,
                   help="--target: skip area only when the detection pixel is "
                        "<= this many times the final pixel (default 4)")
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
    p.add_argument("--dwell-reg", default=1.0, type=float,
                   help="sqrt dwell regulariser: allocate on signal + reg*mean "
                        "(robust to shot-noise-dominated estimates; 0 = plain sqrt)")
    p.add_argument("--dwell-low", default=1.0, type=float)
    p.add_argument("--dwell-high", default=50.0, type=float)
    p.add_argument("--setup-ms", default=500.0, type=float)
    p.add_argument("--no-match-budget", action="store_true",
                   help="Don't normalize dwells to the raster total time")
    p.add_argument("--margin-fine", default=5, type=int,
                   help="Safety margin added around grouped ROIs [fine pixels]")
    p.add_argument("--mc-draws", default=0, type=int,
                   help="Poisson Monte-Carlo draws for the MSE metric (0 = analytic only)")
    p.add_argument("--coarse-scale", default=None, type=float,
                   help="coarse->fine count scale for the reconstruction of the "
                        "skipped pixels (default: measured on the other samples "
                        "in data/, leave-one-sample-out)")
    p.add_argument("--no-show", action="store_true",
                   help="Save figures without opening a window")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

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
    """
    Assemble one scan-plan region. The scanner rasters the rectangle at ONE
    dwell, so `dwell_uniform_ms` uses the MEAN of the desired per-pixel dwells:
    that keeps the region's total time equal to the intended budget (the median
    would collapse to the background level and drop the bright pixels). The exact
    per-pixel dwell map is saved separately (<stem>_dwell_map.npy) for execution
    on a scanner that supports a dwell map.
    """
    if dwell_vals.size:
        mean = float(dwell_vals.mean())
        med, lo, hi = (float(np.median(dwell_vals)),
                       float(dwell_vals.min()), float(dwell_vals.max()))
    else:
        mean = med = lo = hi = fine_dwell
    return {
        "region_id":        rid,
        "x_start_mm":       float(min(x0, x1)),
        "x_end_mm":         float(max(x0, x1)),
        "y_start_mm":       float(min(y0, y1)),
        "y_end_mm":         float(max(y0, y1)),
        "fine_px_mm":       next_px_mm,
        "n_pixels_x":       int(np.ceil(abs(x1 - x0) / next_px_mm)) + 1,
        "n_pixels_y":       int(np.ceil(abs(y1 - y0) / next_px_mm)) + 1,
        "dwell_mean_ms":    mean,
        "dwell_median_ms":  med,
        "dwell_min_ms":     lo,
        "dwell_max_ms":     hi,
        "dwell_uniform_ms": mean,    # what the scanner actually uses
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
    # size_px_mm: get_bounding_boxes counts `size` in CURRENT-grid pixels;
    # group_rois converts them to next_px (fine) units for the gap cost.
    grouped = group_rois(raw, dwell_ms=med_dwell, dx=next_px_mm, dy=next_px_mm,
                         setup_ms=setup_ms, size_px_mm=(grid["dx"], grid["dy"]))
    boxed = add_margin(grouped, coarse_px_mm=grid["dx"], fine_px_mm=next_px_mm,
                       n_fine=margin_fine)
    plan = []
    for i, b in enumerate(boxed, 1):
        x0, x1, y0, y1 = b["mm"]
        vals, (x0c, x1c, y0c, y1c) = _box_dwell_vals(mask, dwell_map, grid,
                                                     x0, x1, y0, y1)
        plan.append(_region_dict(i, x0c, x1c, y0c, y1c, vals, next_px_mm, fine_dwell))
    return plan


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
                                      args.dwell_high, mask=None,
                                      reg=args.dwell_reg)
    elif method == "morpho":
        mask_eff = mask
        dwell_map = np.full_like(comp, args.fine_dwell, dtype=float)
    else:  # roi / combined : footprint + variable dwell
        mask_eff = mask
        dwell_map, _ = allocate_dwell(comp, strat["dwell"], args.dwell_low,
                                      args.dwell_high, mask=mask_eff,
                                      reg=args.dwell_reg)
    return mask_eff, dwell_map


def raster_time_ms(grid, next_px_mm, fine_dwell):
    """Full-frame raster baseline [ms] at next_px_mm resolution.
    Single definition shared by match_budget and emit_outputs, so the budget
    matching and the reported speedup always use the same baseline."""
    # Same pixel-count rule as a full-frame region in _region_dict, so a
    # full-frame plan reports speedup 1.00x (not 1.02x from a bigger baseline).
    w = abs(grid["xdata"].max() - grid["xdata"].min())
    h = abs(grid["ydata"].max() - grid["ydata"].min())
    nx_full = int(np.ceil(w / next_px_mm)) + 1
    ny_full = int(np.ceil(h / next_px_mm)) + 1
    return nx_full * ny_full * fine_dwell


def match_budget(dwell_map, mask_eff, grid, next_px_mm, fine_dwell,
                 dwell_low=None, dwell_high=None, spent_ms=0.0):
    """
    Scale in-mask dwells so the in-mask pixel time equals the raster baseline
    at next_px_mm, then clip to [dwell_low, dwell_high]. Returns the scaled
    dwell_map.

    spent_ms (time already used by earlier cascade levels) is deducted, so the
    whole adaptive acquisition matches ONE raster at next_px_mm.

    This matches the per-pixel budget only: the emitted plan scans whole
    margined bounding boxes plus travel/setup, so its speedup can still be < 1
    (and clipping can leave part of the budget unused).
    """
    raster_total_ms = raster_time_ms(grid, next_px_mm, fine_dwell)
    budget_ms = raster_total_ms - spent_ms
    if budget_ms <= 0:
        raise SystemExit(f"Earlier levels already used {spent_ms/1000:.0f}s, more "
                         f"than the {raster_total_ms/1000:.0f}s raster budget")

    replicas = (grid["dx"] / next_px_mm) * (grid["dy"] / next_px_mm)
    current_total_ms = float((dwell_map * mask_eff).sum()) * replicas
    if current_total_ms > 0:
        scale = budget_ms / current_total_ms
        dwell_map = np.where(mask_eff, dwell_map * scale, dwell_map)
        if dwell_low is not None or dwell_high is not None:
            dwell_map = np.where(mask_eff,
                                 np.clip(dwell_map, dwell_low, dwell_high),
                                 dwell_map)
        used = float((dwell_map * mask_eff).sum()) * replicas / raster_total_ms
        print(f"  Budget matched: dwells scaled by {scale:.3f} "
              f"(this level = {used*100:.0f}% of raster, earlier levels "
              f"{spent_ms/raster_total_ms*100:.0f}%)")
    return dwell_map


def line_scan_plan(grid, mask, dwell_map, next_px_mm, args, spent_ms=0.0):
    """Per-line variable-dwell plan for the NEXT level (utils.line_scan).

    The dwell map lives on the current grid; it is resampled (nearest) onto the
    next level's pixel grid, then - unless --no-match-budget - renormalised so
    the line-scan total (ramps, gap crossings, row moves) plus the earlier
    levels equals a raster of the next level timed with the same model.
    Returns (fine_x, fine_y, requested_dwell, plan, raster_ms).
    """
    x, y = np.asarray(grid["xdata"]), np.asarray(grid["ydata"])
    nx = int(round(abs(x.max() - x.min()) / next_px_mm)) + 1
    ny = int(round(abs(y.max() - y.min()) / next_px_mm)) + 1
    xf = x.min() + np.arange(nx) * next_px_mm
    yf = y.min() + np.arange(ny) * next_px_mm
    m = project_mask(mask, x, y, xf, yf)
    d = project_mask(np.where(mask, dwell_map, 0.0), x, y, xf, yf)
    d = np.where(m, d, 0.0)
    raster_ms = raster_line_time((ny, nx), xf, yf, args.fine_dwell)
    if args.no_match_budget:
        plan = plan_lines(d, xf, yf)
    else:
        d, plan = fit_to_budget(d, raster_ms - spent_ms, xf, yf,
                                lo=args.dwell_low, hi=args.dwell_high)
    return xf, yf, d, plan, raster_ms


def _json_safe(obj):
    """Recursively convert numpy types to JSON; non-finite floats -> null
    (json.dump would otherwise write Infinity/NaN, which is not valid JSON)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def emit_outputs(stem, comp, grid, mask, dwell_map, plan, strat, desc,
                 next_px_mm, args, ground_truth_comp=None, per_level_stats=None,
                 coarse_fill=None, spent_ms=0.0, spent_line_ms=None):
    """Print the report, save JSON + SMAK .dpm regions + figure for one step.
    Returns the summary dict (so callers can log a cascade manifest)."""
    # Time estimate = pixel dwell time + inter-region travel/setup overhead.
    # The overhead uses the shared travel model (trapezoidal profile + per-region
    # setup) over the ACTUAL scan regions' centers, instead of a flat per-region
    # setup — single source of truth with scripts 03/04/06/08.
    scan_time_s = sum(r["dwell_uniform_ms"] * r["n_pixels_x"] * r["n_pixels_y"]
                      for r in plan) / 1000.0
    centers = np.array([[(r["x_start_mm"] + r["x_end_mm"]) / 2.0,
                         (r["y_start_mm"] + r["y_end_mm"]) / 2.0] for r in plan])
    overhead = travel_overhead_from_centers(centers, setup_ms=args.setup_ms)
    level_time_s = scan_time_s + overhead["overhead_ms"] / 1000.0
    # Earlier cascade levels count against the speedup (they were acquired to
    # make this plan possible); a plain raster needs none of them.
    total_time_s = level_time_s + spent_ms / 1000.0
    raster_time_s = raster_time_ms(grid, next_px_mm, args.fine_dwell) / 1000.0
    speedup = raster_time_s / total_time_s if total_time_s > 0 else np.inf

    print(f"\n  === SCAN PLAN (next level @ {next_px_mm*1000:.0f}um) ===")
    print(f"    Regions to scan   : {len(plan)}")
    print(f"    Total fine pixels : ~{sum(r['n_pixels_x']*r['n_pixels_y'] for r in plan):,}")
    print(f"    Estimated time    : {level_time_s:.0f} s ({level_time_s/60:.1f} min)")
    print(f"      travel + setup  : {overhead['overhead_ms']/1000:.1f} s  "
          f"(travel {overhead['travel_ms']/1000:.1f}s over {overhead['n_regions']} regions)")
    if spent_ms:
        print(f"      + earlier levels: {spent_ms/1000:.0f} s  "
              f"-> total {total_time_s:.0f} s")
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
        # Zero the dwell outside the scan footprint (same as the saved .npy):
        # unscanned pixels must contribute zero SNR, otherwise the gain is
        # overstated for sparse footprints (and pinned at 1.0x for morpho).
        snr_ad = mean_snr(ground_truth_comp, np.where(mask, dwell_map, 0.0),
                          t_ref=args.fine_dwell)
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
        "estimate": {"total_time_s": total_time_s, "level_time_s": level_time_s,
                     "earlier_levels_s": spent_ms / 1000.0,
                     "raster_time_s": raster_time_s,
                     "speedup": speedup, "n_regions": len(plan),
                     "travel_ms": overhead["travel_ms"],
                     "setup_total_ms": overhead["setup_total_ms"]},
        "quality": quality,
        # per-level masks are ndarrays (kept in memory for the figures only)
        "cascade_levels": [{k: v for k, v in st.items() if k != "mask"}
                           for st in (per_level_stats or [])] or None,
        "regions": plan,
    }
    # Per-line variable-dwell plan: what the 7-2 scanner can execute
    # (Boltz et al. 2026); the rectangles above are the fallback.
    spent_line = spent_ms if spent_line_ms is None else spent_line_ms
    xf, yf, d_line, lplan, ras_line = line_scan_plan(grid, mask, dwell_map,
                                                     next_px_mm, args, spent_line)
    scanned_f = lplan["dwell_ms"] > 0
    mean_dwell = float(lplan["dwell_ms"][scanned_f].mean()) if scanned_f.any() else 0.0
    line_total_s = (lplan["total_ms"] + spent_line) / 1000.0
    print(f"\n  === LINE-SCAN PLAN (variable dwell per pixel, ramps included) ===")
    print(f"    Rows flown        : {int((lplan['x0'] >= 0).sum())} / {len(yf)}")
    print(f"    Pixels measured   : {scanned_f.mean()*100:.1f}% of the frame, mean "
          f"dwell {mean_dwell:.1f} ms (raster {args.fine_dwell:.0f} ms)")
    print(f"    Time              : {lplan['total_ms']/1000:.0f} s + earlier levels "
          f"{spent_line/1000:.0f} s = {line_total_s:.0f} s  (raster, same model: "
          f"{ras_line/1000:.0f} s)")
    print(f"      overheads       : row moves {lplan['move_ms']/1000:.0f} s, gap "
          f"crossings {(lplan['crossing_ms'][~scanned_f].sum())/1000:.0f} s, "
          f"ramp extra dwell {lplan['extra_dwell']*100:.1f}%")
    if not lplan.get("converged", True):
        print("    WARNING: the dwell bounds make the equal-time budget unreachable "
              "(ROI too small or too large for [dwell-low, dwell-high]); the plan "
              "takes the time printed above, not the raster time")
    line_path = OUTPUT_DIR / f"{stem}_line_plan.npz"
    export_line_plan(line_path, lplan, xf, yf, d_line,
                     meta={"stem": stem, "px_mm": next_px_mm,
                           "raster_ms": ras_line, "earlier_levels_ms": spent_line})
    summary["estimate"].update({
        "line_scan_total_s": line_total_s,
        "line_scan_raster_s": ras_line / 1000.0,
        "line_scan_budget_matched": bool(lplan.get("converged", True)),
        "line_scan_measured_frac": float(scanned_f.mean()),
        "line_scan_mean_dwell_ms": mean_dwell,
        "line_scan_move_s": lplan["move_ms"] / 1000.0,
        "line_scan_extra_dwell": lplan["extra_dwell"]})
    print(f"  Line plan saved   -> {line_path}")

    json_path = OUTPUT_DIR / f"{stem}_scan_plan.json"
    with open(json_path, "w") as f:
        json.dump(_json_safe(summary), f, indent=2)
    print(f"\n  JSON plan saved   -> {json_path}")

    # SMAK "Export scan area to dataqueue" format (see utils.smak_export)
    dpm_path = OUTPUT_DIR / f"{stem}_scan_plan.dpm"
    write_smak_regions(dpm_path, plan, grid.get("energy", 0.0), stem)
    print(f"  SMAK regions saved -> {dpm_path}")

    # Exact per-pixel dwell map (0 = not scanned), on the current grid. The
    # region .pkl above is a uniform-per-rectangle approximation; this preserves
    # the true allocation for a scanner that can execute a dwell map.
    npy_path = OUTPUT_DIR / f"{stem}_dwell_map.npy"
    np.save(npy_path, np.where(mask, dwell_map, 0.0))
    print(f"  Dwell map saved   -> {npy_path}")

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
            return element_ladder(lv, args.max_detect_ratio) if args.target else lv
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

def plan_and_emit(dwell_signal, comp, grid, mask, strat, desc, next_px, stem,
                  args, spent_ms=0.0, spent_line_ms=None, **gt_kwargs):
    """
    Shared planning tail for BOTH modes: dwell map -> optional budget match ->
    scan plan -> outputs. Keeping this in one place guarantees the offline
    simulation validates exactly the plan the stepwise (beamline) mode emits.

    `dwell_signal` is the interest estimate the dwells are computed from (the
    current composite in stepwise mode; the projected previous level in
    simulate mode, so the dwell map never "knows" the fine truth).
    """
    mask_eff, dwell_map = make_dwell_map(dwell_signal, grid, mask, strat, args)
    if not args.no_match_budget and strat["method"] != "morpho":
        dwell_map = match_budget(dwell_map, mask_eff, grid, next_px,
                                 args.fine_dwell, args.dwell_low, args.dwell_high,
                                 spent_ms=spent_ms)
    plan = build_scan_plan(mask_eff, dwell_map, grid, next_px,
                           group=(strat["method"] != "temporal"),
                           setup_ms=args.setup_ms, margin_fine=args.margin_fine,
                           fine_dwell=args.fine_dwell)
    return emit_outputs(stem, comp, grid, mask_eff, dwell_map, plan, strat,
                        desc, next_px, args, spent_ms=spent_ms,
                        spent_line_ms=spent_line_ms, **gt_kwargs)


def coarse_scale_for(args, sample, channels, coarse_px, fine_px):
    """--coarse-scale, else the leave-one-sample-out calibration."""
    if args.coarse_scale is not None:
        return args.coarse_scale
    s, n = calibrate_scale(sample, channels, coarse_px, fine_px)
    if n == 0:
        print(f"  WARNING: no other sample with {coarse_px}um and {fine_px}um "
              f"scans to calibrate the reconstruction; using scale 1.0 "
              f"(pass --coarse-scale)")
    else:
        print(f"  Reconstruction scale {coarse_px}->{fine_px}um: {s:.3f} "
              f"(from {n} other sample{'s' if n > 1 else ''})")
    return s


def save_reconstruction(stem, recon, scanned, grid, args):
    """Write the reconstructed fine map (.npy, counts at the fine dwell) and a
    figure with the scanned area outlined."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    npy = OUTPUT_DIR / f"{stem}_reconstruction.npy"
    np.save(npy, recon)
    fig, ax = plt.subplots(figsize=(8, 6))
    ext = [grid["xdata"].min(), grid["xdata"].max(),
           grid["ydata"].min(), grid["ydata"].max()]
    im = ax.imshow(np.log1p(np.maximum(recon, 0)), origin="lower", extent=ext,
                   cmap="magma", aspect="equal")
    if scanned.any() and not scanned.all():
        ax.contour(grid["xdata"], grid["ydata"], scanned.astype(float), [0.5],
                   colors="cyan", linewidths=0.6)
    ax.set_title(f"{stem}\nreconstruction (cyan: scanned at fine resolution)",
                 fontsize=10)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, label="log(1 + counts)")
    save_and_show(fig, OUTPUT_DIR / f"{stem}_reconstruction.png", show=not args.no_show)
    print(f"  Reconstruction    -> {npy}")


def element_ladder(levels, max_ratio=MAX_DETECT_RATIO):
    """coarse -> detect -> fine for --target: the cheapest intermediate level
    fine enough for spatial skipping (e.g. 250 -> 100 -> 25, skipping 50)."""
    if len(levels) < 3:
        return list(levels)
    final = levels[-1]
    ok = [px for px in levels[1:-1] if px <= max_ratio * final]
    return [levels[0], ok[0], final] if ok else [levels[0], final]


def level_time_ms(d, n_pixels=None):
    """Acquisition time of a scan [ms] (n_pixels defaults to the full frame)."""
    n = d["mapdata"].shape[0] * d["mapdata"].shape[1] if n_pixels is None else n_pixels
    return n * dwell_ms_from_path(d["filepath"])


def run_simulate_target(args, sample):
    """Offline validation of the element-targeted cascade (--target)."""
    ladder = args.levels or element_ladder(available_levels(sample, data_dir=DATA_DIR),
                                           args.max_detect_ratio)
    if len(ladder) < 2:
        raise SystemExit(f"Need >=2 levels for a cascade; found {ladder} for {sample}")
    print(f"\n{'='*70}\n  XRF ELEMENT PLANNER - SIMULATE  ({sample}, {args.target})"
          f"\n{'='*70}")
    print(f"  Ladder: {' -> '.join(f'{p}um' for p in ladder)}")
    levels = []
    for px in ladder:
        path = find_level_file(sample, px, data_dir=DATA_DIR)
        if path is None:
            raise SystemExit(f"No file for {sample} @ {px}um under {DATA_DIR}")
        d = load_xrf(path)
        comp, _, _ = get_composite_map(d, [args.target])
        levels.append({"data": d, "comp": comp, "px": px})
        print(f"  {px}um: {path.name}  {comp.shape}")

    fine, prev = levels[-1], levels[-2]
    strat = choose_element_strategy(prev["px"], fine["px"], args.max_detect_ratio)
    if args.dwell_strategy != "auto":
        strat["dwell"] = args.dwell_strategy
    print(f"\n  Strategy: {strat['method']} / {strat['dwell']}  ({strat['why_method']})")
    if strat["method"] == "roi":
        m_prev = element_roi_mask(prev["comp"], args.roi_k)
        mask = project_mask(m_prev, prev["data"]["xdata"], prev["data"]["ydata"],
                            fine["data"]["xdata"], fine["data"]["ydata"])
        print(f"  {args.target} ROIs @ {prev['px']}um: {m_prev.mean()*100:.1f}% of area")
    else:
        mask = np.ones_like(fine["comp"], dtype=bool)
    # intermediate levels are full frames in this mode
    spent = sum(level_time_ms(lv["data"]) for lv in levels[:-1])
    spent_line = sum(raster_line_time(lv["comp"].shape, lv["data"]["xdata"],
                                      lv["data"]["ydata"],
                                      dwell_ms_from_path(lv["data"]["filepath"]))
                     for lv in levels[:-1])
    dwell_signal = project_coarse_to_fine(prev["comp"], prev["data"], fine["data"])
    # Unscanned fine pixels: graph-harmonic interpolation of the last
    # full-frame level (a measured scan - never the fine truth).
    scale = coarse_scale_for(args, sample, [args.target], prev["px"], fine["px"])
    recon, coarse_fill = reconstruct(
        prev["comp"], prev["data"], fine["data"], mask, fine["comp"], scale,
        dwell_ms_from_path(prev["data"]["filepath"]), args.fine_dwell)
    desc = describe_composite(levels[0]["comp"])
    stem = f"{sample}_{args.target}_simulate"
    plan_and_emit(dwell_signal, fine["comp"], fine["data"], mask, strat, desc,
                  fine["data"]["dx"], stem, args,
                  spent_ms=spent, spent_line_ms=spent_line,
                  ground_truth_comp=fine["comp"],
                  coarse_fill=coarse_fill)
    save_reconstruction(stem, recon, mask, fine["data"], args)


def run_simulate(args):
    sample = args.sample or sample_hint_from_path(args.coarse_pos or args.coarse or "")
    if not sample:
        raise SystemExit("--simulate needs --sample (or a coarse path to infer it from)")
    if args.target:
        return run_simulate_target(args, sample)
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
    # Dwell must be decided from the most informative ALREADY-ACQUIRED level
    # (the one before the final), NOT from the fine truth — otherwise the dwell
    # map "knows" the answer and the MSE/SNR validation is optimistically biased.
    prev = scans_list[-2]
    dwell_signal = project_coarse_to_fine(prev["comp"], prev["data"], fine["data"])
    next_px = fine["data"]["dx"]   # finest level scans at its own resolution
    # Fill for unscanned pixels = graph-harmonic interpolation of the coarsest
    # (full-frame) scan, calibrated on the other samples. It used to be the
    # block mean of the fine truth, which leaks the answer and assumes coarse
    # pixels integrate their whole cell (they do not; report section 14).
    scale = coarse_scale_for(args, sample, ref_channels, ladder[0], ladder[-1])
    recon, coarse_fill = reconstruct(
        coarse0["comp"], coarse0["data"], fine["data"], mask, fine["comp"], scale,
        dwell_ms_from_path(coarse0["data"]["filepath"]), args.fine_dwell)
    # pixels actually acquired at every level before the final one
    spent = sum(level_time_ms(lv["data"], st["n_scanned"])
                for lv, st in zip(scans_list[:-1], per_level_stats[:-1]))
    plan_and_emit(dwell_signal, fine["comp"], fine["data"], mask, strat, desc,
                  next_px, f"{sample}_simulate", args, spent_ms=spent,
                  ground_truth_comp=fine["comp"], per_level_stats=per_level_stats,
                  coarse_fill=coarse_fill)
    save_reconstruction(f"{sample}_simulate", recon, mask, fine["data"], args)


def _final_reconstruction(args, state, data, comp, cur_px):
    """Last level acquired: assemble the final map (fine scan where it was
    planned, graph-harmonic interpolation of the previous level elsewhere)."""
    if "comp" not in state:
        raise SystemExit("This state file predates the reconstruction step; "
                         "re-run the cascade from step 0.")
    prev = {"xdata": state["xdata"], "ydata": state["ydata"]}
    if state["strategy"]["method"] == "temporal":
        scanned = np.ones_like(comp, dtype=bool)
    else:
        scanned = project_mask(state["mask"], prev["xdata"], prev["ydata"],
                               data["xdata"], data["ydata"])
    print(f"\n  Final level @ {cur_px}um: scanned {scanned.mean()*100:.1f}% of the frame")
    scale = coarse_scale_for(args, state["sample"], state["channels"],
                             state["px"], cur_px)
    recon, _ = reconstruct(state["comp"], prev, data, scanned, comp, scale,
                           state["dwell_ms"], dwell_ms_from_path(data["filepath"]))
    stem = f"{state['sample']}_step{state['done_idx'] + 1}_{cur_px}um"
    save_reconstruction(stem, recon, scanned, data, args)
    print("  Cascade complete.")


def _target_step(comp, cur_px, idx, ladder, args):
    """--target footprint/strategy for one stepwise level.
    Before the detection level: plan a uniform full-frame raster (the cheap
    detection scan). At the detection level: element ROIs if they are fine
    enough for the final pixel, else a full-frame sqrt-dwell scan."""
    full = np.ones_like(comp, dtype=bool)
    n_tot = comp.size
    if idx + 2 < len(ladder):
        strat = {"method": "morpho", "dwell": "uniform",
                 "why_method": f"detection scan: full frame @ {ladder[idx+1]}um",
                 "why_dwell": "uniform reference dwell"}
        mask = full
    else:
        strat = choose_element_strategy(cur_px, ladder[-1], args.max_detect_ratio)
        if args.dwell_strategy != "auto":
            strat["dwell"] = args.dwell_strategy
        mask = element_roi_mask(comp, args.roi_k) if strat["method"] == "roi" else full
    total = float(comp.sum())
    st = {"area_frac": mask.sum() / n_tot,
          "signal_in_frac": float(comp[mask].sum()) / total if total > 0 else 0.0}
    return mask, strat, st


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
        spent = state.get("spent_ms", 0.0)
        spent_line = state.get("spent_line_ms", spent)
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
        if idx == len(ladder) - 1 and len(ladder) > 1:
            return _final_reconstruction(args, state, data, comp, cur_px)
        if not args.target:
            mask, _, st = refine_step(comp, data, prev_mask=prev_mask,
                                      prev_data=prev_data, kernel_px=args.kernel,
                                      mode=args.mode, level=args.level)
    else:
        coarse_path = require(args.coarse_pos or args.coarse or find_coarse(),
                              "coarse HDF5")
        data = load_xrf(coarse_path)
        channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
        if args.target:
            channels = [args.target]
        comp, used, _ = get_composite_map(data, channels)
        spent = level_time_ms(data)
        spent_line = raster_line_time(comp.shape, data["xdata"], data["ydata"],
                                      dwell_ms_from_path(data["filepath"]))
        sample = args.sample or sample_hint_from_path(coarse_path) or Path(coarse_path).stem
        cur_px = px_um_of(data)
        ladder = resolve_ladder(args, sample, cur_px)
        # Guard: the coarse scan must be a level of the ladder, else the cascade
        # is silently mislabeled (and every later --resume step is off by one).
        if cur_px in ladder:
            idx = ladder.index(cur_px)
        elif args.force_level:
            idx = 0
        else:
            raise SystemExit(
                f"Coarse scan is {cur_px}um but the ladder is {ladder} "
                f"(coarsest {ladder[0]}um). Pass --levels starting at {cur_px}um, "
                f"or --force-level to treat it as the first level.")
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
        if not args.target:
            mask, _, st = refine_step(comp, data, prev_mask=None,
                                      kernel_px=args.kernel,
                                      mode=args.mode, level=args.level)

    if args.target:
        mask, strat, st = _target_step(comp, cur_px, idx, ladder, args)
        print(f"  {args.target}: {strat['method']} / {strat['dwell']}  "
              f"({strat['why_method']})")
    print(f"  Footprint: in-mask {st['area_frac']*100:.1f}%  "
          f"signal {st['signal_in_frac']*100:.1f}%")

    next_px = next_px_mm_for(ladder, idx, args, cur_px)
    final_level = idx + 1 >= len(ladder) and len(ladder) > 1
    # Per-step stem so each level's plan is archived (not overwritten).
    stem = f"{sample}_step{idx}_{cur_px}um"
    # Only the final level is budget-matched against a fine raster, so only
    # then do the earlier acquisitions count against it.
    plans_final = idx + 2 == len(ladder)
    summary = plan_and_emit(comp, comp, data, mask, strat, desc, next_px,
                            stem, args, spent_ms=spent if plans_final else 0.0,
                            spent_line_ms=spent_line if plans_final else 0.0)
    _append_manifest(sample, idx, cur_px, strat, summary, fresh=not args.resume)

    # Persist state so the next acquired level can resume this cascade.
    # ('channels' is already in the state dict from step 0 / the previous resume.)
    if not final_level:
        state.update({
            "sample": sample, "ladder": ladder, "done_idx": idx,
            "desc": desc, "strategy": strat,
            "mask": mask, "xdata": data["xdata"], "ydata": data["ydata"],
            # this level's map, kept to reconstruct the skipped fine pixels
            "comp": comp, "px": cur_px, "dwell_ms": dwell_ms_from_path(data["filepath"]),
            # acquired so far + the level this plan is about to acquire
            "spent_ms": spent + summary["estimate"]["level_time_s"] * 1000.0,
            # same, timed as line scans (the next level is flown as lines)
            "spent_line_ms": spent_line
            + summary["estimate"]["line_scan_total_s"] * 1000.0
            - (spent_line if plans_final else 0.0),
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
