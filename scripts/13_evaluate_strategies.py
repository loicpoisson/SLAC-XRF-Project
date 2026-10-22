"""
Step-1 robust evaluation of the adaptive strategies (see utils/evaluation.py).

For each sample x target (element or composite) x strategy, at EQUAL TOTAL
TIME (fine raster at 10 ms, earlier coarse levels deducted):
  - decisions (mask, dwell) use ONLY the measured coarse scans (250 / 100 um)
  - unscanned pixels are filled with the measured coarse scan, rescaled by a
    factor calibrated on the OTHER samples
  - scores are computed against fine scans that played no part in the
    decisions: the 10 ms scan ("A", full frame) and, where it exists, the
    independent 25 ms scan ("B", lower noise; partial for UA1_P1)

Strategies
  raster        : uniform 10 ms everywhere (reference, ratio 1 by definition)
  sqrt250       : full frame, dwell ~ sqrt(250um estimate + mean)      [250 -> fine]
  sqrt100       : full frame, dwell ~ sqrt(100um estimate + mean)      [250 -> 100 -> fine]
  roi250        : element ROIs on the 250 um scan, sqrt dwell inside    [250 -> fine]
  roi100        : element ROIs on the 100 um scan, sqrt dwell inside    [250 -> 100 -> fine]
                  (= 10_run_pipeline --target)
  roi100_resbound : roi100 with unscanned pixels filled by the block mean of
                  the truth. NOT achievable, and NOT what a better reconstruction
                  of these scans can reach: the coarse scans do not integrate
                  their whole cell. It bounds what the resolution alone allows.

Criteria: Poisson MSE ratio (raster/adaptive, >1 = adaptive better) with a
block-bootstrap 95% CI; the same ratio in the sqrt-rate domain, where every
pixel weighs alike (the Poisson MSE is dominated by a few bright pixels, see
top1pct_share); hot-spot quantification error and detection recall/precision,
with hot spots counted as real only when present in both independent fine
scans. Scanned pixels are scored as ideal Poisson measurements of the truth.
Fly-scan edge columns (coarse columns 0, 1, last) are excluded.

Usage:
  python scripts/13_evaluate_strategies.py                     # all samples / targets
  python scripts/13_evaluate_strategies.py --samples UB1_P1 --targets Zn.Ka --draws 5
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.hdf5_reader import load_xrf, get_composite_map
from utils.cascade import find_level_file, dwell_ms_from_path
from utils.roi_utils import element_roi_mask
from scipy import ndimage

from utils.validation_utils import (project_mask, project_coarse_to_fine,
                                    coarse_blockmean, _nearest_idx)
from utils.evaluation import (budget_dwell, sqrt_weight, coarse_to_fine_scale,
                              pixel_errors, raster_errors, block_bootstrap_ratio,
                              hot_spots, simulate_tasks, register_rows,
                              sqrt_errors, top_share)
from utils.reconstruction import (coarse_rate, fill_bilinear, fill_harmonic,
                                  fill_kriging, calibrate_scale)
from utils.line_scan import raster_line_time, fit_to_budget
from utils.paths import PROJECT_ROOT, DATA_DIR

SAMPLES = {"UA1_P1": 25, "UB1_P1": 25, "FP1_P1": 50}   # finest full scan per sample
T_REF = 10.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--samples", nargs="+", default=list(SAMPLES))
    p.add_argument("--targets", nargs="+",
                   default=["Fe.Ka", "Zn.Ka", "Mn.Ka", "Cu.Ka", "composite"])
    p.add_argument("--roi-k", type=float, default=1.0)
    p.add_argument("--spot-k", type=float, default=3.0,
                   help="MAD multiplier defining true hot spots on the truth")
    p.add_argument("--draws", type=int, default=10, help="Monte-Carlo draws")
    p.add_argument("--boot", type=int, default=300, help="block-bootstrap resamples")
    p.add_argument("--step2", action="store_true",
                   help="add step-2 strategies: reconstructions of the unscanned "
                        "pixels (bilinear / harmonic / kriging) and recall-"
                        "preserving ROI rules (dwell floor, lower threshold)")
    p.add_argument("--kriging", action="store_true",
                   help="--step2: also residual kriging (~100 s per case, no gain "
                        "over bilinear on UB1 Cu)")
    p.add_argument("--floors", nargs="+", type=float, default=[1.0, 2.0, 3.0],
                   help="--step2: dwell floor [ms] outside the ROIs")
    p.add_argument("--time-model", choices=["pixels", "lines"], default="pixels",
                   help="'pixels': total time = sum of dwells (steps 1-2). "
                        "'lines': unidirectional line scans with acceleration "
                        "ramps, gap crossings and row moves, for the raster and "
                        "every strategy alike; dwells are renormalised so the "
                        "line-scan totals are equal (step 3)")
    p.add_argument("--jobs", type=int, default=6,
                   help="cases evaluated in parallel (processes)")
    p.add_argument("--out", default=None,
                   help="CSV path (default outputs/evaluation_step1.csv, or "
                        "outputs/evaluation_step2.csv with --step2)")
    return p.parse_args()


def _file(sample, px, dwell_ms=10):
    for p in sorted(Path(DATA_DIR).rglob(f"*{sample}_{px}um_{dwell_ms}ms_*.hdf5")):
        return p
    return None


def _maps(data, channels):
    comp, used, _ = get_composite_map(data, channels)
    return comp, used


def load_sample(sample, target):
    """Coarse (250, 100) and fine scans of one target, as count maps."""
    fine_px = SAMPLES[sample]
    paths = {"c250": find_level_file(sample, 250), "c100": find_level_file(sample, 100),
             "A": _file(sample, fine_px, 10), "B": _file(sample, fine_px, 25)}
    data = {k: load_xrf(p) for k, p in paths.items() if p is not None}
    channels = None if target == "composite" else [target]
    maps = {}
    comp, used = _maps(data["c250"], channels)
    maps["c250"] = comp
    for k in ("c100", "A", "B"):
        if k in data:
            maps[k] = _maps(data[k], used)[0]
    dwell = {k: dwell_ms_from_path(paths[k]) for k in data}
    return data, maps, dwell


def calibration(sample, target, level, fine_px):
    """Leave-one-sample-out coarse->fine scale (utils.reconstruction)."""
    ch = None if target == "composite" else [target]
    return calibrate_scale(sample, ch, level, fine_px, candidates=list(SAMPLES))[0]


def _edge_free(data, F):
    """Fine pixels whose coarse column is not a fly-scan edge column (0, 1, last):
    TIME there is 66x / 4x / 1.3x nominal (review F5)."""
    ok = np.ones(len(F["xdata"]), bool)
    for d in data:
        ix = _nearest_idx(d["xdata"], F["xdata"])
        n = len(d["xdata"])
        ok &= ~np.isin(ix, [0, 1, n - 1])
    return ok[None, :]


def _without_edges(cmap):
    """Coarse map with its edge columns set to the median (no fake ROIs)."""
    out = cmap.astype(float).copy()
    out[:, [0, 1, -1]] = np.median(cmap)
    return out


def _confirmed_spots(r_hat, other, valid, k):
    """Hot spots of the scored truth that also appear in the independent scan
    (review F4); without a second scan, all spots of the truth."""
    lab, n = hot_spots(r_hat, valid, k=k)
    if other is None or n == 0:
        return lab, n
    lab_o, _ = hot_spots(other, valid, k=k)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[(lab > 0) & (lab_o > 0)])] = True
    keep[0] = False
    return ndimage.label(keep[lab])


def evaluate(sample, target, args):
    data, maps, dwell = load_sample(sample, target)
    fine_px = SAMPLES[sample]
    F = data["A"]
    P = maps["A"].size
    T = P * T_REF
    # The full 100um frame is charged (conservative: a cascade could scan only
    # the 250um footprint at 100um).
    spent = {250: maps["c250"].size * dwell["c250"],
             100: maps["c250"].size * dwell["c250"] + maps["c100"].size * dwell["c100"]}
    edge_ok = _edge_free([data["c250"], data["c100"]], F)
    est, fill, crate, cvar = {}, {}, {}, {}
    for lv, key in ((250, "c250"), (100, "c100")):
        cmap = _without_edges(maps[key])
        s = calibration(sample, target, lv, fine_px)
        crate[lv] = coarse_rate(cmap, s, dwell[key])           # coarse grid
        cvar[lv] = np.maximum(cmap, 0.0) * s ** 2 / dwell[key] ** 2
        est[lv] = project_coarse_to_fine(crate[lv], data[key], F)
        fill[lv] = est[lv]
    full = np.ones(maps["A"].shape, bool)

    strategies = {"sqrt250": (full, 250), "sqrt100": (full, 100)}
    for lv, key in ((250, "c250"), (100, "c100")):
        m = element_roi_mask(_without_edges(maps[key]), args.roi_k)
        strategies[f"roi{lv}"] = (project_mask(m, data[key]["xdata"], data[key]["ydata"],
                                               F["xdata"], F["ydata"]), lv)
    strategies["roi100_resbound"] = strategies["roi100"]
    # step 2: (mask, level, fill method, dwell floor outside the mask)
    variants = {}
    if args.step2:
        for lv in (250, 100):
            for meth in (("bilinear", "harmonic", "harmonic_c")
                         + (("kriging",) if args.kriging else ())):
                variants[f"roi{lv}+{meth}"] = (strategies[f"roi{lv}"][0], lv, meth, 0.0)
        m05 = project_mask(element_roi_mask(_without_edges(maps["c100"]), 0.5),
                           data["c100"]["xdata"], data["c100"]["ydata"],
                           F["xdata"], F["ydata"])
        variants["roi100_k0.5"] = (m05, 100, "cell", 0.0)
        variants["roi100_k0.5+harmonic"] = (m05, 100, "harmonic", 0.0)
        variants["roi100_k0.5+harmonic_c"] = (m05, 100, "harmonic_c", 0.0)
        for fl in args.floors:
            variants[f"roi100_floor{fl:g}"] = (strategies["roi100"][0], 100, None, fl)

    rA = maps["A"] / dwell["A"]
    valid_A = (maps["A"].sum(axis=1) > 0)[:, None] & edge_ok & full
    # [name, rate, dwell, scoring mask, independent scan, hot-spot mask]
    truths = [["A", rA, dwell["A"], valid_A, None, valid_A]]
    if "B" in maps:
        # Register B to A on the all-element composite, then bring it to A's
        # intensity scale (flux differs by ~5%; review F1, F6).
        compA, usedA, _ = get_composite_map(data["A"])
        compB = _maps(data["B"], usedA)[0]
        _, reg_ok, shifts = register_rows(compB, compA)
        rB = np.empty_like(maps["B"], dtype=float)
        for y, sh in enumerate(shifts):
            rB[y] = np.roll(maps["B"][y], sh)
        rB /= dwell["B"]
        valid_B = (maps["B"].sum(axis=1) > 0)[:, None] & reg_ok & edge_ok
        rB *= rA[valid_B].mean() / max(rB[valid_B].mean(), 1e-12)
        truths[0][4] = rB
        truths[0][5] = valid_A & valid_B
        truths.append(["B", rB, dwell["B"], valid_B, rA, valid_A & valid_B])
        print(f"    B registered to A: x-shift {shifts.min()}..{shifts.max()} px")

    n_cell100 = int(round((100 / fine_px) ** 2))
    if args.time_model == "lines":
        lt = {k: raster_line_time(maps[k].shape, data[k]["xdata"], data[k]["ydata"],
                                  dwell[k]) for k in ("c250", "c100")}
        T_lines = raster_line_time(maps["A"].shape, F["xdata"], F["ydata"], T_REF)
        spent_l = {250: lt["c250"], 100: lt["c250"] + lt["c100"]}

    def to_time_model(d, lv):
        """Equal-time renormalisation under the line-scan model (or no-op).
        Returns the dwell actually OBTAINED (ramps can add dwell, and that
        time is paid in the budget, so it must also be scored)."""
        if args.time_model != "lines":
            return d
        _, plan = fit_to_budget(d, T_lines - spent_l[lv], F["xdata"], F["ydata"])
        return plan["dwell_ms"]
    rows = []
    for tname, r_hat, t_truth, valid, other, spot_valid in truths:
        var_hat = np.maximum(r_hat, 0.0) / t_truth
        # hot spots are confirmed on the area both fine scans cover
        lab, n_spots = _confirmed_spots(r_hat, other, spot_valid, args.spot_k)
        e_ras = raster_errors(r_hat, T_REF)
        q_ras = sqrt_errors(r_hat, t_truth, np.full_like(r_hat, T_REF), full, r_hat)
        base = simulate_tasks(r_hat, np.full_like(r_hat, T_REF), full, r_hat,
                              spot_valid, lab, n_spots, k=args.spot_k,
                              n_draws=args.draws)
        rows.append(_row(sample, target, tname, "raster", full, valid, 1.0, (1.0, 1.0),
                         1.0, top_share(e_ras, valid), base, n_spots))
        for name, (mask, lv) in strategies.items():
            d = to_time_model(budget_dwell(sqrt_weight(est[lv], mask), mask,
                                           T - spent[lv]), lv)
            if name.endswith("resbound"):
                # block mean of the truth: assumes each coarse cell integrated
                # its whole area, which these scans do not (review F2)
                pred, corr = coarse_blockmean(r_hat, data["c100"], F), 1.0 / n_cell100
            else:
                pred, corr = fill[lv], 0.0
            e_ad = pixel_errors(r_hat, var_hat, d, mask, pred, pred_corr=corr)
            ratio = e_ras[valid].sum() / e_ad[valid].sum()
            ci = block_bootstrap_ratio(e_ras, e_ad, valid, n_boot=args.boot)
            q_ad = sqrt_errors(r_hat, t_truth, d, mask, pred)
            sq_ratio = q_ras[valid].sum() / q_ad[valid].sum()
            task = simulate_tasks(r_hat, d, mask, pred, spot_valid, lab, n_spots,
                                  k=args.spot_k, n_draws=args.draws)
            rows.append(_row(sample, target, tname, name, mask, valid, ratio, ci,
                             sq_ratio, top_share(e_ad, valid), task, n_spots))
        rng = np.random.default_rng(0)
        # The simulated ROI acquisition that feeds the reconstruction is drawn
        # from the OTHER fine scan where one exists, so the fill never sees the
        # scanned values of the scored scan (step-2 review F1).
        meas_src = r_hat if other is None else np.where(spot_valid, other, r_hat)
        for name, (roi, lv, meth, floor) in variants.items():
            budget = T - spent[lv]
            if floor > 0:
                # everything measured: floor outside the ROIs, sqrt inside
                n_out = int((~roi).sum())
                in_budget = budget - floor * n_out
                assert in_budget >= 1.0 * roi.sum(), "floor leaves no ROI budget"
                d = budget_dwell(sqrt_weight(est[lv], roi), roi, in_budget)
                d = to_time_model(np.where(roi, d, floor), lv)
                mask, pred = full, est[lv]
            else:
                mask = roi
                d = to_time_model(budget_dwell(sqrt_weight(est[lv], mask), mask,
                                               budget), lv)
                # one simulated acquisition of the ROI feeds the reconstruction
                meas = np.where(mask, rng.poisson(np.maximum(meas_src, 0) * d)
                                / np.maximum(d, 1e-9), 0.0)
                key = "c250" if lv == 250 else "c100"
                if meth == "cell":
                    pred = fill[lv]
                elif meth == "bilinear":
                    pred = fill_bilinear(crate[lv], data[key], F)
                elif meth == "harmonic":
                    pred = fill_harmonic(crate[lv], data[key], F, mask, meas)
                elif meth == "harmonic_c":     # coarse anchors only
                    pred = fill_harmonic(crate[lv], data[key], F, mask,
                                         np.zeros_like(meas), w_fine=0.0)
                else:
                    pred = fill_kriging(crate[lv], data[key], F, mask, meas, d,
                                        cvar[lv])
            e_ad = pixel_errors(r_hat, var_hat, d, mask, pred)
            ratio = e_ras[valid].sum() / e_ad[valid].sum()
            ci = block_bootstrap_ratio(e_ras, e_ad, valid, n_boot=args.boot)
            q_ad = sqrt_errors(r_hat, t_truth, d, mask, pred)
            task = simulate_tasks(r_hat, d, mask, pred, spot_valid, lab, n_spots,
                                  k=args.spot_k, n_draws=args.draws)
            # area = fraction scanned at the ROI dwell (floor variants measure
            # 100 % of the pixels, the rest at the floor)
            rows.append(_row(sample, target, tname, name, roi, valid, ratio, ci,
                             q_ras[valid].sum() / q_ad[valid].sum(),
                             top_share(e_ad, valid), task, n_spots))
    return rows


def _row(sample, target, truth, name, mask, valid, ratio, ci, sq_ratio, top1,
         task, n_spots):
    return {"sample": sample, "target": target, "truth": truth, "strategy": name,
            "area": float(mask[valid].mean()), "mse_ratio": ratio,
            "ci_lo": ci[0], "ci_hi": ci[1], "sqrt_ratio": sq_ratio,
            "top1pct_share": top1, "quant_err": task["quant_err"],
            "recall": task["recall"], "precision": task["precision"],
            "recall_skipped": task["recall_skipped"],
            "mc_rel_std": task["mse_mc_std"] / task["mse_mc_mean"]
            if task["mse_mc_mean"] else np.nan, "n_spots": n_spots}


def main():
    args = parse_args()
    cases = [(s, t) for s in args.samples for t in args.targets]
    rows = []
    if args.jobs > 1 and len(cases) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(cases))) as ex:
            futs = [ex.submit(evaluate, s, t, args) for s, t in cases]
            for (s, t), f in zip(cases, futs):
                rows += f.result()
                print(f"  {s} {t} done", flush=True)
    else:
        for s, t in cases:
            print(f"  {s} {t} ...", flush=True)
            rows += evaluate(s, t, args)
    hdr = (f"{'sample':7} {'target':9} {'tr':2} {'strategy':20} {'area':>5} "
           f"{'MSE ratio [95% CI]':>21} {'sqrt':>5} {'top1%':>5} "
           f"{'quant':>6} {'recall':>6} {'prec':>5}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['sample']:7} {r['target']:9} {r['truth']:2} {r['strategy']:20} "
              f"{r['area']:5.0%} {r['mse_ratio']:6.2f} [{r['ci_lo']:5.2f},{r['ci_hi']:5.2f}] "
              f"{r['sqrt_ratio']:5.2f} {r['top1pct_share']:5.0%} "
              f"{r['quant_err']:6.1%} {r['recall']:6.0%} {r['precision']:5.0%}")
    name = "evaluation_step2.csv" if args.step2 else "evaluation_step1.csv"
    if args.time_model == "lines":
        name = name.replace(".csv", "_lines.csv")
    out = Path(args.out or PROJECT_ROOT / "outputs" / name)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
