"""
Benchmark for adaptive-scan PLANS (rule-based or learned) - the protocol of
scripts/13_evaluate_strategies.py, as a library.

A plan for one sample x target is three maps on the fine grid:
    mask   : pixels scanned at fine resolution
    weight : relative dwell wanted on them (e.g. sqrt of a rate estimate)
    pred   : rate [counts/ms] predicted for every pixel (used where not scanned)
`Case.score(...)` turns it into the reference numbers: Poisson-MSE ratio vs a
uniform raster at EQUAL REAL TIME (line-scan model, coarse levels charged),
block-bootstrap CI, sqrt-domain ratio, hot-spot quantification error and
detection recall / precision at the truth's threshold - against the 10 ms scan
("A") and, where it exists, the independent registered 25 ms scan ("B").

Nothing here may look at a truth to build a plan: `Case.coarse*` are the only
inputs a planner is allowed to use.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

from .hdf5_reader import load_xrf, get_composite_map, get_channel, list_element_channels
from .cascade import find_level_file, dwell_ms_from_path
from .roi_utils import element_roi_mask
from .validation_utils import project_mask, project_coarse_to_fine, _nearest_idx
from .evaluation import (budget_dwell, sqrt_weight, pixel_errors, raster_errors,
                         block_bootstrap_ratio, hot_spots, simulate_tasks,
                         register_rows, sqrt_errors, top_share, sqrt_poisson_var)
from .reconstruction import coarse_rate, fill_harmonic, calibrate_scale
from .line_scan import raster_line_time, fit_to_budget
from .paths import DATA_DIR

SAMPLES = {"UA1_P1": 25, "UB1_P1": 25, "FP1_P1": 50}   # finest full scan per sample
T_REF = 10.0
EDGE_COLS = [0, 1, -1]            # fly-scan edge columns of the coarse scans


def scan_file(sample, px, dwell_ms=10, data_dir=None):
    base = Path(data_dir) if data_dir else DATA_DIR
    return next(iter(sorted(base.rglob(f"*{sample}_{px}um_{dwell_ms}ms_*.hdf5"))), None)


def without_edges(cmap):
    """Coarse map with its edge columns set to the median (no fake ROIs)."""
    out = cmap.astype(float).copy()
    out[:, EDGE_COLS] = np.median(cmap)
    return out


class Case:
    """One sample x target x fine resolution, with its truths."""

    def __init__(self, sample, target, fine_px=None, spot_k=3.0):
        self.sample, self.target = sample, target
        self.fine_px = fine_px or SAMPLES[sample]
        self.spot_k = spot_k
        paths = {"c250": find_level_file(sample, 250),
                 "c100": find_level_file(sample, 100),
                 "A": scan_file(sample, self.fine_px, 10),
                 "B": scan_file(sample, self.fine_px, 25)}
        self.paths = {k: p for k, p in paths.items() if p is not None}
        self.data = {k: load_xrf(p) for k, p in self.paths.items()}
        self.dwell = {k: dwell_ms_from_path(p) for k, p in self.paths.items()}
        ch = None if target == "composite" else [target]
        c250, used, _ = get_composite_map(self.data["c250"], ch)
        self.channels = used
        self.maps = {"c250": c250}
        for k in ("c100", "A", "B"):
            if k in self.data:
                self.maps[k] = get_composite_map(self.data[k], used)[0]
        self.F = self.data["A"]
        self.shape = self.maps["A"].shape
        self.full = np.ones(self.shape, bool)
        self.T = self.maps["A"].size * T_REF
        self.spent = {250: self.maps["c250"].size * self.dwell["c250"]}
        self.spent[100] = self.spent[250] + self.maps["c100"].size * self.dwell["c100"]
        lt = {k: raster_line_time(self.maps[k].shape, self.data[k]["xdata"],
                                  self.data[k]["ydata"], self.dwell[k])
              for k in ("c250", "c100")}
        self.T_lines = raster_line_time(self.shape, self.F["xdata"], self.F["ydata"], T_REF)
        self.spent_lines = {250: lt["c250"], 100: lt["c250"] + lt["c100"]}
        ok = np.ones(self.shape[1], bool)
        for k in ("c250", "c100"):
            d = self.data[k]
            ix = _nearest_idx(d["xdata"], self.F["xdata"])
            ok &= ~np.isin(ix, [0, 1, len(d["xdata"]) - 1])
        self.edge_ok = ok[None, :]
        self._truths = None
        self._loo = {}

    # ── inputs a planner may use ───────────────────────────────────────────────
    def coarse(self, level=100):
        """Target map of a coarse level (counts), edge columns neutralised."""
        return without_edges(self.maps[f"c{level}"])

    def coarse_elements(self, level=100):
        """(ny, nx, C) counts of every element channel of a coarse level and
        the channel names (edge columns neutralised)."""
        d = self.data[f"c{level}"]
        names = [c for c in list_element_channels(d)]
        arr = np.stack([without_edges(get_channel(d, c)) for c in names], axis=2)
        return arr, names

    def project(self, cmap, level=100):
        """Replicate a coarse-grid map over the fine grid."""
        return project_coarse_to_fine(cmap, self.data[f"c{level}"], self.F)

    def project_mask(self, cmask, level=100):
        d = self.data[f"c{level}"]
        return project_mask(cmask, d["xdata"], d["ydata"], self.F["xdata"], self.F["ydata"])

    def loo_scale(self, level=100):
        """Coarse->fine count scale measured on the other samples (cached)."""
        if level not in self._loo:
            ch = None if self.target == "composite" else [self.target]
            self._loo[level] = calibrate_scale(self.sample, ch, level, self.fine_px,
                                               candidates=list(SAMPLES))[0]
        return self._loo[level]

    # ── truths (scoring only) ──────────────────────────────────────────────────
    @property
    def truths(self):
        if self._truths is None:
            self._truths = self._build_truths()
        return self._truths

    def _build_truths(self):
        m, dw = self.maps, self.dwell
        rA = m["A"] / dw["A"]
        valid_A = (m["A"].sum(axis=1) > 0)[:, None] & self.edge_ok & self.full
        out = [dict(name="A", r=rA, t=dw["A"], valid=valid_A, other=None,
                    spot_valid=valid_A)]
        if "B" in m:
            compA, usedA, _ = get_composite_map(self.data["A"])
            compB = get_composite_map(self.data["B"], usedA)[0]
            _, reg_ok, shifts = register_rows(compB, compA)
            rB = np.empty_like(m["B"], dtype=float)
            for y, sh in enumerate(shifts):
                rB[y] = np.roll(m["B"][y], sh)
            rB /= dw["B"]
            valid_B = (m["B"].sum(axis=1) > 0)[:, None] & reg_ok & self.edge_ok
            rB *= rA[valid_B].mean() / max(rB[valid_B].mean(), 1e-12)
            both = valid_A & valid_B
            out[0].update(other=rB, spot_valid=both)
            out.append(dict(name="B", r=rB, t=dw["B"], valid=valid_B, other=rA,
                            spot_valid=both))
        for tr in out:
            lab, n = hot_spots(tr["r"], tr["spot_valid"], k=self.spot_k)
            if tr["other"] is not None and n:
                lab_o, _ = hot_spots(tr["other"], tr["spot_valid"], k=self.spot_k)
                keep = np.zeros(n + 1, bool)
                keep[np.unique(lab[(lab > 0) & (lab_o > 0)])] = True
                keep[0] = False
                lab, n = ndimage.label(keep[lab])
            tr.update(lab=lab, n_spots=n, e_ras=raster_errors(tr["r"], T_REF))
            tr["q_tvar"] = sqrt_poisson_var(np.maximum(tr["r"], 0.0) * tr["t"]) / tr["t"]
            tr["q_ras"] = sqrt_errors(tr["r"], tr["t"], np.full(self.shape, T_REF),
                                      self.full, tr["r"], truth_var=tr["q_tvar"])
        return out

    # ── scoring ────────────────────────────────────────────────────────────────
    def dwell_for(self, mask, weight, level=100, time_model="lines"):
        """Dwell map [ms] spending exactly the raster time minus the coarse
        levels up to `level`; with 'lines', the dwell the line scan delivers."""
        d = budget_dwell(weight, mask, self.T - self.spent[level])
        if time_model != "lines":
            return d
        _, plan = fit_to_budget(d, self.T_lines - self.spent_lines[level],
                                self.F["xdata"], self.F["ydata"])
        if not plan["converged"]:
            raise ValueError("plan cannot spend the raster time within the dwell "
                             "bounds (mask too small or too large): the comparison "
                             "would not be at equal time")
        return plan["dwell_ms"]

    def score(self, mask, weight, pred, level=100, time_model="lines", draws=10,
              boot=300, name="plan"):
        """Reference metrics of a plan, one dict per truth."""
        d = self.dwell_for(mask, weight, level, time_model)
        scanned = d > 0
        rows = []
        for tr in self.truths:
            r, valid = tr["r"], tr["valid"]
            e_ad = pixel_errors(r, np.maximum(r, 0.0) / tr["t"], d, scanned, pred)
            q_ad = sqrt_errors(r, tr["t"], d, scanned, pred, truth_var=tr["q_tvar"])
            ci = block_bootstrap_ratio(tr["e_ras"], e_ad, valid, n_boot=boot)
            task = simulate_tasks(r, d, scanned, pred, tr["spot_valid"], tr["lab"],
                                  tr["n_spots"], k=self.spot_k, n_draws=draws)
            rows.append({
                "sample": self.sample, "target": self.target, "truth": tr["name"],
                "strategy": name, "area": float(mask[valid].mean()),
                "mse_ratio": float(tr["e_ras"][valid].sum() / e_ad[valid].sum()),
                "ci_lo": ci[0], "ci_hi": ci[1],
                "sqrt_ratio": float(tr["q_ras"][valid].sum() / q_ad[valid].sum()),
                "top1pct_share": top_share(e_ad, valid),
                "quant_err": task["quant_err"], "recall": task["recall"],
                "precision": task["precision"],
                "recall_skipped": task["recall_skipped"], "n_spots": tr["n_spots"]})
        return rows

    def primary(self, rows):
        """The row against the most trustworthy truth (B when it exists)."""
        return next((r for r in rows if r["truth"] == "B"), rows[0])

    # ── the rule-based reference plan (script 13: roi100_k<k>+harmonic_c) ────
    def rule_plan(self, k=0.5, level=100, scale=None):
        """(mask, weight, pred) of the current default pipeline: MAD ROIs on the
        coarse target map + 1-pixel ring, sqrt dwell, harmonic fill of the
        coarse points outside the scanned area."""
        key = f"c{level}"
        cmap = self.coarse(level)
        s = self.loo_scale(level) if scale is None else scale
        crate = coarse_rate(cmap, s, self.dwell[key])
        est = self.project(crate, level)
        mask = self.project_mask(element_roi_mask(cmap, k), level)
        pred = fill_harmonic(crate, self.data[key], self.F, mask,
                             np.zeros(self.shape), w_fine=0.0)
        return mask, sqrt_weight(est, mask), pred
