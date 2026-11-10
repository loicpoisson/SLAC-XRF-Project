# Adaptive XRF Imaging — Phase 1 Pipeline

Research project under **Sam Webb** (SSRL, SLAC) — adaptive XRF scanning strategies
for synchrotron beamlines 6-2 / 7-2.

Phase 1 builds a Python pipeline that takes a **coarse** XRF scan (e.g. 250 µm) and
produces an **adaptive fine-scan plan** — deciding *where* to scan (spatial) and
*how long* per pixel (temporal dwell) — with full validation against ground-truth
fine scans, and a paper-aligned **Poisson-MSE quality metric**.

> **Detailed write-up:** [`outputs/progress_report.md`](outputs/progress_report.md)

## Headline finding (2026-10, report §13–16)

Targeting **one element** pays; targeting the composite does not.
`10_run_pipeline.py --target <element>` scans 250 µm, then a full 100 µm frame,
then only that element's 100 µm hot spots at 25 µm with a √-signal dwell, fills
the skipped pixels by graph-harmonic interpolation of the 100 µm points, and
exports the plan as per-line variable-dwell scans. At **equal real time**
(acceleration ramps and row moves included) it beats a uniform raster on
Poisson MSE in 10 / 12 element × sample cases (1.08–1.58×) while losing at most
3 points of hot-spot recall (Fe, Zn, Mn, Cu on UA1 / UB1 / FP1). Hot spots
cannot be found from 250 µm: the coarse scans behave like point samples.

## What the pipeline does

Four families of adaptive strategies, all driven by a quick coarse scan:
1. **Spatial ROI** — threshold + label + bounding boxes (+ travel-aware grouping)
2. **Morphological sample mask** — Otsu-lower + closing to find the sample footprint
3. **Temporal adaptive** — variable dwell per pixel (binary / linear / log / **sqrt**)
4. **Combined** — spatial mask + temporal dwell

A descriptor-based **recommender** (`sample_frac`, `sparsity`, `concentration`,
`dynamic_range`) auto-picks the strategy from the coarse scan alone.

**Quality metric (paper-aligned, `utils/quality.py`):** reconstruction MSE under
Poisson photon noise, with analytic + Monte-Carlo estimates. It shows the
MSE-optimal dwell allocation is **t ∝ √signal** (water-filling) — the `sqrt`
strategy — and that, at equal scan time, it beats a uniform raster (UA1: MSE ratio
**1.07×** on UA1, see the erratum in the progress report), while a naïve binary allocation actually *hurts* MSE (0.14×).

## Quick start

Use the SMAK virtual env on the dev machine (`C:\tools\smak\env\`), or create one
with `pip install -r requirements.txt` (pinned versions the results were produced
with; Python 3.10.11).

### On the beamline — one element, three steps

Before the first run, put the scanner's real numbers in `beamline.json`
(max speed and acceleration, dwell bounds, default ladder). The values shipped
are from Betterton et al. (2020), not measured on 7-2.

```bash
# 1. you have the coarse 250 um scan -> plan of the full-frame 100 um detection scan
python scripts/10_run_pipeline.py --coarse <250um>.hdf5 --target Zn.Ka
# 2. you acquired it -> plan of the 25 um scan of the Zn hot spots
python scripts/10_run_pipeline.py --scan <100um>.hdf5 --resume outputs/<S>_cascade_state.pkl
# 3. you acquired that plan (one or several files) -> final reconstructed map
python scripts/10_run_pipeline.py --scan <25um_a>.hdf5 <25um_b>.hdf5 --resume outputs/<S>_cascade_state.pkl
```
- The cascade settings (`--target`, `--roi-k`, ladder) are fixed at step 1 and
  stored in the state file; later steps reuse them and refuse a contradiction.
  `--levels 250 100 25` overrides the ladder of `beamline.json`.
- Each step writes `<stem>_line_plan.npz` (the next level as unidirectional
  rows with a per-pixel dwell and the stage speed, acceleration ramps included —
  the kind of scan Boltz et al. ran on 7-2), `<stem>_scan_plan.dpm` (SMAK
  dataqueue rectangles, fallback), a JSON summary, the dwell map and a figure.
- The last scan may be **partial**: only the planned pixels need to be there
  (`TIME = 0` elsewhere), with any per-pixel dwell, in one or several files.
  A file at another pixel size is refused; so is a scan covering less than half
  of the plan (`--force-level` overrides both).
- The final map (`<stem>_reconstruction.npy/.png`) holds the measurement where
  it was planned and measured, and elsewhere a graph-harmonic interpolation of
  the 100 um scan, rescaled on the sample itself (ratio over the pixels scanned
  at both resolutions). It is in counts at the median dwell of the scanned
  pixels; `<stem>_dwell_rel.npy` gives each pixel's dwell relative to that and
  `<stem>_scanned_mask.npy` what was really measured.
- `--out-dir` / `--data-dir` choose the folders; `import xrf_pipeline;
  xrf_pipeline.main([...])` runs the same thing from Python.

Not validated yet: the exact file format the 7-2 dataserver needs for a line
plan, and what a real variable-dwell scan file looks like (report §16.4, §17).

### Offline validation — when every resolution already exists on disk
```bash
python scripts/10_run_pipeline.py --simulate --sample UB1_P1 --target Zn.Ka   # one element
python scripts/10_run_pipeline.py --simulate --sample UA1_P1                  # composite cascade
```
The printed MSE ratio is the debiased Poisson MSE of `utils/evaluation.py`, at
equal real time on the line plan, with a 95 % interval — the same method as
script 13 (UB1 Zn: 1.32× here, 1.44× there; the difference is script 13's
exclusion of the fly-scan edge columns and its dwell allocation). Script 13,
with the independent second scan as truth, remains the reference.

### Robust comparison of strategies (independent truth, real coarse fill)
```bash
python scripts/13_evaluate_strategies.py      # -> outputs/evaluation_step1.csv, see report section 14
python scripts/13_evaluate_strategies.py --step2   # + reconstructions / ROI rules, section 15
python scripts/13_evaluate_strategies.py --step2 --time-model lines   # equal REAL time, section 16
```

### Profile samples by sparsity (where does adaptive help?)
```bash
python scripts/11_profile_samples.py                 # per-sample (composite)
python scripts/11_profile_samples.py --per-element   # per element channel
```

### Tests
```bash
python -m pytest tests/ -q        # 118 tests; also run by GitHub Actions on every push
```

## Repo layout
```
SLAC-XRF-Project/
├── README.md                  ← you are here
├── CLAUDE.md                  ← project context for assistants
├── requirements.txt           ← pinned versions
├── beamline.json              ← scanner parameters (speed, acceleration, dwell bounds, ladder)
├── data/                      ← raw HDF5 (gitignored, too large)
├── outputs/                   ← progress report, evaluation CSVs, figures
├── tests/                     ← pytest suite (golden values + unit tests)
├── phase2_ml/                 ← ML notebooks (separate torch env)
└── scripts/
    ├── 10_run_pipeline.py         — PRODUCTION command line (launcher)
    ├── xrf_pipeline.py            — the pipeline itself, importable (stepwise + simulate)
    ├── 11_profile_samples.py      — sparsity profiler (per-sample / per-element)
    ├── 12_report_figures.py       — report figures (MSE per dwell strategy, per element)
    ├── 13_evaluate_strategies.py  — REFERENCE evaluation (independent truth, CIs)
    ├── 01..09_*.py                — historical v1 exploration, see below
    └── utils/
        ├── hdf5_reader.py   — SSRL HDF5 loader + composite map
        ├── roi_utils.py     — thresholding, labeling, TSP, ROI grouping
        ├── strategy.py      — descriptors + recommender rules (single source)
        ├── dwell.py         — dwell allocation (binary/linear/log/sqrt)
        ├── cascade.py       — refine_step / refine_cascade, level discovery
        ├── quality.py       — Poisson MSE / SNR metrics
        ├── validation_utils.py — grid projection, metrics, travel overhead
        ├── plotting.py      — save_and_show (headless-safe figures)
        ├── reconstruction.py — fills of the skipped pixels (graph-harmonic, ...), calibration
        ├── line_scan.py     — per-line variable-dwell plans, acceleration ramps
        ├── evaluation.py    — independent-truth scoring, task metrics, bootstrap
        ├── smak_export.py   — SMAK dataqueue (.dpm) region files
        ├── beamline.py      — loads beamline.json (single source of scanner parameters)
        ├── cli.py           — shared argparse helpers
        └── paths.py         — portable data/output path discovery
```

**Historical scripts 01–09.** They are the v1 exploration described in report §5
(ROI detection, trajectory, morphological mask, hierarchical cascade, Pareto
sweep, temporal and combined dwell, recommender). They still run, and now use the
corrected loader, but **none of the current results comes from them**: their
metrics predate the independent-truth protocol (fill from the fine truth, no
coarse-time charge, own-threshold detection). Use `13_evaluate_strategies.py` for
any number you want to report.

## Key references
- **Paper 1** — Betterton et al. (2020), *Reinforcement Learning for Adaptive
  Illumination with X-rays*, ICRA 2020 — scanner kinematics (v_max = 200 mm/s,
  a_max = 500 mm/s²) used in the travel-cost model.
- **Paper 2** — Boltz, Ratner & Webb (2026), *Adaptive X-ray imaging with
  reinforcement learning*, J. Synchrotron Rad. 33, 162–169 — the temporal /
  equal-time-quality strategy is inspired by this work.

## Status (2026-10)
- Phase 1 pipeline: complete, reviewed, 118 tests (CI); validated in simulation on
  UA1 / UB1 / FP1 with an independent truth (report §14–16).
- Not yet validated on the beamline: the next step is one sample × one element
  at equal time against a raster.
- Open questions for Sam: report §13.2, §14.4, §16.4 (coarse beam footprint,
  quality criterion, line-plan format for 7-2, counting during gap crossings,
  TIME channel).
- Phase 2 (`phase2_ml/`): the U-Net dwell-map POC does not beat the √ heuristic;
  graph methods are surveyed in `Papers/reports/` (local).
```
