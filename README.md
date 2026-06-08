# Adaptive XRF Imaging — Phase 1 Pipeline

Research project under **Sam Webb** (SSRL, SLAC) — adaptive XRF scanning strategies
for synchrotron beamlines 6-2 / 7-2.

Phase 1 builds a Python pipeline that takes a **coarse** XRF scan (e.g. 250 µm) and
produces an **adaptive fine-scan plan** — deciding *where* to scan (spatial) and
*how long* per pixel (temporal dwell) — with full validation against ground-truth
fine scans, and a paper-aligned **Poisson-MSE quality metric**.

> **Detailed write-up:** [`outputs/progress_report.md`](outputs/progress_report.md)

## Headline finding

On these samples the **composite** (sum of all elements) is **dense** (75–82 % of the
area occupied) — so spatial skipping can't beat a fine raster on the composite.
But **per element**, the signal is **sparse hot spots** (~10–15 % of pixels, 50–100 %
concentration; e.g. S, P, Al, Ti). Since XRF imaging targets *specific* elements,
**that is where adaptive scanning pays off** — the pipeline should target a single
element's ROIs, not the composite. (See `scripts/11_profile_samples.py --per-element`.)

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
**1.14×**), while a naïve binary allocation actually *hurts* MSE (0.16×).

## Quick start

Use the SMAK virtual env on the dev machine (`C:\tools\smak\env\`) or
`pip install numpy scipy matplotlib h5py` (+ `pytest` to run the tests).

### Production — iterative, one beamline step at a time
```bash
# step 0: analyse the coarse scan, plan the next (finer) level
python scripts/10_run_pipeline.py --coarse data/.../UA1_P1_250um....hdf5
# step N: feed the level you just acquired, plan the next one
python scripts/10_run_pipeline.py --scan data/.../UA1_P1_100um....hdf5 \
       --resume outputs/UA1_P1_cascade_state.pkl
# focus a single (sparse) element:
python scripts/10_run_pipeline.py --coarse <coarse>.hdf5 --channels Ti.Ka
```
Outputs per step: a text report, `*_scan_plan.json` (regions + dwells),
`*_scan_plan.pkl` (SMAK-compatible rectangles), `*_dwell_map.npy` (exact per-pixel
dwell), and a figure.

### Offline validation — full cascade when all resolutions exist
```bash
python scripts/10_run_pipeline.py --simulate --sample UA1_P1            # capture %, MSE, speedup
python scripts/10_run_pipeline.py --simulate --sample UA1_P1 --dwell-strategy sqrt
```

### Profile samples by sparsity (where does adaptive help?)
```bash
python scripts/11_profile_samples.py                 # per-sample (composite)
python scripts/11_profile_samples.py --per-element   # per element channel
```

### Tests
```bash
python -m pytest tests/ -q        # 39 tests (quality, dwell, roi, cascade, strategy)
```

## Repo layout
```
SLAC-XRF-Project/
├── README.md                  ← you are here
├── CLAUDE.md                  ← project context for assistants
├── data/                      ← raw HDF5 (gitignored, too large)
├── outputs/                   ← figures, reports, scan plans
├── tests/                     ← pytest suite (golden values + unit tests)
└── scripts/
    ├── 01_roi_detection.py        — ROI detection on coarse
    ├── 02_trajectory.py           — TSP trajectory (NN + 2-opt + Or-opt)
    ├── 03_validate_roi.py         — validate ROI vs ground truth
    ├── 04_validate_morphological.py — sample-footprint mask
    ├── 05_validate_hierarchical.py  — multi-resolution cascade
    ├── 06_pareto_sweep.py         — methods comparison + Pareto frontier
    ├── 07_validate_temporal.py    — temporal adaptive dwell (Paper 2 style)
    ├── 08_validate_combined.py    — spatial + temporal
    ├── 09_recommend_strategy.py   — descriptors + auto-recommendation
    ├── 10_run_pipeline.py         — PRODUCTION: iterative cascade (stepwise + simulate)
    ├── 11_profile_samples.py      — sparsity profiler (per-sample / per-element)
    └── utils/
        ├── hdf5_reader.py   — SSRL HDF5 loader + composite map
        ├── roi_utils.py     — thresholding, labeling, TSP, ROI grouping
        ├── strategy.py      — descriptors + recommender rules (single source)
        ├── dwell.py         — dwell allocation (binary/linear/log/sqrt)
        ├── cascade.py       — refine_step / refine_cascade, level discovery
        ├── quality.py       — Poisson MSE / SNR metrics
        ├── validation_utils.py — grid projection, metrics, travel overhead
        ├── plotting.py      — save_and_show (headless-safe figures)
        ├── cli.py           — shared argparse helpers
        └── paths.py         — portable data/output path discovery
```

## Key references
- **Paper 1** — Betterton et al. (2020), *Reinforcement Learning for Adaptive
  Illumination with X-rays*, ICRA 2020 — scanner kinematics (v_max = 200 mm/s,
  a_max = 500 mm/s²) used in the travel-cost model.
- **Paper 2** — Boltz, Ratner & Webb (2026), *Adaptive X-ray imaging with
  reinforcement learning*, J. Synchrotron Rad. 33, 162–169 — the temporal /
  equal-time-quality strategy is inspired by this work.

## Status
- Phase 1 pipeline: complete, iterative, tested (39 tests), validated on UA1 / UB1 / FP1.
- Honest scope: spatial speedup needs **sparse** targets → demonstrated per-element,
  not on the dense composite; temporal `sqrt` improves MSE at equal time on dense samples.
- Open questions for the beamline (see report §10): per-pixel dwell map vs uniform-dwell
  regions, scanner kinematics, confirm per-element focus.
- Phase 2 (ML / RL): scaffolding in `phase2_ml/` (U-Net dwell-map POC).
```
