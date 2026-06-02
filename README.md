# Adaptive XRF Imaging — Phase 1 Pipeline

Research project under Sam Webb (SSRL, SLAC) — adaptive XRF scanning strategies for synchrotron beamlines 6-2 and 7-2.

## What this repo contains

Phase 1 = a complete Python pipeline that takes a coarse XRF scan (e.g. 250 µm) and produces an adaptive fine-scan plan, with full validation against ground-truth fine scans (25 / 50 µm).

Four families of adaptive strategies are implemented and benchmarked:
1. **Spatial ROI** — threshold + label + bounding boxes + TSP-optimized trajectory
2. **Morphological sample mask** — Otsu_lower + closing to find the sample footprint
3. **Temporal adaptive** — variable dwell per pixel (binary / linear / log) inspired by Boltz, Ratner & Webb 2026
4. **Combined** spatial mask + temporal dwell allocation

A simple recommender automatically picks the strategy from coarse-only descriptors (sample fraction, sparsity, concentration, dynamic range).

For a detailed write-up, see [`outputs/progress_report.md`](outputs/progress_report.md).

## Quick start

### Requirements
- Python 3.10
- numpy, scipy, matplotlib, h5py
- Install via the SMAK virtual env at `C:\tools\smak\env\` on the dev machine, or `pip install -r requirements.txt`

### Run on a new coarse scan (production)
```bash
python scripts/10_run_pipeline.py --coarse path/to/my_coarse.hdf5
```
Outputs:
- `outputs/<stem>_scan_plan.json` — full scan plan (regions + dwells)
- `outputs/<stem>_scan_plan.pkl` — SSRL/SMAK-compatible pickle
- `outputs/<stem>_scan_plan.png` — figure of the chosen plan

### Validate against a ground-truth fine scan
```bash
python scripts/06_pareto_sweep.py --samples UA1_P1 UB1_P1
```
Outputs: Pareto curve (signal captured vs speedup) for each method/parameter combination.

## Repo layout
```
SLAC-XRF-Project/
├── README.md                  ← you are here
├── CLAUDE.md                  ← project context for assistants
├── data/                      ← raw HDF5 (gitignored, too large)
├── Papers/                    ← reference PDFs (gitignored)
├── outputs/                   ← figures, reports, scan plans (most gitignored)
├── scripts/
│   ├── 01_roi_detection.py        — ROI detection on coarse
│   ├── 02_trajectory.py           — TSP + 2-opt + Or-opt planning
│   ├── 03_validate_roi.py         — validation: ROI vs ground truth
│   ├── 04_validate_morphological.py  — sample mask
│   ├── 05_validate_hierarchical.py   — multi-resolution cascade
│   ├── 06_pareto_sweep.py         — methods comparison + Pareto frontier
│   ├── 07_validate_temporal.py    — temporal adaptive (Paper 2 style)
│   ├── 08_validate_combined.py    — spatial + temporal
│   ├── 09_recommend_strategy.py   — descriptors + auto-recommendation
│   ├── 10_run_pipeline.py         — PRODUCTION (coarse only, no ground truth)
│   └── utils/
│       ├── hdf5_reader.py         — SSRL HDF5 loader + composite map
│       ├── roi_utils.py           — thresholding, labeling, TSP, merge cost
│       └── validation_utils.py    — projection, metrics, travel overhead
```

## Key references
- **Paper 1**: Betterton et al. (2020), "Reinforcement Learning for Adaptive Illumination with X-rays", ICRA 2020 — scanner kinematics (v_max=200 mm/s, a_max=500 mm/s²) used in our travel cost model.
- **Paper 2**: Boltz, Ratner & Webb (2026), "Adaptive X-ray imaging with reinforcement learning", J. Synchrotron Rad. 33, 162–169 — temporal adaptive strategy is inspired by this work.

## Phase 2 (work in progress)
ML-based extension: train U-Net models on paired (coarse, fine) datasets to predict optimal dwell maps directly. See the `phase2_ml/` folder when added.

## Status
- Phase 1 pipeline: complete, validated on UA1_P1, UB1_P1, FP1_P1
- Production deployment via `10_run_pipeline.py`: works with coarse-only inputs
- Phase 2 ML: not started in this repo yet