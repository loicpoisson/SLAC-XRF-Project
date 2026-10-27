# XRF Adaptive Imaging Project — SLAC SSRL

## Project Context
Research on adaptive XRF imaging at SLAC National Accelerator Laboratory.
Supervised by Sam Webb (co-author of the reference papers below).
Duration: 12+ months.

## Primary Objective (Phase 1)
Build a Python pipeline to:
1. Analyze coarse XRF data (2D/3D arrays)
2. Automatically locate regions of interest (ROI = high intensity areas)
3. Define coordinates/indices that efficiently cover those ROI
4. Output those coordinates for a high-resolution re-scan

This is the foundational step before implementing a full adaptive RL agent.

## Technical Environment
- OS: Windows 11
- Python: 3.10.11
- Venv: C:\tools\smak\env\  (always activate before running); pinned
  versions in `requirements.txt`
- SMAK 3 (Sam Webb, SLAC) installed via pip in this venv
- IDE: VS Code + Claude Code
- Phase 2 (torch) runs in a SEPARATE venv or on Colab — never install torch
  into the smak env

## Data facts (read before touching a loader)
- SMAK `main/mapdata` has `2 + n_labels` planes: planes 0/1 are the stage X/Y
  positions, label i is plane i+2 (SMAK reads `labels.index(ch) + 2`).
  `utils.hdf5_reader.load_xrf` handles it; never index `mapdata` by label
  directly elsewhere.
- The `TIME` channel is the real per-pixel integration time: ~65x nominal in
  column 0, 1.2-4x on the first/last ~20 columns (fly-scan ramps). `load_xrf`
  rescales counts by `median(TIME)/TIME`.
- Coarse 250/100 um scans behave like POINT samples, not cell averages: a
  particle away from a coarse pixel centre barely shows in it.
- UA1_P1 and UB1_P1 have two independent 25 um scans (10 ms and 25 ms; the
  UA1 25 ms one is partial, 218/653 rows) — used as an independent truth.
- Data live in `data/Data_May2026/` (gitignored, ~3.6 GB).

## Reference Papers
1. Betterton et al. (2020) — "Reinforcement Learning for Adaptive
   Illumination with X-rays" — ICRA 2020
   → RL formulation of scanner control, CNN policy, Evolution Strategies
   → Multiple apertures, raster baseline comparison
   → Scanner kinematics used here: v_max 200 mm/s, a_max 500 mm/s²

2. Boltz, Ratner & Webb (2026) — "Adaptive X-ray imaging with
   reinforcement learning" — J. Synchrotron Rad. 33, 162–169
   → Experimental follow-up to paper 1, deployed on SSRL beamline 7-2
   → Gaussian approximation of Poisson noise, direct policy gradients
   → Results: 40x more exposure on ROI, factor 2.5 MSE improvement
   → Executed as unidirectional line scans with a per-pixel dwell (ramps
     inserted for the acceleration limit) — the model of `utils/line_scan.py`

## Key Concepts
- Raster scanning: baseline method (pixel-by-pixel, fixed dwell time)
- ROI: Region of Interest = high intensity areas in XRF map
- Coarse scan → ROI detection → fine scan = target adaptive workflow
- MSE: image quality metric used in both papers (Poisson reconstruction MSE)
- Poisson noise: physical noise model for XRF detectors; the MSE-optimal
  dwell at equal time is t ∝ sqrt(signal)
- Always compare strategies at EQUAL TOTAL TIME, earlier levels included
- SSRL Beamline 6-2 / 7-2: experimental infrastructure

## Current Status (2026-10)
Phase 1 pipeline complete, reviewed and benchmarked; see
`outputs/progress_report.md` (§13-16 are the current results; older sections
predate the corrections listed in its erratum).
- Recommended plan (`scripts/10_run_pipeline.py --target <element>`):
  250 um → full 100 um detection scan → 25 um scan of the element's hot spots
  (MAD k = 0.5), sqrt dwell, skipped pixels reconstructed by graph-harmonic
  interpolation of the 100 um points outside the scanned area, exported as a
  per-line variable-dwell plan (`*_line_plan.npz`).
- At equal real time (line-scan model) it beats a raster on Poisson MSE in
  10/12 element x sample cases (1.08-1.58x) and loses at most 3 points of
  hot-spot recall. ROIs found at 250 um, the composite target, and the full
  250/100/50/25 cascade do not pay.
- Reference evaluation: `scripts/13_evaluate_strategies.py` (independent
  truth, measured coarse fill, MSE + sqrt-MSE + quantification + detection,
  block-bootstrap CIs; `--step2`, `--time-model lines`, `--jobs`).
- Open questions for Sam: report §13.2, §14.4, §16.4 (coarse beam footprint,
  criterion, line-plan format, counting during gap crossings, TIME channel).
- Phase 2 (`phase2_ml/`): U-Net POC does not beat the sqrt heuristic; graph
  methods surveyed in `Papers/reports/GNN pour imagerie XRF adaptative.md`.

## Project Folder Structure
SLAC-XRF-Project/
├── CLAUDE.md          ← this file, auto-read at every session
├── README.md          ← how to run everything
├── requirements.txt   ← pinned versions of the smak venv
├── Papers/            ← reference PDFs + research notes/reports (gitignored)
├── data/              ← raw XRF datasets (gitignored)
├── scripts/
│   ├── 10_run_pipeline.py        ← production pipeline (stepwise / --simulate)
│   ├── 11_profile_samples.py, 12_report_figures.py
│   ├── 13_evaluate_strategies.py ← reference evaluation
│   ├── 01..09_*.py               ← historical v1 exploration (see README)
│   └── utils/                    ← all shared logic (loader, strategy, dwell,
│                                    reconstruction, line_scan, evaluation, ...)
├── tests/             ← pytest suite (`python -m pytest -q tests`)
├── phase2_ml/         ← ML notebooks (separate torch env)
└── outputs/           ← progress report, evaluation CSVs, figures

## Code Conventions
- Always use the smak venv: C:\tools\smak\env\Scripts\activate
- Save outputs to ./outputs/ (generated plans/maps are gitignored)
- Raw data in ./data/
- Shared logic goes in scripts/utils/ (single source of truth); add a test
  for every bug fix
- Never commit scratch outputs of a run; restore outputs/ after experiments
