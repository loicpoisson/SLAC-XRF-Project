# Phase 2 — ML extension

Goal: test whether a **learned model** can predict a better per-pixel dwell map from the
coarse scan than the Phase-1 heuristic. The physics fixes the target: under Poisson noise
and a fixed time budget the MSE-optimal allocation is `t ∝ sqrt(rate)` (see
`scripts/utils/quality.py` and the `'sqrt'` strategy in `scripts/utils/dwell.py`). The
heuristic applies that rule to the blurry coarse map; the model tries to recover the
rule applied to the (unknown) fine map.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_dwell_unet_poc.ipynb` | Feasibility POC: small U-Net (~30k params), coarse 250 µm composite → budget-normalised `sqrt(fine 50 µm)` dwell map, leave-one-sample-out over UA1_P1 / UB1_P1 / FP1_P1, scored with `utils.quality.poisson_mse` at equal total time against uniform raster, the coarse sqrt heuristic and the oracle. Committed **with outputs** from a CPU run. |

### Design (01)
- Loads data **only** through the repo's `scripts/utils` (`hdf5_reader.load_xrf` — SMAK
  plane offset and `median(TIME)/TIME` normalisation — plus `validation_utils`, `dwell`,
  `quality`). No re-implemented loader.
- Leave-one-sample-out (train on 2 samples, test on the 3rd, all 3 folds); input
  normalisation statistics from the training samples only; non-overlapping test tiles.
- On-the-fly random crops (64×64) with flips/rot90; output head bounded to [1, 50] ms;
  fixed epochs (no early stopping / tuning on the test sample); seeds fixed.
- `INPUT_MODE = 'composite'` (default) or `'elements'` (one input channel per element).

### Results of the committed run (CPU, 30 epochs × 40 steps × batch 16, composite input)

Poisson-MSE ratio `MSE_raster / MSE_strategy` on the **held-out** sample, all maps at
mean dwell 10 ms (> 1 = better than raster):

| held-out | (a) uniform | (b) heuristic sqrt (`allocate_dwell`) | (b') sqrt(coarse) | (c) U-Net | (d) oracle sqrt(fine) | headroom closed by (c) |
|---|---|---|---|---|---|---|
| UA1_P1 | 1.000 | 1.014 | 1.009 | **1.016** | 1.124 | +1.7 % |
| UB1_P1 | 1.000 | 1.024 | 1.027 | **1.029** | 1.144 | +1.6 % |
| FP1_P1 | 1.000 | 1.157 | **1.218** | 1.143 | 1.567 | −21 % |

Headroom = `(c − best(b, b')) / (d − best(b, b'))`.

**Reading.** (c) is marginally above the best heuristic on UA1_P1 and UB1_P1 (+0.002 in
ratio, i.e. ~2 % of the available headroom — well within what a different seed or epoch
count could change) and clearly below it on FP1_P1. So on this data the U-Net does **not**
beat the coarse sqrt heuristic in any meaningful way. The U-Net map is closer to the
oracle in log-dwell MSE on all three folds, yet scores worse on FP1_P1: the log loss
weights all pixels equally while the Poisson MSE is dominated by bright pixels. Training
on the Poisson objective directly is the obvious next experiment. Note also how small the
total headroom is (oracle: +12–57 % over raster).

**This is a feasibility check, not evidence**: 3 samples, 3 folds, one seed, no
hyper-parameter search. The oracle (d) uses the fine truth and is only an upper bound.
Limitations are listed at the end of the notebook.

## Running it

**Locally** (needs `torch`; do not install it into the smak env — use a separate venv):
```
set XRF_DATA_DIR=C:\path\to\SLAC-XRF-Project\data\Data_May2026
jupyter nbconvert --to notebook --execute --inplace phase2_ml/notebooks/01_dwell_unet_poc.ipynb
```
The first code cell finds the repo root by walking up from the working directory
(override with `XRF_PROJECT_ROOT`) and adds `scripts/` to `sys.path`. `XRF_EPOCHS`
and `XRF_INPUT_MODE` override the training length and the input mode.
Runtime of the committed run on a laptop CPU: ~36 min (3 folds × 30 epochs, 7–20 min per fold).

**On Colab**: put the repo (at least `scripts/` and `phase2_ml/`) and
`data/Data_May2026/` under `/MyDrive/SLAC-XRF-Project/` on Google Drive (or set
`XRF_PROJECT_ROOT` / `XRF_DATA_DIR` in the config cell), open the notebook in Colab and
run all; the first cell mounts Drive. A GPU is not needed.

## Models to explore (after the POC)

- **U-Net** — supervised image-to-image, what the POC tests
- **SLADS** (Supervised Learning Approach for Dynamic Sampling) — Random Forest / NN for adaptive sampling, designed for microscopy
- **gpCAM** (LBL) — Gaussian Process active learning, no pretraining required
- **Bluesky-Adaptive** (NSLS-II) — direct beamline integration (deployment, not training)

## What to read next
- Phase 1 progress report: [../outputs/progress_report.md](../outputs/progress_report.md)
- Paper 2 (Boltz-Webb 2026) — the reference for the temporal-adaptive idea
- Godaliyadda et al. 2018 — original SLADS paper
- Noack et al. (LBL) — gpCAM publications
