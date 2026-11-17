# Phase 2 — learned planner

Goal: replace the hand-written rule that turns the 100 µm detection scan into a
25 µm scan plan by a model that **uses every element the detector records** and
is trained on full scans of other samples.

Everything is plain Python modules with tests (`scripts/ml/`,
`tests/test_ml.py`) and is scored by the same benchmark as the rule-based
pipeline (`scripts/utils/bench.py`). No notebook is needed.

## The model (`scripts/ml/model.py`)

A graph neural network, `RateGNN`:

- **Nodes** = the pixels of the 100 µm detection scan. Features = all 15
  element channels (scaled per sample), the target element in absolute counts,
  the target in the 250 µm scan, and its local contrast.
- **Message passing** = 3 edge-conditioned layers over the 8 neighbours; a
  small MLP of the relative offset gates each message, so the layers do not
  assume a pixel grid and run on any node set.
- **Read-out** = each fine pixel attends over the 4 coarse nodes around it and
  predicts its rate for the target element.
- The prediction is anchored on the non-learned baseline:
  `rate = scale · (bilinear · exp(g) + softplus(a) · mean)`, so the untrained
  network *is* the bilinear interpolation and learning can only depart from it
  where the data support it.
- **Loss** = Poisson negative log-likelihood of the fine scan's counts — the
  right loss for a noisy photon-counting label (no "ground truth" is assumed).
- ~20 k parameters.

The predicted rate map then drives the three decisions of a plan
(`scripts/ml/plan.py`): where to scan (highest predicted rate), how long
(`√rate`), and what to put in the pixels that are skipped.

## Protocol

- **Leave-one-sample-out.** The model that plans a sample was trained only on
  scans of the *other* samples (UA1 / UB1 at 25 and 50 µm, FP1 at 50 µm).
- Model selection uses spatial blocks held out **inside the training samples**;
  the test sample is never touched before scoring.
- **Controls.** The learned map is compared at the *same scanned area* with
  (a) the current rule, (b) the bilinear map, (c) the smoothed coarse map —
  all through the same planner. Beating the raw rule only shows that denoising
  helps; the model has to beat the **smoothed** control to be worth using.
- Scoring = `utils/bench.py`: equal real time (line-scan model, coarse scans
  charged), independent 25 ms truth where it exists, Poisson MSE with a
  bootstrap interval, √-domain MSE, hot-spot quantification and detection.
  The benchmark reproduces `13_evaluate_strategies.py` to the last digit.

## Run it

PyTorch lives in its **own environment** (never in the smak env). Use Python
3.10–3.12 (the pinned numpy / scipy have no wheels for newer versions):

```bash
python -m venv ~/xrf-ml && source ~/xrf-ml/bin/activate      # Windows: C:\tools\xrf-ml
pip install -r requirements.txt
pip install torch            # CPU: --index-url https://download.pytorch.org/whl/cpu
```

```bash
python scripts/14_train_planner.py                  # 12 models (3 held-out samples x Fe Zn Mn Cu)
python scripts/15_evaluate_planner.py               # -> outputs/evaluation_planner.csv
python scripts/15_evaluate_planner.py --controls-only   # rule + non-learned controls, no PyTorch
python -m pytest -q tests/test_ml.py
```

Training takes about 1 h per model on a laptop CPU (8 cores) and seconds to
minutes on a GPU. Checkpoints go to `phase2_ml/models/` (not versioned).

### On FarmShare

1. Copy the repository and the scans (≈ 700 MB) to the cluster:
   ```bash
   git clone https://github.com/loicpoisson/SLAC-XRF-Project.git && cd SLAC-XRF-Project
   git checkout feature/ml         # until the branch is merged into main
   mkdir -p data/Data_May2026      # then copy the SMW_*_P1_{250,100,50,25}um_*.hdf5 files into it
   ```
2. Create the environment once (first block above).
3. Check the partition names (`sinfo`), adjust the two `ADJUST` lines of
   `phase2_ml/farmshare_train.sbatch`, then:
   ```bash
   sbatch phase2_ml/farmshare_train.sbatch
   ```
   The log is `xrf-planner_<jobid>.log` in the folder you submitted from. If no
   GPU is granted the job runs on CPU (≈ 1 h per model, 12 models).
4. Bring `phase2_ml/models/` and `outputs/evaluation_planner.csv` back.

## Status

Code complete and tested; trained and scored only as far as
`outputs/progress_report.md` §18 says. The earlier U-Net experiment
(`notebooks/01_dwell_unet_poc.ipynb`: coarse composite → dwell map) did not beat
the `√` heuristic and is kept for the record.

## Next models (same data layer, same benchmark)

- **Reconstruction with the fine measurements as nodes**: add the scanned
  25 µm pixels to the graph to predict the skipped ones (a particle cut by the
  ROI border continues outside it).
- **Self-supervised denoising** from the two independent 25 µm scans
  (Noise2Noise), which would improve every map, raster included.
- **Uncertainty-driven acquisition**: predict a rate *distribution* and scan
  where the expected error reduction per second is highest.
