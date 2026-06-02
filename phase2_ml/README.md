# Phase 2 — ML extension

Goal: replace the hand-crafted heuristics of Phase 1 (binary/linear/log dwell) with **learned models** that predict optimal scan parameters directly from the coarse scan.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_dwell_unet_poc.ipynb` | First POC: small U-Net that predicts an "oracle" dwell map from a coarse XRF composite. Built to run on free Google Colab (T4 GPU). |

## Quick start on Colab

1. Upload the whole project folder to Google Drive under `/MyDrive/SLAC-XRF-Project/` (only the `data/` and the relevant scripts are needed at runtime).
2. Open `phase2_ml/notebooks/01_dwell_unet_poc.ipynb` in Colab (right-click in Drive → Open with → Google Colaboratory).
3. Runtime → Change runtime type → **T4 GPU**.
4. Run cells top to bottom. The first cell mounts your Drive.

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
