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
- Venv: C:\tools\smak\env\  (always activate before running)
- SMAK 3 (Sam Webb, SLAC) installed via pip in this venv
- IDE: VS Code + Claude Code

## Python Toolkits Already Explored
- scikit-image (regionprops, threshold_otsu, measure.label)
- scipy.ndimage (label, find_objects, maximum_filter, center_of_mass)
- trackpy (particle detection — highly relevant for sparse XRF)
- bluesky + bluesky-adaptive (synchrotron framework, adaptive scan)
- gpCAM (Gaussian Process adaptive sampling, Kriging-based)
- numpy, scipy.spatial, sklearn (DBSCAN)

## Reference Papers
1. Betterton et al. (2020) — "Reinforcement Learning for Adaptive
   Illumination with X-rays" — ICRA 2020
   → RL formulation of scanner control, CNN policy, Evolution Strategies
   → Multiple apertures, raster baseline comparison

2. Boltz, Ratner & Webb (2026) — "Adaptive X-ray imaging with
   reinforcement learning" — J. Synchrotron Rad. 33, 162–169
   → Experimental follow-up to paper 1, deployed on SSRL beamline 7-2
   → Gaussian approximation of Poisson noise, direct policy gradients
   → Results: 40x more exposure on ROI, factor 2.5 MSE improvement

## Key Concepts
- Raster scanning: baseline method (pixel-by-pixel, fixed dwell time)
- ROI: Region of Interest = high intensity areas in XRF map
- Coarse scan → ROI detection → fine scan = target adaptive workflow
- MSE: image quality metric used in both papers
- Poisson noise: physical noise model for XRF detectors
- SSRL Beamline 6-2 / 7-2: experimental infrastructure

## Current Status
Phase 1 in progress: exploring Python toolkits for ROI detection.
Identified approaches:
- Thresholding (Otsu, sigma-based) + labeling (scipy/skimage)
- 2D peak detection (maximum_filter)
- DBSCAN for grouping nearby ROI
- bluesky-adaptive for future integration with the scanner

## Project Folder Structure
SLAC-XRF-Project/
├── CLAUDE.md     ← this file, auto-read at every session
├── Papers/       ← reference PDFs (reference with @Papers/filename.pdf)
├── data/         ← raw XRF datasets
├── scripts/      ← Python scripts
└── outputs/      ← results, figures, exports

## Code Conventions
- Always use the smak venv: C:\tools\smak\env\Scripts\activate
- Save outputs to ./outputs/
- Raw data in ./data/
