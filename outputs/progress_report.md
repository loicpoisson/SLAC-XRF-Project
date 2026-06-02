# Progress report — Adaptive XRF pipeline (SLAC SSRL)

**Date:** 2026-06-01
**Project:** Adaptive XRF imaging — Phase 1
**Supervision:** Sam Webb (SSRL/SLAC)

---

## 1. Context and objective

### The base problem

In XRF (X-Ray Fluorescence) imaging at a synchrotron, we scan a sample by measuring the XRF emission spectrum at every position (pixel) with a focused beam. A classical scan is **raster**: visit every pixel in order, with a fixed measurement time (`dwell`) per pixel. For example, scanning 1 cm² at 25 µm resolution with a 10 ms dwell takes about 100 minutes.

The problem: most pixels contain nothing of interest (sample matrix, or even empty space). We waste time scanning those as long as the actually interesting regions (ROIs = Regions of Interest, where bright particles or trace elements live).

### The adaptive idea

Instead of a uniform scan, **two families of strategies**:

1. **Spatial adaptive**: do a quick coarse scan (e.g. 250 µm), identify the interesting regions, then only scan those at fine resolution.
2. **Temporal adaptive**: visit every pixel but with variable dwell — long on ROIs (good SNR), short on the background (just enough to sample).

The two can be combined.

### Paper references

- **Paper 1 (Betterton et al., 2020)** — proposes a reinforcement-learning formulation with multiple apertures and a scanner motion cost function (max velocity 200 mm/s, max acceleration 500 mm/s²). Our `travel_time()` parameters come from here.
- **Paper 2 (Boltz, Ratner & Webb, 2026)** — experimentally implements RL adaptive scanning on SSRL beamline 7-2. Uses a Gaussian approximation of Poisson noise and adaptive dwell. Reports 40× more time on ROIs vs raster, and 2.5× better MSE. This is our reference for the **temporal strategy**.

---

## 2. The data

### File structure

The data are SSRL/USDC HDF5 files produced by SMAK (Sam Webb), with:
- `main/mapdata`: a (ny, nx, n_channels) array of XRF counts
- `main/xdata`, `main/ydata`: pixel coordinates in mm
- `main/attrs/labels`: element channel names (`Al.Ka`, `Si.Ka`, `Fe.Ka`, etc.)

Non-element channels (`I0`, `I1`, `TIME`, etc.) are filtered out by `list_element_channels()`.

### Available samples

Three main families:
- **UA1_P1** — 4 resolutions available (250 / 100 / 50 / 25 µm), all at 10 ms dwell
- **UB1_P1** — same, 4 resolutions
- **FP1_P1** — 3 resolutions (250 / 100 / 50 µm), no 25 µm for P1 (only P2 has that)

This means we can compare a coarse (250 µm) detection to the "ground truth" fine (25 or 50 µm) scan on the same samples.

### XRF composite

For ROI detection we build a **composite** = sum of non-constant element channels (a "constant" channel is just instrument floor and carries no spatial information). See [hdf5_reader.py:get_composite_map](../scripts/utils/hdf5_reader.py).

---

## 3. Key concepts

### `dwell` (measurement time)

How long the beam stays on a pixel. The detector counts photons during that time. Poisson statistics: SNR ∝ √(dwell × signal).
- Our reference: uniform dwell = 10 ms.
- Longer dwell → better SNR, but longer total scan.

### `sample_frac` vs `sparsity` (two density measures)

An important distinction:
- `sample_frac` = fraction of pixels where there is **material** (above instrument floor). Measures whether the sample covers the whole scanned area or not.
- `sparsity` = fraction of pixels that are **bright** (above the matrix level).

A sample can be:
- Dense (`sample_frac` ≈ 100%) AND sparse (`sparsity` ≈ 10%) → matrix everywhere, rare particles
- Sparse on both → real off-sample void + a few material zones

**It is `sample_frac` that dictates the maximum possible spatial speedup.**

### Subpixel averaging

When scanning at coarse resolution (e.g. 250 µm) a sample with fine particles (~25 µm), a bright particle (5000 counts on one fine pixel) is "averaged" across the 100 sub-pixels of its coarse pixel:
```
coarse pixel = (99 × background_floor + 1 × particle) / 100
             = (99 × 80 + 5000) / 100 = 130 counts
```
The coarse pixel only looks slightly brighter than the matrix (which is ~85). **A sub-pixel particle becomes nearly invisible at coarse resolution.** This is a physical limit of coarse scanning, not an algorithmic flaw.

### Pareto frontier

A "Pareto frontier" on two axes (here **signal captured** and **speedup**) shows the set of optimal trade-offs: for a given speedup, the best signal one can hope to keep. No method can beat this frontier. The goal is to map it and place each method on it.

### Travel time (scanner motion)

When the scanner hops from one ROI to another it takes time. Model (Paper 1, trapezoidal):
- d ≤ v²/a (= 0.08 mm) → triangular profile: T = 2·√(d/a)
- d > v²/a → trapezoidal profile: T = v/a + d/v
- Plus a per-region setup overhead (~500 ms: positioning, triggering).

On our small samples (23 × 16 mm) travel stays at 0.3% of total time. On samples 10× larger it would still stay <1%. **Travel is not a blocker for our pipeline.**

---

## 4. Workflow / pipeline / decision tree

The full pipeline starting from an unknown sample:

```
   ┌─────────────────────────────────┐
   │ 1. COARSE SCAN 250 µm           │
   │    (fast, ~1-2 min on UA1)      │
   └─────────────┬───────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────┐
   │ 2. COMPUTE DESCRIPTORS           │
   │    - sample_frac (% material)    │
   │    - sparsity   (% bright)       │
   │    - concentration (signal top10)│
   │    - dynamic_range               │
   │    [script 09_recommend...]      │
   └─────────────┬───────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────┐
   │ 3. STRATEGY CHOICE               │
   │                                  │
   │ sample_frac < 30% (sparse)       │
   │   -> Spatial ROI (script 03/06)  │
   │                                  │
   │ sample_frac 30-70% (medium)      │
   │   -> Combined (script 08)        │
   │                                  │
   │ sample_frac > 70% (dense)        │
   │   -> Pure temporal (script 07)   │
   │                                  │
   │ + concentration > 60%            │
   │   -> binary dwell                │
   │ + dynamic_range > 2.5            │
   │   -> log dwell                   │
   │ otherwise                        │
   │   -> linear dwell                │
   └─────────────┬───────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────┐
   │ 4. EXECUTE FINE SCAN             │
   │    - spatial mask (where)        │
   │    - variable dwell (how long)   │
   └─────────────┬───────────────────┘
                 │
                 ▼
   ┌─────────────────────────────────┐
   │ 5. OUTPUTS                       │
   │    - per-element XRF maps        │
   │    - Pareto and descriptor plots │
   │    - metrics report              │
   └─────────────────────────────────┘
```

The decision criteria come from empirical observation: we tried every type and saw that `sample_frac` is the best predictor of the maximum possible spatial speedup (1 / sample_frac).

---

## 5. Implemented scripts (with why, how, and what we learned)

### Script 01 — Basic ROI detection

**File:** [01_roi_detection.py](../scripts/01_roi_detection.py)

**Initial question:** how do we automatically identify interesting regions in a coarse XRF scan?

**Method:**
1. Load the HDF5, compute the composite (sum of non-constant channels)
2. Threshold the composite with a robust method:
   - **MAD**: median + k·MAD (very robust to outliers)
   - **trimmed**: mean + std after dropping the top 5%
   - **IQR**: Q75 + k·IQR (boxplot whisker)
3. The `auto` mode tries all three and keeps the one that maximizes the **Fisher inter-class variance** (background vs signal separation)
4. Label connected components with `scipy.ndimage.label`
5. Compute bounding boxes in pixel and mm coordinates

**Design choices discussed:**
- *Why not Otsu directly?* We tried it: Otsu on the UA1 composite (histogram with very bright outliers) places the threshold between matrix and particles, not between void and matrix. We did implement `_thresh_otsu` but use it via `sample_mask` (cf. script 04), not as a direct threshold.
- *Why k=1.0 by default?* Empirical. k=2.0 (initial default) missed too many ROIs on UA1 (only 73 detected vs ~209 expected).
- *Why min-px=1 by default?* At 250 µm pixel, a real particle **can fit in a single pixel**. Filtering at min-px=2 lost 136 ROIs on UA1.

**What we learned:** ROI detection on the coarse is fragile due to subpixel averaging — this motivated all the variants that follow.

---

### Script 02 — Trajectory planning (TSP)

**File:** [02_trajectory.py](../scripts/02_trajectory.py)

**Initial question:** once the ROIs are detected, in what order should we visit them to minimize total motion time?

**Method:**
1. Apply a **safety margin** around each ROI:
   `margin = ½ × coarse_pixel + N × fine_pixel`
   = 125 µm + 5 × 25 µm = 250 µm per side.
   Justification: Paper 2 reports a scanner positioning error ≈ 35 µm, plus uncertainty on the real extent of the particle (sub-pixel coarse).
2. Compute a **nearest-neighbor** tour (greedy O(N²))
3. Improve with **2-opt** (reverse sub-segments if shorter)
4. Improve with **Or-opt** (move blocks of 1–3 nodes)
5. Print the scan table + total time estimate

**Result on UA1_P1 (209 ROIs):**
- 251.7 mm with NN → 227.1 mm with 2-opt → 216.5 mm with Or-opt
- **Total reduction: −14% on motion distance**
- **BUT: travel = 0.3% of total time** (90 min scan). So TSP optimization has a marginal effect for our case.

**What we learned:** at this sample scale, TSP optimization is intellectually satisfying but low-impact. Keep it for future large samples.

---

### Script 03 — First validation (reference script)

**File:** [03_validate_roi.py](../scripts/03_validate_roi.py)

**Initial question:** if we had only scanned the ROIs detected at coarse, how much of the real signal would we have captured? How much time saved?

**Method:**
1. Run ROI detection on the coarse 250 µm
2. Load the fine 25 µm as "ground truth"
3. For each ROI (with margin), find the fine pixels inside
4. Compute:
   - `signal_captured` = sum(fine_in_ROIs) / sum(fine_total)
   - `area_frac` = pixels_in_ROIs / pixels_total
   - `speedup` = raster_time / adaptive_time

**Results on UA1_P1:**
- ROIs detected: 209
- Pixels scanned: 30% of the area
- **Signal captured: 44%** (target was > 90%!)
- Speedup: 3.3×

**Diagnosis:** 56% of signal missed — clearly visible on the "missed signal" panel of [SMW_UA1_P1_25um_10ms_12000_0_001_validation.png](SMW_UA1_P1_25um_10ms_12000_0_001_validation.png) where bright particles scatter between the ROIs. **This is subpixel averaging in action**: these particles didn't stand out at coarse because they were diluted in the 100-to-1 average of their coarse pixel.

**What we learned:** naive ROI detection doesn't work well on this sample. We need to either lower the threshold (with more false positives) or change strategy entirely. → motivates scripts 04+.

---

### Script 04 — Morphological mask (Option 3)

**File:** [04_validate_morphological.py](../scripts/04_validate_morphological.py)

**Initial question:** what if at coarse we did NOT look for particles (impossible due to subpixel averaging) but only for the **sample footprint** (material vs void)?

**Insight:** the coarse scan can't detect fine particles but it can very reliably distinguish "there is material" vs "there is nothing" (= sample edge).

**Method:**
1. Compute the **low threshold** via `sample_mask()` ([utils/roi_utils.py:sample_mask](../scripts/utils/roi_utils.py)):
   - 3 modes: `otsu_lower`, `percentile`, `otsu_clip`
   - Default `otsu_lower`: take the bottom 50% of pixels, run Otsu on them → finds the void/matrix boundary
2. Morphological cleanup:
   - **closing**: fills small holes inside the sample (keeps the benefit)
   - **opening**: (DISABLED by default — it removes isolated particles!)
3. Project the coarse mask onto the fine grid
4. Measure signal captured / speedup

**Critical choice:** at first we had `closing + opening`. Result: 11% of excluded pixels contained **45% of the signal** — opening was deleting isolated particles at the sample edge. Fix: closing-only by default, `opening_px=0`. See the diff in [roi_utils.py:sample_mask](../scripts/utils/roi_utils.py).

**Results on UA1_P1:**
- Mask = 97.7% of pixels (almost everything)
- **Signal captured: 89%** (vs 44% with script 03)
- Speedup: 1.02× (almost nothing)

**What we learned:** on UA1 the sample covers 95%+ of the scanned area. So we can't gain much spatially → physical ceiling. Morpho captures nearly all the signal but barely saves any time. → motivates cascade and temporal.

---

### Script 05 — Hierarchical multi-resolution cascade (Option 4)

**File:** [05_validate_hierarchical.py](../scripts/05_validate_hierarchical.py)

**Initial question:** if a `250 → 100 → 50 → 25 µm` cascade progressively refined the mask, would we gain more speedup?

**Idea:** each intermediate level scans ONLY inside the previous level's mask. As we better distinguish void/matrix at finer resolution, the mask should tighten step by step.

**Method:**
1. Load the 4 resolutions
2. Level 0: `sample_mask(coarse_250)` → M0
3. Level 1: project M0 onto the 100 µm grid, re-run `sample_mask` on in-mask pixels → M1 ⊂ M0
4. Level 2: same with 50 µm → M2 ⊂ M1
5. Level 3 (final): scan inside M2 at 25 µm
6. Total time = sum(pixels_scanned_at_each_level × dwell)

**Results on UA1_P1 (cascade 250→100→50→25):**
- Mask goes 96.4% → 96.7% → 96.8% → 97.0% (barely any refinement!)
- Total time = 6350 s vs raster 6014 s
- **Speedup: 0.95× (SLOWER than direct raster!)**

**Diagnosis:** on a dense sample (UA1) the intermediate refinement removes barely 1–3% per level. The cost of intermediate scans (1570 s for 100+50 µm) exceeds the gain on the final scan. **The cascade is counter-productive on a dense sample.**

**What we learned:**
- Cascade is only useful if the mask shrinks significantly at each level
- On a dense sample, the physical ceiling (sample_frac ≈ 95%) kills any spatial saving
- → motivates **temporal** adaptive: visit everything, but with variable dwell

---

### Script 06 — Pareto sweep (overview)

**File:** [06_pareto_sweep.py](../scripts/06_pareto_sweep.py)

**Initial question:** instead of one method at a time, plot the **Pareto frontier** signal_captured vs speedup to compare all spatial methods.

**Method:**
1. For each sample, load coarse + fine
2. Sweep 4 k values for ROI (0.3, 0.5, 1.0, 1.5)
3. Sweep 5 level values for Morpho (10, 25, 40, 60, 75)
4. For each config, compute: area_frac, signal_captured, speedup (with and without travel)
5. Save a CSV + a Pareto scatter plot

**Results UA1_P1 + UB1_P1:** [pareto_UA1_P1_UB1_P1_FP1_P1.png](pareto_UA1_P1_UB1_P1_FP1_P1.png)
- Morpho dominates at the bottom (speedup < 1.5×): ~89% signal at 1×
- ROI dominates at the top (speedup > 1.5×): 67% at 1.78×, 46% at 3.0×
- UA1 and UB1 nearly overlap → same kind of sample (dense + diffuse)

**Results FP1_P1:** [pareto_FP1_P1.png](pareto_FP1_P1.png)
- FP1 is **better** than UA1/UB1: ROI k=0.3 captures 89% of signal at 1.36× speedup
- Why? More concentrated signal (73% vs 60% for UA1) AND fine at 50 µm (less subpixel averaging than 25 µm)

**What we learned:**
- The Pareto curves are the **main figure** for Sam Webb
- The best method depends on the sample, there's no universal "best"
- On dense diffuse samples (UA1/UB1) we cap around 3× speedup with 46% signal captured
- On dense concentrated samples (FP1) we reach 1.36× speedup with 89% captured

---

### Script 07 — Temporal adaptive (Paper 2 style)

**File:** [07_validate_temporal.py](../scripts/07_validate_temporal.py)

**Initial question:** what if we don't optimize **space** but **time**? Visit every pixel, but with variable dwell tied to the coarse-projected signal.

**Reference:** this is exactly what the RL of **Paper 2** (Boltz-Webb 2026) does. Their policy learns to allocate time to pixels with signal. They report 40× more time on ROIs and 2.5× better MSE.

**Method:**
1. Project the coarse 250 µm signal onto the fine grid (each coarse pixel → its 100 sub-pixels)
2. Allocate a dwell per fine pixel via a strategy:
   - **binary**: dwell_high if coarse_signal > threshold else dwell_low
   - **linear**: linear interpolation between dwell_low and dwell_high
   - **log**: logarithmic interpolation
3. **`--match-budget` mode**: normalize dwells so total time = raster time (fair comparison at equal time)
4. Compute metrics:
   - **info_ratio** = sum(signal × dwell)_adaptive / sum(signal × dwell)_raster — "photons collected on signal"
   - **bright_pixel_boost** = fraction of time allocated to the top-10% brightest pixels

**Results UA1_P1 with matched budget (100 min = same as raster):**

| Strategy | Info ratio | Bright-pixel boost |
|----------|------------|---------------------|
| **Binary** | **1.62×** | **2.01×** |
| Linear | 1.59× | 1.91× |
| Log | 1.37× | 1.59× |

See [SMW_UA1_P1_250um_10ms_12000_0_001_temporal_binary.png](SMW_UA1_P1_250um_10ms_12000_0_001_temporal_binary.png) — the dwell map clearly shows the correlation: bright pixels → long dwell (yellow), dark pixels → short dwell (blue).

**FP1_P1 binary matched result:** info ratio **2.16×** (even better, signal more concentrated).

**What we learned:**
- Temporal adaptive gives a **strict Pareto gain over raster**: same time, better quality
- Binary > Linear > Log: all-or-nothing allocation is the most efficient
- This is the only family that sacrifices nothing

---

### Script 08 — Combined spatial + temporal

**File:** [08_validate_combined.py](../scripts/08_validate_combined.py)

**Initial question:** are the two families (spatial and temporal) complementary? Can we combine sample_mask (exclude void) + variable dwell (better allocation inside the mask)?

**Method:**
1. Compute the spatial mask via `sample_mask` (level=60 by default)
2. For in-mask pixels, allocate a variable dwell (binary/linear/log)
3. For out-of-mask pixels, dwell = 0 (not scanned)
4. Options:
   - `--match-budget`: scale in-mask dwells so total = raster_total
   - without: keep the spatial speedup + temporal gain

**Results UA1_P1 (combined L=60 binary [1,50] matched):**
- Speedup: 1.00× (matched)
- Signal captured: 80.8% (12% lost via the mask)
- Info ratio: 1.30× (worse than pure temporal 1.62×)
- → **Combined loses on UA1** because losing 12% of signal costs more than gaining via focused allocation

**Results FP1_P1 (combined L=60 binary [1,30] WITHOUT match-budget):**
- **Speedup: 1.44×** (shorter scan)
- Signal captured: 87.9%
- **Info ratio: 1.45×** (on top of the speedup!)
- Bright-pixel boost: 1.46×
- → **Triple gain on FP1**: faster, signal preserved, better SNR on ROIs

See [SMW_FP1_P1_250um_10ms_12000_0_001_combined_otsu_lower_L60_binary_low1.0_high30.0.png](SMW_FP1_P1_250um_10ms_12000_0_001_combined_otsu_lower_L60_binary_low1.0_high30.0.png).

**What we learned:**
- Combined works on **concentrated** samples (FP1) because the mask drops little signal and the dwell exploits the strong concentration
- On **diffuse** samples (UA1/UB1), the mask loses too much, pure temporal stays better
- → choice rule: `concentration > ~70%` and `sample_frac < ~80%` favor combined

---

### Script 09 — Automatic recommender

**File:** [09_recommend_strategy.py](../scripts/09_recommend_strategy.py)

**Initial question:** can we automatically choose the right strategy from a coarse scan alone?

**Method:** computes 4 descriptors on the coarse composite:
1. **`sample_frac`**: fraction of pixels above the Otsu_lower threshold (= material vs void)
2. **`sparsity`**: fraction above the matrix threshold (= particles vs matrix)
3. **`concentration`**: signal_in_top10pct / signal_total
4. **`dynamic_range`**: log10(max / median)

Then applies heuristic rules (calibrated on the 3 tested samples):

```
sample_frac < 30%  -> ROI spatial (sparse sample, big gain possible)
sample_frac 30-70% -> Combined (mixed zone)
sample_frac > 70%  -> Pure temporal (dense, spatial ceiling)

concentration > 60% -> binary dwell
dynamic_range > 2.5 -> log dwell (otherwise)
else                -> linear dwell
```

**Results on our 3 samples:**

| Sample | sample_frac | concentration | dyn_range | Reco |
|--------|-------------|---------------|-----------|------|
| UA1_P1 | 77% | 60% | 2.84 | Temporal + log |
| UB1_P1 | 80% | 68% | 3.49 | Temporal + binary |
| FP1_P1 | 75% | 73% | 3.79 | Temporal + binary |

**Identified limitation:** the current rule sends FP1 to "pure temporal" while empirically **combined** works better. The rule should include: `if dense AND very_concentrated → try combined first`. To be calibrated with more samples.

---

### Script 10 — Production pipeline (coarse-only)

**File:** [10_run_pipeline.py](../scripts/10_run_pipeline.py)

**Initial question:** how do we deploy this on a new sample where we DO NOT have the fine scan?

**Method:**
1. Load only the coarse HDF5
2. Compute descriptors (script 09 logic)
3. Choose strategy automatically (or via `--strategy force_*`)
4. Build the spatial mask and dwell map
5. Budget-match by default so total time ≤ raster baseline (guarantees speedup ≥ 1)
6. Outputs:
   - Printed report (chosen strategy, parameters, time estimate)
   - JSON scan plan (regions + dwells) — directly readable
   - Pickle plan in SMAK-compatible format (rectangles with yrange/zrange/dwell)
   - Figure showing the planned scan

**This is the only script that runs on coarse alone**, with no need for ground truth. It is the deployment target.

---

## 6. Performance optimization (vectorization)

Script 06 (Pareto sweep) on FP1 was initially **stuck** in `group_rois` (the function that merges nearby ROIs). Diagnosis: O(N³) with pure Python loops on N=500 ROIs.

Vectorized functions with numpy broadcasting:

| Function | Before | After | Gain |
|----------|--------|-------|------|
| `group_rois` | O(N³) Python | O(N³) numpy ops | ~100× |
| `nearest_neighbor_tour` | O(N²) Python | O(N²) numpy | ~50× |
| `get_bounding_boxes` | Python loop + `np.where` | `ndimage.find_objects` | ~20× |
| `two_opt_improve` | O(N²)/pass Python | O(N²)/pass numpy | ~50× |

Concrete impact: FP1 sweep went from **stuck** → **30 s**.

Not vectorized: `or_opt_improve` (used only by 02_trajectory.py, already fast after 2-opt).

---

## 7. Consolidated results

### Main findings

1. **Physical ceiling of spatial adaptive on dense samples**
   On UA1/UB1 (sample_frac ≈ 80%) we cannot beat 1.25× spatially without sacrificing signal. This is a physical limit, not algorithmic.

2. **Subpixel averaging kills coarse ROI detection**
   A fine particle (~25 µm) at 5000 counts inside a 250 µm coarse pixel only contributes ~130 counts. The "natural" threshold at 153 (k=1.0) misses it. The hierarchical cascade doesn't help.

3. **Temporal adaptive is a strict Pareto gain**
   At equal time vs raster we capture 1.6× more signal-photons. No sacrifice. This is the universally good method (Paper 2 confirms it experimentally).

4. **Combined is the best on concentrated samples**
   On FP1 (concentration 73%), combined non-budget: speedup **1.44×** + info ratio **1.45×** + signal **88%** simultaneously. Triple gain.

5. **Travel time is negligible**
   On our samples (23 × 16 mm) travel < 0.5% of total time. On samples 10× larger it would still stay <5%. Not a factor in our decisions.

### Figures to look at, in order

1. **Naive ROI detection that fails:** [SMW_UA1_P1_25um_10ms_12000_0_001_validation.png](SMW_UA1_P1_25um_10ms_12000_0_001_validation.png) — right panel shows all the lost signal
2. **Morphological mask:** [SMW_UA1_P1_250um_10ms_12000_0_001_morphological_k2.png](SMW_UA1_P1_250um_10ms_12000_0_001_morphological_k2.png) — 89% captured but 1× speedup
3. **Counter-productive cascade:** [UA1_P1_hierarchical_250_100_50_25.png](UA1_P1_hierarchical_250_100_50_25.png)
4. **Pareto UA1+UB1+FP1:** [pareto_UA1_P1_UB1_P1_FP1_P1.png](pareto_UA1_P1_UB1_P1_FP1_P1.png) — **main figure**
5. **Pareto FP1 (best sample):** [pareto_FP1_P1.png](pareto_FP1_P1.png)
6. **Temporal binary UA1:** [SMW_UA1_P1_250um_10ms_12000_0_001_temporal_binary.png](SMW_UA1_P1_250um_10ms_12000_0_001_temporal_binary.png) — dwell allocation
7. **Combined FP1 (triple gain):** [SMW_FP1_P1_250um_10ms_12000_0_001_combined_otsu_lower_L60_binary_low1.0_high30.0.png](SMW_FP1_P1_250um_10ms_12000_0_001_combined_otsu_lower_L60_binary_low1.0_high30.0.png)

---

## 8. Code architecture

```
scripts/
├── 01_roi_detection.py       — basic ROI detection
├── 02_trajectory.py          — TSP planning (NN + 2-opt + Or-opt)
├── 03_validate_roi.py        — ROI validation against ground truth
├── 04_validate_morphological.py  — Otsu mask + closing
├── 05_validate_hierarchical.py   — multi-resolution cascade
├── 06_pareto_sweep.py        — method sweep + Pareto figure
├── 07_validate_temporal.py   — adaptive dwell
├── 08_validate_combined.py   — spatial + temporal
├── 09_recommend_strategy.py  — descriptors + auto reco
├── 10_run_pipeline.py        — PRODUCTION (coarse only)
└── utils/
    ├── hdf5_reader.py        — HDF5 loader + composite
    ├── roi_utils.py          — detection, thresholding, TSP, merge cost
    └── validation_utils.py   — mask projection, metrics, travel overhead
```

---

## 9. What's next / proposals

Four possible directions, to choose from:

### A. Test on larger samples
We also have `SMW_UA1_1x1_50um_*` and `SMW_FP1_250x250_250um_*`. Bigger samples → more ROIs → more visible travel → validation of vectorization at scale.

### B. Refine the recommender
Calibrate a "dense + very concentrated → combined" rule to better predict FP1. Test on more samples to check robustness.

### C. Unified production script (10_run_pipeline.py) — **DONE**
A single script taking a coarse path, computing descriptors, picking and running the strategy, outputting the final report. Already implemented. This is the **"one-shot" to show Sam Webb**.

### D. Phase 2 — ML training
Train a model on paired (coarse, fine) datasets that predicts the optimal dwell map directly. Candidates:
- **U-Net** as a first POC (PyTorch, image-to-image)
- **SLADS** (Supervised Learning Approach for Dynamic Sampling) — concurrent of gpCAM, designed for adaptive microscopy
- **gpCAM** (LBL) — Gaussian Process active learning, online (no pretraining)

GPU credits from SLAC expected; in the meantime we'll bootstrap on Google Colab.

---

## 10. Limits and future work (beyond this phase)

- **Real-time adaptive loop on the beamline**: currently we simulate using the fine scan as a posteriori ground truth. Phase 2 will integrate with bluesky / bluesky-adaptive to drive the scanner live.
- **RL (Paper 2)**: binary/linear/log strategies are fixed heuristics. Paper 2 uses a learned policy. This is the logical next step after the supervised U-Net.
- **Multi-channel processing**: currently we aggregate into a composite. Detecting on a single key element (e.g. Au.La for gold particles) could be more biologically meaningful.
- **Calibrate scanner parameters**: we use v_max = 200 mm/s and a_max = 500 mm/s² from Paper 1. To be confirmed with the actual values for the SSRL 7-2 / 6-2 scanner used.

---

*End of report.*