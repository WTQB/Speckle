# Speckle Reflection Analysis for Fluorescent Security Tags

## 1) Project Purpose
This repository contains analysis tooling for evaluating and reducing flash-induced specular reflection artifacts (“speckle”) in smartphone images of fluorescent security tags.

The core problem is that flash angle and capture geometry can cause localized glare and diffuse washout, which can distort the apparent material fingerprint used for authentication.

This project aims to build and validate robust metrics that:
- quantify speckle/disruption,
- remain stable under intentional intensity variation in the measurement system,
- and align with what is visually observed in real image data.

---

## 2) High-Level Goals
### Primary goals
- Develop a metric that is representative of visible speckling in real captured images.
- Account for dynamic flash range and expected fluorescence intensity variation.
- Identify strong bimodal distributions that support robust low/high disruption classification.
- Keep visual comparison as the primary validation mechanism.

### Practical goals
- Rapidly compare candidate metrics.
- Understand angular dependence and sample-level behavior.
- Support comparison across tag groups/samples (e.g., HER, FED, FEDS).

---

## 3) Repository Structure
- `/Users/GoergeH/Documents/github/Speckle/SpeckleAnalysis.py`
  - Original/general sweep analysis script for JSON parameter sweeps.
- `/Users/GoergeH/Documents/github/Speckle/flash_disruption_index.py`
  - Main current development script for FDI and experimental bimodal metric discovery.
- `/Users/GoergeH/Documents/github/Speckle/*.json`
  - Sweep result files (positions + capture metadata + existing speckle_area_percent).
- `/Users/GoergeH/Documents/github/Speckle/figures/`
  - Generated figures and metric ranking outputs.
- `/Users/GoergeH/Documents/github/Speckle/FullData/DiscoMotion.zip`
  - Archived raw/extracted scan data source.

---

## 4) Data Model and Inputs

## 4.1 JSON sweep files
Each JSON file represents a position sweep over geometry parameters:
- height
- yaw
- pitch
- roll

Per-position entries typically include:
- `height_mm`, `yaw_degrees`, `pitch_degrees`, `roll_degrees`
- `timestamp`
- `capture_time_seconds`
- `speckle_area_percent` (existing baseline metric)
- server metadata (`server_response`)

## 4.2 Image scan folders
`flash_disruption_index.py` expects extracted scan folders under an image root (default `/tmp/speckle_images/DiscoMotion`).
Each matched scan folder contains `server_data` with yellow patch images named like:
- `frame_*_flash_*.yellow.png`
- optionally `frame_master.yellow.png`

These images are used for metric calculation and visual validation.

## 4.3 Timestamp matching
Positions from JSON are matched to scan folders by nearest timestamp (with tolerance). If the relevant date range is missing from the image directory, a tag will show `0 matched positions`.

---

## 5) Current Analysis Workflow

## 5.1 Pipeline overview
1. Load selected JSON files.
2. Match each position to an extracted scan folder by timestamp.
3. Compute FDI and sub-metrics from yellow patch images (flash-aware, intensity-normalized).
4. Calibrate FDI to reference-like tags (prefer `F53S2/FEDS2` patterns when available).
5. Generate visual-first validation outputs (pooled + per-tag visual comparisons).
6. Run experimental metric sweep to discover strong intrinsic bimodal distributions.
7. Build best-metric dashboard for iterative development.

## 5.2 Why this is metric-first now
Experimental metric ranking no longer uses `speckle_area_percent` to define low/high classes. For each candidate metric:
- low/high cluster split is inferred from the metric’s own histogram valley between peaks,
- the ranking emphasizes intrinsic bimodality + cluster separation + cluster balance.

`speckle_area_percent` is retained only as a diagnostic overlay/reference.

---

## 6) Core Metric Logic

## 6.1 FDI (Flash Disruption Index)
FDI combines multiple sub-metrics computed from the high-flash image and flash-delta behavior:
- color fidelity loss
- saturation loss
- contrast loss
- texture disruption
- blue spatial variance

Important implementation details:
- global intensity normalization is applied per flash frame to reduce bias from intentional fluorescence variation,
- flash sensitivity term is included (growth in disruption across flash range),
- raw FDI is calibrated with a reference baseline and scale.

## 6.2 Experimental distribution sweep
Candidate distributions include (non-exhaustive):
- luminosity statistics,
- color ratios (`B/G`, `B/R`, `(R+G)/B`),
- log-chromaticity (`log(B/G)`, `log(R/G)`),
- normalized channel differences,
- hue-distance-to-yellow,
- Minkowski distances (L1/L2/L3) to ideal yellow,
- flash-delta variants for all above.

Each candidate receives scores for:
- bimodality strength,
- separation of inferred low/high clusters,
- cluster balance (with penalty for tiny minority clusters).

---

## 7) Visual Outputs (Most Important for Development)

The visual comparison outputs are intentionally the crux of evaluation.

## 7.1 Core FDI visuals
- `fdi_visual_comparison.png`
  - Pooled representative visual comparison.
- `fdi_visual_comparison_<TAG>.png`
  - Per-sample visual comparison (recommended for sample-specific tuning).
- `fdi_histogram_trends.png`
  - Histogram diagnostics.

## 7.2 Experimental metric visuals
- `experimental_bimodal_distributions.png`
  - Top candidate metric histograms.
- `experimental_visual_<metric>.png`
  - Visual validation for top experimental metrics.
- `best_metric_dashboard.png`
  - Single-page dashboard for the current best metric:
  - distribution/threshold diagnostics,
  - per-tag cluster rates,
  - diagnostic scatter plots,
  - real-image comparison row,
  - metric-specific pixel histogram row.

## 7.3 Axis-scale consistency
Histogram figures are configured to use consistent y-axis scaling across compared panels/rows to avoid misleading visual interpretation from autoscaling.

---

## 8) Running the Analysis

From repo root:

```bash
python3 flash_disruption_index.py
```

Use explicit inputs if needed:

```bash
python3 flash_disruption_index.py \
  --json /Users/GoergeH/Documents/github/Speckle/F53_FEDS2A_2026-02-26_10-26-33.json \
  --image-dir /path/to/extracted/DiscoMotion
```

Notes:
- `--image-dir` must contain scan folders for the relevant capture dates.
- If a JSON has no matching scan folders, that tag is skipped with a warning.

---

## 9) Current Known Constraints
- If reference sample scan folders are unavailable (e.g., missing date in image directory), reference calibration falls back to the next available non-empty tag.
- Some high-bimodality candidates can still be operationally weak if cluster balance is poor; current scoring penalizes this but should still be visually reviewed.
- Existing `speckle_area_percent` is still useful as a diagnostic baseline, but not treated as ground-truth label for cluster assignment in the experimental sweep.

---

## 10) Suggested Workflow for New Users
1. Confirm image extraction path contains scan folders matching JSON timestamps.
2. Run `flash_disruption_index.py`.
3. Review in this order:
   1. `best_metric_dashboard.png`
   2. `fdi_visual_comparison_<TAG>.png` for each tag
   3. `experimental_visual_<metric>.png` for top candidates
   4. `experimental_bimodality_ranking.csv`
4. Decide if the top metric is visually representative across samples.
5. If not, add/modify candidate distributions and rerun.

---

## 11) Development Direction
Near-term direction is to converge from “candidate ranking” to a single robust production metric by:
- selecting top visually validated candidates,
- testing consistency across tag groups and lighting conditions,
- and refining calibration around the desired low-speckle reference sample (especially matt-coated references like F53S2/FEDS2).

