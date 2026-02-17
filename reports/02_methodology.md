# 02 — Methodology

## Overview
The methodology follows an interpretable-first strategy:

1. data ingestion and validation  
2. preprocessing and leakage-safe splits  
3. feature engineering (classical baselines)  
4. RUL baselines  
5. HI estimation baselines  
6. sequence model extension (LSTM/TCN)  
7. evaluation with reliability interpretation  

## Data ingestion and validation
Minimum checks:
- correct parsing of unit and cycle
- monotonic cycle increase per unit
- no missing values introduced
- sensor sanity inspection (constant/near-constant sensors identified)

Tools:
- `src/data/data_loader.py`
- `src/data/validation.py`

## Splitting and leakage controls
- split by unit (not random rows)
- scaling fitted on training split only
- windowing performed after split to avoid mixing units

## Preprocessing
- standardisation strategy (global vs regime-aware, if required)
- optional RUL clipping (documented if used)
- handling operating settings (retain, cluster, or normalise)

## Feature engineering (classical baselines)
Examples:
- rolling mean/std over window
- deltas and trend slopes
- aggregated last-N-cycle descriptors per unit
- regime features (if needed)

## Models
### RUL baselines
- linear (ridge/lasso)
- tree-based (random forest / gradient boosting)

### HI baselines
- engineered HI (normalised degradation proxy)
- unsupervised approaches (optional) with quality evaluation

### Sequence models (extension)
- LSTM or TCN using windowed sensor sequences
- compared directly against baselines under identical splits

## Evaluation protocol
### RUL
- MAE, RMSE
- unit-level error distribution
- error vs time-to-failure (end-of-life region)

### HI
- monotonicity
- smoothness
- correlation with time-to-failure
- threshold policy behaviour (false alarms vs late warnings)

All metrics and plots are saved into `results/`.
Curated outputs and interpretation are written in `reports/03_results_summary.md`.
