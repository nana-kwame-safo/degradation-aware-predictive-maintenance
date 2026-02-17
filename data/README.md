# Data Setup (NASA CMAPSS)

This project uses the **NASA CMAPSS Turbofan Engine Degradation** dataset to model progressive degradation in multi-sensor time-series and to support reliability-focused predictive maintenance.

The dataset is **not stored in this repository**. This document defines the expected local layout, data dictionary, and reproducibility rules used throughout the project.


## Local folder layout

- `data/raw/` contains the original downloaded CMAPSS files (not committed)
- `data/processed/` contains cleaned and transformed artifacts produced by the pipeline (not committed)

Added `.gitignore` already excludes both directories.


## Expected files

Place the downloaded CMAPSS files under `data/raw/`.

Example (FD001):
- `data/raw/train_FD001.txt`
- `data/raw/test_FD001.txt`
- `data/raw/RUL_FD001.txt`

FD002–FD004 follow the same naming pattern.


## Data dictionary (standard CMAPSS format)

CMAPSS text files are whitespace-delimited with no header. The conventional column meanings are:

- `unit`: engine identifier (integer)
- `cycle`: time index / flight cycle (integer)
- `op_setting_1..3`: operating condition settings (continuous)
- `sensor_1..21`: sensor measurements (continuous)

A typical schema therefore has **26 columns**:

1. `unit`, 2. `cycle`, 3–5. `op_setting_1..3`, 6–26. `sensor_1..21`

Notes:
- Some sensors can be constant or near-constant depending on the subset.
- Operating conditions vary by subset and influence degradation trajectories.


## Labels and targets

### Remaining Useful Life (RUL)

For *training* trajectories (run-to-failure), the standard label is:

- `RUL = max_cycle(unit) - cycle`

For *test* trajectories, the provided file `RUL_FD00x.txt` gives the **true remaining cycles at the final observed cycle** for each unit. The pipeline should reconstruct per-cycle RUL for test units consistently.

Common practice (optional, but must be documented):
- **RUL clipping** to a maximum (e.g., 125) to reduce the impact of early-life noise.

If clipping is used, record:
- clip value
- justification
- effect on evaluation

### Health Index (HI)

HI is treated as a **degradation state signal**, typically:
- bounded (e.g., 0–1)
- decreasing or increasing monotonically with degradation
- smooth enough for threshold-based decisions

HI ground truth is not directly provided; it is constructed and evaluated via quality metrics.


## Dataset variants (FD001–FD004)

CMAPSS includes multiple subsets with different operating regimes and fault modes.

Recommended modelling progression:
1. **FD001** as the baseline subset (simpler operating regime).
2. Extend to FD002–FD004 once baselines are stable.

When you switch subset, record:
- which subset you used
- why it was chosen
- implications for generalisation and evaluation


## Reproducibility and leakage control

This project is assessed as a reliability problem. To preserve validity:

- Split train/validation/test by **unit**, not by row.
- Compute scalers and feature normalisation on **train only**.
- Ensure windowing does not mix units across splits.
- Treat any look-ahead feature (e.g., future sensors) as leakage.


## Processed artifacts (produced by the pipeline)

The project will generate processed files under `data/processed/`, for example:
- cleaned train/test tables (with column names)
- feature tables for classical baselines
- windowed tensors for sequence models

Suggested naming:
- include subset: `FD001`
- include stage: `clean`, `features`, `windows`

Example:
- `data/processed/FD001_clean.parquet`
- `data/processed/FD001_features_baseline.parquet`
- `data/processed/FD001_windows_L50_S1.npz`


## Validation checks (minimum)

Before modelling, confirm:
- each `unit` has strictly increasing `cycle`
- sensor columns parse as numeric
- no missing values introduced by parsing
- basic sanity plots show plausible degradation trends

These checks are implemented in `src/data/validation.py` and should be executed early.
