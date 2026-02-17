# Source Code (`src/`)

This folder contains reusable modules supporting a reproducible, engineering-led modelling pipeline.

The intended flow is:

Data → preprocessing → feature engineering → modelling → evaluation → saved artifacts


## Module map

### `src/config.py`
Central configuration for:
- dataset paths
- dataset subset selection (FD001–FD004)
- window lengths / strides
- optional RUL clipping
- output locations (`results/`)

### `src/data/`
Responsibilities:
- load CMAPSS text files
- assign column names
- perform schema and sanity checks
- build labels (RUL) in a leakage-safe manner

Typical functions:
- `load_cmapss()`
- `build_rul_labels()`
- `validate_trajectories()`

### `src/features/`
Responsibilities:
- classical baseline features (rolling stats, deltas, trend slopes)
- operating regime handling (when needed)
- HI utilities (normalisation, smoothing)

### `src/models/`
Responsibilities:
- interpretable RUL baselines (ridge, RF/XGB)
- HI construction models
- sequence models as controlled extensions (LSTM/TCN)

### `src/evaluation/`
Responsibilities:
- common metrics (MAE/RMSE)
- unit-level error analysis
- reliability-aware evaluation (end-of-life region, asymmetric cost framing)

### `src/utils/`
Responsibilities:
- environment checks
- I/O helpers (save/load metrics, figures)
- run id management


## Engineering conventions

- Avoid data leakage (split by unit; scale on train only).
- Prefer interpretable baselines before deep models.
- Save metrics/figures/tables consistently to `results/`.
- Keep assumptions explicit and documented in `reports/`.


## Entry points (recommended)

As the project develops, add lightweight scripts (either under `src/` or a top-level `scripts/` folder) to make runs reproducible outside notebooks:
- `preprocess.py`
- `train_rul_baselines.py`
- `train_hi_baselines.py`
- `train_sequence_models.py`
- `evaluate.py`
