# Source Code (`src/`)

`src/` contains reusable, testable pipeline code for CMAPSS reliability modeling.

## Engineering Intent

Code in `src/` should remain:
- reproducible
- leakage-safe
- explicit about decision-relevant trade-offs

## Module Responsibilities

### `src/config.py`

Single source of truth for:
- repository paths
- default subset and preprocessing settings
- window/step and clipping controls
- output directories

### `src/data/`

- CMAPSS load/parsing and integrity checks
- train/test RUL construction
- unit-based split, scaling, windowing, and tabular feature utilities

Core APIs:
- `load_cmapss_subset()`
- `unit_train_val_split()`
- `fit_scaler()` / `transform_scaler()`
- `make_windows()` / `make_window_features()`

### `src/features/`

Feature engineering utilities for classical models and HI workflows.

### `src/models/`

Baseline and extension model training logic.

### `src/evaluation/`

Metric computation, stratified error analysis, and plotting helpers.

### `src/utils/`

Environment and general helper utilities.

## How To Run

### Import integrity

```bash
python scripts/check_imports.py
```

Expected output:
- `[PASS]` lines for each required module import.

Artifacts:
- none written.

### Pipeline smoke test

```bash
python scripts/smoke_test.py
```

Expected output:
- `SMOKE_TEST=PASS` with train/val/test, window, and tabular shapes.

Artifacts:
- none written.

### Minimal module baseline run

```bash
python -m src.run_baseline
```

Expected output:
- baseline training summary printed in terminal.

Artifacts written:
- `results/metrics/baseline_train_metrics.json`
- `results/tables/baseline_train_metrics.csv`

### Full baseline experiment run

```bash
python scripts/train_baselines.py --subset FD001 --rul_cap 125 --window 30 --step 1 --val_fraction 0.2 --seed 42
```

Expected output:
- saved artifact paths and model comparison table.

Artifacts written:
- `results/metrics/baselines_FD001.json`
- `results/tables/baseline_comparison_FD001.csv`
- `results/figures/pred_vs_true_<model>_FD001.png`
- `results/figures/error_vs_rul_<model>_FD001.png`

## Decision-Use Framing

Implementation choices in `src/` should preserve decision validity:
- RUL outputs must remain calibrated enough for lead-time policy decisions.
- HI features should reflect monotonic degradation behavior for escalation logic.
- evaluation should expose near-failure error behavior, not only global averages.

## Commit Boundaries

Committed:
- all source code under `src/`

Not committed:
- generated datasets, run artifacts, and model checkpoints produced by execution
