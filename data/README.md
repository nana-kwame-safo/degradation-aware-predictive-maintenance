# Data README

This document defines dataset acquisition, validation, and leakage controls for CMAPSS usage in this repository.

## Purpose

Data handling is designed for reproducible reliability analysis, not only benchmark scoring.

Core requirements:
- auditable target construction
- strict unit-based data partitioning
- deterministic preprocessing behavior

## Official Source (NASA CMAPSS)

Use the official NASA Prognostics Center of Excellence (PCoE) CMAPSS dataset source.

Download all files for the subset(s) you plan to run:
- `train_FD00x.txt`
- `test_FD00x.txt`
- `RUL_FD00x.txt`

Store them under:
- `data/raw/cmapss/`

Example FD001:
- `data/raw/cmapss/train_FD001.txt`
- `data/raw/cmapss/test_FD001.txt`
- `data/raw/cmapss/RUL_FD001.txt`

## Missing `RUL_FD00x.txt`: Required Recovery Path

If you have train/test files but no RUL file:
1. Re-download the matching subset from the same NASA source.
2. Verify file naming matches the subset exactly (`FD001`..`FD004`).
3. Confirm the RUL file is present before running smoke/evaluation scripts.

Without `RUL_FD00x.txt`, test RUL labels cannot be reconstructed and full end-to-end evaluation is expected to fail.

## Local Layout

- `data/raw/cmapss/`: source CMAPSS text files (not committed)
- `data/interim/`: optional intermediate outputs (not committed)
- `data/processed/`: model-ready outputs (not committed)

## Data Schema

CMAPSS raw files are whitespace-delimited with no header.

Expected columns:
- `unit_id`
- `cycle`
- `op_setting_1..3`
- `sensor_1..21`

Total expected raw columns: 26.

## Target Construction

RUL construction implemented in `src/data/data_loader.py`:
- train rows: `max_cycle(unit_id) - cycle`
- test rows: `(last_cycle(unit_id) - cycle) + RUL_end(unit_id)` from `RUL_FD00x.txt`

Optional clipping is applied via configuration and should be recorded in result metadata.

## Leakage Controls

Required controls:
- split by `unit_id` only
- fit scalers on training split only
- apply same scaler to validation/test
- construct windows within unit boundaries only
- avoid future-looking features

Core implementation:
- `src/data/preprocessing.py`
- `src/data/validation.py`

## How To Run

### End-to-end data-path smoke validation

```bash
python scripts/smoke_test.py
```

Expected output:
- `SMOKE_TEST=PASS`
- schema and split path exercised
- train/val/test dataframe shapes
- window and tabular feature shapes

Artifacts:
- none written.

### Full baseline data path with artifact generation

```bash
python scripts/train_baselines.py --subset FD001 --rul_cap 125 --window 30 --step 1 --val_fraction 0.2 --seed 42
```

Expected output:
- leakage-safe split/scaling/window pipeline executes
- saved file paths printed

Artifacts written:
- `results/metrics/baselines_FD001.json`
- `results/tables/baseline_comparison_FD001.csv`
- `results/figures/pred_vs_true_<model>_FD001.png`
- `results/figures/error_vs_rul_<model>_FD001.png`

## Decision-Use Framing

Data pipeline quality directly affects maintenance decisions:
- split leakage inflates reported accuracy and can cause unsafe maintenance intervals.
- incorrect RUL reconstruction biases test metrics and decision thresholds.
- reproducible preprocessing ensures model drift investigations are diagnosable.

## Commit Boundaries

Not committed:
- `data/raw/`
- `data/interim/`
- `data/processed/`

Committed:
- documentation and source code defining data contracts and validation behavior.
