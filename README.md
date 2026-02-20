# Degradation-Aware Predictive Maintenance

Reliability-focused predictive maintenance workflow using NASA CMAPSS turbofan degradation data.

Primary technical outputs:
1. Remaining Useful Life (RUL) estimates.
2. Health Index (HI) trajectories.

Engineering constraints:
- unit-level leakage control
- train-only fit for preprocessing transforms
- reproducible artifact generation under `results/`

## Quickstart

```bash
git clone https://github.com/nana-kwame-safo/degradation-aware-predictive-maintenance.git
cd degradation-aware-predictive-maintenance
conda env create -f environment.yml
conda activate degradation-maintenance
python src/utils/env_check.py
```

Place CMAPSS files under `data/raw/cmapss/`:
- `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
- `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt`
- `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`

```bash
python scripts/smoke_test.py --subset FD001
python -m src.run_baseline --subset FD001 --window 30 --val_fraction 0.2 --seed 42 --rul_cap 125
```

Expected Quickstart behavior:
- environment check runs without exceptions
- smoke test prints dataset/window/tabular shapes and exits `0`
- baseline runner writes metrics/table/figures under `results/`

If you already have `train_*.txt` and `test_*.txt` but are missing `RUL_*.txt`:
1. Re-download the matching subset package from the official NASA CMAPSS source.
2. Ensure subset names match exactly (`FD001`/`FD002`/`FD003`/`FD004`).
3. Confirm `RUL_FD00x.txt` exists before running evaluation.

Without `RUL_FD00x.txt`, labeled test RUL cannot be constructed and evaluation will fail.

See `data/README.md` for schema and validation details.

## How To Run

### Import and environment sanity

```bash
python scripts/check_imports.py
```

Expected output:
- `[PASS]` lines for core imports and a final success line.

Artifacts:
- none written.

### End-to-end smoke test (no heavy training)

```bash
python scripts/smoke_test.py --subset FD001
```

Expected output:
- `SMOKE_TEST=PASS`
- train/validation/test dataframe shapes
- window tensor shapes for train/val/test
- tabular feature shapes and `feature_count=126`

Artifacts:
- none written.

### Baseline training run

```bash
python -m src.run_baseline --subset FD001 --window 30 --val_fraction 0.2 --seed 42 --rul_cap 125
```

Expected output:
- saved metrics/table/figure paths printed in terminal
- model comparison table printed to stdout

Artifacts written:
- `results/metrics/baselines_FD001.json`
- `results/tables/baseline_comparison_FD001.csv`
- `results/figures/pred_vs_true_<model>_FD001.png`
- `results/figures/error_vs_rul_<model>_FD001.png`

## Decision-Use Framing

RUL output supports lead-time decisions:
- map predicted RUL bands to inspection, defer, or planned replacement actions.
- apply tighter review thresholds near end-of-life bands.

HI output supports trend decisions:
- detect monotonic degradation slope shifts.
- trigger deeper diagnostics when HI decline accelerates despite acceptable short-term RUL.

Combined use:
- RUL prioritizes timing of intervention.
- HI supports confidence in degradation direction and escalation urgency.

## Commit Boundaries

Committed:
- source code, scripts, notebooks, and documentation.

Not committed:
- dataset files in `data/raw/`, `data/interim/`, and `data/processed/`.
- generated experiment artifacts in `results/`.
- model binaries/checkpoints and transient logs.
