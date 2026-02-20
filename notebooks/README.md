# Notebooks README

Notebooks are used for staged analysis before logic is stabilized in `src/`.

## Notebook Scope and Expected Outputs

1. `01_exploratory_data_analysis.ipynb`
- Purpose: verify CMAPSS schema, operating-regime behavior, and sensor stability patterns.
- Expected artifacts: exploratory plots in `results/figures/` and summary notes for feature assumptions.

2. `02_preprocessing_and_splits.ipynb`
- Purpose: demonstrate leakage-safe unit splits, scaling policy, and target construction checks.
- Expected artifacts: split diagnostics and preprocessing summary tables in `results/tables/`.

3. `03_rul_baseline_models.ipynb`
- Purpose: train/evaluate baseline RUL models on tabular window features.
- Expected artifacts: baseline metric outputs in `results/metrics/` and comparison figures/tables.

4. `04_health_index_construction.ipynb`
- Purpose: derive and evaluate HI trajectories against degradation progression assumptions.
- Expected artifacts: HI trend plots and HI summary metrics for report integration.

5. `05_sequence_models.ipynb`
- Purpose: evaluate sequence-model extensions after baseline validation.
- Expected artifacts: sequence-model comparison metrics and figures in `results/`.

## Required Notebook Header

Start every notebook with a markdown cell containing:
- purpose
- expected outputs
- assumptions and constraints

## How To Run

### Pre-run validation

```bash
python src/utils/env_check.py
python scripts/smoke_test.py
```

Expected terminal output:
- environment checks pass
- `SMOKE_TEST=PASS` with shape diagnostics

Artifacts:
- none written by these checks.

### Launch notebook environment

```bash
jupyter lab
```

Expected output:
- local Jupyter server URL in terminal.

Artifacts:
- notebook execution state in-memory
- optional autosave updates to `.ipynb` files

### Generate reproducible artifacts from notebook workflows

Notebook cells should write any persistent outputs to:
- `results/metrics/`
- `results/tables/`
- `results/figures/`

Expected artifacts:
- explicit files referenced in notebook markdown and report links.

## Notebook Engineering Rules

- import reusable logic from `src/`, do not duplicate pipeline code in cells
- keep unit-based splitting invariant
- fit transforms on train only
- avoid hidden state dependence; rerun from top before committing
- migrate stable notebook logic into `src/`

## Decision-Use Framing

Notebook analysis should answer maintenance-relevant questions:
- RUL: what intervention lead time is achievable under current error bands?
- HI: does the degradation trend provide early warning ahead of low-RUL regimes?
- Combined: where do RUL and HI agree or disagree on intervention urgency?

## Commit Boundaries

Committed:
- notebook files and documentation

Not committed:
- raw/interim/processed datasets
- generated result artifacts not explicitly curated

Clear oversized or non-essential cell outputs before commit.
