# Results

This folder contains **run artifacts** produced by the pipeline (figures, tables, metrics). Outputs are written to disk to support reproducibility and consistent comparison across baselines.

By default, these artifacts are **not committed to Git** to keep the repository reviewable and to avoid committing large generated files.


## Structure

- `results/figures/` stores visual outputs (degradation trajectories, residual plots, calibration plots)
- `results/tables/` stores comparison tables (baseline ranking, ablation summaries)
- `results/metrics/` stores metric dumps (JSON/CSV per experiment)


## What is recorded

### RUL experiments
Typical artifacts:
- overall MAE/RMSE
- unit-level error distribution
- error vs time-to-failure (end-of-life region)
- optional asymmetric cost summary (late vs early prediction impact)

### HI experiments
Typical artifacts:
- HI traces per unit (qualitative inspection)
- HI quality metrics:
  - monotonicity
  - smoothness
  - correlation with time-to-failure
- threshold policy simulations (false alarm vs missed detection trade-off)


## Naming conventions

Use names that make comparisons traceable:

- Include dataset subset: `FD001`, `FD002`, ...
- Include task: `rul` or `hi`
- Include model family: `ridge`, `rf`, `xgb`, `lstm`, `tcn`, ...
- Include run id or date if needed

Examples:
- `results/metrics/FD001_rul_ridge.json`
- `results/tables/FD001_baseline_comparison.csv`
- `results/figures/FD001_hi_monotonicity.png`


## What to version control

Run artifacts remain in `results/` (ignored).

Curated outputs intended for review should be copied into `reports/`:
- 3–6 key figures
- 1–2 final comparison tables
- brief interpretation paragraphs linked to decisions and limitations
