# Notebooks

Notebooks are used for exploratory analysis, engineering inspection of degradation behaviour, and controlled experimentation.

The goal is to keep notebooks readable and reproducible:
- exploration stays in notebooks,
- validated logic migrates into `src/`.


## Working rules

- Notebooks should import from `src/` rather than duplicating pipeline code.
- Any result shown in a notebook should be saved to `results/` (figure/table/metrics).
- Assumptions and design decisions should be written up in `reports/`.


## Notebook sequence (current files)

### `01_exploratory_data_analysis.ipynb`
- dataset parsing sanity checks
- operating regime inspection and unit trajectory plots
- sensor screening (low variance / redundant sensors)

### `02_preprocessing_and_splits.ipynb`
- leakage-safe splitting by unit
- scaling strategy (global vs regime-aware)
- windowing strategy definition (lookback length, stride)


### `03_rul_baseline_models.ipynb`
- defines baseline target construction (RUL label, optional clipping)
- establishes interpretable baselines (ridge, tree-based)
- evaluates MAE/RMSE and reliability-aware error analysis
- produces a baseline comparison table

### `04_health_index_construction.ipynb`
- builds HI baselines (engineered / unsupervised)
- evaluates HI quality (monotonicity, smoothness, correlation)
- tests threshold behaviour (early warning vs false alarms)

### `05_sequence_models.ipynb`
- introduces sequence models (LSTM/TCN) as controlled extensions
- compares against classical baselines under the same evaluation protocol
- documents stability issues (overfitting, regime sensitivity) where relevant

