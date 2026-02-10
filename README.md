# Degradation-Aware Predictive Maintenance
This project applies reliability-focused machine learning to predictive maintenance by modelling **progressive degradation** in mechanical systems. It delivers two complementary outputs:

1) **Remaining Useful Life (RUL) estimation** (regression)  
2) **Health Index (HI) estimation** (trend-based degradation state)

The emphasis is on **interpretable baselines first**, followed by sequence models, with evaluation framed in engineering and maintenance decision terms.

## Why this project
Predictive maintenance is a reliability problem under uncertainty:
- Failures are rare and costly (risk + downtime)
- False alarms create operational disruption
- Predictions must be interpreted with engineering judgement
- Models must generalise across operating regimes and units

This repository is designed as an industrial-style portfolio project: structured code, reproducible results, and engineering interpretation.

## Dataset
Primary dataset: **NASA CMAPSS Turbofan Engine Degradation**
- Multi-sensor time-series recorded across operating cycles
- Run-to-failure trajectories for multiple units
- Standard benchmarks for RUL and degradation modelling

The dataset is not stored in the repository. See `data/README.md` for setup instructions.

## Project outputs
### A) RUL regression
- Predict remaining cycles to failure
- Compare interpretable models (linear/trees) and later sequence models
- Evaluate using MAE/RMSE and reliability-aware error analysis

### B) Health Index estimation
- Construct a degradation state signal per unit
- Evaluate HI quality via monotonicity, smoothness, and correlation with time-to-failure
- Compare HI-driven decision thresholds vs direct RUL predictions

## Approach (high level)
1. Data ingestion + validation (units, cycles, sensors, operating conditions)
2. Preprocessing (scaling, regime handling, windowing)
3. Feature engineering (rolling stats, trend features)
4. Baseline modelling (interpretable RUL + HI baselines)
5. Reliability-aware evaluation and interpretation
6. Sequence models (LSTM/TCN) as an extension and performance comparison

## Repository structure
- `notebooks/` exploratory analysis and experiments
- `src/` reusable pipeline code (data, features, models, evaluation)
- `results/` figures, tables, and metrics
- `reports/` written technical summaries and interpretation

## Results
All experiment outputs will be tracked in:
- `results/figures/`
- `results/tables/`
- `results/metrics/`

## Practical deployment notes
See `reports/03_deployment_note.md` for how HI and RUL outputs can be used in maintenance planning workflows, including threshold policies and operational considerations.

## Roadmap
- [ ] Data ingestion + preprocessing pipeline
- [ ] Interpretable RUL baselines + evaluation
- [ ] Health Index construction + quality metrics
- [ ] RUL vs HI comparison (decision-focused)
- [ ] Sequence models (LSTM/TCN) + comparison to baselines
- [ ] Final technical report and deployment note
