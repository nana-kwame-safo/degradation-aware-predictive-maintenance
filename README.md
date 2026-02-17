# Degradation-Aware Predictive Maintenance

This project applies reliability-focused machine learning to predictive maintenance by modeling **progressive degradation** in mechanical systems. It delivers two complementary outputs:

1) **Remaining Useful Life (RUL) estimation** (regression)  
2) **Health Index (HI) estimation** (trend-based degradation state)

The emphasis is on **interpretable baselines first**, followed by sequence models, with evaluation framed in engineering and maintenance decision terms.  
The project is developed with an emphasis on **reliability engineering principles, practical maintenance decision-making, and model interpretability**.


## Why this project
Predictive maintenance is a reliability problem under uncertainty:
- Failures are rare and costly (risk + downtime)
- False alarms create operational disruption
- Predictions must be interpreted with engineering judgement
- Models must generalise across operating regimes and units

This repository is designed as an industrial-style portfolio project, prioritising structured code, reproducible results, and engineering interpretation over purely algorithmic performance.



## Dataset
Primary dataset: **NASA CMAPSS Turbofan Engine Degradation**
- Multi-sensor time-series recorded across operating cycles
- Run-to-failure trajectories for multiple units
- Standard benchmark for RUL and degradation modelling

The dataset is not stored in the repository. See `data/README.md` for setup instructions.

---

## Setup

### 1) Clone the repository
```bash
git clone https://github.com/nana-kwame-safo/degradation-aware-predictive-maintenance.git
cd degradation-aware-predictive-maintenance
```

### 2) Create the environment (Conda)
```bash
conda env create -f environment.yml
conda activate degradation-maintenance
```

### 3) Verify the environment
```bash
python src/utils/env_check.py
```

### 4) Data setup
Download the CMAPSS dataset and place files under:
- `data/raw/`

See `data/README.md` for details.

---

## Project outputs

### A) Remaining Useful Life (RUL) regression
- Predict remaining cycles to failure
- Compare interpretable models (linear and tree-based) and later sequence models
- Evaluate using MAE/RMSE and reliability-aware error analysis, with attention to behaviour near end-of-life

### B) Health Index estimation
- Construct a degradation state signal per unit
- Evaluate HI quality using monotonicity, smoothness, and correlation with time-to-failure
- Compare HI-driven decision thresholds against direct RUL predictions


## Approach (engineering-led, high level)
1. Data ingestion and validation (units, cycles, sensors, operating conditions)
2. Preprocessing (scaling, regime handling, windowing)
3. Feature engineering (rolling statistics, trend-sensitive features)
4. Baseline modelling (interpretable RUL and HI approaches)
5. Reliability-aware evaluation and interpretation
6. Sequence models (LSTM/TCN) as an extension and performance comparison



## Repository structure
- `data/` dataset layout and setup instructions (raw/processed not committed)
- `notebooks/` exploratory analysis and experiments
- `src/` reusable pipeline code (data, features, models, evaluation)
- `results/` run artifacts (figures, tables, metrics; not committed by default)
- `reports/` technical reports, results interpretation, and deployment notes (portfolio-ready)

## Results
Experiment outputs are written to:
- `results/figures/`
- `results/tables/`
- `results/metrics/`

Curated figures and final tables intended for review can be placed under `reports/` for version control.


## Practical deployment notes
See `reports/04_deployment_note.md` for how HI and RUL outputs can be translated into maintenance planning workflows, including threshold policies, lead-time considerations, and operational trade-offs.


## Roadmap
- [ ] Data ingestion and preprocessing pipeline
- [ ] Interpretable RUL baselines and evaluation
- [ ] Health Index construction and quality metrics
- [ ] RUL vs HI comparison (decision-focused)
- [ ] Sequence models (LSTM/TCN) and comparison to baselines
- [ ] Final technical report and deployment note
