# 03 — Results Summary

This report summarises experimental results for both project outputs:

1) **Remaining Useful Life (RUL) estimation** (regression)  
2) **Health Index (HI) estimation** (trend-based degradation state)

Results are presented with emphasis on **reliability-aware interpretation** and practical maintenance decision relevance.

---

## 3.1 Experimental scope

### Dataset subset(s)
- Subset(s) evaluated: FD00_
- Units used for training: ___
- Units used for validation: ___
- Units used for test: ___

### Leakage control confirmation
- Split performed by **unit**: Yes / No  
- Scaling fitted on training only: Yes / No  
- Windowing performed after split: Yes / No  

### Core modelling tasks
- RUL regression: Yes / No  
- HI estimation: Yes / No  

---

## 3.2 Baseline results — RUL regression

### Models compared (interpretable baselines)
| Model | Feature set | MAE | RMSE | Notes |
|------|-------------|-----|------|------|
| Linear (Ridge/Lasso) | Baseline features |  |  |  |
| Random Forest | Baseline features |  |  |  |
| Gradient Boosting (XGBoost/LightGBM) | Baseline features |  |  |  |

### Reliability-aware error analysis
The following analyses are required to interpret RUL performance in decision terms:

- **Error distribution across units**  
  Are errors consistent, or dominated by a subset of units?

- **Error vs time-to-failure (end-of-life sensitivity)**  
  Do errors increase near failure where maintenance decisions are most sensitive?

- **Over-prediction vs under-prediction behaviour**  
  Under-prediction can trigger early maintenance (costly).  
  Over-prediction can cause late intervention (riskier).  
  Summarise directional bias:

| Metric | Value |
|--------|-------|
| Mean error (signed) |   |
| % over-predictions |   |
| % under-predictions |   |

### Key findings (RUL)
Write 3–6 bullet points that answer:
- Which baseline is strongest and why?
- What failure modes appear (regime shift, unstable sensors, noise)?
- Where does the model struggle (early life vs late life)?

### 3.2.1 Placeholder interpretation template (RUL bands + EOL behaviour)
Use this template once `scripts/train_baselines.py` artifacts are available:

- **RUL band [0,30):**  
  MAE/RMSE = ___ / ___; interpretation of near-failure behavior: ___

- **RUL band [30,60):**  
  MAE/RMSE = ___ / ___; interpretation of transition-to-failure behavior: ___

- **RUL band [60,125]:**  
  MAE/RMSE = ___ / ___; interpretation of early-to-mid life behavior: ___

- **End-of-life (EOL) judgement:**  
  Is model error acceptable for maintenance triggering near failure? Yes/No and why.

- **Operational implication:**  
  Does the model bias toward over-prediction (late risk) or under-prediction (early maintenance cost)?

---

## 3.3 Health Index (HI) estimation results

HI is evaluated as a **degradation state representation** rather than a direct prediction target.

### HI construction methods compared
| Method | Description | Notes |
|--------|-------------|------|
| Engineered HI (normalised degradation proxy) |  |  |
| Unsupervised HI (e.g., PCA/AE/1D embedding) (optional) |  |  |

### HI quality metrics
For each method, report:

| Method | Monotonicity ↑ | Smoothness ↑ | Correlation with TTF ↑ | Notes |
|--------|-----------------|--------------|--------------------------|------|
| Engineered HI |  |  |  |  |
| Unsupervised HI |  |  |  |  |

### Visual inspection (required)
Include representative HI plots for multiple units:
- stable degradation trend
- noisy/ambiguous trend
- failure-case example

Summarise whether the HI is:
- physically plausible
- stable enough for thresholding
- consistent across operating conditions

### Key findings (HI)
Write 3–6 bullet points that answer:
- Which HI is most decision-useful?
- Where does HI break down?
- Are there regime effects or sensor anomalies?

---

## 3.4 RUL vs HI comparison (decision framing)

This section compares the two outputs in maintenance decision terms.

### Decision suitability comparison
| Criterion | RUL (direct) | HI (trend-based) | Notes |
|----------|--------------|------------------|------|
| Interpretability |  |  |  |
| Stability across regimes |  |  |  |
| Sensitivity near end-of-life |  |  |  |
| Ease of thresholding |  |  |  |
| Useful lead-time for planning |  |  |  |

### Practical interpretation
Discuss:
- When HI is a safer monitoring tool than RUL
- When direct RUL is more actionable than HI
- Whether a combined strategy improves reliability decision-making

---

## 3.5 Sequence model results (extension, once implemented)

This section should be filled after sequence models are introduced.

### Models
- LSTM: Yes / No
- TCN: Yes / No

### Comparison against baselines
| Model | MAE | RMSE | Notes |
|------|-----|------|------|
| Best classical baseline |  |  |  |
| LSTM |  |  |  |
| TCN |  |  |  |

Discuss:
- performance gains vs increased complexity
- stability and overfitting behaviour
- interpretability limitations
- deployment implications (compute, monitoring)

---

## 3.6 Limitations and failure modes

This project is benchmark-driven. Record limitations clearly:

- Generalisation limits (FD001 vs multi-regime subsets)
- Sensitivity to sensor drift/noise
- Dependence on preprocessing assumptions (scaling, clipping)
- HI ambiguity (multiple plausible degradation mappings)
- Differences from real industrial data (missingness, maintenance actions, partial failures)

---

## 3.7 Outputs saved

List key saved artifacts (paths), for traceability:

- Metrics: `results/metrics/...`
- Tables: `results/tables/...`
- Figures: `results/figures/...`

Curated outputs used in this report should be copied into `reports/figures/` or embedded as links.
