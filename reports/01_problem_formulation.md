# 01 — Problem Formulation

## Objective
This project models progressive degradation to support reliability-focused predictive maintenance through two complementary outputs:

1) Remaining Useful Life (RUL) estimation (regression)  
2) Health Index (HI) estimation (degradation state)

The objective is not only predictive performance, but decision usefulness under maintenance constraints.

## Engineering framing
Predictive maintenance decisions are affected by:
- failure criticality and risk
- false alarm cost (unnecessary maintenance, downtime)
- missed detection cost (run-to-failure risk)
- lead-time requirements (planning, spares, scheduling)

## Dataset
NASA CMAPSS turbofan engine degradation dataset (FD001 initially; extendable to FD002–FD004).

## Success criteria (project-level)
Minimum evidence of success includes:
- credible baselines with transparent assumptions
- leakage-safe evaluation by unit
- reliability-aware error analysis (especially near end-of-life)
- HI quality metrics and threshold behaviour analysis
- clear limitations and failure modes

## Constraints and assumptions
- Unit-level independence (train/test split by unit)
- Sensors reflect degradation + operating conditions
- RUL target definition follows standard CMAPSS conventions
- Optional RUL clipping must be explicitly justified if used

## Deliverables
- baseline comparison table (classical models)
- HI construction approaches + HI quality evaluation
- sequence model extension (LSTM/TCN) with controlled comparison
- deployment note linking outputs to maintenance actions
