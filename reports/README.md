# Reports

This folder contains written technical summaries intended to support both:
- an industrial-style portfolio narrative, and
- an MSc dissertation-style write-up.

The reporting structure is designed to connect modelling outputs to reliability engineering interpretation and maintenance decision-making.


## Planned documents

### `01_problem_formulation.md`
- problem definition and scope (RUL + HI)
- dataset selection and assumptions
- what counts as success (engineering criteria)
- leakage controls and evaluation framing

### `02_methodology.md`
- preprocessing design decisions
- feature engineering choices (and why)
- baseline model justification (interpretability-first)
- sequence model extension plan

### `03_results_summary.md`
- baseline comparisons (RUL and HI)
- error analysis with attention to end-of-life behaviour
- robustness checks (across units/regimes)
- limitations and failure modes

### `04_deployment_note.md`
- translating RUL/HI into maintenance actions
- threshold policies, lead-time considerations
- false alarm vs missed detection trade-offs
- monitoring and retraining considerations


## What goes in reports vs results

- `results/` contains run artifacts (ignored by Git)
- `reports/` contains curated figures/tables and written interpretation (tracked)

Expected outcome to be considered: 
- a small set of curated figures/tables
- clear interpretation and limitations
- an explicit deployment note linking outputs to decisions
