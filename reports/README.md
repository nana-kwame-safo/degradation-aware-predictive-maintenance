# Reports README

`reports/` contains curated technical interpretation of experimental outputs.

## Purpose

This directory is the decision narrative layer:
- convert metrics/plots into maintenance implications
- document assumptions and limitations
- preserve a traceable argument from data to operational recommendation

## Report File Roles

- `01_problem_formulation.md`: defines maintenance objective, operational constraints, and target decision horizon.
- `02_methodology.md`: documents data pipeline, leakage controls, feature/label derivation, and model protocol.
- `03_results_and_discussion.md`: summarizes quantitative results, error diagnostics, and decision trade-offs.
- `04_deployment_note.md`: records deployment assumptions, monitoring requirements, and integration risks.

## How To Run

### Generate report inputs

```bash
python scripts/train_baselines.py --subset FD001 --rul_cap 125 --window 30 --step 1 --val_fraction 0.2 --seed 42
```

Expected terminal output:
- saved metrics/table/figures paths
- model comparison summary table

Artifacts written:
- `results/metrics/baselines_FD001.json`
- `results/tables/baseline_comparison_FD001.csv`
- `results/figures/pred_vs_true_<model>_FD001.png`
- `results/figures/error_vs_rul_<model>_FD001.png`

### Assemble report content

```bash
ls -1 reports/*.md
```

Expected terminal output:
- markdown files available for update and review.

Artifacts:
- updated report markdown content in `reports/*.md`.

Optional export, if `pandoc` is installed:

```bash
pandoc reports/03_results_and_discussion.md -o reports/03_results_and_discussion.pdf
```

Expected artifact:
- `reports/03_results_and_discussion.pdf`

## Decision-Use Framing

Report conclusions should map outputs to concrete maintenance choices:
- RUL metrics map to replacement lead-time policies and alert thresholds.
- HI trends map to condition-monitoring escalation and inspection priority.
- discordance between RUL and HI should trigger explicit uncertainty discussion.

Required decision statements:
- intervention window recommendation
- expected false alarm and missed-detection trade-off
- operational limitations for deployment scope

## Commit Boundaries

Committed:
- report markdown files
- curated report-ready tables/figures copied intentionally for long-term traceability

Not committed:
- bulk generated run artifacts that remain under `results/`
