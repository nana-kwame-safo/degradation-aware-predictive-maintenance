# 04 — Deployment Note (Maintenance Decision Use)

## Intended operational use
Two outputs map to two decision modes:

- **RUL** supports scheduling (when to intervene)
- **HI** supports monitoring (is degradation accelerating / stable?)

## Threshold policies (HI)
Define:
- alert threshold
- warning threshold (optional)
- persistence rule (avoid reacting to noise)

Discuss:
- false alarms vs late warnings
- lead time requirements
- calibration needs (site-specific behaviour)

## Planning policies (RUL)
Discuss:
- minimum actionable lead time
- how to treat uncertainty (safety margin)
- consequence of underestimation vs overestimation

## Monitoring and retraining
Include:
- drift monitoring (sensor distribution changes)
- retraining triggers
- revalidation requirements (unit-level evaluation)

## Practical limitations
State clearly:
- CMAPSS is a benchmark; real assets may differ
- sensor availability and quality constraints
- maintenance records and failure labels may be imperfect
