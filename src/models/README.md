# Models README

Canonical baseline API lives in `src/models/baseline_models.py`.

Use these factories and helpers:
- `train_ridge`
- `train_elasticnet`
- `train_random_forest`
- `train_xgboost`
- `train_lightgbm`
- `predict`

Canonical baseline pipeline entrypoint:
- `python -m src.run_baseline`

Compatibility wrapper:
- `python scripts/train_baselines.py` (delegates to `src.run_baseline.main()` only)
