"""
Model package exports for baseline tabular regressors.

This namespace exposes canonical training/prediction helpers used by
``src.run_baseline`` and downstream evaluation scripts.
"""

from .baseline_models import (
    predict,
    train_elasticnet,
    train_lightgbm,
    train_random_forest,
    train_ridge,
    train_xgboost,
)

__all__ = [
    "train_ridge",
    "train_elasticnet",
    "train_random_forest",
    "train_xgboost",
    "train_lightgbm",
    "predict",
]
