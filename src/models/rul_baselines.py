"""
Compatibility baseline helpers for early milestone code paths.

This module preserves the original ``train_rul_baseline`` interface while
delegating implementation to the canonical factories in
``src.models.baseline_models``.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from .baseline_models import predict, train_random_forest, train_ridge


def train_rul_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "ridge",
    random_state: int = 42,
) -> Tuple[object, np.ndarray]:
    """
    Train a legacy baseline regressor and return in-sample predictions.

    Args:
        X_train: Training feature matrix with shape ``(N, F)``.
        y_train: Training targets with shape ``(N,)``.
        model_name: Baseline selector. Supported values: ``"ridge"``, ``"rf"``.
        random_state: Reproducibility seed for supported models.

    Returns:
        Tuple[object, np.ndarray]: ``(model, y_pred_train)``.

    Raises:
        ValueError: If ``model_name`` is unsupported.
    """
    warnings.warn(
        "train_rul_baseline() is a compatibility API. "
        "Prefer src.models.baseline_models factories instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    name = model_name.lower().strip()
    if name == "ridge":
        model = train_ridge(X_train, y_train, seed=random_state)
    elif name == "rf":
        model = train_random_forest(X_train, y_train, seed=random_state)
    else:
        raise ValueError(f"Unknown baseline model '{model_name}'. Use 'ridge' or 'rf'.")

    return model, predict(model, X_train)
