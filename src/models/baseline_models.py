"""
Classical baseline model factories for tabular RUL regression features.

Responsibilities:
- Provide explicit, reproducible model constructors used by the canonical
  baseline runner.
- Keep baseline hyperparameters centralized and audit-friendly.

Pipeline fit:
- Inputs are tabular feature matrices generated from window tensors
  (typically shape ``(N, F)``) and aligned targets ``(N,)``.
- Outputs are fitted estimator objects and float prediction arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge


DEFAULT_SEED = 42


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> Ridge:
    """
    Train a Ridge regression baseline.

    Args:
        X_train: Training feature matrix with shape ``(N, F)``.
        y_train: Training targets with shape ``(N,)``.
        seed: Random seed for solver reproducibility.

    Returns:
        Ridge: Fitted Ridge model.

    Raises:
        ValueError: Propagated by scikit-learn for invalid input shapes/values.
    """
    model = Ridge(
        alpha=1.0,
        solver="sag",
        random_state=seed,
        max_iter=5000,
        tol=1e-3,
    )
    model.fit(X_train, y_train)
    return model


def train_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> ElasticNet:
    """
    Train an ElasticNet regression baseline.

    Args:
        X_train: Training feature matrix with shape ``(N, F)``.
        y_train: Training targets with shape ``(N,)``.
        seed: Random seed for reproducibility.

    Returns:
        ElasticNet: Fitted ElasticNet model.

    Raises:
        ValueError: Propagated by scikit-learn for invalid input shapes/values.
    """
    model = ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=50000,
        tol=1e-3,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> RandomForestRegressor:
    """
    Train a RandomForest regression baseline.

    Args:
        X_train: Training feature matrix with shape ``(N, F)``.
        y_train: Training targets with shape ``(N,)``.
        seed: Random seed for tree/bootstrap reproducibility.

    Returns:
        RandomForestRegressor: Fitted random-forest model.

    Raises:
        ValueError: Propagated by scikit-learn for invalid input shapes/values.
    """
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    """
    Train an XGBoost regression baseline.

    Args:
        X_train: Training feature matrix with shape ``(N, F)``.
        y_train: Training targets with shape ``(N,)``.
        seed: Random seed for reproducibility.

    Returns:
        Any: Fitted ``xgboost.XGBRegressor`` instance.

    Raises:
        ImportError: If XGBoost is not installed.
        ValueError: Propagated by estimator fitting for invalid inputs.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is not installed.") from exc

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    """
    Train a LightGBM regression baseline.

    Args:
        X_train: Training feature matrix with shape ``(N, F)``.
        y_train: Training targets with shape ``(N,)``.
        seed: Random seed for reproducibility.

    Returns:
        Any: Fitted ``lightgbm.LGBMRegressor`` instance.

    Raises:
        ImportError: If LightGBM is not installed.
        ValueError: Propagated by estimator fitting for invalid inputs.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("lightgbm is not installed.") from exc

    model = LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Generate float predictions from a fitted baseline model.

    Args:
        model: Fitted estimator exposing ``predict``.
        X: Feature matrix with shape ``(N, F)``.

    Returns:
        np.ndarray: Float predictions with shape ``(N,)``.

    Raises:
        AttributeError: If model has no ``predict`` method.
        ValueError: If estimator prediction output cannot be cast to float.
    """
    return np.asarray(model.predict(X), dtype=float)
