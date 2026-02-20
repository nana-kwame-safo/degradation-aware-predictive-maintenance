"""
Evaluation metrics for reliability-oriented RUL modelling.

This module provides compact, JSON-serializable summaries intended for:
- model comparison tables
- artifact logging
- decision-aware interpretation by RUL region
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


ArrayLike = Iterable[float]


def _as_float_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array-like input. Received shape: {arr.shape}")
    if arr.size == 0:
        raise ValueError("Metric input is empty.")
    return arr


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """
    Compute standard regression metrics for RUL estimation.

    Returns:
        JSON-serializable dictionary with MAE and RMSE.
    """
    y_t = _as_float_array(y_true)
    y_p = _as_float_array(y_pred)
    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    return {
        "MAE": float(mean_absolute_error(y_t, y_p)),
        "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
    }


def rul_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """Backward-compatible alias for historical callers."""
    return regression_metrics(y_true=y_true, y_pred=y_pred)


def stratified_metrics_by_rul_bins(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    bins: Optional[List[float]] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute MAE/RMSE by RUL bands.

    Bin interpretation for ``bins=[0,30,60,125]``:
    - ``[0,30)``
    - ``[30,60)``
    - ``[60,125]``
    - ``>125`` (included only if samples exist)

    Returns:
        Dict keyed by human-readable band label.
    """
    if bins is None:
        bins = [0.0, 30.0, 60.0, 125.0]

    y_t = _as_float_array(y_true)
    y_p = _as_float_array(y_pred)
    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    bins_arr = np.asarray(bins, dtype=float)
    if bins_arr.ndim != 1 or bins_arr.size < 2:
        raise ValueError("bins must contain at least two increasing values.")
    if not np.all(np.diff(bins_arr) > 0):
        raise ValueError(f"bins must be strictly increasing. Received: {bins}")

    out: Dict[str, Dict[str, Optional[float]]] = {}

    for i in range(len(bins_arr) - 1):
        left = float(bins_arr[i])
        right = float(bins_arr[i + 1])

        if i < len(bins_arr) - 2:
            mask = (y_t >= left) & (y_t < right)
            label = f"[{int(left)},{int(right)})"
        else:
            mask = (y_t >= left) & (y_t <= right)
            label = f"[{int(left)},{int(right)}]"

        n = int(mask.sum())
        if n == 0:
            out[label] = {"n": 0, "MAE": None, "RMSE": None}
        else:
            out[label] = {
                "n": n,
                "MAE": float(mean_absolute_error(y_t[mask], y_p[mask])),
                "RMSE": float(np.sqrt(mean_squared_error(y_t[mask], y_p[mask]))),
            }

    overflow_mask = y_t > float(bins_arr[-1])
    overflow_n = int(overflow_mask.sum())
    if overflow_n > 0:
        label = f">{int(bins_arr[-1])}"
        out[label] = {
            "n": overflow_n,
            "MAE": float(mean_absolute_error(y_t[overflow_mask], y_p[overflow_mask])),
            "RMSE": float(np.sqrt(mean_squared_error(y_t[overflow_mask], y_p[overflow_mask]))),
        }

    return out


def unit_level_error_summary(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    unit_ids: Optional[Iterable[int]] = None,
) -> Optional[Dict[str, object]]:
    """
    Summarize absolute error at unit level when unit ids are available.

    Returns ``None`` if ``unit_ids`` is not provided.
    """
    if unit_ids is None:
        return None

    y_t = _as_float_array(y_true)
    y_p = _as_float_array(y_pred)
    u = np.asarray(list(unit_ids))

    if y_t.shape[0] != y_p.shape[0] or y_t.shape[0] != u.shape[0]:
        raise ValueError("y_true, y_pred, and unit_ids must have the same length.")

    abs_err = np.abs(y_t - y_p)
    per_unit: Dict[str, float] = {}

    for uid in sorted(np.unique(u)):
        mask = u == uid
        per_unit[str(int(uid))] = float(abs_err[mask].mean())

    vals = np.asarray(list(per_unit.values()), dtype=float)
    return {
        "n_units": int(vals.size),
        "mean_unit_mae": float(vals.mean()),
        "std_unit_mae": float(vals.std(ddof=0)),
        "min_unit_mae": float(vals.min()),
        "max_unit_mae": float(vals.max()),
        "per_unit_mae": per_unit,
    }
