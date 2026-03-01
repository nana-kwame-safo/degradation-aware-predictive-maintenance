"""
Metric computation utilities for reliability-oriented RUL model evaluation.

Responsibilities:
- Compute global regression accuracy metrics.
- Produce stratified diagnostics across RUL operating regions.
- Quantify directional/asymmetric error behavior for maintenance risk.
- Aggregate per-unit and summary statistics for operational reporting.

Pipeline fit:
- Consumes aligned ``y_true``/``y_pred`` arrays (shape ``(N,)``) and optional
  ``unit_ids``.
- Produces JSON-serializable dictionaries and table-friendly row structures.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.arrays import as_float_1d, validate_same_length


ArrayLike = Iterable[float]


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """
    Compute overall MAE and RMSE.

    Args:
        y_true: Ground-truth RUL values, shape ``(N,)``.
        y_pred: Predicted RUL values, shape ``(N,)``.

    Returns:
        Dict[str, float]: ``{"MAE": ..., "RMSE": ...}``.

    Raises:
        ValueError: If inputs are not non-empty 1D arrays with matching length.
    """
    y_t = as_float_1d(y_true, "y_true")
    y_p = as_float_1d(y_pred, "y_pred")
    validate_same_length(y_t, y_p)

    return {
        "MAE": float(mean_absolute_error(y_t, y_p)),
        "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
    }


def rul_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """
    Backward-compatible alias for :func:`regression_metrics`.

    Args:
        y_true: Ground-truth RUL values, shape ``(N,)``.
        y_pred: Predicted RUL values, shape ``(N,)``.

    Returns:
        Dict[str, float]: ``{"MAE": ..., "RMSE": ...}``.

    Raises:
        ValueError: Propagated from :func:`regression_metrics`.
    """
    return regression_metrics(y_true=y_true, y_pred=y_pred)


def stratified_metrics_by_rul_bins(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    bins: Optional[List[float]] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute MAE/RMSE by sequential RUL bin edges.

    Args:
        y_true: Ground-truth RUL values, shape ``(N,)``.
        y_pred: Predicted RUL values, shape ``(N,)``.
        bins: Strictly increasing bin edges. Default is ``[0, 30, 60, 125]``.

    Returns:
        Dict[str, Dict[str, Optional[float]]]: Mapping from bin label to
        ``{"n", "MAE", "RMSE"}``. Includes overflow bin ``">max_edge"`` when present.

    Raises:
        ValueError: If input arrays are invalid/misaligned or bin definitions are invalid.

    Assumptions & leakage boundaries:
        - Stratification is purely evaluation-time slicing.
        - No model fitting or threshold tuning is performed here.
    """
    if bins is None:
        bins = [0.0, 30.0, 60.0, 125.0]

    y_t = as_float_1d(y_true, "y_true")
    y_p = as_float_1d(y_pred, "y_pred")
    validate_same_length(y_t, y_p)

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
            "RMSE": float(
                np.sqrt(mean_squared_error(y_t[overflow_mask], y_p[overflow_mask]))
            ),
        }

    return out


def stratified_metrics_by_rul_bands(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    bands: Sequence[Tuple[int, int]],
) -> List[Dict[str, Optional[float]]]:
    """
    Compute MAE/RMSE for explicit inclusive RUL bands.

    Args:
        y_true: Ground-truth RUL values, shape ``(N,)``.
        y_pred: Predicted RUL values, shape ``(N,)``.
        bands: Sequence of inclusive ``(low, high)`` tuples.

    Returns:
        List[Dict[str, Optional[float]]]: One row per band with keys
        ``band``, ``band_low``, ``band_high``, ``n``, ``MAE``, ``RMSE``.

    Raises:
        ValueError: If arrays are invalid/misaligned, band list is empty, or
            any band bounds are invalid.
    """
    y_t = as_float_1d(y_true, "y_true")
    y_p = as_float_1d(y_pred, "y_pred")
    validate_same_length(y_t, y_p)

    if not bands:
        raise ValueError("bands must not be empty.")

    rows: List[Dict[str, Optional[float]]] = []
    for low, high in bands:
        low_i = int(low)
        high_i = int(high)
        if low_i < 0:
            raise ValueError(f"Band lower bound must be >= 0. Received: {low_i}")
        if high_i < low_i:
            raise ValueError(
                f"Band upper bound must be >= lower bound. Received: ({low_i}, {high_i})"
            )

        mask = (y_t >= float(low_i)) & (y_t <= float(high_i))
        n = int(mask.sum())

        row: Dict[str, Optional[float]] = {
            "band": f"{low_i}-{high_i}",
            "band_low": low_i,
            "band_high": high_i,
            "n": n,
            "MAE": None,
            "RMSE": None,
        }
        if n > 0:
            row["MAE"] = float(mean_absolute_error(y_t[mask], y_p[mask]))
            row["RMSE"] = float(np.sqrt(mean_squared_error(y_t[mask], y_p[mask])))

        rows.append(row)

    return rows


def error_asymmetry_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    eol_band: Tuple[int, int] = (0, 20),
    severe_threshold: float = 10.0,
) -> Dict[str, Optional[float]]:
    """
    Compute directional error diagnostics relevant to maintenance risk.

    Args:
        y_true: Ground-truth RUL values, shape ``(N,)``.
        y_pred: Predicted RUL values, shape ``(N,)``.
        eol_band: Inclusive EOL band ``(low, high)`` on true RUL.
        severe_threshold: Severe overestimation cutoff on ``pred - true``.

    Returns:
        Dict[str, Optional[float]]: Directional-bias and overestimation metrics,
        including sample counts and EOL-band metadata.

    Raises:
        ValueError: If arrays are invalid/misaligned, band bounds are invalid,
            or ``severe_threshold <= 0``.

    Assumptions & leakage boundaries:
        - Error is defined as ``pred - true``.
        - Metrics are observational diagnostics; no decision threshold is fit.
    """
    y_t = as_float_1d(y_true, "y_true")
    y_p = as_float_1d(y_pred, "y_pred")
    validate_same_length(y_t, y_p)

    low, high = int(eol_band[0]), int(eol_band[1])
    if low < 0 or high < low:
        raise ValueError(f"Invalid eol_band: {eol_band}")
    if severe_threshold <= 0:
        raise ValueError(f"severe_threshold must be > 0. Received: {severe_threshold}")

    err = y_p - y_t
    eol_mask = (y_t >= float(low)) & (y_t <= float(high))
    n_eol = int(eol_mask.sum())

    out: Dict[str, Optional[float]] = {
        "bias_overall": float(err.mean()),
        "bias_eol": None,
        "pct_overestimation_eol": None,
        "pct_severe_overestimation_eol": None,
        "n_total": int(y_t.shape[0]),
        "n_eol": n_eol,
        "eol_low": low,
        "eol_high": high,
        "severe_overestimate_threshold": float(severe_threshold),
    }

    if n_eol > 0:
        eol_err = err[eol_mask]
        out["bias_eol"] = float(eol_err.mean())
        out["pct_overestimation_eol"] = float(100.0 * np.mean(eol_err > 0.0))
        out["pct_severe_overestimation_eol"] = float(
            100.0 * np.mean(eol_err >= float(severe_threshold))
        )

    return out


def unit_level_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    unit_ids: Iterable[int],
) -> pd.DataFrame:
    """
    Aggregate prediction errors by ``unit_id``.

    Args:
        y_true: Ground-truth RUL values, shape ``(N,)``.
        y_pred: Predicted RUL values, shape ``(N,)``.
        unit_ids: Unit identifiers aligned to samples, shape ``(N,)``.

    Returns:
        pd.DataFrame: One row per unit with columns
        ``unit_id``, ``n_samples``, ``MAE``, ``RMSE``, ``bias``.

    Raises:
        ValueError: If arrays are invalid/misaligned or the output would be empty.
    """
    y_t = as_float_1d(y_true, "y_true")
    y_p = as_float_1d(y_pred, "y_pred")
    units = np.asarray(list(unit_ids), dtype=int)
    if units.ndim != 1:
        raise ValueError(f"unit_ids must be 1D. Received shape: {units.shape}")

    validate_same_length(y_t, y_p, units)

    rows: List[Dict[str, float]] = []
    for uid in sorted(np.unique(units)):
        mask = units == uid
        y_t_u = y_t[mask]
        y_p_u = y_p[mask]
        err_u = y_p_u - y_t_u
        rows.append(
            {
                "unit_id": int(uid),
                "n_samples": int(mask.sum()),
                "MAE": float(mean_absolute_error(y_t_u, y_p_u)),
                "RMSE": float(np.sqrt(mean_squared_error(y_t_u, y_p_u))),
                "bias": float(err_u.mean()),
            }
        )

    out = pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)
    if out.empty:
        raise ValueError("unit_level_metrics produced an empty table.")

    return out


def summarize_unit_metrics(
    unit_df: pd.DataFrame,
    worst_fraction: float = 0.10,
) -> Dict[str, object]:
    """
    Summarize per-unit metrics and identify worst-performing units by MAE.

    Args:
        unit_df: Per-unit dataframe from :func:`unit_level_metrics`.
        worst_fraction: Fraction of worst-MAE units to summarize in ``(0, 1]``.

    Returns:
        Dict[str, object]: Aggregate statistics and worst-unit identifiers.

    Raises:
        ValueError: If required columns are missing, dataframe is empty, or
            ``worst_fraction`` is out of range.
    """
    required = {"unit_id", "MAE", "RMSE"}
    missing = sorted(required.difference(unit_df.columns))
    if missing:
        raise ValueError(f"summarize_unit_metrics missing columns: {missing}")
    if unit_df.empty:
        raise ValueError("summarize_unit_metrics received an empty dataframe.")
    if not (0.0 < float(worst_fraction) <= 1.0):
        raise ValueError(f"worst_fraction must be in (0,1]. Received: {worst_fraction}")

    mae = unit_df["MAE"].to_numpy(dtype=float)
    rmse = unit_df["RMSE"].to_numpy(dtype=float)
    n_units = int(unit_df.shape[0])

    n_worst = max(1, int(np.ceil(n_units * float(worst_fraction))))
    worst = unit_df.sort_values("MAE", ascending=False).head(n_worst)

    return {
        "n_units": n_units,
        "mae_mean": float(np.mean(mae)),
        "mae_median": float(np.median(mae)),
        "mae_std": float(np.std(mae, ddof=0)),
        "mae_p90": float(np.percentile(mae, 90)),
        "rmse_mean": float(np.mean(rmse)),
        "rmse_median": float(np.median(rmse)),
        "rmse_std": float(np.std(rmse, ddof=0)),
        "rmse_p90": float(np.percentile(rmse, 90)),
        "worst_10pct_unit_count": int(n_worst),
        "worst_10pct_mean_mae": float(worst["MAE"].mean()),
        "worst_10pct_unit_ids": [int(v) for v in worst["unit_id"].tolist()],
    }


def unit_level_error_summary(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    unit_ids: Optional[Iterable[int]] = None,
) -> Optional[Dict[str, object]]:
    """
    Backward-compatible compact summary wrapper for unit-level diagnostics.

    Args:
        y_true: Ground-truth RUL values, shape ``(N,)``.
        y_pred: Predicted RUL values, shape ``(N,)``.
        unit_ids: Optional unit identifiers, shape ``(N,)``.

    Returns:
        Optional[Dict[str, object]]: ``None`` if ``unit_ids`` is ``None``;
        otherwise compact summary with MAE distribution and per-unit mapping.

    Raises:
        ValueError: Propagated from :func:`unit_level_metrics` for invalid inputs.
    """
    if unit_ids is None:
        return None

    unit_df = unit_level_metrics(y_true=y_true, y_pred=y_pred, unit_ids=unit_ids)
    vals = unit_df["MAE"].to_numpy(dtype=float)
    per_unit = {
        str(int(row.unit_id)): float(row.MAE) for row in unit_df.itertuples(index=False)
    }
    return {
        "n_units": int(vals.size),
        "mean_unit_mae": float(vals.mean()),
        "std_unit_mae": float(vals.std(ddof=0)),
        "min_unit_mae": float(vals.min()),
        "max_unit_mae": float(vals.max()),
        "per_unit_mae": per_unit,
    }
