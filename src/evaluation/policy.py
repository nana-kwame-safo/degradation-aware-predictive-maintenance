"""
Policy-oriented evaluation for threshold-based maintenance triggering.

Responsibilities:
- Convert per-window RUL predictions into per-unit trigger outcomes.
- Quantify trigger quality using operationally meaningful rates and lead-time stats.

Pipeline fit:
- Inputs are aligned model outputs and metadata from ``src.run_baseline``:
  ``meta_df`` with ``unit_id``, ``cycle_end``, ``true_rul`` plus ``y_pred``.
- Outputs are:
  1) a per-unit trigger outcome dataframe
  2) aggregate policy metrics for reporting and threshold comparison.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.arrays import as_float_1d


ArrayLike = Iterable[float]


def _validate_meta(meta_df: pd.DataFrame) -> None:
    required = {"unit_id", "cycle_end", "true_rul"}
    missing = sorted(required.difference(meta_df.columns))
    if missing:
        raise ValueError(f"meta_df missing required columns: {missing}")
    if meta_df.empty:
        raise ValueError("meta_df is empty.")


def _validate_alignment(
    meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    if int(meta_df.shape[0]) != int(y_true.shape[0]):
        raise ValueError(
            "metadata row count must match y_true length. "
            f"Received meta={meta_df.shape[0]}, y_true={y_true.shape[0]}."
        )
    if int(y_true.shape[0]) != int(y_pred.shape[0]):
        raise ValueError(
            f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )

    meta_true = meta_df["true_rul"].to_numpy(dtype=float)
    if not np.allclose(meta_true, y_true):
        raise ValueError(
            "meta_df.true_rul is not aligned with y_true values in the same row order."
        )


def first_trigger(
    meta_df: pd.DataFrame,
    y_pred: ArrayLike,
    threshold: int,
) -> pd.DataFrame:
    """
    Return the first threshold-crossing trigger event per unit.

    Trigger rule:
        A trigger occurs at the earliest row where ``predicted_rul <= threshold``
        within a unit's trajectory ordered by ``cycle_end``.

    Args:
        meta_df: Metadata dataframe with columns ``unit_id``, ``cycle_end``,
            ``true_rul`` and shape ``(N, 3+)``.
        y_pred: Predicted RUL values aligned to ``meta_df``, shape ``(N,)``.
        threshold: Non-negative trigger threshold on predicted RUL.

    Returns:
        pd.DataFrame: One row per unit with columns:
        - ``unit_id`` (int)
        - ``triggered`` (bool)
        - ``trigger_cycle_end`` (float/int; NaN if not triggered)
        - ``pred_rul_at_trigger`` (float; NaN if not triggered)
        - ``true_rul_at_trigger`` (float; NaN if not triggered)

    Raises:
        ValueError: If metadata schema is invalid, arrays are misaligned, or
            threshold is negative.

    Assumptions & leakage boundaries:
        - Uses prediction outputs only; does not modify model state.
        - Operates per unit after sorting by observed timeline.
    """
    _validate_meta(meta_df)
    y_p = as_float_1d(y_pred, "y_pred")
    if int(meta_df.shape[0]) != int(y_p.shape[0]):
        raise ValueError(
            "metadata row count must match y_pred length. "
            f"Received meta={meta_df.shape[0]}, y_pred={y_p.shape[0]}."
        )

    thr = int(threshold)
    if thr < 0:
        raise ValueError(f"threshold must be >= 0. Received: {thr}")

    work = meta_df.loc[:, ["unit_id", "cycle_end", "true_rul"]].copy()
    work["unit_id"] = work["unit_id"].astype(int)
    work["cycle_end"] = work["cycle_end"].astype(int)
    work["true_rul"] = work["true_rul"].astype(float)
    work["y_pred"] = y_p.astype(float)

    rows = []
    for unit_id, g in work.groupby("unit_id", sort=True):
        g = g.sort_values("cycle_end", ascending=True).reset_index(drop=True)
        triggered_rows = g[g["y_pred"] <= float(thr)]
        if triggered_rows.empty:
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "triggered": False,
                    "trigger_cycle_end": np.nan,
                    "pred_rul_at_trigger": np.nan,
                    "true_rul_at_trigger": np.nan,
                }
            )
        else:
            first = triggered_rows.iloc[0]
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "triggered": True,
                    "trigger_cycle_end": int(first["cycle_end"]),
                    "pred_rul_at_trigger": float(first["y_pred"]),
                    "true_rul_at_trigger": float(first["true_rul"]),
                }
            )

    out = pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)
    if out.empty:
        raise ValueError("first_trigger produced an empty table.")
    return out


def policy_metrics(
    meta_df: pd.DataFrame,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    threshold: int = 20,
    eol_critical: int = 5,
    false_alarm_margin: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    """
    Compute per-unit and aggregate metrics for a maintenance trigger policy.

    Definitions:
        - trigger rule: first time ``pred_rul <= threshold``
        - false alarm: triggered and ``true_rul_at_trigger > threshold + false_alarm_margin``
        - late trigger: triggered and ``true_rul_at_trigger <= eol_critical``
        - missed trigger: no trigger observed for unit trajectory
        - lead time: ``true_rul_at_trigger`` for triggered units

    Args:
        meta_df: Metadata dataframe with columns ``unit_id``, ``cycle_end``,
            ``true_rul`` and shape ``(N, 3+)``.
        y_true: True RUL values aligned to metadata, shape ``(N,)``.
        y_pred: Predicted RUL values aligned to metadata, shape ``(N,)``.
        threshold: Non-negative trigger threshold on predicted RUL.
        eol_critical: Non-negative true-RUL cutoff defining late triggers.
        false_alarm_margin: Non-negative offset above threshold defining false alarms.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
        1. ``unit_df`` with columns:
           ``unit_id``, ``triggered``, ``trigger_cycle_end``,
           ``pred_rul_at_trigger``, ``true_rul_at_trigger``, ``lead_time``,
           ``is_false_alarm``, ``is_late_trigger``, ``missed_trigger``.
        2. ``summary`` dictionary with keys:
           ``threshold``, ``eol_critical``, ``false_alarm_margin``,
           ``n_units``, ``n_triggered``, ``trigger_rate``, ``false_alarm_rate``,
           ``late_trigger_rate``, ``missed_trigger_rate``,
           ``lead_time_mean``, ``lead_time_median``, ``lead_time_p10``,
           ``lead_time_p90``.

    Raises:
        ValueError: If metadata schema/alignment checks fail or any threshold-style
            parameter is negative.

    Assumptions & leakage boundaries:
        - ``meta_df.true_rul`` must align row-wise with ``y_true``.
        - Metrics are post-hoc evaluation outputs; no policy fitting occurs here.
    """
    _validate_meta(meta_df)
    y_t = as_float_1d(y_true, "y_true")
    y_p = as_float_1d(y_pred, "y_pred")
    _validate_alignment(meta_df=meta_df, y_true=y_t, y_pred=y_p)

    thr = int(threshold)
    eol = int(eol_critical)
    margin = int(false_alarm_margin)
    if thr < 0:
        raise ValueError(f"threshold must be >= 0. Received: {thr}")
    if eol < 0:
        raise ValueError(f"eol_critical must be >= 0. Received: {eol}")
    if margin < 0:
        raise ValueError(f"false_alarm_margin must be >= 0. Received: {margin}")

    unit_df = first_trigger(meta_df=meta_df, y_pred=y_p, threshold=thr).copy()

    unit_df["lead_time"] = unit_df["true_rul_at_trigger"]
    unit_df["is_false_alarm"] = (
        unit_df["triggered"]
        & (unit_df["true_rul_at_trigger"] > float(thr + margin))
    )
    unit_df["is_late_trigger"] = (
        unit_df["triggered"] & (unit_df["true_rul_at_trigger"] <= float(eol))
    )
    unit_df["missed_trigger"] = ~unit_df["triggered"]

    unit_df["triggered"] = unit_df["triggered"].astype(bool)
    unit_df["is_false_alarm"] = unit_df["is_false_alarm"].astype(bool)
    unit_df["is_late_trigger"] = unit_df["is_late_trigger"].astype(bool)
    unit_df["missed_trigger"] = unit_df["missed_trigger"].astype(bool)

    n_units = int(unit_df.shape[0])
    n_triggered = int(unit_df["triggered"].sum())
    n_false_alarms = int(unit_df["is_false_alarm"].sum())
    n_late = int(unit_df["is_late_trigger"].sum())
    n_missed = int(unit_df["missed_trigger"].sum())

    lead_times = unit_df.loc[unit_df["triggered"], "lead_time"].to_numpy(dtype=float)

    summary: Dict[str, Optional[float]] = {
        "threshold": float(thr),
        "eol_critical": float(eol),
        "false_alarm_margin": float(margin),
        "n_units": float(n_units),
        "n_triggered": float(n_triggered),
        "trigger_rate": float(n_triggered / n_units),
        "false_alarm_rate": None,
        "late_trigger_rate": None,
        "missed_trigger_rate": float(n_missed / n_units),
        "lead_time_mean": None,
        "lead_time_median": None,
        "lead_time_p10": None,
        "lead_time_p90": None,
    }

    if n_triggered > 0:
        summary["false_alarm_rate"] = float(n_false_alarms / n_triggered)
        summary["late_trigger_rate"] = float(n_late / n_triggered)
        summary["lead_time_mean"] = float(np.mean(lead_times))
        summary["lead_time_median"] = float(np.median(lead_times))
        summary["lead_time_p10"] = float(np.percentile(lead_times, 10))
        summary["lead_time_p90"] = float(np.percentile(lead_times, 90))

    ordered_cols = [
        "unit_id",
        "triggered",
        "trigger_cycle_end",
        "pred_rul_at_trigger",
        "true_rul_at_trigger",
        "lead_time",
        "is_false_alarm",
        "is_late_trigger",
        "missed_trigger",
    ]
    unit_df = unit_df.loc[:, ordered_cols].sort_values("unit_id").reset_index(drop=True)

    return unit_df, summary
