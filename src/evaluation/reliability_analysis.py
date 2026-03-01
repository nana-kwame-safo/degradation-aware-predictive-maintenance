"""
Decision-focused reliability diagnostics for RUL predictions.

Responsibilities:
- Evaluate threshold-alert behavior via confusion-style metrics.
- Compare policy behavior across multiple trigger thresholds.
- Compute asymmetric cost summaries emphasizing late-intervention risk.

Pipeline fit:
- Consumes aligned ``y_true`` and ``y_pred`` arrays of shape ``(N,)``.
- Produces table/JSON-friendly dictionaries used by ``src.run_baseline``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.utils.arrays import as_float_1d, validate_same_length


ArrayLike = Iterable[float]


def alert_threshold_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    threshold: int = 30,
) -> Dict[str, Optional[float]]:
    """
    Compute confusion-style alert metrics for one trigger threshold.

    Args:
        y_true: Ground-truth RUL values, shape ``(N,)``.
        y_pred: Predicted RUL values, shape ``(N,)``.
        threshold: Non-negative alert threshold on RUL.

    Returns:
        Dict[str, Optional[float]]: Counts and rates including
        ``tp``, ``fp``, ``fn``, ``tn``, ``precision``, ``recall``,
        ``false_alarm_rate``, ``miss_rate``, and alert-time true-RUL stats.

    Raises:
        ValueError: If arrays are invalid/misaligned or threshold is negative.

    Assumptions & leakage boundaries:
        - Alert event is defined by ``predicted_rul <= threshold``.
        - Ground-truth alert-needed condition is ``true_rul <= threshold``.
    """
    y_t = as_float_1d(y_true, "y_true")
    y_p = as_float_1d(y_pred, "y_pred")
    validate_same_length(y_t, y_p)

    thr = int(threshold)
    if thr < 0:
        raise ValueError(f"threshold must be >= 0. Received: {thr}")

    actual_alert = y_t <= float(thr)
    pred_alert = y_p <= float(thr)

    tp = int(np.sum(pred_alert & actual_alert))
    fp = int(np.sum(pred_alert & ~actual_alert))
    fn = int(np.sum(~pred_alert & actual_alert))
    tn = int(np.sum(~pred_alert & ~actual_alert))

    n_total = int(y_t.shape[0])
    n_actual_alert = int(np.sum(actual_alert))
    n_pred_alert = int(np.sum(pred_alert))

    precision = None if n_pred_alert == 0 else float(tp / n_pred_alert)
    recall = None if n_actual_alert == 0 else float(tp / n_actual_alert)
    false_alarm_rate = None if (fp + tn) == 0 else float(fp / (fp + tn))
    miss_rate = None if n_actual_alert == 0 else float(fn / n_actual_alert)

    # True RUL at points where the model would trigger an alert.
    if n_pred_alert == 0:
        mean_true_rul_at_alert = None
        median_true_rul_at_alert = None
    else:
        y_at_alert = y_t[pred_alert]
        mean_true_rul_at_alert = float(np.mean(y_at_alert))
        median_true_rul_at_alert = float(np.median(y_at_alert))

    return {
        "threshold": thr,
        "n_total": n_total,
        "n_actual_alert": n_actual_alert,
        "n_pred_alert": n_pred_alert,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
        "mean_true_rul_at_alert": mean_true_rul_at_alert,
        "median_true_rul_at_alert": median_true_rul_at_alert,
    }


def alert_threshold_sweep(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    thresholds: Sequence[int] = (10, 20, 30),
) -> List[Dict[str, Optional[float]]]:
    """
    Evaluate alert-policy behavior across multiple thresholds.

    Args:
        y_true: Ground-truth RUL values, shape ``(N,)``.
        y_pred: Predicted RUL values, shape ``(N,)``.
        thresholds: Candidate non-negative thresholds.

    Returns:
        List[Dict[str, Optional[float]]]: One metric dictionary per unique threshold,
        sorted ascending by threshold.

    Raises:
        ValueError: If thresholds are empty or include negative values.
    """
    if not thresholds:
        raise ValueError("thresholds must not be empty.")

    rows: List[Dict[str, Optional[float]]] = []
    seen = set()
    for raw in thresholds:
        thr = int(raw)
        if thr < 0:
            raise ValueError(f"threshold values must be >= 0. Received: {thr}")
        if thr in seen:
            continue
        seen.add(thr)
        rows.append(alert_threshold_metrics(y_true=y_true, y_pred=y_pred, threshold=thr))

    rows.sort(key=lambda x: int(x["threshold"]))
    return rows


def weighted_error_cost(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    early_weight: float = 1.0,
    late_weight: float = 2.0,
    severe_late_threshold: float = 10.0,
    severe_late_multiplier: float = 2.0,
    eol_band: Sequence[int] = (0, 20),
) -> Dict[str, Optional[float]]:
    """
    Aggregate asymmetric maintenance cost from signed prediction errors.

    Args:
        y_true: Ground-truth RUL values, shape ``(N,)``.
        y_pred: Predicted RUL values, shape ``(N,)``.
        early_weight: Cost multiplier for early actions (underestimation).
        late_weight: Cost multiplier for late actions (overestimation).
        severe_late_threshold: Threshold beyond which late error is considered severe.
        severe_late_multiplier: Additional multiplier for severe late component.
        eol_band: Inclusive ``(low, high)`` true-RUL band for EOL cost reporting.

    Returns:
        Dict[str, Optional[float]]: Cost components and aggregates including
        ``early_cycles_sum``, ``late_cycles_sum``, ``severe_late_cycles_sum``,
        ``cost_sum``, ``cost_mean``, and ``cost_mean_eol``.

    Raises:
        ValueError: If arrays are invalid/misaligned, weights are invalid, or
            EOL band bounds are invalid.

    Assumptions & leakage boundaries:
        - Error definition is ``pred - true``.
        - Function is a deterministic post-hoc aggregation only.
    """
    y_t = as_float_1d(y_true, "y_true")
    y_p = as_float_1d(y_pred, "y_pred")
    validate_same_length(y_t, y_p)

    ew = float(early_weight)
    lw = float(late_weight)
    severe_thr = float(severe_late_threshold)
    severe_mult = float(severe_late_multiplier)
    eol_low = int(eol_band[0])
    eol_high = int(eol_band[1])

    if ew < 0 or lw < 0:
        raise ValueError(
            f"early_weight and late_weight must be >= 0. Received: {ew}, {lw}"
        )
    if severe_thr <= 0:
        raise ValueError(
            f"severe_late_threshold must be > 0. Received: {severe_late_threshold}"
        )
    if severe_mult < 1.0:
        raise ValueError(
            "severe_late_multiplier must be >= 1.0. "
            f"Received: {severe_late_multiplier}"
        )
    if eol_low < 0 or eol_high < eol_low:
        raise ValueError(f"Invalid eol_band: {tuple(eol_band)}")

    err = y_p - y_t
    early_cycles = np.maximum(0.0, -err)
    late_cycles = np.maximum(0.0, err)
    severe_late_cycles = np.maximum(0.0, late_cycles - severe_thr)

    n_total = int(y_t.shape[0])
    base_cost = (ew * early_cycles) + (lw * late_cycles)
    severe_extra = (lw * (severe_mult - 1.0) * severe_late_cycles)
    total_cost = base_cost + severe_extra

    eol_mask = (y_t >= float(eol_low)) & (y_t <= float(eol_high))
    n_eol = int(np.sum(eol_mask))
    if n_eol == 0:
        mean_cost_eol = None
    else:
        mean_cost_eol = float(np.mean(total_cost[eol_mask]))

    return {
        "n_total": n_total,
        "n_eol": n_eol,
        "early_weight": ew,
        "late_weight": lw,
        "severe_late_threshold": severe_thr,
        "severe_late_multiplier": severe_mult,
        "eol_low": eol_low,
        "eol_high": eol_high,
        "early_cycles_sum": float(np.sum(early_cycles)),
        "late_cycles_sum": float(np.sum(late_cycles)),
        "severe_late_cycles_sum": float(np.sum(severe_late_cycles)),
        "cost_sum": float(np.sum(total_cost)),
        "cost_mean": float(np.mean(total_cost)),
        "cost_mean_eol": mean_cost_eol,
    }
