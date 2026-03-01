"""
Matplotlib plotting utilities for baseline evaluation artifacts.

Responsibilities:
- Generate reproducible static figures for model diagnostics.
- Enforce minimal input validation before writing files.

Pipeline fit:
- Inputs are aligned prediction arrays and evaluation tables produced in
  ``src.run_baseline``.
- Outputs are PNG files under ``results/figures``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

# Force matplotlib cache/config into a writable project-local directory.
_MPL_DIR = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _prepare_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_pred_vs_true(y_true, y_pred, title: str, path: str | Path) -> None:
    """
    Plot predicted-vs-true RUL scatter with identity reference line.

    Args:
        y_true: True RUL values, array-like shape ``(N,)``.
        y_pred: Predicted RUL values, array-like shape ``(N,)``.
        title: Figure title.
        path: Output image path.

    Returns:
        None.

    Raises:
        ValueError: If plotting internals receive invalid array values.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    out = _prepare_path(path)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(y_t, y_p, s=10, alpha=0.5)

    lo = float(min(y_t.min(), y_p.min()))
    hi = float(max(y_t.max(), y_p.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_error_vs_rul(y_true, y_pred, title: str, path: str | Path) -> None:
    """
    Plot signed prediction error against true RUL.

    Args:
        y_true: True RUL values, array-like shape ``(N,)``.
        y_pred: Predicted RUL values, array-like shape ``(N,)``.
        title: Figure title.
        path: Output image path.

    Returns:
        None.

    Raises:
        ValueError: If plotting internals receive invalid array values.

    Assumptions & leakage boundaries:
        - Error is defined as ``prediction - truth``.
        - This is a post-hoc visualization with no model-side effects.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    err = y_p - y_t
    out = _prepare_path(path)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(y_t, err, s=10, alpha=0.5)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Prediction Error (Pred - True)")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_unit_mae_hist(
    unit_mae_df: pd.DataFrame, path: str | Path, title: str
) -> None:
    """
    Plot histogram of per-unit MAE values.

    Args:
        unit_mae_df: Dataframe containing required ``MAE`` column.
        path: Output image path.
        title: Figure title.

    Returns:
        None.

    Raises:
        ValueError: If ``MAE`` column is missing or empty.
    """
    if "MAE" not in unit_mae_df.columns:
        raise ValueError("plot_unit_mae_hist requires an 'MAE' column.")

    vals = unit_mae_df["MAE"].to_numpy(dtype=float)
    if vals.size == 0:
        raise ValueError("plot_unit_mae_hist received an empty MAE series.")

    out = _prepare_path(path)
    bins = int(min(25, max(6, np.sqrt(vals.size))))

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(vals, bins=bins, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Unit MAE")
    ax.set_ylabel("Count")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_policy_timelines(
    meta_df: pd.DataFrame,
    y_true: Iterable[float],
    y_pred: Iterable[float],
    selected_units: Sequence[int],
    threshold: int,
    path: str | Path,
    title: str,
) -> None:
    """
    Plot per-unit true/predicted RUL timelines with threshold trigger marker.

    Args:
        meta_df: Metadata dataframe with required columns ``unit_id`` and
            ``cycle_end`` and row count ``N``.
        y_true: True RUL values aligned to ``meta_df``, shape ``(N,)``.
        y_pred: Predicted RUL values aligned to ``meta_df``, shape ``(N,)``.
        selected_units: Unit identifiers to visualize.
        threshold: Non-negative trigger threshold.
        path: Output image path.
        title: Figure title.

    Returns:
        None.

    Raises:
        ValueError: If required columns are missing, array lengths are mismatched,
            threshold is invalid, or selected units are absent from metadata.

    Assumptions & leakage boundaries:
        - Trigger marker is first point where ``y_pred <= threshold`` per unit.
        - Function only visualizes provided arrays and metadata.
    """
    required = {"unit_id", "cycle_end"}
    missing = sorted(required.difference(meta_df.columns))
    if missing:
        raise ValueError(
            f"plot_policy_timelines meta_df missing required columns: {missing}"
        )

    y_t = np.asarray(list(y_true), dtype=float)
    y_p = np.asarray(list(y_pred), dtype=float)
    if y_t.ndim != 1 or y_p.ndim != 1:
        raise ValueError("plot_policy_timelines expects 1D y_true and y_pred.")
    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError(
            f"y_true and y_pred length mismatch: {y_t.shape[0]} vs {y_p.shape[0]}"
        )
    if int(meta_df.shape[0]) != int(y_t.shape[0]):
        raise ValueError(
            "meta_df row count must match y arrays length. "
            f"Received meta={meta_df.shape[0]}, y={y_t.shape[0]}."
        )
    if not selected_units:
        raise ValueError("selected_units must not be empty.")

    thr = int(threshold)
    if thr < 0:
        raise ValueError(f"threshold must be >= 0. Received: {thr}")

    work = meta_df.loc[:, ["unit_id", "cycle_end"]].copy()
    work["unit_id"] = work["unit_id"].astype(int)
    work["cycle_end"] = work["cycle_end"].astype(int)
    work["y_true"] = y_t
    work["y_pred"] = y_p

    units = [int(u) for u in selected_units]
    missing_units = sorted(set(units).difference(set(work["unit_id"].unique())))
    if missing_units:
        raise ValueError(
            f"selected_units contains ids not present in meta_df: {missing_units}"
        )

    out = _prepare_path(path)
    n = len(units)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, max(2.4 * n, 3.0)), sharex=False)
    if n == 1:
        axes = [axes]

    for i, (ax, uid) in enumerate(zip(axes, units)):
        g = (
            work.loc[work["unit_id"] == uid]
            .sort_values("cycle_end", ascending=True)
            .reset_index(drop=True)
        )
        ax.plot(g["cycle_end"], g["y_true"], label="True RUL", linewidth=1.6)
        ax.plot(g["cycle_end"], g["y_pred"], label="Predicted RUL", linewidth=1.4)
        ax.axhline(float(thr), linestyle="--", linewidth=1.1, color="red", label="Threshold")

        trigger = g[g["y_pred"] <= float(thr)]
        if not trigger.empty:
            first = trigger.iloc[0]
            ax.scatter(
                [float(first["cycle_end"])],
                [float(first["y_pred"])],
                marker="x",
                s=40,
                color="black",
                label="Trigger",
                zorder=4,
            )

        ax.set_ylabel(f"Unit {uid}")
        ax.grid(True, linewidth=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Cycle End")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)
