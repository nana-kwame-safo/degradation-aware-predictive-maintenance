"""
Matplotlib-only plotting helpers for baseline evaluation artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

# Force matplotlib cache/config into a writable project-local directory.
_MPL_DIR = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _prepare_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_pred_vs_true(y_true, y_pred, title: str, path: str | Path) -> None:
    """
    Scatter plot of predicted vs true RUL with ideal reference line.
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
    Scatter plot of signed prediction error against true RUL.

    Error is defined as: ``prediction - truth``.
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
