"""
Feature engineering for degradation modelling.

Baseline-first approach:
- Convert time-series sensor streams into fixed-length window features.
- Use simple rolling mean/std as interpretable engineered features.

Later extensions (not in this baseline file):
- trend/derivative features
- operating regime normalization
- learned embeddings
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def make_window_features(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    sensor_cols: List[str],
    window: int,
    target: str,
    clip_rul: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert per-cycle sensor readings into fixed-length window features per unit:
    - rolling mean (window)
    - rolling std (window)

    The sample is aligned to the end of each rolling window.

    Parameters
    ----------
    df:
        Input dataframe containing unit/time columns, sensors, and target.
    unit_col, time_col:
        Column names for unit identifier and time index.
    sensor_cols:
        Columns used as sensor inputs.
    window:
        Rolling window size (cycles).
    target:
        Target column name (e.g., 'rul').
    clip_rul:
        Optional cap for RUL targets (common for CMAPSS).

    Returns
    -------
    X, y:
        numpy arrays of features and targets.
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available: {sorted(df.columns)}")

    df = df.sort_values([unit_col, time_col]).copy()

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for _, g in df.groupby(unit_col):
        g = g.reset_index(drop=True)
        if len(g) < window:
            continue

        roll_mean = g[sensor_cols].rolling(window).mean().iloc[window - 1 :]
        roll_std = g[sensor_cols].rolling(window).std(ddof=0).iloc[window - 1 :]

        X = pd.concat(
            [roll_mean.add_suffix("_mean"), roll_std.add_suffix("_std")],
            axis=1,
        )
        y = g[target].iloc[window - 1 :].to_numpy(dtype=float)

        if clip_rul is not None and target.lower() == "rul":
            y = np.minimum(y, float(clip_rul))

        X_list.append(X.to_numpy(dtype=float))
        y_list.append(y)

    if not X_list:
        return np.empty((0, 0)), np.empty((0,))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    return X_all, y_all
