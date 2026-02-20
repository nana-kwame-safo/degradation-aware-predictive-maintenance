"""
Leakage-safe preprocessing utilities for CMAPSS baseline modelling.

Design assumptions:
- Splits are unit-based (never row-based).
- Scalers are fit on training data only.
- Window extraction never crosses unit boundaries.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def _validate_feature_columns(
    df: pd.DataFrame, feature_cols: Sequence[str], context: str
) -> List[str]:
    cols = list(feature_cols)
    if not cols:
        raise ValueError(f"{context}: feature_cols is empty.")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing feature columns: {missing}")

    return cols


def _validate_unit_integrity(
    df: pd.DataFrame, context: str, require_cycle: bool = False
) -> None:
    if df.empty:
        raise ValueError(f"{context}: dataframe is empty.")
    if "unit_id" not in df.columns:
        raise ValueError(f"{context}: missing required column 'unit_id'.")
    if df["unit_id"].isna().any():
        raise ValueError(f"{context}: found NaN in unit_id.")
    if (df["unit_id"] <= 0).any():
        raise ValueError(f"{context}: unit_id values must be > 0.")

    if require_cycle:
        if "cycle" not in df.columns:
            raise ValueError(f"{context}: missing required column 'cycle'.")
        if df["cycle"].isna().any():
            raise ValueError(f"{context}: found NaN in cycle.")
        if (df["cycle"] <= 0).any():
            raise ValueError(f"{context}: cycle values must be > 0.")

        if df.duplicated(subset=["unit_id", "cycle"]).any():
            raise ValueError(f"{context}: duplicate (unit_id, cycle) rows detected.")

        ordered = df.sort_values(["unit_id", "cycle"])
        bad_units: List[int] = []
        for uid, g in ordered.groupby("unit_id", sort=True):
            diffs = g["cycle"].diff().dropna()
            if not (diffs > 0).all():
                bad_units.append(int(uid))
                if len(bad_units) >= 5:
                    break
        if bad_units:
            raise ValueError(
                f"{context}: cycles are not strictly increasing for units: {bad_units}"
            )


def _validate_finite_values(
    df: pd.DataFrame, cols: Sequence[str], context: str
) -> None:
    arr = df[list(cols)].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{context}: found non-finite values in feature columns.")


# -----------------------------------------------------------------------------
# Core baseline API
# -----------------------------------------------------------------------------


def unit_train_val_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by unit_id only.

    Rationale:
    row-level splitting leaks temporal information from the same unit into both
    train and validation partitions, which inflates reported performance.
    """
    _validate_unit_integrity(df, context="unit_train_val_split", require_cycle=False)

    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1). Received: {val_fraction}")

    unit_ids = np.array(sorted(df["unit_id"].unique()), dtype=int)
    if unit_ids.size < 2:
        raise ValueError(
            "At least two units are required for train/validation splitting."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(unit_ids)

    n_val_units = max(1, int(round(unit_ids.size * float(val_fraction))))
    n_val_units = min(n_val_units, unit_ids.size - 1)
    val_units = set(unit_ids[:n_val_units].tolist())

    train_df = df[~df["unit_id"].isin(val_units)].copy()
    val_df = df[df["unit_id"].isin(val_units)].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            "unit_train_val_split produced an empty partition. "
            f"Check val_fraction={val_fraction} and number of units={unit_ids.size}."
        )

    return train_df, val_df


def fit_scaler(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> StandardScaler:
    """
    Fit a StandardScaler on training rows only.

    This is the leakage boundary: validation/test statistics must not influence
    transform parameters used during model fitting.
    """
    _validate_unit_integrity(train_df, context="fit_scaler", require_cycle=False)
    cols = _validate_feature_columns(train_df, feature_cols, context="fit_scaler")
    _validate_finite_values(train_df, cols, context="fit_scaler")

    scaler = StandardScaler()
    scaler.fit(train_df[cols].to_numpy(dtype=float))
    return scaler


def transform_scaler(
    df: pd.DataFrame, scaler: StandardScaler, feature_cols: Sequence[str]
) -> pd.DataFrame:
    """Apply a pre-fitted scaler to a dataframe copy."""
    _validate_unit_integrity(df, context="transform_scaler", require_cycle=False)
    cols = _validate_feature_columns(df, feature_cols, context="transform_scaler")
    _validate_finite_values(df, cols, context="transform_scaler")

    out = df.copy()
    out[cols] = scaler.transform(out[cols].to_numpy(dtype=float))
    return out


def make_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build fixed-length sensor windows and aligned RUL targets.

    Target assignment:
    - Each window target is the RUL at the window end index.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0. Received: {window}")
    if step <= 0:
        raise ValueError(f"step must be > 0. Received: {step}")

    _validate_unit_integrity(df, context="make_windows", require_cycle=True)
    cols = _validate_feature_columns(df, feature_cols, context="make_windows")

    required = {"rul", *cols}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"make_windows: missing required columns: {missing}")

    if df["rul"].isna().any():
        raise ValueError("make_windows: found NaN in rul column.")
    if (df["rul"] < 0).any():
        raise ValueError("make_windows: found negative rul values.")

    _validate_finite_values(df, cols, context="make_windows")

    x_list: List[np.ndarray] = []
    y_list: List[float] = []

    ordered = df.sort_values(["unit_id", "cycle"]).copy()
    for _, g in ordered.groupby("unit_id", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < window:
            continue

        x_values = g[cols].to_numpy(dtype=float)
        y_values = g["rul"].to_numpy(dtype=float)

        for end in range(window - 1, len(g), step):
            start = end - window + 1
            x_list.append(x_values[start : end + 1])
            y_list.append(float(y_values[end]))

    if not x_list:
        raise ValueError(
            "Window construction produced zero samples. "
            "Check window/step settings and minimum trajectory lengths."
        )

    x_arr = np.stack(x_list, axis=0)
    y_arr = np.asarray(y_list, dtype=float)
    return x_arr, y_arr


def make_window_features(x: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate each window into tabular features.

    Per channel statistics:
    - mean, std, min, max, last, slope

    For 21 channels this yields 126 features.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"x must have shape (N, W, d). Received shape: {arr.shape}")

    n_samples, window, n_features = arr.shape
    if n_samples == 0:
        raise ValueError("x has zero samples; cannot compute window features.")
    if window < 2:
        raise ValueError("window length must be >= 2 to compute slope features.")

    t = np.arange(window, dtype=float)
    t_centered = t - t.mean()
    denom = float(np.sum(t_centered**2))

    mean_feat = arr.mean(axis=1)
    std_feat = arr.std(axis=1, ddof=0)
    min_feat = arr.min(axis=1)
    max_feat = arr.max(axis=1)
    last_feat = arr[:, -1, :]
    slope_feat = (arr * t_centered[None, :, None]).sum(axis=1) / denom

    features = np.concatenate(
        [mean_feat, std_feat, min_feat, max_feat, last_feat, slope_feat],
        axis=1,
    )

    names: List[str] = []
    for stat in ["mean", "std", "min", "max", "last", "slope"]:
        for i in range(n_features):
            names.append(f"sensor_{i + 1}_{stat}")

    return features, names


# -----------------------------------------------------------------------------
# Backward-compatible wrappers
# -----------------------------------------------------------------------------


def split_by_unit(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper around unit_train_val_split."""
    if unit_col != "unit_id":
        raise ValueError("split_by_unit currently supports unit_col='unit_id' only.")
    return unit_train_val_split(df=df, val_fraction=val_fraction, seed=random_state)


def scale_sensor_columns(
    train_df: pd.DataFrame,
    sensor_cols: Iterable[str],
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> Tuple[
    pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], StandardScaler
]:
    """Fit on train, transform train/val/test with the same scaler."""
    cols = list(sensor_cols)
    scaler = fit_scaler(train_df=train_df, feature_cols=cols)
    out_train = transform_scaler(train_df, scaler=scaler, feature_cols=cols)
    out_val = (
        transform_scaler(val_df, scaler=scaler, feature_cols=cols)
        if val_df is not None
        else None
    )
    out_test = (
        transform_scaler(test_df, scaler=scaler, feature_cols=cols)
        if test_df is not None
        else None
    )
    return out_train, out_val, out_test, scaler


def assert_unit_disjoint(
    train_df: pd.DataFrame, val_df: pd.DataFrame, unit_col: str = "unit_id"
) -> None:
    """Raise if train and validation partitions share any unit identifier."""
    if unit_col not in train_df.columns or unit_col not in val_df.columns:
        raise ValueError(
            f"assert_unit_disjoint requires column '{unit_col}' in both dataframes."
        )
    overlap = set(train_df[unit_col].unique()).intersection(
        set(val_df[unit_col].unique())
    )
    if overlap:
        raise ValueError(
            f"Train/validation unit leakage detected. Overlap: {sorted(overlap)}"
        )


def generate_unit_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "rul",
    unit_col: str = "unit_id",
    time_col: str = "cycle",
    window_size: int = 30,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible window generator returning metadata arrays."""
    if target_col != "rul" or unit_col != "unit_id" or time_col != "cycle":
        raise ValueError(
            "generate_unit_windows currently expects target_col='rul', unit_col='unit_id', time_col='cycle'."
        )

    x_arr, y_arr = make_windows(
        df=df, feature_cols=feature_cols, window=window_size, step=stride
    )

    unit_ids: List[int] = []
    end_cycles: List[int] = []
    ordered = df.sort_values(["unit_id", "cycle"]).copy()
    for uid, g in ordered.groupby("unit_id", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < window_size:
            continue
        for end in range(window_size - 1, len(g), stride):
            unit_ids.append(int(uid))
            end_cycles.append(int(g.loc[end, "cycle"]))

    return (
        x_arr,
        y_arr,
        np.asarray(unit_ids, dtype=int),
        np.asarray(end_cycles, dtype=int),
    )


def build_tabular_baseline_features(
    df: pd.DataFrame,
    sensor_cols: Sequence[str],
    target_col: str = "rul",
    unit_col: str = "unit_id",
    time_col: str = "cycle",
    window_size: int = 30,
) -> pd.DataFrame:
    """Backward-compatible tabular baseline feature builder."""
    if target_col != "rul" or unit_col != "unit_id" or time_col != "cycle":
        raise ValueError(
            "build_tabular_baseline_features currently expects target_col='rul', "
            "unit_col='unit_id', time_col='cycle'."
        )

    x_arr, y_arr = make_windows(
        df=df, feature_cols=sensor_cols, window=window_size, step=1
    )
    feats, names = make_window_features(x_arr)

    rows: List[dict] = []
    ordered = df.sort_values(["unit_id", "cycle"]).copy()
    idx = 0
    for uid, g in ordered.groupby("unit_id", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < window_size:
            continue
        for end in range(window_size - 1, len(g), 1):
            record = {
                "unit_id": int(uid),
                "cycle": int(g.loc[end, "cycle"]),
                "rul": float(y_arr[idx]),
            }
            for j, name in enumerate(names):
                record[name] = float(feats[idx, j])
            rows.append(record)
            idx += 1

    return pd.DataFrame(rows)
