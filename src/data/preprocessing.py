"""
Leakage-safe preprocessing utilities for baseline-ready CMAPSS features.

Responsibilities:
- Partition trajectories by unit to prevent temporal leakage.
- Fit/apply feature scaling with explicit train-only boundaries.
- Convert per-cycle sensor frames into window tensors and tabular statistics.

Pipeline fit:
- Input: labeled CMAPSS frames with required columns (``unit_id``, ``cycle``, ``rul``).
- Output: split/scaled dataframes, window tensors ``(N, W, d)``, metadata frames,
  and tabular feature matrices for classical regressors.

Assumptions & leakage boundaries:
- Splits are by unit identity, never by row.
- Scaler parameters are learned from training rows only.
- Each window is contained within a single unit trajectory.
"""

from __future__ import annotations

import warnings
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
    Split a dataframe into train/validation partitions by ``unit_id`` only.

    Args:
        df: Input dataframe containing ``unit_id``.
        val_fraction: Fraction of units assigned to validation in ``(0, 1)``.
        seed: Random seed for deterministic unit shuffling.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: ``(train_df, val_df)``.

    Raises:
        ValueError: If schema is invalid, fraction is out of bounds, there are too
            few units, or either partition becomes empty.

    Assumptions & leakage boundaries:
        - Unit-level split avoids train/validation leakage from shared trajectories.
        - No row-level random split is performed.
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
    Fit ``StandardScaler`` on training rows for selected feature columns.

    Args:
        train_df: Training dataframe containing the requested feature columns.
        feature_cols: Ordered feature column names to scale.

    Returns:
        StandardScaler: Fitted scaler with train-only statistics.

    Raises:
        ValueError: If dataframe integrity, feature existence, or finite-value
            checks fail.

    Assumptions & leakage boundaries:
        - This function must be called on training data only.
        - Validation/test rows must be transformed via :func:`transform_scaler`.
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
    """
    Apply a pre-fitted scaler to a dataframe copy.

    Args:
        df: Input dataframe to transform.
        scaler: Previously fitted ``StandardScaler`` instance.
        feature_cols: Ordered feature columns to transform.

    Returns:
        pd.DataFrame: Copy of ``df`` with transformed ``feature_cols``.

    Raises:
        ValueError: If dataframe integrity, feature existence, or finite-value
            checks fail.

    Assumptions & leakage boundaries:
        - ``scaler`` is expected to come from train-only fitting.
        - Non-feature columns are preserved unchanged.
    """
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
    return_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build fixed-length per-unit windows and aligned RUL targets.

    Args:
        df: Input dataframe with required columns ``unit_id``, ``cycle``, ``rul``
            and all ``feature_cols``.
        feature_cols: Ordered feature columns used as channels.
        window: Window length ``W`` (must be ``> 0``).
        step: Stride between window end indices (must be ``> 0``).
        return_meta: If ``True``, return aligned metadata frame.

    Returns:
        Tuple[np.ndarray, np.ndarray] or
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        - ``X`` with shape ``(N, W, d)``
        - ``y`` with shape ``(N,)`` (RUL at window end)
        - optional ``meta_df`` with columns ``unit_id``, ``cycle_end``, ``true_rul``

    Raises:
        ValueError: If schema checks fail, values are invalid, or no windows are
            produced for the requested parameters.

    Assumptions & leakage boundaries:
        - Windows never cross unit boundaries.
        - ``cycle`` must be strictly increasing per unit.
        - Targets are end-aligned: ``y[i] = rul`` at each window end index.
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
    meta_rows: List[dict[str, float | int]] = []

    ordered = df.sort_values(["unit_id", "cycle"]).copy()
    for uid, g in ordered.groupby("unit_id", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < window:
            continue

        x_values = g[cols].to_numpy(dtype=float)
        y_values = g["rul"].to_numpy(dtype=float)
        cycle_values = g["cycle"].to_numpy(dtype=int)

        for end in range(window - 1, len(g), step):
            start = end - window + 1
            target = float(y_values[end])
            x_list.append(x_values[start : end + 1])
            y_list.append(target)
            if return_meta:
                meta_rows.append(
                    {
                        "unit_id": int(uid),
                        "cycle_end": int(cycle_values[end]),
                        "true_rul": target,
                    }
                )

    if not x_list:
        raise ValueError(
            "Window construction produced zero samples. "
            "Check window/step settings and minimum trajectory lengths."
        )

    x_arr = np.stack(x_list, axis=0)
    y_arr = np.asarray(y_list, dtype=float)
    if not return_meta:
        return x_arr, y_arr

    meta_df = pd.DataFrame(meta_rows, columns=["unit_id", "cycle_end", "true_rul"])
    if meta_df.shape[0] != y_arr.shape[0]:
        raise ValueError("make_windows metadata rows are not aligned with targets.")
    return x_arr, y_arr, meta_df


def make_window_features(
    x: np.ndarray, feature_cols: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate each window tensor into tabular baseline features.

    Args:
        x: Window tensor with shape ``(N, W, d)``.
        feature_cols: Ordered channel names of length ``d``.

    Returns:
        Tuple[np.ndarray, List[str]]:
        - ``features`` with shape ``(N, 6 * d)``
        - ``feature_names`` in deterministic stat-major order:
          ``{col}_{mean|std|min|max|last|slope}``

    Raises:
        ValueError: If input tensor dimensionality is invalid, samples are empty,
            window length is too short, or ``feature_cols`` do not match ``d``.

    Assumptions & leakage boundaries:
        - This function is a deterministic transformation of existing windows.
        - No fitting or data-dependent global state is introduced.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"x must have shape (N, W, d). Received shape: {arr.shape}")

    n_samples, window, n_features = arr.shape
    if n_samples == 0:
        raise ValueError("x has zero samples; cannot compute window features.")
    if window < 2:
        raise ValueError("window length must be >= 2 to compute slope features.")

    cols = list(feature_cols)
    if len(cols) != n_features:
        raise ValueError(
            "feature_cols length must match window channel count. "
            f"len(feature_cols)={len(cols)} != n_features={n_features}."
        )
    if len(set(cols)) != len(cols):
        raise ValueError("feature_cols must be unique.")

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
        for col in cols:
            names.append(f"{col}_{stat}")

    return features, names


# -----------------------------------------------------------------------------
# Backward-compatible wrappers
# -----------------------------------------------------------------------------


def _warn_deprecated(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated and will be removed in a future release. "
        f"Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def split_by_unit(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deprecated wrapper for :func:`unit_train_val_split`.

    Args:
        df: Input dataframe.
        unit_col: Unit identifier column (must be ``"unit_id"``).
        val_fraction: Validation fraction by unit.
        random_state: Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: ``(train_df, val_df)``.

    Raises:
        ValueError: If unsupported ``unit_col`` is supplied or split validation fails.
    """
    _warn_deprecated(
        "split_by_unit",
        "unit_train_val_split(df, val_fraction, seed)",
    )
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
    """
    Deprecated wrapper combining scaler fit and transform steps.

    Args:
        train_df: Training dataframe used to fit scaler and transform train.
        sensor_cols: Feature columns to scale.
        val_df: Optional validation dataframe to transform.
        test_df: Optional test dataframe to transform.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], StandardScaler]:
        ``(scaled_train, scaled_val, scaled_test, scaler)``.

    Raises:
        ValueError: Propagated from canonical scaling functions.

    Assumptions & leakage boundaries:
        - Scaler is fit on ``train_df`` only.
        - ``val_df`` and ``test_df`` are transformed with the same scaler.
    """
    _warn_deprecated(
        "scale_sensor_columns",
        "fit_scaler(...) and transform_scaler(...)",
    )
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


def generate_unit_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "rul",
    unit_col: str = "unit_id",
    time_col: str = "cycle",
    window_size: int = 30,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deprecated metadata-array wrapper for :func:`make_windows`.

    Args:
        df: Input dataframe with required windowing columns.
        feature_cols: Ordered feature columns used as channels.
        target_col: Target column name (must be ``"rul"``).
        unit_col: Unit identifier column (must be ``"unit_id"``).
        time_col: Time column name (must be ``"cycle"``).
        window_size: Window length ``W``.
        stride: Window stride.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ``(X, y, unit_ids, end_cycles)`` aligned by sample index.

    Raises:
        ValueError: If unsupported legacy column names are provided or canonical
            window generation fails.
    """
    _warn_deprecated(
        "generate_unit_windows",
        "make_windows(df, feature_cols, window, step, return_meta=True)",
    )
    if target_col != "rul" or unit_col != "unit_id" or time_col != "cycle":
        raise ValueError(
            "generate_unit_windows currently expects target_col='rul', unit_col='unit_id', time_col='cycle'."
        )

    x_arr, y_arr, meta_df = make_windows(
        df=df,
        feature_cols=feature_cols,
        window=window_size,
        step=stride,
        return_meta=True,
    )
    return (
        x_arr,
        y_arr,
        meta_df["unit_id"].to_numpy(dtype=int),
        meta_df["cycle_end"].to_numpy(dtype=int),
    )


def build_tabular_baseline_features(
    df: pd.DataFrame,
    sensor_cols: Sequence[str],
    target_col: str = "rul",
    unit_col: str = "unit_id",
    time_col: str = "cycle",
    window_size: int = 30,
) -> pd.DataFrame:
    """
    Deprecated wrapper to build tabular baseline dataframe from raw trajectories.

    Args:
        df: Input dataframe with unit/cycle/rul and sensor columns.
        sensor_cols: Ordered sensor columns used for window channels.
        target_col: Target column name (must be ``"rul"``).
        unit_col: Unit identifier column (must be ``"unit_id"``).
        time_col: Time column name (must be ``"cycle"``).
        window_size: Window length ``W``.

    Returns:
        pd.DataFrame: Table with metadata columns ``unit_id``, ``cycle``, ``rul``
        followed by deterministic tabular window features.

    Raises:
        ValueError: If unsupported legacy column names are provided or canonical
            window/feature generation fails.
    """
    _warn_deprecated(
        "build_tabular_baseline_features",
        "make_windows(...) and make_window_features(..., feature_cols)",
    )
    if target_col != "rul" or unit_col != "unit_id" or time_col != "cycle":
        raise ValueError(
            "build_tabular_baseline_features currently expects target_col='rul', "
            "unit_col='unit_id', time_col='cycle'."
        )

    x_arr, _, meta_df = make_windows(
        df=df,
        feature_cols=sensor_cols,
        window=window_size,
        step=1,
        return_meta=True,
    )
    feats, names = make_window_features(x_arr, feature_cols=sensor_cols)
    meta_out = meta_df.rename(columns={"cycle_end": "cycle", "true_rul": "rul"})
    feat_out = pd.DataFrame(feats, columns=names)
    return pd.concat(
        [meta_out[["unit_id", "cycle", "rul"]].reset_index(drop=True), feat_out],
        axis=1,
    )
