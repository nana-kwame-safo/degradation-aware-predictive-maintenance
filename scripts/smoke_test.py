#!/usr/bin/env python3
"""
End-to-end CMAPSS FD001 smoke test.

Scope:
- load labeled CMAPSS trajectories
- validate schema and temporal ordering
- run leakage-safe split and scaling
- build sequence windows and tabular baseline features

This script intentionally excludes model fitting to keep runtime low.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.data.data_loader import CMAPSSPaths, load_cmapss_subset
from src.data.preprocessing import (
    fit_scaler,
    make_window_features,
    make_windows,
    transform_scaler,
    unit_train_val_split,
)
from src.data.validation import validate_basic_schema, validate_unit_monotonic_cycles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight end-to-end CMAPSS smoke test."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="FD001",
        help="CMAPSS subset to validate (FD001..FD004).",
    )
    parser.add_argument(
        "--rul_cap", type=int, default=125, help="Optional RUL clip value."
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Validation split fraction by unit.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for unit split."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Window length for sequence construction.",
    )
    parser.add_argument("--step", type=int, default=1, help="Window stride.")
    return parser.parse_args()


def _format_shape(shape: Iterable[int]) -> str:
    return str(tuple(int(v) for v in shape))


def _require_sensor_columns(df_columns: Iterable[str]) -> list[str]:
    sensor_cols = [c for c in df_columns if c.startswith("sensor_")]
    if len(sensor_cols) != 21:
        raise ValueError(
            "Expected 21 sensor columns (sensor_1..sensor_21), "
            f"found {len(sensor_cols)}."
        )
    return sensor_cols


def _assert_unit_disjoint(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    train_units = set(train_df["unit_id"].unique())
    val_units = set(val_df["unit_id"].unique())
    overlap = train_units.intersection(val_units)
    if overlap:
        raise ValueError(
            "Unit leakage detected in train/validation split. "
            f"Overlapping unit_id values: {sorted(overlap)}"
        )


def main() -> int:
    cfg = Config()
    args = _parse_args()
    subset = args.subset.strip().upper()
    val_fraction = float(args.val_fraction)
    seed = int(args.seed)
    window = int(args.window)
    step = int(args.step)
    rul_cap = int(args.rul_cap)

    raw_dir = cfg.cmapss_raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"CMAPSS raw directory not found: {raw_dir}. "
            "Place FD001 files under data/raw/cmapss/."
        )

    # Load pre-labeled train/test splits from official CMAPSS files.
    train_df, test_df = load_cmapss_subset(
        paths=CMAPSSPaths(root_dir=raw_dir),
        subset=subset,
        rul_cap=rul_cap,
    )

    # Schema and temporal checks guard against malformed inputs.
    validate_basic_schema(train_df)
    validate_basic_schema(test_df)
    validate_unit_monotonic_cycles(train_df)
    validate_unit_monotonic_cycles(test_df)
    if "rul" not in train_df.columns or "rul" not in test_df.columns:
        raise ValueError("Expected 'rul' column in both train and test frames.")

    # Unit-level partitioning prevents cycle leakage across splits.
    train_split, val_split = unit_train_val_split(
        df=train_df,
        val_fraction=val_fraction,
        seed=seed,
    )
    _assert_unit_disjoint(train_split, val_split)

    # Fit scaler on training split only; reuse it for val/test.
    sensor_cols = _require_sensor_columns(train_split.columns)
    scaler = fit_scaler(train_split, feature_cols=sensor_cols)
    train_scaled = transform_scaler(
        train_split, scaler=scaler, feature_cols=sensor_cols
    )
    val_scaled = transform_scaler(val_split, scaler=scaler, feature_cols=sensor_cols)
    test_scaled = transform_scaler(test_df, scaler=scaler, feature_cols=sensor_cols)

    # Window tensors are used for sequence models and tabular aggregation.
    x_train_w, y_train = make_windows(
        train_scaled, feature_cols=sensor_cols, window=window, step=step
    )
    x_val_w, y_val = make_windows(
        val_scaled, feature_cols=sensor_cols, window=window, step=step
    )
    x_test_w, y_test = make_windows(
        test_scaled, feature_cols=sensor_cols, window=window, step=step
    )

    x_train_tab, feature_names = make_window_features(x_train_w)
    x_val_tab, val_feature_names = make_window_features(x_val_w)
    x_test_tab, test_feature_names = make_window_features(x_test_w)

    expected_feature_count = 126
    if x_train_tab.shape[1] != expected_feature_count:
        raise ValueError(
            f"Tabular feature width mismatch. Expected {expected_feature_count}, "
            f"got {x_train_tab.shape[1]}."
        )
    if feature_names != val_feature_names or feature_names != test_feature_names:
        raise ValueError("Tabular feature definitions differ across splits.")

    # Deterministic diagnostics for CI/local reproducibility checks.
    print("SMOKE_TEST=PASS")
    print(f"{subset} train_df={_format_shape(train_df.shape)}")
    print(f"{subset} val_df={_format_shape(val_split.shape)}")
    print(f"{subset} test_df={_format_shape(test_df.shape)}")
    print(
        f"windows_train X={_format_shape(x_train_w.shape)} y={_format_shape(y_train.shape)}"
    )
    print(
        f"windows_val   X={_format_shape(x_val_w.shape)} y={_format_shape(y_val.shape)}"
    )
    print(
        f"windows_test  X={_format_shape(x_test_w.shape)} y={_format_shape(y_test.shape)}"
    )
    print(f"tabular_train={_format_shape(x_train_tab.shape)}")
    print(f"tabular_val={_format_shape(x_val_tab.shape)}")
    print(f"tabular_test={_format_shape(x_test_tab.shape)}")
    print(f"feature_count={len(feature_names)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
