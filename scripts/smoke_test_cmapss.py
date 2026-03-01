#!/usr/bin/env python3
"""
Minimal CMAPSS baseline sanity check.

Checks:
1) FD001 loads from data/raw/cmapss
2) schema validation passes
3) train/test RUL columns are constructed
4) unit-based split is disjoint
5) train-only scaling runs without leakage
6) window generation returns non-empty tensors
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.data.data_loader import CMAPSSPaths, load_cmapss_subset
from src.data.preprocessing import (
    fit_scaler,
    make_windows,
    transform_scaler,
    unit_train_val_split,
)
from src.data.validation import validate_basic_schema, validate_unit_monotonic_cycles


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> int:
    cfg = Config()
    raw_dir = Path(cfg.cmapss_raw_dir)
    subset = "FD001"

    try:
        if not raw_dir.exists():
            raise FileNotFoundError(f"CMAPSS directory not found: {raw_dir}")

        train_df, test_df = load_cmapss_subset(
            paths=CMAPSSPaths(root_dir=raw_dir),
            subset=subset,
            rul_cap=cfg.clip_rul,
        )
        _ok(f"Loaded {subset} from {raw_dir} | train={train_df.shape}, test={test_df.shape}")

        validate_basic_schema(train_df)
        validate_basic_schema(test_df)
        validate_unit_monotonic_cycles(train_df)
        validate_unit_monotonic_cycles(test_df)
        _ok("Schema and monotonic cycle validation passed")

        if "rul" not in train_df.columns or "rul" not in test_df.columns:
            raise ValueError("Missing 'rul' column in train/test outputs")
        if train_df["rul"].isna().any() or test_df["rul"].isna().any():
            raise ValueError("Found NaNs in constructed RUL targets")
        _ok("Train/test RUL targets constructed successfully")

        train_split, val_split = unit_train_val_split(
            df=train_df,
            val_fraction=cfg.val_fraction,
            seed=cfg.random_state,
        )
        overlap = set(train_split["unit_id"].unique()).intersection(
            set(val_split["unit_id"].unique())
        )
        if overlap:
            raise ValueError(
                f"Train/validation unit leakage detected. Overlap: {sorted(overlap)}"
            )
        _ok(f"Unit-based split passed | train_rows={len(train_split)}, val_rows={len(val_split)}")

        scaler = fit_scaler(train_df=train_split, feature_cols=cfg.sensor_cols)
        scaled_train = transform_scaler(
            train_split, scaler=scaler, feature_cols=cfg.sensor_cols
        )
        scaled_val = transform_scaler(
            val_split, scaler=scaler, feature_cols=cfg.sensor_cols
        )
        scaled_test = transform_scaler(
            test_df, scaler=scaler, feature_cols=cfg.sensor_cols
        )
        if len(scaler.mean_) != len(cfg.sensor_cols):
            raise ValueError("Scaler was not fit on expected sensor columns")

        # Leakage guard: scaler means should match the unscaled training split means.
        train_means = train_split[cfg.sensor_cols].mean().to_numpy(dtype=float)
        if not np.allclose(scaler.mean_, train_means):
            raise ValueError("Scaler statistics do not match train split means")

        _ok(
            "Leakage-safe scaling passed | "
            f"scaled_train={scaled_train.shape}, scaled_val={scaled_val.shape}, scaled_test={scaled_test.shape}"
        )

        X, y, meta_df = make_windows(
            df=scaled_train,
            feature_cols=cfg.sensor_cols,
            window=cfg.window,
            step=cfg.step,
            return_meta=True,
        )
        if X.shape[0] == 0:
            raise ValueError("Window generation returned zero samples")
        if (
            y.shape[0] != X.shape[0]
            or meta_df.shape[0] != X.shape[0]
            or set(meta_df.columns) != {"unit_id", "cycle_end", "true_rul"}
        ):
            raise ValueError("Window outputs have inconsistent sample counts")
        _ok(f"Window generation passed | X={X.shape}, y={y.shape}")

        print("[SUCCESS] CMAPSS smoke test completed.")
        return 0
    except Exception as exc:
        _fail(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
