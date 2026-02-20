#!/usr/bin/env python3
"""
Train baseline RUL models on CMAPSS with leakage-safe preprocessing.

Pipeline overview:
1) Load CMAPSS subset with robust target construction.
2) Split training data by unit_id (no unit leakage).
3) Fit scaler on training split only; transform val/test.
4) Build sequence windows and aggregate into tabular baseline features.
5) Train Ridge, ElasticNet, and RandomForest baselines.
6) Evaluate overall and stratified by RUL bands.
7) Save metrics/tables/figures under results/.

This script includes simple assertion checks as smoke guards, without pytest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Runtime guard for restricted/container environments where threaded BLAS/OpenMP
# can fail due shared-memory limitations.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

# Ensure repository root is importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.data_loader import CMAPSSPaths, load_cmapss_subset
from src.data.preprocessing import (
    fit_scaler,
    make_window_features,
    make_windows,
    transform_scaler,
    unit_train_val_split,
)
from src.evaluation.metrics import (
    regression_metrics,
    stratified_metrics_by_rul_bins,
    unit_level_error_summary,
)
from src.evaluation.plots import plot_error_vs_rul, plot_pred_vs_true
from src.models.baseline_models import (
    predict,
    train_elasticnet,
    train_random_forest,
    train_ridge,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leakage-safe baseline models for CMAPSS RUL.")
    parser.add_argument("--subset", type=str, default="FD001", help="CMAPSS subset (FD001..FD004)")
    parser.add_argument("--rul_cap", type=int, default=125, help="Optional RUL cap applied in loader")
    parser.add_argument("--window", type=int, default=30, help="Window length for sequence construction")
    parser.add_argument("--step", type=int, default=1, help="Window stride")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Validation fraction by unit")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    return parser.parse_args()


def _sensor_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("sensor_")]
    if not cols:
        raise ValueError("No sensor columns found. Expected columns starting with 'sensor_'.")
    return cols


def _window_end_unit_ids(df: pd.DataFrame, window: int, step: int) -> List[int]:
    """Reconstruct unit ids aligned with make_windows output ordering."""
    out: List[int] = []
    ordered = df.sort_values(["unit_id", "cycle"]).copy()
    for uid, g in ordered.groupby("unit_id", sort=True):
        if len(g) < window:
            continue
        for end in range(window - 1, len(g), step):
            _ = end
            out.append(int(uid))
    return out


def _prepare_results_dirs(base: Path) -> Tuple[Path, Path, Path]:
    metrics_dir = base / "results" / "metrics"
    tables_dir = base / "results" / "tables"
    figures_dir = base / "results" / "figures"
    for d in (metrics_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)
    return metrics_dir, tables_dir, figures_dir


def main() -> int:
    args = _parse_args()
    subset = args.subset.strip().upper()

    raw_dir = REPO_ROOT / "data" / "raw" / "cmapss"
    if not raw_dir.exists():
        raise FileNotFoundError(f"CMAPSS directory not found: {raw_dir}")

    # ------------------------------------------------------------------
    # 1) Load CMAPSS subset with RUL labels
    # ------------------------------------------------------------------
    train_df, test_df = load_cmapss_subset(
        paths=CMAPSSPaths(root_dir=raw_dir),
        subset=subset,
        rul_cap=int(args.rul_cap),
    )

    assert "rul" in train_df.columns and "rul" in test_df.columns, "RUL targets were not constructed."
    assert train_df["rul"].notna().all() and test_df["rul"].notna().all(), "RUL contains NaN values."

    sensor_cols = _sensor_columns(train_df)

    # ------------------------------------------------------------------
    # 2) Leakage-safe split by unit
    # ------------------------------------------------------------------
    train_split, val_split = unit_train_val_split(
        df=train_df,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )

    train_units = set(train_split["unit_id"].unique().tolist())
    val_units = set(val_split["unit_id"].unique().tolist())
    assert len(train_units.intersection(val_units)) == 0, "Unit leakage detected between train and validation."

    # ------------------------------------------------------------------
    # 3) Leakage-safe scaling (fit train only)
    # ------------------------------------------------------------------
    scaler = fit_scaler(train_df=train_split, feature_cols=sensor_cols)
    train_scaled = transform_scaler(train_split, scaler=scaler, feature_cols=sensor_cols)
    val_scaled = transform_scaler(val_split, scaler=scaler, feature_cols=sensor_cols)
    test_scaled = transform_scaler(test_df, scaler=scaler, feature_cols=sensor_cols)

    # ------------------------------------------------------------------
    # 4) Window generation (sensor channels only)
    # ------------------------------------------------------------------
    X_train_w, y_train = make_windows(train_scaled, feature_cols=sensor_cols, window=args.window, step=args.step)
    X_val_w, y_val = make_windows(val_scaled, feature_cols=sensor_cols, window=args.window, step=args.step)
    X_test_w, y_test = make_windows(test_scaled, feature_cols=sensor_cols, window=args.window, step=args.step)

    assert X_train_w.ndim == 3 and X_val_w.ndim == 3 and X_test_w.ndim == 3, "Window tensors must be 3D."
    assert X_train_w.shape[0] == y_train.shape[0], "Train window/target length mismatch."
    assert X_val_w.shape[0] == y_val.shape[0], "Val window/target length mismatch."
    assert X_test_w.shape[0] == y_test.shape[0], "Test window/target length mismatch."

    print(
        "[SHAPES] windows "
        f"train={X_train_w.shape}, val={X_val_w.shape}, test={X_test_w.shape}"
    )

    # ------------------------------------------------------------------
    # 5) Window aggregation for classical baselines
    # ------------------------------------------------------------------
    X_train, feature_names = make_window_features(X_train_w)
    X_val, feature_names_val = make_window_features(X_val_w)
    X_test, feature_names_test = make_window_features(X_test_w)

    assert feature_names == feature_names_val == feature_names_test, "Feature naming mismatch across splits."
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature dimension mismatch across splits."

    print(
        "[SHAPES] tabular "
        f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    # ------------------------------------------------------------------
    # 6) Train baselines and evaluate
    # ------------------------------------------------------------------
    metrics_dir, tables_dir, figures_dir = _prepare_results_dirs(REPO_ROOT)

    models = {
        "ridge": train_ridge(X_train, y_train),
        "elasticnet": train_elasticnet(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train, seed=int(args.seed)),
    }

    rul_bins = [0, 30, 60, 125]
    val_unit_ids = _window_end_unit_ids(val_scaled, window=args.window, step=args.step)
    test_unit_ids = _window_end_unit_ids(test_scaled, window=args.window, step=args.step)

    assert len(val_unit_ids) == len(y_val), "Validation unit-id alignment mismatch."
    assert len(test_unit_ids) == len(y_test), "Test unit-id alignment mismatch."

    summary_rows: List[Dict[str, float]] = []
    metrics_payload: Dict[str, object] = {
        "config": {
            "subset": subset,
            "rul_cap": int(args.rul_cap),
            "window": int(args.window),
            "step": int(args.step),
            "val_fraction": float(args.val_fraction),
            "seed": int(args.seed),
            "rul_bins": rul_bins,
        },
        "shapes": {
            "train_windows": list(X_train_w.shape),
            "val_windows": list(X_val_w.shape),
            "test_windows": list(X_test_w.shape),
            "train_features": list(X_train.shape),
            "val_features": list(X_val.shape),
            "test_features": list(X_test.shape),
        },
        "models": {},
    }

    for model_name, model in models.items():
        y_val_pred = predict(model, X_val)
        y_test_pred = predict(model, X_test)

        val_overall = regression_metrics(y_val, y_val_pred)
        test_overall = regression_metrics(y_test, y_test_pred)

        val_bands = stratified_metrics_by_rul_bins(y_val, y_val_pred, bins=rul_bins)
        test_bands = stratified_metrics_by_rul_bins(y_test, y_test_pred, bins=rul_bins)

        val_unit = unit_level_error_summary(y_val, y_val_pred, unit_ids=val_unit_ids)
        test_unit = unit_level_error_summary(y_test, y_test_pred, unit_ids=test_unit_ids)

        metrics_payload["models"][model_name] = {
            "validation": {
                "overall": val_overall,
                "by_rul_band": val_bands,
                "unit_summary": val_unit,
            },
            "test": {
                "overall": test_overall,
                "by_rul_band": test_bands,
                "unit_summary": test_unit,
            },
        }

        summary_rows.append(
            {
                "model": model_name,
                "val_MAE": float(val_overall["MAE"]),
                "val_RMSE": float(val_overall["RMSE"]),
                "test_MAE": float(test_overall["MAE"]),
                "test_RMSE": float(test_overall["RMSE"]),
            }
        )

        plot_pred_vs_true(
            y_true=y_test,
            y_pred=y_test_pred,
            title=f"Predicted vs True RUL ({model_name}, {subset})",
            path=figures_dir / f"pred_vs_true_{model_name}_{subset}.png",
        )
        plot_error_vs_rul(
            y_true=y_test,
            y_pred=y_test_pred,
            title=f"Error vs True RUL ({model_name}, {subset})",
            path=figures_dir / f"error_vs_rul_{model_name}_{subset}.png",
        )

    # ------------------------------------------------------------------
    # 7) Persist artifacts
    # ------------------------------------------------------------------
    metrics_path = metrics_dir / f"baselines_{subset}.json"
    table_path = tables_dir / f"baseline_comparison_{subset}.csv"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    table_df = pd.DataFrame(summary_rows).sort_values("test_RMSE", ascending=True).reset_index(drop=True)
    table_df.to_csv(table_path, index=False)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved table: {table_path}")
    print(f"Saved figures in: {figures_dir}")
    print("Baseline training complete.")
    print(table_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
