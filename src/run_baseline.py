"""
Module entry point for Milestone 1 baseline experiments.

This runner executes the full leakage-safe baseline path and writes artifacts to
an output directory suitable for reproducible reporting.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Runtime guard for restricted/container environments where threaded BLAS/OpenMP
# can fail due shared-memory limitations.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

from src.config import Config
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
    parser = argparse.ArgumentParser(
        description="Run Milestone 1 CMAPSS baseline experiments."
    )
    parser.add_argument(
        "--subset", type=str, default="FD001", help="CMAPSS subset (FD001..FD004)."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Window length for sequence construction.",
    )
    parser.add_argument("--step", type=int, default=1, help="Window stride.")
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Validation split fraction by unit.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--rul_cap", type=int, default=125, help="Optional RUL clip value."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base output directory. Metrics/tables/figures subfolders are created under this path.",
    )
    return parser.parse_args()


def _sensor_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("sensor_")]
    if len(cols) != 21:
        raise ValueError(f"Expected 21 sensor columns, found {len(cols)}.")
    return cols


def _window_end_unit_ids(df: pd.DataFrame, window: int, step: int) -> List[int]:
    """Reconstruct unit_ids aligned with make_windows output ordering."""
    unit_ids: List[int] = []
    ordered = df.sort_values(["unit_id", "cycle"]).copy()
    for uid, g in ordered.groupby("unit_id", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < window:
            continue
        for _ in range(window - 1, len(g), step):
            unit_ids.append(int(uid))
    return unit_ids


def _prepare_output_dirs(base_output_dir: Path) -> Tuple[Path, Path, Path]:
    metrics_dir = base_output_dir / "metrics"
    tables_dir = base_output_dir / "tables"
    figures_dir = base_output_dir / "figures"
    for out_dir in (metrics_dir, tables_dir, figures_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir, tables_dir, figures_dir


def main() -> int:
    args = _parse_args()
    cfg = Config()

    subset = args.subset.strip().upper()
    output_root = Path(args.output_dir)

    raw_dir = Path(cfg.cmapss_raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"CMAPSS directory not found: {raw_dir}. Place files under data/raw/cmapss/."
        )

    train_df, test_df = load_cmapss_subset(
        paths=CMAPSSPaths(root_dir=raw_dir),
        subset=subset,
        rul_cap=int(args.rul_cap),
    )

    sensor_cols = _sensor_columns(train_df)

    train_split, val_split = unit_train_val_split(
        df=train_df,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    overlap = set(train_split["unit_id"].unique()).intersection(
        set(val_split["unit_id"].unique())
    )
    if overlap:
        raise ValueError(
            f"Unit leakage detected between train and validation: {sorted(overlap)}"
        )

    scaler = fit_scaler(train_df=train_split, feature_cols=sensor_cols)
    train_scaled = transform_scaler(
        train_split, scaler=scaler, feature_cols=sensor_cols
    )
    val_scaled = transform_scaler(val_split, scaler=scaler, feature_cols=sensor_cols)
    test_scaled = transform_scaler(test_df, scaler=scaler, feature_cols=sensor_cols)

    x_train_w, y_train = make_windows(
        train_scaled,
        feature_cols=sensor_cols,
        window=int(args.window),
        step=int(args.step),
    )
    x_val_w, y_val = make_windows(
        val_scaled,
        feature_cols=sensor_cols,
        window=int(args.window),
        step=int(args.step),
    )
    x_test_w, y_test = make_windows(
        test_scaled,
        feature_cols=sensor_cols,
        window=int(args.window),
        step=int(args.step),
    )

    x_train_tab, feature_names = make_window_features(x_train_w)
    x_val_tab, feature_names_val = make_window_features(x_val_w)
    x_test_tab, feature_names_test = make_window_features(x_test_w)

    if feature_names != feature_names_val or feature_names != feature_names_test:
        raise ValueError(
            "Tabular feature definitions are inconsistent across train/val/test."
        )

    metrics_dir, tables_dir, figures_dir = _prepare_output_dirs(output_root)

    models = {
        "ridge": train_ridge(x_train_tab, y_train),
        "elasticnet": train_elasticnet(x_train_tab, y_train),
        "random_forest": train_random_forest(x_train_tab, y_train, seed=int(args.seed)),
    }

    rul_bins = [0, 30, 60, int(args.rul_cap)]
    val_unit_ids = _window_end_unit_ids(
        val_scaled, window=int(args.window), step=int(args.step)
    )
    test_unit_ids = _window_end_unit_ids(
        test_scaled, window=int(args.window), step=int(args.step)
    )

    if len(val_unit_ids) != len(y_val):
        raise ValueError(
            "Validation unit_id reconstruction does not match y_val length."
        )
    if len(test_unit_ids) != len(y_test):
        raise ValueError("Test unit_id reconstruction does not match y_test length.")

    summary_rows: List[Dict[str, float]] = []
    metrics_payload: Dict[str, object] = {
        "config": {
            "subset": subset,
            "window": int(args.window),
            "step": int(args.step),
            "val_fraction": float(args.val_fraction),
            "seed": int(args.seed),
            "rul_cap": int(args.rul_cap),
            "output_dir": str(output_root),
            "rul_bins": rul_bins,
        },
        "shapes": {
            "train_df": list(train_df.shape),
            "val_df": list(val_split.shape),
            "test_df": list(test_df.shape),
            "train_windows": list(x_train_w.shape),
            "val_windows": list(x_val_w.shape),
            "test_windows": list(x_test_w.shape),
            "train_features": list(x_train_tab.shape),
            "val_features": list(x_val_tab.shape),
            "test_features": list(x_test_tab.shape),
            "feature_count": len(feature_names),
        },
        "models": {},
    }

    for model_name, model in models.items():
        y_val_pred = predict(model, x_val_tab)
        y_test_pred = predict(model, x_test_tab)

        val_overall = regression_metrics(y_val, y_val_pred)
        test_overall = regression_metrics(y_test, y_test_pred)

        metrics_payload["models"][model_name] = {
            "validation": {
                "overall": val_overall,
                "by_rul_band": stratified_metrics_by_rul_bins(
                    y_val, y_val_pred, bins=rul_bins
                ),
                "unit_summary": unit_level_error_summary(
                    y_val, y_val_pred, unit_ids=val_unit_ids
                ),
            },
            "test": {
                "overall": test_overall,
                "by_rul_band": stratified_metrics_by_rul_bins(
                    y_test, y_test_pred, bins=rul_bins
                ),
                "unit_summary": unit_level_error_summary(
                    y_test, y_test_pred, unit_ids=test_unit_ids
                ),
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

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("test_RMSE", ascending=True)
        .reset_index(drop=True)
    )

    metrics_path = metrics_dir / f"baselines_{subset}.json"
    table_path = tables_dir / f"baseline_comparison_{subset}.csv"

    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    summary_df.to_csv(table_path, index=False)

    print("Baseline run complete")
    print(f"subset={subset} | output_dir={output_root}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved table:   {table_path}")
    print(f"saved figures: {figures_dir}")
    print("\nModel summary (sorted by test_RMSE):")
    print(summary_df.to_string(index=False, justify="center"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
