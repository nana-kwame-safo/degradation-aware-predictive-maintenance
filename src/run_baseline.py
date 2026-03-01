"""
Canonical end-to-end baseline experiment runner for CMAPSS RUL evaluation.

Responsibilities:
- Parse experiment configuration from CLI arguments.
- Execute leakage-safe split/scale/window/tabular preprocessing.
- Train baseline regressors and generate predictions for validation/test.
- Compute reliability, asymmetry, and policy-oriented metrics.
- Persist metrics/tables/figures under the configured results directory.

Pipeline fit:
- Upstream input: labeled CMAPSS train/test frames from ``src.data.data_loader``.
- Downstream outputs: reproducible artifacts consumed by reports and comparisons.

Input/output contract:
- Input tensors after preprocessing follow ``X.shape == (N, W, d)`` and tabular
  matrices ``(N, F)``.
- Outputs include JSON metrics payloads, CSV tables, and PNG figures.

Assumptions & leakage boundaries:
- Train/validation split is unit-based.
- Scaling is fit on train split only and reused for validation/test.
- Evaluation and policy analysis operate on fixed trained-model predictions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Runtime guard for restricted/container environments where threaded BLAS/OpenMP
# can fail due shared-memory limitations.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from src.config import Config
from src.data.data_loader import CMAPSSPaths, load_cmapss_subset
from src.data.preprocessing import (
    fit_scaler,
    make_windows,
    make_window_features,
    transform_scaler,
    unit_train_val_split,
)
from src.evaluation.metrics import (
    error_asymmetry_metrics,
    regression_metrics,
    stratified_metrics_by_rul_bands,
    summarize_unit_metrics,
    unit_level_metrics,
)
from src.evaluation.plots import (
    plot_error_vs_rul,
    plot_policy_timelines,
    plot_pred_vs_true,
    plot_unit_mae_hist,
)
from src.evaluation.policy import policy_metrics
from src.evaluation.reliability_analysis import (
    alert_threshold_sweep,
    weighted_error_cost,
)
from src.models.baseline_models import (
    predict,
    train_elasticnet,
    train_lightgbm,
    train_random_forest,
    train_ridge,
    train_xgboost,
)


def _parse_rul_cap(raw: str) -> Optional[int]:
    value = raw.strip().lower()
    if value in {"none", "null"}:
        return None
    cap = int(value)
    if cap <= 0:
        raise argparse.ArgumentTypeError(
            "rul_cap must be a positive integer or 'none'."
        )
    return cap


def _parse_alert_thresholds(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",")]
    values: List[int] = []
    for p in parts:
        if not p:
            continue
        thr = int(p)
        if thr < 0:
            raise argparse.ArgumentTypeError(
                f"alert thresholds must be >= 0. Received: {thr}"
            )
        values.append(thr)

    if not values:
        raise argparse.ArgumentTypeError(
            "alert_thresholds must include at least one integer value."
        )

    return sorted(set(values))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Milestone 2 CMAPSS baseline experiments."
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
        "--rul_cap",
        type=_parse_rul_cap,
        default=125,
        help="RUL cap as positive integer, or 'none' to disable clipping.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base output directory. metrics/tables/figures subfolders are created here.",
    )
    parser.add_argument(
        "--alert_thresholds",
        type=_parse_alert_thresholds,
        default=[10, 20, 30],
        help="Comma-separated alert thresholds for decision sweep (e.g., 10,20,30).",
    )
    parser.add_argument(
        "--early_weight",
        type=float,
        default=1.0,
        help="Cost weight for underestimation (maintenance too early).",
    )
    parser.add_argument(
        "--late_weight",
        type=float,
        default=2.0,
        help="Cost weight for overestimation (late intervention risk).",
    )
    parser.add_argument(
        "--severe_late_threshold",
        type=float,
        default=10.0,
        help="Late-error cycles above this threshold incur extra penalty.",
    )
    parser.add_argument(
        "--severe_late_multiplier",
        type=float,
        default=2.0,
        help="Multiplier applied to severe late-error cost component.",
    )
    parser.add_argument(
        "--policy_threshold",
        type=int,
        default=20,
        help="Trigger policy threshold on predicted RUL.",
    )
    parser.add_argument(
        "--policy_eol_critical",
        type=int,
        default=5,
        help="True RUL threshold for late-trigger flag.",
    )
    parser.add_argument(
        "--policy_false_alarm_margin",
        type=int,
        default=10,
        help="Margin above threshold used to flag false alarms.",
    )
    parser.add_argument(
        "--policy_include_val",
        action="store_true",
        help="Also run policy evaluation on validation split (default: test only).",
    )
    return parser.parse_args()


def _sensor_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("sensor_")]
    if len(cols) != 21:
        raise ValueError(f"Expected 21 sensor columns, found {len(cols)}.")
    return cols


def _prepare_output_dirs(base_output_dir: Path) -> Tuple[Path, Path, Path]:
    metrics_dir = base_output_dir / "metrics"
    tables_dir = base_output_dir / "tables"
    figures_dir = base_output_dir / "figures"
    for out_dir in (metrics_dir, tables_dir, figures_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir, tables_dir, figures_dir


def _build_window_split(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Tuple[int, int, int], List[str]]:
    x_w, y, metadata = make_windows(
        df=df,
        feature_cols=feature_cols,
        window=window,
        step=step,
        return_meta=True,
    )
    x_tab, feature_names = make_window_features(x_w, feature_cols=feature_cols)

    if metadata.shape[0] != y.shape[0]:
        raise ValueError("Window metadata length mismatch.")

    return x_tab, y, metadata, tuple(int(v) for v in x_w.shape), feature_names


def _eol_rmse_from_rows(rows: List[Dict[str, Optional[float]]]) -> Optional[float]:
    for row in rows:
        if int(row["band_low"]) == 0 and int(row["band_high"]) == 20:
            value = row["RMSE"]
            return None if value is None else float(value)
    return None


def _persistence_predictions(
    metadata: pd.DataFrame,
    y_reference: np.ndarray,
    rul_cap: Optional[int],
    train_mean_rul: float,
) -> np.ndarray:
    if rul_cap is not None:
        preds = np.maximum(
            0.0, float(rul_cap) - metadata["cycle_end"].to_numpy(dtype=float)
        )
        return preds.astype(float)
    return np.full(
        shape=y_reference.shape[0], fill_value=float(train_mean_rul), dtype=float
    )


def _select_boosting_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[Optional[str], Optional[object], Optional[str]]:
    """Try XGBoost first, then LightGBM. Return name/model/error message."""
    try:
        model = train_xgboost(x_train, y_train, seed=seed)
        return "xgboost", model, None
    except Exception as xgb_exc:
        xgb_msg = str(xgb_exc)

    try:
        model = train_lightgbm(x_train, y_train, seed=seed)
        return "lightgbm", model, None
    except Exception as lgbm_exc:
        return (
            None,
            None,
            f"xgboost unavailable ({xgb_msg}); lightgbm unavailable ({lgbm_exc})",
        )


def _validate_metadata_alignment(meta_df: pd.DataFrame, y: np.ndarray, name: str) -> None:
    required = {"unit_id", "cycle_end", "true_rul"}
    missing = sorted(required.difference(meta_df.columns))
    if missing:
        raise ValueError(f"{name} metadata missing required columns: {missing}")
    if int(meta_df.shape[0]) != int(y.shape[0]):
        raise ValueError(
            f"{name} metadata length mismatch: meta={meta_df.shape[0]}, y={y.shape[0]}."
        )
    if not np.allclose(meta_df["true_rul"].to_numpy(dtype=float), y):
        raise ValueError(f"{name} metadata true_rul is not aligned with target y.")


def _select_policy_units(meta_df: pd.DataFrame, seed: int, max_units: int = 5) -> List[int]:
    units = np.asarray(sorted(meta_df["unit_id"].unique()), dtype=int)
    if units.size == 0:
        raise ValueError("Cannot select policy plot units from empty metadata.")
    if units.size <= max_units:
        return [int(v) for v in units.tolist()]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(units, size=max_units, replace=False)
    return sorted(int(v) for v in chosen.tolist())


def _policy_artifact_paths(
    tables_dir: Path,
    metrics_dir: Path,
    subset: str,
    model_name: str,
    split_name: str,
) -> Tuple[Path, Path, Path]:
    if split_name == "test":
        eval_path = tables_dir / f"policy_eval_{subset}_{model_name}.csv"
        summary_path = tables_dir / f"policy_summary_{subset}_{model_name}.csv"
        metrics_path = metrics_dir / f"policy_eval_{subset}_{model_name}.json"
        return eval_path, summary_path, metrics_path

    eval_path = tables_dir / f"policy_eval_{subset}_{model_name}_{split_name}.csv"
    summary_path = tables_dir / f"policy_summary_{subset}_{model_name}_{split_name}.csv"
    metrics_path = metrics_dir / f"policy_eval_{subset}_{model_name}_{split_name}.json"
    return eval_path, summary_path, metrics_path


def main() -> int:
    """
    Execute the baseline experiment pipeline from CLI arguments.

    Args:
        None. Arguments are parsed from process CLI via :func:`_parse_args`.

    Returns:
        int: Process exit code (``0`` on success).

    Raises:
        FileNotFoundError: If expected CMAPSS raw directory is unavailable.
        ValueError: If preprocessing contracts, alignment checks, or metric
            preconditions fail.

    Assumptions & leakage boundaries:
        - Train/validation units are disjoint.
        - Feature scaling uses training statistics only.
        - Window metadata and targets remain row-aligned throughout evaluation.
    """
    args = _parse_args()
    cfg = Config()

    subset = args.subset.strip().upper()
    output_root = Path(args.output_dir)
    rul_cap: Optional[int] = args.rul_cap
    alert_thresholds = [int(v) for v in args.alert_thresholds]

    raw_dir = Path(cfg.cmapss_raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"CMAPSS directory not found: {raw_dir}. Place files under data/raw/cmapss/."
        )

    train_df, test_df = load_cmapss_subset(
        paths=CMAPSSPaths(root_dir=raw_dir),
        subset=subset,
        rul_cap=rul_cap,
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

    x_train, y_train, train_meta, train_w_shape, feature_names = _build_window_split(
        df=train_scaled,
        feature_cols=sensor_cols,
        window=int(args.window),
        step=int(args.step),
    )
    x_val, y_val, val_meta, val_w_shape, feature_names_val = _build_window_split(
        df=val_scaled,
        feature_cols=sensor_cols,
        window=int(args.window),
        step=int(args.step),
    )
    x_test, y_test, test_meta, test_w_shape, feature_names_test = _build_window_split(
        df=test_scaled,
        feature_cols=sensor_cols,
        window=int(args.window),
        step=int(args.step),
    )

    if feature_names != feature_names_val or feature_names != feature_names_test:
        raise ValueError("Tabular feature definitions are inconsistent across splits.")
    _validate_metadata_alignment(train_meta, y_train, name="train")
    _validate_metadata_alignment(val_meta, y_val, name="validation")
    _validate_metadata_alignment(test_meta, y_test, name="test")

    metrics_dir, tables_dir, figures_dir = _prepare_output_dirs(output_root)

    models: Dict[str, object] = {
        "ridge": train_ridge(x_train, y_train, seed=int(args.seed)),
        "elasticnet": train_elasticnet(x_train, y_train, seed=int(args.seed)),
        "random_forest": train_random_forest(x_train, y_train, seed=int(args.seed)),
    }

    boosting_name, boosting_model, boosting_error = _select_boosting_model(
        x_train=x_train,
        y_train=y_train,
        seed=int(args.seed),
    )
    if boosting_name is not None and boosting_model is not None:
        models[boosting_name] = boosting_model
    elif boosting_error is not None:
        print(f"[INFO] Boosting baseline skipped: {boosting_error}")

    train_mean_rul = float(np.mean(y_train))

    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name, model in models.items():
        predictions[model_name] = {
            "validation": predict(model, x_val),
            "test": predict(model, x_test),
        }

    predictions["persistence"] = {
        "validation": _persistence_predictions(
            metadata=val_meta,
            y_reference=y_val,
            rul_cap=rul_cap,
            train_mean_rul=train_mean_rul,
        ),
        "test": _persistence_predictions(
            metadata=test_meta,
            y_reference=y_test,
            rul_cap=rul_cap,
            train_mean_rul=train_mean_rul,
        ),
    }

    band_upper = (
        int(rul_cap)
        if rul_cap is not None
        else int(max(51, np.ceil(max(y_val.max(), y_test.max()))))
    )
    if band_upper < 51:
        band_upper = 51
    bands = [(0, 20), (21, 50), (51, band_upper)]
    policy_plot_units = _select_policy_units(test_meta, seed=int(args.seed), max_units=5)
    policy_splits = ["test"]
    if args.policy_include_val:
        policy_splits.append("val")

    overall_payload: Dict[str, object] = {}
    stratified_payload: Dict[str, object] = {}
    asymmetry_payload: Dict[str, object] = {}
    unit_summary_payload: Dict[str, object] = {}
    decision_payload: Dict[str, object] = {}
    policy_payload: Dict[str, object] = {}

    comparison_rows: List[Dict[str, object]] = []
    stratified_rows: List[Dict[str, object]] = []
    asymmetry_rows: List[Dict[str, object]] = []
    unit_summary_rows: List[Dict[str, object]] = []
    decision_threshold_rows: List[Dict[str, object]] = []
    decision_cost_rows: List[Dict[str, object]] = []

    for model_name, pred in predictions.items():
        y_val_pred = pred["validation"]
        y_test_pred = pred["test"]

        val_overall = regression_metrics(y_val, y_val_pred)
        test_overall = regression_metrics(y_test, y_test_pred)

        val_strat = stratified_metrics_by_rul_bands(y_val, y_val_pred, bands=bands)
        test_strat = stratified_metrics_by_rul_bands(y_test, y_test_pred, bands=bands)

        val_asym = error_asymmetry_metrics(
            y_val, y_val_pred, eol_band=(0, 20), severe_threshold=10.0
        )
        test_asym = error_asymmetry_metrics(
            y_test, y_test_pred, eol_band=(0, 20), severe_threshold=10.0
        )
        val_threshold_sweep = alert_threshold_sweep(
            y_true=y_val,
            y_pred=y_val_pred,
            thresholds=alert_thresholds,
        )
        test_threshold_sweep = alert_threshold_sweep(
            y_true=y_test,
            y_pred=y_test_pred,
            thresholds=alert_thresholds,
        )
        val_weighted_cost = weighted_error_cost(
            y_true=y_val,
            y_pred=y_val_pred,
            early_weight=float(args.early_weight),
            late_weight=float(args.late_weight),
            severe_late_threshold=float(args.severe_late_threshold),
            severe_late_multiplier=float(args.severe_late_multiplier),
            eol_band=(0, 20),
        )
        test_weighted_cost = weighted_error_cost(
            y_true=y_test,
            y_pred=y_test_pred,
            early_weight=float(args.early_weight),
            late_weight=float(args.late_weight),
            severe_late_threshold=float(args.severe_late_threshold),
            severe_late_multiplier=float(args.severe_late_multiplier),
            eol_band=(0, 20),
        )

        unit_df = unit_level_metrics(
            y_true=y_test,
            y_pred=y_test_pred,
            unit_ids=test_meta["unit_id"].to_numpy(dtype=int),
        )
        unit_path = tables_dir / f"unit_metrics_{model_name}_{subset}.csv"
        unit_df.to_csv(unit_path, index=False)
        unit_summary = summarize_unit_metrics(unit_df)
        plot_unit_mae_hist(
            unit_mae_df=unit_df,
            path=figures_dir / f"unit_mae_hist_{model_name}_{subset}.png",
            title=f"Unit MAE Histogram ({model_name}, {subset}, test)",
        )

        policy_by_split: Dict[str, object] = {}
        for split_name in policy_splits:
            if split_name == "test":
                split_meta = test_meta
                split_y_true = y_test
                split_y_pred = y_test_pred
            elif split_name == "val":
                split_meta = val_meta
                split_y_true = y_val
                split_y_pred = y_val_pred
            else:
                raise ValueError(f"Unsupported policy split: {split_name}")

            policy_unit_df, policy_summary = policy_metrics(
                meta_df=split_meta,
                y_true=split_y_true,
                y_pred=split_y_pred,
                threshold=int(args.policy_threshold),
                eol_critical=int(args.policy_eol_critical),
                false_alarm_margin=int(args.policy_false_alarm_margin),
            )

            (
                policy_eval_path,
                policy_summary_path,
                policy_metrics_path,
            ) = _policy_artifact_paths(
                tables_dir=tables_dir,
                metrics_dir=metrics_dir,
                subset=subset,
                model_name=model_name,
                split_name=split_name,
            )

            policy_unit_df.to_csv(policy_eval_path, index=False)
            pd.DataFrame(
                [
                    {
                        "model": model_name,
                        "split": split_name,
                        **policy_summary,
                    }
                ]
            ).to_csv(policy_summary_path, index=False)
            policy_metrics_payload = {
                "model": model_name,
                "split": split_name,
                "summary": policy_summary,
            }
            policy_metrics_path.write_text(
                json.dumps(policy_metrics_payload, indent=2),
                encoding="utf-8",
            )

            policy_by_split[split_name] = {
                "summary": policy_summary,
                "unit_rows": int(policy_unit_df.shape[0]),
                "paths": {
                    "policy_eval": str(policy_eval_path),
                    "policy_summary": str(policy_summary_path),
                    "policy_metrics": str(policy_metrics_path),
                },
            }

        overall_payload[model_name] = {
            "validation": val_overall,
            "test": test_overall,
        }
        stratified_payload[model_name] = {
            "validation": val_strat,
            "test": test_strat,
        }
        asymmetry_payload[model_name] = {
            "validation": val_asym,
            "test": test_asym,
        }
        unit_summary_payload[model_name] = unit_summary
        policy_payload[model_name] = policy_by_split
        decision_payload[model_name] = {
            "validation": {
                "alert_threshold_sweep": val_threshold_sweep,
                "weighted_error_cost": val_weighted_cost,
            },
            "test": {
                "alert_threshold_sweep": test_threshold_sweep,
                "weighted_error_cost": test_weighted_cost,
            },
        }

        for split_name, split_rows in (("validation", val_strat), ("test", test_strat)):
            for row in split_rows:
                stratified_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        **row,
                    }
                )

        for split_name, asym in (("validation", val_asym), ("test", test_asym)):
            asymmetry_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    **asym,
                }
            )
        for split_name, threshold_rows in (
            ("validation", val_threshold_sweep),
            ("test", test_threshold_sweep),
        ):
            for row in threshold_rows:
                decision_threshold_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        **row,
                    }
                )
        for split_name, cost_row in (
            ("validation", val_weighted_cost),
            ("test", test_weighted_cost),
        ):
            decision_cost_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    **cost_row,
                }
            )

        unit_summary_rows.append(
            {
                "model": model_name,
                **unit_summary,
                "worst_10pct_unit_ids": "|".join(
                    str(v) for v in unit_summary["worst_10pct_unit_ids"]
                ),
            }
        )

        eol_test_rmse = _eol_rmse_from_rows(test_strat)
        test_policy_summary = policy_by_split["test"]["summary"]
        comparison_rows.append(
            {
                "model": model_name,
                "val_MAE": float(val_overall["MAE"]),
                "val_RMSE": float(val_overall["RMSE"]),
                "test_MAE": float(test_overall["MAE"]),
                "test_RMSE": float(test_overall["RMSE"]),
                "test_RMSE_0_20": eol_test_rmse,
                "test_bias_eol": test_asym["bias_eol"],
                "trigger_rate": test_policy_summary["trigger_rate"],
                "false_alarm_rate": test_policy_summary["false_alarm_rate"],
                "missed_trigger_rate": test_policy_summary["missed_trigger_rate"],
                "median_lead_time": test_policy_summary["lead_time_median"],
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
        plot_policy_timelines(
            meta_df=test_meta,
            y_true=y_test,
            y_pred=y_test_pred,
            selected_units=policy_plot_units,
            threshold=int(args.policy_threshold),
            path=figures_dir / f"policy_timeline_{model_name}_{subset}.png",
            title=(
                f"Policy Timelines ({model_name}, {subset}, test, "
                f"threshold={int(args.policy_threshold)})"
            ),
        )

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values("test_RMSE", ascending=True)
        .reset_index(drop=True)
    )
    stratified_df = pd.DataFrame(stratified_rows)
    asymmetry_df = pd.DataFrame(asymmetry_rows)
    unit_summary_df = (
        pd.DataFrame(unit_summary_rows)
        .sort_values("mae_mean", ascending=True)
        .reset_index(drop=True)
    )
    decision_threshold_df = pd.DataFrame(decision_threshold_rows)
    decision_cost_df = pd.DataFrame(decision_cost_rows)

    metrics_path = metrics_dir / f"baselines_{subset}.json"
    comparison_path = tables_dir / f"baseline_comparison_{subset}.csv"
    stratified_path = tables_dir / f"baseline_stratified_metrics_{subset}.csv"
    asymmetry_path = tables_dir / f"baseline_error_asymmetry_{subset}.csv"
    unit_summary_path = tables_dir / f"baseline_unit_summary_{subset}.csv"
    decision_threshold_path = tables_dir / f"baseline_alert_thresholds_{subset}.csv"
    decision_cost_path = tables_dir / f"baseline_weighted_cost_{subset}.csv"

    comparison_df.to_csv(comparison_path, index=False)
    stratified_df.to_csv(stratified_path, index=False)
    asymmetry_df.to_csv(asymmetry_path, index=False)
    unit_summary_df.to_csv(unit_summary_path, index=False)
    decision_threshold_df.to_csv(decision_threshold_path, index=False)
    decision_cost_df.to_csv(decision_cost_path, index=False)

    metrics_payload: Dict[str, object] = {
        "config": {
            "subset": subset,
            "window": int(args.window),
            "step": int(args.step),
            "val_fraction": float(args.val_fraction),
            "seed": int(args.seed),
            "rul_cap": rul_cap,
            "output_dir": str(output_root),
            "bands": [{"low": low, "high": high} for low, high in bands],
            "alert_thresholds": alert_thresholds,
            "early_weight": float(args.early_weight),
            "late_weight": float(args.late_weight),
            "severe_late_threshold": float(args.severe_late_threshold),
            "severe_late_multiplier": float(args.severe_late_multiplier),
            "policy_threshold": int(args.policy_threshold),
            "policy_eol_critical": int(args.policy_eol_critical),
            "policy_false_alarm_margin": int(args.policy_false_alarm_margin),
            "policy_splits": policy_splits,
            "policy_plot_units": policy_plot_units,
        },
        "shapes": {
            "train_df": list(train_df.shape),
            "val_df": list(val_split.shape),
            "test_df": list(test_df.shape),
            "train_windows": list(train_w_shape),
            "val_windows": list(val_w_shape),
            "test_windows": list(test_w_shape),
            "train_features": list(x_train.shape),
            "val_features": list(x_val.shape),
            "test_features": list(x_test.shape),
            "feature_count": len(feature_names),
            "train_metadata_rows": int(train_meta.shape[0]),
            "val_metadata_rows": int(val_meta.shape[0]),
            "test_metadata_rows": int(test_meta.shape[0]),
        },
        "overall": overall_payload,
        "stratified": stratified_payload,
        "asymmetry": asymmetry_payload,
        "unit_summary": unit_summary_payload,
        "policy": policy_payload,
        "decision": decision_payload,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    concise = comparison_df[
        [
            "model",
            "test_RMSE",
            "test_RMSE_0_20",
            "trigger_rate",
            "false_alarm_rate",
            "median_lead_time",
        ]
    ].copy()
    concise = concise.rename(columns={"test_RMSE_0_20": "EOL_RMSE(0-20)"})

    print("Baseline run complete")
    print(f"subset={subset} | output_dir={output_root}")
    print(f"saved metrics:   {metrics_path}")
    print(f"saved comparison:{comparison_path}")
    print(f"saved stratified:{stratified_path}")
    print(f"saved asymmetry: {asymmetry_path}")
    print(f"saved unit sum.: {unit_summary_path}")
    print(f"saved thresholds:{decision_threshold_path}")
    print(f"saved cost tbl.:{decision_cost_path}")
    print("saved policy:    policy_eval_<subset>_<model>.csv/.json and policy_summary_<subset>_<model>.csv")
    print(f"saved figures:   {figures_dir}")
    print("\nConcise model summary (test policy-focused):")
    print(concise.to_string(index=False, justify="center"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
