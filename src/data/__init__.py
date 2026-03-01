"""
Data package public surface for CMAPSS ingestion and preprocessing.

This package groups:
- loading/label construction contracts (``data_loader``)
- leakage-safe preprocessing primitives (``preprocessing``)
- legacy compatibility wrappers explicitly kept for migration support.
"""

from .data_loader import CMAPSSPaths, cmapss_summary, load_cmapss_subset
from .preprocessing import (
    fit_scaler,
    generate_unit_windows,
    make_window_features,
    make_windows,
    scale_sensor_columns,
    split_by_unit,
    transform_scaler,
    unit_train_val_split,
    build_tabular_baseline_features,
)

__all__ = [
    "CMAPSSPaths",
    "load_cmapss_subset",
    "cmapss_summary",
    "unit_train_val_split",
    "fit_scaler",
    "transform_scaler",
    "make_windows",
    "make_window_features",
    "split_by_unit",
    "scale_sensor_columns",
    "generate_unit_windows",
    "build_tabular_baseline_features",
]
