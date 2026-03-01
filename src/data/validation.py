"""
Lightweight schema and trajectory validation utilities for CMAPSS-like frames.

Responsibilities:
- Validate minimum structural columns needed by the data pipeline.
- Enforce monotonic cycle progression within each unit trajectory.

Pipeline fit:
- Called early after data loading and before split/scaling/windowing.
- Provides fail-fast diagnostics for malformed data contracts.
"""

from __future__ import annotations

import pandas as pd


def _resolve_unit_column(df: pd.DataFrame) -> str:
    """
    Resolve the unit identifier column name used in the dataframe.
    Supports both legacy 'unit' and current 'unit_id' naming.
    """
    if "unit" in df.columns:
        return "unit"
    if "unit_id" in df.columns:
        return "unit_id"
    raise ValueError("Missing required unit column. Expected one of: ['unit', 'unit_id']")


def validate_basic_schema(df: pd.DataFrame) -> None:
    """
    Validate minimal required schema for CMAPSS-like dataframes.

    Args:
        df: Input dataframe expected to include one unit identifier column
            (``unit`` or ``unit_id``) and ``cycle``.

    Returns:
        None.

    Raises:
        ValueError: If required columns are missing, contain NaN values, or
            if ``cycle`` contains non-positive values.

    Assumptions & leakage boundaries:
        - Validation is structural only; no train/validation/test partitioning.
        - Supports legacy ``unit`` and canonical ``unit_id`` naming.
    """
    unit_col = _resolve_unit_column(df)
    required = {unit_col, "cycle"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[unit_col].isna().any() or df["cycle"].isna().any():
        raise ValueError("Found NaNs in unit or cycle columns.")

    if (df["cycle"] <= 0).any():
        raise ValueError("Cycle must be positive (cycle > 0).")


def validate_unit_monotonic_cycles(df: pd.DataFrame) -> None:
    """
    Validate monotonic cycle progression within each unit trajectory.

    Args:
        df: Input dataframe containing unit and cycle columns.

    Returns:
        None.

    Raises:
        ValueError: If any unit contains non-monotonic ``cycle`` ordering.

    Assumptions & leakage boundaries:
        - Sorting by ``(unit, cycle)`` is applied before monotonic checks.
        - This check prevents invalid temporal sequences from entering
          downstream window generation.
    """
    # Sort defensively to make check reliable.
    unit_col = _resolve_unit_column(df)
    df_sorted = df.sort_values([unit_col, "cycle"])
    bad = df_sorted.groupby(unit_col)["cycle"].apply(lambda s: not s.is_monotonic_increasing)

    if bad.any():
        bad_units = bad[bad].index.tolist()
        raise ValueError(f"Non-monotonic cycle sequences found for units: {bad_units}")
