"""
Basic dataset validation.

Validation is a professional standard:
- catches schema issues early
- prevents silent errors and data leakage
- makes notebooks safer and more repeatable
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
    Validate the minimal expected schema for CMAPSS-like data.

    Expected columns at minimum:
    - unit (engine identifier)
    - cycle (time step / cycle count)

    Parameters
    ----------
    df:
        Input dataset.

    Raises
    ------
    ValueError:
        If required columns are missing or invalid.
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
    Check that cycles increase monotonically within each unit.

    This is a strong sanity check for time series data.

    Parameters
    ----------
    df:
        Dataset containing unit and cycle columns.

    Raises
    ------
    ValueError:
        If cycles are not monotonic within a unit.
    """
    # Sort defensively to make check reliable.
    unit_col = _resolve_unit_column(df)
    df_sorted = df.sort_values([unit_col, "cycle"])
    bad = df_sorted.groupby(unit_col)["cycle"].apply(lambda s: not s.is_monotonic_increasing)

    if bad.any():
        bad_units = bad[bad].index.tolist()
        raise ValueError(f"Non-monotonic cycle sequences found for units: {bad_units}")
