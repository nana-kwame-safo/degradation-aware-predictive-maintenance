"""
Basic dataset validation.

Validation is a professional standard:
- catches schema issues early
- prevents silent errors and data leakage
- makes notebooks safer and more repeatable
"""

from __future__ import annotations

import pandas as pd


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
    required = {"unit", "cycle"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["unit"].isna().any() or df["cycle"].isna().any():
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
    df_sorted = df.sort_values(["unit", "cycle"])
    bad = df_sorted.groupby("unit")["cycle"].apply(lambda s: not s.is_monotonic_increasing)

    if bad.any():
        bad_units = bad[bad].index.tolist()
        raise ValueError(f"Non-monotonic cycle sequences found for units: {bad_units}")
