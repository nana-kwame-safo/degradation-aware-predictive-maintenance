"""
CMAPSS data loading and target construction utilities.

Engineering assumptions:
- Input files follow the standard NASA CMAPSS naming convention.
- Parsing is fail-fast: malformed schema or subset mismatch raises immediately.
- RUL construction is explicit for both train (run-to-failure) and test
  (partial trajectories merged with RUL_FD00x end-of-sequence targets).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

VALID_SUBSETS = {"FD001", "FD002", "FD003", "FD004"}


def _normalize_subset(subset: str) -> str:
    subset_norm = subset.strip().upper()
    if subset_norm not in VALID_SUBSETS:
        raise ValueError(
            f"Invalid subset '{subset}'. Expected one of: {sorted(VALID_SUBSETS)}"
        )
    return subset_norm


# -----------------------------------------------------------------------------
# Column definitions
# -----------------------------------------------------------------------------


def cmapss_feature_columns() -> Tuple[List[str], List[str], List[str]]:
    """Return operating-setting columns, sensor columns, and their concatenation."""
    settings = [f"op_setting_{i}" for i in range(1, 4)]
    sensors = [f"sensor_{i}" for i in range(1, 22)]
    return settings, sensors, settings + sensors


def cmapss_all_columns() -> List[str]:
    """Return the full CMAPSS raw schema (26 columns)."""
    settings, sensors, _ = cmapss_feature_columns()
    return ["unit_id", "cycle"] + settings + sensors


# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CMAPSSPaths:
    """
    Strongly-typed CMAPSS path container.

    Expected repository layout:
        data/raw/cmapss/
          train_FD00x.txt
          test_FD00x.txt
          RUL_FD00x.txt
    """

    root_dir: Path

    def validate_root_dir(self) -> None:
        """Validate that the configured CMAPSS root directory exists and is usable."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"CMAPSS root directory not found: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"CMAPSS root path is not a directory: {self.root_dir}"
            )
        if self.root_dir.name.lower() != "cmapss":
            raise ValueError(
                "CMAPSS root_dir should point to the 'cmapss' directory "
                f"(for example: data/raw/cmapss). Received: {self.root_dir}"
            )

    def train_file(self, subset: str) -> Path:
        return self.root_dir / f"train_{_normalize_subset(subset)}.txt"

    def test_file(self, subset: str) -> Path:
        return self.root_dir / f"test_{_normalize_subset(subset)}.txt"

    def rul_file(self, subset: str) -> Path:
        return self.root_dir / f"RUL_{_normalize_subset(subset)}.txt"


# -----------------------------------------------------------------------------
# Low-level file readers
# -----------------------------------------------------------------------------


def _read_cmapss_txt(path: Path) -> pd.DataFrame:
    """
    Read one CMAPSS train/test text file with robust whitespace parsing.

    Defensive checks:
    - file must exist
    - column count must match expected CMAPSS schema
    - unit_id and cycle must be positive integers
    """
    if not path.exists():
        raise FileNotFoundError(f"CMAPSS file not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")

    expected_cols = cmapss_all_columns()
    if df.shape[1] != len(expected_cols):
        raise ValueError(
            f"Unexpected column count in {path}. "
            f"Expected {len(expected_cols)}, got {df.shape[1]}."
        )

    df.columns = expected_cols

    # Parse all numeric columns explicitly so malformed tokens fail early.
    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    if (df["unit_id"] <= 0).any():
        raise ValueError(f"Invalid unit_id values in {path}: unit_id must be > 0.")
    if (df["cycle"] <= 0).any():
        raise ValueError(f"Invalid cycle values in {path}: cycle must be > 0.")

    return df


def _read_rul_targets(path: Path) -> pd.Series:
    """
    Read CMAPSS test-end RUL targets from RUL_FD00x.txt.

    The file contains one row per test unit in ascending unit order.
    The value is RUL at the final observed cycle for that unit.
    """
    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")

    rul_df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    rul_df = rul_df.dropna(axis=1, how="all")
    if rul_df.shape[1] != 1:
        raise ValueError(
            f"Unexpected RUL file shape in {path}: expected 1 column, got {rul_df.shape[1]}."
        )

    rul = pd.to_numeric(rul_df.iloc[:, 0], errors="raise").astype(int)
    if rul.empty:
        raise ValueError(f"RUL file is empty: {path}")
    if (rul < 0).any():
        raise ValueError(f"RUL file contains negative values: {path}")

    rul.index = np.arange(1, len(rul) + 1)
    rul.name = "rul_end"
    return rul


# -----------------------------------------------------------------------------
# Dataset integrity checks
# -----------------------------------------------------------------------------


def validate_cmapss_dataframe(df: pd.DataFrame, name: str = "cmapss") -> None:
    """
    Validate CMAPSS frame integrity.

    Required invariants:
    - full raw CMAPSS schema is present
    - unit_id and cycle are positive
    - (unit_id, cycle) keys are unique
    - cycle order is strictly increasing within each unit
    """
    if df.empty:
        raise ValueError(f"[{name}] dataframe is empty.")

    required = set(cmapss_all_columns())
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")

    if df[cmapss_all_columns()].isna().any().any():
        raise ValueError(f"[{name}] Found NaN values in required columns.")

    if (df["unit_id"] <= 0).any():
        raise ValueError(f"[{name}] unit_id must be > 0.")
    if (df["cycle"] <= 0).any():
        raise ValueError(f"[{name}] cycle must be > 0.")

    if df.duplicated(subset=["unit_id", "cycle"]).any():
        raise ValueError(f"[{name}] Duplicate (unit_id, cycle) rows detected.")

    bad_units: List[int] = []
    ordered = df.sort_values(["unit_id", "cycle"])
    for uid, g in ordered.groupby("unit_id", sort=True):
        diffs = g["cycle"].diff().dropna()
        if not (diffs > 0).all():
            bad_units.append(int(uid))
            if len(bad_units) >= 5:
                break
    if bad_units:
        raise ValueError(
            f"[{name}] Cycle sequence is not strictly increasing for units: {bad_units}"
        )


# -----------------------------------------------------------------------------
# Target construction
# -----------------------------------------------------------------------------


def add_train_rul(
    df_train: pd.DataFrame, rul_cap: Optional[int] = None
) -> pd.DataFrame:
    """
    Add per-row train RUL labels for run-to-failure trajectories.

    Assumption:
        Each training unit runs until failure.

    Formula:
        RUL(t) = max_cycle(unit_id) - cycle(t)
    """
    required = {"unit_id", "cycle"}
    missing = sorted(required.difference(df_train.columns))
    if missing:
        raise ValueError(f"add_train_rul missing required columns: {missing}")

    df = df_train.copy()
    max_cycle = df.groupby("unit_id")["cycle"].max()
    df = df.join(max_cycle.rename("max_cycle"), on="unit_id")
    df["rul"] = (df["max_cycle"] - df["cycle"]).astype(int)
    df = df.drop(columns=["max_cycle"])

    if (df["rul"] < 0).any():
        raise ValueError("add_train_rul produced negative RUL values.")

    if rul_cap is not None:
        if int(rul_cap) <= 0:
            raise ValueError(f"rul_cap must be > 0 when provided. Received: {rul_cap}")
        df["rul"] = df["rul"].clip(upper=int(rul_cap))

    return df


def add_test_rul(
    df_test: pd.DataFrame, rul_end: pd.Series, rul_cap: Optional[int] = None
) -> pd.DataFrame:
    """
    Add per-row test RUL labels using RUL_FD00x.txt values.

    Assumptions:
    - test trajectories are truncated before failure
    - RUL_FD00x has one value per test unit in ascending unit_id order

    For each unit:
        last_cycle = max observed cycle in test file
        rul_end = cycles remaining after last observed cycle (from RUL file)
        RUL(cycle) = (last_cycle - cycle) + rul_end
    """
    required = {"unit_id", "cycle"}
    missing = sorted(required.difference(df_test.columns))
    if missing:
        raise ValueError(f"add_test_rul missing required columns: {missing}")

    if rul_end.index.duplicated().any():
        raise ValueError("add_test_rul received duplicated unit indices in rul_end.")

    df = df_test.copy()

    test_units = sorted(int(u) for u in df["unit_id"].unique())
    missing_targets = [u for u in test_units if u not in rul_end.index]
    extra_targets = [int(u) for u in rul_end.index if int(u) not in test_units]
    if missing_targets:
        raise ValueError(
            "RUL targets missing for test units: "
            f"{missing_targets}. Check subset alignment between test and RUL files."
        )
    if extra_targets:
        raise ValueError(
            "RUL file has units not present in test data: "
            f"{extra_targets}. Check subset alignment between test and RUL files."
        )

    last_cycle = df.groupby("unit_id")["cycle"].max().rename("last_cycle")
    df = df.join(last_cycle, on="unit_id")
    df["rul_end"] = df["unit_id"].map(rul_end).astype(int)
    df["rul"] = (df["last_cycle"] - df["cycle"] + df["rul_end"]).astype(int)
    df = df.drop(columns=["last_cycle", "rul_end"])

    if (df["rul"] < 0).any():
        raise ValueError("add_test_rul produced negative RUL values.")

    if rul_cap is not None:
        if int(rul_cap) <= 0:
            raise ValueError(f"rul_cap must be > 0 when provided. Received: {rul_cap}")
        df["rul"] = df["rul"].clip(upper=int(rul_cap))

    return df


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def load_cmapss_subset(
    paths: CMAPSSPaths,
    subset: str = "FD001",
    rul_cap: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load one CMAPSS subset and return train/test dataframes with RUL labels.

    Returns:
        train_df: CMAPSS train rows with added `rul`
        test_df: CMAPSS test rows with added `rul` reconstructed via RUL_FD00x
    """
    subset_norm = _normalize_subset(subset)
    paths.validate_root_dir()

    train_path = paths.train_file(subset_norm)
    test_path = paths.test_file(subset_norm)
    rul_path = paths.rul_file(subset_norm)

    df_train = _read_cmapss_txt(train_path)
    df_test = _read_cmapss_txt(test_path)
    rul_end = _read_rul_targets(rul_path)

    validate_cmapss_dataframe(df_train, name=f"train_{subset_norm}")
    validate_cmapss_dataframe(df_test, name=f"test_{subset_norm}")

    n_test_units = int(df_test["unit_id"].nunique())
    if len(rul_end) != n_test_units:
        raise ValueError(
            f"RUL target count mismatch for {subset_norm}: "
            f"expected {n_test_units} units, got {len(rul_end)} targets."
        )

    df_train = add_train_rul(df_train, rul_cap=rul_cap)
    df_test = add_test_rul(df_test, rul_end=rul_end, rul_cap=rul_cap)

    if df_train["rul"].isna().any() or df_test["rul"].isna().any():
        raise ValueError(
            f"RUL construction produced NaN values for subset {subset_norm}."
        )

    return df_train, df_test


def cmapss_summary(df: pd.DataFrame) -> Dict[str, Optional[int]]:
    """Return a compact dataframe summary for logging and diagnostics."""
    return {
        "n_rows": int(len(df)),
        "n_units": int(df["unit_id"].nunique()),
        "cycle_min": int(df["cycle"].min()),
        "cycle_max": int(df["cycle"].max()),
        "rul_min": int(df["rul"].min()) if "rul" in df.columns else None,
        "rul_max": int(df["rul"].max()) if "rul" in df.columns else None,
    }
