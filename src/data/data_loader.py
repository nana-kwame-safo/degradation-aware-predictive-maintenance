"""
CMAPSS ingestion and label-construction module for the baseline pipeline.

Responsibilities:
- Load raw CMAPSS train/test text files from the repository data layout.
- Validate schema and trajectory integrity before downstream preprocessing.
- Construct row-level RUL targets for both train and test splits.

Pipeline fit:
- Upstream: raw files in ``data/raw/cmapss``.
- Downstream: validated, labeled ``pd.DataFrame`` objects consumed by
  leakage-safe splitting/scaling/windowing in ``src.data.preprocessing``.

Input/output contract:
- Input files follow NASA CMAPSS naming:
  ``train_FD00x.txt``, ``test_FD00x.txt``, ``RUL_FD00x.txt``.
- Output frames include CMAPSS raw columns plus ``rul``.

Assumptions & leakage boundaries:
- Train trajectories are run-to-failure and are labeled by remaining cycles.
- Test trajectories are truncated and require ``RUL_FD00x`` end targets.
- This module does not split or scale; it only produces validated labeled data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import cmapss_columns

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
    """
    Deprecated alias for ``src.config.cmapss_columns``.

    Args:
        None.

    Returns:
        Tuple[List[str], List[str], List[str]]: Operating-setting columns,
        sensor columns, and concatenated feature columns.

    Raises:
        None directly. Emits ``DeprecationWarning``.
    """
    warnings.warn(
        "cmapss_feature_columns() is deprecated. "
        "Use src.config.cmapss_columns() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cmapss_columns()


def cmapss_all_columns() -> List[str]:
    """
    Return the full CMAPSS raw schema.

    Args:
        None.

    Returns:
        List[str]: Ordered column list with shape-defining keys:
        ``["unit_id", "cycle"] + op_settings(3) + sensors(21)``.

    Raises:
        None.
    """
    settings, sensors, _ = cmapss_columns()
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
        """
        Validate that the configured CMAPSS root directory is usable.

        Args:
            None.

        Returns:
            None.

        Raises:
            FileNotFoundError: If root directory does not exist.
            NotADirectoryError: If root path is not a directory.
            ValueError: If directory name is not ``cmapss``.
        """
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
        """
        Resolve train file path for a CMAPSS subset.

        Args:
            subset: Subset identifier (``FD001``..``FD004``).

        Returns:
            Path: Expected train file path.

        Raises:
            ValueError: If subset is invalid.
        """
        return self.root_dir / f"train_{_normalize_subset(subset)}.txt"

    def test_file(self, subset: str) -> Path:
        """
        Resolve test file path for a CMAPSS subset.

        Args:
            subset: Subset identifier (``FD001``..``FD004``).

        Returns:
            Path: Expected test file path.

        Raises:
            ValueError: If subset is invalid.
        """
        return self.root_dir / f"test_{_normalize_subset(subset)}.txt"

    def rul_file(self, subset: str) -> Path:
        """
        Resolve test-end RUL file path for a CMAPSS subset.

        Args:
            subset: Subset identifier (``FD001``..``FD004``).

        Returns:
            Path: Expected RUL target file path.

        Raises:
            ValueError: If subset is invalid.
        """
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

    Args:
        df: Candidate CMAPSS dataframe with raw columns.
        name: Label used in error messages for provenance.

    Returns:
        None.

    Raises:
        ValueError: If schema, positivity, uniqueness, or cycle ordering checks fail.

    Assumptions & leakage boundaries:
        - Frame must contain full raw schema from :func:`cmapss_all_columns`.
        - Validation enforces temporal ordering but does not split data.
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
    Add per-row RUL labels for train trajectories.

    Args:
        df_train: Train dataframe with required columns ``unit_id`` and ``cycle``.
        rul_cap: Optional positive cap for clipping the resulting ``rul``.

    Returns:
        pd.DataFrame: Copy of ``df_train`` with integer ``rul`` column.

    Raises:
        ValueError: If required columns are missing, cap is invalid, or negative
            RUL values are produced.

    Assumptions & leakage boundaries:
        - Each unit in training runs to failure.
        - Formula: ``RUL(t) = max_cycle(unit) - cycle(t)``.
        - No statistics from test data are used.
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
    Add per-row RUL labels for test trajectories.

    Args:
        df_test: Test dataframe with required columns ``unit_id`` and ``cycle``.
        rul_end: Per-unit end-of-trajectory RUL values indexed by ``unit_id``.
        rul_cap: Optional positive cap for clipping the resulting ``rul``.

    Returns:
        pd.DataFrame: Copy of ``df_test`` with integer ``rul`` column.

    Raises:
        ValueError: If required columns are missing, target alignment fails,
            cap is invalid, or negative RUL values are produced.

    Assumptions & leakage boundaries:
        - Test trajectories are truncated before failure.
        - ``rul_end`` provides one value per test unit.
        - Formula: ``RUL(cycle) = (last_cycle - cycle) + rul_end(unit)``.
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
    Load one CMAPSS subset and return labeled train/test dataframes.

    Args:
        paths: Validated CMAPSS path container.
        subset: Subset identifier in ``{"FD001","FD002","FD003","FD004"}``.
        rul_cap: Optional positive cap for ``rul`` clipping.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
        ``(train_df, test_df)`` with CMAPSS raw columns plus ``rul``.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If subset is invalid, schema/ordering checks fail, RUL
            targets are misaligned, or label construction produces invalid values.

    Assumptions & leakage boundaries:
        - This function only loads and labels; no split/scaling/windowing occurs.
        - Test labels are reconstructed using official ``RUL_FD00x`` files.
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
    """
    Produce a compact dataframe summary for logging/diagnostics.

    Args:
        df: CMAPSS dataframe, optionally including ``rul``.

    Returns:
        Dict[str, Optional[int]]: Summary fields
        ``n_rows``, ``n_units``, ``cycle_min``, ``cycle_max``,
        ``rul_min`` and ``rul_max`` (the last two may be ``None``).

    Raises:
        KeyError: If required structural columns are missing.
    """
    return {
        "n_rows": int(len(df)),
        "n_units": int(df["unit_id"].nunique()),
        "cycle_min": int(df["cycle"].min()),
        "cycle_max": int(df["cycle"].max()),
        "rul_min": int(df["rul"].min()) if "rul" in df.columns else None,
        "rul_max": int(df["rul"].max()) if "rul" in df.columns else None,
    }
