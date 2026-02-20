"""
Central configuration for the project.

Design goals:
- Keep paths and core constants here so notebooks/scripts do not hardcode values.
- Support reproducibility (single source of truth).
- Provide both module-level constants and a Config dataclass for convenience.

Engineering note:
- Keep this file stable; changing constants mid-project can invalidate comparisons.
- If you change a core constant (e.g., RUL clip), record it in results/metrics logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

# Project root is assumed to be the repository root (one level above /src).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Data directories (raw data should NOT be committed to Git).
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_INTERIM_DIR: Path = DATA_DIR / "interim"        # optional: cleaned but not final
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"    # final modeling-ready data

# CMAPSS location (recommended layout: data/raw/cmapss/*.txt)
CMAPSS_RAW_DIR: Path = DATA_RAW_DIR / "cmapss"

# Standard output locations for reproducible experiment artifacts.
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"
METRICS_DIR: Path = RESULTS_DIR / "metrics"


def ensure_project_dirs() -> None:
    """
    Create expected output directories if missing.

    This avoids silent failures where scripts try to write metrics/figures
    into folders that do not yet exist.
    """
    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, METRICS_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

RANDOM_STATE: int = 42

# -----------------------------------------------------------------------------
# CMAPSS / RUL modelling defaults
# -----------------------------------------------------------------------------

# RUL is often clipped to reduce label variance and stabilise training.
# Common values in literature: 125 or 130 depending on preprocessing choices.
RUL_CLIP: int = 125

# Unit-based split default (prevents leakage across cycles).
VAL_FRACTION: float = 0.2

# Windowing defaults for sequence models / rolling feature construction.
WINDOW_SIZE: int = 30
STEP_SIZE: int = 1


# -----------------------------------------------------------------------------
# Column naming helpers (single source of truth)
# -----------------------------------------------------------------------------

def cmapss_columns() -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
        op_settings: [op_setting_1..3]
        sensors:     [sensor_1..sensor_21]
        features:    op_settings + sensors
    """
    op_settings = [f"op_setting_{i}" for i in range(1, 4)]
    sensors = [f"sensor_{i}" for i in range(1, 22)]
    features = op_settings + sensors
    return op_settings, sensors, features


# -----------------------------------------------------------------------------
# Config object for scripts
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """
    Lightweight configuration object used by scripts and experiments.

    This wraps module constants but makes it easier to pass configuration around
    (e.g., to training/evaluation functions) and later serialize to YAML/JSON.
    """

    # Directories
    project_root: Path = PROJECT_ROOT
    data_raw_dir: Path = DATA_RAW_DIR
    data_interim_dir: Path = DATA_INTERIM_DIR
    data_processed_dir: Path = DATA_PROCESSED_DIR
    cmapss_raw_dir: Path = CMAPSS_RAW_DIR

    results_dir: Path = RESULTS_DIR
    figures_dir: Path = FIGURES_DIR
    tables_dir: Path = TABLES_DIR
    metrics_dir: Path = METRICS_DIR

    # Dataset selection
    cmapss_subset: str = "FD001"  # FD001, FD002, FD003, FD004

    # Splitting / reproducibility
    val_fraction: float = VAL_FRACTION
    random_state: int = RANDOM_STATE

    # Feature / window settings
    window: int = WINDOW_SIZE
    step: int = STEP_SIZE

    # Target engineering
    clip_rul: int = RUL_CLIP

    # Baseline modelling (interpretable first)
    baseline_model: str = "ridge"  # ridge | rf | xgb (later)

    # Feature columns (refine later after EDA/feature selection)
    op_setting_cols: List[str] = field(default_factory=lambda: [f"op_setting_{i}" for i in range(1, 4)])
    sensor_cols: List[str] = field(default_factory=lambda: [f"sensor_{i}" for i in range(1, 22)])

    @property
    def feature_cols(self) -> List[str]:
        """Operating conditions + sensor columns used as model inputs."""
        return self.op_setting_cols + self.sensor_cols
