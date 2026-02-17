"""
Central configuration for the project.

Keep paths and core constants here so notebooks/scripts do not hardcode values.
This supports reproducibility and makes experiments easier to track.
"""

from __future__ import annotations

from pathlib import Path

# Project root is assumed to be the repository root (one level above /src).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Data directories (raw data should NOT be committed to Git).
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

# Standard output locations for reproducible experiment artifacts.
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"
METRICS_DIR: Path = RESULTS_DIR / "metrics"

# Reproducibility controls.
RANDOM_STATE: int = 42

# CMAPSS / RUL modelling defaults.
# RUL is often clipped to reduce label variance and stabilize training.
RUL_CLIP: int = 125

# Windowing defaults for sequence/feature construction.
WINDOW_SIZE: int = 30
STEP_SIZE: int = 1
