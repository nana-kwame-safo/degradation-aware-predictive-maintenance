"""
Data loading utilities for the NASA CMAPSS dataset.

CMAPSS files are typically whitespace-delimited .txt files without a header.
They contain:
- unit id
- cycle index
- operating settings
- sensor measurements
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_cmapss_txt(path: Path, col_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load a CMAPSS .txt file (whitespace-separated values).

    Parameters
    ----------
    path:
        Path to the CMAPSS .txt file (e.g., train_FD001.txt).
    col_names:
        Optional list of column names to apply.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # CMAPSS is whitespace-separated; regex handles variable spaces.
    df = pd.read_csv(path, sep=r"\s+", header=None)

    if col_names is not None:
        if len(col_names) != df.shape[1]:
            raise ValueError(
                f"Provided col_names length={len(col_names)} does not match "
                f"data columns={df.shape[1]}."
            )
        df.columns = col_names

    return df
