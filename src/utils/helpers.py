"""
Utility helpers.

These functions keep the project consistent:
- standardized directory creation
- standardized JSON saving/loading for metrics and configs
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path:
        Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """
    Save a dictionary as a JSON file with readable formatting.

    Parameters
    ----------
    obj:
        Dictionary to save.
    path:
        Output file path.
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.

    Parameters
    ----------
    path:
        JSON file path.

    Returns
    -------
    dict
        Parsed JSON as dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
