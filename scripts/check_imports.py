#!/usr/bin/env python3
"""
Lightweight import smoke check for core project modules.

Usage:
    python scripts/check_imports.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

# Ensure repository root is importable when running directly from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULES = [
    "src.config",
    "src.data.data_loader",
    "src.data.preprocessing",
    "src.data.validation",
    "src.features.feature_engineering",
    "src.models.baseline_models",
    "src.models.rul_baselines",
    "src.evaluation.metrics",
    "src.evaluation.plots",
    "src.run_baseline",
    "scripts.train_baselines",
]


def main() -> int:
    print("Running import checks...")
    failures: list[str] = []

    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[PASS] {module_name}")
        except Exception as exc:
            failures.append(module_name)
            print(f"[FAIL] {module_name}: {exc}")
            traceback.print_exc()

    if failures:
        print("\nImport check failed for modules:")
        for module_name in failures:
            print(f"- {module_name}")
        return 1

    print("\nAll import checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
