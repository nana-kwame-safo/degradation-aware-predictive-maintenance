#!/usr/bin/env python3
"""
Remove local caches and generated artifacts that should not be committed.

This script is safe by default for repository placeholders:
- never deletes README.md
- never deletes .gitkeep
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_FILENAMES = {"README.md", ".gitkeep"}
RESULTS_OUTPUT_DIRS = (
    REPO_ROOT / "results" / "figures",
    REPO_ROOT / "results" / "tables",
    REPO_ROOT / "results" / "metrics",
)


def _is_protected(path: Path) -> bool:
    return path.name in PROTECTED_FILENAMES


def _remove_path(path: Path, dry_run: bool) -> None:
    relative = path.relative_to(REPO_ROOT)
    if dry_run:
        print(f"[dry-run] remove {relative}")
        return

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    print(f"removed {relative}")


def _clean_pycache(dry_run: bool) -> int:
    removed = 0
    pycache_dirs = sorted(
        {path for path in REPO_ROOT.rglob("__pycache__") if path.is_dir()},
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for cache_dir in pycache_dirs:
        _remove_path(cache_dir, dry_run=dry_run)
        removed += 1
    return removed


def _clean_mpl_fontlists(dry_run: bool) -> int:
    removed = 0
    fontlists = sorted(
        path
        for path in REPO_ROOT.glob("**/.mplconfig/fontlist-*.json")
        if path.is_file()
    )
    for fontlist in fontlists:
        _remove_path(fontlist, dry_run=dry_run)
        removed += 1
    return removed


def _clean_results_outputs(dry_run: bool) -> int:
    removed = 0
    for output_dir in RESULTS_OUTPUT_DIRS:
        if not output_dir.exists():
            continue

        files = sorted(path for path in output_dir.rglob("*") if path.is_file())
        for file_path in files:
            if _is_protected(file_path):
                continue
            _remove_path(file_path, dry_run=dry_run)
            removed += 1

        dirs = sorted(
            (path for path in output_dir.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for dir_path in dirs:
            if dir_path == output_dir:
                continue
            try:
                if dry_run:
                    if not any(dir_path.iterdir()):
                        print(f"[dry-run] remove {dir_path.relative_to(REPO_ROOT)}")
                        removed += 1
                else:
                    dir_path.rmdir()
                    print(f"removed {dir_path.relative_to(REPO_ROOT)}")
                    removed += 1
            except OSError:
                continue
    return removed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete local caches and generated results while preserving README/.gitkeep placeholders."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dry_run = bool(args.dry_run)

    removed_count = 0
    removed_count += _clean_pycache(dry_run=dry_run)
    removed_count += _clean_mpl_fontlists(dry_run=dry_run)
    removed_count += _clean_results_outputs(dry_run=dry_run)

    mode = "dry run completed" if dry_run else "cleanup completed"
    print(f"{mode}: removed {removed_count} path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
