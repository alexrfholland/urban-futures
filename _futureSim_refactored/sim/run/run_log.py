"""
Persistent run log — appends one row per batch run to a CSV at the repo root.

The log lives at ``_data-refactored/run_log.csv`` so it persists across runs
and is easy to inspect.  Each row records:

    timestamp, name, output_root, description

Use ``get_last_output_root()`` to retrieve the most recent root path.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

_LOG_PATH = (
    Path(__file__).resolve().parents[3]  # repo root
    / "_data-refactored"
    / "run_log.csv"
)

_FIELDS = ["timestamp", "name", "output_root", "description"]


def _ensure_header() -> None:
    if _LOG_PATH.exists() and _LOG_PATH.stat().st_size > 0:
        return
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(_FIELDS)


def append_run_log(
    name: str,
    output_root: str,
    description: str = "",
) -> None:
    """Append a single row to the run log."""
    _ensure_header()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([stamp, name, output_root, description])
    print(f"Run log → {_LOG_PATH}")


def get_last_output_root() -> str | None:
    """Return the output_root from the most recent log entry, or None."""
    if not _LOG_PATH.exists():
        return None
    with open(_LOG_PATH, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]["output_root"]


def print_log(n: int = 10) -> None:
    """Print the last *n* entries."""
    if not _LOG_PATH.exists():
        print("No run log found.")
        return
    with open(_LOG_PATH, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Run log is empty.")
        return
    for row in rows[-n:]:
        print(f"  {row['timestamp']}  {row['name']:<40}  {row['output_root']}")
        if row.get("description"):
            print(f"    {row['description']}")


if __name__ == "__main__":
    print_log()
