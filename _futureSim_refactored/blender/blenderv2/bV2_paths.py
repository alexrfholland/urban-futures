"""Shared path settings for the Blender v2 pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

try:
    from _futureSim_refactored.paths import BLENDERV2_INPUTS_ROOT
except ImportError:
    BLENDERV2_INPUTS_ROOT = Path(__file__).resolve().parents[3] / "_data-refactored" / "blenderv2" / "inputs"


DATA_BUNDLE_ROOT_ENV = "BV2_DATA_BUNDLE_ROOT"


def iter_blender_input_roots() -> Iterable[Path]:
    """Yield canonical local bV2 input roots.

    Single source of truth for where the instancer, world-attributes, and
    bioenvelope builders look for node-df CSVs, VTKs, and bioenvelope PLYs.
    All three stages share the same `<root>/<category>/<site>/<file>` layout,
    so one env var governs all of them:

    - `BV2_DATA_BUNDLE_ROOT` takes precedence when set and pointing at an
      existing directory
    - falls back to the repo-local default at
      `_data-refactored/blenderv2/inputs/` when that exists
    """
    seen: set[Path] = set()

    env_value = os.environ.get(DATA_BUNDLE_ROOT_ENV, "").strip()
    if env_value:
        candidate = Path(env_value)
        if candidate.exists():
            seen.add(candidate)
            yield candidate

    if BLENDERV2_INPUTS_ROOT.exists() and BLENDERV2_INPUTS_ROOT not in seen:
        yield BLENDERV2_INPUTS_ROOT
