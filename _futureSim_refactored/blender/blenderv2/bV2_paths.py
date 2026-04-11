"""Shared path settings for the Blender v2 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:
    from _futureSim_refactored.paths import BLENDERV2_INPUTS_ROOT
except ImportError:
    BLENDERV2_INPUTS_ROOT = Path(__file__).resolve().parents[3] / "_data-refactored" / "blenderv2" / "inputs"


def iter_blender_input_roots() -> Iterable[Path]:
    """Yield canonical local bV2 input roots."""

    if BLENDERV2_INPUTS_ROOT.exists():
        yield BLENDERV2_INPUTS_ROOT
