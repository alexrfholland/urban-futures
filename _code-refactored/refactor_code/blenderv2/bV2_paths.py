"""Shared path settings for the Blender v2 pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


_TRUTHY = {"1", "true", "yes", "on"}
_DEFAULT_TEMP_LOCAL_ROOT = r"E:\2026 Arboreal Futures\blenderv2\inputs\v3-5"
_DEFAULT_REMOTE_REPO_ROOT = r"Z:\MF 2026 Arboreal Futures\blender\inputs\v3-5"
_LEGACY_BUNDLE_ROOTS = (
    Path(r"E:\2026 Arboreal Futures\blender\inputs\v3 tests\simv3recruitanddecaytweaks"),
    Path(r"D:\2026 Arboreal Futures\urban-futures\_data-refactored\v3engine_outputs"),
    Path(r"D:\2026 Arboreal Futures\data"),
)


BLENDER_USE_REMOTE = os.environ.get("BLENDER_USE_REMOTE", "0").strip().lower() in _TRUTHY
TEMP_LOCAL_ROOT = Path(
    os.environ.get("BLENDER_TEMP_REPO", _DEFAULT_TEMP_LOCAL_ROOT)
).expanduser()
BLENDER_REPO_ROOT = Path(
    os.environ.get("BLENDER_REPO_ROOT", _DEFAULT_REMOTE_REPO_ROOT)
).expanduser()

# Backward-compatible alias while the naming settles.
BLENDER_TEMP_REPO = TEMP_LOCAL_ROOT


def iter_blender_input_roots() -> Iterable[Path]:
    """Yield bundle roots in the current preferred order.

    The intended working root is the local temp copy. Older local bundle roots
    remain as fallbacks until the explicit sync wrapper is in place.
    """

    seen: set[Path] = set()
    for candidate in (TEMP_LOCAL_ROOT, *_LEGACY_BUNDLE_ROOTS):
        resolved = Path(candidate)
        if resolved in seen:
            continue
        if resolved.exists():
            seen.add(resolved)
            yield resolved
