from __future__ import annotations

"""
Registry of named PyVista render-setting schemas used by the refactored
scenario renderers.

Current schemas:
- engine3-proposals
"""

from refactor_code.scenario.pyvista_render_settings.engine3_proposals import (
    ENGINE3_PROPOSALS_RENDER_SETTINGS,
)
from refactor_code.scenario.pyvista_render_settings.shared import (
    PYVISTA_SHARED_RENDER_SETTINGS,
)


PYVISTA_RENDER_SETTINGS = {
    "engine3-proposals": ENGINE3_PROPOSALS_RENDER_SETTINGS,
}


__all__ = [
    "PYVISTA_SHARED_RENDER_SETTINGS",
    "ENGINE3_PROPOSALS_RENDER_SETTINGS",
    "PYVISTA_RENDER_SETTINGS",
]
