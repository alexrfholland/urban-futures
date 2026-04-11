from __future__ import annotations

"""
PyVista render settings for the custom v3 proposal renders.

Used by:
- `engine3-proposals_interventions`
- `engine3-proposals`
"""

from typing import Final

from _futureSim_refactored.outputs.report.pyvista_render_settings.shared import (
    PYVISTA_SHARED_RENDER_SETTINGS,
)


ENGINE3_PROPOSALS_RENDER_SETTINGS: Final[dict[str, object]] = {
    **PYVISTA_SHARED_RENDER_SETTINGS,
    "point_size": 4.0,
    "render_points_as_spheres": False,
}
