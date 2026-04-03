from __future__ import annotations

"""
PyVista render settings for the `engine3-proposals` custom v3 proposal render.
"""

from typing import Final

from refactor_code.scenario.pyvista_render_settings.shared import (
    PYVISTA_SHARED_RENDER_SETTINGS,
)


ENGINE3_PROPOSALS_RENDER_SETTINGS: Final[dict[str, object]] = {
    **PYVISTA_SHARED_RENDER_SETTINGS,
    "point_size": 4.0,
    "render_points_as_spheres": False,
}
