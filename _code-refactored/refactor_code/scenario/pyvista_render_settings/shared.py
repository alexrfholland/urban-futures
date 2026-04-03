from __future__ import annotations

"""
Shared PyVista render-setting defaults for refactored scenario renders.
"""

from typing import Final


PYVISTA_SHARED_RENDER_SETTINGS: Final[dict[str, object]] = {
    "window_width": 2200,
    "window_height": 1600,
    "background": "#FFFFFF",
    "lighting": False,
    "eye_dome_lighting": True,
}
