from __future__ import annotations

"""
Shared render setup for v4 PyVista renderers.

Consolidates the camera presets, colour palettes, helper functions, and
PyVista plotter settings used by:

- `render_proposal_v4.py` — proposal-and-interventions renders
- `render_debug_recruit.py` — recruit diagnostic layers
- `render_debug_release_control.py` — release-control diagnostic

Nothing in this module is v3-specific: it reuses the `ENGINE3_PROPOSALS`
plotter settings and the colour palette that the v4 pipeline has
standardised on (v3 hex values mapped by position to the v4 framebuffer
integer slots in `blender/bexport/proposal_framebuffers.py`).
"""

import colorsys
from pathlib import Path
from typing import Final

import numpy as np
import pyvista as pv
from PIL import ImageFont

from _futureSim_refactored.outputs.report.pyvista_render_settings.engine3_proposals import (
    ENGINE3_PROPOSALS_RENDER_SETTINGS,
)


# ── camera presets ─────────────────────────────────────────────────────────

CAMERAS: Final[dict[str, dict[str, object]]] = {
    "trimmed-parade": {
        "position": (-710.5999, 155.0484, 780.0399),
        "focal_point": (52.8332, 109.8565, 57.0),
        "view_up": (0.6873, -0.0101, 0.7263),
        "view_angle": 30.0,
    },
    "city": {
        "position": (827.5661, 49.1329, 880.7060),
        "focal_point": (288.2490, -22.3298, 333.8400),
        "view_up": (-0.7116, -0.0063, 0.7026),
        "view_angle": 28.1718,
    },
    "uni": {
        "position": (-76.7619, -879.4925, 863.1532),
        "focal_point": (-13.3853, 44.2705, 63.1832),
        "view_up": (-0.0381, 0.6557, 0.7541),
        "view_angle": 30.0,
    },
}


# ── colour palettes ────────────────────────────────────────────────────────

def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


WHITE_RGB: Final[tuple[int, int, int]] = hex_rgb("#FFFFFF")

FOREST_SIZE_HEX: Final[dict[str, str]] = {
    "small": "#AADB5E",
    "medium": "#9AB9DE",
    "large": "#F99F76",
    "senescing": "#EB9BC5",
    "snag": "#FCE358",
    "fallen": "#82CBB9",
    "decayed": "#5F867E",
    # Terminal absent state; should normally be filtered before render/export.
    "early-tree-death": "#6E6E6E",
    "artificial": "#FF0000",
}

DEADWOOD_BASE_HEX: Final[dict[str, str]] = {
    "fallen": "#8F89BF",
    "decayed": "#5F867E",
}

PROPOSAL_HEX: Final[dict[str, dict[int, str]]] = {
    "blender_proposal-deploy-structure": {
        2: "#FF0000",  # adapt-utility-pole
        3: "#8F89BF",  # translocate-log
        4: "#CE6DD9",  # upgrade-feature
    },
    "blender_proposal-decay": {
        2: "#B83B6B",  # buffer-feature
        3: "#D9638C",  # brace-feature
    },
    "blender_proposal-recruit": {
        2: "#C5E28E",  # rewild-smaller-patch
        3: "#5CB85C",  # rewild-larger-patch
    },
    "blender_proposal-colonise": {
        2: "#5CB85C",  # larger-patches-rewild (ground)
        3: "#8CCC4F",  # enrich-envelope
        4: "#B87A38",  # roughen-envelope
    },
}

FOREST_SIZE_RGB: Final[dict[str, tuple[int, int, int]]] = {
    key: hex_rgb(value) for key, value in FOREST_SIZE_HEX.items()
}
DEADWOOD_BASE_RGB: Final[dict[str, tuple[int, int, int]]] = {
    key: hex_rgb(value) for key, value in DEADWOOD_BASE_HEX.items()
}
PROPOSAL_RGB: Final[dict[str, dict[int, tuple[int, int, int]]]] = {
    array_name: {state: hex_rgb(hex_value) for state, hex_value in state_map.items()}
    for array_name, state_map in PROPOSAL_HEX.items()
}

PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP: Final[list[str]] = [
    "blender_proposal-release-control",
    "blender_proposal-colonise",
    "blender_proposal-recruit",
    "blender_proposal-decay",
    "blender_proposal-deploy-structure",
]

RELEASE_CONTROL_SATURATION: Final[dict[int, float]] = {
    1: 0.20,  # rejected
    2: 0.50,  # reduce-canopy-pruning
    3: 1.00,  # eliminate-canopy-pruning
}

RELEASE_CONTROL_FOREST_SIZE_KEYS: Final[list[str]] = [
    "small",
    "medium",
    "large",
    "senescing",
    "snag",
]


# ── plotter settings ───────────────────────────────────────────────────────

CUSTOM_RENDER_SETTINGS: Final[dict[str, object]] = ENGINE3_PROPOSALS_RENDER_SETTINGS


# ── array helpers ──────────────────────────────────────────────────────────

def rgb_to_uint8_array(rgb: tuple[int, int, int], count: int) -> np.ndarray:
    array = np.empty((count, 3), dtype=np.uint8)
    array[:] = np.asarray(rgb, dtype=np.uint8)
    return array


def normalize_str_array(values: np.ndarray) -> np.ndarray:
    return np.char.lower(np.asarray(values).astype(str))


def release_control_rgb(forest_size: str, state: int) -> tuple[int, int, int] | None:
    if state == 0:
        return None
    base = FOREST_SIZE_RGB.get(forest_size)
    if base is None:
        return None
    saturation_scale = RELEASE_CONTROL_SATURATION.get(state)
    if saturation_scale is None:
        return None
    r, g, b = [component / 255.0 for component in base]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, min(1.0, s * saturation_scale))
    rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
    return (round(rr * 255), round(gg * 255), round(bb * 255))


# ── rendering ──────────────────────────────────────────────────────────────

def render_png(mesh: pv.PolyData, site: str, output_path: Path, rgb_values: np.ndarray) -> None:
    """Render a mesh to PNG using the shared plotter settings and site camera."""
    settings = CUSTOM_RENDER_SETTINGS
    camera = CAMERAS[site]
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(int(settings["window_width"]), int(settings["window_height"])),
    )
    plotter.set_background(str(settings["background"]))
    plotter.add_mesh(
        mesh,
        scalars=rgb_values,
        rgb=True,
        render_points_as_spheres=bool(settings["render_points_as_spheres"]),
        point_size=float(settings["point_size"]),
        lighting=bool(settings["lighting"]),
    )
    if bool(settings["eye_dome_lighting"]):
        plotter.enable_eye_dome_lighting()
    plotter.camera_position = [
        camera["position"],
        camera["focal_point"],
        camera["view_up"],
    ]
    plotter.camera.view_angle = camera["view_angle"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(output_path))
    plotter.close()


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ["Aptos.ttf", "Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()
