from __future__ import annotations

"""
Source-of-truth constants for the custom v3 proposal render schema.

This schema is for the validation-style proposal renders built from the
`blender_proposal-*` integer framebuffer arrays plus `forest_size`.

Composition, from highest to lowest priority:

1. deploy-structure
2. decay
3. recruit
4. colonise
5. release-control
6. deadwood base
7. white fallback

Notes:
- proposal families only render accepted intervention states
- `not-assessed` and `rejected` are not rendered for deploy-structure, decay,
  recruit, or colonise
- release-control uses the current `forest_size` hue with relative saturation:
  - rejected = 20%
  - reduce-pruning = 50%
  - eliminate-pruning = 100%
- deadwood base is only used where no higher-priority proposal color is present
"""

from typing import Final

from refactor_code.blender.proposal_framebuffers import (
    DEFAULT_OUTPUT_COLUMNS,
    FRAMEBUFFER_STATE_MAPPINGS,
)
from refactor_code.scenario.pyvista_render_settings.engine3_proposals import (
    ENGINE3_PROPOSALS_RENDER_SETTINGS,
)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


WHITE_RGB: Final[tuple[int, int, int]] = _hex_rgb("#FFFFFF")

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

PROPOSAL_LAYER_ORDER_TOP_TO_BOTTOM: Final[list[str]] = [
    "blender_proposal-deploy-structure",
    "blender_proposal-decay",
    "blender_proposal-recruit",
    "blender_proposal-colonise",
    "blender_proposal-release-control",
]

PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP: Final[list[str]] = list(
    reversed(PROPOSAL_LAYER_ORDER_TOP_TO_BOTTOM)
)

PROPOSAL_INT_MAPPING: Final[dict[str, dict[int, str]]] = {
    DEFAULT_OUTPUT_COLUMNS[family]: {value: label for label, value in mapping.items()}
    for family, mapping in FRAMEBUFFER_STATE_MAPPINGS.items()
}

PROPOSAL_HEX: Final[dict[str, dict[int, str]]] = {
    "blender_proposal-deploy-structure": {
        2: "#FF0000",  # adapt-utility-pole
        3: "#8F89BF",  # translocated-log
        4: "#CE6DD9",  # upgrade-feature
    },
    "blender_proposal-decay": {
        2: "#B83B6B",  # buffer-feature
        3: "#D9638C",  # brace-feature
    },
    "blender_proposal-recruit": {
        2: "#C5E28E",  # buffer-feature
        3: "#5CB85C",  # rewild-ground
    },
    "blender_proposal-colonise": {
        2: "#5CB85C",  # rewild-ground
        3: "#8CCC4F",  # enrich-envelope
        4: "#B87A38",  # roughen-envelope
    },
}

PROPOSAL_LABELS: Final[dict[str, dict[int, str]]] = PROPOSAL_INT_MAPPING

RELEASE_CONTROL_SATURATION: Final[dict[int, float]] = {
    1: 0.20,  # rejected
    2: 0.50,  # reduce-pruning
    3: 1.00,  # eliminate-pruning
}

RELEASE_CONTROL_FOREST_SIZE_KEYS: Final[list[str]] = [
    "small",
    "medium",
    "large",
    "senescing",
    "snag",
]

CUSTOM_RENDER_SETTINGS: Final[dict[str, object]] = ENGINE3_PROPOSALS_RENDER_SETTINGS
