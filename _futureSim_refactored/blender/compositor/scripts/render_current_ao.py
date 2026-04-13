"""Fast runner for compositor_ao.blend.

Three-input runner: existing_condition_positive + positive_state +
positive_priority_state.

Environment variables (all paths, all required except blend/scene):
    COMPOSITOR_EXISTING_EXR     existing_condition_positive EXR
    COMPOSITOR_PATHWAY_EXR      positive_state EXR
    COMPOSITOR_PRIORITY_EXR     positive_priority_state EXR
    COMPOSITOR_OUTPUT_DIR       output directory
    COMPOSITOR_BLEND_PATH       optional override (default: canonical compositor_ao.blend)
    COMPOSITOR_SCENE_NAME       optional (default: Current)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _fast_runner_core import CANONICAL_ROOT, FastRunnerConfig, run_fast_render  # noqa: E402

NAME = "render_current_ao"
BLEND_DEFAULT = CANONICAL_ROOT / "compositor_ao.blend"


def main() -> None:
    existing = os.environ.get("COMPOSITOR_EXISTING_EXR", "")
    pathway = os.environ.get("COMPOSITOR_PATHWAY_EXR", "")
    priority = os.environ.get("COMPOSITOR_PRIORITY_EXR", "")
    out = os.environ.get("COMPOSITOR_OUTPUT_DIR", "")
    if not (existing and pathway and priority and out):
        raise ValueError(
            "Required env vars: COMPOSITOR_EXISTING_EXR, COMPOSITOR_PATHWAY_EXR, "
            "COMPOSITOR_PRIORITY_EXR, COMPOSITOR_OUTPUT_DIR"
        )
    blend = os.environ.get("COMPOSITOR_BLEND_PATH", str(BLEND_DEFAULT))
    scene = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")

    config = FastRunnerConfig(
        name=NAME,
        blend_path=Path(blend),
        scene_name=scene,
        exr_inputs={
            "AO::EXR Existing": Path(existing),
            "AO::EXR Pathway": Path(pathway),
            "AO::EXR Priority": Path(priority),
        },
        file_output_node="AO::Outputs",
        output_dir=Path(out),
    )
    run_fast_render(config)


if __name__ == "__main__":
    main()
