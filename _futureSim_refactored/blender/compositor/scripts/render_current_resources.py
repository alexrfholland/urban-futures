"""Fast runner for compositor_resources.blend.

Three-input runner: positive_state + positive_priority_state + trending_state.

Environment variables:
    COMPOSITOR_PATHWAY_EXR      positive_state EXR
    COMPOSITOR_PRIORITY_EXR     positive_priority_state EXR
    COMPOSITOR_TRENDING_EXR     trending_state EXR
    COMPOSITOR_OUTPUT_DIR       output directory
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

NAME = "render_current_resources"
BLEND_DEFAULT = CANONICAL_ROOT / "compositor_resources.blend"


def main() -> None:
    pathway = os.environ.get("COMPOSITOR_PATHWAY_EXR", "")
    priority = os.environ.get("COMPOSITOR_PRIORITY_EXR", "")
    trending = os.environ.get("COMPOSITOR_TRENDING_EXR", "")
    out = os.environ.get("COMPOSITOR_OUTPUT_DIR", "")
    if not (pathway and priority and trending and out):
        raise ValueError(
            "Required env vars: COMPOSITOR_PATHWAY_EXR, COMPOSITOR_PRIORITY_EXR, "
            "COMPOSITOR_TRENDING_EXR, COMPOSITOR_OUTPUT_DIR"
        )
    blend = os.environ.get("COMPOSITOR_BLEND_PATH", str(BLEND_DEFAULT))
    scene = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")

    config = FastRunnerConfig(
        name=NAME,
        blend_path=Path(blend),
        scene_name=scene,
        exr_inputs={
            "Resources::EXR Pathway": Path(pathway),
            "Resources::EXR Priority": Path(priority),
            "Resources::EXR Trending": Path(trending),
        },
        file_output_node="Resources::Outputs",
        output_dir=Path(out),
    )
    run_fast_render(config)


if __name__ == "__main__":
    main()
