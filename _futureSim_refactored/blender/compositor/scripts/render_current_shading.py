"""Fast runner for compositor_shading.blend.

Seven-input runner. Targets the aggregate 'Current Shading ::Outputs' node
(the 6-slot consolidated output). The per-family File Output nodes remain
muted.

Environment variables:
    COMPOSITOR_EXISTING_EXR              existing_condition_positive EXR
    COMPOSITOR_EXISTING_TRENDING_EXR     existing_condition_trending EXR
    COMPOSITOR_PATHWAY_EXR               positive_state EXR
    COMPOSITOR_PRIORITY_EXR              positive_priority_state EXR
    COMPOSITOR_BIOENVELOPE_EXR           bioenvelope_positive EXR
    COMPOSITOR_BIOENVELOPE_TRENDING_EXR  bioenvelope_trending EXR
    COMPOSITOR_OUTPUT_DIR                output directory
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

NAME = "render_current_shading"
BLEND_DEFAULT = CANONICAL_ROOT / "compositor_shading.blend"


def main() -> None:
    existing = os.environ.get("COMPOSITOR_EXISTING_EXR", "")
    existing_trend = os.environ.get("COMPOSITOR_EXISTING_TRENDING_EXR", "")
    pathway = os.environ.get("COMPOSITOR_PATHWAY_EXR", "")
    priority = os.environ.get("COMPOSITOR_PRIORITY_EXR", "")
    bio_pos = os.environ.get("COMPOSITOR_BIOENVELOPE_EXR", "")
    bio_trend = os.environ.get("COMPOSITOR_BIOENVELOPE_TRENDING_EXR", "")
    out = os.environ.get("COMPOSITOR_OUTPUT_DIR", "")
    required = [existing, existing_trend, pathway, priority, bio_pos, bio_trend, out]
    if not all(required):
        raise ValueError(
            "Required env vars: COMPOSITOR_EXISTING_EXR, COMPOSITOR_EXISTING_TRENDING_EXR, "
            "COMPOSITOR_PATHWAY_EXR, COMPOSITOR_PRIORITY_EXR, COMPOSITOR_BIOENVELOPE_EXR, "
            "COMPOSITOR_BIOENVELOPE_TRENDING_EXR, COMPOSITOR_OUTPUT_DIR"
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
            "Current BioEnvelope :: EXR BioEnvelope": Path(bio_pos),
            "Current BioEnvelope :: EXR Trending": Path(bio_trend),
            "Current Shading :: BioEnvelope Helper :: EXR Existing Positive": Path(existing),
            "Current Shading :: BioEnvelope Helper :: EXR Existing Trending": Path(existing_trend),
        },
        file_output_node="Current Shading ::Outputs",
        output_dir=Path(out),
    )
    run_fast_render(config)


if __name__ == "__main__":
    main()
