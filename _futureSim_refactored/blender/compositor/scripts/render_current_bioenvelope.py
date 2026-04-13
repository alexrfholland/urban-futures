"""Fast runner for compositor_bioenvelope.blend.

Three-input runner: existing_condition_positive + bioenvelope_positive +
bioenvelope_trending.

Environment variables:
    COMPOSITOR_EXISTING_EXR              existing_condition_positive EXR
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

NAME = "render_current_bioenvelope"
BLEND_DEFAULT = CANONICAL_ROOT / "compositor_bioenvelope.blend"


def main() -> None:
    existing = os.environ.get("COMPOSITOR_EXISTING_EXR", "")
    bio_pos = os.environ.get("COMPOSITOR_BIOENVELOPE_EXR", "")
    bio_trend = os.environ.get("COMPOSITOR_BIOENVELOPE_TRENDING_EXR", "")
    out = os.environ.get("COMPOSITOR_OUTPUT_DIR", "")
    if not (existing and bio_pos and bio_trend and out):
        raise ValueError(
            "Required env vars: COMPOSITOR_EXISTING_EXR, COMPOSITOR_BIOENVELOPE_EXR, "
            "COMPOSITOR_BIOENVELOPE_TRENDING_EXR, COMPOSITOR_OUTPUT_DIR"
        )
    blend = os.environ.get("COMPOSITOR_BLEND_PATH", str(BLEND_DEFAULT))
    scene = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")

    config = FastRunnerConfig(
        name=NAME,
        blend_path=Path(blend),
        scene_name=scene,
        exr_inputs={
            "Current BioEnvelope :: EXR Existing": Path(existing),
            "Current BioEnvelope :: EXR BioEnvelope": Path(bio_pos),
            "Current BioEnvelope :: EXR Trending": Path(bio_trend),
        },
        file_output_node="Current BioEnvelope ::Outputs",
        output_dir=Path(out),
    )
    run_fast_render(config)


if __name__ == "__main__":
    main()
