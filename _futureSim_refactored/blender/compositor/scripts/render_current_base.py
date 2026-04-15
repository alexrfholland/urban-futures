"""Fast runner for compositor_base.blend.

Single-input runner: existing_condition_positive EXR. The canonical blend owns
a File Output node named 'Current Base Outputs :: Outputs' with 10 slots (one
per labeled reroute). `rebuild_file_output=True` because those slots are fed
from reroutes whose upstream is a CompositorNodeGroup output — the pattern §7d
in COMPOSITOR_RUN-INSTRUCTIONS.md warns about.

Environment variables:
    COMPOSITOR_EXISTING_EXR   existing_condition_positive EXR (required)
    COMPOSITOR_OUTPUT_DIR     output directory (required)
    COMPOSITOR_BLEND_PATH     optional override
    COMPOSITOR_SCENE_NAME     optional override (default: Current)
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

NAME = "render_current_base"
BLEND_DEFAULT = CANONICAL_ROOT / "compositor_base.blend"


def main() -> None:
    existing = os.environ.get("COMPOSITOR_EXISTING_EXR", "")
    out = os.environ.get("COMPOSITOR_OUTPUT_DIR", "")
    if not (existing and out):
        raise ValueError("Required env vars: COMPOSITOR_EXISTING_EXR, COMPOSITOR_OUTPUT_DIR")
    blend = os.environ.get("COMPOSITOR_BLEND_PATH", str(BLEND_DEFAULT))
    scene = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")

    config = FastRunnerConfig(
        name=NAME,
        blend_path=Path(blend),
        scene_name=scene,
        exr_inputs={
            "Current Base Outputs :: EXR Existing": Path(existing),
        },
        file_output_node="Current Base Outputs :: Outputs",
        output_dir=Path(out),
        rebuild_file_output=True,
    )
    run_fast_render(config)


if __name__ == "__main__":
    main()
