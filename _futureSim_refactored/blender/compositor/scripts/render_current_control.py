"""Fast runner for compositor_control.blend.

Reads the `control` layer from a state EXR and emits a greyscale PNG where
each tree-ownership class maps to a desat amount (0.8 street / 0.5 park /
0.0 reserve / 0.0 improved). Intended as a mask for a Hue/Sat -100
adjustment layer clipped to the sizes group inside a PSB.

Environment variables:
    COMPOSITOR_EXR            state EXR with a `control` layer (required)
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

NAME = "render_current_control"
BLEND_DEFAULT = CANONICAL_ROOT / "compositor_control.blend"


def main() -> None:
    exr = os.environ.get("COMPOSITOR_EXR", "")
    out = os.environ.get("COMPOSITOR_OUTPUT_DIR", "")
    if not (exr and out):
        raise ValueError("Required env vars: COMPOSITOR_EXR, COMPOSITOR_OUTPUT_DIR")
    blend = os.environ.get("COMPOSITOR_BLEND_PATH", str(BLEND_DEFAULT))
    scene = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")

    config = FastRunnerConfig(
        name=NAME,
        blend_path=Path(blend),
        scene_name=scene,
        exr_inputs={
            "Current Control :: EXR": Path(exr),
        },
        file_output_node="Current Control :: Outputs",
        output_dir=Path(out),
        rebuild_file_output=False,
    )
    run_fast_render(config)


if __name__ == "__main__":
    main()
