"""Fast runner for proposal_only_layers.blend.

Single-input runner: one state EXR. Run once per branch.

Environment variables:
    COMPOSITOR_EXR           state EXR (required)
    COMPOSITOR_OUTPUT_DIR    output directory (required)
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

NAME = "render_current_proposal_only"
BLEND_DEFAULT = CANONICAL_ROOT / "proposal_only_layers.blend"


def main() -> None:
    exr = os.environ.get("COMPOSITOR_EXR", "")
    out = os.environ.get("COMPOSITOR_OUTPUT_DIR", "")
    if not (exr and out):
        raise ValueError("Required env vars: COMPOSITOR_EXR, COMPOSITOR_OUTPUT_DIR")
    blend = os.environ.get("COMPOSITOR_BLEND_PATH", str(BLEND_DEFAULT))
    scene = os.environ.get("COMPOSITOR_SCENE_NAME", "ProposalOnly")

    config = FastRunnerConfig(
        name=NAME,
        blend_path=Path(blend),
        scene_name=scene,
        exr_inputs={"EXR": Path(exr)},
        file_output_node="ProposalOnlyOutput",
        output_dir=Path(out),
    )
    run_fast_render(config)


if __name__ == "__main__":
    main()
