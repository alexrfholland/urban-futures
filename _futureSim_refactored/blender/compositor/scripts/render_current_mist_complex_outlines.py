"""Thin runner for compositor_mist_complex_outlines.blend.

Single-input runner: one state EXR. Run once per branch.

Environment variables:
    COMPOSITOR_EXR           state EXR to render (required)
    COMPOSITOR_OUTPUT_DIR    output directory (required)

Output naming:
    The canonical blend owns one semantic output slot:
        whole_forest_outline_v8_t10

    At runtime the runner derives the actual PNG stem from the EXR filename:
        <state>__whole_forest_outline_v8_t10.png
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import bpy

_THIS = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _fast_runner_core import CANONICAL_ROOT, FastRunnerConfig, run_fast_render  # noqa: E402

NAME = "render_current_mist_complex_outlines"
BLEND_DEFAULT = CANONICAL_ROOT / "compositor_mist_complex_outlines.blend"


def derive_output_name(exr_path: Path) -> str:
    stem = re.sub(r"__(?:2k|4k|8k)(?:64s)?$", "", exr_path.stem)
    parts = stem.split("__", 1)
    derived = parts[1] if len(parts) == 2 else stem
    if not derived:
        raise ValueError(f"Could not derive output stem from EXR path: {exr_path}")
    return f"{derived}__whole_forest_outline_v8_t10_"


def main() -> None:
    exr = os.environ.get("COMPOSITOR_EXR", "")
    out = os.environ.get("COMPOSITOR_OUTPUT_DIR", "")
    if not (exr and out):
        raise ValueError("Required env vars: COMPOSITOR_EXR, COMPOSITOR_OUTPUT_DIR")
    exr_path = Path(exr)
    blend = os.environ.get("COMPOSITOR_BLEND_PATH", str(BLEND_DEFAULT))
    scene = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")
    derived_slot_path = derive_output_name(exr_path)

    def extra_setup(tree: bpy.types.NodeTree) -> None:
        output = tree.nodes.get("MistComplexOutlines::Outputs")
        if output is None:
            raise ValueError("Missing node: 'MistComplexOutlines::Outputs'")
        if len(output.file_slots) != 1:
            raise RuntimeError(
                f"MistComplexOutlines::Outputs expected 1 slot, found {len(output.file_slots)}"
            )
        output.file_slots[0].path = derived_slot_path

    config = FastRunnerConfig(
        name=NAME,
        blend_path=Path(blend),
        scene_name=scene,
        exr_inputs={"MistComplexOutlines::EXR Input": exr_path},
        file_output_node="MistComplexOutlines::Outputs",
        output_dir=Path(out),
        extra_setup=extra_setup,
        rebuild_file_output=True,
    )
    run_fast_render(config)


if __name__ == "__main__":
    main()
