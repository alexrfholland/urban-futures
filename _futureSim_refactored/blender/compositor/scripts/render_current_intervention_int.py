"""Fast runner for compositor_intervention_int.blend.

Single-input runner. Wires one bioenvelope EXR into the canonical
intervention_int graph, including the AOV socket prefixed
'intervention_bioenvelope_ply' (Blender truncates the full AOV name).

Environment variables:
    COMPOSITOR_EXR           bioenvelope EXR path (required)
    COMPOSITOR_OUTPUT_DIR    output directory (required)
    COMPOSITOR_BLEND_PATH    optional override for the canonical blend
    COMPOSITOR_SCENE_NAME    optional scene override (default: Current)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy

_THIS = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _fast_runner_core import CANONICAL_ROOT, FastRunnerConfig, run_fast_render  # noqa: E402

NAME = "render_current_intervention_int"
BLEND_DEFAULT = CANONICAL_ROOT / "compositor_intervention_int.blend"
SCENE_DEFAULT = "Current"
EXR_NODE = "InterventionInt::EXR Input"
RAW_HUB_NODE = "InterventionInt::Raw"
FILE_OUTPUT_NODE = "InterventionInt::Outputs"
AOV_SOCKET_PREFIX = "intervention_bioenvelope_ply"


def wire_aov(tree: bpy.types.NodeTree) -> None:
    """Reconnect the truncated AOV socket on the EXR node to the raw hub."""
    exr_node = tree.nodes.get(EXR_NODE)
    raw_hub = tree.nodes.get(RAW_HUB_NODE)
    if exr_node is None or raw_hub is None:
        raise ValueError(f"Missing node: {EXR_NODE!r} or {RAW_HUB_NODE!r}")
    aov = None
    for sock in exr_node.outputs:
        if sock.name.startswith(AOV_SOCKET_PREFIX):
            aov = sock
            break
    if aov is None:
        available = [s.name for s in exr_node.outputs]
        raise ValueError(
            f"No AOV socket starting with {AOV_SOCKET_PREFIX!r}. Available: {available}"
        )
    for link in list(raw_hub.inputs[0].links):
        tree.links.remove(link)
    tree.links.new(aov, raw_hub.inputs[0])
    print(f"[{NAME}] wired AOV {aov.name!r} -> {RAW_HUB_NODE!r}")


def main() -> None:
    exr = os.environ.get("COMPOSITOR_EXR", "")
    out = os.environ.get("COMPOSITOR_OUTPUT_DIR", "")
    blend = os.environ.get("COMPOSITOR_BLEND_PATH", str(BLEND_DEFAULT))
    scene = os.environ.get("COMPOSITOR_SCENE_NAME", SCENE_DEFAULT)
    if not exr:
        raise ValueError("COMPOSITOR_EXR is required")
    if not out:
        raise ValueError("COMPOSITOR_OUTPUT_DIR is required")

    config = FastRunnerConfig(
        name=NAME,
        blend_path=Path(blend),
        scene_name=scene,
        exr_inputs={EXR_NODE: Path(exr)},
        file_output_node=FILE_OUTPUT_NODE,
        output_dir=Path(out),
        extra_setup=wire_aov,
    )
    run_fast_render(config)


if __name__ == "__main__":
    main()
