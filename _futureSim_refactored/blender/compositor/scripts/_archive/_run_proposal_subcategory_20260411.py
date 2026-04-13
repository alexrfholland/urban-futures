"""Dev runner for the v4 proposal-subcategory dev blends.

Target blends (all in temp_blends/template_development/...recruit_smalls_fix/):
  - proposal_only_layers_v4_subcategories.blend
  - proposal_outline_layers_v4_subcategories.blend
  - proposal_colored_depth_outlines_v4_subcategories.blend

Each blend has:
  - one CompositorNodeImage named "EXR"
  - one main CompositorNodeOutputFile with 7 file_slots (one per proposal category)
  - optional stray "_orphan_File_Output" nodes that this runner mutes

This runner is intentionally thin. It repaths the single EXR input, points the
main output node at COMPOSITOR_OUTPUT_DIR, applies a COMPOSITOR_FILENAME_PREFIX
to every file slot so successive state runs in the same folder don't collide,
and renders. It never saves the blend, so each invocation is fully isolated
and the working-copy blend on disk stays clean.

Env vars (all required except FILENAME_PREFIX):
  COMPOSITOR_BLEND_PATH       working copy of one dev blend
  COMPOSITOR_OUTPUT_DIR       flat directory for PNG outputs
  COMPOSITOR_EXR              EXR feeding the single 'EXR' image node
  COMPOSITOR_OUTPUT_NODE_NAME name of the main CompositorNodeOutputFile
  COMPOSITOR_SCENE_NAME       scene whose node_tree to operate on
  COMPOSITOR_FILENAME_PREFIX  prefix prepended as "<prefix>__" to every slot
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
from _exr_header import read_exr_dimensions  # noqa: E402


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"env var {name} not set")
    return value


BLEND_PATH = Path(_required_env("COMPOSITOR_BLEND_PATH")).expanduser()
OUTPUT_DIR = Path(_required_env("COMPOSITOR_OUTPUT_DIR")).expanduser()
EXR_PATH = Path(_required_env("COMPOSITOR_EXR")).expanduser()
OUTPUT_NODE_NAME = _required_env("COMPOSITOR_OUTPUT_NODE_NAME")
SCENE_NAME = _required_env("COMPOSITOR_SCENE_NAME")
FILENAME_PREFIX = os.environ.get("COMPOSITOR_FILENAME_PREFIX", "").strip()


def log(message: str) -> None:
    print(f"[run_proposal_subcategory] {message}")


def set_standard_view(scene: bpy.types.Scene) -> None:
    try:
        scene.display_settings.display_device = "sRGB"
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def strip_prefix(raw: str) -> str:
    """Drop trailing underscore and any pre-existing '<word>__' prefix.

    The blend's on-disk slot paths use single underscores internally (e.g.
    'proposal-only_release-control_'), so splitting on '__' safely separates
    an adhoc state prefix from the canonical stem.
    """
    raw = raw.rstrip("_")
    return raw.split("__", 1)[1] if "__" in raw else raw


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(f"blend: {BLEND_PATH}")
    if not EXR_PATH.exists():
        raise FileNotFoundError(f"exr: {EXR_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))

    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"scene {SCENE_NAME!r} missing or has no node_tree")
    node_tree = scene.node_tree

    exr_node = node_tree.nodes.get("EXR")
    if exr_node is None or exr_node.bl_idname != "CompositorNodeImage":
        raise ValueError("expected CompositorNodeImage named 'EXR'")
    if exr_node.image is None:
        raise ValueError("'EXR' node has no image datablock")
    exr_node.image.filepath = str(EXR_PATH)
    exr_node.image.reload()

    out_node = node_tree.nodes.get(OUTPUT_NODE_NAME)
    if out_node is None or out_node.bl_idname != "CompositorNodeOutputFile":
        raise ValueError(f"expected CompositorNodeOutputFile named {OUTPUT_NODE_NAME!r}")

    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile" and node.name != OUTPUT_NODE_NAME:
            node.mute = True
    out_node.mute = False

    width, height = read_exr_dimensions(str(EXR_PATH))
    scene.use_nodes = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    set_standard_view(scene)

    out_node.base_path = str(OUTPUT_DIR)
    out_node.format.file_format = "PNG"
    out_node.format.color_mode = "RGBA"
    out_node.format.color_depth = "8"

    for slot in out_node.file_slots:
        stem = strip_prefix(slot.path)
        if FILENAME_PREFIX:
            slot.path = f"{FILENAME_PREFIX}__{stem}_"
        else:
            slot.path = f"{stem}_"

    scene.render.filepath = str(OUTPUT_DIR / "_discard_render.png")
    bpy.ops.render.render(write_still=True, scene=scene.name)

    for rendered in sorted(OUTPUT_DIR.glob("*_0001.png")):
        final = rendered.with_name(rendered.name.replace("_0001", ""))
        if final.exists():
            final.unlink()
        rendered.replace(final)

    discard = OUTPUT_DIR / "_discard_render.png"
    if discard.exists():
        discard.unlink()

    log(f"done: {OUTPUT_DIR}  prefix={FILENAME_PREFIX or '(none)'}  exr={EXR_PATH.name}")


if __name__ == "__main__":
    main()
