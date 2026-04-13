"""Thin runner: compositor_ao.blend on parade single-state yr180 EXRs (latest timestamp).

Opens the canonical AO blend, repaths 3 EXR inputs, renders each output
slot individually through the Composite node, exits without saving.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import bpy

_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _exr_header import read_exr_dimensions

CANONICAL_BLEND = (
    REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
    / "canonical_templates" / "compositor_ao.blend"
)
EXR_DIR = (
    REPO_ROOT / "_data-refactored" / "blenderv2" / "output"
    / "20260412_232449_parade_single-state_yr180_8k"
)
EXISTING_EXR = EXR_DIR / "parade_single-state_yr180__existing_condition_positive__8k.exr"
PATHWAY_EXR  = EXR_DIR / "parade_single-state_yr180__positive_state__8k.exr"
PRIORITY_EXR = EXR_DIR / "parade_single-state_yr180__positive_priority_state__8k.exr"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    REPO_ROOT / "_data-refactored" / "compositor" / "outputs"
    / "4.10" / "parade_single-state_yr180" / f"ao__{stamp}"
)


def log(msg: str) -> None:
    print(f"[run_ao_parade] {msg}")


def require_node(tree, name):
    n = tree.nodes.get(name)
    if n is None:
        raise ValueError(f"Missing node: {name}")
    return n


def require_node_by_type(tree, bl_idname):
    for n in tree.nodes:
        if n.bl_idname == bl_idname:
            return n
    raise ValueError(f"Missing node type: {bl_idname}")


def set_standard_view(scene):
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


def render_socket_to_png(scene, tree, socket, out_path):
    composite = require_node_by_type(tree, "CompositorNodeComposite")
    viewer = require_node_by_type(tree, "CompositorNodeViewer")
    for link in list(composite.inputs[0].links):
        tree.links.remove(link)
    for link in list(viewer.inputs[0].links):
        tree.links.remove(link)
    tree.links.new(socket, composite.inputs[0])
    tree.links.new(socket, viewer.inputs[0])
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    log(f"Wrote {out_path}")


def main():
    for p in (EXISTING_EXR, PATHWAY_EXR, PRIORITY_EXR):
        if not p.exists():
            raise FileNotFoundError(p)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(CANONICAL_BLEND))
    scene = bpy.data.scenes.get("Current")
    if scene is None or scene.node_tree is None:
        raise ValueError("Scene 'Current' not found")

    tree = scene.node_tree
    set_standard_view(scene)

    # Repath the 3 EXR inputs
    for node_name, exr_path in [
        ("AO::EXR Existing", EXISTING_EXR),
        ("AO::EXR Pathway", PATHWAY_EXR),
        ("AO::EXR Priority", PRIORITY_EXR),
    ]:
        node = require_node(tree, node_name)
        img = bpy.data.images.load(str(exr_path), check_existing=True)
        node.image = img
        log(f"Loaded {exr_path.name} -> {node_name}")

    # Detect resolution from EXR header
    w, h = read_exr_dimensions(str(EXISTING_EXR))
    log(f"Resolution {w}x{h}")
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1

    # Mute all File Output nodes
    for n in tree.nodes:
        if n.bl_idname == "CompositorNodeOutputFile":
            n.mute = True

    # Render each slot individually through Composite node
    fo = require_node(tree, "AO::Outputs")
    for i, slot in enumerate(fo.file_slots):
        stem = slot.path.rstrip("_")
        if fo.inputs[i].is_linked:
            src_socket = fo.inputs[i].links[0].from_socket
            render_socket_to_png(scene, tree, src_socket, OUTPUT_DIR / f"{stem}.png")
        else:
            log(f"SKIP slot {stem} (unlinked)")

    log("done")


if __name__ == "__main__":
    main()
