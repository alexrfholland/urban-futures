"""Apply all four diagnosed fixes to proposal_colored_depth_outlines_v2.blend.

Fixes applied in this script (saved back to the .blend):
  1. Add a Camera object and assign as scene.camera
  2. Create a Composite sink wired to the same source as File Output
  3. Set scene.render.filepath to a valid Windows path (flush stale Mac path)
  4. Fix File Output slot[0] path from default 'Image' to a stable prefix

Creates (on disk, not in .blend):
  - E:\\2026 Arboreal Futures\\blenderv2\\renders\\alextest\\renders\\
"""
from __future__ import annotations

import os
from pathlib import Path

import bpy

BLEND = Path(os.environ["TARGET_BLEND"])
SCENE_NAME = "ProposalColoredDepthOutlines"

# Where rendered PNGs will land. File Output node's base_path already points
# here per the user's edit; we just make sure the folder exists.
OUTPUT_DIR = Path(r"E:\2026 Arboreal Futures\blenderv2\renders\alextest\renders")

# Scene-level render.filepath — placeholder used by the Composite sink only.
# Must be a valid path on this OS so Blender's render pipeline can initialize.
SCENE_DISCARD = OUTPUT_DIR / "_discard_render_"

# Slot[0] prefix. The source socket is 'Set Alpha.002'.'Image' which the user
# wired up to render the decay pass (per frame mapping). Use an unambiguous
# prefix so the final filename becomes e.g. proposal-depth-colour_decay_0001.png
SLOT0_PREFIX = "proposal-depth-colour_decay_"


def log(msg: str) -> None:
    print(f"[fix_v2] {msg}")


def main() -> None:
    log(f"open {BLEND}")
    bpy.ops.wm.open_mainfile(filepath=str(BLEND))

    scene = bpy.data.scenes[SCENE_NAME]
    tree = scene.node_tree

    # ---- 1. Camera ----
    if scene.camera is None:
        cam = bpy.data.objects.get("FixCamera")
        if cam is None:
            cam_data = bpy.data.cameras.new("FixCamera")
            cam = bpy.data.objects.new("FixCamera", cam_data)
            scene.collection.objects.link(cam)
            log("created FixCamera object")
        scene.camera = cam
        log(f"assigned scene.camera = {cam.name!r}")
    else:
        log(f"scene.camera already set: {scene.camera.name!r}")

    # ---- 2. Composite sink ----
    file_out = next(
        (n for n in tree.nodes if n.bl_idname == "CompositorNodeOutputFile"),
        None,
    )
    if file_out is None:
        raise RuntimeError("expected a CompositorNodeOutputFile in the scene")
    log(f"file output: {file_out.name!r} base_path={file_out.base_path!r}")

    # Find source driving slot[0]
    slot0_sock = file_out.inputs[0]
    if not slot0_sock.is_linked:
        raise RuntimeError("File Output slot[0] is not linked — nothing to render")
    src_sock = slot0_sock.links[0].from_socket
    log(f"source: {slot0_sock.links[0].from_node.name!r}.{src_sock.name!r}")

    composite = next(
        (n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"),
        None,
    )
    if composite is None:
        composite = tree.nodes.new("CompositorNodeComposite")
        composite.name = "Composite"
        composite.location = (1400, -900)
        log("created Composite sink")
    # (re)wire it
    for link in list(composite.inputs["Image"].links):
        tree.links.remove(link)
    tree.links.new(src_sock, composite.inputs["Image"])
    log("wired Composite.Image <- source")

    # ---- 3. Scene render.filepath ----
    old_fp = scene.render.filepath
    scene.render.filepath = str(SCENE_DISCARD)
    log(f"scene.render.filepath: {old_fp!r} -> {scene.render.filepath!r}")

    # ---- 4. File Output slot[0] path ----
    slot0 = file_out.file_slots[0]
    if slot0.path != SLOT0_PREFIX:
        log(f"slot[0] path: {slot0.path!r} -> {SLOT0_PREFIX!r}")
        slot0.path = SLOT0_PREFIX

    # ---- Supporting sanity: render settings ----
    scene.render.resolution_x = 7680
    scene.render.resolution_y = 4320
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1

    # Create the output directory on disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"ensured on disk: {OUTPUT_DIR}")

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND), copy=False)
    log(f"saved {BLEND.name}")


if __name__ == "__main__":
    main()
