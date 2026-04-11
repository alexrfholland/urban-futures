"""Modify normals blend to add per-axis (x/y/z) slots and render against baseline EXRs.

Operates on a working copy of compositor_normals.blend (no canonical mutation).

What this does (in-memory + saved back to the working copy):
  1. Repath the 3 EXR Image nodes to baseline EXRs
  2. For each Workflow Group RGBA normal output, add a Separate Color (RGB)
     and 3 Combine Color (RGB) nodes — one per channel — so each axis
     becomes its own RGBA grayscale image (x→R, y→G, z→B), alpha preserved.
  3. Replace the File Output node's slots with 12 entries:
        existing_condition_normal_full_         (combined RGB normal)
        existing_condition_normal_x_            (R channel)
        existing_condition_normal_y_            (G channel)
        existing_condition_normal_z_            (B channel)
        priority_tree_normal_                    (combined)
        priority_tree_normal_x_
        priority_tree_normal_y_
        priority_tree_normal_z_
        pathway_tree_normal_                     (combined)
        pathway_tree_normal_x_
        pathway_tree_normal_y_
        pathway_tree_normal_z_
  4. Add a camera (the blend has none — render pipeline needs one)
  5. Set valid scene render.filepath
  6. Configure resolution from EXR
  7. Render via animation=True (Blender 4.2 File Output workaround)
  8. Rename outputs to drop the trailing _0001
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import bpy

BLEND = Path(os.environ["WORK_BLEND"])
PATHWAY_EXR = Path(os.environ["PATHWAY_EXR"])
PRIORITY_EXR = Path(os.environ["PRIORITY_EXR"])
EXISTING_EXR = Path(os.environ["EXISTING_EXR"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
SCENE_NAME = "Current"

# Mapping from workflow group output socket name to (slot prefix, EXR identifier)
OUTPUT_TO_PREFIX = {
    "to_normals__composite":           "pathway_tree_normal",
    "existing_condition_normal_full":  "existing_condition_normal_full",
    "priority_tree_normal":            "priority_tree_normal",
}


def log(msg: str) -> None:
    print(f"[normals_xyz] {msg}")


def detect_resolution(exr_path: Path, image: bpy.types.Image) -> tuple[int, int]:
    try:
        result = subprocess.run(
            ["oiiotool", "--info", "-v", str(exr_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(r"(\d+)\s*x\s*(\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    width, height = image.size[:]
    if width > 0 and height > 0:
        return width, height
    return 7680, 4320


def repath_exr(node: bpy.types.Node, filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    # Fresh load — reload() leaves size at 0 in background
    node.image = bpy.data.images.load(str(filepath), check_existing=False)
    log(f"repathed {node.name!r} -> {filepath.name}")


def main() -> None:
    bpy.ops.wm.open_mainfile(filepath=str(BLEND))
    scene = bpy.data.scenes[SCENE_NAME]
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    tree = scene.node_tree

    # ---- 1. Repath EXRs ----
    repath_exr(tree.nodes["Normals::EXR Pathway"], PATHWAY_EXR)
    repath_exr(tree.nodes["Normals::EXR Priority"], PRIORITY_EXR)
    repath_exr(tree.nodes["Normals::EXR Existing"], EXISTING_EXR)

    # ---- 2. Find workflow group + its RGBA outputs ----
    wf = tree.nodes["Normals::Workflow Group"]
    log(f"workflow group has {len(wf.outputs)} outputs")
    for i, sock in enumerate(wf.outputs):
        log(f"  output[{i}] {sock.name!r} type={sock.type}")

    # Build per-axis chains (Separate Color + 3 Combine Colors)
    # Layout: place new nodes to the right of the workflow group, stacked
    base_x = wf.location.x + 600
    base_y = wf.location.y
    spacing_y = -350

    axis_combine: dict[str, dict[str, bpy.types.Node]] = {}  # output_name -> {x: node, y:, z:}

    for idx, out_name in enumerate(OUTPUT_TO_PREFIX.keys()):
        out_sock = wf.outputs[out_name]
        anchor_y = base_y + idx * spacing_y

        sep = tree.nodes.new("CompositorNodeSeparateColor")
        sep.name = f"_xyz_sep_{out_name}"
        sep.label = f"Separate {out_name}"
        sep.mode = "RGB"
        sep.location = (base_x, anchor_y)
        tree.links.new(out_sock, sep.inputs[0])

        chans = {}
        for ch_idx, axis in enumerate(("x", "y", "z")):
            comb = tree.nodes.new("CompositorNodeCombineColor")
            comb.name = f"_xyz_comb_{out_name}_{axis}"
            comb.label = f"{out_name} {axis}"
            comb.mode = "RGB"
            comb.location = (base_x + 250, anchor_y + ch_idx * 200)
            # Wire R, G, B all to the same channel value to make a grayscale image
            ch_sock = sep.outputs[ch_idx]  # 0=R, 1=G, 2=B
            tree.links.new(ch_sock, comb.inputs[0])  # Red
            tree.links.new(ch_sock, comb.inputs[1])  # Green
            tree.links.new(ch_sock, comb.inputs[2])  # Blue
            tree.links.new(sep.outputs[3], comb.inputs[3])  # Alpha pass-through
            chans[axis] = comb
        axis_combine[out_name] = chans
        log(f"built x/y/z chain for {out_name!r}")

    # ---- 3. Build fresh File Output node (Blender 4.2 saved-File-Output bug
    #         workaround: modifying a saved node's slots silently fails to fire
    #         in background render — must create a new in-memory node) ----
    old_fout = tree.nodes.get("Normals::Outputs")
    if old_fout is not None:
        for i in range(len(old_fout.inputs)):
            for link in list(old_fout.inputs[i].links):
                tree.links.remove(link)
        old_fout.mute = True
        old_fout.name = "_orphan_Normals::Outputs"
        old_fout.label = "_orphan"
    fout = tree.nodes.new("CompositorNodeOutputFile")
    fout.name = "Normals::Outputs"
    fout.label = "Normals::Outputs"
    fout.location = (old_fout.location[0], old_fout.location[1] - 50) if old_fout else (1800, 0)

    # Slot order: existing combined+x/y/z, priority combined+x/y/z, pathway combined+x/y/z
    slot_specs: list[tuple[str, bpy.types.NodeSocket]] = []

    def add_group(out_name: str, prefix: str):
        slot_specs.append((f"{prefix}_", wf.outputs[out_name]))
        for axis in ("x", "y", "z"):
            slot_specs.append((f"{prefix}_{axis}_", axis_combine[out_name][axis].outputs[0]))

    add_group("existing_condition_normal_full", "existing_condition_normal_full")
    add_group("priority_tree_normal", "priority_tree_normal")
    add_group("to_normals__composite", "pathway_tree_normal")

    fout.file_slots[0].path = slot_specs[0][0]
    for path, _ in slot_specs[1:]:
        fout.file_slots.new(path)
    for i, (path, src) in enumerate(slot_specs):
        fout.file_slots[i].path = path
        tree.links.new(src, fout.inputs[i])
        log(f"slot[{i}] {path!r} <- {src.node.name!r}.{src.name!r}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fout.base_path = str(OUTPUT_DIR)
    fout.format.file_format = "PNG"
    fout.format.color_mode = "RGBA"
    fout.format.color_depth = "8"

    # ---- 4. Camera ----
    if scene.camera is None:
        cam_data = bpy.data.cameras.new("FixCamera")
        cam = bpy.data.objects.new("FixCamera", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
        log("created and assigned FixCamera")

    # ---- 5. Scene render settings ----
    width, height = detect_resolution(PATHWAY_EXR, tree.nodes["Normals::EXR Pathway"].image)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
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
    scene.render.filepath = str(OUTPUT_DIR / "_discard_render_")
    log(f"scene render config: {width}x{height} -> {OUTPUT_DIR}")

    # Wire Composite sink to first existing output (compositor needs it to evaluate)
    existing_composite = tree.nodes.get("Normals::Composite")
    if existing_composite is not None:
        for link in list(existing_composite.inputs[0].links):
            tree.links.remove(link)
        tree.links.new(wf.outputs["to_normals__composite"], existing_composite.inputs[0])
        log("wired Normals::Composite <- workflow group")

    # Save modified blend (working copy only)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND), copy=False)
    log(f"saved working copy {BLEND.name}")

    # ---- 6. Render ----
    log("rendering animation=True ...")
    bpy.ops.render.render(animation=True)
    log("render complete")

    # ---- 7. Rename outputs ----
    for rendered in sorted(OUTPUT_DIR.glob("*_0001.png")):
        final = rendered.with_name(rendered.name.replace("_0001", ""))
        if final.exists():
            final.unlink()
        rendered.replace(final)
        log(f"renamed {rendered.name} -> {final.name}")

    discard = OUTPUT_DIR / "_discard_render_.png"
    for d in OUTPUT_DIR.glob("_discard_render_*"):
        d.unlink()


if __name__ == "__main__":
    main()
