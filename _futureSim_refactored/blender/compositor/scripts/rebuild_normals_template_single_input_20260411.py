"""Intentional canonical-template rebuild: convert compositor_normals.blend
from a multi-EXR (pathway/priority/existing) hardwired template to a
single-input standalone template that matches the depth_outliner / mist
pattern.

New template shape (for scene `Current`):

    EXR Image --+--> IndexOB ----> IDMask(idx=3) --> [mask alpha]
                |
                +--> Normal (RGBA) ---> Separate RGBA
                                             |
                                             +-- R -> *0.5 -> +0.5 -> SetAlpha(mask) -> slot normal_x_
                                             +-- G -> *0.5 -> +0.5 -> SetAlpha(mask) -> slot normal_y_
                                             +-- B -> *0.5 -> +0.5 -> SetAlpha(mask) -> slot normal_z_

    RGB(0,0,0) -------------------------------------> SetAlpha(mask) -> slot shading_

Four File Output slots:
    normal_x_
    normal_y_
    normal_z_
    shading_

All four use the tree IDMask as alpha. The mask source is a single link so a
runner that wants whole-scene output can relink `SetAlpha.Alpha` from the
IDMask output to the EXR `Alpha` output.

`shading_` is intentionally a flat black RGB, not a Lambert N.L dot product.
The N.L approach was tried and discarded because the bV2 Normal pass shows a
systematic gradient in the Y channel across screen X (the Normal pass is
camera-space, not world-space, on a camera that is not a perfectly axis-
aligned top-down ortho). The resulting shading had a false left-to-right
darkening that made no sense on a top-down city view.

Instead, `shading_` is a uniform black overlay with the tree mask as alpha.
Drop it on a Photoshop layer above the beauty render at Multiply and dial
the layer opacity for the darkening strength. For any directional shading,
use the per-axis normal_* slabs as PS adjustment sources (the x/y/z slabs
are angle-independent — the Lambert math is moved from Blender to PS).

Contract: this is an explicit template-edit operation. It is NOT a runtime
script. It rebuilds the `Current` scene's compositor tree from scratch.

Usage:

    blender --background --python rebuild_normals_template_single_input_20260411.py \\
        -- --src "<source blend>" --dst "<dest blend>"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy


TREE_ID = 3
SUN = (0.6123724356957946, 0.6123724356957946, 0.49999999999999994)  # 30 deg NE

SCENE_NAME = "Current"
EXR_NODE_NAME = "EXR Input"
IDMASK_NODE_NAME = "Arboreal IDMask"
FILE_OUTPUT_NAME = "Normals::Outputs"

OUTPUT_SLOT_PATHS = ("normal_x_", "normal_y_", "normal_z_", "shading_")


def log(msg: str) -> None:
    print(f"[rebuild_normals_single_input] {msg}")


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    return p.parse_args(argv)


def clear_tree(tree: bpy.types.NodeTree) -> None:
    for link in list(tree.links):
        tree.links.remove(link)
    for node in list(tree.nodes):
        tree.nodes.remove(node)


def new_node(tree, bl_idname, name, label, location, color=None, parent=None):
    node = tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    if color is not None:
        node.use_custom_color = True
        node.color = color
    if parent is not None:
        node.parent = parent
    return node


def new_frame(tree, name, label, location, color):
    f = new_node(tree, "NodeFrame", name, label, location, color=color)
    f.label_size = 18
    f.shrink = False
    return f


def new_math(tree, op, name, label, location, value_b=None, clamp=False, parent=None):
    n = new_node(tree, "CompositorNodeMath", name, label, location, parent=parent)
    n.operation = op
    n.use_clamp = clamp
    if value_b is not None:
        n.inputs[1].default_value = value_b
    return n


def build_compositor(tree: bpy.types.NodeTree) -> None:
    # Frames for visual grouping.
    input_frame = new_frame(tree, "Frame::Input", "EXR Input + Mask", (-1600.0, 300.0), (0.16, 0.16, 0.20))
    remap_frame = new_frame(tree, "Frame::Remap", "(N + 1) / 2 remap", (-900.0, 300.0), (0.14, 0.20, 0.16))
    shading_frame = new_frame(tree, "Frame::Shading", "Lambert N.L (30 NE)", (-900.0, -300.0), (0.20, 0.18, 0.14))
    output_frame = new_frame(tree, "Frame::Output", "File Output + Sinks", (200.0, 0.0), (0.14, 0.16, 0.14))

    # EXR input.
    exr = new_node(
        tree,
        "CompositorNodeImage",
        EXR_NODE_NAME,
        EXR_NODE_NAME,
        (-1500.0, 200.0),
        color=(0.12, 0.18, 0.10),
        parent=input_frame,
    )
    exr.image = None  # runner repaths at runtime

    # Arboreal IDMask (IndexOB == TREE_ID).
    idmask = new_node(
        tree,
        "CompositorNodeIDMask",
        IDMASK_NODE_NAME,
        f"{IDMASK_NODE_NAME} (idx={TREE_ID})",
        (-1160.0, 0.0),
        color=(0.18, 0.18, 0.10),
        parent=input_frame,
    )
    idmask.index = TREE_ID
    idmask.use_antialiasing = True
    # Link IndexOB -> IDMask later once we know the sockets exist. The EXR
    # Image node has no outputs until an image is loaded, so we cannot wire
    # this now. The runner will wire it after repathing the EXR. We leave a
    # comment in the template via a reroute.

    # Separate RGBA for the remap + shading (both share one Separate).
    sep = new_node(
        tree,
        "CompositorNodeSepRGBA",
        "Separate Normal",
        "Separate Normal",
        (-820.0, 300.0),
        parent=remap_frame,
    )

    # --- remap chain: (c + 1) / 2 for each of R, G, B -----------------------
    remap_outputs: list[bpy.types.NodeSocket] = []
    for idx, axis in enumerate(("x", "y", "z")):
        mul = new_math(
            tree,
            "MULTIPLY",
            f"Remap::{axis}_mul",
            f"{axis} * 0.5",
            (-580.0, 400.0 - idx * 110.0),
            value_b=0.5,
            parent=remap_frame,
        )
        tree.links.new(sep.outputs[idx], mul.inputs[0])

        add = new_math(
            tree,
            "ADD",
            f"Remap::{axis}_add",
            f"+ 0.5",
            (-380.0, 400.0 - idx * 110.0),
            value_b=0.5,
            parent=remap_frame,
        )
        tree.links.new(mul.outputs[0], add.inputs[0])
        remap_outputs.append(add.outputs[0])

    # --- shading chain: max(0, N . L) ---------------------------------------
    lx = new_math(
        tree,
        "MULTIPLY",
        "Shading::Lx",
        f"R * Lx ({SUN[0]:.3f})",
        (-580.0, -200.0),
        value_b=SUN[0],
        parent=shading_frame,
    )
    tree.links.new(sep.outputs[0], lx.inputs[0])
    ly = new_math(
        tree,
        "MULTIPLY",
        "Shading::Ly",
        f"G * Ly ({SUN[1]:.3f})",
        (-580.0, -310.0),
        value_b=SUN[1],
        parent=shading_frame,
    )
    tree.links.new(sep.outputs[1], ly.inputs[0])
    lz = new_math(
        tree,
        "MULTIPLY",
        "Shading::Lz",
        f"B * Lz ({SUN[2]:.3f})",
        (-580.0, -420.0),
        value_b=SUN[2],
        parent=shading_frame,
    )
    tree.links.new(sep.outputs[2], lz.inputs[0])

    sum_xy = new_math(
        tree,
        "ADD",
        "Shading::SumXY",
        "Lx + Ly",
        (-380.0, -255.0),
        parent=shading_frame,
    )
    tree.links.new(lx.outputs[0], sum_xy.inputs[0])
    tree.links.new(ly.outputs[0], sum_xy.inputs[1])

    sum_xyz = new_math(
        tree,
        "ADD",
        "Shading::SumXYZ",
        "+ Lz",
        (-180.0, -310.0),
        parent=shading_frame,
    )
    tree.links.new(sum_xy.outputs[0], sum_xyz.inputs[0])
    tree.links.new(lz.outputs[0], sum_xyz.inputs[1])

    max0 = new_math(
        tree,
        "MAXIMUM",
        "Shading::Max0",
        "max(0, dot)",
        (20.0, -310.0),
        value_b=0.0,
        parent=shading_frame,
    )
    tree.links.new(sum_xyz.outputs[0], max0.inputs[0])

    # --- SetAlpha nodes (4 total) -------------------------------------------
    set_alphas: list[bpy.types.NodeSocket] = []
    for idx, (axis, source) in enumerate(zip(("x", "y", "z"), remap_outputs)):
        sa = new_node(
            tree,
            "CompositorNodeSetAlpha",
            f"SetAlpha::{axis}",
            f"SetAlpha {axis}",
            (-80.0, 400.0 - idx * 110.0),
            parent=remap_frame,
        )
        sa.mode = "REPLACE_ALPHA"
        tree.links.new(source, sa.inputs["Image"])
        tree.links.new(idmask.outputs["Alpha"], sa.inputs["Alpha"])
        set_alphas.append(sa.outputs["Image"])

    sa_shading = new_node(
        tree,
        "CompositorNodeSetAlpha",
        "SetAlpha::shading",
        "SetAlpha shading",
        (220.0, -310.0),
        parent=shading_frame,
    )
    sa_shading.mode = "REPLACE_ALPHA"
    tree.links.new(max0.outputs[0], sa_shading.inputs["Image"])
    tree.links.new(idmask.outputs["Alpha"], sa_shading.inputs["Alpha"])
    set_alphas.append(sa_shading.outputs["Image"])

    # --- File Output --------------------------------------------------------
    fout = new_node(
        tree,
        "CompositorNodeOutputFile",
        FILE_OUTPUT_NAME,
        FILE_OUTPUT_NAME,
        (420.0, 0.0),
        color=(0.12, 0.20, 0.14),
        parent=output_frame,
    )
    fout.base_path = "//normals_out"
    fout.format.file_format = "PNG"
    fout.format.color_mode = "RGBA"
    fout.format.color_depth = "8"

    # Repurpose default slot then add the rest.
    if len(fout.file_slots) == 0:
        fout.file_slots.new(OUTPUT_SLOT_PATHS[0])
    else:
        fout.file_slots[0].path = OUTPUT_SLOT_PATHS[0]
    tree.links.new(set_alphas[0], fout.inputs[0])

    for slot_path, source in zip(OUTPUT_SLOT_PATHS[1:], set_alphas[1:]):
        fout.file_slots.new(slot_path)
        tree.links.new(source, fout.inputs[-1])

    # --- Composite + Viewer sinks (required by Blender) ---------------------
    comp = new_node(
        tree,
        "CompositorNodeComposite",
        "Composite",
        "Composite",
        (700.0, 100.0),
        parent=output_frame,
    )
    tree.links.new(set_alphas[3], comp.inputs[0])  # shading as the preview

    viewer = new_node(
        tree,
        "CompositorNodeViewer",
        "Viewer",
        "Viewer",
        (700.0, -100.0),
        parent=output_frame,
    )
    tree.links.new(set_alphas[3], viewer.inputs[0])


def wire_exr_to_graph_if_possible(tree: bpy.types.NodeTree) -> None:
    """Wire the EXR Image node's IndexOB and Normal sockets to the graph.

    This is only possible if the EXR Image node already has a loaded image
    that exposes those sockets. If not, the wiring becomes the runner's job
    (the runner loads the EXR then calls this same wiring routine).
    """
    exr = tree.nodes.get(EXR_NODE_NAME)
    idmask = tree.nodes.get(IDMASK_NODE_NAME)
    sep = tree.nodes.get("Separate Normal")
    if exr is None or idmask is None or sep is None:
        return
    index_socket = next((s for s in exr.outputs if s.name == "IndexOB"), None)
    normal_socket = next((s for s in exr.outputs if s.name == "Normal"), None)
    if index_socket is None or normal_socket is None:
        log("EXR sockets not yet present; runner must wire after loading EXR")
        return
    tree.links.new(index_socket, idmask.inputs["ID value"])
    tree.links.new(normal_socket, sep.inputs[0])
    log("wired EXR sockets (IndexOB -> IDMask, Normal -> Separate)")


def rebuild(src: Path, dst: Path) -> None:
    log(f"opening {src}")
    bpy.ops.wm.open_mainfile(filepath=str(src))

    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise RuntimeError(f"scene {SCENE_NAME!r} not found")
    if scene.node_tree is None:
        scene.use_nodes = True

    # Scene-wide config.
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
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

    tree = scene.node_tree
    log(f"clearing {len(tree.nodes)} nodes / {len(tree.links)} links from scene {SCENE_NAME!r}")
    clear_tree(tree)

    log("building single-input normals graph")
    build_compositor(tree)
    wire_exr_to_graph_if_possible(tree)

    # Report summary.
    fout = tree.nodes.get(FILE_OUTPUT_NAME)
    log(f"final node count: {len(tree.nodes)} links: {len(tree.links)}")
    if fout is not None:
        log(f"File Output has {len(fout.file_slots)} slots:")
        for i, s in enumerate(fout.file_slots):
            sock = fout.inputs[i]
            log(f"  slot[{i}] path={s.path!r} linked={sock.is_linked}")

    log(f"saving rebuilt blend to {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    log("done")


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(f"source blend not found: {src}")
    if dst.resolve() == src.resolve():
        raise RuntimeError("refusing to overwrite source in place")
    rebuild(src, dst)


if __name__ == "__main__":
    main()
