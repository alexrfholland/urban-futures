"""Build a development-only normals assessment blend.

This is intentionally not canonical. It creates one single-input compositor
blend that renders:

- normal_x
- normal_y
- normal_z
- ambient-lifted Lambert shading for a left/right angle band

The point is to obey the compositor contract shape for assessment work:
one EXR per compositor run, all owned outputs from one render call.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
OUTPUT_BLEND = (
    REPO_ROOT
    / "_data-refactored"
    / "compositor"
    / "temp_blends"
    / "template_development"
    / "compositor_normals_single_input_assess_ambient20_band__20260414.blend"
)

TREE_ID = 3
AMBIENT_FLOOR = 0.2
GAMMA = 0.85
ANGLES = (10, 20, 30, 40, 50, 60, 70, 80)

SCENE_NAME = "Current"
EXR_NODE_NAME = "EXR Input"
IDMASK_NODE_NAME = "Arboreal IDMask"
FILE_OUTPUT_NAME = "NormalsAssess::Outputs"
OUTPUT_SLOT_PATHS = ("normal_x_", "normal_y_", "normal_z_")


def log(msg: str) -> None:
    print(f"[_build_normals_assess_ambient20_band] {msg}")


def normalize(x: float, y: float, z: float) -> tuple[float, float, float]:
    length = math.sqrt(x * x + y * y + z * z)
    return (x / length, y / length, z / length)


def angle_vector(direction: str, angle_deg: int) -> tuple[float, float, float]:
    elev = math.radians(angle_deg)
    horiz = math.cos(elev)
    z = math.sin(elev)
    x = horiz / math.sqrt(2.0)
    y = horiz / math.sqrt(2.0)
    if direction == "left":
        x = -x
    return normalize(x, y, z)


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
    frame = new_node(tree, "NodeFrame", name, label, location, color=color)
    frame.label_size = 18
    frame.shrink = False
    return frame


def new_math(tree, op, name, label, location, value_b=None, clamp=False, parent=None):
    node = new_node(tree, "CompositorNodeMath", name, label, location, parent=parent)
    node.operation = op
    node.use_clamp = clamp
    if value_b is not None:
        node.inputs[1].default_value = value_b
    return node


def build_compositor(tree: bpy.types.NodeTree) -> None:
    input_frame = new_frame(tree, "Frame::Input", "EXR Input + Mask", (-1900.0, 320.0), (0.16, 0.16, 0.20))
    remap_frame = new_frame(tree, "Frame::Remap", "(N + 1) / 2 remap", (-1200.0, 320.0), (0.14, 0.20, 0.16))
    shading_frame = new_frame(tree, "Frame::ShadingBand", "Ambient20 Gamma085 Lambert Angle Band", (-1180.0, -980.0), (0.20, 0.18, 0.14))
    output_frame = new_frame(tree, "Frame::Output", "File Output + Sinks", (980.0, -60.0), (0.14, 0.16, 0.14))

    exr = new_node(
        tree,
        "CompositorNodeImage",
        EXR_NODE_NAME,
        EXR_NODE_NAME,
        (-1800.0, 210.0),
        color=(0.12, 0.18, 0.10),
        parent=input_frame,
    )
    exr.image = None

    idmask = new_node(
        tree,
        "CompositorNodeIDMask",
        IDMASK_NODE_NAME,
        f"{IDMASK_NODE_NAME} (idx={TREE_ID})",
        (-1460.0, 10.0),
        color=(0.18, 0.18, 0.10),
        parent=input_frame,
    )
    idmask.index = TREE_ID
    idmask.use_antialiasing = True

    sep = new_node(
        tree,
        "CompositorNodeSepRGBA",
        "Separate Normal",
        "Separate Normal",
        (-1120.0, 320.0),
        parent=remap_frame,
    )

    remap_outputs: list[bpy.types.NodeSocket] = []
    for idx, axis in enumerate(("x", "y", "z")):
        y = 420.0 - idx * 120.0
        mul = new_math(
            tree,
            "MULTIPLY",
            f"Remap::{axis}_mul",
            f"{axis} * 0.5",
            (-860.0, y),
            value_b=0.5,
            parent=remap_frame,
        )
        tree.links.new(sep.outputs[idx], mul.inputs[0])
        add = new_math(
            tree,
            "ADD",
            f"Remap::{axis}_add",
            f"+ 0.5",
            (-640.0, y),
            value_b=0.5,
            parent=remap_frame,
        )
        tree.links.new(mul.outputs[0], add.inputs[0])
        remap_outputs.append(add.outputs[0])

    output_images: list[tuple[str, bpy.types.NodeSocket]] = []
    for idx, (axis, source) in enumerate(zip(("x", "y", "z"), remap_outputs)):
        y = 420.0 - idx * 120.0
        set_alpha = new_node(
            tree,
            "CompositorNodeSetAlpha",
            f"SetAlpha::{axis}",
            f"SetAlpha {axis}",
            (-400.0, y),
            parent=remap_frame,
        )
        set_alpha.mode = "REPLACE_ALPHA"
        tree.links.new(source, set_alpha.inputs["Image"])
        tree.links.new(idmask.outputs["Alpha"], set_alpha.inputs["Alpha"])
        output_images.append((OUTPUT_SLOT_PATHS[idx], set_alpha.outputs["Image"]))

    preview_source = None
    base_y = -160.0
    row_gap = 160.0
    col_gap = 520.0
    for col, direction in enumerate(("left", "right")):
        for row, angle in enumerate(ANGLES):
            sun = angle_vector(direction, angle)
            x = -980.0 + col * col_gap
            y = base_y - row * row_gap
            tag = f"ambient20_gamma085_{direction}_{angle}deg"

            lx = new_math(
                tree,
                "MULTIPLY",
                f"Shading::{tag}::Lx",
                f"{tag} R*Lx",
                (x, y),
                value_b=sun[0],
                parent=shading_frame,
            )
            ly = new_math(
                tree,
                "MULTIPLY",
                f"Shading::{tag}::Ly",
                f"{tag} G*Ly",
                (x, y - 45.0),
                value_b=sun[1],
                parent=shading_frame,
            )
            lz = new_math(
                tree,
                "MULTIPLY",
                f"Shading::{tag}::Lz",
                f"{tag} B*Lz",
                (x, y - 90.0),
                value_b=sun[2],
                parent=shading_frame,
            )
            tree.links.new(sep.outputs[0], lx.inputs[0])
            tree.links.new(sep.outputs[1], ly.inputs[0])
            tree.links.new(sep.outputs[2], lz.inputs[0])

            sum_xy = new_math(
                tree,
                "ADD",
                f"Shading::{tag}::SumXY",
                f"{tag} xy",
                (x + 180.0, y - 20.0),
                parent=shading_frame,
            )
            sum_xyz = new_math(
                tree,
                "ADD",
                f"Shading::{tag}::SumXYZ",
                f"{tag} xyz",
                (x + 360.0, y - 45.0),
                parent=shading_frame,
            )
            max0 = new_math(
                tree,
                "MAXIMUM",
                f"Shading::{tag}::Max0",
                f"{tag} max0",
                (x + 540.0, y - 45.0),
                value_b=0.0,
                parent=shading_frame,
            )
            lift_mul = new_math(
                tree,
                "MULTIPLY",
                f"Shading::{tag}::LiftMul",
                f"{tag} *0.8",
                (x + 720.0, y - 45.0),
                value_b=1.0 - AMBIENT_FLOOR,
                parent=shading_frame,
            )
            lift_add = new_math(
                tree,
                "ADD",
                f"Shading::{tag}::LiftAdd",
                f"{tag} +0.2",
                (x + 900.0, y - 45.0),
                value_b=AMBIENT_FLOOR,
                parent=shading_frame,
            )
            gamma = new_math(
                tree,
                "POWER",
                f"Shading::{tag}::Gamma",
                f"{tag} ^0.85",
                (x + 1080.0, y - 45.0),
                value_b=GAMMA,
                parent=shading_frame,
            )
            set_alpha = new_node(
                tree,
                "CompositorNodeSetAlpha",
                f"SetAlpha::{tag}",
                f"SetAlpha {tag}",
                (x + 1280.0, y - 45.0),
                parent=shading_frame,
            )
            set_alpha.mode = "REPLACE_ALPHA"

            tree.links.new(lx.outputs[0], sum_xy.inputs[0])
            tree.links.new(ly.outputs[0], sum_xy.inputs[1])
            tree.links.new(sum_xy.outputs[0], sum_xyz.inputs[0])
            tree.links.new(lz.outputs[0], sum_xyz.inputs[1])
            tree.links.new(sum_xyz.outputs[0], max0.inputs[0])
            tree.links.new(max0.outputs[0], lift_mul.inputs[0])
            tree.links.new(lift_mul.outputs[0], lift_add.inputs[0])
            tree.links.new(lift_add.outputs[0], gamma.inputs[0])
            tree.links.new(gamma.outputs[0], set_alpha.inputs["Image"])
            tree.links.new(idmask.outputs["Alpha"], set_alpha.inputs["Alpha"])

            slot_path = f"shading__{tag}_"
            output_images.append((slot_path, set_alpha.outputs["Image"]))
            if preview_source is None:
                preview_source = set_alpha.outputs["Image"]

    fout = new_node(
        tree,
        "CompositorNodeOutputFile",
        FILE_OUTPUT_NAME,
        FILE_OUTPUT_NAME,
        (1220.0, 40.0),
        color=(0.12, 0.20, 0.14),
        parent=output_frame,
    )
    fout.base_path = "//normals_assess_out"
    fout.format.file_format = "PNG"
    fout.format.color_mode = "RGBA"
    fout.format.color_depth = "8"

    if len(fout.file_slots) == 0:
        fout.file_slots.new(output_images[0][0])
    else:
        fout.file_slots[0].path = output_images[0][0]
    tree.links.new(output_images[0][1], fout.inputs[0])

    for slot_path, source in output_images[1:]:
        fout.file_slots.new(slot_path)
        tree.links.new(source, fout.inputs[-1])

    comp = new_node(
        tree,
        "CompositorNodeComposite",
        "Composite",
        "Composite",
        (1520.0, 180.0),
        parent=output_frame,
    )
    viewer = new_node(
        tree,
        "CompositorNodeViewer",
        "Viewer",
        "Viewer",
        (1520.0, 20.0),
        parent=output_frame,
    )
    if preview_source is not None:
        tree.links.new(preview_source, comp.inputs[0])
        tree.links.new(preview_source, viewer.inputs[0])


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = SCENE_NAME
    scene.use_nodes = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    try:
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    tree = scene.node_tree
    clear_tree(tree)
    build_compositor(tree)

    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))
    log(f"Saved {OUTPUT_BLEND}")


if __name__ == "__main__":
    main()
