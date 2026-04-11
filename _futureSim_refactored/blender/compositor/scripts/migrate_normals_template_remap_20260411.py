"""Intentional canonical-template edit: remap the normals workflow to emit
per-axis grayscale slabs plus a prebaked Lambert shading PNG.

This script is an explicit template-edit operation per the Compositor Template
Contract. It is NOT a runtime render-time script. It modifies compositor graph
structure and saves the result to a target path.

Usage (background mode):

    blender --background \\
        --python migrate_normals_template_remap_20260411.py \\
        -- \\
        --src "<path to compositor_normals.blend>" \\
        --dst "<path to working-copy or canonical destination>"

If --dst is omitted, the script writes next to --src with a `_remapped` suffix.

What this does:

For each of the three normals workflows (pathway, priority, existing), it taps
the alpha-masked Normal RGBA output of `Normals::Workflow Group`, separates
R/G/B/A, and builds four new output chains:

  * x-slab       : (R + 1) / 2   masked
  * y-slab       : (G + 1) / 2   masked
  * z-slab       : (B + 1) / 2   masked
  * shading      : max(0, N . L) masked, where L = normalize(0.3, 0.3, 0.9)

The existing `Normals::Outputs` File Output node is deleted and replaced with a
new one that has 12 slots (3 workflows * 4 outputs), written as 8-bit RGBA PNG
where RGB is the grayscale value and Alpha is the tree mask so Photoshop picks
up the mask automatically on drop.

The inner `normals_workflow_group` node group is NOT modified. All new logic
lives at the top level of the compositor tree in a new frame.

Contract notes:

- target scene is `Current`
- canonical template edit, not a runtime script
- runs on a working copy by default; do not point --dst at the canonical blend
  unless you are intentionally promoting.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy


# Sun vector for the prebaked Lambert shading output. Chosen as a high NE sun
# (roughly 70 degrees elevation) so the baked shading reads as soft daylight.
SUN_RAW = (0.3, 0.3, 0.9)
_SUN_LEN = math.sqrt(sum(c * c for c in SUN_RAW))
SUN = tuple(c / _SUN_LEN for c in SUN_RAW)  # normalized

# Workflow identifier -> (group output socket name, output filename stem)
# NB: the group's pathway output is confusingly named `to_normals__composite`
# because it is also wired to the scene Composite/Viewer sinks.
WORKFLOWS = [
    ("pathway", "to_normals__composite", "pathway_tree_normal"),
    ("priority", "priority_tree_normal", "priority_tree_normal"),
    ("existing", "existing_condition_normal_full", "existing_condition_normal_full"),
]

GROUP_NODE_NAME = "Normals::Workflow Group"
OLD_FILE_OUTPUT_NAME = "Normals::Outputs"
NEW_FILE_OUTPUT_NAME = "Normals::Outputs"
NEW_FRAME_NAME = "Normals::Remap Frame"
NEW_FRAME_LABEL = "Remap + Shading (PS-friendly)"


def log(msg: str) -> None:
    print(f"[migrate_normals_template_remap] {msg}")


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Normals template remap migration")
    parser.add_argument("--src", required=True, help="Path to source .blend")
    parser.add_argument("--dst", default=None, help="Path to write migrated .blend")
    return parser.parse_args(argv)


def new_math(
    tree: bpy.types.NodeTree,
    op: str,
    name: str,
    label: str,
    location: tuple[float, float],
    value_b: float | None = None,
    clamp: bool = False,
    parent: bpy.types.Node | None = None,
) -> bpy.types.Node:
    node = tree.nodes.new("CompositorNodeMath")
    node.operation = op
    node.name = name
    node.label = label
    node.location = location
    node.use_clamp = clamp
    if value_b is not None:
        node.inputs[1].default_value = value_b
    if parent is not None:
        node.parent = parent
    return node


def new_separate_rgba(
    tree: bpy.types.NodeTree,
    name: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None = None,
) -> bpy.types.Node:
    node = tree.nodes.new("CompositorNodeSepRGBA")
    node.name = name
    node.label = name
    node.location = location
    if parent is not None:
        node.parent = parent
    return node


def new_set_alpha(
    tree: bpy.types.NodeTree,
    image_socket,
    alpha_socket,
    name: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None = None,
) -> bpy.types.Node:
    node = tree.nodes.new("CompositorNodeSetAlpha")
    node.mode = "REPLACE_ALPHA"
    node.name = name
    node.label = name
    node.location = location
    if parent is not None:
        node.parent = parent
    tree.links.new(image_socket, node.inputs["Image"])
    tree.links.new(alpha_socket, node.inputs["Alpha"])
    return node


def new_frame(
    tree: bpy.types.NodeTree,
    name: str,
    label: str,
    location: tuple[float, float],
    color: tuple[float, float, float],
) -> bpy.types.Node:
    frame = tree.nodes.new("NodeFrame")
    frame.name = name
    frame.label = label
    frame.location = location
    frame.use_custom_color = True
    frame.color = color
    frame.label_size = 18
    frame.shrink = False
    return frame


def delete_old_file_output(tree: bpy.types.NodeTree) -> None:
    old = tree.nodes.get(OLD_FILE_OUTPUT_NAME)
    if old is None:
        log(f"no existing {OLD_FILE_OUTPUT_NAME} node to delete")
        return
    log(f"deleting old {OLD_FILE_OUTPUT_NAME} with {len(old.file_slots)} slots")
    tree.nodes.remove(old)


def build_workflow_branch(
    tree: bpy.types.NodeTree,
    frame: bpy.types.Node,
    group_node: bpy.types.Node,
    workflow_name: str,
    group_output_name: str,
    stem: str,
    row_y: float,
) -> dict[str, bpy.types.NodeSocket]:
    """Build the Separate/remap/dot/SetAlpha chain for one workflow.

    Returns a dict mapping suffix ('x', 'y', 'z', 'shading') to the output
    socket that should be wired into a File Output slot.
    """

    if group_output_name not in group_node.outputs:
        raise KeyError(
            f"group output {group_output_name!r} not found on {group_node.name!r}"
        )
    source = group_node.outputs[group_output_name]

    x0 = -400.0

    sep = new_separate_rgba(
        tree,
        f"Normals::Remap::{workflow_name}::Separate",
        (x0, row_y),
        parent=frame,
    )
    tree.links.new(source, sep.inputs[0])

    # -- axis slab branches: (c + 1) / 2 -------------------------------------
    slabs: dict[str, bpy.types.NodeSocket] = {}
    for idx, axis in enumerate(("x", "y", "z")):
        channel_socket = sep.outputs[idx]  # 0=R, 1=G, 2=B

        mul = new_math(
            tree,
            "MULTIPLY",
            f"Normals::Remap::{workflow_name}::{axis}_mul",
            f"{axis} * 0.5",
            (x0 + 220.0, row_y + 120.0 - idx * 60.0),
            value_b=0.5,
            parent=frame,
        )
        tree.links.new(channel_socket, mul.inputs[0])

        add = new_math(
            tree,
            "ADD",
            f"Normals::Remap::{workflow_name}::{axis}_add",
            f"+ 0.5",
            (x0 + 420.0, row_y + 120.0 - idx * 60.0),
            value_b=0.5,
            parent=frame,
        )
        tree.links.new(mul.outputs[0], add.inputs[0])

        set_alpha = new_set_alpha(
            tree,
            add.outputs[0],
            sep.outputs[3],  # alpha from the mask
            f"Normals::Remap::{workflow_name}::{axis}_setalpha",
            (x0 + 640.0, row_y + 120.0 - idx * 60.0),
            parent=frame,
        )
        slabs[axis] = set_alpha.outputs[0]

    # -- shading branch: max(0, N . L) ---------------------------------------
    lx = new_math(
        tree,
        "MULTIPLY",
        f"Normals::Remap::{workflow_name}::shading_lx",
        f"R * Lx",
        (x0 + 220.0, row_y - 120.0),
        value_b=SUN[0],
        parent=frame,
    )
    tree.links.new(sep.outputs[0], lx.inputs[0])

    ly = new_math(
        tree,
        "MULTIPLY",
        f"Normals::Remap::{workflow_name}::shading_ly",
        f"G * Ly",
        (x0 + 220.0, row_y - 180.0),
        value_b=SUN[1],
        parent=frame,
    )
    tree.links.new(sep.outputs[1], ly.inputs[0])

    lz = new_math(
        tree,
        "MULTIPLY",
        f"Normals::Remap::{workflow_name}::shading_lz",
        f"B * Lz",
        (x0 + 220.0, row_y - 240.0),
        value_b=SUN[2],
        parent=frame,
    )
    tree.links.new(sep.outputs[2], lz.inputs[0])

    sum_xy = new_math(
        tree,
        "ADD",
        f"Normals::Remap::{workflow_name}::shading_sum_xy",
        "Lx + Ly",
        (x0 + 420.0, row_y - 150.0),
        parent=frame,
    )
    tree.links.new(lx.outputs[0], sum_xy.inputs[0])
    tree.links.new(ly.outputs[0], sum_xy.inputs[1])

    sum_xyz = new_math(
        tree,
        "ADD",
        f"Normals::Remap::{workflow_name}::shading_sum_xyz",
        "+ Lz",
        (x0 + 620.0, row_y - 180.0),
        parent=frame,
    )
    tree.links.new(sum_xy.outputs[0], sum_xyz.inputs[0])
    tree.links.new(lz.outputs[0], sum_xyz.inputs[1])

    max0 = new_math(
        tree,
        "MAXIMUM",
        f"Normals::Remap::{workflow_name}::shading_max0",
        "max(0, dot)",
        (x0 + 820.0, row_y - 180.0),
        value_b=0.0,
        parent=frame,
    )
    tree.links.new(sum_xyz.outputs[0], max0.inputs[0])

    shading_set_alpha = new_set_alpha(
        tree,
        max0.outputs[0],
        sep.outputs[3],
        f"Normals::Remap::{workflow_name}::shading_setalpha",
        (x0 + 1020.0, row_y - 180.0),
        parent=frame,
    )
    slabs["shading"] = shading_set_alpha.outputs[0]

    return slabs


def build_file_output(
    tree: bpy.types.NodeTree,
    frame: bpy.types.Node,
    branches: list[tuple[str, str, dict[str, bpy.types.NodeSocket]]],
) -> bpy.types.Node:
    """Create the new File Output node with 12 slots."""

    fout = tree.nodes.new("CompositorNodeOutputFile")
    fout.name = NEW_FILE_OUTPUT_NAME
    fout.label = NEW_FILE_OUTPUT_NAME
    fout.location = (1000.0, 0.0)
    fout.parent = frame
    fout.base_path = "//normals_remap_test"
    fout.format.file_format = "PNG"
    fout.format.color_mode = "RGBA"
    fout.format.color_depth = "8"

    # Build full slot list first, then add them. Repurpose the default slot
    # created with the node for the first entry to avoid Blender-version
    # differences in how file_slots.remove() is called.
    all_slots: list[tuple[str, bpy.types.NodeSocket]] = []
    for workflow_name, stem, slabs in branches:
        for suffix in ("x", "y", "z", "shading"):
            if suffix == "shading":
                slot_path = (
                    stem.replace("_tree_normal", "_tree_shading")
                    .replace("_normal_full", "_shading_full")
                    + "_"
                )
            else:
                slot_path = f"{stem}_{suffix}_"
            all_slots.append((slot_path, slabs[suffix]))

    # Default File Output nodes ship with a single "Image" slot. Repurpose it
    # as the first entry, then append the remaining 11.
    if len(fout.file_slots) == 0:
        fout.file_slots.new(all_slots[0][0])
    else:
        fout.file_slots[0].path = all_slots[0][0]
    tree.links.new(all_slots[0][1], fout.inputs[0])

    for slot_path, source_socket in all_slots[1:]:
        fout.file_slots.new(slot_path)
        tree.links.new(source_socket, fout.inputs[-1])

    return fout


def migrate(src: Path, dst: Path) -> None:
    log(f"opening {src}")
    bpy.ops.wm.open_mainfile(filepath=str(src))

    scene = bpy.data.scenes.get("Current")
    if scene is None:
        raise RuntimeError("scene 'Current' not found in source blend")

    tree = scene.node_tree
    if tree is None:
        raise RuntimeError("scene 'Current' has no compositor node tree")

    group_node = tree.nodes.get(GROUP_NODE_NAME)
    if group_node is None:
        raise RuntimeError(f"compositor node {GROUP_NODE_NAME!r} not found")

    log("deleting old File Output node")
    delete_old_file_output(tree)

    log("creating remap frame")
    frame = new_frame(
        tree,
        NEW_FRAME_NAME,
        NEW_FRAME_LABEL,
        (0.0, -600.0),
        (0.14, 0.20, 0.16),
    )

    log("building per-workflow remap branches")
    branches: list[tuple[str, str, dict[str, bpy.types.NodeSocket]]] = []
    for idx, (workflow_name, group_output_name, stem) in enumerate(WORKFLOWS):
        row_y = 400.0 - idx * 500.0
        log(f"  workflow {workflow_name} -> {stem}_(x|y|z|shading)")
        slabs = build_workflow_branch(
            tree,
            frame,
            group_node,
            workflow_name,
            group_output_name,
            stem,
            row_y,
        )
        branches.append((workflow_name, stem, slabs))

    log("building new File Output node with 12 slots")
    fout = build_file_output(tree, frame, branches)
    log(f"  new File Output has {len(fout.file_slots)} slots")
    for i, slot in enumerate(fout.file_slots):
        sock = fout.inputs[i]
        linked = sock.is_linked
        log(f"    slot[{i}] path={slot.path!r} linked={linked}")

    log(f"saving migrated blend to {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    log("done")


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"source blend not found: {src}")
    if args.dst is None:
        dst = src.with_name(src.stem + "_remapped.blend")
    else:
        dst = Path(args.dst)

    if dst.resolve() == src.resolve():
        raise RuntimeError("refusing to overwrite source in place; provide --dst")

    migrate(src, dst)


if __name__ == "__main__":
    main()
