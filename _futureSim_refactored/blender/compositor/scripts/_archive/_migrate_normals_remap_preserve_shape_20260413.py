"""Minimal canonical-template edit: insert `(N + 1) / 2` remap between the
Normals::Workflow Group and the File Output node, preserving the existing
3-output shape (existing / pathway / priority).

Why
---
The canonical `compositor_normals.blend` writes the workflow group's raw
world-space Normal RGBA straight into a PNG 8-bit File Output. Negative
channel values clip to 0 on write, which destroys roughly half of the world
normal data and yields an output that superficially resembles a view-space
normal map. Adding per-channel `(c + 1) / 2` before the File Output encodes
the full [-1, +1] range into the [0, 1] PNG range losslessly.

What this script does
---------------------
Inside scene `Current`:

  1. For each of the three group outputs (existing / pathway / priority),
     build: SeparateRGBA -> three MathChains(c+1 then *0.5) -> CombineRGBA,
     preserving the original alpha from the group output.
  2. Relink the File Output node's three slots to the three new combined
     RGBA sockets. Slot paths are preserved.
  3. The `Normals::Workflow Group` node group itself is NOT touched.
  4. Existing `Normals::Composite` / Viewer / extra sinks remain wired to
     the original pre-remap RGBA so in-Blender inspection stays truthful.

Contract
--------
This is an explicit template-edit operation, not a runtime script. Run
against a working copy; only copy the result onto the canonical once
verified.

Usage
-----
    blender --background --python _migrate_normals_remap_preserve_shape_20260413.py \\
        -- --src <src .blend> --dst <dst .blend>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


SCENE_NAME = "Current"
GROUP_NODE_NAME = "Normals::Workflow Group"
FILE_OUTPUT_NAME = "Normals::Outputs"

WORKFLOWS = [
    ("pathway",  "to_normals__composite",           "pathway_tree_normal_"),
    ("priority", "priority_tree_normal",            "priority_tree_normal_"),
    ("existing", "existing_condition_normal_full",  "existing_condition_normal_full_"),
]

REMAP_FRAME_NAME = "Normals::Remap Frame 20260413"
REMAP_FRAME_LABEL = "(N + 1) / 2  [PNG-safe remap]"


def log(msg: str) -> None:
    print(f"[migrate_normals_remap_preserve_shape] {msg}", flush=True)


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


def new_frame(tree, name, label, location):
    f = tree.nodes.new("NodeFrame")
    f.name = name
    f.label = label
    f.location = location
    f.use_custom_color = True
    f.color = (0.12, 0.22, 0.18)
    f.label_size = 18
    f.shrink = False
    return f


def remap_chain(tree, rgba_source, row_y, frame, workflow_name):
    """Build (N+1)/2 + alpha-preserving chain. Return the CombineRGBA output."""
    x0 = 400.0

    sep = tree.nodes.new("CompositorNodeSepRGBA")
    sep.name = f"Normals::Remap::{workflow_name}::Separate"
    sep.label = f"{workflow_name} separate"
    sep.location = (x0, row_y)
    sep.parent = frame
    tree.links.new(rgba_source, sep.inputs[0])

    combine = tree.nodes.new("CompositorNodeCombRGBA")
    combine.name = f"Normals::Remap::{workflow_name}::Combine"
    combine.label = f"{workflow_name} combine"
    combine.location = (x0 + 880.0, row_y)
    combine.parent = frame

    for idx, axis in enumerate(("R", "G", "B")):
        add = tree.nodes.new("CompositorNodeMath")
        add.operation = "ADD"
        add.name = f"Normals::Remap::{workflow_name}::{axis}_add"
        add.label = f"{axis} + 1"
        add.location = (x0 + 220.0, row_y + 60.0 - idx * 55.0)
        add.parent = frame
        add.inputs[1].default_value = 1.0
        tree.links.new(sep.outputs[idx], add.inputs[0])

        mul = tree.nodes.new("CompositorNodeMath")
        mul.operation = "MULTIPLY"
        mul.name = f"Normals::Remap::{workflow_name}::{axis}_mul"
        mul.label = f"* 0.5"
        mul.location = (x0 + 440.0, row_y + 60.0 - idx * 55.0)
        mul.parent = frame
        mul.inputs[1].default_value = 0.5
        tree.links.new(add.outputs[0], mul.inputs[0])

        tree.links.new(mul.outputs[0], combine.inputs[idx])

    # Preserve the original alpha from the RGBA source (mask).
    tree.links.new(sep.outputs[3], combine.inputs[3])
    return combine.outputs["Image"]


def migrate(src: Path, dst: Path) -> None:
    log(f"opening {src}")
    bpy.ops.wm.open_mainfile(filepath=str(src))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise RuntimeError(f"scene {SCENE_NAME!r} missing")
    tree = scene.node_tree
    if tree is None:
        raise RuntimeError(f"scene {SCENE_NAME!r} has no compositor tree")
    group_node = tree.nodes.get(GROUP_NODE_NAME)
    if group_node is None:
        raise RuntimeError(f"node {GROUP_NODE_NAME!r} missing")
    fout = tree.nodes.get(FILE_OUTPUT_NAME)
    if fout is None:
        raise RuntimeError(f"node {FILE_OUTPUT_NAME!r} missing")

    # Map slot paths to slot indices so we can rewire without assuming order.
    slot_index_by_path: dict[str, int] = {}
    for i, slot in enumerate(fout.file_slots):
        slot_index_by_path[slot.path] = i
    log(f"existing File Output slots: {slot_index_by_path}")

    # Place the remap frame off to the right of the group node.
    gx, gy = group_node.location
    frame = new_frame(tree, REMAP_FRAME_NAME, REMAP_FRAME_LABEL, (gx + 600.0, gy))

    for row_idx, (workflow_name, group_output_name, slot_path) in enumerate(WORKFLOWS):
        source_sock = group_node.outputs.get(group_output_name)
        if source_sock is None:
            raise RuntimeError(f"group output {group_output_name!r} missing on {GROUP_NODE_NAME!r}")
        row_y = gy + 400.0 - row_idx * 360.0
        log(f"  building remap for {workflow_name} (group out {group_output_name!r} -> slot {slot_path!r})")
        remapped_sock = remap_chain(tree, source_sock, row_y, frame, workflow_name)

        slot_i = slot_index_by_path.get(slot_path)
        if slot_i is None:
            raise RuntimeError(f"File Output slot {slot_path!r} not found")
        # Drop any existing link into this slot input, then wire the remapped
        # RGBA in.
        slot_input = fout.inputs[slot_i]
        for link in list(slot_input.links):
            tree.links.remove(link)
        tree.links.new(remapped_sock, slot_input)

    log(f"saving to {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    log("done")


def main() -> None:
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if src == dst:
        raise RuntimeError("refusing to overwrite src in place")
    migrate(src, dst)


if __name__ == "__main__":
    main()
