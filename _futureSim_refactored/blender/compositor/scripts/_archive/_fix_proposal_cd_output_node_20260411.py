"""One-shot fixup for proposal_colored_depth_outlines_v4_subcategories.blend.

The original blend's 'ProposalColoredDepthOutput' node is silently ignored by
Blender's compositor evaluator (confirmed 2026-04-11: rebuilding the node from
scratch with identical slots and links makes it write all 7 PNGs). We suspect
some data-corruption from a Blender upgrade or accidental edit.

This script:
  1. Opens the canonical working-copy blend (passed via argv after --).
  2. Rebuilds ProposalColoredDepthOutput node with the same 7 slots + links.
  3. Saves the blend in place.

Usage:
  blender --background --factory-startup --python <this> -- <blend_path>
"""
from __future__ import annotations
import sys
from pathlib import Path
import bpy

NODE_NAME = "ProposalColoredDepthOutput"


def main() -> None:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("usage: blender --background --python ... -- <blend_path>")
    blend = Path(argv[argv.index("--") + 1])
    print(f"OPEN {blend}")
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = None
    for s in bpy.data.scenes:
        if s.node_tree and NODE_NAME in s.node_tree.nodes:
            scene = s
            break
    if scene is None:
        raise SystemExit(f"no scene contains {NODE_NAME}")
    nt = scene.node_tree
    old = nt.nodes[NODE_NAME]
    print(f"SCENE {scene.name!r}  capturing {len(old.file_slots)} slots...")
    # Capture slots and their source links
    captured: list[tuple[str, list[tuple[str, str]]]] = []
    for i, inp in enumerate(old.inputs):
        if i >= len(old.file_slots):
            break
        slot = old.file_slots[i]
        srcs = [(lk.from_node.name, lk.from_socket.name) for lk in inp.links]
        captured.append((slot.path, srcs))
        print(f"  [{i}] {slot.path!r} <- {srcs}")
    # Capture node position/props so the UI looks sane on next open
    loc = tuple(old.location)
    width = old.width
    label = old.label
    base_path_fallback = old.base_path  # not the valid dir but preserves the string if needed
    # Remove old, create new
    nt.nodes.remove(old)
    new = nt.nodes.new("CompositorNodeOutputFile")
    new.name = NODE_NAME
    new.label = label or NODE_NAME
    new.location = loc
    new.width = width
    new.format.file_format = "PNG"
    new.format.color_mode = "RGBA"
    new.format.color_depth = "8"
    new.base_path = base_path_fallback or ""
    # Slots (first slot is already present, just rename it)
    first_path, first_srcs = captured[0]
    new.file_slots[0].path = first_path
    for path, _ in captured[1:]:
        new.file_slots.new(path)
    # Link
    for i, (path, srcs) in enumerate(captured):
        for from_name, from_sock in srcs:
            from_node = nt.nodes.get(from_name)
            if from_node is None:
                print(f"  MISSING source node {from_name!r} for slot {path!r}")
                continue
            from_socket = from_node.outputs.get(from_sock)
            if from_socket is None:
                print(f"  MISSING source socket {from_sock!r} on {from_name!r}")
                continue
            nt.links.new(from_socket, new.inputs[i])
    print(f"REBUILT node with {len(new.file_slots)} slots")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    print(f"SAVED {blend}")


if __name__ == "__main__":
    main()
