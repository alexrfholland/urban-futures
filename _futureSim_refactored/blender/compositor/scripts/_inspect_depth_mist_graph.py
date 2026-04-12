"""Ad-hoc: dump the compositor node graph inside the depth-outliner and mist
workflow groups of the canonical blend files. Lists every node, its type,
and its links so we can understand the edge-detection pipeline.

Usage:
    COMPOSITOR_BLEND_PATH=<...blend> \
    blender -b --factory-startup -P _inspect_depth_mist_graph.py
"""
from __future__ import annotations

import os

import bpy


def log(msg: str) -> None:
    print(f"[inspect_graph] {msg}")


def dump_tree(tree, label: str, depth: int = 0) -> None:
    indent = "  " * depth
    log(f"{indent}==== {label} ({len(tree.nodes)} nodes, {len(tree.links)} links) ====")
    for n in tree.nodes:
        extra = ""
        if n.bl_idname == "CompositorNodeImage":
            img_name = n.image.name if n.image else "None"
            extra = f" image={img_name}"
        elif n.bl_idname == "CompositorNodeOutputFile":
            slots = [s.path for s in n.file_slots]
            extra = f" slots={slots}"
        elif n.bl_idname == "CompositorNodeGroup":
            ng_name = n.node_tree.name if n.node_tree else "None"
            extra = f" -> {ng_name}"
        elif n.bl_idname in ("CompositorNodeMath", "CompositorNodeMixRGB"):
            extra = f" op={getattr(n, 'operation', '?')} blend={getattr(n, 'blend_type', '?')}"
        elif n.bl_idname == "CompositorNodeFilter":
            extra = f" filter_type={n.filter_type}"
        elif n.bl_idname == "CompositorNodeMapRange":
            extra = ""
            for s in n.inputs:
                try:
                    v = s.default_value
                    extra += f" {s.name}={v:.4f}"
                except:
                    pass
        log(f"{indent}  [{n.bl_idname}] name={n.name!r} label={n.label!r}{extra}")

    log(f"{indent}  -- links --")
    for link in tree.links:
        log(f"{indent}  {link.from_node.name}.{link.from_socket.name} -> {link.to_node.name}.{link.to_socket.name}")

    # Recurse into compositor node groups
    for n in tree.nodes:
        if n.bl_idname == "CompositorNodeGroup" and n.node_tree:
            dump_tree(n.node_tree, f"{label} / {n.node_tree.name}", depth + 1)


def main() -> None:
    blend_path = os.environ.get("COMPOSITOR_BLEND_PATH")
    if not blend_path:
        raise RuntimeError("missing env: COMPOSITOR_BLEND_PATH")
    log(f"opening {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    for scene in bpy.data.scenes:
        if not (scene.use_nodes and scene.node_tree):
            continue
        dump_tree(scene.node_tree, f"scene[{scene.name}]")


if __name__ == "__main__":
    main()
