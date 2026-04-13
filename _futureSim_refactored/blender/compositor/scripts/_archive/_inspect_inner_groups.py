"""Dump depth-outliner_viewlayer_group and mist-outliner_viewlayer_group."""
from __future__ import annotations
import os
import bpy


def log(msg: str) -> None:
    print(f"[inner_groups] {msg}")


def dump_ng(ng) -> None:
    log(f"==== {ng.name} ({len(ng.nodes)} nodes, {len(ng.links)} links) ====")
    for n in ng.nodes:
        extra = ""
        if n.bl_idname == "CompositorNodeFilter":
            extra = f" filter_type={n.filter_type}"
        elif n.bl_idname in ("CompositorNodeMath",):
            extra = f" op={n.operation}"
        elif n.bl_idname == "CompositorNodeMapRange":
            for s in n.inputs:
                try:
                    extra += f" {s.name}={s.default_value:.4f}"
                except:
                    pass
        elif n.bl_idname == "CompositorNodeGroup":
            extra = f" -> {n.node_tree.name if n.node_tree else None}"
        log(f"  [{n.bl_idname}] name={n.name!r} label={n.label!r}{extra}")
        # dump input defaults for key nodes
        if n.bl_idname in ("CompositorNodeDilateErode",):
            log(f"    distance={n.distance}")
        if n.bl_idname == "CompositorNodeBlur":
            log(f"    size_x={n.size_x} size_y={n.size_y}")

    log("  -- links --")
    for link in ng.links:
        log(f"  {link.from_node.name}.{link.from_socket.name} -> {link.to_node.name}.{link.to_socket.name}")

    # Recurse into sub-groups
    for n in ng.nodes:
        if n.bl_idname == "CompositorNodeGroup" and n.node_tree:
            dump_ng(n.node_tree)


def main() -> None:
    blend_path = os.environ.get("COMPOSITOR_BLEND_PATH")
    if not blend_path:
        raise RuntimeError("missing env: COMPOSITOR_BLEND_PATH")
    log(f"opening {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    targets = ["depth-outliner_viewlayer_group", "mist-outliner_viewlayer_group"]
    for ng in bpy.data.node_groups:
        if ng.name in targets:
            dump_ng(ng)


if __name__ == "__main__":
    main()
