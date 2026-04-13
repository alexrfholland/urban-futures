"""Read-only: for compositor_mist.blend and compositor_depth_outliner.blend,
dump the links from the EXR Input node into the *::viewlayer_group, and
the socket names / link chain through the group so we can decide whether
`EXR Input.IndexOB` can be relinked to `EXR Input.resource_tree_mask` at
runtime without changing the canonical.
"""
import bpy
from pathlib import Path

ROOT = Path(r"D:/2026 Arboreal Futures/urban-futures/_futureSim_refactored/blender/compositor/canonical_templates")
SPECS = [
    ("compositor_mist.blend", "MistOutlines::EXR Input", "MistOutlines::viewlayer_group", "MistOutlines::Outputs"),
    ("compositor_depth_outliner.blend", "DepthOutliner::EXR Input", "DepthOutliner::viewlayer_group", "DepthOutliner::Outputs"),
]

for blend_name, exr_name, group_name, fo_name in SPECS:
    bpy.ops.wm.open_mainfile(filepath=str(ROOT / blend_name))
    print(f"\n=== {blend_name} ===")
    tree = bpy.data.scenes["Current"].node_tree
    exr = tree.nodes.get(exr_name)
    grp = tree.nodes.get(group_name)
    fo = tree.nodes.get(fo_name)
    if exr is None or grp is None or fo is None:
        print(f"  MISSING — exr={exr is not None} grp={grp is not None} fo={fo is not None}")
        continue

    print(f"  Links FROM {exr_name!r}:")
    for sock in exr.outputs:
        for link in sock.links:
            print(f"    {sock.name!r} -> {link.to_node.name!r}.{link.to_socket.name!r}")

    print(f"  Links INTO {group_name!r}:")
    for sock in grp.inputs:
        if sock.is_linked:
            src = sock.links[0].from_socket
            print(f"    input {sock.name!r}  <- {src.node.name!r}.{src.name!r}")
        else:
            print(f"    input {sock.name!r}  <unlinked>")

    print(f"  Outputs of {group_name!r}:")
    for sock in grp.outputs:
        for link in sock.links:
            print(f"    {sock.name!r} -> {link.to_node.name!r}.{link.to_socket.name!r}")

    print(f"  File Output {fo_name!r} slots:")
    for i, slot in enumerate(fo.file_slots):
        sock = fo.inputs[i]
        if sock.is_linked:
            lk = sock.links[0]
            print(f"    slot[{i}] path={slot.path!r} <- {lk.from_node.name!r}.{lk.from_socket.name!r}")
        else:
            print(f"    slot[{i}] path={slot.path!r} <unlinked>")

    print(f"  Inside group.node_tree {grp.node_tree.name!r}:")
    for n in grp.node_tree.nodes:
        if n.bl_idname == "NodeGroupInput":
            print(f"    GroupInput outputs (what the caller's inputs become inside):")
            for s in n.outputs:
                for link in s.links:
                    print(f"      {s.name!r} -> {link.to_node.name!r}({link.to_node.bl_idname}).{link.to_socket.name!r}")
