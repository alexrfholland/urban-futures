"""Inspect the DepthOutliner node group outputs and internal nodes."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["INSPECT_BLEND"])
SCENE = os.environ["INSPECT_SCENE"]

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes[SCENE]
tree = scene.node_tree

# find the group node
for n in tree.nodes:
    if n.bl_idname == "CompositorNodeGroup":
        print(f"[depth] group node: {n.name!r}")
        print(f"[depth] group name: {n.node_tree.name!r}")
        ng = n.node_tree
        # Blender 4.x uses interface.items_tree
        print(f"[depth] group interface ({len(ng.interface.items_tree)}):")
        for it in ng.interface.items_tree:
            io = getattr(it, "in_out", "?")
            sock = getattr(it, "socket_type", "?")
            print(f"  {io} {it.name!r} type={sock}")
        # Also the group node instance's inputs/outputs
        print(f"[depth] node outputs ({len(n.outputs)}):")
        for o in n.outputs:
            print(f"  out {o.name!r} type={o.type}")
        print(f"[depth] internal nodes ({len(ng.nodes)}):")
        for nn in ng.nodes:
            print(f"  {nn.bl_idname:30s} name={nn.name!r}")
        print(f"[depth] internal links ({len(ng.links)}):")
        for ll in ng.links:
            print(f"  {ll.from_node.name!r}.{ll.from_socket.name!r} -> {ll.to_node.name!r}.{ll.to_socket.name!r}")
        break
