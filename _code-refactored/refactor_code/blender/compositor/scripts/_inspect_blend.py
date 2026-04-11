"""Dump full state of a blend compositor tree."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["INSPECT_BLEND"])
SCENE = os.environ["INSPECT_SCENE"]
bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes[SCENE]
tree = scene.node_tree

print(f"[inspect] {len(tree.nodes)} nodes, {len(tree.links)} links")
print("[inspect] NODES:")
for n in tree.nodes:
    extra = ""
    if n.bl_idname == "CompositorNodeMath":
        extra = f" op={n.operation} v0={n.inputs[0].default_value:.3f} v1={n.inputs[1].default_value:.3f}"
    if n.bl_idname == "CompositorNodeSetAlpha":
        extra = f" mode={n.mode}"
    if n.bl_idname == "CompositorNodeRGB":
        v = n.outputs[0].default_value
        extra = f" color=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f},{v[3]:.3f})"
    print(f"  {n.bl_idname:28s} name={n.name!r}{extra}")
print("[inspect] LINKS:")
for l in tree.links:
    print(f"  {l.from_node.name!r}.{l.from_socket.name!r} -> {l.to_node.name!r}.{l.to_socket.name!r}")
