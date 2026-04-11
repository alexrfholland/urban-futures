"""Inspect file slots on a CompositorNodeOutputFile."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["INSPECT_BLEND"])
SCENE = os.environ["INSPECT_SCENE"]
NODE = os.environ["INSPECT_NODE"]

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes[SCENE]
node = scene.node_tree.nodes[NODE]
print(f"[slots] {NODE}: base_path={node.base_path!r}")
print(f"[slots] file_slots ({len(node.file_slots)}):")
for i, s in enumerate(node.file_slots):
    socket = node.inputs[i]
    linked = "LINKED" if socket.is_linked else "unlinked"
    print(f"  [{i}] path={s.path!r} socket_name={socket.name!r} {linked}")
