"""Inspect frame parenting to understand proposal chain groupings."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["INSPECT_BLEND"])
SCENE = os.environ["INSPECT_SCENE"]

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes[SCENE]
tree = scene.node_tree

# Group nodes by parent frame
frames = {}
orphans = []
for n in tree.nodes:
    if n.bl_idname == "NodeFrame":
        frames.setdefault(n.name, {"label": n.label, "children": []})
    else:
        parent = n.parent
        if parent is not None:
            frames.setdefault(parent.name, {"label": parent.label, "children": []})
            frames[parent.name]["children"].append(n)
        else:
            orphans.append(n)

# First ensure all frames even empty ones are present
for n in tree.nodes:
    if n.bl_idname == "NodeFrame":
        frames.setdefault(n.name, {"label": n.label, "children": []})

print(f"[frames] {len(frames)} frames")
for fname, info in frames.items():
    print(f"\n[frame] {fname!r} label={info['label']!r}")
    for c in info["children"]:
        extra = ""
        if c.bl_idname == "CompositorNodeRGB":
            v = c.outputs[0].default_value
            extra = f" color=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})"
        if c.bl_idname == "CompositorNodeMath":
            extra = f" op={c.operation} thresh={c.inputs[1].default_value:.3f}"
        if c.bl_idname == "CompositorNodeSetAlpha":
            extra = f" mode={c.mode}"
        print(f"  {c.bl_idname:28s} name={c.name!r}{extra}")

print(f"\n[frames] {len(orphans)} orphan (unparented) nodes:")
for o in orphans:
    print(f"  {o.bl_idname:28s} name={o.name!r}")

# Also check output slot order (proposal order in file_slots)
fout = tree.nodes["ProposalColoredDepthOutput"]
print(f"\n[slots] ProposalColoredDepthOutput file_slots order:")
for i, s in enumerate(fout.file_slots):
    sock = fout.inputs[i]
    src = ""
    if sock.is_linked:
        l = sock.links[0]
        src = f"{l.from_node.name!r}.{l.from_socket.name!r}"
    print(f"  [{i}] path={s.path!r} socket_name={sock.name!r} <- {src}")

# EXR output sockets (to see the 'canonical' order)
exr = tree.nodes["EXR"]
print(f"\n[exr] EXR output sockets (proposals only):")
for s in exr.outputs:
    if s.name.startswith("proposal-"):
        print(f"  {s.name!r}")
