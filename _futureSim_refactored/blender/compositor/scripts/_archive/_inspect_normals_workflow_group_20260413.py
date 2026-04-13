"""Dump every node + link in the 'normals_workflow_group' node group.

Goal: find anything that transforms normals into a view-space-looking result,
or confirm the group passes them through as world-space.
"""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["NORMALS_BLEND"])
bpy.ops.wm.open_mainfile(filepath=str(BLEND))

group = bpy.data.node_groups.get("normals_workflow_group")
if group is None:
    print("no node group 'normals_workflow_group'")
    raise SystemExit(1)

print(f"=== normals_workflow_group ===")
print(f"nodes: {len(group.nodes)}  links: {len(group.links)}")
print()

# Dump every node with key attributes.
print("-- nodes --")
for n in group.nodes:
    attrs = [f"bl_idname={n.bl_idname}"]
    for attr in ("operation", "blend_type", "use_clamp", "label"):
        if hasattr(n, attr):
            v = getattr(n, attr)
            if v not in (None, "", False):
                attrs.append(f"{attr}={v!r}")
    if n.bl_idname == "CompositorNodeImage" and n.image is not None:
        attrs.append(f"image={n.image.name!r}")
    if n.bl_idname == "CompositorNodeRLayers":
        attrs.append(f"scene={n.scene.name if n.scene else None}")
        attrs.append(f"layer={n.layer}")
    print(f"  {n.name!r}  {' | '.join(attrs)}")
    for i, sock in enumerate(n.inputs):
        src = "<unlinked>"
        if sock.is_linked:
            l = sock.links[0]
            src = f"{l.from_node.name!r}.{l.from_socket.name!r}"
        # Print default for inputs if scalar/vector/color
        dv = getattr(sock, "default_value", None)
        try:
            if hasattr(dv, "__len__"):
                dv_str = f"({', '.join(f'{v:+.3f}' for v in dv)})"
            else:
                dv_str = f"{float(dv):+.3f}"
        except Exception:
            dv_str = ""
        print(f"      in [{i}] {sock.name!r} {sock.type} <- {src}  default={dv_str}")
    for i, sock in enumerate(n.outputs):
        consumers = [f"{l.to_node.name!r}.{l.to_socket.name!r}" for l in sock.links]
        print(f"      out[{i}] {sock.name!r} {sock.type} -> {consumers if consumers else '<no consumers>'}")
