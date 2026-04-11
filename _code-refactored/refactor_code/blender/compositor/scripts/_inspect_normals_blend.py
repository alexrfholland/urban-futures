"""Detailed inspection of the standalone compositor_normals.blend."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["NORMALS_BLEND"])
bpy.ops.wm.open_mainfile(filepath=str(BLEND))

scene = bpy.data.scenes["Current"]
tree = scene.node_tree

print(f"\n=== {BLEND.name} ===\n")
print(f"total nodes: {len(tree.nodes)}")
print(f"total links: {len(tree.links)}\n")

# All Image nodes (EXR inputs)
print("Image nodes (EXR/file inputs):")
for n in tree.nodes:
    if n.bl_idname == "CompositorNodeImage":
        img = n.image
        print(f"  {n.name!r} label={n.label!r}")
        if img:
            print(f"    image.name={img.name!r} filepath={img.filepath!r}")
            print(f"    layers={[l.name for l in img.layers] if hasattr(img,'layers') else 'n/a'}")
        else:
            print(f"    <no image>")

# Group nodes
print("\nGroup nodes:")
for n in tree.nodes:
    if n.bl_idname == "CompositorNodeGroup":
        print(f"  {n.name!r} label={n.label!r}")
        if n.node_tree is not None:
            print(f"    group_tree={n.node_tree.name!r}")
            print(f"    inputs:")
            for i, sock in enumerate(n.inputs):
                src = "<unlinked>"
                if sock.is_linked:
                    l = sock.links[0]
                    src = f"{l.from_node.name!r}.{l.from_socket.name!r}"
                print(f"      [{i}] {sock.name!r} type={sock.type} <- {src}")
            print(f"    outputs:")
            for i, sock in enumerate(n.outputs):
                consumers = [f"{l.to_node.name!r}.{l.to_socket.name!r}" for l in sock.links]
                print(f"      [{i}] {sock.name!r} type={sock.type} -> {consumers if consumers else '<no consumers>'}")

# File Output nodes
print("\nFile Output nodes:")
for n in tree.nodes:
    if n.bl_idname == "CompositorNodeOutputFile":
        print(f"  {n.name!r} base_path={n.base_path!r}")
        print(f"    format: {n.format.file_format} {n.format.color_mode} {n.format.color_depth}")
        for i, s in enumerate(n.file_slots):
            sock = n.inputs[i]
            src = "<unlinked>"
            if sock.is_linked:
                l = sock.links[0]
                src = f"{l.from_node.name!r}.{l.from_socket.name!r}"
            print(f"    slot[{i}] path={s.path!r} <- {src}")

# Composite nodes
print("\nComposite sinks:")
for n in tree.nodes:
    if n.bl_idname == "CompositorNodeComposite":
        sock = n.inputs[0]
        src = "<unlinked>"
        if sock.is_linked:
            l = sock.links[0]
            src = f"{l.from_node.name!r}.{l.from_socket.name!r}"
        print(f"  {n.name!r} <- {src}")

print()
