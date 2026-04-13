"""Test by swapping in a brand-new File Output node alongside the existing one."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["TEST_BLEND"])
EXR = Path(os.environ["TEST_EXR"])
OUT = Path(os.environ["TEST_OUT"])

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes["ProposalColoredDepthOutlines"]
bpy.context.window.scene = scene
tree = scene.node_tree

exr_node = tree.nodes["EXR"]
exr_node.image = bpy.data.images.load(str(EXR), check_existing=False)

scene.render.resolution_x = 7680
scene.render.resolution_y = 4320
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True
scene.render.use_compositing = True
scene.frame_start = 1
scene.frame_end = 1
scene.frame_current = 1

OUT.mkdir(parents=True, exist_ok=True)

# Find the masked output sockets by walking from the File Output's current links
old_fout = tree.nodes["ProposalColoredDepthOutput"]
print(f"[fresh] old File Output has {len(old_fout.file_slots)} slots")
source_sockets = []
for i, s in enumerate(old_fout.file_slots):
    sock = old_fout.inputs[i]
    if sock.is_linked:
        l = sock.links[0]
        source_sockets.append((s.path, l.from_socket))
        print(f"  [{i}] path={s.path!r} <- {l.from_node.name!r}.{l.from_socket.name!r}")

# Build a fresh File Output node
fresh = tree.nodes.new("CompositorNodeOutputFile")
fresh.name = "FreshOutput"
fresh.base_path = str(OUT)
fresh.format.file_format = "PNG"
fresh.format.color_mode = "RGBA"
fresh.format.color_depth = "8"
fresh.location = (2000, 0)

# remove default slot, add ours
while len(fresh.file_slots) > 1:
    fresh.file_slots.remove(fresh.file_slots[-1])
fresh.file_slots[0].path = source_sockets[0][0]
for path, _src in source_sockets[1:]:
    fresh.file_slots.new(path)

for i, (path, src) in enumerate(source_sockets):
    fresh.file_slots[i].path = path
    tree.links.new(src, fresh.inputs[i])
    print(f"[fresh] linked slot[{i}] {path!r}")

# Disconnect old fout so only fresh writes
for i in range(len(old_fout.inputs)):
    sock = old_fout.inputs[i]
    for l in list(sock.links):
        tree.links.remove(l)

# Force Composite to point at the first source too
composite = next((n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"), None)
if composite is not None:
    for l in list(composite.inputs["Image"].links):
        tree.links.remove(l)
    tree.links.new(source_sockets[0][1], composite.inputs["Image"])

scene.render.filepath = str(OUT / "_discard_.png")
print("[fresh] rendering...")
bpy.ops.render.render(animation=True)
print("[fresh] render complete")

print(f"[fresh] files in {OUT}:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name} ({p.stat().st_size} bytes)")
