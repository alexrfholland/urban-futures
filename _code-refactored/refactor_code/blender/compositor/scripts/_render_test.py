"""Minimal render test — load blend, force-load fresh EXR, render."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["TEST_BLEND"])
EXR = Path(os.environ["TEST_EXR"])
OUT = Path(os.environ["TEST_OUT"])

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes["ProposalColoredDepthOutlines"]
bpy.context.window.scene = scene

# Replace EXR image with a fresh load (reload() alone leaves size=(0,0))
exr_node = scene.node_tree.nodes["EXR"]
old_img = exr_node.image
new_img = bpy.data.images.load(str(EXR), check_existing=False)
exr_node.image = new_img
print(f"[test] fresh load: size={tuple(new_img.size)} source={new_img.source}")

# In background Blender, image.size stays (0,0) until render accesses the
# image; hard-code 8K resolution (matches EXR). Render still reads from file.
w, h = 7680, 4320
if new_img.size[0] > 0:
    w, h = new_img.size[:2]
print(f"[test] using resolution {w}x{h}")
scene.render.resolution_x = w
scene.render.resolution_y = h
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "8"
scene.render.film_transparent = True
scene.render.use_compositing = True
scene.render.use_sequencer = False
scene.frame_start = 1
scene.frame_end = 1
scene.frame_current = 1

OUT.mkdir(parents=True, exist_ok=True)
fout = scene.node_tree.nodes["ProposalColoredDepthOutput"]
fout.base_path = str(OUT)
print(f"[test] fout base_path={fout.base_path!r}")

scene.render.filepath = str(OUT / "_discard_.png")
print(f"[test] rendering at {w}x{h} (animation=True to fire File Output)...")
bpy.ops.render.render(animation=True)
print("[test] render complete")

print(f"[test] files in {OUT}:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name} ({p.stat().st_size} bytes)")
