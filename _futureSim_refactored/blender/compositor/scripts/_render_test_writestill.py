"""Test rebuilt blend with write_still=True vs animation=True."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["TEST_BLEND"])
EXR = Path(os.environ["TEST_EXR"])
OUT = Path(os.environ["TEST_OUT"])
MODE = os.environ.get("TEST_MODE", "still")

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes["ProposalColoredDepthOutlines"]
bpy.context.window.scene = scene

exr_node = scene.node_tree.nodes["EXR"]
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
fout = scene.node_tree.nodes["ProposalColoredDepthOutput"]
fout.base_path = str(OUT)

scene.render.filepath = str(OUT / "_discard_.png")
print(f"[test] mode={MODE}")
if MODE == "anim":
    bpy.ops.render.render(animation=True)
else:
    bpy.ops.render.render(write_still=True)
print("[test] done")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name} ({p.stat().st_size} bytes)")
