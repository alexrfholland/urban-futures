"""Minimal render of proposal_colored_depth_outlines.blend — no graph edits.

Only changes made at runtime:
  - repath EXR (filepath + reload, then fresh load if size is 0)
  - set resolution to 8K
  - set File Output base_path
  - render animation (write_still won't fire File Output slots)
"""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["BLEND"])
EXR = Path(os.environ["EXR"])
OUT = Path(os.environ["OUT"])

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes["ProposalColoredDepthOutlines"]
bpy.context.window.scene = scene

exr_node = scene.node_tree.nodes["EXR"]
exr_node.image.filepath = str(EXR)
exr_node.image.reload()
if exr_node.image.size[0] == 0:
    # Background Blender: reload sometimes leaves size (0,0). Fresh load.
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

scene.render.filepath = str(OUT / "_discard_")
print("[min] animation render...")
bpy.ops.render.render(animation=True)
print("[min] done")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name} ({p.stat().st_size} bytes)")
