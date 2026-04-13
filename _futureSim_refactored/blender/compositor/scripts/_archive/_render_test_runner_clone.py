"""Clone of the runner pattern to debug why File Output isn't firing."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["TEST_BLEND"])
EXR = Path(os.environ["TEST_EXR"])
OUT = Path(os.environ["TEST_OUT"])

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes.get("ProposalColoredDepthOutlines")
print(f"[rc] scene name: {scene.name}")
print(f"[rc] context scene before: {bpy.context.scene.name}")

# Runner DOES NOT set bpy.context.window.scene — does that matter?
# Let's try WITHOUT setting context scene first.
exr_node = scene.node_tree.nodes["EXR"]
# Runner uses reload; we've seen that leaves size=(0,0). Force fresh load.
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
print(f"[rc] fout.base_path={fout.base_path!r}")

scene.render.filepath = str(OUT / "_discard_render.png")
bpy.context.window.scene = scene
print("[rc] rendering with animation=True (no scene param)...")
bpy.ops.render.render(animation=True)
print("[rc] done")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name} ({p.stat().st_size} bytes)")
