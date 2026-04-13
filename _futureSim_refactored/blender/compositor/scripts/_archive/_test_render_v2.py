"""Test render v2 blend with NO graph edits — just open and render.

If the four fixes from _fix_v2_for_render.py actually unblock the pipeline,
this should write a file into
    E:\\2026 Arboreal Futures\\blenderv2\\renders\\alextest\\renders\\

No runtime File Output rebuild. No Composite-sink injection. No EXR repath.
Pure test of whether the saved blend can render as-is.
"""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["TARGET_BLEND"])
OUT = Path(r"E:\2026 Arboreal Futures\blenderv2\renders\alextest\renders")

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes["ProposalColoredDepthOutlines"]
if bpy.context.window is not None:
    bpy.context.window.scene = scene

print(f"[test] camera: {scene.camera.name if scene.camera else '<NONE>'}")
print(f"[test] render.filepath: {scene.render.filepath!r}")

# Clear the output dir so we can tell what this render wrote
for p in OUT.iterdir() if OUT.exists() else []:
    if p.is_file():
        p.unlink()

print("[test] render(animation=True)...")
bpy.ops.render.render(animation=True)
print("[test] done")

print("[test] contents after render:")
if OUT.exists():
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name} ({p.stat().st_size} bytes)")
else:
    print("  <output dir does not exist>")
