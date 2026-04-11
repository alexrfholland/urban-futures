"""Summarize current EXR path, scenes, and File Output nodes of a blend."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["SUMMARY_BLEND"])
bpy.ops.wm.open_mainfile(filepath=str(BLEND))

print(f"\n=== {BLEND.name} ===")
print(f"scenes: {[s.name for s in bpy.data.scenes]}")

for scene in bpy.data.scenes:
    print(f"\n[scene] {scene.name!r}")
    print(f"  camera: {scene.camera.name if scene.camera else '<NONE>'}")
    print(f"  use_compositing: {scene.render.use_compositing}")
    print(f"  render.filepath: {scene.render.filepath!r}")
    if not scene.use_nodes or scene.node_tree is None:
        continue
    tree = scene.node_tree

    exr_node = tree.nodes.get("EXR")
    if exr_node is not None and exr_node.image is not None:
        print(f"  EXR node image.filepath: {exr_node.image.filepath!r}")
    else:
        print("  EXR node: <not found>")

    fouts = [n for n in tree.nodes if n.bl_idname == "CompositorNodeOutputFile"]
    comps = [n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"]
    print(f"  Composite sinks: {len(comps)}")
    print(f"  File Output nodes: {len(fouts)}")
    for fn in fouts:
        print(f"    {fn.name!r} base_path={fn.base_path!r}")
        for i, s in enumerate(fn.file_slots):
            sock = fn.inputs[i]
            src = "<unlinked>"
            if sock.is_linked:
                l = sock.links[0]
                src = f"{l.from_node.name!r}.{l.from_socket.name!r}"
            print(f"      slot[{i}] path={s.path!r} <- {src}")
print()
