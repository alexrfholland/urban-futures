"""Open a blend, repath its EXR, reload, and list Image node output sockets."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["PROBE_BLEND"])
EXR = Path(os.environ["PROBE_EXR"])
SCENE = os.environ["PROBE_SCENE"]

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes[SCENE]
exr_node = scene.node_tree.nodes["EXR"]
exr_node.image.filepath = str(EXR)
exr_node.image.reload()

# Force update so outputs populate
bpy.context.view_layer.update()

print(f"[probe] image size: {exr_node.image.size[:]}")
print(f"[probe] image source: {exr_node.image.source}")
print(f"[probe] {len(exr_node.outputs)} output sockets:")
for s in exr_node.outputs:
    print(f"  {s.name!r}  type={s.type}  enabled={s.enabled}")
