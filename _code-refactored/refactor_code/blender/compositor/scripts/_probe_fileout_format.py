"""Probe a CompositorNodeOutputFile's format & per-slot overrides."""
import os
from pathlib import Path
import bpy

BLEND = Path(os.environ["PROBE_BLEND"])
SCENE = os.environ["PROBE_SCENE"]
NODE = os.environ["PROBE_NODE"]

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
scene = bpy.data.scenes[SCENE]
node = scene.node_tree.nodes[NODE]

print(f"[fout] node.base_path={node.base_path!r}")
print(f"[fout] active_input_index={node.active_input_index}")
print(f"[fout] format.file_format={node.format.file_format}")
print(f"[fout] format.color_mode={node.format.color_mode}")
print(f"[fout] format.color_depth={node.format.color_depth}")
print(f"[fout] format.quality={node.format.quality}")
print()
print(f"[fout] {len(node.file_slots)} slots:")
for i, s in enumerate(node.file_slots):
    sock = node.inputs[i]
    print(f"  [{i}] path={s.path!r} use_node_format={s.use_node_format}")
    if not s.use_node_format:
        print(f"      format={s.format.file_format} color={s.format.color_mode} depth={s.format.color_depth}")

# scene output settings
print()
print(f"[scene] use_compositing={scene.render.use_compositing}")
print(f"[scene] use_nodes={scene.use_nodes}")
print(f"[scene] frame_start={scene.frame_start} frame_end={scene.frame_end} frame_current={scene.frame_current}")
