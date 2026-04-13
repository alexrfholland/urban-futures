"""Read the current default values of the three gain Value nodes."""
import bpy
from pathlib import Path

BLEND = Path(r"D:/2026 Arboreal Futures/urban-futures/_futureSim_refactored/blender/compositor/canonical_templates/compositor_mist.blend")
bpy.ops.wm.open_mainfile(filepath=str(BLEND))
tree = bpy.data.scenes["Current"].node_tree
vl = tree.nodes.get("MistOutlines::viewlayer_group").node_tree
for name in ("thin_gain", "fine_gain", "extra_thin_gain"):
    n = vl.nodes.get(name)
    print(f"{name} = {n.outputs[0].default_value!r}")

# also dump the ColorRamp stops in thin_alpha so we know what thresholds we're working against
ta = vl.nodes.get("thin_alpha").node_tree
for cr_name in ("Color Ramp", "Color Ramp.001"):
    cr = ta.nodes.get(cr_name)
    print(f"\nthin_alpha::{cr_name} interp={cr.color_ramp.interpolation}")
    for i, el in enumerate(cr.color_ramp.elements):
        print(f"  stop[{i}] pos={el.position:.4f} color={list(el.color)}")
