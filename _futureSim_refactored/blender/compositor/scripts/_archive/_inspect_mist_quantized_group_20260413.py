"""Dump the internal graph of the nested `mist_quantized` group inside
MistOutlines::viewlayer_group — this is where the actual kirsch edge filter
+ threshold/quantize live.
"""
import bpy
from pathlib import Path

BLEND = Path(r"D:/2026 Arboreal Futures/urban-futures/_futureSim_refactored/blender/compositor/canonical_templates/compositor_mist.blend")

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
tree = bpy.data.scenes["Current"].node_tree
outer = tree.nodes.get("MistOutlines::viewlayer_group")
quant = outer.node_tree.nodes.get("mist_quantized")
print(f"mist_quantized node_tree: {quant.node_tree.name!r}")
gt = quant.node_tree

for n in gt.nodes:
    props = []
    for attr in ("filter_type", "size_x", "size_y", "operation",
                 "use_clamp", "factor", "distance", "sigma_color", "sigma_space"):
        if hasattr(n, attr):
            props.append(f"{attr}={getattr(n, attr)!r}")
    defaults = []
    for inp in n.inputs:
        if not inp.is_linked and hasattr(inp, "default_value"):
            try:
                v = inp.default_value
                if hasattr(v, "__len__") and not isinstance(v, str):
                    v = list(v)
                defaults.append(f"{inp.name}={v}")
            except Exception:
                pass
    print(f"  [{n.bl_idname}] {n.name!r}  {'  '.join(props)}")
    for d in defaults:
        print(f"      default: {d}")

print("\nLinks:")
for link in gt.links:
    print(f"  {link.from_node.name!r}.{link.from_socket.name!r} -> {link.to_node.name!r}.{link.to_socket.name!r}")
