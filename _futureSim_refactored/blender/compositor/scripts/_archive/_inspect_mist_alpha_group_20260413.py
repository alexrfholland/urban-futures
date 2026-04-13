"""Dump the internal graph of the `thin_alpha` nested group — this is the
actual line-extraction step (quantized image + gain -> mask alpha).
"""
import bpy
from pathlib import Path

BLEND = Path(r"D:/2026 Arboreal Futures/urban-futures/_futureSim_refactored/blender/compositor/canonical_templates/compositor_mist.blend")

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
tree = bpy.data.scenes["Current"].node_tree
outer = tree.nodes.get("MistOutlines::viewlayer_group")
vl_tree = outer.node_tree

for probe_name in ("thin_alpha", "fine_alpha", "extra_thin_alpha"):
    n = vl_tree.nodes.get(probe_name)
    gt = n.node_tree
    print(f"\n=== {probe_name} -> node_tree {gt.name!r} ===")
    for nd in gt.nodes:
        props = []
        for attr in ("filter_type", "size_x", "size_y", "operation",
                     "use_clamp", "factor", "distance", "sigma_color",
                     "sigma_space", "x", "y", "falloff"):
            if hasattr(nd, attr):
                props.append(f"{attr}={getattr(nd, attr)!r}")
        defaults = []
        for inp in nd.inputs:
            if not inp.is_linked and hasattr(inp, "default_value"):
                try:
                    v = inp.default_value
                    if hasattr(v, "__len__") and not isinstance(v, str):
                        v = list(v)
                    defaults.append(f"{inp.name}={v}")
                except Exception:
                    pass
        print(f"  [{nd.bl_idname}] {nd.name!r}  {'  '.join(props)}")
        for d in defaults:
            print(f"      default: {d}")
    print("  Links:")
    for link in gt.links:
        print(f"    {link.from_node.name!r}.{link.from_socket.name!r} -> {link.to_node.name!r}.{link.to_socket.name!r}")
    # only need to dump the shared tree once
    break
