"""Dump the internal graph of MistOutlines::viewlayer_group so we can see
which nodes/parameters drive the 'broken up' lines on library trees.
Read-only.
"""
import bpy
from pathlib import Path

BLEND = Path(r"D:/2026 Arboreal Futures/urban-futures/_futureSim_refactored/blender/compositor/canonical_templates/compositor_mist.blend")

bpy.ops.wm.open_mainfile(filepath=str(BLEND))
tree = bpy.data.scenes["Current"].node_tree
grp = tree.nodes.get("MistOutlines::viewlayer_group")
assert grp is not None
gt = grp.node_tree
print(f"group node_tree: {gt.name!r}")

# All nodes
for n in gt.nodes:
    props = []
    for attr in ("filter_type", "size_x", "size_y", "use_bokeh",
                 "use_relative", "operation", "use_clamp",
                 "falloff", "x", "y", "factor",
                 "sigma_color", "sigma_space"):
        if hasattr(n, attr):
            v = getattr(n, attr)
            props.append(f"{attr}={v!r}")
    # inputs with default values
    defaults = []
    for inp in n.inputs:
        if not inp.is_linked and hasattr(inp, "default_value"):
            try:
                val = inp.default_value
                if hasattr(val, "__len__") and not isinstance(val, str):
                    val = list(val)
                defaults.append(f"{inp.name}={val}")
            except Exception:
                pass
    print(f"  [{n.bl_idname}] {n.name!r}  {'  '.join(props)}")
    if defaults:
        for d in defaults:
            print(f"      default: {d}")

print("\nLinks:")
for link in gt.links:
    print(f"  {link.from_node.name!r}.{link.from_socket.name!r} -> {link.to_node.name!r}.{link.to_socket.name!r}")
