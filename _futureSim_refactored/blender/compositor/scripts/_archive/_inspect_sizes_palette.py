"""Ad-hoc: dump all color-defining nodes inside the canonical compositor_sizes.blend
node groups: CompositorNodeRGB, CompositorNodeValToRGB (color ramps), and any
group-socket default_values that look like RGBA.

Usage:
    COMPOSITOR_BLEND_PATH=<...compositor_sizes.blend> \
    blender -b --factory-startup -P _inspect_sizes_palette.py
"""
from __future__ import annotations

import os

import bpy


def log(message: str) -> None:
    print(f"[inspect_sizes_palette] {message}")


def dump_colorramp(n) -> None:
    cr = n.color_ramp
    log(f"    ColorRamp name={n.name!r} label={n.label!r} mode={cr.color_mode} interp={cr.interpolation}")
    for i, e in enumerate(cr.elements):
        c = e.color
        log(f"      stop[{i}] pos={e.position:.6f} rgba=({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}, {c[3]:.6f})")


def dump_group_defaults(ng) -> None:
    # Dump default_value of every node's inputs/outputs for Groups or Mix nodes
    for n in ng.nodes:
        idname = n.bl_idname
        if idname == "CompositorNodeRGB":
            v = n.outputs[0].default_value
            log(f"    RGB name={n.name!r} label={n.label!r} rgba=({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, {v[3]:.6f})")
        elif idname == "CompositorNodeValToRGB":
            dump_colorramp(n)
        elif idname in ("CompositorNodeMixRGB", "CompositorNodeMix"):
            # Mix nodes hold color default inputs
            for s in n.inputs:
                try:
                    v = s.default_value
                except Exception:
                    continue
                if hasattr(v, "__len__") and len(v) == 4:
                    log(f"    Mix name={n.name!r} input={s.name!r} rgba=({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, {v[3]:.6f})")
        elif idname == "CompositorNodeGroup":
            # Group input default values
            for s in n.inputs:
                try:
                    v = s.default_value
                except Exception:
                    continue
                if hasattr(v, "__len__") and len(v) == 4:
                    log(f"    GroupIn name={n.name!r} input={s.name!r} rgba=({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, {v[3]:.6f})")


def main() -> None:
    blend_path = os.environ.get("COMPOSITOR_BLEND_PATH")
    if not blend_path:
        raise RuntimeError("missing env: COMPOSITOR_BLEND_PATH")
    log(f"opening {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    log("==== bpy.data.node_groups (compositor) ====")
    for ng in bpy.data.node_groups:
        if ng.bl_idname != "CompositorNodeTree":
            continue
        log(f"node_group: {ng.name}")
        dump_group_defaults(ng)

    # Also dump scene compositor tree (for instances of groups with overridden defaults)
    for scene in bpy.data.scenes:
        if not (scene.use_nodes and scene.node_tree is not None):
            continue
        log(f"==== scene[{scene.name}] top-level ====")
        tree = scene.node_tree
        for n in tree.nodes:
            idname = n.bl_idname
            if idname != "CompositorNodeGroup":
                continue
            # Only care about Sizes:: groups
            if "Sizes::" not in n.name and "Sizes::" not in (n.label or ""):
                continue
            log(f"  group inst: name={n.name!r} label={n.label!r} -> {n.node_tree.name if n.node_tree else None}")
            for s in n.inputs:
                try:
                    v = s.default_value
                except Exception:
                    continue
                if hasattr(v, "__len__") and len(v) == 4:
                    log(f"    input {s.name!r} rgba=({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, {v[3]:.6f})")
                elif isinstance(v, (int, float)):
                    log(f"    input {s.name!r} scalar={v}")


if __name__ == "__main__":
    main()
