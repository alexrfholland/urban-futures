"""Remove the baked-normal override from v2WorldAOV.

Context
-------
v2WorldAOV wires three geometry attributes (orig_normal_x/y/z) through a
Combine XYZ node into the Toon BSDF's Normal input. Cycles writes the SHADING
NORMAL into the Normal render pass, so this override makes the pass reflect
the baked per-vertex attributes rather than the actual surface normal —
i.e. the Normal EXR pass ceases to be world-space and shows camera/view-
correlated artifacts across parallel surfaces.

Fix
---
Remove the 3 Attribute nodes ('orig_normal_x/y/z'), the Combine XYZ node,
and the link into Toon BSDF.Normal. Leaving that socket unlinked makes
Cycles use the real geometry shading normal, which is world-space.

The point-cloud attributes themselves are unaffected — they're still written
by the VTK export pipeline (export_rewilded_envelopes.py) and can be read
by any future node that wants them. We're just deleting the in-material
override that was misusing them as a shading-normal source.

Usage
-----
    blender --background --python _fix_v2worldaov_normal_override_20260413.py \\
        -- --blend <path.blend>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


MATERIAL_NAME = "v2WorldAOV"
ATTR_NAMES = {"orig_normal_x", "orig_normal_y", "orig_normal_z"}


def log(msg: str) -> None:
    print(f"[fix_v2worldaov_normal] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    return p.parse_args(argv)


def verify_no_other_consumers(mat: bpy.types.Material) -> None:
    """Abort if anything reads orig_normal_* other than via the 3 expected
    Attribute nodes OR the single Combine XYZ feeding the Toon BSDF Normal
    input (both are expected and will be removed)."""
    tree = mat.node_tree
    suspect = []
    for n in tree.nodes:
        if n.bl_idname == "ShaderNodeAttribute" and n.attribute_name in ATTR_NAMES:
            continue  # will be removed
        if n.bl_idname == "ShaderNodeCombineXYZ":
            combine_out = n.outputs[0]
            consumers = [l.to_node.bl_idname for l in combine_out.links]
            if consumers == ["ShaderNodeBsdfToon"]:
                continue  # expected; will be removed
        for sock in n.inputs:
            if not sock.is_linked:
                continue
            src = sock.links[0].from_node
            if src.bl_idname == "ShaderNodeAttribute" and src.attribute_name in ATTR_NAMES:
                suspect.append(f"{n.name!r}.{sock.name!r} <- Attribute {src.attribute_name!r}")
    if suspect:
        raise RuntimeError(
            f"orig_normal_* referenced by unexpected consumers in {mat.name!r}: {suspect}"
        )


def verify_other_materials_clean() -> None:
    for mat in bpy.data.materials:
        if mat.name == MATERIAL_NAME:
            continue
        if not mat.use_nodes or mat.node_tree is None:
            continue
        for n in mat.node_tree.nodes:
            if n.bl_idname == "ShaderNodeAttribute" and n.attribute_name in ATTR_NAMES:
                raise RuntimeError(
                    f"material {mat.name!r} also reads {n.attribute_name!r} — aborting"
                )


def apply_fix(mat: bpy.types.Material) -> None:
    tree = mat.node_tree
    # Find the Toon BSDF and its Combine XYZ source on the Normal input.
    toon = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeBsdfToon"), None)
    if toon is None:
        raise RuntimeError("Toon BSDF not found in v2WorldAOV")
    normal_in = toon.inputs.get("Normal")
    if normal_in is None:
        raise RuntimeError("Toon BSDF has no Normal input")
    nodes_to_remove: list[bpy.types.Node] = []
    if normal_in.is_linked:
        combine = normal_in.links[0].from_node
        if combine.bl_idname != "ShaderNodeCombineXYZ":
            raise RuntimeError(
                f"Toon BSDF Normal is driven by {combine.bl_idname}, not Combine XYZ — aborting"
            )
        nodes_to_remove.append(combine)
        # Walk back to the 3 Attribute nodes.
        for s in combine.inputs:
            if s.is_linked:
                src = s.links[0].from_node
                if src.bl_idname == "ShaderNodeAttribute" and src.attribute_name in ATTR_NAMES:
                    nodes_to_remove.append(src)
    else:
        log("Toon BSDF Normal already unlinked — nothing to do")
        return
    log(f"removing {len(nodes_to_remove)} nodes: {[n.name for n in nodes_to_remove]}")
    for n in nodes_to_remove:
        tree.nodes.remove(n)


def main() -> None:
    args = parse_args()
    blend = Path(args.blend).resolve()
    if not blend.exists():
        raise FileNotFoundError(blend)
    log(f"opening {blend}")
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is None:
        raise RuntimeError(f"material {MATERIAL_NAME!r} not found in {blend}")
    verify_other_materials_clean()
    verify_no_other_consumers(mat)
    apply_fix(mat)
    log(f"saving {blend}")
    bpy.ops.wm.save_mainfile(filepath=str(blend))
    log("done")


if __name__ == "__main__":
    main()
