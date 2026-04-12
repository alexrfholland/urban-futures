"""Enumerate EXR input hooks for a batch of canonical compositor blends.

Usage:
    blender -b -P _inspect_canonical_blend_hooks.py -- <blend1> <blend2> ...

For each blend, opens it, finds the 'Current' scene's node tree, and prints
the Image (EXR input) nodes with their name, label, and currently bound
image filepath. This is the minimum needed to decide how to wire a given
EXR family into the blend per COMPOSITOR_SYNC_CONTRACT.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy


def inspect_blend(blend_path: Path) -> None:
    print(f"\n===== {blend_path.name} =====")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.data.scenes.get("Current")
    if scene is None:
        # fall back: single scene
        if len(bpy.data.scenes) == 1:
            scene = bpy.data.scenes[0]
            print(f"  (no 'Current' scene — using only scene {scene.name!r})")
        else:
            print(f"  NO 'Current' SCENE; scenes={[s.name for s in bpy.data.scenes]}")
            return
    if scene.node_tree is None:
        print("  no compositor node tree")
        return
    tree = scene.node_tree
    image_nodes = [n for n in tree.nodes if n.bl_idname == "CompositorNodeImage"]
    print(f"  scene={scene.name!r}  image_input_nodes={len(image_nodes)}")
    for n in image_nodes:
        img = n.image.filepath if n.image else "<no image>"
        print(f"    - name={n.name!r}  label={n.label!r}  image={img!r}")


def main() -> None:
    argv = sys.argv
    if "--" not in argv:
        print("usage: blender -b -P _inspect_canonical_blend_hooks.py -- <blend1> <blend2> ...")
        return
    blends = [Path(a) for a in argv[argv.index("--") + 1 :]]
    for blend in blends:
        if not blend.exists():
            print(f"MISSING: {blend}")
            continue
        inspect_blend(blend)


if __name__ == "__main__":
    main()
