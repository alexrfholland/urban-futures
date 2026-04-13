"""One-off inspection of canonical_templates/compositor_sizes.blend.

Dumps nodes, frames, File Output slots, and Image input nodes in the
'Current' scene so we can decide how to build a thin runner for it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        blend = Path(argv[argv.index("--") + 1])
    else:
        print("usage: blender -b -P _inspect_compositor_sizes_blend.py -- <blend>")
        return
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.data.scenes.get("Current")
    if scene is None or scene.node_tree is None:
        print("no 'Current' scene or no node tree")
        return
    tree = scene.node_tree
    print(f"[Current] nodes={len(tree.nodes)} links={len(tree.links)}")

    print()
    print("== Image (EXR input) nodes ==")
    for n in tree.nodes:
        if n.bl_idname == "CompositorNodeImage":
            img = n.image.filepath if n.image else "<no image>"
            print(f"  name={n.name!r}  label={n.label!r}  image={img!r}")

    print()
    print("== File Output nodes ==")
    for n in tree.nodes:
        if n.bl_idname == "CompositorNodeOutputFile":
            slots = [s.path for s in n.file_slots]
            print(f"  name={n.name!r}  label={n.label!r}  base_path={n.base_path!r}")
            for i, slot in enumerate(n.file_slots):
                src = "unlinked"
                if n.inputs[i].is_linked:
                    link = n.inputs[i].links[0]
                    src = f"{link.from_node.name}.{link.from_socket.name}"
                print(f"    slot[{i}]  path={slot.path!r}  <- {src}")

    print()
    print("== Frames ==")
    for n in tree.nodes:
        if n.bl_idname == "NodeFrame":
            parent = n.parent.name if n.parent else None
            print(f"  name={n.name!r}  label={n.label!r}  parent={parent}")

    print()
    print("== IDMask nodes ==")
    for n in tree.nodes:
        if n.bl_idname == "CompositorNodeIDMask":
            print(f"  name={n.name!r}  index={n.index}")

    print()
    print("== Top-level node type histogram ==")
    from collections import Counter
    hist = Counter(n.bl_idname for n in tree.nodes)
    for ty, cnt in sorted(hist.items(), key=lambda x: -x[1]):
        print(f"  {cnt:3d}  {ty}")


if __name__ == "__main__":
    main()
