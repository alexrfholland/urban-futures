"""Targeted inspection: find the Sizes workflow region in the merged template.

Looks in the 'Current' scene only. Prints:
- every node whose name starts with 'Sizes::'
- every frame, with label, in the Current scene
- any frame whose label or name contains 'size' (case-insensitive)
- for any matched frame, lists its child nodes
"""
from __future__ import annotations

from pathlib import Path

import bpy


COMPOSITOR_ROOT = Path(
    r"d:/2026 Arboreal Futures/urban-futures/_code-refactored/refactor_code/blender/compositor"
)
MERGED = COMPOSITOR_ROOT / "canonical_templates" / "edge_lab_final_template_safe_rebuild_20260405.blend"


def children_of(tree: bpy.types.NodeTree, frame: bpy.types.Node) -> list[bpy.types.Node]:
    kids = []
    for n in tree.nodes:
        cur = n
        while getattr(cur, "parent", None) is not None:
            cur = cur.parent
            if cur == frame:
                kids.append(n)
                break
    return kids


def main() -> None:
    bpy.ops.wm.open_mainfile(filepath=str(MERGED))
    scene = bpy.data.scenes.get("Current")
    if scene is None or scene.node_tree is None:
        print("No Current scene")
        return
    tree = scene.node_tree
    print(f"[Current] nodes={len(tree.nodes)} links={len(tree.links)}")

    print()
    print("== nodes starting with 'Sizes::' ==")
    sizes_nodes = [n for n in tree.nodes if n.name.startswith("Sizes::")]
    print(f"count={len(sizes_nodes)}")
    for n in sizes_nodes:
        parent = n.parent.name if n.parent else None
        extra = ""
        if n.bl_idname == "CompositorNodeOutputFile":
            slots = [s.path for s in n.file_slots]
            extra = f" slots={slots}"
        print(f"  {n.bl_idname:30s} {n.name!r:60s} parent={parent!r}{extra}")

    print()
    print("== frames containing 'size' (case-insensitive) in name or label ==")
    for f in tree.nodes:
        if f.bl_idname != "NodeFrame":
            continue
        if "size" in f.name.lower() or "size" in (f.label or "").lower():
            print(f"  frame name={f.name!r}  label={f.label!r}")

    print()
    print("== all frames with no parent (top-level) ==")
    for f in tree.nodes:
        if f.bl_idname != "NodeFrame":
            continue
        if f.parent is None:
            print(f"  frame name={f.name!r}  label={f.label!r}")

    print()
    print("== output file nodes whose name contains 'size' or whose slot paths contain 'size' ==")
    for n in tree.nodes:
        if n.bl_idname != "CompositorNodeOutputFile":
            continue
        slot_names = [s.path for s in n.file_slots]
        hit = "size" in n.name.lower() or any("size" in s.lower() for s in slot_names)
        if hit:
            parent = n.parent.name if n.parent else None
            print(f"  {n.name!r:60s} parent={parent!r}  slots={slot_names}")


if __name__ == "__main__":
    main()
