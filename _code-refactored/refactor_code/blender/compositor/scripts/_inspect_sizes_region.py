"""Dump the Sizes region of the merged template and the shading standalone for comparison.

Read-only inspection. Prints:
- frame names
- Sizes-scoped nodes in the merged template (frame, inputs, outputs, branch nodes)
- socket labels on every image node that appears to feed the Sizes branches
- the shading standalone's frame as the multi-source reference pattern
  (multiple EXR input nodes -> parallel branches -> one Outputs node with N slots)
"""
from __future__ import annotations

import os
from pathlib import Path

import bpy


COMPOSITOR_ROOT = Path(
    r"d:/2026 Arboreal Futures/urban-futures/_code-refactored/refactor_code/blender/compositor"
)
CANONICAL_ROOT = COMPOSITOR_ROOT / "canonical_templates"

MERGED = CANONICAL_ROOT / "edge_lab_final_template_safe_rebuild_20260405.blend"
SHADING = CANONICAL_ROOT / "compositor_shading.blend"


def header(msg: str) -> None:
    print()
    print("=" * 78)
    print(msg)
    print("=" * 78)


def dump_frames(tree: bpy.types.NodeTree) -> None:
    frames = [n for n in tree.nodes if n.bl_idname == "NodeFrame"]
    print(f"[frames] {len(frames)}")
    for f in frames:
        print(f"  frame: {f.name!r}  label={f.label!r}")


def dump_nodes_matching(tree: bpy.types.NodeTree, needle: str) -> None:
    needle_lower = needle.lower()
    hits = [n for n in tree.nodes if needle_lower in n.name.lower() or needle_lower in (n.label or "").lower()]
    print(f"[match '{needle}'] {len(hits)} nodes")
    for n in hits:
        parent = n.parent.name if n.parent else None
        extra = ""
        if n.bl_idname == "CompositorNodeImage" and n.image is not None:
            extra = f" image={n.image.name!r} filepath={n.image.filepath!r}"
            socket_names = [s.name for s in n.outputs]
            extra += f" output_sockets={socket_names}"
        if n.bl_idname == "CompositorNodeOutputFile":
            slot_names = [s.path for s in n.file_slots]
            extra = f" base_path={n.base_path!r} slots={slot_names}"
        print(f"  {n.bl_idname:28s} name={n.name!r}  parent={parent!r}{extra}")


def dump_children_of_frame(tree: bpy.types.NodeTree, frame_name: str) -> None:
    frame = tree.nodes.get(frame_name)
    if frame is None:
        print(f"[children-of {frame_name!r}] MISSING")
        return
    children = []
    for n in tree.nodes:
        cur = n
        while getattr(cur, "parent", None) is not None:
            cur = cur.parent
            if cur == frame:
                children.append(n)
                break
    print(f"[children-of {frame_name!r}] {len(children)} nodes")
    for n in children:
        extra = ""
        if n.bl_idname == "CompositorNodeImage" and n.image is not None:
            socket_names = [s.name for s in n.outputs]
            extra = f" image={n.image.name!r} sockets={socket_names}"
        if n.bl_idname == "CompositorNodeOutputFile":
            slot_names = [s.path for s in n.file_slots]
            extra = f" slots={slot_names}"
        print(f"  {n.bl_idname:28s} name={n.name!r}{extra}")


def dump_links_into_frame(tree: bpy.types.NodeTree, frame_name: str) -> None:
    frame = tree.nodes.get(frame_name)
    if frame is None:
        return

    def in_frame(node: bpy.types.Node) -> bool:
        cur = node
        while getattr(cur, "parent", None) is not None:
            cur = cur.parent
            if cur == frame:
                return True
        return False

    crossing = []
    for link in tree.links:
        from_in = in_frame(link.from_node)
        to_in = in_frame(link.to_node)
        if to_in and not from_in:
            crossing.append(link)
    print(f"[links crossing into {frame_name!r}] {len(crossing)}")
    for link in crossing:
        print(
            f"  {link.from_node.name!r}.{link.from_socket.name!r} "
            f"-> {link.to_node.name!r}.{link.to_socket.name!r}"
        )


def inspect_blend(path: Path, label: str, probe_frames: list[str], probe_needles: list[str]) -> None:
    header(f"{label}  ::  {path.name}")
    if not path.exists():
        print(f"MISSING: {path}")
        return
    bpy.ops.wm.open_mainfile(filepath=str(path))
    for scene in bpy.data.scenes:
        tree = scene.node_tree
        if tree is None:
            continue
        print(f"[scene {scene.name!r}] nodes={len(tree.nodes)} links={len(tree.links)}")
        dump_frames(tree)
        for needle in probe_needles:
            dump_nodes_matching(tree, needle)
        for frame_name in probe_frames:
            dump_children_of_frame(tree, frame_name)
            dump_links_into_frame(tree, frame_name)


def main() -> None:
    inspect_blend(
        MERGED,
        "MERGED TEMPLATE",
        probe_frames=["Sizes::FamilyFrame", "Current Shading :: Frame"],
        probe_needles=["Sizes", "size", "Tree Size", "tree_size"],
    )
    inspect_blend(
        SHADING,
        "SHADING STANDALONE (multi-source reference pattern)",
        probe_frames=["Current Shading :: Frame"],
        probe_needles=["Current Shading", "EXR Pathway", "EXR Priority", "EXR Trending"],
    )


if __name__ == "__main__":
    main()
