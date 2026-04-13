"""Read-only introspection of compositor_mist.blend and
compositor_depth_outliner.blend — reports the Image/EXR input node setup,
the layer bindings on multilayer sockets, the File Output slot paths and
links, and any view-transform / format flags that influence output depth.

No edits. No rebuild. Just print what the canonicals actually are.
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy


_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
CANONICAL = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor" / "canonical_templates"


def dump_image_node(node: bpy.types.Node) -> None:
    img = node.image
    print(f"    image={img.name if img else None!r} "
          f"filepath={img.filepath if img else None!r} "
          f"source={img.source if img else None!r}")
    if img is not None:
        try:
            layers = [l.name for l in img.layers]
        except Exception:
            layers = ["<no .layers attr>"]
        print(f"    exr layers on image: {layers}")
    print(f"    node.layer={getattr(node, 'layer', None)!r}")
    print(f"    node.view={getattr(node, 'view', None)!r}")
    print(f"    outputs:")
    for s in node.outputs:
        print(f"      - {s.name!r} enabled={s.enabled} links={len(s.links)}")


def dump_file_output(node: bpy.types.Node) -> None:
    print(f"    base_path={node.base_path!r} mute={node.mute}")
    fmt = node.format
    print(f"    format: file_format={fmt.file_format} color_mode={fmt.color_mode} "
          f"color_depth={fmt.color_depth} "
          f"compression={getattr(fmt, 'compression', '?')} "
          f"view_transform_applied={getattr(fmt, 'view_settings', '?')}")
    for i, slot in enumerate(node.file_slots):
        sock = node.inputs[i]
        use_override = getattr(slot, "use_node_format", True)
        src = ""
        if sock.is_linked:
            lk = sock.links[0]
            src = f" <- {lk.from_node.name!r}.{lk.from_socket.name!r}"
        print(f"      slot[{i}] path={slot.path!r} use_node_format={use_override}"
              f" linked={sock.is_linked}{src}")
        if not use_override:
            sf = slot.format
            print(f"        slot format: file_format={sf.file_format} "
                  f"color_mode={sf.color_mode} color_depth={sf.color_depth}")


def dump_scene(scene: bpy.types.Scene) -> None:
    print(f"  scene={scene.name!r} use_nodes={scene.use_nodes} "
          f"use_compositing={scene.render.use_compositing}")
    print(f"    view_transform={scene.view_settings.view_transform} "
          f"look={scene.view_settings.look} "
          f"exposure={scene.view_settings.exposure} "
          f"gamma={scene.view_settings.gamma}")
    print(f"    display_device={scene.display_settings.display_device}")
    tree = scene.node_tree
    if tree is None:
        return
    image_nodes = [n for n in tree.nodes if n.bl_idname == "CompositorNodeImage"]
    fo_nodes = [n for n in tree.nodes if n.bl_idname == "CompositorNodeOutputFile"]
    print(f"    image nodes: {[n.name for n in image_nodes]}")
    for n in image_nodes:
        print(f"  --- Image node {n.name!r} ---")
        dump_image_node(n)
    print(f"    file output nodes: {[n.name for n in fo_nodes]}")
    for n in fo_nodes:
        print(f"  --- FileOutput node {n.name!r} ---")
        dump_file_output(n)
    print(f"    all muted nodes: {[n.name for n in tree.nodes if n.mute]}")


for blend_name in ("compositor_mist.blend", "compositor_depth_outliner.blend"):
    blend = CANONICAL / blend_name
    print(f"\n================ {blend_name} ================")
    print(f"path: {blend}  exists={blend.is_file()}")
    if not blend.is_file():
        continue
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    for scene in bpy.data.scenes:
        dump_scene(scene)
