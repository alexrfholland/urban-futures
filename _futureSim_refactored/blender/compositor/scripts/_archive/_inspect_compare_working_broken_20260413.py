"""Compare compositor_mist vs compositor_intervention_int — why does one work?"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy

_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
CANONICAL = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor" / "canonical_templates"


def dump(name, blend_name, fo_name):
    blend = CANONICAL / blend_name
    print(f"\n=== {name}: {blend_name} ===")
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.data.scenes.get("Current")
    if scene is None:
        scene = bpy.data.scenes[0]
    tree = scene.node_tree
    print(f"  scene={scene.name!r} camera={scene.camera}")
    print(f"  scene.render.resolution={scene.render.resolution_x}x{scene.render.resolution_y}")
    print(f"  frame {scene.frame_start}-{scene.frame_end} current={scene.frame_current}")
    print(f"  use_nodes={scene.use_nodes} use_compositing={scene.render.use_compositing}")
    print(f"  image_settings.file_format={scene.render.image_settings.file_format}")
    print(f"  filepath={scene.render.filepath!r}")

    fo = tree.nodes.get(fo_name)
    print(f"  FO {fo_name!r}:")
    print(f"    mute={fo.mute} hide={fo.hide} base_path={fo.base_path!r}")
    print(f"    format.file_format={fo.format.file_format}")
    print(f"    format.color_mode={fo.format.color_mode}")
    print(f"    format.color_depth={fo.format.color_depth}")
    print(f"    inputs count={len(fo.inputs)} slots count={len(fo.file_slots)}")
    for i, slot in enumerate(fo.file_slots):
        sock = fo.inputs[i]
        linked = sock.is_linked
        src = ""
        if linked:
            lk = sock.links[0]
            from_node = lk.from_node
            src = f" <- {from_node.name!r} ({from_node.bl_idname}) .{lk.from_socket.name!r}"
            # Go one more hop up
            if from_node.bl_idname == "CompositorNodeGroup":
                gt = from_node.node_tree
                out = gt.nodes.get("Group Output") if gt else None
                if out is not None:
                    group_sock = next((gs for gs in out.inputs if gs.name == lk.from_socket.name), None)
                    if group_sock and group_sock.is_linked:
                        glk = group_sock.links[0]
                        src += f"  [group internal <- {glk.from_node.name!r} ({glk.from_node.bl_idname}) .{glk.from_socket.name!r}]"
        print(f"    slot[{i}] path={slot.path!r} linked={linked}{src}")

    print(f"  compositing flag: scene.node_tree = {scene.node_tree is scene.node_tree}")
    print(f"  node_tree has_links={len(tree.links)} has_nodes={len(tree.nodes)}")


dump("BROKEN", "compositor_mist.blend", "MistOutlines::Outputs")
dump("WORKING", "compositor_intervention_int.blend", "InterventionInt::Outputs")
dump("BROKEN", "compositor_depth_outliner.blend", "DepthOutliner::Outputs")
