"""Diagnose why mist + depth_outliner File Output slots don't fire."""
from __future__ import annotations

import sys
from pathlib import Path

import bpy

_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
CANONICAL = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor" / "canonical_templates"

for blend_name, fo_name in [
    ("compositor_mist.blend", "MistOutlines::Outputs"),
    ("compositor_depth_outliner.blend", "DepthOutliner::Outputs"),
]:
    blend = CANONICAL / blend_name
    print(f"\n=== {blend_name} ===")
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    for scene in bpy.data.scenes:
        tree = scene.node_tree
        print(f"  scene={scene.name!r} use_nodes={scene.use_nodes} node_tree={tree is not None} "
              f"use_compositing={scene.render.use_compositing} camera={scene.camera}")
        if tree is None:
            continue
        fo = tree.nodes.get(fo_name)
        print(f"  FO {fo_name!r}: exists={fo is not None}", end="")
        if fo is None:
            print()
            continue
        print(f" mute={fo.mute} base_path={fo.base_path!r}")
        for i, slot in enumerate(fo.file_slots):
            sock = fo.inputs[i]
            linked = sock.is_linked
            src = ""
            if linked:
                lk = sock.links[0]
                src = f"  <- {lk.from_node.name!r}.{lk.from_socket.name!r} (from_node muted={lk.from_node.mute})"
            print(f"    slot[{i}] path={slot.path!r} linked={linked}{src}")
        # Print upstream chain of muted nodes
        print(f"  all muted nodes: {[n.name for n in tree.nodes if n.mute]}")
        print(f"  composite nodes: {[n.name for n in tree.nodes if n.bl_idname == 'CompositorNodeComposite']}")
        print(f"  render layers nodes: {[(n.name, n.scene.name if n.scene else None, n.layer if hasattr(n,'layer') else None) for n in tree.nodes if n.bl_idname == 'CompositorNodeRLayers']}")
        print(f"  image nodes: {[n.name for n in tree.nodes if n.bl_idname == 'CompositorNodeImage']}")
