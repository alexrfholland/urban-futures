"""Probe a single library EXR via CompositorNodeImage to enumerate sockets.

Usage:
    blender -b --factory-startup -P _probe_library_exr_sockets_20260411.py -- <exr_path>
"""
from __future__ import annotations

import sys
import bpy


def main() -> None:
    argv = sys.argv
    if "--" not in argv:
        print("usage: blender -b -P _probe_library_exr_sockets_20260411.py -- <exr>")
        return
    exr = argv[argv.index("--") + 1]

    # Create a fresh scene with a compositor node tree
    scene = bpy.data.scenes.new("probe")
    scene.use_nodes = True
    tree = scene.node_tree

    img = bpy.data.images.load(exr)
    node = tree.nodes.new("CompositorNodeImage")
    node.image = img
    # Force multilayer pass discovery
    # Referencing the layers property triggers Blender to populate sockets
    try:
        layers = img.layers if hasattr(img, "layers") else None
    except Exception:
        layers = None

    print(f"EXR: {exr}")
    print(f"image.type={img.type}  source={img.source}  file_format={img.file_format}")
    print(f"CompositorNodeImage sockets ({len(node.outputs)}):")
    for s in node.outputs:
        enabled = getattr(s, "enabled", True)
        print(f"  - {s.name!r}  enabled={enabled}")


if __name__ == "__main__":
    main()
