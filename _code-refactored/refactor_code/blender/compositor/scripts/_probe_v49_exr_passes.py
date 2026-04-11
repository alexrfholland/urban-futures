"""One-off probe: list the render passes in the v4.9 city EXRs.

Loads each EXR via bpy.data.images.load and reports the image name,
size, and (for multilayer EXRs) the layer list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        paths = [Path(p) for p in argv[argv.index("--") + 1 :]]
    else:
        paths = []

    if not paths:
        print("no EXR paths provided")
        return

    for exr_path in paths:
        print(f"\n=== {exr_path.name} ===")
        if not exr_path.exists():
            print("  MISSING")
            continue
        img = bpy.data.images.load(str(exr_path), check_existing=False)
        print(f"  size: {tuple(img.size)}")
        print(f"  source: {img.source}")
        print(f"  type: {img.type}")
        print(f"  depth: {img.depth}")
        print(f"  file_format: {img.file_format}")
        # Multilayer EXRs expose .render_slots only for render buffers, not
        # file-loaded images. For file-loaded multilayer EXRs we have to
        # create a compositor Image node to see the passes.

    # Build a temporary compositor scene and inspect sockets.
    print("\n=== socket probe via compositor Image nodes ===")
    scene = bpy.data.scenes[0]
    scene.use_nodes = True
    tree = scene.node_tree
    # Clear the tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)

    for exr_path in paths:
        if not exr_path.exists():
            continue
        img = bpy.data.images.get(exr_path.name) or bpy.data.images.load(str(exr_path))
        node = tree.nodes.new("CompositorNodeImage")
        node.image = img
        # Force evaluate
        bpy.context.view_layer.update()
        print(f"\n  {exr_path.name}:")
        for s in node.outputs:
            print(f"    socket {s.name!r}  type={s.type}  enabled={s.enabled}")


if __name__ == "__main__":
    main()
