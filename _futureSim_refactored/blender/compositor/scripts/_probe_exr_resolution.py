"""One-off: read true pixel dimensions of EXR files via a compositor pipeline.

`bpy.data.images.load(...).size` returns (0, 0) for EXRs that have not yet
been evaluated, so instead we load each EXR into a tiny compositor graph,
trigger an eval, and then inspect `img.size`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv
    paths = [Path(p) for p in argv[argv.index("--") + 1 :]] if "--" in argv else []
    if not paths:
        print("no EXR paths")
        return

    scene = bpy.data.scenes[0]
    scene.use_nodes = True
    tree = scene.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)

    for p in paths:
        if not p.exists():
            print(f"{p.name}: MISSING")
            continue
        # Load and attach to a compositor Image node, then force update.
        img = bpy.data.images.load(str(p), check_existing=False)
        # Force pixel decode by reading the header into a temp scene.
        scene.render.resolution_x = 1
        scene.render.resolution_y = 1
        node = tree.nodes.new("CompositorNodeImage")
        node.image = img
        try:
            img.reload()
        except Exception:
            pass
        # Access img.pixels[0] forces decode in Blender 4.x.
        try:
            _ = img.pixels[0]
        except Exception as e:
            print(f"  {p.name}: pixels access failed: {e}")
        w, h = int(img.size[0]), int(img.size[1])
        print(f"{p.name}: {w} x {h}")


if __name__ == "__main__":
    main()
