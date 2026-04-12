"""Headless inspector for proposal-related compositor blends.

Run with:
    blender --background --factory-startup --python <this script> -- <blend_path>

Prints a concise summary of EXR input image nodes and output-file nodes.
Dev tooling only; does not modify anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        blend_path = Path(argv[argv.index("--") + 1])
    else:
        raise SystemExit("usage: blender --background --python ... -- <blend_path>")

    print(f"INSPECT {blend_path}")
    if not blend_path.exists():
        print("  MISSING")
        return

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    for scene in bpy.data.scenes:
        nt = scene.node_tree
        if nt is None:
            continue
        print(f"  SCENE {scene.name} use_nodes={scene.use_nodes}")
        image_nodes = [n for n in nt.nodes if n.bl_idname == "CompositorNodeImage"]
        output_nodes = [n for n in nt.nodes if n.bl_idname == "CompositorNodeOutputFile"]
        print(f"    IMAGE_INPUTS ({len(image_nodes)}):")
        for n in sorted(image_nodes, key=lambda x: x.name):
            fp = n.image.filepath if n.image is not None else "<no image>"
            print(f"      - {n.name!r} image={fp}")
        print(f"    OUTPUT_FILE ({len(output_nodes)}):")
        for n in sorted(output_nodes, key=lambda x: x.name):
            slot_paths = [s.path for s in n.file_slots]
            print(f"      - {n.name!r} base={n.base_path!r} slots={slot_paths}")


if __name__ == "__main__":
    main()
