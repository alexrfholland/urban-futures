"""Throwaway test runner for the normals remap working copy.

Thin runner per the Compositor Template Contract:
- opens the remapped working-copy blend
- repaths the three EXR Image nodes to v4.9 city EXRs
- points the new File Output node at the exploration output folder
- renders frame 1
- strips the `_0001` suffix from the output filenames

This is a working-copy / exploration helper. It does NOT modify canonical
templates and does NOT edit graph structure.

Usage:
    blender --background --python _test_render_normals_remap_20260411.py \\
        -- \\
        --blend "<path to working-copy .blend>" \\
        --exr-pathway "<path>" \\
        --exr-priority "<path>" \\
        --exr-existing "<path>" \\
        --out "<output folder>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


SCENE_NAME = "Current"
GROUP_NODE_NAME = "Normals::Workflow Group"
FILE_OUTPUT_NAME = "Normals::Outputs"

EXR_NODE_NAMES = {
    "pathway": "Normals::EXR Pathway",
    "priority": "Normals::EXR Priority",
    "existing": "Normals::EXR Existing",
}


def log(msg: str) -> None:
    print(f"[test_render_normals_remap] {msg}")


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--exr-pathway", required=True)
    p.add_argument("--exr-priority", required=True)
    p.add_argument("--exr-existing", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def repath_image_node(tree: bpy.types.NodeTree, node_name: str, exr_path: Path) -> None:
    node = tree.nodes.get(node_name)
    if node is None:
        raise RuntimeError(f"image node {node_name!r} not found")
    if not exr_path.exists():
        raise FileNotFoundError(f"EXR not found: {exr_path}")
    img = bpy.data.images.load(str(exr_path), check_existing=True)
    img.source = "FILE"
    node.image = img
    log(f"  {node_name} -> {exr_path.name}")


def detect_resolution(exr_path: Path) -> tuple[int, int]:
    img = bpy.data.images.load(str(exr_path), check_existing=True)
    w, h = int(img.size[0]), int(img.size[1])
    if w <= 0 or h <= 0:
        return 3840, 2160
    return w, h


def strip_frame_suffix(out_dir: Path) -> None:
    for p in sorted(out_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        name = p.name
        if "_0001" in name:
            new_name = name.replace("_0001", "")
            target = p.with_name(new_name)
            if target.exists():
                target.unlink()
            p.replace(target)
            log(f"  renamed {name} -> {new_name}")


def main() -> None:
    args = parse_args()
    blend_path = Path(args.blend)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"opening {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise RuntimeError(f"scene {SCENE_NAME!r} not found")

    tree = scene.node_tree
    if tree is None:
        raise RuntimeError("no node tree on scene")

    log("repathing EXR inputs")
    pathway_exr = Path(args.exr_pathway)
    priority_exr = Path(args.exr_priority)
    existing_exr = Path(args.exr_existing)
    repath_image_node(tree, EXR_NODE_NAMES["pathway"], pathway_exr)
    repath_image_node(tree, EXR_NODE_NAMES["priority"], priority_exr)
    repath_image_node(tree, EXR_NODE_NAMES["existing"], existing_exr)

    fout = tree.nodes.get(FILE_OUTPUT_NAME)
    if fout is None:
        raise RuntimeError(f"{FILE_OUTPUT_NAME!r} not found")
    fout.base_path = str(out_dir)
    log(f"File Output base_path = {fout.base_path}")

    w, h = detect_resolution(pathway_exr)
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    try:
        scene.display_settings.display_device = "sRGB"
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    log(f"rendering {w}x{h}")
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=False)

    log("stripping _0001 suffix from output filenames")
    strip_frame_suffix(out_dir)

    log("done")


if __name__ == "__main__":
    main()
