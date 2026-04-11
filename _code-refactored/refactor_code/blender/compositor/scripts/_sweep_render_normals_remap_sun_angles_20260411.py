"""Throwaway angle-sweep runner for the normals remap working copy.

Loads the remapped working-copy blend once, then for each sun vector in a
list it:
 1. patches the `shading_lx/ly/lz` Math node default_values on all three
    workflow branches (pathway, priority, existing)
 2. sets the File Output base_path to an angle-specific subfolder
 3. renders frame 1
 4. strips the `_0001` suffix from the rendered filenames

This is an exploration helper. It does NOT modify the canonical template or
save anything back to the blend on disk.

Note: this runner re-renders the x/y/z slab outputs for every angle even
though they don't depend on the sun vector. The compositor is fast enough
that the waste doesn't matter; it also gives a trivial way to confirm the
slabs are identical across angles.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy


SCENE_NAME = "Current"
FILE_OUTPUT_NAME = "Normals::Outputs"
EXR_NODE_NAMES = {
    "pathway": "Normals::EXR Pathway",
    "priority": "Normals::EXR Priority",
    "existing": "Normals::EXR Existing",
}
WORKFLOWS = ("pathway", "priority", "existing")


def log(msg: str) -> None:
    print(f"[sweep_render_normals] {msg}")


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
    p.add_argument("--out-root", required=True)
    return p.parse_args(argv)


def sun_vector(elevation_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    """Convert (elevation, azimuth) to a unit sun vector.

    elevation: angle above the horizon (0 = horizon, 90 = zenith)
    azimuth:   clockwise from north (0 = N, 90 = E, 180 = S, 270 = W)
    """
    elev = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    x = math.cos(elev) * math.sin(az)  # east
    y = math.cos(elev) * math.cos(az)  # north
    z = math.sin(elev)                  # up
    length = math.sqrt(x * x + y * y + z * z)
    return (x / length, y / length, z / length)


def patch_sun_vector(tree: bpy.types.NodeTree, L: tuple[float, float, float]) -> None:
    """Write L into the shading_lx/ly/lz Math nodes for all workflows."""
    for workflow in WORKFLOWS:
        for axis_idx, axis_letter in enumerate(("x", "y", "z")):
            node_name = f"Normals::Remap::{workflow}::shading_l{axis_letter}"
            node = tree.nodes.get(node_name)
            if node is None:
                raise RuntimeError(f"math node {node_name!r} not found")
            node.inputs[1].default_value = L[axis_idx]


def repath_image_node(tree: bpy.types.NodeTree, node_name: str, exr_path: Path) -> None:
    node = tree.nodes.get(node_name)
    if node is None:
        raise RuntimeError(f"image node {node_name!r} not found")
    if not exr_path.exists():
        raise FileNotFoundError(f"EXR not found: {exr_path}")
    img = bpy.data.images.load(str(exr_path), check_existing=True)
    img.source = "FILE"
    node.image = img


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


def configure_scene(scene: bpy.types.Scene, pathway_exr: Path) -> None:
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


def main() -> None:
    args = parse_args()
    blend_path = Path(args.blend)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    log(f"opening {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise RuntimeError(f"scene {SCENE_NAME!r} not found")
    tree = scene.node_tree
    if tree is None:
        raise RuntimeError("no node tree on scene")

    pathway_exr = Path(args.exr_pathway)
    repath_image_node(tree, EXR_NODE_NAMES["pathway"], pathway_exr)
    repath_image_node(tree, EXR_NODE_NAMES["priority"], Path(args.exr_priority))
    repath_image_node(tree, EXR_NODE_NAMES["existing"], Path(args.exr_existing))

    fout = tree.nodes.get(FILE_OUTPUT_NAME)
    if fout is None:
        raise RuntimeError(f"{FILE_OUTPUT_NAME!r} not found")

    configure_scene(scene, pathway_exr)
    bpy.context.window.scene = scene

    # Angle sweep: NE azimuth (45 deg), four elevations from high to low.
    angles = [
        ("ne_elev60", 60.0, 45.0),
        ("ne_elev45", 45.0, 45.0),
        ("ne_elev30", 30.0, 45.0),
        ("ne_elev15", 15.0, 45.0),
    ]

    for label, elev, az in angles:
        L = sun_vector(elev, az)
        log(f"angle {label}: elev={elev} az={az} L={L}")
        patch_sun_vector(tree, L)

        angle_dir = out_root / label
        angle_dir.mkdir(parents=True, exist_ok=True)
        fout.base_path = str(angle_dir)

        bpy.ops.render.render(write_still=False)
        strip_frame_suffix(angle_dir)
        log(f"  wrote {label} -> {angle_dir}")

    log("done")


if __name__ == "__main__":
    main()
