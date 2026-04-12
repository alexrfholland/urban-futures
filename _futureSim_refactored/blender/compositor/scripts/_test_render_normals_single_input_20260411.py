"""Thin runner for the rebuilt single-input normals template.

Per the Compositor Template Contract, this runner does not touch graph
structure — it only:

  1. opens the rebuilt working-copy blend
  2. loads the EXR into the `EXR Input` Image node
  3. wires IndexOB -> Arboreal IDMask.ID value and Normal -> Separate Normal
     (these sockets only appear on the Image node after an image is loaded,
     which is why the template build script cannot wire them itself)
  4. sets the File Output base_path to an output folder
  5. renames the File Output slot prefixes so files land with a
     scenario-aware stem (e.g. `pathway_tree_normal_x.png`)
  6. renders frame 1
  7. strips the `_0001` frame suffix from the output filenames

Usage:

    blender --background --python _test_render_normals_single_input_20260411.py \\
        -- \\
        --blend "<single-input working copy blend>" \\
        --exr "<path to v4.9 multilayer EXR>" \\
        --out "<output folder>" \\
        --stem "<scenario stem e.g. pathway_tree>"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import bpy


# Load the sibling EXR header helper without relying on package imports
# (this script is run via `blender --background --python ...` which does not
# put the script's directory on sys.path as a package).
_HELPER_PATH = Path(__file__).resolve().parent / "_exr_header.py"
_spec = importlib.util.spec_from_file_location("_exr_header", _HELPER_PATH)
_exr_header = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exr_header)
read_exr_dimensions = _exr_header.read_exr_dimensions


SCENE_NAME = "Current"
EXR_NODE_NAME = "EXR Input"
IDMASK_NODE_NAME = "Arboreal IDMask"
SEPARATE_NODE_NAME = "Separate Normal"
FILE_OUTPUT_NAME = "Normals::Outputs"

# Base slot path suffixes defined by the template rebuild.
SLOT_SUFFIXES = ("normal_x_", "normal_y_", "normal_z_", "shading_")


def log(msg: str) -> None:
    print(f"[test_render_normals_single_input] {msg}")


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--exr", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stem", required=True, help="e.g. pathway_tree, priority_tree, existing_condition")
    return p.parse_args(argv)


def load_exr_into_node(tree: bpy.types.NodeTree, exr_path: Path) -> bpy.types.Node:
    node = tree.nodes.get(EXR_NODE_NAME)
    if node is None:
        raise RuntimeError(f"image node {EXR_NODE_NAME!r} not found")
    if not exr_path.exists():
        raise FileNotFoundError(f"EXR not found: {exr_path}")
    img = bpy.data.images.load(str(exr_path), check_existing=True)
    img.source = "FILE"
    node.image = img
    log(f"  {EXR_NODE_NAME} -> {exr_path.name}")
    return node


def wire_exr_sockets(tree: bpy.types.NodeTree) -> None:
    exr = tree.nodes.get(EXR_NODE_NAME)
    idmask = tree.nodes.get(IDMASK_NODE_NAME)
    sep = tree.nodes.get(SEPARATE_NODE_NAME)
    if exr is None or idmask is None or sep is None:
        raise RuntimeError("required nodes missing from tree")

    # Force Blender to evaluate the image so sockets appear.
    bpy.context.view_layer.update()

    index_socket = next((s for s in exr.outputs if s.name == "IndexOB"), None)
    normal_socket = next((s for s in exr.outputs if s.name == "Normal"), None)
    if index_socket is None:
        raise RuntimeError("EXR Image node has no IndexOB output — wrong EXR?")
    if normal_socket is None:
        raise RuntimeError("EXR Image node has no Normal output — wrong EXR?")

    # Clear any stale links on the target sockets.
    for link in list(tree.links):
        if link.to_node is idmask and link.to_socket.name == "ID value":
            tree.links.remove(link)
        if link.to_node is sep and link.to_socket == sep.inputs[0]:
            tree.links.remove(link)

    tree.links.new(index_socket, idmask.inputs["ID value"])
    tree.links.new(normal_socket, sep.inputs[0])
    log("  wired IndexOB -> IDMask, Normal -> Separate")


def relabel_file_slots(fout: bpy.types.Node, stem: str) -> None:
    """Rename the File Output slot paths to use a scenario stem prefix.

    Template slot paths are `normal_x_`, `normal_y_`, `normal_z_`, `shading_`.
    We rewrite them to `{stem}_normal_x_`, `{stem}_shading_`, etc.
    """
    mapping = {
        "normal_x_": f"{stem}_normal_x_",
        "normal_y_": f"{stem}_normal_y_",
        "normal_z_": f"{stem}_normal_z_",
        "shading_": f"{stem}_shading_",
    }
    for slot in fout.file_slots:
        new_path = mapping.get(slot.path)
        if new_path is not None:
            slot.path = new_path
    log(f"  relabeled file slots with stem {stem!r}")


def detect_resolution(exr_path: Path) -> tuple[int, int]:
    """Read the true EXR displayWindow and return (width, height).

    Raises loudly if the file cannot be parsed. NEVER falls back to a
    hardcoded resolution — a silent fallback would mean an 8K EXR gets
    rendered at 4K with no warning (this has happened once; don't repeat
    it). See COMPOSITOR_TEMPLATE_CONTRACT.md § Input resolution rule.
    """
    w, h = read_exr_dimensions(exr_path)
    if w <= 0 or h <= 0:
        raise RuntimeError(f"EXR {exr_path.name} reports invalid dims {w}x{h}")
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
    exr_path = Path(args.exr)
    out_dir = Path(args.out)
    stem = args.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"opening {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise RuntimeError(f"scene {SCENE_NAME!r} not found")

    tree = scene.node_tree
    if tree is None:
        raise RuntimeError("no node tree on scene")

    log("loading EXR")
    load_exr_into_node(tree, exr_path)

    log("wiring EXR sockets")
    wire_exr_sockets(tree)

    fout = tree.nodes.get(FILE_OUTPUT_NAME)
    if fout is None:
        raise RuntimeError(f"{FILE_OUTPUT_NAME!r} not found")
    fout.base_path = str(out_dir)
    relabel_file_slots(fout, stem)
    log(f"File Output base_path = {fout.base_path}")

    w, h = detect_resolution(exr_path)
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
