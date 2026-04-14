"""Run the development ambient20 normals assessment blend once per EXR.

This is a non-canonical helper for assessment work. It preserves the contract
shape by doing one compositor render per EXR while letting the blend own the
full output band.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import bpy


_HELPER_PATH = Path(__file__).resolve().parent / "_exr_header.py"
_spec = importlib.util.spec_from_file_location("_exr_header", _HELPER_PATH)
_exr_header = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exr_header)
read_exr_dimensions = _exr_header.read_exr_dimensions

SCENE_NAME = "Current"
EXR_NODE_NAME = "EXR Input"
IDMASK_NODE_NAME = "Arboreal IDMask"
SEPARATE_NODE_NAME = "Separate Normal"
FILE_OUTPUT_NAME = "NormalsAssess::Outputs"


def log(msg: str) -> None:
    print(f"[_render_normals_assess_ambient20_band] {msg}")


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
    p.add_argument("--stem", required=True)
    return p.parse_args(argv)


def load_exr_into_node(tree: bpy.types.NodeTree, exr_path: Path) -> None:
    node = tree.nodes.get(EXR_NODE_NAME)
    if node is None:
        raise RuntimeError(f"image node {EXR_NODE_NAME!r} not found")
    img = bpy.data.images.load(str(exr_path), check_existing=True)
    img.source = "FILE"
    node.image = img
    log(f"  {EXR_NODE_NAME} -> {exr_path.name}")


def wire_exr_sockets(tree: bpy.types.NodeTree) -> None:
    exr = tree.nodes.get(EXR_NODE_NAME)
    idmask = tree.nodes.get(IDMASK_NODE_NAME)
    sep = tree.nodes.get(SEPARATE_NODE_NAME)
    if exr is None or idmask is None or sep is None:
        raise RuntimeError("required nodes missing from tree")
    bpy.context.view_layer.update()
    index_socket = next((s for s in exr.outputs if s.name == "IndexOB"), None)
    normal_socket = next((s for s in exr.outputs if s.name == "Normal"), None)
    if index_socket is None or normal_socket is None:
        raise RuntimeError("EXR Image node missing IndexOB or Normal output")
    for link in list(tree.links):
        if link.to_node is idmask and link.to_socket.name == "ID value":
            tree.links.remove(link)
        if link.to_node is sep and link.to_socket == sep.inputs[0]:
            tree.links.remove(link)
    tree.links.new(index_socket, idmask.inputs["ID value"])
    tree.links.new(normal_socket, sep.inputs[0])
    log("  wired IndexOB -> IDMask, Normal -> Separate")


def prefix_slot_paths(fout: bpy.types.Node, stem: str) -> None:
    for slot in fout.file_slots:
        base = slot.path.rstrip("_")
        slot.path = f"{stem}_{base}_"


def strip_frame_suffix(out_dir: Path) -> None:
    for path in sorted(out_dir.glob("*_0001.png")):
        final = path.with_name(path.name.replace("_0001", ""))
        if final.exists():
            final.unlink()
        path.replace(final)
        log(f"  renamed {path.name} -> {final.name}")


def cleanup_discard(out_dir: Path) -> None:
    for discard in out_dir.glob("_discard_render*"):
        try:
            discard.unlink()
        except OSError:
            pass


def main() -> None:
    args = parse_args()
    blend = Path(args.blend)
    exr = Path(args.exr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"opening {blend}")
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise RuntimeError(f"scene {SCENE_NAME!r} not found or has no node tree")
    tree = scene.node_tree

    load_exr_into_node(tree, exr)
    wire_exr_sockets(tree)

    fout = tree.nodes.get(FILE_OUTPUT_NAME)
    if fout is None:
        raise RuntimeError(f"{FILE_OUTPUT_NAME!r} not found")
    fout.base_path = str(out_dir)
    prefix_slot_paths(fout, args.stem)

    width, height = read_exr_dimensions(exr)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    scene.render.filepath = str(out_dir / "_discard_render.png")
    try:
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    log(f"rendering {width}x{height}")
    bpy.context.window.scene = scene
    bpy.ops.render.render(animation=True, scene=scene.name)
    strip_frame_suffix(out_dir)
    cleanup_discard(out_dir)
    log("done")


if __name__ == "__main__":
    main()
