"""Assessment helper for single-input normals + Lambert shading sweeps.

This is not a canonical family runner. It is a repo helper for comparing
Lambert shading variants on one EXR while keeping the normals slabs alongside
them in a single flat folder.

Contract notes:
- opens an existing working-copy blend
- repaths the EXR input
- performs runtime-only socket wiring and optional sun patching
- renders one frame as an animation
- never saves graph changes back into the blend
"""
from __future__ import annotations

import argparse
import importlib.util
import math
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
FILE_OUTPUT_NAME = "Normals::Outputs"
SHADING_NODE_NAMES = {"x": "Shading::Lx", "y": "Shading::Ly", "z": "Shading::Lz"}
SHADING_MAX_NODE_NAME = "Shading::Max0"
SHADING_SETALPHA_NAME = "SetAlpha::shading"


def log(msg: str) -> None:
    print(f"[_render_normals_single_input_assess] {msg}")


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
    p.add_argument("--mode", choices=("full", "shading-only"), default="full")
    p.add_argument("--shading-label", default=None)
    p.add_argument("--sun", default=None, help="Optional x,y,z Lambert vector")
    p.add_argument(
        "--ambient-floor",
        type=float,
        default=0.0,
        help="Optional ambient floor for lifted Lambert: floor + (1-floor) * max(0, N.L)",
    )
    return p.parse_args(argv)


def load_exr_into_node(tree: bpy.types.NodeTree, exr_path: Path) -> None:
    node = tree.nodes.get(EXR_NODE_NAME)
    if node is None:
        raise RuntimeError(f"image node {EXR_NODE_NAME!r} not found")
    if not exr_path.exists():
        raise FileNotFoundError(f"EXR not found: {exr_path}")
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


def parse_sun_vector(raw: str) -> tuple[float, float, float]:
    xyz = tuple(float(p.strip()) for p in raw.split(","))
    if len(xyz) != 3:
        raise ValueError(f"--sun must have 3 comma-separated values, got {raw!r}")
    length = math.sqrt(sum(v * v for v in xyz))
    if length <= 0.0:
        raise ValueError(f"--sun must not be zero-length, got {raw!r}")
    return tuple(v / length for v in xyz)


def patch_sun_vector(tree: bpy.types.NodeTree, sun: tuple[float, float, float]) -> None:
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        node = tree.nodes.get(SHADING_NODE_NAMES[axis])
        if node is None:
            raise RuntimeError(f"runtime shading node {SHADING_NODE_NAMES[axis]!r} not found")
        node.inputs[1].default_value = sun[idx]
        node.label = f"{axis.upper()} * L{axis.upper()} ({sun[idx]:.3f})"
    log(f"  patched Lambert sun to ({sun[0]:.3f}, {sun[1]:.3f}, {sun[2]:.3f})")


def apply_ambient_floor(tree: bpy.types.NodeTree, floor: float) -> None:
    if floor <= 0.0:
        return
    if not 0.0 < floor < 1.0:
        raise ValueError(f"--ambient-floor must be between 0 and 1, got {floor}")

    max0 = tree.nodes.get(SHADING_MAX_NODE_NAME)
    set_alpha = tree.nodes.get(SHADING_SETALPHA_NAME)
    if max0 is None or set_alpha is None:
        raise RuntimeError("ambient-floor patch requires Shading::Max0 and SetAlpha::shading")

    for link in list(set_alpha.inputs["Image"].links):
        tree.links.remove(link)

    mul = tree.nodes.new("CompositorNodeMath")
    mul.name = "Assess::AmbientLiftMul"
    mul.label = f"* {1.0 - floor:.3f}"
    mul.operation = "MULTIPLY"
    mul.location = (max0.location.x + 180.0, max0.location.y + 30.0)
    mul.inputs[1].default_value = 1.0 - floor

    add = tree.nodes.new("CompositorNodeMath")
    add.name = "Assess::AmbientLiftAdd"
    add.label = f"+ {floor:.3f}"
    add.operation = "ADD"
    add.location = (max0.location.x + 380.0, max0.location.y + 30.0)
    add.inputs[1].default_value = floor

    tree.links.new(max0.outputs[0], mul.inputs[0])
    tree.links.new(mul.outputs[0], add.inputs[0])
    tree.links.new(add.outputs[0], set_alpha.inputs["Image"])
    log(f"  patched ambient floor to {floor:.3f}")


def configure_file_output(
    tree: bpy.types.NodeTree,
    fout: bpy.types.Node,
    stem: str,
    mode: str,
    shading_label: str | None,
) -> None:
    full_mapping = {
        "normal_x_": f"{stem}_normal_x_",
        "normal_y_": f"{stem}_normal_y_",
        "normal_z_": f"{stem}_normal_z_",
        "shading_": f"{stem}_shading_",
    }
    shading_path = f"{stem}_shading__{shading_label}_" if shading_label else f"{stem}_shading_"
    for idx, slot in enumerate(fout.file_slots):
        if mode == "shading-only" and slot.path != "shading_":
            for link in list(fout.inputs[idx].links):
                tree.links.remove(link)
            slot.path = f"_disabled_{idx}_"
            continue
        new_path = shading_path if slot.path == "shading_" else full_mapping.get(slot.path)
        if new_path is not None:
            slot.path = new_path


def strip_frame_suffix(out_dir: Path) -> None:
    for p in sorted(out_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        name = p.name
        if "_0001" in name:
            target = p.with_name(name.replace("_0001", ""))
            if target.exists():
                target.unlink()
            p.replace(target)
            log(f"  renamed {name} -> {target.name}")


def cleanup_discard(out_dir: Path) -> None:
    for discard in out_dir.glob("_discard_render*"):
        try:
            discard.unlink()
        except OSError:
            pass


def main() -> None:
    args = parse_args()
    blend_path = Path(args.blend)
    exr_path = Path(args.exr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"opening {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise RuntimeError(f"scene {SCENE_NAME!r} not found or has no node tree")
    tree = scene.node_tree

    load_exr_into_node(tree, exr_path)
    wire_exr_sockets(tree)
    if args.sun:
        patch_sun_vector(tree, parse_sun_vector(args.sun))
    if args.ambient_floor:
        apply_ambient_floor(tree, args.ambient_floor)

    fout = tree.nodes.get(FILE_OUTPUT_NAME)
    if fout is None:
        raise RuntimeError(f"{FILE_OUTPUT_NAME!r} not found")
    fout.base_path = str(out_dir)
    configure_file_output(tree, fout, args.stem, args.mode, args.shading_label)

    width, height = read_exr_dimensions(exr_path)
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
    except Exception:
        pass
    try:
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
