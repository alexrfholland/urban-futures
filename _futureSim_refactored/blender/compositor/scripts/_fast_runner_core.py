"""Shared helper for fast compositor runners.

Implements the Render Execution Rule from COMPOSITOR_TEMPLATE_CONTRACT.md:
single-frame animation render via `bpy.ops.render.render(animation=True, ...)`.

A fast runner provides a small config describing:
- the canonical blend and scene to open
- which EXR input nodes to repath with which EXR paths
- the File Output node whose base_path to set
- the output directory

This helper does the rest:
- opens the blend
- repaths each EXR input via `image.filepath = ...; image.reload()`
- detects resolution from the first EXR header
- sets the File Output base_path (leaves it unmuted)
- sets scene.render.filepath to a discard PNG inside the output dir
- renders one frame as an animation
- strips the `_0001` / `0001.png` frame suffix from the output filenames
- cleans up the discard render
- audits that every expected File Output slot landed on disk
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import bpy

_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _exr_header import read_exr_dimensions  # noqa: E402

CANONICAL_ROOT = (
    REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
    / "canonical_templates"
)


@dataclass
class FastRunnerConfig:
    """Config for a single fast compositor render."""

    name: str
    """Short label used in log lines."""

    blend_path: Path
    """Absolute path to the canonical blend to open."""

    scene_name: str
    """Name of the scene inside the blend (usually 'Current')."""

    exr_inputs: dict[str, Path] = field(default_factory=dict)
    """Mapping of EXR image-node name in the blend -> EXR path on disk."""

    file_output_node: str = ""
    """Name of the File Output node whose base_path should be redirected."""

    output_dir: Path = field(default_factory=Path)
    """Where renders go."""

    resolution_from: Path | None = None
    """EXR to read displayWindow from. Defaults to first entry in exr_inputs."""

    extra_setup: Callable[[bpy.types.NodeTree], None] | None = None
    """Optional hook for runner-specific wiring (e.g. AOV socket routing)."""

    mute_other_file_outputs: bool = True
    """If True, mute every File Output node except `file_output_node`."""

    rebuild_file_output: bool = False
    """If True, rebuild the File Output node fresh in-memory before rendering.
    Workaround for Blender 4.2 where some saved File Output nodes (notably those
    whose inputs come directly from CompositorNodeGroup outputs) do not get
    evaluated at render time. The rebuilt node has identical slot paths and
    wiring but is created fresh so Blender considers it an active sink."""


def log(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}")


def _require_node(tree: bpy.types.NodeTree, name: str) -> bpy.types.Node:
    node = tree.nodes.get(name)
    if node is None:
        raise ValueError(f"Missing node: {name!r}")
    return node


def _repath_exr_input(tree: bpy.types.NodeTree, node_name: str, exr_path: Path, prefix: str) -> None:
    node = _require_node(tree, node_name)
    if node.image is None:
        node.image = bpy.data.images.load(str(exr_path), check_existing=True)
        log(prefix, f"loaded {exr_path.name} -> {node_name!r}")
        return
    node.image.filepath = str(exr_path)
    node.image.reload()
    log(prefix, f"repathed {node_name!r} -> {exr_path.name}")


def _ensure_camera(scene: bpy.types.Scene, prefix: str) -> None:
    """animation=True requires scene.camera to be set, even for compositor-only
    renders. If the canonical blend lacks a camera, attach the first camera
    in bpy.data or create a transient one."""
    if scene.camera is not None:
        return
    cam_obj = next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
    if cam_obj is None:
        cam_data = bpy.data.cameras.new("_compositor_dummy_cam")
        cam_obj = bpy.data.objects.new("_compositor_dummy_cam", cam_data)
        scene.collection.objects.link(cam_obj)
        log(prefix, "attached transient dummy camera")
    else:
        log(prefix, f"attached existing camera {cam_obj.name!r}")
    scene.camera = cam_obj


def _set_standard_view(scene: bpy.types.Scene) -> None:
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


def _rename_frame_suffix_outputs(output_dir: Path, prefix: str) -> None:
    # Blender's animation=True path appends frame padding ('_0001.png' for the
    # File Output slots, and '0001.png' directly after scene.render.filepath for
    # the discard render path). Strip both forms.
    for rendered in sorted(output_dir.glob("*_0001.png")):
        final = rendered.with_name(rendered.name.replace("_0001", ""))
        if final.exists():
            final.unlink()
        rendered.replace(final)
        log(prefix, f"renamed {rendered.name} -> {final.name}")


def _rebuild_file_output(tree: bpy.types.NodeTree, fo_name: str, base_path: str,
                         prefix: str) -> bpy.types.Node:
    """Recreate a File Output node with the same slot paths and input wiring.

    The saved node is left in the tree as a disconnected orphan (muted).
    Returns the freshly-built replacement node, which takes over `fo_name`."""
    old = tree.nodes[fo_name]
    source_slots: list[tuple[str, bpy.types.NodeSocket]] = []
    for i, slot in enumerate(old.file_slots):
        sock = old.inputs[i]
        if not sock.is_linked:
            raise RuntimeError(f"{fo_name} slot[{i}] {slot.path!r} is unlinked")
        source_slots.append((slot.path, sock.links[0].from_socket))
    old_location = tuple(old.location)
    old_format = old.format
    fmt_file_format = old_format.file_format
    fmt_color_mode = old_format.color_mode
    fmt_color_depth = old_format.color_depth

    for i in range(len(old.inputs)):
        for link in list(old.inputs[i].links):
            tree.links.remove(link)
    old.name = f"_orphan_{fo_name}"
    old.label = "_orphan"
    old.mute = True

    fresh = tree.nodes.new("CompositorNodeOutputFile")
    fresh.name = fo_name
    fresh.label = fo_name
    fresh.base_path = base_path
    fresh.format.file_format = fmt_file_format
    fresh.format.color_mode = fmt_color_mode
    fresh.format.color_depth = fmt_color_depth
    fresh.location = old_location

    while len(fresh.file_slots) > 1:
        fresh.file_slots.remove(fresh.file_slots[-1])
    fresh.file_slots[0].path = source_slots[0][0]
    for path, _src in source_slots[1:]:
        fresh.file_slots.new(path)
    for i, (path, src) in enumerate(source_slots):
        fresh.file_slots[i].path = path
        tree.links.new(src, fresh.inputs[i])
    log(prefix, f"rebuilt {fo_name!r} in-memory ({len(source_slots)} slots)")
    return fresh


def _cleanup_discard(output_dir: Path) -> None:
    for discard in output_dir.glob("_discard_render*"):
        try:
            discard.unlink()
        except OSError:
            pass


def run_fast_render(config: FastRunnerConfig) -> list[str]:
    """Execute one fast compositor render per the Render Execution Rule.

    Returns the list of expected slot stems (one per File Output slot).
    Raises on missing inputs or missing output files.
    """
    prefix = config.name

    if not config.blend_path.exists():
        raise FileNotFoundError(f"Canonical blend not found: {config.blend_path}")
    for node_name, exr_path in config.exr_inputs.items():
        if not exr_path.exists():
            raise FileNotFoundError(f"EXR not found for {node_name!r}: {exr_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(config.blend_path))
    scene = bpy.data.scenes.get(config.scene_name)
    if scene is None or scene.node_tree is None:
        raise ValueError(
            f"Scene {config.scene_name!r} not found or has no node_tree "
            f"in {config.blend_path}"
        )
    tree = scene.node_tree

    # Repath all EXR inputs.
    for node_name, exr_path in config.exr_inputs.items():
        _repath_exr_input(tree, node_name, exr_path, prefix)

    # Detect resolution from the chosen EXR header.
    res_source = config.resolution_from
    if res_source is None:
        res_source = next(iter(config.exr_inputs.values()))
    width, height = read_exr_dimensions(str(res_source))
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    log(prefix, f"resolution {width}x{height} from {res_source.name}")

    # Core render settings.
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    _set_standard_view(scene)
    _ensure_camera(scene, prefix)

    # Single-frame animation render.
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1

    # Optional runner-specific wiring (AOV sockets, etc).
    if config.extra_setup is not None:
        config.extra_setup(tree)

    # Target File Output node.
    if config.rebuild_file_output:
        fo = _rebuild_file_output(
            tree, config.file_output_node, str(config.output_dir), prefix
        )
    else:
        fo = _require_node(tree, config.file_output_node)
        fo.base_path = str(config.output_dir)
    fo.mute = False
    expected_slots = [slot.path.rstrip("_") for slot in fo.file_slots]

    if config.mute_other_file_outputs:
        for node in tree.nodes:
            if (
                node.bl_idname == "CompositorNodeOutputFile"
                and node.name != config.file_output_node
            ):
                node.mute = True

    # Discard target for scene.render.
    scene.render.filepath = str(config.output_dir / "_discard_render.png")

    log(prefix, f"rendering animation (frame 1, {len(expected_slots)} slots)...")
    bpy.ops.render.render(animation=True, scene=scene.name)
    log(prefix, "render done")

    _rename_frame_suffix_outputs(config.output_dir, prefix)
    _cleanup_discard(config.output_dir)

    missing: list[str] = []
    present: list[str] = []
    for stem in expected_slots:
        png = config.output_dir / f"{stem}.png"
        if png.exists():
            present.append(stem)
        else:
            missing.append(stem)
    log(prefix, f"{len(present)}/{len(expected_slots)} slots present")
    if missing:
        log(prefix, f"MISSING: {missing}")
        raise RuntimeError(
            f"{config.name}: {len(missing)} slots missing after render: {missing}"
        )
    return expected_slots
