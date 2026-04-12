"""Thin runner for size outlines from the canonical size_outline_layers blend."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import bpy

REPO_ROOT = Path(__file__).resolve().parents[4]
COMPOSITOR_ROOT = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
CANONICAL_ROOT = COMPOSITOR_ROOT / "canonical_templates"


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


BLEND_PATH = env_path(
    "COMPOSITOR_BLEND_PATH",
    str(CANONICAL_ROOT / "size_outline_layers.blend"),
)
EXR_PATH = env_path(
    "COMPOSITOR_SIZE_OUTLINE_EXR",
    "",
)
OUTPUT_DIR = env_path(
    "COMPOSITOR_SIZE_OUTLINE_OUTPUT_DIR",
    "",
)
SCENE_NAME = os.environ.get("COMPOSITOR_SCENE_NAME", "SizeOutline")


def log(message: str) -> None:
    print(f"[render_current_size_outline] {message}")


def set_standard_view(scene: bpy.types.Scene) -> None:
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


def detect_resolution(exr_path: Path, image: bpy.types.Image) -> tuple[int, int]:
    try:
        result = subprocess.run(
            ["oiiotool", "--info", "-v", str(exr_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(r"(\d+)\s*x\s*(\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    width, height = image.size[:]
    if width > 0 and height > 0:
        return width, height
    return 7680, 4320


def rename_outputs(output_dir: Path) -> None:
    for rendered in sorted(output_dir.glob("*_0001.png")):
        final = rendered.with_name(rendered.name.replace("_0001", ""))
        if final.exists():
            final.unlink()
        rendered.replace(final)
        log(f"Renamed {rendered.name} -> {final.name}")


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(f"Canonical blend not found: {BLEND_PATH}")
    if not EXR_PATH or not Path(EXR_PATH).exists():
        raise FileNotFoundError(f"EXR not found: {EXR_PATH}")
    if not OUTPUT_DIR:
        raise ValueError("COMPOSITOR_SIZE_OUTLINE_OUTPUT_DIR is required")

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {BLEND_PATH}")

    node_tree = scene.node_tree

    exr_node = node_tree.nodes.get("EXR")
    if exr_node is None or exr_node.image is None:
        raise ValueError("Missing 'EXR' image node in canonical blend")
    exr_node.image.filepath = str(EXR_PATH)
    exr_node.image.reload()

    width, height = detect_resolution(Path(EXR_PATH), exr_node.image)
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
    set_standard_view(scene)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_node = node_tree.nodes.get("SizeOutlineOutput")
    if output_node is None:
        raise ValueError("Missing 'SizeOutlineOutput' file output node")
    output_node.base_path = str(output_dir)

    scene.render.filepath = str(output_dir / "_discard_render.png")
    bpy.ops.render.render(write_still=True, scene=scene.name)

    rename_outputs(output_dir)

    discard = output_dir / "_discard_render.png"
    if discard.exists():
        discard.unlink()

    for slot in output_node.file_slots:
        stem = slot.path.rstrip("_")
        out_path = output_dir / f"{stem}.png"
        if out_path.exists():
            log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
