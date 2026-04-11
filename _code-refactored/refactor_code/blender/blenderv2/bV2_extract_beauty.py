"""Headless Blender script: extract tonemapped beauty (Combined) PNG from bV2 multilayer EXRs.

Reads a newline-separated list of EXR paths from BV2_EXR_LIST (or one path from
BV2_EXR), builds a minimal compositor graph (Image -> Composite), applies the
Filmic view transform, and renders each EXR's Combined pass to:

    <exr_stem>__beauty.png           (full-resolution, sRGB 16-bit)
    <exr_stem>__beauty_preview.png   (downscaled preview, sRGB 8-bit)

This is a thin runner around Blender's compositor/view-transform — it does not
own graph logic (see COMPOSITOR_TEMPLATE_CONTRACT.md). No PIL / no OIIO.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy


PREVIEW_WIDTH = 1920  # preview long-edge; height derived from EXR aspect
VIEW_TRANSFORM = os.environ.get("BV2_VIEW_TRANSFORM", "Filmic").strip() or "Filmic"
LOOK = os.environ.get("BV2_LOOK", "None").strip() or "None"
EXPOSURE = float(os.environ.get("BV2_EXPOSURE", "0.0"))
GAMMA = float(os.environ.get("BV2_GAMMA", "1.0"))
# bV2 pipeline standard render resolution (from bV2_build_scene defaults).
# Override per-run with BV2_RES_X / BV2_RES_Y.
RES_X = int(os.environ.get("BV2_RES_X", "7680"))
RES_Y = int(os.environ.get("BV2_RES_Y", "4320"))


def log(*args) -> None:
    print("[bV2_extract_beauty]", *args, flush=True)


def resolve_exr_list() -> list[Path]:
    single = os.environ.get("BV2_EXR", "").strip()
    if single:
        return [Path(single)]
    list_raw = os.environ.get("BV2_EXR_LIST", "").strip()
    if not list_raw:
        raise RuntimeError("Set BV2_EXR (single path) or BV2_EXR_LIST (newline-separated paths)")
    paths: list[Path] = []
    for line in list_raw.splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line))
    if not paths:
        raise RuntimeError("BV2_EXR_LIST contained no paths")
    return paths


def configure_scene_view_transform(scene: bpy.types.Scene) -> None:
    vs = scene.view_settings
    vs.view_transform = VIEW_TRANSFORM
    vs.look = LOOK
    vs.exposure = EXPOSURE
    vs.gamma = GAMMA
    scene.display_settings.display_device = "sRGB"


def build_compositor_graph(scene: bpy.types.Scene) -> tuple[bpy.types.Node, bpy.types.Node]:
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    image_node = tree.nodes.new(type="CompositorNodeImage")
    image_node.name = "BV2_BEAUTY_IMAGE"
    image_node.location = (0, 0)
    comp_node = tree.nodes.new(type="CompositorNodeComposite")
    comp_node.name = "BV2_BEAUTY_COMPOSITE"
    comp_node.location = (400, 0)
    comp_node.use_alpha = True
    return image_node, comp_node


def set_render_png(scene: bpy.types.Scene, out_path: Path, width: int, height: int, depth: str) -> None:
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = depth
    scene.render.image_settings.compression = 15
    scene.render.filepath = str(out_path)
    scene.render.use_file_extension = False


def clear_loaded_images() -> None:
    for img in list(bpy.data.images):
        if img.users == 0:
            bpy.data.images.remove(img)


def extract_one(exr_path: Path) -> dict[str, str]:
    if not exr_path.exists():
        raise FileNotFoundError(exr_path)
    log("LOAD", exr_path.name)

    scene = bpy.context.scene
    configure_scene_view_transform(scene)
    image_node, comp_node = build_compositor_graph(scene)

    img = bpy.data.images.load(str(exr_path), check_existing=False)
    image_node.image = img
    # Force the image to resolve metadata (size, layers, passes) — multilayer
    # EXRs lazily populate these until reload/update is called.
    img.reload()
    bpy.context.view_layer.update()

    # For multilayer EXRs, select the first available render-layer so its
    # Combined/Image socket is exposed.
    try:
        layer_enum = image_node.bl_rna.properties["layer"].enum_items
        layer_names = [item.identifier for item in layer_enum]
        log("  layers:", layer_names)
        if layer_names:
            image_node.layer = layer_names[0]
    except Exception as e:
        log("  layer enum inspect skipped:", repr(e))

    bpy.context.view_layer.update()

    log("  sockets:", [(s.name, s.enabled) for s in image_node.outputs])

    # Connect first available "Image" output to Composite Image input
    out_socket = None
    for socket in image_node.outputs:
        if socket.name == "Image" and socket.enabled:
            out_socket = socket
            break
    if out_socket is None and image_node.outputs:
        out_socket = image_node.outputs[0]
    if out_socket is None:
        raise RuntimeError(f"No output sockets on Image node for {exr_path.name}")

    scene.node_tree.links.new(out_socket, comp_node.inputs["Image"])

    # Multilayer EXR `image.size` is unreliable in headless Blender 4.2 — it
    # returns (0, 0) until pixels are accessed in memory. Since bV2 renders
    # have a fixed pipeline resolution, we use RES_X/RES_Y from env (defaulting
    # to the bV2 8K standard). The compositor re-samples if they mismatch.
    width, height = RES_X, RES_Y
    log("  render size:", f"{width}x{height}")

    stem = exr_path.stem
    out_dir = exr_path.parent
    full_path = out_dir / f"{stem}__beauty.png"
    preview_path = out_dir / f"{stem}__beauty_preview.png"

    # Full-resolution render via compositor
    set_render_png(scene, full_path, width, height, "16")
    bpy.ops.render.render(write_still=True)
    log("WROTE", full_path.name)

    # Preview render (downscaled)
    aspect = height / width
    pw = min(PREVIEW_WIDTH, width)
    ph = max(1, int(round(pw * aspect)))
    set_render_png(scene, preview_path, pw, ph, "8")
    bpy.ops.render.render(write_still=True)
    log("WROTE", preview_path.name, f"({pw}x{ph})")

    # Cleanup image datablock to release memory before next EXR
    image_node.image = None
    bpy.data.images.remove(img)
    clear_loaded_images()

    return {
        "exr": str(exr_path),
        "beauty_full": str(full_path),
        "beauty_preview": str(preview_path),
    }


def main() -> None:
    exrs = resolve_exr_list()
    log(f"Extracting beauty from {len(exrs)} EXR(s)")
    log(f"view_transform={VIEW_TRANSFORM} look={LOOK} exposure={EXPOSURE} gamma={GAMMA}")
    results: list[dict[str, str]] = []
    for exr in exrs:
        try:
            results.append(extract_one(exr))
        except Exception as e:
            log("ERROR", exr.name, repr(e))
            raise
    log("DONE", f"{len(results)} extracted")
    for r in results:
        log("  ", Path(r["beauty_preview"]).name)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[bV2_extract_beauty] FATAL {e!r}", flush=True)
        sys.exit(1)
