from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import bpy


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


BLEND_PATH = env_path(
    "EDGE_LAB_BLEND_PATH",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/edge_lab_final_template.blend",
)
OUTPUT_DIR = env_path(
    "EDGE_LAB_OUTPUT_DIR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/outputs/edge_lab_final_template_bioenvelope",
)
EXISTING_EXR = env_path(
    "EDGE_LAB_EXISTING_EXR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6-city/city-existing_condition.exr",
)
TRENDING_EXR = env_path(
    "EDGE_LAB_TRENDING_EXR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6-city/city-trending_state.exr",
)
BIOENVELOPE_EXR = env_path(
    "EDGE_LAB_BIOENVELOPE_EXR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6-city/city-city_bioenvelope.exr",
)
BIOENVELOPE_TRENDING_EXR = env_path(
    "EDGE_LAB_BIOENVELOPE_TRENDING_EXR",
    os.environ.get("EDGE_LAB_BIOENVELOPE_TRENDING", str(TRENDING_EXR)),
)
SCENE_NAME = os.environ.get("EDGE_LAB_SCENE_NAME", "Current")


def log(message: str) -> None:
    print(f"[render_edge_lab_current_bioenvelopes] {message}")


def set_standard_view(scene: bpy.types.Scene) -> None:
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


def require_node(node_tree: bpy.types.NodeTree, name: str) -> bpy.types.Node:
    node = node_tree.nodes.get(name)
    if node is None:
        raise ValueError(f"Missing node: {name}")
    return node


def require_any_node(node_tree: bpy.types.NodeTree, names: list[str]) -> bpy.types.Node:
    for name in names:
        node = node_tree.nodes.get(name)
        if node is not None:
            return node
    raise ValueError(f"Missing node from candidates: {names}")


def require_node_by_type(node_tree: bpy.types.NodeTree, bl_idname: str) -> bpy.types.Node:
    for node in node_tree.nodes:
        if node.bl_idname == bl_idname:
            return node
    raise ValueError(f"Missing node of type: {bl_idname}")


def require_node_by_label(
    node_tree: bpy.types.NodeTree,
    label: str,
    bl_idname: str | None = None,
) -> bpy.types.Node:
    for node in node_tree.nodes:
        if node.label != label:
            continue
        if bl_idname is not None and node.bl_idname != bl_idname:
            continue
        return node
    raise ValueError(f"Missing node with label: {label}")


def find_workflow_output_node(node_tree: bpy.types.NodeTree) -> bpy.types.Node | None:
    for name in ("Current BioEnvelope ::Outputs", "Current BioEnvelope::Outputs"):
        node = node_tree.nodes.get(name)
        if node is not None and node.bl_idname == "CompositorNodeOutputFile":
            return node
    return None


def repath_exr_node(node: bpy.types.Node, filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(f"EXR not found: {filepath}")
    if node.image is None:
        raise ValueError(f"Node '{node.name}' has no image")
    node.image.filepath = str(filepath)
    node.image.reload()


def detect_resolution(exr_paths: list[Path], images: list[bpy.types.Image]) -> tuple[int, int]:
    for path in exr_paths:
        try:
            result = subprocess.run(
                ["oiiotool", "--info", "-v", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            match = re.search(r"(\d+)\s*x\s*(\d+)", result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
        except Exception:
            continue
    for image in images:
        width, height = image.size[:]
        if width > 0 and height > 0:
            return width, height
    return 3840, 2160


def rename_output(rendered_path: Path) -> Path:
    final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
    if rendered_path.exists():
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")
    return final_path


def move_from_fallback_dir(fallback_path: Path, final_path: Path) -> Path:
    if fallback_path.exists():
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            final_path.unlink()
        fallback_path.replace(final_path)
        log(f"Moved {fallback_path} -> {final_path}")
    return final_path


def rename_family_outputs(output_dir: Path) -> None:
    for rendered_path in sorted(output_dir.glob("*_0001.png")):
        final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")


def render_socket_to_png(
    scene: bpy.types.Scene,
    node_tree: bpy.types.NodeTree,
    image_socket,
    output_path: Path,
) -> None:
    composite = require_node_by_type(node_tree, "CompositorNodeComposite")
    viewer = require_node_by_type(node_tree, "CompositorNodeViewer")
    for link in list(composite.inputs[0].links):
        node_tree.links.remove(link)
    for link in list(viewer.inputs[0].links):
        node_tree.links.remove(link)
    node_tree.links.new(image_socket, composite.inputs[0])
    node_tree.links.new(image_socket, viewer.inputs[0])
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    log(f"Wrote {output_path}")


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(BLEND_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {BLEND_PATH}")

    node_tree = scene.node_tree
    set_standard_view(scene)

    existing = require_any_node(node_tree, ["Current BioEnvelope :: EXR Existing", "AO::EXR Existing"])
    exr_bio = require_node(node_tree, "Current BioEnvelope :: EXR BioEnvelope")
    exr_trending = require_node(node_tree, "Current BioEnvelope :: EXR Trending")
    repath_exr_node(existing, EXISTING_EXR)
    repath_exr_node(exr_bio, BIOENVELOPE_EXR)
    repath_exr_node(exr_trending, BIOENVELOPE_TRENDING_EXR)

    palette_suffixes = (
        "full-image",
        "exoskeleton",
        "brownroof",
        "otherground",
        "rewilded",
        "footprintdepaved",
        "livingfacade",
        "greenroof",
    )
    stems = (
        [f"base_bioenvelope_{suffix}" for suffix in palette_suffixes]
        + [f"positive_bioenvelope_{suffix}" for suffix in palette_suffixes]
        + [f"trending_bioenvelope_{suffix}" for suffix in palette_suffixes]
        + [
            "positive_bioenvelope_outlines-depth",
            "positive_bioenvelope_outlines-simple",
            "base_bioenvelope_outlines-depth",
            "base_bioenvelope_outlines-simple",
            "trending_bioenvelope_outlines-depth",
            "trending_bioenvelope_outlines-simple",
        ]
    )
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile":
            node.mute = True

    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    width, height = detect_resolution(
        [BIOENVELOPE_EXR, TRENDING_EXR, EXISTING_EXR],
        [exr_bio.image, exr_trending.image, existing.image],
    )
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    set_standard_view(scene)

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    workflow_output = find_workflow_output_node(node_tree)
    if workflow_output is not None:
        workflow_group = require_any_node(
            node_tree,
            [
                "Current BioEnvelope :: Workflow Group",
                "Current BioEnvelope::Workflow Group",
            ],
        )
        workflow_output.base_path = str(OUTPUT_DIR)
        workflow_output.format.file_format = "PNG"
        workflow_output.format.color_mode = "RGBA"
        workflow_output.format.color_depth = "8"
        workflow_output.mute = False
        scene.render.filepath = str(OUTPUT_DIR / "_discard_render.png")
        bpy.ops.render.render(write_still=True, scene=scene.name)
        rename_family_outputs(OUTPUT_DIR)
        for slot in workflow_output.file_slots:
            stem = slot.path.rstrip("_")
            out_path = OUTPUT_DIR / f"{stem}.png"
            # Blender sometimes skips newly added workflow slots on this node tree.
            # Fall back to a direct socket render so the saved blend stays the source of truth.
            if not out_path.exists():
                source_socket = workflow_group.outputs.get(stem) or workflow_group.outputs.get(
                    stem.replace("-", "_")
                )
                if source_socket is not None:
                    render_socket_to_png(scene, node_tree, source_socket, out_path)
            if out_path.exists():
                log(f"Wrote {out_path}")
        return

    outputs = []
    for stem in stems:
        outputs.append(
            (
                require_node_by_label(node_tree, stem, "NodeReroute"),
                OUTPUT_DIR / f"{stem}.png",
            )
        )

    final_paths = []
    for node, output_path in outputs:
        render_socket_to_png(scene, node_tree, node.outputs[0], output_path)
        final_paths.append(output_path)
    for path in final_paths:
        log(f"Wrote {path}")


if __name__ == "__main__":
    main()
