from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = Path(
    os.environ.get(
        "EDGE_LAB_EXR_ROOT",
        str(REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6_baseline-city"),
    )
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "edge_detection_lab"
    / "outputs"
    / "exr_city_blender_baseline_depth_outliner"
)
DEFAULT_BLEND_PATH = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "edge_detection_lab"
    / "exr_city_blender_baseline_depth_outliner.blend"
)

TREE_ID = 3
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)
OUTLINER_THRESHOLD = 0.13409094512462616


def preferred_exr_path(stem: str) -> Path:
    exr_8k = EXR_ROOT / f"{stem}_8k.exr"
    exr_default = EXR_ROOT / f"{stem}.exr"
    return exr_8k if exr_8k.exists() else exr_default


OUTPUT_DIR = Path(os.environ.get("EDGE_LAB_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
BLEND_PATH = Path(os.environ.get("EDGE_LAB_BLEND_PATH", str(DEFAULT_BLEND_PATH)))
PATHWAY_EXR = Path(os.environ.get("EDGE_LAB_PATHWAY_EXR", str(preferred_exr_path("pathway_state"))))
PRIORITY_EXR = Path(os.environ.get("EDGE_LAB_PRIORITY_EXR", str(preferred_exr_path("priority"))))
TRENDING_EXR = Path(os.environ.get("EDGE_LAB_TRENDING_EXR", str(preferred_exr_path("trending_state"))))


def log(message: str) -> None:
    print(f"[render_exr_arboreal_depth_outliner_baseline_blender] {message}")


def clear_node_tree(node_tree: bpy.types.NodeTree) -> None:
    for link in list(node_tree.links):
        node_tree.links.remove(link)
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def new_node(node_tree, bl_idname, name: str, label: str, location, color=None):
    node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    if color is not None:
        node.use_custom_color = True
        node.color = color
    return node


def frame_node(node_tree, name: str, label: str, location, color):
    node = new_node(node_tree, "NodeFrame", name, label, location, color=color)
    node.label_size = 18
    node.shrink = False
    return node


def image_node(node_tree, path: Path, name: str, label: str, location):
    node = new_node(node_tree, "CompositorNodeImage", name, label, location, color=(0.12, 0.18, 0.10))
    node.image = bpy.data.images.load(str(path), check_existing=True)
    return node


def id_mask_node(node_tree, index_socket, index: int, name: str, label: str, location):
    node = new_node(node_tree, "CompositorNodeIDMask", name, label, location, color=(0.18, 0.18, 0.10))
    node.index = index
    node.use_antialiasing = True
    ensure_link(node_tree, index_socket, node.inputs["ID value"])
    return node


def math_node(node_tree, operation: str, name: str, label: str, location, clamp: bool = True):
    node = new_node(node_tree, "CompositorNodeMath", name, label, location, color=(0.18, 0.16, 0.20))
    node.operation = operation
    node.use_clamp = clamp
    return node


def normalize_node(node_tree, value_socket, name: str, label: str, location):
    node = new_node(node_tree, "CompositorNodeNormalize", name, label, location, color=(0.16, 0.18, 0.20))
    ensure_link(node_tree, value_socket, node.inputs[0])
    return node


def filter_node(node_tree, image_socket, name: str, label: str, location, filter_type: str):
    node = new_node(node_tree, "CompositorNodeFilter", name, label, location, color=(0.18, 0.16, 0.10))
    node.filter_type = filter_type
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    return node


def ramp_node(node_tree, fac_socket, name: str, label: str, location, threshold: float):
    node = new_node(node_tree, "CompositorNodeValToRGB", name, label, location, color=(0.18, 0.14, 0.14))
    node.color_ramp.interpolation = "CONSTANT"
    node.color_ramp.elements[0].position = 0.0
    node.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    node.color_ramp.elements[1].position = threshold
    node.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    ensure_link(node_tree, fac_socket, node.inputs["Fac"])
    return node


def rgb_to_bw_node(node_tree, image_socket, name: str, label: str, location):
    node = new_node(node_tree, "CompositorNodeRGBToBW", name, label, location, color=(0.14, 0.16, 0.20))
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    return node


def set_alpha_node(node_tree, image_socket, alpha_socket, name: str, label: str, location, mode: str):
    node = new_node(node_tree, "CompositorNodeSetAlpha", name, label, location, color=(0.16, 0.20, 0.16))
    node.mode = mode
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    ensure_link(node_tree, alpha_socket, node.inputs["Alpha"])
    return node


def rgb_node(node_tree, name: str, label: str, location, rgba):
    node = new_node(node_tree, "CompositorNodeRGB", name, label, location, color=(0.18, 0.12, 0.20))
    node.outputs[0].default_value = rgba
    return node


def output_node(node_tree, image_socket, stem: str, location) -> Path:
    node = new_node(node_tree, "CompositorNodeOutputFile", f"Output {stem}", f"Output {stem}", location, color=(0.12, 0.20, 0.14))
    node.base_path = str(OUTPUT_DIR)
    node.format.file_format = "PNG"
    node.format.color_mode = "RGBA"
    node.format.color_depth = "8"
    node.file_slots[0].path = f"{stem}_"
    ensure_link(node_tree, image_socket, node.inputs[0])
    return OUTPUT_DIR / f"{stem}_0001.png"


def rename_output(rendered_path: Path) -> Path:
    final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
    if rendered_path.exists():
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")
    return final_path


def render_scene(scene: bpy.types.Scene, *, write_still: bool = False) -> None:
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=write_still, scene=scene.name)


def render_socket_to_png(scene: bpy.types.Scene, node_tree: bpy.types.NodeTree, image_socket, output_path: Path) -> None:
    composite = node_tree.nodes["Composite"]
    viewer = node_tree.nodes["Viewer"]
    ensure_link(node_tree, image_socket, composite.inputs[0])
    ensure_link(node_tree, image_socket, viewer.inputs[0])
    scene.render.filepath = str(output_path)
    render_scene(scene, write_still=True)
    log(f"Wrote {output_path}")


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


def build_outliner_branch(node_tree, signal_socket, mask_socket, prefix: str, y: float, parent, signal_name: str, stem_name: str) -> tuple[list[Path], bpy.types.Node]:
    signal_norm = normalize_node(
        node_tree,
        signal_socket,
        f"{prefix}_{signal_name}_normalize",
        f"{prefix} {signal_name} normalize",
        (-980.0, y),
    )
    signal_norm.parent = parent
    depth_prepped = set_alpha_node(
        node_tree,
        signal_norm.outputs[0],
        mask_socket,
        f"{prefix}_{signal_name}_prepped",
        f"{prefix} {signal_name} prepped",
        (-760.0, y),
        "REPLACE_ALPHA",
    )
    depth_prepped.parent = parent
    kirsch = filter_node(
        node_tree,
        signal_norm.outputs[0],
        f"{prefix}_{signal_name}_kirsch",
        f"{prefix} {signal_name} kirsch",
        (-540.0, y),
        "KIRSCH",
    )
    kirsch.parent = parent
    ramp = ramp_node(
        node_tree,
        kirsch.outputs["Image"],
        f"{prefix}_{signal_name}_ramp",
        f"{prefix} {signal_name} hard threshold",
        (-320.0, y),
        OUTLINER_THRESHOLD,
    )
    ramp.parent = parent
    ramp_bw = rgb_to_bw_node(
        node_tree,
        ramp.outputs["Image"],
        f"{prefix}_{signal_name}_ramp_bw",
        f"{prefix} {signal_name} ramp bw",
        (-100.0, y),
    )
    ramp_bw.parent = parent
    masked_alpha = math_node(
        node_tree,
        "MULTIPLY",
        f"{prefix}_{signal_name}_masked_alpha",
        f"{prefix} {signal_name} masked alpha",
        (120.0, y),
    )
    masked_alpha.parent = parent
    ensure_link(node_tree, ramp_bw.outputs["Val"], masked_alpha.inputs[0])
    ensure_link(node_tree, mask_socket, masked_alpha.inputs[1])
    edge_rgb = rgb_node(
        node_tree,
        f"{prefix}_{signal_name}_edge_rgb",
        f"{prefix} {signal_name} edge rgb",
        (120.0, y - 180.0),
        EDGE_COLOR_LINEAR,
    )
    edge_rgb.parent = parent
    edge_masked = set_alpha_node(
        node_tree,
        edge_rgb.outputs[0],
        masked_alpha.outputs["Value"],
        f"{prefix}_{signal_name}_edge_masked",
        f"{prefix} {signal_name} edge masked",
        (360.0, y),
        "REPLACE_ALPHA",
    )
    edge_masked.parent = parent
    prepped_output = output_node(
        node_tree,
        depth_prepped.outputs["Image"],
        f"{prefix}_{stem_name}_normalized_visible_arboreal",
        (840.0, y + 120.0),
    )
    node_tree.nodes[f"Output {prefix}_{stem_name}_normalized_visible_arboreal"].parent = parent
    edge_output = output_node(node_tree, edge_masked.outputs["Image"], f"{prefix}_{stem_name}_outliner", (620.0, y - 120.0))
    node_tree.nodes[f"Output {prefix}_{stem_name}_outliner"].parent = parent
    return [prepped_output, edge_output], edge_masked


def build_scene(scene: bpy.types.Scene) -> list[Path]:
    scene.use_nodes = True
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
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    inputs_frame = frame_node(node_tree, "Frame Inputs", "EXR Inputs", (-1600.0, 600.0), (0.16, 0.16, 0.16))
    masks_frame = frame_node(node_tree, "Frame Masks", "Visible Arboreal Masks", (-1280.0, 500.0), (0.18, 0.18, 0.14))
    pathway_frame = frame_node(node_tree, "Frame Pathway", "Pathway Depth Outliner", (-1080.0, 760.0), (0.16, 0.20, 0.16))
    priority_frame = frame_node(node_tree, "Frame Priority", "Priority Depth Outliner", (-1080.0, 220.0), (0.16, 0.20, 0.16))
    trending_frame = frame_node(node_tree, "Frame Trending", "Trending Depth Outliner", (-1080.0, -320.0), (0.16, 0.20, 0.16))

    pathway = image_node(node_tree, PATHWAY_EXR, "EXR Pathway", "EXR Pathway", (-1500.0, 620.0))
    pathway.parent = inputs_frame
    priority = image_node(node_tree, PRIORITY_EXR, "EXR Priority", "EXR Priority", (-1500.0, 80.0))
    priority.parent = inputs_frame
    trending = image_node(node_tree, TRENDING_EXR, "EXR Trending", "EXR Trending", (-1500.0, -460.0))
    trending.parent = inputs_frame

    width, height = detect_resolution([PATHWAY_EXR, PRIORITY_EXR, TRENDING_EXR], [pathway.image, priority.image, trending.image])
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    mask_visible_pathway = id_mask_node(
        node_tree,
        pathway.outputs["IndexOB"],
        TREE_ID,
        "mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (-1240.0, 500.0),
    )
    mask_visible_pathway.parent = masks_frame
    mask_visible_trending = id_mask_node(
        node_tree,
        trending.outputs["IndexOB"],
        TREE_ID,
        "mask_visible-arboreal_trending",
        "mask_visible-arboreal_trending",
        (-1240.0, -340.0),
    )
    mask_visible_trending.parent = masks_frame
    mask_all_priority = id_mask_node(
        node_tree,
        priority.outputs["IndexOB"],
        TREE_ID,
        "mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (-1240.0, -60.0),
    )
    mask_all_priority.parent = masks_frame
    mask_visible_priority = math_node(
        node_tree,
        "MULTIPLY",
        "mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (-1020.0, 80.0),
    )
    mask_visible_priority.parent = masks_frame
    ensure_link(node_tree, mask_all_priority.outputs["Alpha"], mask_visible_priority.inputs[0])
    ensure_link(node_tree, mask_visible_pathway.outputs["Alpha"], mask_visible_priority.inputs[1])

    rendered_paths: list[Path] = []
    depth_edge_nodes: dict[str, bpy.types.Node] = {}
    mist_edge_nodes: dict[str, bpy.types.Node] = {}
    depth_paths, depth_edge_nodes["pathway"] = build_outliner_branch(
        node_tree, pathway.outputs["Depth"], mask_visible_pathway.outputs["Alpha"], "pathway", 620.0, pathway_frame, "depth", "depth"
    )
    rendered_paths.extend(depth_paths)
    mist_paths, mist_edge_nodes["pathway"] = build_outliner_branch(
        node_tree, pathway.outputs["Mist"], mask_visible_pathway.outputs["Alpha"], "pathway", 260.0, pathway_frame, "mist", "mist"
    )
    rendered_paths.extend(mist_paths)
    depth_paths, depth_edge_nodes["priority"] = build_outliner_branch(
        node_tree, priority.outputs["Depth"], mask_visible_priority.outputs["Value"], "priority", 80.0, priority_frame, "depth", "depth"
    )
    rendered_paths.extend(depth_paths)
    mist_paths, mist_edge_nodes["priority"] = build_outliner_branch(
        node_tree, priority.outputs["Mist"], mask_visible_priority.outputs["Value"], "priority", -280.0, priority_frame, "mist", "mist"
    )
    rendered_paths.extend(mist_paths)
    depth_paths, depth_edge_nodes["trending"] = build_outliner_branch(
        node_tree, trending.outputs["Depth"], mask_visible_trending.outputs["Alpha"], "trending", -460.0, trending_frame, "depth", "depth"
    )
    rendered_paths.extend(depth_paths)
    mist_paths, mist_edge_nodes["trending"] = build_outliner_branch(
        node_tree, trending.outputs["Mist"], mask_visible_trending.outputs["Alpha"], "trending", -820.0, trending_frame, "mist", "mist"
    )
    rendered_paths.extend(mist_paths)

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (1120.0, 140.0))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (1120.0, -60.0))
    composite.parent = pathway_frame
    viewer.parent = pathway_frame
    combined = new_node(
        node_tree,
        "CompositorNodeAlphaOver",
        "Combined Final Outliner",
        "Combined Final Outliner",
        (860.0, 140.0),
    )
    combined.parent = pathway_frame
    combined.premul = 1.0
    ensure_link(node_tree, depth_edge_nodes["pathway"].outputs["Image"], combined.inputs[1])
    ensure_link(node_tree, depth_edge_nodes["priority"].outputs["Image"], combined.inputs[2])
    ensure_link(node_tree, depth_edge_nodes["trending"].outputs["Image"], combined.inputs[0])
    ensure_link(node_tree, combined.outputs["Image"], composite.inputs[0])
    ensure_link(node_tree, combined.outputs["Image"], viewer.inputs[0])
    return rendered_paths


def finalize_render(scene: bpy.types.Scene, rendered_paths: list[Path]) -> None:
    render_scene(scene, write_still=False)
    for path in rendered_paths:
        rename_output(path)
    node_tree = scene.node_tree
    render_socket_to_png(
        scene,
        node_tree,
        node_tree.nodes["pathway_depth_edge_masked"].outputs["Image"],
        OUTPUT_DIR / "pathway_depth_outliner.png",
    )
    render_socket_to_png(
        scene,
        node_tree,
        node_tree.nodes["priority_depth_edge_masked"].outputs["Image"],
        OUTPUT_DIR / "priority_depth_outliner.png",
    )
    render_socket_to_png(
        scene,
        node_tree,
        node_tree.nodes["trending_depth_edge_masked"].outputs["Image"],
        OUTPUT_DIR / "trending_depth_outliner.png",
    )
    render_socket_to_png(
        scene,
        node_tree,
        node_tree.nodes["pathway_mist_edge_masked"].outputs["Image"],
        OUTPUT_DIR / "pathway_mist_outliner.png",
    )
    render_socket_to_png(
        scene,
        node_tree,
        node_tree.nodes["priority_mist_edge_masked"].outputs["Image"],
        OUTPUT_DIR / "priority_mist_outliner.png",
    )
    render_socket_to_png(
        scene,
        node_tree,
        node_tree.nodes["trending_mist_edge_masked"].outputs["Image"],
        OUTPUT_DIR / "trending_mist_outliner.png",
    )


def main() -> None:
    scene = bpy.context.scene
    rendered_paths = build_scene(scene)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
        log(f"Saved {BLEND_PATH}")
    except RuntimeError as exc:
        log(f"Skipping blend save: {exc}")
    finalize_render(scene, rendered_paths)


if __name__ == "__main__":
    main()
