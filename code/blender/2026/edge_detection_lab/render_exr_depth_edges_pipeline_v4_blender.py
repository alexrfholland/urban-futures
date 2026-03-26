from __future__ import annotations

import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"
OUTPUT_DIR = Path(
    os.environ.get(
        "EDGE_LAB_OUTPUT_DIR",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_v4" / "01_depth_edges"),
    )
)
BLEND_PATH = Path(
    os.environ.get(
        "EDGE_LAB_BLEND_PATH",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_depth_edges_v4.blend"),
    )
)

PATHWAY_EXR = Path(os.environ.get("EDGE_LAB_PATHWAY_EXR", str(EXR_ROOT / "city-pathway_state.exr")))
PRIORITY_EXR = Path(os.environ.get("EDGE_LAB_PRIORITY_EXR", str(EXR_ROOT / "city-city_priority.exr")))
TRENDING_EXR = Path(os.environ.get("EDGE_LAB_TRENDING_EXR", str(EXR_ROOT / "city-trending_state.exr")))

TREE_ID = 3
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)


def log(message: str) -> None:
    print(f"[render_exr_depth_edges_pipeline_v4_blender] {message}")


def clear_node_tree(node_tree: bpy.types.NodeTree) -> None:
    for link in list(node_tree.links):
        node_tree.links.remove(link)
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def new_node(
    node_tree: bpy.types.NodeTree,
    bl_idname: str,
    name: str,
    label: str,
    location: tuple[float, float],
    color: tuple[float, float, float] | None = None,
):
    node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    if color is not None:
        node.use_custom_color = True
        node.color = color
    return node


def image_node(node_tree: bpy.types.NodeTree, path: Path, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeImage", name, label, location, color=(0.12, 0.18, 0.10))
    node.image = bpy.data.images.load(str(path), check_existing=True)
    return node


def id_mask_node(node_tree: bpy.types.NodeTree, index_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeIDMask", name, label, location, color=(0.18, 0.18, 0.10))
    node.index = TREE_ID
    node.use_antialiasing = True
    ensure_link(node_tree, index_socket, node.inputs["ID value"])
    return node


def math_node(
    node_tree: bpy.types.NodeTree,
    operation: str,
    name: str,
    label: str,
    location: tuple[float, float],
    clamp: bool = True,
):
    node = new_node(node_tree, "CompositorNodeMath", name, label, location, color=(0.18, 0.16, 0.20))
    node.operation = operation
    node.use_clamp = clamp
    return node


def set_alpha_node(
    node_tree: bpy.types.NodeTree,
    image_socket,
    alpha_socket,
    name: str,
    label: str,
    location: tuple[float, float],
):
    node = new_node(node_tree, "CompositorNodeSetAlpha", name, label, location, color=(0.16, 0.20, 0.16))
    node.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    ensure_link(node_tree, alpha_socket, node.inputs["Alpha"])
    return node


def normalize_node(node_tree: bpy.types.NodeTree, value_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeNormalize", name, label, location, color=(0.16, 0.18, 0.20))
    ensure_link(node_tree, value_socket, node.inputs[0])
    return node


def rgb_to_bw_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeRGBToBW", name, label, location, color=(0.14, 0.16, 0.20))
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    return node


def filter_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float], filter_type: str):
    node = new_node(node_tree, "CompositorNodeFilter", name, label, location, color=(0.18, 0.16, 0.10))
    node.filter_type = filter_type
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    return node


def threshold_ramp(
    node_tree: bpy.types.NodeTree,
    value_socket,
    name: str,
    label: str,
    location: tuple[float, float],
    low: float,
    high: float,
):
    node = new_node(node_tree, "CompositorNodeValToRGB", name, label, location, color=(0.18, 0.14, 0.14))
    node.color_ramp.interpolation = "LINEAR"
    node.color_ramp.elements[0].position = low
    node.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    node.color_ramp.elements[1].position = high
    node.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    ensure_link(node_tree, value_socket, node.inputs["Fac"])
    return node


def rgb_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], rgba: tuple[float, float, float, float]):
    node = new_node(node_tree, "CompositorNodeRGB", name, label, location, color=(0.18, 0.12, 0.20))
    node.outputs[0].default_value = rgba
    return node


def separate_rgba_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeSepRGBA", name, label, location, color=(0.14, 0.18, 0.20))
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    return node


def output_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float], stem: str) -> Path:
    node = new_node(node_tree, "CompositorNodeOutputFile", name, label, location, color=(0.12, 0.20, 0.14))
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


def configure_scene(scene: bpy.types.Scene) -> None:
    scene.use_nodes = True
    scene.render.resolution_x = 3840
    scene.render.resolution_y = 2160
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
        scene.view_settings.look = "None"
    except Exception:
        pass


def build_prep_stage(scene: bpy.types.Scene) -> list[Path]:
    configure_scene(scene)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    pathway = image_node(node_tree, PATHWAY_EXR, "EXR Pathway", "EXR Pathway", (-1500.0, 520.0))
    priority = image_node(node_tree, PRIORITY_EXR, "EXR Priority", "EXR Priority", (-1500.0, 100.0))
    trending = image_node(node_tree, TRENDING_EXR, "EXR Trending", "EXR Trending", (-1500.0, -320.0))

    mask_visible_pathway = id_mask_node(
        node_tree,
        pathway.outputs["IndexOB"],
        "mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (-1220.0, 380.0),
    )
    mask_visible_trending = id_mask_node(
        node_tree,
        trending.outputs["IndexOB"],
        "mask_visible-arboreal_trending",
        "mask_visible-arboreal_trending",
        (-1220.0, -460.0),
    )
    mask_all_priority = id_mask_node(
        node_tree,
        priority.outputs["IndexOB"],
        "mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (-1220.0, -40.0),
    )
    mask_visible_priority = math_node(
        node_tree,
        "MULTIPLY",
        "mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (-1000.0, -40.0),
    )
    ensure_link(node_tree, mask_all_priority.outputs["Alpha"], mask_visible_priority.inputs[0])
    ensure_link(node_tree, mask_visible_pathway.outputs["Alpha"], mask_visible_priority.inputs[1])

    outputs: list[Path] = []
    layer_specs = [
        ("pathway", pathway, mask_visible_pathway.outputs["Alpha"], 520.0),
        ("priority", priority, mask_visible_priority.outputs["Value"], 100.0),
        ("trending", trending, mask_visible_trending.outputs["Alpha"], -320.0),
    ]
    for prefix, exr_node, alpha_socket, y in layer_specs:
        depth_normalized = normalize_node(
            node_tree,
            exr_node.outputs["Depth"],
            f"{prefix}_depth_normalized",
            f"{prefix}_depth_normalized",
            (-980.0, y),
        )
        depth_masked = set_alpha_node(
            node_tree,
            depth_normalized.outputs[0],
            alpha_socket,
            f"{prefix}_depth_normalized_visible_arboreal",
            f"{prefix}_depth_normalized_visible_arboreal",
            (-760.0, y),
        )
        outputs.append(
            output_node(
                node_tree,
                depth_masked.outputs["Image"],
                f"Output {prefix} Prepped Depth",
                f"Output {prefix} Prepped Depth",
                (-520.0, y),
                f"{prefix}_depth_normalized_visible_arboreal",
            )
        )

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (240.0, -80.0))
    ensure_link(node_tree, trending.outputs["Image"], composite.inputs[0])
    return outputs


def build_edge_stage(scene: bpy.types.Scene, prefix: str, prepared_png: Path) -> None:
    configure_scene(scene)
    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    prepared = image_node(
        node_tree,
        prepared_png,
        f"{prefix}_prepared_depth_png",
        f"{prefix}_prepared_depth_png",
        (-1200.0, 180.0),
    )
    prepared_bw = rgb_to_bw_node(
        node_tree,
        prepared.outputs["Image"],
        f"{prefix}_prepared_bw",
        f"{prefix}_prepared_bw",
        (-940.0, 180.0),
    )
    kirsch = filter_node(
        node_tree,
        prepared_bw.outputs["Val"],
        f"{prefix}_edge_kirsch",
        f"{prefix}_edge_kirsch",
        (-700.0, 180.0),
        "KIRSCH",
    )
    edge_normalized = normalize_node(
        node_tree,
        kirsch.outputs[0],
        f"{prefix}_edge_normalized",
        f"{prefix}_edge_normalized",
        (-460.0, 180.0),
    )
    threshold = threshold_ramp(
        node_tree,
        edge_normalized.outputs[0],
        f"{prefix}_edge_threshold",
        f"{prefix}_edge_threshold",
        (-220.0, 180.0),
        0.12,
        0.30,
    )
    threshold_bw = rgb_to_bw_node(
        node_tree,
        threshold.outputs["Image"],
        f"{prefix}_edge_threshold_bw",
        f"{prefix}_edge_threshold_bw",
        (20.0, 180.0),
    )
    prepared_rgba = separate_rgba_node(
        node_tree,
        prepared.outputs["Image"],
        f"{prefix}_prepared_rgba",
        f"{prefix}_prepared_rgba",
        (20.0, -20.0),
    )
    masked_edge_alpha = math_node(
        node_tree,
        "MULTIPLY",
        f"{prefix}_edge_masked_alpha",
        f"{prefix}_edge_masked_alpha",
        (260.0, 120.0),
    )
    ensure_link(node_tree, threshold_bw.outputs["Val"], masked_edge_alpha.inputs[0])
    ensure_link(node_tree, prepared_rgba.outputs["A"], masked_edge_alpha.inputs[1])

    edge_rgb = rgb_node(
        node_tree,
        f"{prefix}_edge_purple_rgb",
        f"{prefix}_edge_purple_rgb",
        (260.0, -40.0),
        EDGE_COLOR_LINEAR,
    )
    purple_edges = set_alpha_node(
        node_tree,
        edge_rgb.outputs[0],
        masked_edge_alpha.outputs["Value"],
        f"{prefix}_edge_purple",
        f"{prefix}_edge_purple",
        (520.0, 120.0),
    )
    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (780.0, 120.0))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (780.0, -60.0))
    ensure_link(node_tree, purple_edges.outputs["Image"], composite.inputs[0])
    ensure_link(node_tree, purple_edges.outputs["Image"], viewer.inputs[0])


def render_stage_one(scene: bpy.types.Scene) -> list[Path]:
    rendered_paths = build_prep_stage(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    log(f"Saved {BLEND_PATH}")
    bpy.ops.render.render(write_still=False)
    return [rename_output(path) for path in rendered_paths]


def render_stage_two(scene: bpy.types.Scene, prepared_pngs: list[Path]) -> list[Path]:
    edge_paths: list[Path] = []
    for prepared_png in prepared_pngs:
        prefix = prepared_png.name.replace("_depth_normalized_visible_arboreal.png", "")
        edge_path = OUTPUT_DIR / f"{prefix}_depth_kirsch_purple_edges.png"
        build_edge_stage(scene, prefix, prepared_png)
        scene.render.filepath = str(edge_path)
        bpy.ops.render.render(write_still=True)
        edge_paths.append(edge_path)
        log(f"Rendered {edge_path.name}")
    return edge_paths


def main() -> None:
    scene = bpy.context.scene
    prepared_pngs = render_stage_one(scene)
    render_stage_two(scene, prepared_pngs)


if __name__ == "__main__":
    main()
