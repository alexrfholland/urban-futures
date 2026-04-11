from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"
DEFAULT_INPUT_DIR = DEFAULT_DATA_ROOT / "inputs"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "outputs"
DEFAULT_BLEND_PATH = DEFAULT_DATA_ROOT / "edge_detection_lab.blend"
DEFAULT_SCENE_NAME = "edge_detection_lab"

INPUT_SPECS = (
    {
        "key": "beauty",
        "aliases": ("beauty", "color", "albedo", "rgb"),
        "colorspace": "sRGB",
        "label": "beauty / luma source",
    },
    {
        "key": "depth",
        "aliases": ("depth", "z", "distance"),
        "colorspace": "Non-Color",
        "label": "depth source",
    },
    {
        "key": "normal",
        "aliases": ("normal", "normals"),
        "colorspace": "Non-Color",
        "label": "normal source",
    },
    {
        "key": "mask",
        "aliases": ("mask", "alpha", "matte", "silhouette"),
        "colorspace": "Non-Color",
        "label": "mask source",
    },
)

VARIANT_SPECS = (
    {
        "name": "beauty_sobel_soft",
        "source": "beauty",
        "filter_type": "SOBEL",
        "blur_pixels": 1,
        "threshold_low": 0.10,
        "threshold_high": 0.24,
        "notes": "General-purpose edges from the beauty pass with a small blur.",
    },
    {
        "name": "depth_laplace_sparse",
        "source": "depth",
        "filter_type": "LAPLACE",
        "blur_pixels": 0,
        "threshold_low": 0.06,
        "threshold_high": 0.18,
        "notes": "Crisp structural lines from depth with minimal prefiltering.",
    },
    {
        "name": "normal_kirsch_fine",
        "source": "normal",
        "filter_type": "KIRSCH",
        "blur_pixels": 0,
        "threshold_low": 0.12,
        "threshold_high": 0.30,
        "notes": "Directional emphasis for fine façade and roof edges.",
    },
    {
        "name": "mask_silhouette_band",
        "source": "mask",
        "filter_type": "SOBEL",
        "blur_pixels": 0,
        "threshold_low": 0.18,
        "threshold_high": 0.32,
        "notes": "Silhouette band from a mask or alpha-style pass.",
        "silhouette": True,
    },
    {
        "name": "hybrid_max_consensus",
        "source": "beauty",
        "filter_type": "PREWITT",
        "blur_pixels": 1,
        "threshold_low": 0.14,
        "threshold_high": 0.26,
        "notes": "Max-combined consensus edge from beauty, depth, and normal inputs.",
        "hybrid": True,
    },
)


def log(message: str) -> None:
    print(f"[edge_detection_lab] {message}")


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


def cli_path(flag: str, default: Path) -> Path:
    argv = list(sys.argv)
    if "--" not in argv:
        return default
    extra = argv[argv.index("--") + 1 :]
    for index, token in enumerate(extra):
        if token == flag and index + 1 < len(extra):
            return Path(extra[index + 1])
    return default


def resolve_path(flag: str, env_name: str, default: Path) -> Path:
    return cli_path(flag, env_path(env_name, default))


def ensure_scene(name: str) -> bpy.types.Scene:
    scene = bpy.data.scenes.get(name)
    if scene is None:
        scene = bpy.data.scenes.new(name)
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    scene.use_nodes = True
    return scene


def clear_node_tree(node_tree: bpy.types.NodeTree) -> None:
    for link in list(node_tree.links):
        node_tree.links.remove(link)
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def ensure_node(node_tree: bpy.types.NodeTree, bl_idname: str, name: str, location: tuple[float, float]):
    node = node_tree.nodes.get(name)
    if node is None or node.bl_idname != bl_idname:
        if node is not None:
            node_tree.nodes.remove(node)
        node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = name
    node.location = location
    return node


def set_colorspace(image: bpy.types.Image, colorspace: str) -> None:
    try:
        image.colorspace_settings.name = colorspace
    except Exception:
        pass


def find_input_image(input_dir: Path, spec: dict) -> Path | None:
    exact = input_dir / f"{spec['key']}.png"
    if exact.exists():
        return exact

    candidates = sorted(input_dir.glob("*.png"))
    for candidate in candidates:
        stem = candidate.stem.lower()
        if any(alias in stem for alias in spec["aliases"]):
            return candidate
    return None


def build_input_nodes(node_tree: bpy.types.NodeTree, input_dir: Path) -> dict[str, bpy.types.Node]:
    input_nodes = {}
    x = -2200.0
    for index, spec in enumerate(INPUT_SPECS):
        node = ensure_node(node_tree, "CompositorNodeImage", f"Input::{spec['key']}", (x, 120.0 - index * 260.0))
        node.label = spec["label"]
        image_path = find_input_image(input_dir, spec)
        if image_path is not None:
            image = bpy.data.images.load(str(image_path), check_existing=True)
            set_colorspace(image, spec["colorspace"])
            node.image = image
            log(f"Loaded {spec['key']} from {image_path}")
        else:
            node.image = None
            log(f"No {spec['key']} image found under {input_dir}; leaving input node empty")
        input_nodes[spec["key"]] = node
    return input_nodes


def threshold_node(node_tree: bpy.types.NodeTree, name: str, location: tuple[float, float], low: float, high: float):
    ramp = ensure_node(node_tree, "CompositorNodeValToRGB", name, location)
    ramp.color_ramp.interpolation = "LINEAR"
    ramp.color_ramp.elements[0].position = low
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = high
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    return ramp


def build_filtered_branch(
    node_tree: bpy.types.NodeTree,
    source_socket,
    base_name: str,
    location_y: float,
    filter_type: str,
    blur_pixels: int,
    threshold_low: float,
    threshold_high: float,
):
    start_x = -1200.0
    bw = ensure_node(node_tree, "CompositorNodeRGBToBW", f"{base_name}::BW", (start_x, location_y))
    blur = None
    if blur_pixels > 0:
        blur = ensure_node(node_tree, "CompositorNodeBlur", f"{base_name}::Blur", (start_x + 260.0, location_y))
        blur.filter_type = "GAUSS"
        blur.use_relative = False
        blur.size_x = blur_pixels
        blur.size_y = blur_pixels

    edge = ensure_node(node_tree, "CompositorNodeFilter", f"{base_name}::Filter", (start_x + 520.0, location_y))
    edge.filter_type = filter_type

    normalize = ensure_node(node_tree, "CompositorNodeNormalize", f"{base_name}::Normalize", (start_x + 780.0, location_y))
    ramp = threshold_node(node_tree, f"{base_name}::Threshold", (start_x + 1040.0, location_y), threshold_low, threshold_high)

    ensure_link(node_tree, source_socket, bw.inputs[0])
    previous = bw.outputs[0]
    if blur is not None:
        ensure_link(node_tree, previous, blur.inputs[0])
        previous = blur.outputs[0]
    ensure_link(node_tree, previous, edge.inputs[0])
    ensure_link(node_tree, edge.outputs[0], normalize.inputs[0])
    ensure_link(node_tree, normalize.outputs[0], ramp.inputs[0])
    return ramp.outputs[0]


def build_silhouette_branch(
    node_tree: bpy.types.NodeTree,
    source_socket,
    base_name: str,
    location_y: float,
    threshold_low: float,
    threshold_high: float,
):
    grow = ensure_node(node_tree, "CompositorNodeDilateErode", f"{base_name}::Grow", (-1200.0, location_y))
    grow.mode = "DISTANCE"
    grow.distance = 5
    shrink = ensure_node(node_tree, "CompositorNodeDilateErode", f"{base_name}::Shrink", (-940.0, location_y))
    shrink.mode = "DISTANCE"
    shrink.distance = -5
    subtract = ensure_node(node_tree, "CompositorNodeMath", f"{base_name}::Band", (-680.0, location_y))
    subtract.operation = "SUBTRACT"
    normalize = ensure_node(node_tree, "CompositorNodeNormalize", f"{base_name}::Normalize", (-420.0, location_y))
    ramp = threshold_node(node_tree, f"{base_name}::Threshold", (-160.0, location_y), threshold_low, threshold_high)

    ensure_link(node_tree, source_socket, grow.inputs[0])
    ensure_link(node_tree, source_socket, shrink.inputs[0])
    ensure_link(node_tree, grow.outputs[0], subtract.inputs[0])
    ensure_link(node_tree, shrink.outputs[0], subtract.inputs[1])
    ensure_link(node_tree, subtract.outputs[0], normalize.inputs[0])
    ensure_link(node_tree, normalize.outputs[0], ramp.inputs[0])
    return ramp.outputs[0]


def build_hybrid_branch(
    node_tree: bpy.types.NodeTree,
    source_nodes: dict[str, bpy.types.Node],
    base_name: str,
    location_y: float,
    filter_type: str,
    blur_pixels: int,
    threshold_low: float,
    threshold_high: float,
):
    beauty = build_filtered_branch(
        node_tree,
        source_nodes["beauty"].outputs[0],
        f"{base_name}::Beauty",
        location_y + 180.0,
        filter_type,
        blur_pixels,
        threshold_low,
        threshold_high,
    )
    depth = build_filtered_branch(
        node_tree,
        source_nodes["depth"].outputs[0],
        f"{base_name}::Depth",
        location_y,
        "LAPLACE",
        0,
        threshold_low,
        threshold_high,
    )
    normal = build_filtered_branch(
        node_tree,
        source_nodes["normal"].outputs[0],
        f"{base_name}::Normal",
        location_y - 180.0,
        "KIRSCH",
        0,
        threshold_low,
        threshold_high,
    )

    max_a = ensure_node(node_tree, "CompositorNodeMath", f"{base_name}::MaxA", (320.0, location_y + 120.0))
    max_a.operation = "MAXIMUM"
    max_b = ensure_node(node_tree, "CompositorNodeMath", f"{base_name}::MaxB", (580.0, location_y + 120.0))
    max_b.operation = "MAXIMUM"
    normalize = ensure_node(node_tree, "CompositorNodeNormalize", f"{base_name}::Normalize", (840.0, location_y + 120.0))
    ramp = threshold_node(node_tree, f"{base_name}::Threshold", (1100.0, location_y + 120.0), threshold_low, threshold_high)

    ensure_link(node_tree, beauty, max_a.inputs[0])
    ensure_link(node_tree, depth, max_a.inputs[1])
    ensure_link(node_tree, max_a.outputs[0], max_b.inputs[0])
    ensure_link(node_tree, normal, max_b.inputs[1])
    ensure_link(node_tree, max_b.outputs[0], normalize.inputs[0])
    ensure_link(node_tree, normalize.outputs[0], ramp.inputs[0])
    return ramp.outputs[0]


def build_output(node_tree: bpy.types.NodeTree, source_socket, variant_name: str, output_dir: Path, location_y: float):
    node = ensure_node(node_tree, "CompositorNodeOutputFile", f"Output::{variant_name}", (1500.0, location_y))
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    node.base_path = str(variant_dir)
    node.format.file_format = "PNG"
    node.format.color_mode = "RGBA"
    node.format.color_depth = "8"
    node.file_slots[0].path = f"{variant_name}_"
    ensure_link(node_tree, source_socket, node.inputs[0])
    return node


def build_lab_scene(scene: bpy.types.Scene, input_dir: Path, output_dir: Path) -> None:
    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    scene.render.resolution_x = 3840
    scene.render.resolution_y = 2160
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.use_compositing = True

    input_nodes = build_input_nodes(node_tree, input_dir)

    variant_outputs: dict[str, bpy.types.NodeSocket] = {}
    for index, spec in enumerate(VARIANT_SPECS):
        location_y = 360.0 - index * 260.0
        if spec.get("silhouette"):
            source_socket = build_silhouette_branch(
                node_tree,
                input_nodes[spec["source"]].outputs[0],
                f"Variant::{spec['name']}",
                location_y,
                spec["threshold_low"],
                spec["threshold_high"],
            )
        elif spec.get("hybrid"):
            source_socket = build_hybrid_branch(
                node_tree,
                input_nodes,
                f"Variant::{spec['name']}",
                location_y,
                spec["filter_type"],
                spec["blur_pixels"],
                spec["threshold_low"],
                spec["threshold_high"],
            )
        else:
            source_socket = build_filtered_branch(
                node_tree,
                input_nodes[spec["source"]].outputs[0],
                f"Variant::{spec['name']}",
                location_y,
                spec["filter_type"],
                spec["blur_pixels"],
                spec["threshold_low"],
                spec["threshold_high"],
            )
        variant_outputs[spec["name"]] = source_socket
        build_output(node_tree, source_socket, spec["name"], output_dir, location_y)
        log(f"Configured variant: {spec['name']} ({spec['notes']})")

    composite = ensure_node(node_tree, "CompositorNodeComposite", "Composite", (1860.0, 140.0))
    viewer = ensure_node(node_tree, "CompositorNodeViewer", "Viewer", (1860.0, -140.0))
    ensure_link(node_tree, variant_outputs["hybrid_max_consensus"], composite.inputs[0])
    ensure_link(node_tree, variant_outputs["hybrid_max_consensus"], viewer.inputs[0])


def save_blend(blend_path: Path) -> None:
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    log(f"Saved scaffold blend: {blend_path}")


def main() -> None:
    input_dir = resolve_path("--inputs", "B2026_EDGE_LAB_INPUT_DIR", DEFAULT_INPUT_DIR)
    output_dir = resolve_path("--outputs", "B2026_EDGE_LAB_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    blend_path = resolve_path("--blend", "B2026_EDGE_LAB_BLEND_PATH", DEFAULT_BLEND_PATH)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = ensure_scene(os.environ.get("B2026_EDGE_LAB_SCENE_NAME", DEFAULT_SCENE_NAME))
    build_lab_scene(scene, input_dir, output_dir)
    save_blend(blend_path)

    if os.environ.get("B2026_EDGE_LAB_RENDER", "").strip() == "1":
        log("Rendering is enabled, but this scaffold does not auto-render until inputs are ready.")


if __name__ == "__main__":
    main()
