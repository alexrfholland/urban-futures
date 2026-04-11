from __future__ import annotations

import os
from pathlib import Path

import bpy

REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
COMPOSITOR_ROOT = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
CANONICAL_ROOT = COMPOSITOR_ROOT / "canonical_templates"
OUTPUT_BASE = REPO_ROOT / "_data-refactored" / "compositor" / "outputs"
DEFAULT_EXR = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "edge_detection_lab"
    / "inputs"
    / "LATEST_REMOTE_EXRS"
    / "simv3-7_20260405_8k64s_simv3-7"
    / "parade_timeline"
    / "parade_timeline__positive_state__8k64s.exr"
)

EXR_PATH = Path(
    os.environ.get(
        "COMPOSITOR_PROPOSAL_OUTLINE_EXR",
        str(DEFAULT_EXR),
    )
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get(
        "COMPOSITOR_PROPOSAL_OUTLINE_OUTPUT_DIR",
        str(OUTPUT_BASE / "proposal_outline_layers"),
    )
).expanduser()
BLEND_PATH = Path(
    os.environ.get(
        "COMPOSITOR_PROPOSAL_OUTLINE_BLEND",
        str(CANONICAL_ROOT / "proposal_outline_layers.blend"),
    )
).expanduser()

OUTLINE_RGBA = (1.0, 0.0, 0.0, 1.0)
PROPOSAL_CHANNELS = (
    ("proposal-release-control", "proposal-outline_release-control"),
    ("proposal-decay", "proposal-outline_decay"),
    ("proposal-recruit", "proposal-outline_recruit"),
    ("proposal-colonise", "proposal-outline_colonise"),
    ("proposal-deploy-structure", "proposal-outline_deploy-structure"),
)


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
):
    node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    return node


def socket_by_name(node: bpy.types.Node, preferred_name: str):
    for socket in node.outputs:
        if socket.name == preferred_name:
            return socket
    normalized = preferred_name.lower().replace(" ", "_")
    for socket in node.outputs:
        if socket.name.lower().replace(" ", "_") == normalized:
            return socket
    raise KeyError(f"Missing socket {preferred_name} on {node.name}")


def detect_resolution(image: bpy.types.Image) -> tuple[int, int]:
    width, height = image.size[:]
    if width > 0 and height > 0:
        return width, height
    return 7680, 4320


def configure_scene(scene: bpy.types.Scene, width: int, height: int) -> None:
    scene.use_nodes = True
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


def clear_output_root(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("*.png"):
        path.unlink()


def add_output_node(node_tree: bpy.types.NodeTree, rendered, output_dir: Path):
    output = new_node(
        node_tree,
        "CompositorNodeOutputFile",
        "ProposalOutlineOutput",
        "ProposalOutlineOutput",
        (1760.0, 0.0),
    )
    output.base_path = str(output_dir)
    output.format.file_format = "PNG"
    output.format.color_mode = "RGBA"
    output.format.color_depth = "8"

    while len(output.file_slots) > 1:
        output.file_slots.remove(output.file_slots[-1])
    output.file_slots[0].path = f"{rendered[0][0]}_"

    for slug, _socket in rendered[1:]:
        output.file_slots.new(slug)

    for index, (slug, socket) in enumerate(rendered):
        output.file_slots[index].path = f"{slug}_"
        ensure_link(node_tree, socket, output.inputs[index])

    return output


def build_outline_mask(node_tree, value_socket, slug: str, x: float, y: float):
    threshold = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{slug}::threshold",
        "threshold",
        (x, y),
    )
    threshold.operation = "GREATER_THAN"
    threshold.inputs[1].default_value = 1.0
    ensure_link(node_tree, value_socket, threshold.inputs[0])

    dilate = new_node(
        node_tree,
        "CompositorNodeDilateErode",
        f"{slug}::dilate",
        "dilate",
        (x + 260.0, y - 40.0),
    )
    dilate.mode = "DISTANCE"
    dilate.distance = 1
    ensure_link(node_tree, threshold.outputs["Value"], dilate.inputs["Mask"])

    erode = new_node(
        node_tree,
        "CompositorNodeDilateErode",
        f"{slug}::erode",
        "erode",
        (x + 260.0, y + 80.0),
    )
    erode.mode = "DISTANCE"
    erode.distance = -1
    ensure_link(node_tree, threshold.outputs["Value"], erode.inputs["Mask"])

    edge_band = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{slug}::edge-band",
        "edge-band",
        (x + 520.0, y),
    )
    edge_band.operation = "SUBTRACT"
    edge_band.use_clamp = True
    ensure_link(node_tree, dilate.outputs["Mask"], edge_band.inputs[0])
    ensure_link(node_tree, erode.outputs["Mask"], edge_band.inputs[1])

    edge_binary = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{slug}::edge-binary",
        "edge-binary",
        (x + 760.0, y),
    )
    edge_binary.operation = "GREATER_THAN"
    edge_binary.inputs[1].default_value = 0.0
    ensure_link(node_tree, edge_band.outputs["Value"], edge_binary.inputs[0])

    rgb = new_node(node_tree, "CompositorNodeRGB", f"{slug}::rgb", "rgb", (x + 1020.0, y))
    rgb.outputs[0].default_value = OUTLINE_RGBA

    rgba = new_node(node_tree, "CompositorNodeSetAlpha", f"{slug}::rgba", "rgba", (x + 1280.0, y))
    rgba.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, rgb.outputs[0], rgba.inputs["Image"])
    ensure_link(node_tree, edge_binary.outputs["Value"], rgba.inputs["Alpha"])
    return rgba.outputs["Image"]


def main() -> None:
    if not EXR_PATH.exists():
        raise FileNotFoundError(EXR_PATH)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "ProposalOutline"
    scene.use_nodes = True
    node_tree = scene.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    exr_node = new_node(node_tree, "CompositorNodeImage", "EXR", "EXR", (0.0, 0.0))
    exr_node.image = bpy.data.images.load(str(EXR_PATH), check_existing=True)
    width, height = detect_resolution(exr_node.image)
    configure_scene(scene, width, height)

    rendered = []
    for row, (channel_name, slug) in enumerate(PROPOSAL_CHANNELS):
        y = 360.0 - row * 300.0
        image_socket = build_outline_mask(node_tree, socket_by_name(exr_node, channel_name), slug, 240.0, y)
        rendered.append((slug, image_socket))

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (1760.0, 280.0))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (1760.0, 140.0))
    ensure_link(node_tree, rendered[0][1], composite.inputs[0])
    ensure_link(node_tree, rendered[0][1], viewer.inputs[0])

    BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_output_root(OUTPUT_DIR)
    add_output_node(node_tree, rendered, OUTPUT_DIR)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    scene.render.filepath = str(OUTPUT_DIR / "_discard_render.png")
    bpy.ops.render.render(write_still=True, scene=scene.name)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))


if __name__ == "__main__":
    main()
