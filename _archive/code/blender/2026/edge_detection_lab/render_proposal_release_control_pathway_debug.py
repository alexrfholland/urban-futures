from __future__ import annotations

import os
from pathlib import Path

import bpy


EXR_PATH = Path(
    os.environ.get(
        "EDGE_LAB_PATHWAY_EXR",
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/inputs/parade_8k_network_20260402/parade_pathway_state_8k.exr",
    )
).expanduser()
BLEND_PATH = Path(
    os.environ.get(
        "EDGE_LAB_RELEASE_CONTROL_BLEND",
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/proposal_release_control_pathway_debug.blend",
    )
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get(
        "EDGE_LAB_RELEASE_CONTROL_OUTPUT_DIR",
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/outputs/proposal_release_control_pathway_debug_20260402",
    )
).expanduser()

RELEASE_CONTROL_SPECS = (
    (1, "rejected", 0.2),
    (2, "reduce-pruning", 0.5),
    (3, "eliminate-pruning", 1.0),
)
DECAY_SPECS = (
    (2, "buffer-feature", (0.72, 0.23, 0.42, 1.0)),
    (3, "brace-feature", (0.85, 0.39, 0.55, 1.0)),
)
RECRUIT_SPECS = (
    (2, "buffer-feature", (0.77, 0.89, 0.56, 1.0)),
    (3, "rewild-ground", (0.36, 0.72, 0.34, 1.0)),
)
COLONISE_SPECS = (
    (2, "rewild-ground", (0.36, 0.72, 0.34, 1.0)),
    (3, "enrich-envelope", (0.55, 0.80, 0.31, 1.0)),
    (4, "roughen-envelope", (0.72, 0.48, 0.22, 1.0)),
)
DEPLOY_STRUCTURE_SPECS = (
    (2, "adapt-utility-pole", (1.0, 0.0, 0.0, 1.0)),
    (3, "translocated-log", (0.56, 0.54, 0.75, 1.0)),
    (4, "upgrade-feature", (0.81, 0.43, 0.85, 1.0)),
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


def build_value_match_socket(node_tree, value_socket, target_value: int, x: float, y: float, stem: str):
    compare = new_node(node_tree, "CompositorNodeMath", f"{stem}::compare", "compare", (x, y))
    compare.operation = "COMPARE"
    compare.use_clamp = True
    compare.inputs[1].default_value = float(target_value)
    compare.inputs[2].default_value = 0.0
    ensure_link(node_tree, value_socket, compare.inputs[0])
    return compare.outputs["Value"]


def build_gray_fill(node_tree, alpha_socket, gray: float, x: float, y: float, stem: str):
    rgb = new_node(node_tree, "CompositorNodeRGB", f"{stem}::rgb", "rgb", (x, y))
    rgb.outputs[0].default_value = (gray, gray, gray, 1.0)

    set_alpha = new_node(node_tree, "CompositorNodeSetAlpha", f"{stem}::rgba", "rgba", (x + 460.0, y))
    set_alpha.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, rgb.outputs[0], set_alpha.inputs["Image"])
    ensure_link(node_tree, alpha_socket, set_alpha.inputs["Alpha"])
    return set_alpha.outputs["Image"]


def build_color_fill(node_tree, alpha_socket, rgba: tuple[float, float, float, float], x: float, y: float, stem: str):
    rgb = new_node(node_tree, "CompositorNodeRGB", f"{stem}::rgb", "rgb", (x, y))
    rgb.outputs[0].default_value = rgba

    set_alpha = new_node(node_tree, "CompositorNodeSetAlpha", f"{stem}::rgba", "rgba", (x + 460.0, y))
    set_alpha.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, rgb.outputs[0], set_alpha.inputs["Image"])
    ensure_link(node_tree, alpha_socket, set_alpha.inputs["Alpha"])
    return set_alpha.outputs["Image"]


def configure_scene(scene: bpy.types.Scene, width: int, height: int) -> None:
    scene.use_nodes = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
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


def detect_resolution(image: bpy.types.Image) -> tuple[int, int]:
    width, height = image.size[:]
    if width > 0 and height > 0:
        return width, height
    return 7680, 4320


def clear_output_root(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("*.png"):
        path.unlink()


def add_file_output_node(
    node_tree: bpy.types.NodeTree,
    rendered: list[tuple[str, object]],
    output_dir: Path,
) -> bpy.types.Node:
    output = new_node(
        node_tree,
        "CompositorNodeOutputFile",
        "ReleaseControlOutput",
        "ReleaseControlOutput",
        (1760.0, 40.0),
    )
    output.base_path = str(output_dir)
    output.format.file_format = "PNG"
    output.format.color_mode = "RGBA"
    output.format.color_depth = "8"

    while len(output.file_slots) > 1:
        output.file_slots.remove(output.file_slots[-1])
    output.file_slots[0].path = f"{rendered[0][0]}_"

    for slug, socket in rendered[1:]:
        output.file_slots.new(slug)

    for index, (slug, socket) in enumerate(rendered):
        output.file_slots[index].path = f"{slug}_"
        ensure_link(node_tree, socket, output.inputs[index])

    return output


def main() -> None:
    if not EXR_PATH.exists():
        raise FileNotFoundError(EXR_PATH)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "ReleaseControlPathway"
    scene.use_nodes = True
    node_tree = scene.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    exr_node = new_node(node_tree, "CompositorNodeImage", "EXR Pathway", "EXR Pathway", (0.0, 0.0))
    exr_node.image = bpy.data.images.load(str(EXR_PATH), check_existing=True)
    release_control_socket = socket_by_name(exr_node, "proposal-release-control")
    decay_socket = socket_by_name(exr_node, "proposal-decay")
    recruit_socket = socket_by_name(exr_node, "proposal-recruit")
    colonise_socket = socket_by_name(exr_node, "proposal-colonise")
    deploy_structure_socket = socket_by_name(exr_node, "proposal-deploy-structure")

    width, height = detect_resolution(exr_node.image)
    configure_scene(scene, width, height)

    rendered = []
    for index, slug, gray in RELEASE_CONTROL_SPECS:
        y = 280.0 - (index - 1) * 260.0
        stem = f"release-control::{slug}"
        mask_socket = build_value_match_socket(node_tree, release_control_socket, index, 240.0, y, stem)
        image_socket = build_gray_fill(node_tree, mask_socket, gray, 980.0, y, stem)
        rendered.append((f"proposal-release-control-{slug}", image_socket))

    for row, (index, slug, rgba) in enumerate(DECAY_SPECS):
        y = -560.0 - row * 260.0
        stem = f"decay::{slug}"
        mask_socket = build_value_match_socket(node_tree, decay_socket, index, 240.0, y, stem)
        image_socket = build_color_fill(node_tree, mask_socket, rgba, 980.0, y, stem)
        rendered.append((f"proposal-decay-{slug}", image_socket))

    for row, (index, slug, rgba) in enumerate(RECRUIT_SPECS):
        y = -1220.0 - row * 260.0
        stem = f"recruit::{slug}"
        mask_socket = build_value_match_socket(node_tree, recruit_socket, index, 240.0, y, stem)
        image_socket = build_color_fill(node_tree, mask_socket, rgba, 980.0, y, stem)
        rendered.append((f"proposal-recruit-{slug}", image_socket))

    for row, (index, slug, rgba) in enumerate(COLONISE_SPECS):
        y = -1880.0 - row * 260.0
        stem = f"colonise::{slug}"
        mask_socket = build_value_match_socket(node_tree, colonise_socket, index, 240.0, y, stem)
        image_socket = build_color_fill(node_tree, mask_socket, rgba, 980.0, y, stem)
        rendered.append((f"proposal-colonise-{slug}", image_socket))

    for row, (index, slug, rgba) in enumerate(DEPLOY_STRUCTURE_SPECS):
        y = -2760.0 - row * 260.0
        stem = f"deploy-structure::{slug}"
        mask_socket = build_value_match_socket(node_tree, deploy_structure_socket, index, 240.0, y, stem)
        image_socket = build_color_fill(node_tree, mask_socket, rgba, 980.0, y, stem)
        rendered.append((f"proposal-deploy-structure-{slug}", image_socket))

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (1760.0, 320.0))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (1760.0, 180.0))
    ensure_link(node_tree, rendered[0][1], composite.inputs[0])
    ensure_link(node_tree, rendered[0][1], viewer.inputs[0])

    BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_output_root(OUTPUT_DIR)
    add_file_output_node(node_tree, rendered, OUTPUT_DIR)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    scene.render.filepath = str(OUTPUT_DIR / "_discard_render.png")
    bpy.ops.render.render(write_still=True, scene=scene.name)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))


if __name__ == "__main__":
    main()
