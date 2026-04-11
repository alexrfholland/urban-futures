from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = Path(
    os.environ.get(
        "EDGE_LAB_EXR_ROOT",
        str(REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "EDGE_LAB_OUTPUT_DIR",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_v2" / "02_normals"),
    )
)
BLEND_PATH = Path(
    os.environ.get(
        "EDGE_LAB_BLEND_PATH",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_normals_lab_v2.blend"),
    )
)

PATHWAY_EXR = Path(os.environ.get("EDGE_LAB_PATHWAY_EXR", str(EXR_ROOT / "pathway_state.exr")))
PRIORITY_EXR = Path(os.environ.get("EDGE_LAB_PRIORITY_EXR", str(EXR_ROOT / "priority.exr")))
EXISTING_EXR = Path(os.environ.get("EDGE_LAB_EXISTING_EXR", str(EXR_ROOT / "existing_condition.exr")))

TREE_ID = 3


def detect_resolution_from_exr(path: Path) -> tuple[int, int]:
    try:
        info = subprocess.check_output(
            ["oiiotool", "--info", "-v", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        match = re.search(r":\s+(\d+)\s+x\s+(\d+),", info)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    image = bpy.data.images.load(str(path), check_existing=True)
    width, height = image.size[:]
    if width > 0 and height > 0:
        return int(width), int(height)
    return 3840, 2160


def log(message: str) -> None:
    print(f"[render_exr_normals_v2_blender] {message}")


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


def frame_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], color: tuple[float, float, float]):
    node = new_node(node_tree, "NodeFrame", name, label, location, color=color)
    node.label_size = 18
    node.shrink = False
    return node


def image_node(node_tree: bpy.types.NodeTree, path: Path, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeImage", name, label, location, color=(0.12, 0.18, 0.10))
    node.image = bpy.data.images.load(str(path), check_existing=True)
    return node


def id_mask_node(
    node_tree: bpy.types.NodeTree,
    index_socket,
    name: str,
    label: str,
    location: tuple[float, float],
):
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
):
    node = new_node(node_tree, "CompositorNodeMath", name, label, location, color=(0.18, 0.16, 0.20))
    node.operation = operation
    node.use_clamp = True
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


def output_node(
    node_tree: bpy.types.NodeTree,
    image_socket,
    name: str,
    label: str,
    location: tuple[float, float],
    stem: str,
):
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


def build_scene(scene: bpy.types.Scene) -> list[Path]:
    render_width, render_height = detect_resolution_from_exr(PATHWAY_EXR)
    scene.use_nodes = True
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
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
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    inputs_frame = frame_node(node_tree, "Frame Inputs", "EXR Inputs", (-1740.0, 700.0), (0.16, 0.16, 0.16))
    masks_frame = frame_node(node_tree, "Frame Masks", "Visible Arboreal Masks", (-1320.0, 520.0), (0.18, 0.18, 0.14))
    outputs_frame = frame_node(node_tree, "Frame Outputs", "Normals Outputs", (-980.0, 640.0), (0.16, 0.20, 0.16))

    pathway = image_node(node_tree, PATHWAY_EXR, "EXR Pathway", "EXR Pathway", (-1600.0, 500.0))
    pathway.parent = inputs_frame
    priority = image_node(node_tree, PRIORITY_EXR, "EXR Priority", "EXR Priority", (-1600.0, 120.0))
    priority.parent = inputs_frame
    existing = image_node(node_tree, EXISTING_EXR, "EXR Existing", "EXR Existing", (-1600.0, -260.0))
    existing.parent = inputs_frame

    mask_pathway = id_mask_node(
        node_tree,
        pathway.outputs["IndexOB"],
        "mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (-1240.0, 360.0),
    )
    mask_pathway.parent = masks_frame
    mask_all_priority = id_mask_node(
        node_tree,
        priority.outputs["IndexOB"],
        "mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (-1240.0, -20.0),
    )
    mask_all_priority.parent = masks_frame
    mask_visible_priority = math_node(
        node_tree,
        "MULTIPLY",
        "mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (-1020.0, -20.0),
    )
    mask_visible_priority.parent = masks_frame
    ensure_link(node_tree, mask_all_priority.outputs["Alpha"], mask_visible_priority.inputs[0])
    ensure_link(node_tree, mask_pathway.outputs["Alpha"], mask_visible_priority.inputs[1])

    pathway_normal = set_alpha_node(
        node_tree,
        pathway.outputs["Normal"],
        mask_pathway.outputs["Alpha"],
        "Normal Pathway Trees",
        "Normal Pathway Trees",
        (-900.0, 500.0),
    )
    pathway_normal.parent = outputs_frame
    priority_normal = set_alpha_node(
        node_tree,
        priority.outputs["Normal"],
        mask_visible_priority.outputs["Value"],
        "Normal Priority Trees",
        "Normal Priority Trees",
        (-900.0, 120.0),
    )
    priority_normal.parent = outputs_frame
    existing_normal = set_alpha_node(
        node_tree,
        existing.outputs["Normal"],
        existing.outputs["Alpha"],
        "Normal Existing Full",
        "Normal Existing Full",
        (-900.0, -260.0),
    )
    existing_normal.parent = outputs_frame

    rendered_paths = [
        output_node(
            node_tree,
            pathway_normal.outputs["Image"],
            "Output Pathway Normal",
            "Output Pathway Normal",
            (-520.0, 500.0),
            "pathway_tree_normal",
        ),
        output_node(
            node_tree,
            priority_normal.outputs["Image"],
            "Output Priority Normal",
            "Output Priority Normal",
            (-520.0, 120.0),
            "priority_tree_normal",
        ),
        output_node(
            node_tree,
            existing_normal.outputs["Image"],
            "Output Existing Normal",
            "Output Existing Normal",
            (-520.0, -260.0),
            "existing_condition_normal_full",
        ),
    ]
    for name in ("Output Pathway Normal", "Output Priority Normal", "Output Existing Normal"):
        node_tree.nodes[name].parent = outputs_frame

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (-120.0, 120.0))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (-120.0, -80.0))
    composite.parent = outputs_frame
    viewer.parent = outputs_frame
    ensure_link(node_tree, pathway_normal.outputs["Image"], composite.inputs[0])
    ensure_link(node_tree, pathway_normal.outputs["Image"], viewer.inputs[0])
    return rendered_paths


def main() -> None:
    scene = bpy.context.scene
    rendered_paths = build_scene(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    log(f"Saved {BLEND_PATH}")
    bpy.ops.render.render(write_still=False)
    for path in rendered_paths:
        rename_output(path)


if __name__ == "__main__":
    main()
