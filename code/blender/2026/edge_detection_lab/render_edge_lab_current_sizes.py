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
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/outputs/edge_lab_final_template_sizes",
)
PATHWAY_EXR = env_path(
    "EDGE_LAB_PATHWAY_EXR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6-city/city-pathway_state.exr",
)
PRIORITY_EXR = env_path(
    "EDGE_LAB_PRIORITY_EXR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6-city/city-city_priority.exr",
)
EXISTING_EXR = env_path(
    "EDGE_LAB_EXISTING_EXR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6-city/city-existing_condition.exr",
)
TRENDING_EXR = env_path(
    "EDGE_LAB_TRENDING_EXR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6-city/city-trending_state.exr",
)
SCENE_NAME = os.environ.get("EDGE_LAB_SCENE_NAME", "Current")
OUTPUT_FILTER = {
    item.strip()
    for item in os.environ.get("EDGE_LAB_OUTPUT_FILTER", "").split(",")
    if item.strip()
}


def log(message: str) -> None:
    print(f"[render_edge_lab_current_sizes] {message}")


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


def repath_exr_node(node: bpy.types.Node, filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    if node.image is None:
        raise ValueError(f"Node '{node.name}' has no image")
    node.image.filepath = str(filepath)
    node.image.reload()


def find_output_socket(node: bpy.types.Node, name: str):
    for socket in node.outputs:
        if socket.name == name:
            return socket
    return None


def preferred_image_socket(node: bpy.types.Node):
    image = find_output_socket(node, "Image")
    if image is not None and getattr(image, "enabled", True):
        return image
    combined = find_output_socket(node, "Combined")
    if combined is not None and getattr(combined, "enabled", True):
        return combined
    if image is not None:
        return image
    return None


def reconnect_image_links(node_tree: bpy.types.NodeTree, node: bpy.types.Node) -> None:
    image = find_output_socket(node, "Image")
    preferred = preferred_image_socket(node)
    if image is None or preferred is None or image == preferred:
        return
    targets = [link.to_socket for link in list(image.links)]
    for link in list(image.links):
        node_tree.links.remove(link)
    for target in targets:
        node_tree.links.new(preferred, target)


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


def configure_scene(scene: bpy.types.Scene, exr_paths: list[Path], images: list[bpy.types.Image]) -> None:
    width, height = detect_resolution(exr_paths, images)
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
    set_standard_view(scene)


def mute_all_output_nodes(node_tree: bpy.types.NodeTree) -> None:
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile":
            node.mute = True


def configure_output_nodes(output_nodes: list[bpy.types.Node], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for node in output_nodes:
        node.base_path = str(output_dir)
        node.format.file_format = "PNG"
        node.format.color_mode = "RGBA"
        node.format.color_depth = "8"
        node.mute = False


def rename_family_outputs(output_dir: Path) -> None:
    for rendered_path in sorted(output_dir.glob("*_0001.png")):
        final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")


def keep_only_filtered_outputs(output_dir: Path, allowed_stems: set[str]) -> None:
    for path in output_dir.glob("*.png"):
        if path.stem not in allowed_stems:
            path.unlink()
            log(f"Removed filtered-out output {path.name}")


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


def output_nodes(node_tree: bpy.types.NodeTree) -> list[bpy.types.Node]:
    nodes = [
        node
        for node in node_tree.nodes
        if node.bl_idname == "CompositorNodeOutputFile" and node.name.startswith("Sizes::")
    ]
    if not nodes:
        raise ValueError("No Sizes output nodes found")
    return sorted(nodes, key=lambda node: node.name)


def output_stem(node: bpy.types.Node) -> str:
    if not node.file_slots:
        raise ValueError(f"Output node '{node.name}' has no file slot")
    return node.file_slots[0].path.rstrip("_")


def linked_source_socket(node: bpy.types.Node):
    if not node.inputs[0].links:
        raise ValueError(f"Output node '{node.name}' is not linked")
    return node.inputs[0].links[0].from_socket


def selected_slots(node: bpy.types.Node) -> list[tuple[int, str]]:
    slots = [(index, slot.path.rstrip("_")) for index, slot in enumerate(node.file_slots)]
    if not OUTPUT_FILTER:
        return slots
    selected = [(index, stem) for index, stem in slots if stem in OUTPUT_FILTER]
    if not selected:
        raise ValueError(f"No Sizes outputs matched EDGE_LAB_OUTPUT_FILTER={sorted(OUTPUT_FILTER)}")
    return selected


def linked_source_socket_for_slot(node: bpy.types.Node, index: int):
    input_socket = node.inputs[index]
    if not input_socket.links:
        raise ValueError(f"Output node '{node.name}' slot {index} is not linked")
    return input_socket.links[0].from_socket


def filtered_output_nodes(nodes: list[bpy.types.Node]) -> list[bpy.types.Node]:
    if not OUTPUT_FILTER:
        return nodes
    filtered = [node for node in nodes if output_stem(node) in OUTPUT_FILTER]
    if filtered:
        return filtered
    if len(nodes) == 1:
        slot_stems = {slot.path.rstrip("_") for slot in nodes[0].file_slots}
        matched = slot_stems & OUTPUT_FILTER
        if matched:
            return nodes
    raise ValueError(f"No Sizes outputs matched EDGE_LAB_OUTPUT_FILTER={sorted(OUTPUT_FILTER)}")


def allowed_output_stems(nodes: list[bpy.types.Node]) -> set[str]:
    stems: set[str] = set()
    for node in nodes:
        for slot in node.file_slots:
            stems.add(slot.path.rstrip("_"))
    return stems


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(BLEND_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {BLEND_PATH}")

    node_tree = scene.node_tree
    existing = require_any_node(node_tree, ["Sizes::EXR Existing", "AO::EXR Existing"])
    pathway = require_any_node(node_tree, ["Sizes::EXR Pathway", "AO::EXR Pathway"])
    priority = require_any_node(node_tree, ["Sizes::EXR Priority", "AO::EXR Priority"])
    trending = require_any_node(node_tree, ["Sizes::EXR Trending", "Resources::EXR Trending"])

    repath_exr_node(existing, EXISTING_EXR)
    repath_exr_node(pathway, PATHWAY_EXR)
    repath_exr_node(priority, PRIORITY_EXR)
    repath_exr_node(trending, TRENDING_EXR)
    reconnect_image_links(node_tree, existing)
    reconnect_image_links(node_tree, pathway)
    reconnect_image_links(node_tree, priority)
    reconnect_image_links(node_tree, trending)

    configure_scene(
        scene,
        [EXISTING_EXR, PATHWAY_EXR, PRIORITY_EXR, TRENDING_EXR],
        [node.image for node in (existing, pathway, priority, trending) if node.image is not None],
    )
    mute_all_output_nodes(node_tree)

    all_output_nodes = output_nodes(node_tree)
    selected_output_nodes = filtered_output_nodes(all_output_nodes)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    if len(all_output_nodes) == 1:
        # Sizes file outputs do not render reliably as compositor sinks, even pre-consolidation.
        # Keep the consolidated node for layout/organization, but render each linked slot directly.
        node = selected_output_nodes[0]
        for index, stem in selected_slots(node):
            render_socket_to_png(
                scene,
                node_tree,
                linked_source_socket_for_slot(node, index),
                OUTPUT_DIR / f"{stem}.png",
            )
    else:
        configure_output_nodes(selected_output_nodes, OUTPUT_DIR)
        bpy.ops.render.render(write_still=False, scene=scene.name)
        rename_family_outputs(OUTPUT_DIR)


if __name__ == "__main__":
    main()
