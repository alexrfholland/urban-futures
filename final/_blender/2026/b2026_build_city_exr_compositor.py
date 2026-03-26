"""
Build a compositor-only Blender project for the city scene using EXR image
inputs instead of live Render Layers nodes.

Expected EXR layout:
  {source-blend-stem}-{scene}/{scene}-{viewlayer}.exr

This script does not generate EXRs. Run it after the per-view-layer EXRs exist.
"""

import bpy
import os
from pathlib import Path


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def safe_name(value: str) -> str:
    return "".join(char if char not in '\\/:*?"<>|' else "_" for char in value).strip()


SOURCE_BLEND_PATH = Path(
    env_str(
        "B2026_SOURCE_BLEND_PATH",
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6.blend",
    )
)
SOURCE_SCENE_NAME = env_str("B2026_SOURCE_SCENE_NAME", "city")
OUTPUT_SCENE_NAME = env_str("B2026_OUTPUT_SCENE_NAME", "city_exr_compositor")
OUTPUT_BLEND_PATH = Path(
    env_str(
        "B2026_OUTPUT_BLEND_PATH",
        f"/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/{safe_name(SOURCE_BLEND_PATH.stem)}-{safe_name(OUTPUT_SCENE_NAME)}.blend",
    )
)
EXR_ROOT_OVERRIDE = env_str("B2026_EXR_ROOT", "")
INPUT_NODE_PREFIX = "City EXR :: "
HIDDEN_HELPER_PREFIXES = (
    "ViewLayer EXR :: ",
    "Snapshot Setup :: ",
    "Snapshot EXR :: ",
)
IMAGE_SOCKET_MAP = {
    "Image": "Combined",
}


def log(message: str) -> None:
    print(f"[city_exr_compositor] {message}")


def require_source_blend() -> Path:
    if not SOURCE_BLEND_PATH.exists():
        raise FileNotFoundError(f"Source blend not found: {SOURCE_BLEND_PATH}")
    return SOURCE_BLEND_PATH


def require_scene(scene_name: str) -> bpy.types.Scene:
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' was not found.")
    return scene


def exr_root_for_source(scene_name: str) -> Path:
    if EXR_ROOT_OVERRIDE.strip():
        return Path(EXR_ROOT_OVERRIDE)
    return SOURCE_BLEND_PATH.parent / f"{SOURCE_BLEND_PATH.stem}-{safe_name(scene_name)}"


def exr_path_for_layer(scene_name: str, view_layer_name: str) -> Path:
    return exr_root_for_source(scene_name) / f"{safe_name(scene_name)}-{safe_name(view_layer_name)}.exr"


def ensure_nodes_enabled(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    scene.use_nodes = True
    if scene.node_tree is None:
        raise ValueError(f"Scene '{scene.name}' does not have a compositor node tree.")
    return scene.node_tree


def copy_rna_properties(source, target, excluded: set[str]) -> None:
    for prop in source.bl_rna.properties:
        identifier = prop.identifier
        if identifier in excluded or prop.is_readonly or prop.type == "COLLECTION":
            continue
        try:
            setattr(target, identifier, getattr(source, identifier))
        except Exception:
            continue


def copy_socket_defaults(source_node: bpy.types.Node, target_node: bpy.types.Node) -> None:
    for source_socket, target_socket in zip(source_node.inputs, target_node.inputs):
        if hasattr(source_socket, "default_value"):
            try:
                target_socket.default_value = source_socket.default_value
            except Exception:
                pass


def copy_scene_settings(source_scene: bpy.types.Scene, target_scene: bpy.types.Scene) -> None:
    copy_rna_properties(source_scene.render, target_scene.render, {"rna_type"})
    copy_rna_properties(source_scene.view_settings, target_scene.view_settings, {"rna_type"})
    copy_rna_properties(source_scene.display_settings, target_scene.display_settings, {"rna_type"})
    copy_rna_properties(
        source_scene.sequencer_colorspace_settings,
        target_scene.sequencer_colorspace_settings,
        {"rna_type"},
    )
    if hasattr(source_scene, "cycles") and hasattr(target_scene, "cycles"):
        copy_rna_properties(source_scene.cycles, target_scene.cycles, {"rna_type"})
    target_scene.render.engine = source_scene.render.engine
    target_scene.render.film_transparent = source_scene.render.film_transparent
    target_scene.render.resolution_x = source_scene.render.resolution_x
    target_scene.render.resolution_y = source_scene.render.resolution_y
    target_scene.render.resolution_percentage = source_scene.render.resolution_percentage
    target_scene.frame_start = source_scene.frame_start
    target_scene.frame_end = source_scene.frame_end
    target_scene.frame_current = source_scene.frame_current
    if source_scene.camera is not None:
        target_scene.camera = source_scene.camera


def clone_compositor_tree(source_tree: bpy.types.NodeTree, target_tree: bpy.types.NodeTree) -> None:
    for node in list(target_tree.nodes):
        target_tree.nodes.remove(node)

    node_map: dict[bpy.types.Node, bpy.types.Node] = {}

    for source_node in source_tree.nodes:
        target_node = target_tree.nodes.new(source_node.bl_idname)
        node_map[source_node] = target_node

    for source_node, target_node in node_map.items():
        copy_rna_properties(
            source_node,
            target_node,
            {
                "rna_type",
                "dimensions",
                "location_absolute",
                "select",
                "show_options",
                "show_preview",
                "show_texture",
                "show_extra",
                "inputs",
                "outputs",
                "internal_links",
                "interface",
                "warning_propagation",
            },
        )
        target_node.name = source_node.name
        target_node.label = source_node.label
        target_node.location = source_node.location.copy()
        target_node.width = source_node.width
        target_node.height = source_node.height
        target_node.hide = source_node.hide
        target_node.mute = source_node.mute
        target_node.use_custom_color = source_node.use_custom_color
        if source_node.use_custom_color:
            target_node.color = source_node.color
        copy_socket_defaults(source_node, target_node)

    for source_node, target_node in node_map.items():
        if source_node.parent and source_node.parent in node_map:
            target_node.parent = node_map[source_node.parent]

    for source_link in source_tree.links:
        from_index = next(
            (index for index, socket in enumerate(source_link.from_node.outputs) if socket == source_link.from_socket),
            None,
        )
        to_index = next(
            (index for index, socket in enumerate(source_link.to_node.inputs) if socket == source_link.to_socket),
            None,
        )
        if from_index is None or to_index is None:
            continue
        try:
            target_tree.links.new(
                node_map[source_link.from_node].outputs[from_index],
                node_map[source_link.to_node].inputs[to_index],
            )
        except Exception:
            continue


def image_socket_name(render_socket_name: str) -> str:
    return IMAGE_SOCKET_MAP.get(render_socket_name, render_socket_name)


def remove_helper_nodes(node_tree: bpy.types.NodeTree) -> None:
    helper_nodes = [
        node
        for node in node_tree.nodes
        if any(node.name.startswith(prefix) or node.label.startswith(prefix) for prefix in HIDDEN_HELPER_PREFIXES)
    ]
    for node in helper_nodes:
        node_tree.nodes.remove(node)


def collect_datablocks_for_export(scene: bpy.types.Scene) -> set:
    data_blocks = {scene}
    visited_trees = set()

    def visit_tree(node_tree: bpy.types.NodeTree) -> None:
        if node_tree in visited_trees:
            return
        visited_trees.add(node_tree)
        data_blocks.add(node_tree)
        for node in node_tree.nodes:
            node_group = getattr(node, "node_tree", None)
            if node_group is not None:
                visit_tree(node_group)
            image = getattr(node, "image", None)
            if image is not None:
                data_blocks.add(image)

    visit_tree(scene.node_tree)
    return data_blocks


def create_blank_scene(name: str) -> bpy.types.Scene:
    scene = bpy.data.scenes.new(name)
    scene.use_nodes = True
    return scene


def load_exr_image(exr_path: Path) -> bpy.types.Image:
    if not exr_path.exists():
        raise FileNotFoundError(
            f"Missing EXR input: {exr_path}. "
            "Run the EXR export step first, then rerun this compositor build."
        )
    return bpy.data.images.load(str(exr_path), check_existing=True)


def replace_render_layers_with_exr_images(
    node_tree: bpy.types.NodeTree,
    source_scene_name: str,
) -> tuple[list[str], list[str]]:
    loaded_images: dict[str, bpy.types.Image] = {}
    rewired: list[str] = []
    missing: list[str] = []
    render_nodes = [node for node in node_tree.nodes if node.bl_idname == "CompositorNodeRLayers"]

    for render_node in render_nodes:
        layer_name = render_node.layer or source_scene_name
        exr_path = exr_path_for_layer(source_scene_name, layer_name)
        image = loaded_images.get(layer_name)
        if image is None:
            try:
                image = load_exr_image(exr_path)
            except FileNotFoundError:
                missing.append(layer_name)
                continue
            loaded_images[layer_name] = image

        image_node = node_tree.nodes.new("CompositorNodeImage")
        image_node.name = f"{INPUT_NODE_PREFIX}{layer_name}"
        image_node.label = image_node.name
        image_node.location = render_node.location.copy()
        image_node.width = render_node.width
        image_node.height = render_node.height
        image_node.hide = render_node.hide
        image_node.mute = render_node.mute
        image_node.use_custom_color = True
        image_node.color = (0.12, 0.18, 0.10)
        image_node.image = image
        if render_node.parent is not None:
            image_node.parent = render_node.parent

        outgoing_links = [link for link in node_tree.links if link.from_node == render_node]
        for link in outgoing_links:
            source_socket_name = link.from_socket.name
            replacement_name = image_socket_name(source_socket_name)
            replacement_socket = image_node.outputs.get(replacement_name)
            if replacement_socket is None:
                missing.append(f"{layer_name}:{source_socket_name}")
                continue
            target_socket = link.to_socket
            node_tree.links.remove(link)
            node_tree.links.new(replacement_socket, target_socket)
            rewired.append(f"{layer_name}:{source_socket_name}")

        node_tree.nodes.remove(render_node)

    return sorted(set(rewired)), sorted(set(missing))


def export_output_blend(scene: bpy.types.Scene) -> None:
    OUTPUT_BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_BLEND_PATH.exists():
        OUTPUT_BLEND_PATH.unlink()
    bpy.data.libraries.write(str(OUTPUT_BLEND_PATH), collect_datablocks_for_export(scene))


def main() -> None:
    require_source_blend()
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE_BLEND_PATH))

    source_scene = require_scene(SOURCE_SCENE_NAME)
    output_scene = create_blank_scene(OUTPUT_SCENE_NAME)
    copy_scene_settings(source_scene, output_scene)

    clone_compositor_tree(source_scene.node_tree, output_scene.node_tree)
    remove_helper_nodes(output_scene.node_tree)
    rewired, missing = replace_render_layers_with_exr_images(output_scene.node_tree, source_scene.name)

    export_output_blend(output_scene)

    output_scene["b2026_exr_root"] = str(exr_root_for_source(source_scene.name))
    log(f"Source blend: {SOURCE_BLEND_PATH}")
    log(f"Source scene: {source_scene.name}")
    log(f"Output scene: {output_scene.name}")
    log(f"Output blend: {OUTPUT_BLEND_PATH}")
    log(f"EXR root: {exr_root_for_source(source_scene.name)}")
    log(f"Rewired sockets: {', '.join(rewired) if rewired else 'none'}")
    if missing:
        log(f"Missing EXR inputs or sockets: {', '.join(missing)}")


if __name__ == "__main__":
    main()
