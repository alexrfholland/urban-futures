"""
Build a lightweight compositor-only Blender project from the city scene.

The source scene is copied from a temp blend, the compositor tree is cloned,
helper EXR-export nodes are removed, and live Render Layers nodes are replaced
with multilayer EXR image inputs.

The output scene preserves:
- 4K render settings
- Standard/sRGB color management
- the five city view-layer names

The output project intentionally contains no model objects.
"""

from __future__ import annotations

import os
from pathlib import Path

import bpy


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def safe_name(value: str) -> str:
    return "".join(char if char not in '\\/:*?"<>|' else "_" for char in value).strip()


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DEFAULT_SOURCE_BLEND_PATH = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "tmp" / "2026 futures heroes6_compositor_export_tmp.blend"
SOURCE_BLEND_PATH = Path(env_str("B2026_SOURCE_BLEND_PATH", str(DEFAULT_SOURCE_BLEND_PATH)))
REQUESTED_SOURCE_SCENE_NAME = env_str("B2026_SOURCE_SCENE_NAME", "City")
OUTPUT_SCENE_NAME = env_str("B2026_OUTPUT_SCENE_NAME", "City")
OUTPUT_BLEND_PATH = Path(
    env_str(
        "B2026_OUTPUT_BLEND_PATH",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "city_exr_compositor_lightweight.blend"),
    )
)
TEST_RENDER_PATH = Path(
    env_str(
        "B2026_TEST_RENDER_PATH",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "city_exr_compositor_lightweight_test.png"),
    )
)
EXR_ROOT_OVERRIDE = env_str("B2026_EXR_ROOT", "")
RUN_TEST = env_flag("B2026_RUN_TEST", True)
IMAGE_NODE_PREFIX = "EXR :: "
HIDDEN_HELPER_PREFIXES = (
    "ViewLayer EXR :: ",
    "Snapshot Setup :: ",
    "Snapshot EXR :: ",
)
IMAGE_SOCKET_MAP = {
    "Image": "Combined",
    "world_design_bioenvelope_simple": "world_design_bioenvelope_simp",
}
EXPECTED_VIEW_LAYERS = (
    "pathway_state",
    "existing_condition",
    "city_priority",
    "city_bioenvelope",
    "trending_state",
)
CANONICAL_IMAGE_LAYER_NAMES = {
    "pathway_state": "pathway_state",
    "existing_condition": "existing_condition",
    "city_priority": "priority",
    "city_bioenvelope": "bioenvelope",
    "trending_state": "trending_state",
}


def log(message: str) -> None:
    print(f"[build_city_exr_compositor_lightweight] {message}")


def require_source_blend() -> Path:
    if not SOURCE_BLEND_PATH.exists():
        raise FileNotFoundError(f"Source blend not found: {SOURCE_BLEND_PATH}")
    return SOURCE_BLEND_PATH


def resolve_source_scene(requested_name: str) -> bpy.types.Scene:
    exact = bpy.data.scenes.get(requested_name)
    if exact is not None:
        return exact

    lowered = requested_name.casefold()
    for scene in bpy.data.scenes:
        if scene.name.casefold() == lowered:
            log(f"Requested source scene '{requested_name}' was not found; using case-insensitive match '{scene.name}'.")
            return scene

    alias = bpy.data.scenes.get("city")
    if alias is not None and requested_name.casefold() == "city":
        log("Requested source scene 'City' was not found; falling back to source scene 'city'.")
        return alias

    available = ", ".join(scene.name for scene in bpy.data.scenes)
    raise ValueError(f"Scene '{requested_name}' was not found. Available scenes: {available}")


def exr_root_for_source(source_scene_name: str) -> Path:
    if EXR_ROOT_OVERRIDE.strip():
        return Path(EXR_ROOT_OVERRIDE)
    logical_blend_stem = SOURCE_BLEND_PATH.stem.removesuffix("_compositor_export_tmp")
    return REPO_ROOT / "data" / "blender" / "2026" / f"{logical_blend_stem}-{safe_name(source_scene_name)}"


def candidate_exr_paths(source_scene_name: str, view_layer_name: str) -> list[Path]:
    root = exr_root_for_source(source_scene_name)
    stem = f"{safe_name(source_scene_name)}-{safe_name(view_layer_name)}"
    return [
        root / f"{stem}0000.exr",
        root / f"{stem}.exr",
    ]


def resolve_exr_path(source_scene_name: str, view_layer_name: str) -> Path:
    candidates = [path for path in candidate_exr_paths(source_scene_name, view_layer_name) if path.exists()]
    if not candidates:
        raise FileNotFoundError(
            f"No EXR found for '{source_scene_name}:{view_layer_name}'. "
            f"Tried: {', '.join(str(path) for path in candidate_exr_paths(source_scene_name, view_layer_name))}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


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
    target_scene.camera = None
    target_scene.world = None


def ensure_view_layer_contract(source_scene: bpy.types.Scene, target_scene: bpy.types.Scene) -> None:
    source_names = [view_layer.name for view_layer in source_scene.view_layers]
    if source_names != list(EXPECTED_VIEW_LAYERS):
        log(f"Source view-layer order differs from expected contract: {source_names}")

    if target_scene.view_layers:
        target_scene.view_layers[0].name = source_names[0]
        while len(target_scene.view_layers) > 1:
            target_scene.view_layers.remove(target_scene.view_layers[-1])
    else:
        target_scene.view_layers.new(name=source_names[0])

    for view_layer_name in source_names[1:]:
        target_scene.view_layers.new(name=view_layer_name)

    for source_view_layer, target_view_layer in zip(source_scene.view_layers, target_scene.view_layers):
        copy_rna_properties(
            source_view_layer,
            target_view_layer,
            {"rna_type", "layer_collection", "aovs", "lightgroups", "eevee", "cycles"},
        )
        if hasattr(source_view_layer, "cycles") and hasattr(target_view_layer, "cycles"):
            copy_rna_properties(source_view_layer.cycles, target_view_layer.cycles, {"rna_type"})
        existing_aovs = {aov.name: aov for aov in target_view_layer.aovs}
        for source_aov in source_view_layer.aovs:
            target_aov = existing_aovs.get(source_aov.name)
            if target_aov is None:
                target_aov = target_view_layer.aovs.add()
            target_aov.name = source_aov.name
            try:
                target_aov.type = source_aov.type
            except Exception:
                pass


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


def remove_helper_nodes(node_tree: bpy.types.NodeTree) -> None:
    helper_nodes = [
        node
        for node in node_tree.nodes
        if any(node.name.startswith(prefix) or node.label.startswith(prefix) for prefix in HIDDEN_HELPER_PREFIXES)
    ]
    for node in helper_nodes:
        node_tree.nodes.remove(node)


def candidate_image_socket_names(render_socket_name: str) -> list[str]:
    mapped = IMAGE_SOCKET_MAP.get(render_socket_name)
    candidates = [render_socket_name]
    if mapped:
        candidates.insert(0, mapped)
    if render_socket_name.endswith("_simple"):
        candidates.append(render_socket_name.removesuffix("_simple") + "_simp")
    if render_socket_name.endswith("_simp"):
        candidates.append(render_socket_name.removesuffix("_simp") + "_simple")
    return list(dict.fromkeys(candidates))


def resolve_image_socket(image_node: bpy.types.Node, render_socket_name: str):
    for candidate in candidate_image_socket_names(render_socket_name):
        socket = image_node.outputs.get(candidate)
        if socket is not None:
            return socket, candidate
    return None, None


def load_exr_image(exr_path: Path) -> bpy.types.Image:
    image = bpy.data.images.load(str(exr_path), check_existing=True)
    try:
        image.colorspace_settings.name = "Linear"
    except Exception:
        pass
    return image


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
        try:
            exr_path = resolve_exr_path(source_scene_name, layer_name)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue

        image = loaded_images.get(layer_name)
        if image is None:
            image = load_exr_image(exr_path)
            loaded_images[layer_name] = image

        image_node = node_tree.nodes.new("CompositorNodeImage")
        image_layer_name = CANONICAL_IMAGE_LAYER_NAMES.get(layer_name, layer_name)
        image_node.name = f"{IMAGE_NODE_PREFIX}{image_layer_name}"
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
            replacement_socket, used_name = resolve_image_socket(image_node, source_socket_name)
            if replacement_socket is None:
                missing.append(f"{layer_name}:{source_socket_name}")
                continue
            target_socket = link.to_socket
            node_tree.links.remove(link)
            node_tree.links.new(replacement_socket, target_socket)
            rewired.append(f"{layer_name}:{source_socket_name}->{used_name}")

        node_tree.nodes.remove(render_node)

    return sorted(set(rewired)), sorted(set(missing))


def collect_datablocks_for_export(scene: bpy.types.Scene) -> set:
    data_blocks = {scene}
    if scene.world is not None:
        data_blocks.add(scene.world)

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


def export_output_blend(scene: bpy.types.Scene) -> None:
    OUTPUT_BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_BLEND_PATH.exists():
        OUTPUT_BLEND_PATH.unlink()
    bpy.data.libraries.write(str(OUTPUT_BLEND_PATH), collect_datablocks_for_export(scene))


def verify_output_blend() -> None:
    bpy.ops.wm.open_mainfile(filepath=str(OUTPUT_BLEND_PATH))
    scene = bpy.data.scenes.get(OUTPUT_SCENE_NAME)
    if scene is None:
        raise RuntimeError(f"Output scene '{OUTPUT_SCENE_NAME}' was not found in {OUTPUT_BLEND_PATH}")

    render_nodes = [node for node in scene.node_tree.nodes if node.bl_idname == "CompositorNodeRLayers"]
    image_nodes = [node for node in scene.node_tree.nodes if node.bl_idname == "CompositorNodeImage" and node.name.startswith(IMAGE_NODE_PREFIX)]
    if render_nodes:
        raise RuntimeError(f"Output scene still contains Render Layers nodes: {[node.name for node in render_nodes]}")
    if len(image_nodes) < len(EXPECTED_VIEW_LAYERS):
        raise RuntimeError(f"Expected at least {len(EXPECTED_VIEW_LAYERS)} EXR image nodes, found {len(image_nodes)}")
    if scene.objects:
        raise RuntimeError(f"Output scene should be model-free, but contains objects: {[obj.name for obj in scene.objects[:10]]}")
    if [view_layer.name for view_layer in scene.view_layers] != list(EXPECTED_VIEW_LAYERS):
        raise RuntimeError(f"Unexpected output view layers: {[view_layer.name for view_layer in scene.view_layers]}")

    scene.render.filepath = str(TEST_RENDER_PATH)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    bpy.ops.render.render(write_still=True, scene=scene.name)

    if not TEST_RENDER_PATH.exists():
        raise RuntimeError(f"Test render did not write: {TEST_RENDER_PATH}")

    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND_PATH))
    log(f"Verified output scene '{scene.name}' with {len(image_nodes)} EXR image nodes and {len(scene.view_layers)} view layers.")
    log(f"Test render: {TEST_RENDER_PATH}")


def main() -> None:
    require_source_blend()
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE_BLEND_PATH))

    source_scene = resolve_source_scene(REQUESTED_SOURCE_SCENE_NAME)
    output_scene = create_blank_scene(OUTPUT_SCENE_NAME)
    copy_scene_settings(source_scene, output_scene)
    ensure_view_layer_contract(source_scene, output_scene)

    source_tree = ensure_nodes_enabled(source_scene)
    output_tree = ensure_nodes_enabled(output_scene)
    clone_compositor_tree(source_tree, output_tree)
    remove_helper_nodes(output_tree)
    rewired, missing = replace_render_layers_with_exr_images(output_tree, source_scene.name)

    if missing:
        raise RuntimeError(f"Missing EXR inputs or socket mappings: {missing}")

    output_scene["b2026_source_blend_path"] = str(SOURCE_BLEND_PATH)
    output_scene["b2026_source_scene_name"] = source_scene.name
    output_scene["b2026_exr_root"] = str(exr_root_for_source(source_scene.name))

    export_output_blend(output_scene)
    log(f"Source blend: {SOURCE_BLEND_PATH}")
    log(f"Source scene: {source_scene.name}")
    log(f"Output scene: {output_scene.name}")
    log(f"Output blend: {OUTPUT_BLEND_PATH}")
    log(f"EXR root: {exr_root_for_source(source_scene.name)}")
    log(f"Rewired sockets: {', '.join(rewired) if rewired else 'none'}")

    if RUN_TEST:
        verify_output_blend()


if __name__ == "__main__":
    main()
