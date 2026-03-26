import bpy
import os
from datetime import datetime
from pathlib import Path


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


TARGET_SCENE_NAME = env_str("B2026_TARGET_SCENE_NAME", "city")
PATHWAY_VIEW_LAYER_NAME = env_str("B2026_PATHWAY_VIEW_LAYER_NAME", "pathway_state")
EXISTING_VIEW_LAYER_NAME = env_str("B2026_EXISTING_VIEW_LAYER_NAME", "existing_condition")
SNAPSHOT_SCENE_NAME = env_str("B2026_SNAPSHOT_SCENE_NAME", f"{TARGET_SCENE_NAME} Snapshot")
SNAPSHOT_FOLDER_SUFFIX = "_snapshots"
SNAPSHOT_IMAGE_NODE_PREFIX = "Snapshot EXR :: "
PREPARE_SCENE_OUTPUTS = env_bool("B2026_PREPARE_SCENE_OUTPUTS", True)
EXPORT_SNAPSHOT_BLEND = env_bool("B2026_EXPORT_SNAPSHOT_BLEND", True)
SAVE_SOURCE_BLEND = env_bool("B2026_SAVE_SOURCE_BLEND", True)
KEEP_SNAPSHOT_SCENE_IN_SOURCE_BLEND = env_bool("B2026_KEEP_SNAPSHOT_SCENE_IN_SOURCE_BLEND", False)
REPLACE_EXISTING_SNAPSHOT_SCENE = env_bool("B2026_REPLACE_EXISTING_SNAPSHOT_SCENE", True)

IMAGE_SOCKET_MAP = {
    "Image": "Combined",
}

HELPER_FRAME_NAME = "Snapshot Setup :: EXR Outputs"
HELPER_RENDER_NODE_PREFIX = "Snapshot Setup :: Render Layers :: "
HELPER_OUTPUT_NODE_PREFIX = "Snapshot Setup :: File Output :: "
NODE_PROP_EXCLUDE = {
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
}
SOCKET_PROP_EXCLUDE = {
    "rna_type",
    "bl_idname",
    "bl_label",
    "bl_subtype_label",
    "is_linked",
    "is_unavailable",
    "is_multi_input",
    "is_icon_visible",
    "is_inactive",
    "is_panel_toggle",
    "identifier",
    "link_limit",
    "links",
    "name",
    "node",
    "type",
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[snapshot {timestamp}] {message}")


def require_saved_blend() -> Path:
    if not bpy.data.filepath:
        raise ValueError("Save the Blender file before running the snapshot script.")
    return Path(bpy.data.filepath)


def require_scene(scene_name: str) -> bpy.types.Scene:
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' was not found.")
    return scene


def snapshot_root_folder() -> Path:
    blend_path = require_saved_blend()
    folder = blend_path.parent / f"{blend_path.stem}{SNAPSHOT_FOLDER_SUFFIX}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def scene_exr_prefix(scene_name: str, view_layer_name: str) -> Path:
    return snapshot_root_folder() / f"{slugify(scene_name)}_{slugify(view_layer_name)}"


def current_frame_suffix(scene: bpy.types.Scene) -> str:
    return f"{scene.frame_current:04d}"


def scene_exr_path(scene: bpy.types.Scene, view_layer_name: str) -> Path:
    prefix = scene_exr_prefix(scene.name, view_layer_name)
    return Path(f"{prefix}{current_frame_suffix(scene)}.exr")


def timestamped_snapshot_blend_path() -> Path:
    blend_path = require_saved_blend()
    root = snapshot_root_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{blend_path.stem}_snapshot_{timestamp}.blend"
    counter = 1
    while candidate.exists():
        candidate = root / f"{blend_path.stem}_snapshot_{timestamp}_{counter:02d}.blend"
        counter += 1
    return candidate


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
        copy_rna_properties(source_socket, target_socket, SOCKET_PROP_EXCLUDE)
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
    target_scene.world = source_scene.world


def clone_compositor_tree(source_tree: bpy.types.NodeTree, target_tree: bpy.types.NodeTree) -> None:
    for node in list(target_tree.nodes):
        target_tree.nodes.remove(node)

    node_map: dict[bpy.types.Node, bpy.types.Node] = {}

    for source_node in source_tree.nodes:
        target_node = target_tree.nodes.new(source_node.bl_idname)
        node_map[source_node] = target_node

    for source_node, target_node in node_map.items():
        copy_rna_properties(source_node, target_node, NODE_PROP_EXCLUDE)
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
        from_index = next((i for i, socket in enumerate(source_link.from_node.outputs) if socket == source_link.from_socket), None)
        to_index = next((i for i, socket in enumerate(source_link.to_node.inputs) if socket == source_link.to_socket), None)
        if from_index is None or to_index is None:
            continue
        try:
            target_tree.links.new(
                node_map[source_link.from_node].outputs[from_index],
                node_map[source_link.to_node].inputs[to_index],
            )
        except Exception:
            continue


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


def create_snapshot_scene() -> bpy.types.Scene:
    existing = bpy.data.scenes.get(SNAPSHOT_SCENE_NAME)
    if existing is not None:
        if not REPLACE_EXISTING_SNAPSHOT_SCENE:
            raise ValueError(f"Snapshot scene '{SNAPSHOT_SCENE_NAME}' already exists.")
        if bpy.context.window is not None and bpy.context.window.scene == existing:
            fallback = next((scene for scene in bpy.data.scenes if scene != existing), None)
            if fallback is not None:
                bpy.context.window.scene = fallback
        bpy.data.scenes.remove(existing)

    scene = bpy.data.scenes.new(SNAPSHOT_SCENE_NAME)
    scene.use_nodes = True
    return scene


def helper_frame(node_tree: bpy.types.NodeTree) -> bpy.types.Node:
    frame = node_tree.nodes.get(HELPER_FRAME_NAME)
    if frame is None:
        frame = node_tree.nodes.new("NodeFrame")
        frame.name = HELPER_FRAME_NAME
        frame.label = HELPER_FRAME_NAME
        frame.location = (-2600, 1200)
        frame.use_custom_color = True
        frame.color = (0.16, 0.15, 0.08)
    return frame


def helper_render_node_name(view_layer_name: str) -> str:
    return f"{HELPER_RENDER_NODE_PREFIX}{view_layer_name}"


def helper_output_node_name(view_layer_name: str) -> str:
    return f"{HELPER_OUTPUT_NODE_PREFIX}{view_layer_name}"


def ensure_helper_render_node(
    scene: bpy.types.Scene,
    node_tree: bpy.types.NodeTree,
    view_layer_name: str,
    x: float,
    y: float,
    parent: bpy.types.Node,
) -> bpy.types.Node:
    node = node_tree.nodes.get(helper_render_node_name(view_layer_name))
    if node is None:
        node = node_tree.nodes.new("CompositorNodeRLayers")
        node.name = helper_render_node_name(view_layer_name)
    node.label = node.name
    node.layer = view_layer_name
    if hasattr(node, "scene"):
        node.scene = scene
    node.location = (x, y)
    node.parent = parent
    node.use_custom_color = True
    node.color = (0.14, 0.12, 0.18)
    return node


def clear_file_slots(output_node: bpy.types.Node) -> None:
    while len(output_node.inputs):
        output_node.file_slots.remove(output_node.inputs[0])


def ensure_helper_output_node(
    node_tree: bpy.types.NodeTree,
    render_node: bpy.types.Node,
    view_layer_name: str,
    output_prefix: Path,
    x: float,
    y: float,
    parent: bpy.types.Node,
) -> bpy.types.Node:
    node = node_tree.nodes.get(helper_output_node_name(view_layer_name))
    if node is None:
        node = node_tree.nodes.new("CompositorNodeOutputFile")
        node.name = helper_output_node_name(view_layer_name)
    node.label = node.name
    node.location = (x, y)
    node.parent = parent
    node.use_custom_color = True
    node.color = (0.12, 0.18, 0.10)
    node.base_path = str(output_prefix)
    node.format.file_format = "OPEN_EXR_MULTILAYER"
    node.format.color_mode = "RGBA"
    node.format.color_depth = "32"
    node.format.exr_codec = "ZIP"

    clear_file_slots(node)
    for socket in render_node.outputs:
        node.file_slots.new(socket.name)

    for input_socket in node.inputs:
        for link in list(input_socket.links):
            node_tree.links.remove(link)

    for socket in render_node.outputs:
        target_socket = node.inputs.get(socket.name)
        if target_socket is None:
            continue
        node_tree.links.new(socket, target_socket)

    return node


def prepare_scene_exr_outputs(scene: bpy.types.Scene) -> dict[str, Path]:
    node_tree = scene.node_tree
    frame = helper_frame(node_tree)
    frame.location = (-2600, 1200)

    output_paths: dict[str, Path] = {}
    configs = (
        (PATHWAY_VIEW_LAYER_NAME, -2400, 1200),
        (EXISTING_VIEW_LAYER_NAME, -2400, 600),
    )

    for view_layer_name, x, y in configs:
        output_prefix = scene_exr_prefix(scene.name, view_layer_name)
        render_node = ensure_helper_render_node(scene, node_tree, view_layer_name, x, y, frame)
        ensure_helper_output_node(node_tree, render_node, view_layer_name, output_prefix, x + 520, y, frame)
        output_paths[view_layer_name] = Path(f"{output_prefix}{current_frame_suffix(scene)}.exr")

    return output_paths


def latest_archived_exr(scene_name: str, view_layer_name: str) -> Path | None:
    root = snapshot_root_folder()
    snapshot_name = f"{slugify(scene_name)}_snapshot.blend"
    archive_dirs = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    for folder in archive_dirs:
        if not (folder / snapshot_name).exists():
            continue
        candidate = folder / f"{view_layer_name}.exr"
        if candidate.exists():
            return candidate
    return None


def resolve_snapshot_input_exr(scene: bpy.types.Scene, view_layer_name: str) -> Path:
    direct_path = scene_exr_path(scene, view_layer_name)
    if direct_path.exists():
        return direct_path

    legacy_temp = snapshot_root_folder() / f"temp_{view_layer_name}.exr"
    if legacy_temp.exists():
        return legacy_temp

    archived = latest_archived_exr(scene.name, view_layer_name)
    if archived is not None:
        return archived

    raise FileNotFoundError(
        f"No EXR available for '{scene.name}:{view_layer_name}'. "
        f"Render the scene once after the snapshot EXR outputs are prepared."
    )


def image_socket_name(render_socket_name: str) -> str:
    return IMAGE_SOCKET_MAP.get(render_socket_name, render_socket_name)


def remove_helper_nodes(node_tree: bpy.types.NodeTree) -> None:
    helper_nodes = [
        node for node in node_tree.nodes
        if node.name == HELPER_FRAME_NAME
        or node.name.startswith(HELPER_RENDER_NODE_PREFIX)
        or node.name.startswith(HELPER_OUTPUT_NODE_PREFIX)
    ]
    for node in helper_nodes:
        node_tree.nodes.remove(node)


def replace_render_layers_with_snapshot_images(
    node_tree: bpy.types.NodeTree,
    exr_paths_by_layer: dict[str, Path],
) -> tuple[list[str], list[str]]:
    loaded_images = {
        layer_name: bpy.data.images.load(str(exr_path), check_existing=True)
        for layer_name, exr_path in exr_paths_by_layer.items()
    }

    rewired: list[str] = []
    missing: list[str] = []
    render_nodes = [node for node in node_tree.nodes if node.bl_idname == "CompositorNodeRLayers"]

    for render_node in render_nodes:
        layer_name = render_node.layer or PATHWAY_VIEW_LAYER_NAME
        exr_image = loaded_images.get(layer_name)
        if exr_image is None:
            missing.append(f"{layer_name}:image")
            continue

        image_node = node_tree.nodes.new("CompositorNodeImage")
        image_node.name = f"{SNAPSHOT_IMAGE_NODE_PREFIX}{layer_name}"
        image_node.label = image_node.name
        image_node.location = render_node.location.copy()
        image_node.width = render_node.width
        image_node.hide = render_node.hide
        image_node.mute = render_node.mute
        image_node.use_custom_color = True
        image_node.color = (0.12, 0.18, 0.10)
        image_node.image = exr_image
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


def export_snapshot_blend(scene: bpy.types.Scene) -> Path:
    path = timestamped_snapshot_blend_path()
    bpy.data.libraries.write(str(path), collect_datablocks_for_export(scene))
    return path


def cleanup_snapshot_scene(snapshot_scene: bpy.types.Scene, source_scene: bpy.types.Scene) -> None:
    if KEEP_SNAPSHOT_SCENE_IN_SOURCE_BLEND:
        return
    if bpy.context.window is not None:
        bpy.context.window.scene = source_scene
    bpy.data.scenes.remove(snapshot_scene)


def main() -> None:
    source_scene = require_scene(TARGET_SCENE_NAME)
    missing_layers = [
        name
        for name in (PATHWAY_VIEW_LAYER_NAME, EXISTING_VIEW_LAYER_NAME)
        if source_scene.view_layers.get(name) is None
    ]
    if missing_layers:
        raise ValueError(
            f"Scene '{TARGET_SCENE_NAME}' is missing required view layers: {', '.join(missing_layers)}"
        )

    root_folder = snapshot_root_folder()
    prepared_paths = {}
    if PREPARE_SCENE_OUTPUTS:
        prepared_paths = prepare_scene_exr_outputs(source_scene)
        log(
            "Prepared live EXR outputs: "
            + ", ".join(f"{layer} -> {path}" for layer, path in prepared_paths.items())
        )

    snapshot_scene = create_snapshot_scene()
    copy_scene_settings(source_scene, snapshot_scene)
    clone_compositor_tree(source_scene.node_tree, snapshot_scene.node_tree)
    remove_helper_nodes(snapshot_scene.node_tree)

    snapshot_exrs = {
        PATHWAY_VIEW_LAYER_NAME: resolve_snapshot_input_exr(source_scene, PATHWAY_VIEW_LAYER_NAME),
        EXISTING_VIEW_LAYER_NAME: resolve_snapshot_input_exr(source_scene, EXISTING_VIEW_LAYER_NAME),
    }
    rewired, missing = replace_render_layers_with_snapshot_images(snapshot_scene.node_tree, snapshot_exrs)

    snapshot_blend = None
    if EXPORT_SNAPSHOT_BLEND:
        snapshot_blend = export_snapshot_blend(snapshot_scene)
        log(f"Exported compositor snapshot blend to {snapshot_blend}")

    if SAVE_SOURCE_BLEND:
        bpy.ops.wm.save_mainfile()
        log(f"Saved source blend: {bpy.data.filepath}")

    cleanup_snapshot_scene(snapshot_scene, source_scene)

    log(f"Snapshot root folder: {root_folder}")
    log(f"Snapshot input EXRs: {snapshot_exrs[PATHWAY_VIEW_LAYER_NAME]}, {snapshot_exrs[EXISTING_VIEW_LAYER_NAME]}")
    log(f"Rewired outputs: {', '.join(rewired) if rewired else 'none'}")
    if missing:
        log(f"Missing snapshot outputs: {', '.join(missing)}")
    if prepared_paths:
        log(
            "Render the source scene normally to write the EXRs: "
            + ", ".join(str(path) for path in prepared_paths.values())
        )


if __name__ == "__main__":
    main()
