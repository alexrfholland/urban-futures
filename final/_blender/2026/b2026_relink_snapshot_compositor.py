import bpy
import os
from pathlib import Path


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


TARGET_SCENE_NAMES = env_list("B2026_TARGET_SCENE_NAMES", ["parade", "city"])
SNAPSHOT_SCENE_NAME = env_str("B2026_SNAPSHOT_SCENE_NAME", "futures_compositing_pipeline")
SNAPSHOT_BLEND_PATH = env_str(
    "B2026_SNAPSHOT_BLEND_PATH",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/initial snapshot/futures_compositing_pipeline.blend",
)
PATHWAY_VIEW_LAYER_NAME = env_str("B2026_PATHWAY_VIEW_LAYER_NAME", "pathway_state")
EXISTING_VIEW_LAYER_NAME = env_str("B2026_EXISTING_VIEW_LAYER_NAME", "existing_condition")
PATHWAY_RENDER_NODE_NAME = env_str("B2026_PATHWAY_RENDER_NODE_NAME", "Render Layers")
EXISTING_RENDER_NODE_NAME = env_str(
    "B2026_EXISTING_RENDER_NODE_NAME",
    "Render Layers Existing Condition",
)
SNAPSHOT_IMAGE_NODE_PREFIX = env_str("B2026_SNAPSHOT_IMAGE_NODE_PREFIX", "Snapshot EXR :: ")
ARCHIVE_EXISTING_TARGET_SCENES = env_bool("B2026_ARCHIVE_EXISTING_TARGET_SCENES", True)
REPLACE_EXISTING_ARCHIVE_SCENES = env_bool("B2026_REPLACE_EXISTING_ARCHIVE_SCENES", True)
CLEANUP_IMPORTED_SNAPSHOT_SCENE = env_bool("B2026_CLEANUP_IMPORTED_SNAPSHOT_SCENE", True)

IMAGE_SOCKET_TO_RENDER_SOCKET = {
    "Combined": "Image",
}

NODE_PROP_EXCLUDE = {
    "rna_type",
    "dimensions",
    "location_absolute",
    "layer",
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


def require_scene(scene_name: str) -> bpy.types.Scene:
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' was not found.")
    return scene


def remove_scene_if_present(scene_name: str) -> None:
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        return
    if bpy.context.window and bpy.context.window.scene == scene:
        fallback = next((candidate for candidate in bpy.data.scenes if candidate != scene), None)
        if fallback is None:
            raise ValueError(f"Cannot remove scene '{scene_name}' because it is the only scene in the file.")
        bpy.context.window.scene = fallback
    bpy.data.scenes.remove(scene)


def archive_scene_if_requested(scene: bpy.types.Scene) -> bpy.types.Scene | None:
    if not ARCHIVE_EXISTING_TARGET_SCENES:
        return None

    archive_name = f"{slugify(scene.name)}_compositor_archive"
    existing = bpy.data.scenes.get(archive_name)
    if existing is not None:
        if not REPLACE_EXISTING_ARCHIVE_SCENES:
            raise ValueError(f"Archive scene '{archive_name}' already exists.")
        remove_scene_if_present(archive_name)

    archive_scene = scene.copy()
    archive_scene.name = archive_name
    return archive_scene


def import_snapshot_scene(snapshot_blend_path: Path) -> bpy.types.Scene:
    with bpy.data.libraries.load(str(snapshot_blend_path), link=False) as (data_from, data_to):
        scene_names = [name for name in data_from.scenes if name == SNAPSHOT_SCENE_NAME]
        if not scene_names:
            raise ValueError(
                f"No scene named '{SNAPSHOT_SCENE_NAME}' was found in snapshot blend {snapshot_blend_path}"
            )
        data_to.scenes = scene_names

    imported_scene = next((scene for scene in data_to.scenes if scene is not None), None)
    if imported_scene is None:
        raise ValueError(f"Failed to import '{SNAPSHOT_SCENE_NAME}' from {snapshot_blend_path}")
    return imported_scene


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
        from_index = next((i for i, s in enumerate(source_link.from_node.outputs) if s == source_link.from_socket), None)
        to_index = next((i for i, s in enumerate(source_link.to_node.inputs) if s == source_link.to_socket), None)
        if from_index is None or to_index is None:
            continue
        try:
            target_tree.links.new(
                node_map[source_link.from_node].outputs[from_index],
                node_map[source_link.to_node].inputs[to_index],
            )
        except Exception:
            continue


def render_node_name_for_layer(layer_name: str) -> str:
    if layer_name == EXISTING_VIEW_LAYER_NAME:
        return EXISTING_RENDER_NODE_NAME
    return PATHWAY_RENDER_NODE_NAME


def ensure_render_layers_node(
    node_tree: bpy.types.NodeTree,
    target_scene: bpy.types.Scene,
    layer_name: str,
    anchor_node: bpy.types.Node,
) -> bpy.types.CompositorNodeRLayers:
    target_name = render_node_name_for_layer(layer_name)
    node = node_tree.nodes.get(target_name)
    if node is not None and node.bl_idname == "CompositorNodeRLayers":
        node.layer = layer_name
        if hasattr(node, "scene"):
            node.scene = target_scene
        node.location = anchor_node.location.copy()
        return node

    candidates = [node for node in node_tree.nodes if node.bl_idname == "CompositorNodeRLayers"]
    for candidate in candidates:
        if getattr(candidate, "layer", "") == layer_name:
            candidate.name = target_name
            candidate.label = target_name
            if hasattr(candidate, "scene"):
                candidate.scene = target_scene
            candidate.layer = layer_name
            candidate.location = anchor_node.location.copy()
            return candidate

    node = node_tree.nodes.new("CompositorNodeRLayers")
    node.name = target_name
    node.label = target_name
    node.layer = layer_name
    if hasattr(node, "scene"):
        node.scene = target_scene
    node.location = anchor_node.location.copy()
    return node


def render_socket_name(image_socket_name: str) -> str:
    return IMAGE_SOCKET_TO_RENDER_SOCKET.get(image_socket_name, image_socket_name)


def reconnect_snapshot_image_to_render_layers(
    node_tree: bpy.types.NodeTree,
    image_node: bpy.types.CompositorNodeImage,
    render_node: bpy.types.CompositorNodeRLayers,
) -> tuple[list[str], list[str]]:
    outgoing_links = [link for link in node_tree.links if link.from_node == image_node]
    rewired: list[str] = []
    missing: list[str] = []

    for link in outgoing_links:
        output_name = render_socket_name(link.from_socket.name)
        replacement_socket = render_node.outputs.get(output_name)
        if replacement_socket is None:
            missing.append(f"{render_node.layer}:{output_name}")
            continue

        target_socket = link.to_socket
        node_tree.links.remove(link)
        node_tree.links.new(replacement_socket, target_socket)
        rewired.append(f"{render_node.layer}:{output_name}")

    return sorted(set(rewired)), sorted(set(missing))


def snapshot_image_nodes(node_tree: bpy.types.NodeTree) -> list[bpy.types.CompositorNodeImage]:
    nodes = []
    for node in node_tree.nodes:
        if node.bl_idname != "CompositorNodeImage":
            continue
        if node.name.startswith(SNAPSHOT_IMAGE_NODE_PREFIX) or node.label.startswith(SNAPSHOT_IMAGE_NODE_PREFIX):
            nodes.append(node)
    if not nodes:
        raise ValueError("No snapshot EXR image nodes were found in the imported compositor.")
    return nodes


def infer_layer_name(image_node: bpy.types.CompositorNodeImage) -> str:
    if getattr(image_node, "layer", ""):
        return image_node.layer
    for text in (image_node.name, image_node.label):
        if text.startswith(SNAPSHOT_IMAGE_NODE_PREFIX):
            return text.removeprefix(SNAPSHOT_IMAGE_NODE_PREFIX).strip()
    return PATHWAY_VIEW_LAYER_NAME


def cleanup_snapshot_artifacts(imported_snapshot_scene: bpy.types.Scene, imported_images: list[bpy.types.Image]) -> None:
    if CLEANUP_IMPORTED_SNAPSHOT_SCENE:
        if bpy.context.window and bpy.context.window.scene == imported_snapshot_scene:
            fallback = next((scene for scene in bpy.data.scenes if scene != imported_snapshot_scene), None)
            if fallback is not None:
                bpy.context.window.scene = fallback
        bpy.data.scenes.remove(imported_snapshot_scene)

    for image in imported_images:
        if image is not None and image.users == 0:
            bpy.data.images.remove(image)


def apply_snapshot_to_scene(
    target_scene: bpy.types.Scene,
    imported_snapshot_scene: bpy.types.Scene,
) -> tuple[list[str], list[str], list[bpy.types.Image]]:
    target_scene.use_nodes = True
    clone_compositor_tree(imported_snapshot_scene.node_tree, target_scene.node_tree)

    node_tree = target_scene.node_tree
    images_to_cleanup: list[bpy.types.Image] = []
    rewired_all: list[str] = []
    missing_all: list[str] = []

    for image_node in snapshot_image_nodes(node_tree):
        layer_name = infer_layer_name(image_node)
        if target_scene.view_layers.get(layer_name) is None:
            raise ValueError(
                f"Target scene '{target_scene.name}' is missing required live view layer '{layer_name}'."
            )
        images_to_cleanup.append(image_node.image)
        render_node = ensure_render_layers_node(node_tree, target_scene, layer_name, image_node)
        rewired, missing = reconnect_snapshot_image_to_render_layers(node_tree, image_node, render_node)
        rewired_all.extend(rewired)
        missing_all.extend(missing)
        if not image_node.outputs["Combined"].links:
            node_tree.nodes.remove(image_node)

    return sorted(set(rewired_all)), sorted(set(missing_all)), images_to_cleanup


def main() -> None:
    snapshot_blend_path = Path(SNAPSHOT_BLEND_PATH)
    if not snapshot_blend_path.exists():
        raise FileNotFoundError(f"Snapshot blend not found: {snapshot_blend_path}")

    imported_snapshot_scene = import_snapshot_scene(snapshot_blend_path)

    if bpy.context.window is not None:
        bpy.context.window.scene = require_scene(TARGET_SCENE_NAMES[0])

    imported_images: list[bpy.types.Image] = []
    results = []
    for scene_name in TARGET_SCENE_NAMES:
        target_scene = require_scene(scene_name)
        archive_scene_if_requested(target_scene)
        rewired, missing, image_refs = apply_snapshot_to_scene(target_scene, imported_snapshot_scene)
        imported_images.extend(image_refs)
        results.append((scene_name, rewired, missing))

    cleanup_snapshot_artifacts(imported_snapshot_scene, imported_images)

    if bpy.context.window is not None:
        bpy.context.window.scene = require_scene(TARGET_SCENE_NAMES[0])

    bpy.ops.wm.save_mainfile()

    print(f"Imported snapshot compositor from: {snapshot_blend_path}")
    for scene_name, rewired, missing in results:
        print(f"Target scene: {scene_name}")
        print(f"Rewired outputs: {', '.join(rewired) if rewired else 'none'}")
        if missing:
            print(f"Unmatched snapshot outputs left disconnected: {', '.join(missing)}")
        else:
            print("All snapshot outputs were relinked to live Render Layers.")


if __name__ == "__main__":
    main()
