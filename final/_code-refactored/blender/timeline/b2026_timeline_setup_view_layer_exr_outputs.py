import bpy
import os
from pathlib import Path


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


TARGET_SCENE_NAME = env_str("B2026_TARGET_SCENE_NAME", "")
OUTPUT_DIR_OVERRIDE = env_str("B2026_OUTPUT_DIR", "")
OUTPUT_BASENAME_OVERRIDE = env_str("B2026_OUTPUT_BASENAME", "")
OUTPUT_TAG_OVERRIDE = env_str("B2026_OUTPUT_TAG", "")
TARGET_VIEW_LAYERS_OVERRIDE = tuple(
    name.strip()
    for name in env_str("B2026_TARGET_VIEW_LAYERS", "").split(",")
    if name.strip()
)
FRAME_NAME = "View Layer EXR Outputs"
NODE_PREFIX = "ViewLayer EXR :: "
FRAME_NODE_NAME = f"{NODE_PREFIX}{FRAME_NAME}"
RENDER_NODE_PREFIX = f"{NODE_PREFIX}Render Layers :: "
OUTPUT_NODE_PREFIX = f"{NODE_PREFIX}Output :: "
HANDLER_NAME = "b2026_view_layer_exr_rename_post"


def safe_name(value: str) -> str:
    return "".join(char if char not in '\\/:*?"<>|' else "_" for char in value).strip()


def require_saved_blend() -> Path:
    if not bpy.data.filepath:
        raise ValueError("Save the Blender file before setting up EXR outputs.")
    return Path(bpy.data.filepath)


def require_scene() -> bpy.types.Scene:
    if TARGET_SCENE_NAME:
        scene = bpy.data.scenes.get(TARGET_SCENE_NAME)
        if scene is None:
            raise ValueError(f"Scene '{TARGET_SCENE_NAME}' was not found.")
        return scene

    scene = bpy.context.scene
    if scene is None:
        raise ValueError("No active scene is available.")
    return scene


def output_folder_for_scene(scene: bpy.types.Scene) -> Path:
    if OUTPUT_DIR_OVERRIDE:
        folder = Path(OUTPUT_DIR_OVERRIDE)
        folder.mkdir(parents=True, exist_ok=True)
        return folder
    blend_path = require_saved_blend()
    folder = blend_path.parent / f"{blend_path.stem}-{safe_name(scene.name)}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def output_stem(scene_name: str, view_layer_name: str) -> str:
    if OUTPUT_BASENAME_OVERRIDE:
        stem = f"{safe_name(OUTPUT_BASENAME_OVERRIDE)}_{safe_name(view_layer_name)}"
        if OUTPUT_TAG_OVERRIDE:
            stem = f"{stem}_{safe_name(OUTPUT_TAG_OVERRIDE)}"
        return stem
    return f"{safe_name(scene_name)}-{safe_name(view_layer_name)}"


def ensure_nodes_enabled(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    scene.use_nodes = True
    if scene.node_tree is None:
        raise ValueError(f"Scene '{scene.name}' does not have a compositor node tree.")
    return scene.node_tree


def remove_existing_nodes(node_tree: bpy.types.NodeTree) -> None:
    for node in list(node_tree.nodes):
        if node.name.startswith(NODE_PREFIX):
            node_tree.nodes.remove(node)


def new_node(
    node_tree: bpy.types.NodeTree,
    bl_idname: str,
    name: str,
    label: str,
    location: tuple[float, float],
    parent=None,
    color=None,
):
    node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    if parent is not None:
        node.parent = parent
    if color is not None:
        node.use_custom_color = True
        node.color = color
    return node


def ensure_frame(node_tree: bpy.types.NodeTree, location: tuple[float, float]):
    frame = new_node(
        node_tree,
        "NodeFrame",
        FRAME_NODE_NAME,
        FRAME_NAME,
        location,
    )
    frame.use_custom_color = True
    frame.color = (0.14, 0.14, 0.14)
    frame.shrink = False
    return frame


def ensure_render_layer_node(
    node_tree: bpy.types.NodeTree,
    scene: bpy.types.Scene,
    view_layer_name: str,
    location: tuple[float, float],
    parent,
):
    node = new_node(
        node_tree,
        "CompositorNodeRLayers",
        f"{RENDER_NODE_PREFIX}{view_layer_name}",
        f"Render Layers :: {view_layer_name}",
        location,
        parent,
        (0.18, 0.16, 0.10),
    )
    node.scene = scene
    node.layer = view_layer_name
    return node


def ensure_file_output_node(
    node_tree: bpy.types.NodeTree,
    scene: bpy.types.Scene,
    view_layer_name: str,
    output_prefix: Path,
    location: tuple[float, float],
    parent,
):
    node = new_node(
        node_tree,
        "CompositorNodeOutputFile",
        f"{OUTPUT_NODE_PREFIX}{view_layer_name}",
        f"EXR Output :: {view_layer_name}",
        location,
        parent,
        (0.12, 0.20, 0.14),
    )
    node.base_path = str(output_prefix)
    node.format.file_format = "OPEN_EXR_MULTILAYER"
    node.format.color_mode = "RGBA"
    node.format.color_depth = "32"
    node.format.exr_codec = "ZIP"
    return node


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def clear_file_slots(output_node: bpy.types.Node) -> None:
    while len(output_node.inputs):
        output_node.file_slots.remove(output_node.inputs[0])


def expected_frame_path(folder: Path, scene_name: str, view_layer_name: str, frame: int) -> Path:
    return folder / f"{output_stem(scene_name, view_layer_name)}{frame:04d}.exr"


def expected_final_path(folder: Path, scene_name: str, view_layer_name: str) -> Path:
    return folder / f"{output_stem(scene_name, view_layer_name)}.exr"


def rename_rendered_exrs(scene: bpy.types.Scene) -> None:
    folder = output_folder_for_scene(scene)
    frame = scene.frame_current
    view_layers = [
        view_layer
        for view_layer in scene.view_layers
        if not TARGET_VIEW_LAYERS_OVERRIDE or view_layer.name in TARGET_VIEW_LAYERS_OVERRIDE
    ]
    for view_layer in view_layers:
        rendered = expected_frame_path(folder, scene.name, view_layer.name, frame)
        final = expected_final_path(folder, scene.name, view_layer.name)
        if not rendered.exists():
            continue
        if final.exists():
            final.unlink()
        rendered.replace(final)
        print(f"Renamed {rendered.name} -> {final.name}")


def remove_handler() -> None:
    for handler in list(bpy.app.handlers.render_post):
        if getattr(handler, "__name__", "") == HANDLER_NAME:
            bpy.app.handlers.render_post.remove(handler)


def register_handler() -> None:
    remove_handler()

    def _handler(scene, _depsgraph=None):
        rename_rendered_exrs(scene)

    _handler.__name__ = HANDLER_NAME
    bpy.app.handlers.render_post.append(_handler)


def build_outputs(scene: bpy.types.Scene) -> Path:
    node_tree = ensure_nodes_enabled(scene)
    remove_existing_nodes(node_tree)
    folder = output_folder_for_scene(scene)
    view_layers = [
        view_layer
        for view_layer in scene.view_layers
        if not TARGET_VIEW_LAYERS_OVERRIDE or view_layer.name in TARGET_VIEW_LAYERS_OVERRIDE
    ]

    start_x = -2200
    start_y = 1200
    row_gap = 260
    frame = ensure_frame(node_tree, (start_x - 120, start_y + 120))

    for index, view_layer in enumerate(view_layers):
        y = start_y - index * row_gap
        output_prefix = folder / output_stem(scene.name, view_layer.name)
        render_node = ensure_render_layer_node(
            node_tree,
            scene,
            view_layer.name,
            (start_x, y),
            frame,
        )
        output_node = ensure_file_output_node(
            node_tree,
            scene,
            view_layer.name,
            output_prefix,
            (start_x + 380, y),
            frame,
        )
        clear_file_slots(output_node)
        enabled_sockets = [socket for socket in render_node.outputs if getattr(socket, "enabled", True)]
        for socket in enabled_sockets:
            output_node.file_slots.new(socket.name)

        for socket in enabled_sockets:
            target_socket = output_node.inputs.get(socket.name)
            if target_socket is None:
                continue
            ensure_link(node_tree, socket, target_socket)

    register_handler()
    scene["b2026_view_layer_exr_output_dir"] = str(folder)
    return folder


def main():
    scene = require_scene()
    base_path = build_outputs(scene)
    view_layers = [
        view_layer
        for view_layer in scene.view_layers
        if not TARGET_VIEW_LAYERS_OVERRIDE or view_layer.name in TARGET_VIEW_LAYERS_OVERRIDE
    ]
    print(f"Prepared per-view-layer EXR outputs for scene: {scene.name}")
    print(f"Output folder: {base_path}")
    for view_layer in view_layers:
        print(f"- {view_layer.name} -> {expected_final_path(base_path, scene.name, view_layer.name).name}")
    print("Render the scene once to write the EXRs.")


if __name__ == "__main__":
    main()
