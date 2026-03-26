import bpy
import os


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


PATHWAY_LAYER_NAME = env_str("B2026_PATHWAY_LAYER_NAME", "pathway_state")
EXISTING_LAYER_NAME = env_str("B2026_EXISTING_LAYER_NAME", "existing_condition")
PRIMARY_RENDER_NODE_NAME = env_str("B2026_PRIMARY_RENDER_NODE_NAME", "Render Layers")
EXISTING_RENDER_NODE_NAME = env_str(
    "B2026_EXISTING_RENDER_NODE_NAME", "Render Layers Existing Condition"
)
TARGET_SCENE_NAMES = [
    scene_name.strip()
    for scene_name in env_str("B2026_TARGET_SCENE_NAMES", "city,parade").split(",")
    if scene_name.strip()
]

SCENE_COLLECTION_EXCLUDES = {
    "city": [
        "city_envelope",
        "City_Manager",
        "City_Camera",
        "City_Year_60",
        "City_Year_180",
        "City_Year_180_Legacy",
    ],
    "parade": [
        "Parade_envelope",
        "Parade_Manager",
        "Parade_Camera",
        "Parade_Year_60",
        "Parade_Year_180",
        "Parade_Year_180_Trending_Archive",
        "Parade_Year_0_Archive",
    ],
}

BIO_TOPO_RAMP = [
    (0.0, (1.0, 1.0, 1.0, 1.0)),
    (0.25, (0.9560, 0.4508, 0.0887, 1.0)),
    (0.5, (0.2622, 0.0823, 0.0030, 1.0)),
    (0.75, (0.3564, 0.4508, 0.1529, 1.0)),
    (1.0, (0.0931, 0.1329, 0.0319, 1.0)),
]


def log(message: str) -> None:
    print(f"[restore_existing_condition] {message}")


def require_scene(scene_name: str) -> bpy.types.Scene:
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' not found.")
    return scene


def ensure_view_layer(scene: bpy.types.Scene, layer_name: str) -> bpy.types.ViewLayer:
    layer = scene.view_layers.get(layer_name)
    if layer is None:
        layer = scene.view_layers.new(name=layer_name)
    return layer


def rename_primary_view_layer(scene: bpy.types.Scene) -> bpy.types.ViewLayer:
    current = scene.view_layers.get(PATHWAY_LAYER_NAME)
    if current is not None:
        return current

    legacy = scene.view_layers.get("ViewLayer")
    if legacy is None:
        raise ValueError(
            f"Scene '{scene.name}' has neither '{PATHWAY_LAYER_NAME}' nor 'ViewLayer'."
        )
    legacy.name = PATHWAY_LAYER_NAME
    return legacy


def ensure_aov(view_layer: bpy.types.ViewLayer, aov_name: str) -> None:
    if view_layer.aovs.get(aov_name) is None:
        view_layer.aovs.add()
        view_layer.aovs[-1].name = aov_name


def copy_view_layer_contract(
    source: bpy.types.ViewLayer, target: bpy.types.ViewLayer
) -> None:
    target.use_pass_z = source.use_pass_z
    target.use_pass_normal = source.use_pass_normal
    target.use_pass_object_index = source.use_pass_object_index
    target.use_pass_ambient_occlusion = source.use_pass_ambient_occlusion
    for aov in source.aovs:
        ensure_aov(target, aov.name)


def set_layer_collection_exclude_recursive(
    layer_collection: bpy.types.LayerCollection,
    collection_name: str,
    exclude: bool,
) -> bool:
    if layer_collection.name == collection_name:
        layer_collection.exclude = exclude
        return True
    for child in layer_collection.children:
        if set_layer_collection_exclude_recursive(child, collection_name, exclude):
            return True
    return False


def set_top_level_collection_visibility(
    view_layer: bpy.types.ViewLayer,
    included_names: set[str],
    excluded_names: set[str],
) -> None:
    for child in view_layer.layer_collection.children:
        if child.name in excluded_names:
            child.exclude = True
        elif child.name in included_names:
            child.exclude = False


def apply_color_ramp(node: bpy.types.Node, elements: list[tuple[float, tuple[float, float, float, float]]]) -> None:
    ramp = node.color_ramp
    while len(ramp.elements) > 1:
        ramp.elements.remove(ramp.elements[-1])
    ramp.elements[0].position = elements[0][0]
    ramp.elements[0].color = elements[0][1]
    for position, color in elements[1:]:
        element = ramp.elements.new(position)
        element.color = color


def restore_bio_ramps(scene: bpy.types.Scene) -> None:
    if not scene.use_nodes or scene.node_tree is None:
        return
    for node_name in ["Bio__Color Ramp.004", "Bio__Color Ramp.011"]:
        node = scene.node_tree.nodes.get(node_name)
        if node is None or not hasattr(node, "color_ramp"):
            log(f"Scene '{scene.name}' is missing '{node_name}', skipping ramp restore.")
            continue
        apply_color_ramp(node, BIO_TOPO_RAMP)


def configure_existing_condition_layer(
    scene: bpy.types.Scene,
    view_layer: bpy.types.ViewLayer,
) -> None:
    excluded_names = set(SCENE_COLLECTION_EXCLUDES.get(scene.name, []))
    top_level_names = {child.name for child in view_layer.layer_collection.children}
    included_names = top_level_names - excluded_names
    set_top_level_collection_visibility(view_layer, included_names, excluded_names)
    for collection_name in excluded_names:
        if not set_layer_collection_exclude_recursive(
            view_layer.layer_collection, collection_name, True
        ):
            log(
                f"Scene '{scene.name}' does not contain collection '{collection_name}' for existing condition."
            )


def ensure_render_layers_node(
    node_tree: bpy.types.NodeTree,
    scene: bpy.types.Scene,
    node_name: str,
    layer_name: str,
    location: tuple[float, float],
) -> bpy.types.CompositorNodeRLayers:
    node = node_tree.nodes.get(node_name)
    if node is None or node.bl_idname != "CompositorNodeRLayers":
        node = node_tree.nodes.new("CompositorNodeRLayers")
        node.name = node_name
        node.label = node_name
    node.location = location
    if hasattr(node, "scene"):
        node.scene = scene
    node.layer = layer_name
    return node


def relink_existing_condition_underlay(
    scene: bpy.types.Scene,
    primary_layer: bpy.types.ViewLayer,
) -> None:
    if not scene.use_nodes or scene.node_tree is None:
        raise ValueError(f"Scene '{scene.name}' has no compositor node tree.")

    node_tree = scene.node_tree
    primary_render_node = node_tree.nodes.get(PRIMARY_RENDER_NODE_NAME)
    if primary_render_node is None or primary_render_node.bl_idname != "CompositorNodeRLayers":
        raise ValueError(
            f"Scene '{scene.name}' is missing primary render node '{PRIMARY_RENDER_NODE_NAME}'."
        )
    if hasattr(primary_render_node, "scene"):
        primary_render_node.scene = scene
    primary_render_node.layer = primary_layer.name

    target_node = node_tree.nodes.get("Bio__Set Alpha.023")
    if target_node is None or target_node.bl_idname != "CompositorNodeSetAlpha":
        log(
            f"Scene '{scene.name}' does not have 'Bio__Set Alpha.023'; skipping existing-condition relink."
        )
        return

    existing_render_node = ensure_render_layers_node(
        node_tree,
        scene,
        EXISTING_RENDER_NODE_NAME,
        EXISTING_LAYER_NAME,
        (target_node.location.x - 520.0, target_node.location.y + 60.0),
    )

    image_input = target_node.inputs.get("Image")
    if image_input is None:
        raise ValueError(f"'Bio__Set Alpha.023' in scene '{scene.name}' has no Image input.")

    for link in list(image_input.links):
        node_tree.links.remove(link)
    node_tree.links.new(existing_render_node.outputs["Image"], image_input)


def main() -> None:
    for scene_name in TARGET_SCENE_NAMES:
        scene = require_scene(scene_name)
        scene.render.film_transparent = True
        pathway_layer = rename_primary_view_layer(scene)
        existing_layer = ensure_view_layer(scene, EXISTING_LAYER_NAME)

        copy_view_layer_contract(pathway_layer, existing_layer)
        configure_existing_condition_layer(scene, existing_layer)
        relink_existing_condition_underlay(scene, pathway_layer)
        restore_bio_ramps(scene)

        log(
            f"Configured scene '{scene.name}' with view layers: "
            f"{[view_layer.name for view_layer in scene.view_layers]}"
        )

    bpy.ops.wm.save_mainfile()
    log(f"Saved {bpy.data.filepath}")


if __name__ == "__main__":
    main()
