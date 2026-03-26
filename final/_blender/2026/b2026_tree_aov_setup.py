import bpy


TARGET_SCENE_NAME = "parade-senescent"
TARGET_VIEW_LAYER_NAME = "ViewLayer"
MATERIAL_PREFIXES = ("RESOURCES", "STUPIDRESOURCES")

AOV_SPECS = (
    {"aov_name": "size", "attr_name": "size", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "control", "attr_name": "control", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "node_type", "attr_name": "node_type", "type": "VALUE", "attribute_type": "INSTANCER"},
    {
        "aov_name": "tree_interventions",
        "attr_name": "tree_interventions",
        "type": "VALUE",
        "attribute_type": "INSTANCER",
    },
    {
        "aov_name": "tree_proposals",
        "attr_name": "tree_proposals",
        "type": "VALUE",
        "attribute_type": "INSTANCER",
    },
    {"aov_name": "improvement", "attr_name": "improvement", "type": "VALUE", "attribute_type": "INSTANCER"},
    {
        "aov_name": "canopy_resistance",
        "attr_name": "canopy_resistance",
        "type": "VALUE",
        "attribute_type": "INSTANCER",
    },
    {"aov_name": "node_id", "attr_name": "node_id", "type": "VALUE", "attribute_type": "INSTANCER"},
)


def require_scene(scene_name):
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' was not found.")
    return scene


def require_view_layer(scene, view_layer_name):
    view_layer = scene.view_layers.get(view_layer_name)
    if view_layer is not None:
        return view_layer
    if scene.view_layers:
        return scene.view_layers[0]
    raise ValueError(f"Scene '{scene.name}' has no view layers.")


def iter_target_view_layers(scene, view_layer_name):
    if view_layer_name is None:
        return list(scene.view_layers)
    return [require_view_layer(scene, view_layer_name)]


def ensure_aov(view_layer, spec):
    for existing in view_layer.aovs:
        if existing.name == spec["aov_name"]:
            existing.type = spec["type"]
            return existing

    aov = view_layer.aovs.add()
    aov.name = spec["aov_name"]
    aov.type = spec["type"]
    return aov


def relevant_materials():
    return [
        material
        for material in bpy.data.materials
        if material.use_nodes
        and material.node_tree is not None
        and material.name.startswith(MATERIAL_PREFIXES)
    ]


def ensure_attribute_node(node_tree, spec, location):
    node_name = f"AOV Attribute {spec['aov_name']}"
    node = node_tree.nodes.get(node_name)
    if node is None or node.bl_idname != "ShaderNodeAttribute":
        node = node_tree.nodes.new("ShaderNodeAttribute")
    node.name = node_name
    node.label = node_name
    node.location = location
    node.attribute_name = spec["attr_name"]
    if hasattr(node, "attribute_type"):
        node.attribute_type = spec["attribute_type"]
    return node


def ensure_aov_output_node(node_tree, spec, location):
    node_name = f"AOV Output {spec['aov_name']}"
    node = node_tree.nodes.get(node_name)
    if node is None or node.bl_idname != "ShaderNodeOutputAOV":
        node = node_tree.nodes.new("ShaderNodeOutputAOV")
    node.name = node_name
    node.label = node_name
    node.location = location
    node.aov_name = spec["aov_name"]
    return node


def ensure_link(node_tree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def patch_material(material):
    nodes = material.node_tree.nodes
    aov_nodes = [node for node in nodes if node.bl_idname == "ShaderNodeOutputAOV"]
    base_x = max((node.location.x for node in aov_nodes), default=400.0)
    base_y = max((node.location.y for node in aov_nodes), default=300.0)

    for index, spec in enumerate(AOV_SPECS):
        y = base_y - (index * 180.0)
        attr_node = ensure_attribute_node(material.node_tree, spec, (base_x - 320.0, y))
        aov_node = ensure_aov_output_node(material.node_tree, spec, (base_x, y))
        ensure_link(material.node_tree, attr_node.outputs["Fac"], aov_node.inputs["Value"])


def main():
    scene = require_scene(TARGET_SCENE_NAME)

    ensured_aovs = []
    target_view_layers = iter_target_view_layers(scene, TARGET_VIEW_LAYER_NAME)
    for view_layer in target_view_layers:
        for spec in AOV_SPECS:
            ensure_aov(view_layer, spec)
        ensured_aovs.append((view_layer.name, [spec["aov_name"] for spec in AOV_SPECS]))

    patched_materials = []
    for material in relevant_materials():
        patch_material(material)
        patched_materials.append(material.name)

    print(f"Scene: {scene.name}")
    print(f"View layers: {[view_layer.name for view_layer in target_view_layers]}")
    print(f"AOVs ensured: {ensured_aovs}")
    print(f"Materials patched: {patched_materials}")


if __name__ == "__main__":
    main()
