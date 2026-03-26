import bpy


TARGET_SCENE_NAMES = ("parade", "city")
TARGET_GROUP_NAMES = (
    "Background Cubes",
    "Background - Large pts Cubes",
    "Background - Large pts.001 Cubes",
    "Background.001 Cubes",
)
SAVE_FILE = True

WORLD_AOV_SPECS = (
    {
        "aov_name": "world_sim_turns",
        "attr_name": "sim_Turns",
        "type": "VALUE",
        "attribute_type": "INSTANCER",
    },
    {
        "aov_name": "world_sim_nodes",
        "attr_name": "sim_Nodes",
        "type": "VALUE",
        "attribute_type": "INSTANCER",
    },
    {
        "aov_name": "world_design_bioenvelope",
        "attr_name": "scenario_bioEnvelope",
        "type": "VALUE",
        "attribute_type": "INSTANCER",
    },
    {
        "aov_name": "world_design_bioenvelope_simple",
        "attr_name": "scenario_bioEnvelopeSimple",
        "type": "VALUE",
        "attribute_type": "INSTANCER",
    },
    {
        "aov_name": "world_sim_matched",
        "attr_name": "sim_Matched",
        "type": "VALUE",
        "attribute_type": "INSTANCER",
    },
)


def ensure_aov(view_layer, spec):
    for existing in view_layer.aovs:
        if existing.name == spec["aov_name"]:
            existing.type = spec["type"]
            return existing

    aov = view_layer.aovs.add()
    aov.name = spec["aov_name"]
    aov.type = spec["type"]
    return aov


def material_from_set_material_nodes(node_group):
    materials = []
    for node in node_group.nodes:
        if node.bl_idname != "GeometryNodeSetMaterial":
            continue
        material = node.inputs["Material"].default_value
        if material is not None:
            materials.append(material)
    return materials


def relevant_materials():
    seen = set()
    materials = []
    for group_name in TARGET_GROUP_NAMES:
        node_group = bpy.data.node_groups.get(group_name)
        if node_group is None:
            continue
        for material in material_from_set_material_nodes(node_group):
            if material.name in seen or not material.use_nodes or material.node_tree is None:
                continue
            seen.add(material.name)
            materials.append(material)
    return materials


def ensure_attribute_node(node_tree, spec, location):
    node_name = f"World AOV Attribute {spec['aov_name']}"
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
    node_name = f"World AOV Output {spec['aov_name']}"
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
    node_tree = material.node_tree
    nodes = node_tree.nodes
    base_x = max((node.location.x for node in nodes), default=400.0) + 250.0
    base_y = max((node.location.y for node in nodes), default=200.0)

    for index, spec in enumerate(WORLD_AOV_SPECS):
        y = base_y - (index * 180.0)
        attr_node = ensure_attribute_node(node_tree, spec, (base_x - 320.0, y))
        aov_node = ensure_aov_output_node(node_tree, spec, (base_x, y))
        ensure_link(node_tree, attr_node.outputs["Fac"], aov_node.inputs["Value"])


def patch_scenes():
    ensured = {}
    for scene_name in TARGET_SCENE_NAMES:
        scene = bpy.data.scenes.get(scene_name)
        if scene is None:
            continue
        ensured[scene.name] = {}
        for view_layer in scene.view_layers:
            for spec in WORLD_AOV_SPECS:
                ensure_aov(view_layer, spec)
            ensured[scene.name][view_layer.name] = [spec["aov_name"] for spec in WORLD_AOV_SPECS]
    return ensured


def main():
    ensured = patch_scenes()
    materials = relevant_materials()
    for material in materials:
        patch_material(material)

    print("World AOVs ensured:")
    for scene_name, view_layers in ensured.items():
        print(f"Scene: {scene_name}")
        for view_layer_name, aovs in view_layers.items():
            print(f"  {view_layer_name}: {aovs}")

    print(f"Materials patched: {[material.name for material in materials]}")

    if SAVE_FILE:
        bpy.ops.wm.save_mainfile()
        print("Saved current blend file")


if __name__ == "__main__":
    main()
