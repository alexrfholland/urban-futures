import bpy


TARGET_SCENE_NAME = "city-site"
SAVE_FILE = False


ENVELOPE_AOV_SPECS = [
    {"aov_name": "bioSimple", "attr_name": "scenario_bioEnvelope_simple_int", "attribute_type": "GEOMETRY"},
    {"aov_name": "bioEnvelopeType", "attr_name": "scenario_bioEnvelope_int", "attribute_type": "GEOMETRY"},
    {"aov_name": "sim_Turns", "attr_name": "sim_Turns", "attribute_type": "GEOMETRY"},
]

CITY_TREE_AOV_SPECS = [
    {"aov_name": "control", "attr_name": "control", "attribute_type": "INSTANCER"},
    {"aov_name": "node_type", "attr_name": "node_type", "attribute_type": "INSTANCER"},
]


def ensure_view_layer_aov(view_layer, aov_name, aov_type="VALUE"):
    for aov in view_layer.aovs:
        if aov.name == aov_name:
            return aov

    aov = view_layer.aovs.add()
    aov.name = aov_name
    aov.type = aov_type
    return aov


def find_or_create_attribute_node(node_tree, attribute_name, attribute_type, x, y):
    for node in node_tree.nodes:
        if node.type == "ATTRIBUTE" and node.attribute_name == attribute_name:
            if hasattr(node, "attribute_type"):
                node.attribute_type = attribute_type
            return node

    node = node_tree.nodes.new("ShaderNodeAttribute")
    node.attribute_name = attribute_name
    if hasattr(node, "attribute_type"):
        node.attribute_type = attribute_type
    node.location = (x, y)
    return node


def find_or_create_aov_node(node_tree, aov_name, x, y):
    for node in node_tree.nodes:
        if node.type == "OUTPUT_AOV" and node.aov_name == aov_name:
            return node

    node = node_tree.nodes.new("ShaderNodeOutputAOV")
    node.aov_name = aov_name
    node.name = f"AOV Output {aov_name}"
    node.location = (x, y)
    return node


def ensure_link(node_tree, from_socket, to_socket):
    for link in node_tree.links:
        if link.from_socket == from_socket and link.to_socket == to_socket:
            return
    node_tree.links.new(from_socket, to_socket)


def envelope_materials():
    matches = []
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue

        attrs = {
            node.attribute_name
            for node in material.node_tree.nodes
            if node.type == "ATTRIBUTE"
        }
        if "scenario_bioEnvelope_int" in attrs or "scenario_bioEnvelope_simple_int" in attrs:
            matches.append(material)
    return matches


def tree_materials():
    return [
        material
        for material in bpy.data.materials
        if material.use_nodes and material.node_tree is not None and material.name.startswith("Tree Resource")
    ]


def patch_material(material, specs):
    node_tree = material.node_tree

    for index, spec in enumerate(specs):
        y = 250 - (index * 180)
        attribute_node = find_or_create_attribute_node(
            node_tree,
            spec["attr_name"],
            spec["attribute_type"],
            -600,
            y,
        )
        aov_node = find_or_create_aov_node(node_tree, spec["aov_name"], -250, y)
        ensure_link(node_tree, attribute_node.outputs["Fac"], aov_node.inputs["Value"])


def patch_scene(scene):
    for view_layer in scene.view_layers:
        for spec in ENVELOPE_AOV_SPECS + CITY_TREE_AOV_SPECS:
            ensure_view_layer_aov(view_layer, spec["aov_name"])


def iter_scene_objects(scene):
    seen = set()

    def walk_collection(collection):
        for obj in collection.objects:
            if obj.name not in seen:
                seen.add(obj.name)
                yield obj
        for child in collection.children:
            yield from walk_collection(child)

    yield from walk_collection(scene.collection)


def ensure_constant_int_point_attribute(obj, attr_name, value):
    if obj.type != "MESH" or obj.data is None:
        return

    mesh = obj.data
    point_count = len(mesh.vertices)
    if point_count == 0:
        return

    attr = mesh.attributes.get(attr_name)
    if attr is None:
        attr = mesh.attributes.new(name=attr_name, type="INT", domain="POINT")

    attr.data.foreach_set("value", [value] * point_count)


def patch_city_point_clouds(scene):
    patched = []
    for obj in iter_scene_objects(scene):
        if obj.name.startswith("TreePositions"):
            ensure_constant_int_point_attribute(obj, "node_type", 0)
            patched.append(obj.name)
        elif obj.name.startswith("LogPositions"):
            ensure_constant_int_point_attribute(obj, "node_type", 2)
            patched.append(obj.name)
    return patched


def main():
    scene = bpy.data.scenes.get(TARGET_SCENE_NAME)
    if not scene:
        raise ValueError(
            f"Scene '{TARGET_SCENE_NAME}' not found. Run the city append script first or change TARGET_SCENE_NAME."
        )

    patch_scene(scene)

    patched_objects = patch_city_point_clouds(scene)

    patched_materials = []
    for material in envelope_materials():
        patch_material(material, ENVELOPE_AOV_SPECS)
        patched_materials.append(material.name)

    patched_tree_materials = []
    for material in tree_materials():
        patch_material(material, CITY_TREE_AOV_SPECS)
        patched_tree_materials.append(material.name)

    print(f"Patched scene: {scene.name}")
    print(f"Patched city point clouds: {patched_objects}")
    print(f"Patched envelope materials: {patched_materials}")
    print(f"Patched tree materials: {patched_tree_materials}")

    if SAVE_FILE:
        bpy.ops.wm.save_mainfile()
        print("Saved current blend file")


if __name__ == "__main__":
    main()
