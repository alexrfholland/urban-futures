import bpy


SOURCE_GROUP_NAMES = (
    "Background",
    "Background.001",
    "Background - Large pts",
    "Background - Large pts.001",
)
TARGET_SPECS = (
    {
        "prefix": "trimmed-parade_base",
        "target_collection": "World-cubes",
    },
    {
        "prefix": "trimmed-parade_highResRoad",
        "target_collection": "World-cubes",
    },
    {
        "prefix": "city_buildings",
        "target_collection": "City_World-cubes",
    },
    {
        "prefix": "city_highResRoad",
        "target_collection": "City_World-cubes",
    },
)
GROUP_SUFFIX = " Cubes"
OBJECT_SUFFIX = "_cubes"
VOXEL_SIZE = (0.25, 0.25, 0.25)
CREATE_DUPLICATE_OBJECTS = True
CITY_WORLD_CUBE_VIEW_LAYER_EXCLUDES = {
    "pathway_state": False,
    "existing_condition": False,
    "city_priority": True,
    "city_bioenvelope": False,
    "trending_state": False,
}
WORLD_AOV_SPECS = (
    {
        "aov_name": "world_sim_turns",
        "attr_name": "sim_Turns",
        "attribute_type": "INSTANCER",
    },
    {
        "aov_name": "world_sim_nodes",
        "attr_name": "sim_Nodes",
        "attribute_type": "INSTANCER",
    },
    {
        "aov_name": "world_design_bioenvelope",
        "attr_name": "scenario_bioEnvelope",
        "attribute_type": "INSTANCER",
    },
    {
        "aov_name": "world_design_bioenvelope_simple",
        "attr_name": "scenario_bioEnvelopeSimple",
        "attribute_type": "INSTANCER",
    },
    {
        "aov_name": "world_sim_matched",
        "attr_name": "sim_Matched",
        "attribute_type": "INSTANCER",
    },
)


def collection_tree_contains(root_collection, target_collection):
    if root_collection == target_collection:
        return True

    for child in root_collection.children:
        if collection_tree_contains(child, target_collection):
            return True

    return False


def find_scenes_for_object(obj):
    scenes = []
    for scene in bpy.data.scenes:
        if any(collection_tree_contains(scene.collection, collection) for collection in obj.users_collection):
            scenes.append(scene)
    return scenes


def remove_group_if_present(name):
    node_group = bpy.data.node_groups.get(name)
    if node_group is not None:
        bpy.data.node_groups.remove(node_group)


def ensure_target_collection(collection_name, source_obj):
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        collection = bpy.data.collections.new(collection_name)

    scenes = find_scenes_for_object(source_obj)
    if not scenes and bpy.context.scene is not None:
        scenes = [bpy.context.scene]

    for scene in scenes:
        if scene.collection.children.get(collection.name) is None:
            scene.collection.children.link(collection)
        apply_collection_view_layer_state(scene, collection)

    return collection


def find_layer_collection(layer_collection, collection_name):
    if layer_collection.collection.name == collection_name:
        return layer_collection

    for child in layer_collection.children:
        match = find_layer_collection(child, collection_name)
        if match is not None:
            return match

    return None


def apply_collection_view_layer_state(scene, collection):
    if scene.name != "city" or collection.name != "City_WorldCubes":
        return

    for view_layer_name, exclude in CITY_WORLD_CUBE_VIEW_LAYER_EXCLUDES.items():
        view_layer = scene.view_layers.get(view_layer_name)
        if view_layer is None:
            continue
        layer_collection = find_layer_collection(view_layer.layer_collection, collection.name)
        if layer_collection is None:
            continue
        layer_collection.exclude = exclude


def duplicate_material_for_instances(material):
    if material is None:
        return None

    duplicate_name = f"{material.name}{GROUP_SUFFIX}"
    existing_duplicate = bpy.data.materials.get(duplicate_name)
    if existing_duplicate is not None:
        bpy.data.materials.remove(existing_duplicate, do_unlink=True)

    duplicate = material.copy()
    duplicate.name = duplicate_name

    if duplicate.node_tree is not None:
        for node in duplicate.node_tree.nodes:
            if node.bl_idname == "ShaderNodeAttribute":
                node.attribute_type = "INSTANCER"
        ensure_world_aov_outputs(duplicate)

    return duplicate


def ensure_link(node_tree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


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


def ensure_world_aov_outputs(material):
    node_tree = material.node_tree
    nodes = node_tree.nodes
    base_x = max((node.location.x for node in nodes), default=400.0) + 250.0
    base_y = max((node.location.y for node in nodes), default=200.0)

    for index, spec in enumerate(WORLD_AOV_SPECS):
        y = base_y - (index * 180.0)
        attr_node = ensure_attribute_node(node_tree, spec, (base_x - 320.0, y))
        aov_node = ensure_aov_output_node(node_tree, spec, (base_x, y))
        ensure_link(node_tree, attr_node.outputs["Fac"], aov_node.inputs["Value"])


def get_upstream_points_socket(set_material):
    geometry_input = set_material.inputs.get("Geometry")
    if geometry_input is None or not geometry_input.links:
        return None

    source_socket = geometry_input.links[0].from_socket
    source_node = source_socket.node

    if source_node.bl_idname == "GeometryNodeSetPointRadius":
        points_input = source_node.inputs.get("Points")
        if points_input and points_input.links:
            return points_input.links[0].from_socket

    return source_socket


def remove_node_if_orphaned(node_group, node):
    if node is None:
        return
    if node.inputs and any(socket.links for socket in node.inputs):
        return
    if node.outputs and any(socket.links for socket in node.outputs):
        return
    node_group.nodes.remove(node)


def build_cube_group(source_group_name):
    source_group = bpy.data.node_groups.get(source_group_name)
    if source_group is None:
        print(f"Skipping missing source group: {source_group_name}")
        return None

    new_group_name = f"{source_group_name}{GROUP_SUFFIX}"
    remove_group_if_present(new_group_name)

    new_group = source_group.copy()
    new_group.name = new_group_name

    nodes = new_group.nodes
    links = new_group.links

    set_material = next(
        (node for node in nodes if node.bl_idname == "GeometryNodeSetMaterial"),
        None,
    )
    if set_material is None:
        print(f"Skipping {source_group_name}: no Set Material node found")
        return None

    upstream_socket = get_upstream_points_socket(set_material)
    if upstream_socket is None:
        print(f"Skipping {source_group_name}: no upstream points geometry found")
        return None

    downstream_sockets = [link.to_socket for link in set_material.outputs["Geometry"].links]
    original_material = set_material.inputs["Material"].default_value
    duplicate_material = duplicate_material_for_instances(original_material)

    old_links = list(set_material.inputs["Geometry"].links) + list(set_material.outputs["Geometry"].links)
    for link in old_links:
        links.remove(link)

    cube_node = nodes.new("GeometryNodeMeshCube")
    cube_node.name = "Cube Voxel"
    cube_node.location = (set_material.location.x - 520, set_material.location.y - 120)
    cube_node.inputs["Size"].default_value = VOXEL_SIZE

    instance_node = nodes.new("GeometryNodeInstanceOnPoints")
    instance_node.name = "Instance Cubes"
    instance_node.location = (set_material.location.x - 260, set_material.location.y)
    instance_node.inputs["Pick Instance"].default_value = False

    links.new(upstream_socket, instance_node.inputs["Points"])
    links.new(cube_node.outputs["Mesh"], instance_node.inputs["Instance"])
    links.new(instance_node.outputs["Instances"], set_material.inputs["Geometry"])

    if duplicate_material is not None:
        set_material.inputs["Material"].default_value = duplicate_material

    for socket in downstream_sockets:
        links.new(set_material.outputs["Geometry"], socket)

    point_radius_node = next(
        (node for node in nodes if node.bl_idname == "GeometryNodeSetPointRadius"),
        None,
    )
    value_node = next(
        (node for node in nodes if node.bl_idname == "ShaderNodeValue"),
        None,
    )
    remove_node_if_orphaned(new_group, point_radius_node)
    remove_node_if_orphaned(new_group, value_node)

    print(f"Created cube-instancing group: {new_group.name}")
    return new_group


def build_group_map():
    group_map = {}
    for source_group_name in SOURCE_GROUP_NAMES:
        new_group = build_cube_group(source_group_name)
        if new_group is not None:
            group_map[source_group_name] = new_group
    return group_map


def get_target_spec(obj):
    for spec in TARGET_SPECS:
        if obj.name.startswith(spec["prefix"]):
            return spec
    return None


def duplicate_with_cube_group(obj, replacement_group, target_collection_name):
    duplicate_name = f"{obj.name}{OBJECT_SUFFIX}"
    existing_duplicate = bpy.data.objects.get(duplicate_name)
    if existing_duplicate is not None:
        bpy.data.objects.remove(existing_duplicate, do_unlink=True)

    duplicate = obj.copy()
    duplicate.data = obj.data
    duplicate.animation_data_clear()
    duplicate.name = duplicate_name

    target_collection = ensure_target_collection(target_collection_name, obj)
    target_collection.objects.link(duplicate)

    for modifier in duplicate.modifiers:
        if modifier.type == "NODES" and modifier.node_group is not None:
            modifier.node_group = replacement_group
            break

    return duplicate.name


def create_duplicate_objects(group_map):
    created = []

    for obj in bpy.data.objects:
        if obj.name.endswith(OBJECT_SUFFIX):
            continue

        target_spec = get_target_spec(obj)
        if target_spec is None:
            continue

        for modifier in obj.modifiers:
            if modifier.type != "NODES" or modifier.node_group is None:
                continue

            replacement_group = group_map.get(modifier.node_group.name)
            if replacement_group is None:
                continue

            duplicate_name = duplicate_with_cube_group(
                obj,
                replacement_group,
                target_spec["target_collection"],
            )
            created.append((obj.name, duplicate_name, replacement_group.name, target_spec["target_collection"]))
            break

    return created


def main():
    group_map = build_group_map()
    print(f"Built cube groups: {sorted(group_map)}")

    created_objects = []
    if CREATE_DUPLICATE_OBJECTS:
        created_objects = create_duplicate_objects(group_map)

    print(f"Voxel size: {VOXEL_SIZE}")
    print(f"Created duplicate objects: {created_objects}")


if __name__ == "__main__":
    main()
