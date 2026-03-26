import math
from pathlib import Path

import bpy


OUTPUT_BLEND = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6_hoddle-worldcubes.blend"
)
TARGET_GROUPS = (
    "Background.001 Cubes",
    "Background - Large pts.001 Cubes",
)
ROTATION_DEGREES_Z = -20.0
TARGET_OBJECTS = (
    "city_buildings.001_cubes",
    "city_highResRoad.001_cubes",
)


def ensure_link(node_tree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def rotate_group(node_group):
    nodes = node_group.nodes
    links = node_group.links

    cube_node = nodes.get("Cube Voxel")
    instance_node = nodes.get("Instance Cubes")
    if cube_node is None or instance_node is None:
        raise ValueError(f"Group '{node_group.name}' is missing Cube Voxel or Instance Cubes")

    transform_node = nodes.get("Hoddle Grid Rotate")
    if transform_node is None or transform_node.bl_idname != "GeometryNodeTransform":
        transform_node = nodes.new("GeometryNodeTransform")
    transform_node.name = "Hoddle Grid Rotate"
    transform_node.label = "Hoddle Grid Rotate"
    transform_node.location = (
        (cube_node.location.x + instance_node.location.x) / 2.0,
        cube_node.location.y,
    )
    transform_node.inputs["Rotation"].default_value[2] = math.radians(ROTATION_DEGREES_Z)

    ensure_link(node_group, cube_node.outputs["Mesh"], transform_node.inputs["Geometry"])
    ensure_link(node_group, transform_node.outputs["Geometry"], instance_node.inputs["Instance"])


def find_layer_collection(layer_collection, collection_name):
    if layer_collection.collection.name == collection_name:
        return layer_collection

    for child in layer_collection.children:
        match = find_layer_collection(child, collection_name)
        if match is not None:
            return match

    return None


def inspect_city_world_cube_visibility():
    scene = bpy.data.scenes.get("city")
    if scene is None:
        return []

    states = []
    for view_layer in scene.view_layers:
        layer_collection = find_layer_collection(view_layer.layer_collection, "City_WorldCubes")
        if layer_collection is None:
            continue
        states.append((view_layer.name, layer_collection.exclude))
    return states


def hide_target_objects_in_viewport():
    hidden = []
    for object_name in TARGET_OBJECTS:
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            continue
        obj.hide_viewport = True
        hidden.append(object_name)
    return hidden


def main():
    aligned = []
    for group_name in TARGET_GROUPS:
        node_group = bpy.data.node_groups.get(group_name)
        if node_group is None:
            print(f"Skipping missing node group: {group_name}")
            continue
        rotate_group(node_group)
        print(f"Aligned node group: {group_name} ({ROTATION_DEGREES_Z} deg Z)")
        aligned.append(group_name)

    if not aligned:
        raise ValueError("No target node groups were found to align.")

    hidden = hide_target_objects_in_viewport()

    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))
    print(f"Saved: {OUTPUT_BLEND}")
    print(f"Viewport-hidden objects: {hidden}")
    print(f"City_WorldCubes visibility: {inspect_city_world_cube_visibility()}")


if __name__ == "__main__":
    main()
