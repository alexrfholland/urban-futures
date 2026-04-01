import bpy
from mathutils import Vector
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract
import b2026_timeline_runtime_flags as runtime_flags


CLIP_NODE_NAME = "Clip Box Cull"
POST_INSTANCE_CLIP_NODE_NAME = "Post Instance Clip Box Cull"
REALIZE_INSTANCES_NODE_NAME = "Realize Instances For Clip"
ENABLE_CLIPBOX_PATCHING = True
DEFAULT_CLIP_BOX_SCALE = (40.0, 40.0, 20.0)
TREE_CLIP_BUFFER_XY = 5.0
TREE_CLIP_BUFFER_Z = 0.0
CITY_CLIP_BOX_NAME = "City_ClipBox"
CITY_CLIP_BOX_Z_PADDING = 10.0
CITY_CLIP_BOX_XY_PADDING = 10.0
PARADE_CLIP_BOX_Z_PADDING = 5.0
PARADE_CLIP_BOX_XY_PADDING = 5.0
PARADE_POINT_OBJECT_PREFIXES = (
    "trimmed-parade_base",
    "trimmed-parade_highResRoad",
)
CITY_POINT_OBJECT_PREFIXES = (
    "city_buildings",
    "city_highResRoad",
)
PARADE_GROUND_OBJECT_PREFIXES = (
    "trimmed-parade_180_rewilded",
    "trimmed-parade_positive_1_envelope_scenarioYR",
)
CITY_GROUND_OBJECT_PREFIXES = (
    "city_1_envelope_scenarioYR",
    "city_1_envelopes_scenarioYR",
)
PARADE_TREE_OBJECT_PREFIXES = ("TreePositions_trimmed-parade_", "LogPositions_trimmed-parade_")
CITY_TREE_OBJECT_PREFIXES = ("TreePositions_city_", "LogPositions_city_")
PARADE_TREE_TEMPLATE_GROUP_NAMES = ("instance_template",)
CITY_TREE_TEMPLATE_GROUP_NAMES = ()
UNI_POINT_OBJECT_PREFIXES = (
    "uni_base",
    "uni_highResRoad",
)
UNI_GROUND_OBJECT_PREFIXES = (
    "uni_positive_1_envelope_scenarioYR",
    "uni_trending_1_envelope_scenarioYR",
)
UNI_TREE_OBJECT_PREFIXES = (
    "TreePositions_uni_",
    "LogPositions_uni_",
    "PolePositions_uni_",
)
UNI_TREE_TEMPLATE_GROUP_NAMES = ("instance_template",)


SITE_SPECS = (
    {
        "label": "parade",
        "clip_box_name": "ClipBox",
        "point_clip_group_name": "ClipBox Cull Points",
        "mesh_clip_group_name": "ClipBox Cull Mesh",
        "tree_clip_group_name": "ClipBox Cull Trees",
        "point_object_prefixes": PARADE_POINT_OBJECT_PREFIXES,
        "ground_object_prefixes": PARADE_GROUND_OBJECT_PREFIXES,
        "tree_object_prefixes": PARADE_TREE_OBJECT_PREFIXES,
        "tree_template_group_names": PARADE_TREE_TEMPLATE_GROUP_NAMES,
        "bbox_object_prefixes": PARADE_POINT_OBJECT_PREFIXES + PARADE_GROUND_OBJECT_PREFIXES,
        "collection_name": "Parade_Manager",
        "inherit_xy_from": None,
        "xy_padding": PARADE_CLIP_BOX_XY_PADDING,
        "z_padding": PARADE_CLIP_BOX_Z_PADDING,
    },
    {
        "label": "city",
        "clip_box_name": CITY_CLIP_BOX_NAME,
        "point_clip_group_name": "City ClipBox Cull Points",
        "mesh_clip_group_name": "City ClipBox Cull Mesh",
        "tree_clip_group_name": "City ClipBox Cull Trees",
        "point_object_prefixes": CITY_POINT_OBJECT_PREFIXES,
        "ground_object_prefixes": CITY_GROUND_OBJECT_PREFIXES,
        "tree_object_prefixes": CITY_TREE_OBJECT_PREFIXES,
        "tree_template_group_names": CITY_TREE_TEMPLATE_GROUP_NAMES,
        "collection_name": "City_Manager",
        "inherit_xy_from": None,
        "xy_padding": CITY_CLIP_BOX_XY_PADDING,
        "z_padding": CITY_CLIP_BOX_Z_PADDING,
    },
    {
        "label": "uni",
        "clip_box_name": scene_contract.SITE_CONTRACTS["uni"]["live_clip_object"],
        "point_clip_group_name": "Uni ClipBox Cull Points",
        "mesh_clip_group_name": "Uni ClipBox Cull Mesh",
        "tree_clip_group_name": "Uni ClipBox Cull Trees",
        "point_object_prefixes": UNI_POINT_OBJECT_PREFIXES,
        "ground_object_prefixes": UNI_GROUND_OBJECT_PREFIXES,
        "tree_object_prefixes": UNI_TREE_OBJECT_PREFIXES,
        "tree_template_group_names": UNI_TREE_TEMPLATE_GROUP_NAMES,
        "collection_name": scene_contract.get_collection_name("uni", "manager", legacy=True),
        "inherit_xy_from": None,
        "xy_padding": CITY_CLIP_BOX_XY_PADDING,
        "z_padding": CITY_CLIP_BOX_Z_PADDING,
    },
)


def iter_target_objects(object_prefixes):
    for obj in bpy.data.objects:
        if obj.name.endswith("_cubes"):
            continue
        if any(obj.name.startswith(prefix) for prefix in object_prefixes):
            yield obj


def compute_world_bbox(object_prefixes):
    min_corner = [float("inf"), float("inf"), float("inf")]
    max_corner = [float("-inf"), float("-inf"), float("-inf")]
    found = False

    for obj in iter_target_objects(object_prefixes):
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                min_corner[axis] = min(min_corner[axis], world_corner[axis])
                max_corner[axis] = max(max_corner[axis], world_corner[axis])
        found = True

    if not found:
        return None

    return tuple(min_corner), tuple(max_corner)


def get_bbox_prefixes(spec):
    explicit_prefixes = spec.get("bbox_object_prefixes")
    if explicit_prefixes is not None:
        return tuple(explicit_prefixes)
    return (
        tuple(spec["point_object_prefixes"]) +
        tuple(spec["ground_object_prefixes"]) +
        tuple(spec["tree_object_prefixes"])
    )


def get_collection(collection_name):
    if not collection_name:
        return None
    return bpy.data.collections.get(collection_name)


def move_object_to_collection(obj, collection_name):
    collection = get_collection(collection_name)
    if collection is None:
        return

    if collection not in obj.users_collection:
        collection.objects.link(obj)

    for existing_collection in list(obj.users_collection):
        if existing_collection != collection:
            existing_collection.objects.unlink(obj)


def resolve_clip_box_transform(spec):
    clip_box_name = spec["clip_box_name"]
    inherited_box = bpy.data.objects.get(spec["inherit_xy_from"]) if spec["inherit_xy_from"] else None
    bbox = compute_world_bbox(get_bbox_prefixes(spec))
    if bbox is None:
        if inherited_box is not None:
            return inherited_box.location.copy(), tuple(inherited_box.rotation_euler), tuple(inherited_box.scale)
        scene = bpy.context.scene
        cursor_location = scene.cursor.location.copy() if scene else (0.0, 0.0, 0.0)
        return cursor_location, (0.0, 0.0, 0.0), DEFAULT_CLIP_BOX_SCALE

    bbox_min, bbox_max = bbox
    location = (
        (bbox_min[0] + bbox_max[0]) / 2.0,
        (bbox_min[1] + bbox_max[1]) / 2.0,
        (bbox_min[2] + bbox_max[2]) / 2.0,
    )
    width_x = bbox_max[0] - bbox_min[0]
    width_y = bbox_max[1] - bbox_min[1]
    height = bbox_max[2] - bbox_min[2]

    if inherited_box is not None:
        rotation = tuple(inherited_box.rotation_euler)
        scale_x = inherited_box.scale.x
        scale_y = inherited_box.scale.y
        base_scale_z = inherited_box.scale.z
    else:
        rotation = (0.0, 0.0, 0.0)
        scale_x, scale_y, base_scale_z = DEFAULT_CLIP_BOX_SCALE
        xy_padding = spec.get("xy_padding", 0.0)
        scale_x = max(scale_x, width_x / 2.0 + xy_padding)
        scale_y = max(scale_y, width_y / 2.0 + xy_padding)

    z_padding = spec.get("z_padding", 0.0)
    scale_z = max(base_scale_z, height / 2.0 + z_padding)
    return location, rotation, (scale_x, scale_y, scale_z)


def ensure_clip_box(spec):
    clip_box_name = spec["clip_box_name"]
    location, rotation, scale = resolve_clip_box_transform(spec)
    clip_box = bpy.data.objects.get(clip_box_name)
    if clip_box is not None:
        clip_box.location = location
        clip_box.rotation_euler = rotation
        clip_box.scale = scale
        clip_box.display_type = "BOUNDS"
        clip_box.hide_render = True
        clip_box.show_name = True
        clip_box.show_in_front = True
        move_object_to_collection(clip_box, spec["collection_name"])
        print(
            f"Updated {clip_box_name} to {tuple(round(v, 3) for v in clip_box.location)} "
            f"with scale {tuple(round(v, 3) for v in clip_box.scale)}"
        )
        return clip_box

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location, rotation=rotation)
    clip_box = bpy.context.active_object
    clip_box.name = clip_box_name
    clip_box.display_type = "BOUNDS"
    clip_box.hide_render = True
    clip_box.show_name = True
    clip_box.show_in_front = True
    clip_box.scale = scale
    move_object_to_collection(clip_box, spec["collection_name"])

    print(
        f"Created {clip_box_name} at {tuple(round(v, 3) for v in clip_box.location)} "
        f"with scale {tuple(round(v, 3) for v in clip_box.scale)}"
    )
    return clip_box


def new_geometry_socket(node_group, name, in_out):
    return node_group.interface.new_socket(
        name=name,
        in_out=in_out,
        socket_type="NodeSocketGeometry",
    )


def make_compare_node(node_group, operation, location):
    node = node_group.nodes.new("FunctionNodeCompare")
    node.data_type = "FLOAT"
    node.operation = operation
    node.location = location
    return node


def build_clip_group(name, clip_box, delete_domain, xy_buffer=0.0, z_buffer=0.0):
    existing = bpy.data.node_groups.get(name)
    if existing is not None:
        bpy.data.node_groups.remove(existing)

    node_group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    new_geometry_socket(node_group, "Geometry", "INPUT")
    new_geometry_socket(node_group, "Geometry", "OUTPUT")

    nodes = node_group.nodes
    links = node_group.links

    group_input = nodes.new("NodeGroupInput")
    group_input.location = (-980, 0)

    group_output = nodes.new("NodeGroupOutput")
    group_output.location = (760, 0)

    transform_info = nodes.new("GeometryNodeObjectInfo")
    transform_info.location = (-1200, -140)
    transform_info.transform_space = "RELATIVE"
    transform_info.inputs["Object"].default_value = clip_box
    transform_info.inputs["As Instance"].default_value = False

    geometry_info = nodes.new("GeometryNodeObjectInfo")
    geometry_info.location = (-1200, -440)
    geometry_info.transform_space = "ORIGINAL"
    geometry_info.inputs["Object"].default_value = clip_box
    geometry_info.inputs["As Instance"].default_value = False

    bound_box = nodes.new("GeometryNodeBoundBox")
    bound_box.location = (-980, -440)

    position = nodes.new("GeometryNodeInputPosition")
    position.location = (-980, 180)

    offset = nodes.new("ShaderNodeVectorMath")
    offset.operation = "SUBTRACT"
    offset.location = (-760, 180)

    inverse_rotate = nodes.new("ShaderNodeVectorRotate")
    inverse_rotate.rotation_type = "EULER_XYZ"
    inverse_rotate.invert = True
    inverse_rotate.location = (-540, 180)

    unscale = nodes.new("ShaderNodeVectorMath")
    unscale.operation = "DIVIDE"
    unscale.location = (-320, 180)

    separate_position = nodes.new("ShaderNodeSeparateXYZ")
    separate_position.location = (-100, 180)

    buffer_vector = nodes.new("ShaderNodeCombineXYZ")
    buffer_vector.location = (-980, -20)
    buffer_vector.inputs["X"].default_value = xy_buffer
    buffer_vector.inputs["Y"].default_value = xy_buffer
    buffer_vector.inputs["Z"].default_value = z_buffer

    local_buffer = nodes.new("ShaderNodeVectorMath")
    local_buffer.operation = "DIVIDE"
    local_buffer.location = (-980, -180)

    inset_min = nodes.new("ShaderNodeVectorMath")
    inset_min.operation = "ADD"
    inset_min.location = (-760, -260)

    inset_max = nodes.new("ShaderNodeVectorMath")
    inset_max.operation = "SUBTRACT"
    inset_max.location = (-760, -500)

    separate_min = nodes.new("ShaderNodeSeparateXYZ")
    separate_min.location = (-540, -260)

    separate_max = nodes.new("ShaderNodeSeparateXYZ")
    separate_max.location = (-540, -500)

    x_min = make_compare_node(node_group, "GREATER_EQUAL", (120, 260))
    x_max = make_compare_node(node_group, "LESS_EQUAL", (120, 160))
    y_min = make_compare_node(node_group, "GREATER_EQUAL", (120, 20))
    y_max = make_compare_node(node_group, "LESS_EQUAL", (120, -80))
    z_min = make_compare_node(node_group, "GREATER_EQUAL", (120, -220))
    z_max = make_compare_node(node_group, "LESS_EQUAL", (120, -320))

    and_x = nodes.new("FunctionNodeBooleanMath")
    and_x.operation = "AND"
    and_x.location = (360, 210)

    and_y = nodes.new("FunctionNodeBooleanMath")
    and_y.operation = "AND"
    and_y.location = (360, -30)

    and_z = nodes.new("FunctionNodeBooleanMath")
    and_z.operation = "AND"
    and_z.location = (360, -270)

    and_xy = nodes.new("FunctionNodeBooleanMath")
    and_xy.operation = "AND"
    and_xy.location = (580, 90)

    and_xyz = nodes.new("FunctionNodeBooleanMath")
    and_xyz.operation = "AND"
    and_xyz.location = (800, 0)

    outside = nodes.new("FunctionNodeBooleanMath")
    outside.operation = "NOT"
    outside.location = (1000, 0)

    delete_geometry = nodes.new("GeometryNodeDeleteGeometry")
    delete_geometry.domain = delete_domain
    delete_geometry.location = (1000, 180)

    links.new(geometry_info.outputs["Geometry"], bound_box.inputs["Geometry"])
    links.new(position.outputs["Position"], offset.inputs[0])
    links.new(transform_info.outputs["Location"], offset.inputs[1])
    links.new(offset.outputs["Vector"], inverse_rotate.inputs["Vector"])
    links.new(transform_info.outputs["Rotation"], inverse_rotate.inputs["Rotation"])
    links.new(inverse_rotate.outputs["Vector"], unscale.inputs[0])
    links.new(transform_info.outputs["Scale"], unscale.inputs[1])
    links.new(unscale.outputs["Vector"], separate_position.inputs["Vector"])
    links.new(buffer_vector.outputs["Vector"], local_buffer.inputs[0])
    links.new(transform_info.outputs["Scale"], local_buffer.inputs[1])
    links.new(bound_box.outputs["Min"], inset_min.inputs[0])
    links.new(local_buffer.outputs["Vector"], inset_min.inputs[1])
    links.new(bound_box.outputs["Max"], inset_max.inputs[0])
    links.new(local_buffer.outputs["Vector"], inset_max.inputs[1])
    links.new(inset_min.outputs["Vector"], separate_min.inputs["Vector"])
    links.new(inset_max.outputs["Vector"], separate_max.inputs["Vector"])

    links.new(separate_position.outputs["X"], x_min.inputs[0])
    links.new(separate_min.outputs["X"], x_min.inputs[1])
    links.new(separate_position.outputs["X"], x_max.inputs[0])
    links.new(separate_max.outputs["X"], x_max.inputs[1])
    links.new(separate_position.outputs["Y"], y_min.inputs[0])
    links.new(separate_min.outputs["Y"], y_min.inputs[1])
    links.new(separate_position.outputs["Y"], y_max.inputs[0])
    links.new(separate_max.outputs["Y"], y_max.inputs[1])
    links.new(separate_position.outputs["Z"], z_min.inputs[0])
    links.new(separate_min.outputs["Z"], z_min.inputs[1])
    links.new(separate_position.outputs["Z"], z_max.inputs[0])
    links.new(separate_max.outputs["Z"], z_max.inputs[1])

    links.new(x_min.outputs["Result"], and_x.inputs[0])
    links.new(x_max.outputs["Result"], and_x.inputs[1])
    links.new(y_min.outputs["Result"], and_y.inputs[0])
    links.new(y_max.outputs["Result"], and_y.inputs[1])
    links.new(z_min.outputs["Result"], and_z.inputs[0])
    links.new(z_max.outputs["Result"], and_z.inputs[1])
    links.new(and_x.outputs["Boolean"], and_xy.inputs[0])
    links.new(and_y.outputs["Boolean"], and_xy.inputs[1])
    links.new(and_xy.outputs["Boolean"], and_xyz.inputs[0])
    links.new(and_z.outputs["Boolean"], and_xyz.inputs[1])
    links.new(and_xyz.outputs["Boolean"], outside.inputs[0])

    links.new(group_input.outputs["Geometry"], delete_geometry.inputs["Geometry"])
    links.new(outside.outputs["Boolean"], delete_geometry.inputs["Selection"])
    links.new(delete_geometry.outputs["Geometry"], group_output.inputs["Geometry"])

    return node_group


def ensure_clip_groups(spec, clip_box):
    point_group = build_clip_group(
        spec["point_clip_group_name"],
        clip_box,
        delete_domain="POINT",
    )
    mesh_group = build_clip_group(
        spec["mesh_clip_group_name"],
        clip_box,
        delete_domain="FACE",
    )
    tree_group = build_clip_group(
        spec["tree_clip_group_name"],
        clip_box,
        delete_domain="POINT",
        xy_buffer=TREE_CLIP_BUFFER_XY,
        z_buffer=TREE_CLIP_BUFFER_Z,
    )
    return point_group, mesh_group, tree_group


def ensure_clip_node(node_group, clip_group, location):
    return ensure_named_clip_node(node_group, CLIP_NODE_NAME, clip_group, location)


def ensure_named_clip_node(node_group, node_name, clip_group, location):
    clip_node = node_group.nodes.get(node_name)
    if clip_node is None or clip_node.bl_idname != "GeometryNodeGroup":
        if clip_node is not None:
            node_group.nodes.remove(clip_node)
        clip_node = node_group.nodes.new("GeometryNodeGroup")
        clip_node.name = node_name

    clip_node.node_tree = clip_group
    clip_node.location = location
    return clip_node


def bypass_geometry_node(node_group, node):
    if node is None:
        return False

    input_socket = node.inputs.get("Geometry") if hasattr(node, "inputs") else None
    output_socket = node.outputs.get("Geometry") if hasattr(node, "outputs") else None
    source_socket = None
    if input_socket is not None and input_socket.links:
        source_socket = input_socket.links[0].from_socket

    downstream_sockets = []
    if output_socket is not None:
        downstream_sockets = [link.to_socket for link in list(output_socket.links)]

    for link in list(node_group.links):
        if link.from_node == node or link.to_node == node:
            node_group.links.remove(link)

    if source_socket is not None:
        for socket in downstream_sockets:
            node_group.links.new(source_socket, socket)

    node_group.nodes.remove(node)
    return True


def unpatch_node_group(node_group):
    changed = False

    post_instance_clip = node_group.nodes.get(POST_INSTANCE_CLIP_NODE_NAME)
    if post_instance_clip is not None:
        changed = bypass_geometry_node(node_group, post_instance_clip) or changed

    realize_node = node_group.nodes.get(REALIZE_INSTANCES_NODE_NAME)
    if realize_node is not None:
        changed = bypass_geometry_node(node_group, realize_node) or changed

    clip_node = node_group.nodes.get(CLIP_NODE_NAME)
    if clip_node is not None:
        changed = bypass_geometry_node(node_group, clip_node) or changed

    return changed


def insert_clip_after_socket(node_group, source_socket, clip_group, location):
    clip_node = ensure_clip_node(node_group, clip_group, location)

    downstream_sockets = []

    for link in list(node_group.links):
        if link.to_socket == clip_node.inputs["Geometry"]:
            node_group.links.remove(link)

    if "Geometry" in clip_node.outputs:
        for link in list(node_group.links):
            if link.from_socket == clip_node.outputs["Geometry"]:
                downstream_sockets.append(link.to_socket)

    for link in list(node_group.links):
        if link.from_socket == source_socket and link.to_node != clip_node:
            downstream_sockets.append(link.to_socket)

    deduped = []
    seen = set()
    for socket in downstream_sockets:
        key = (socket.node.name, socket.name)
        if key not in seen:
            seen.add(key)
            deduped.append(socket)

    for link in list(node_group.links):
        if link.from_socket == source_socket:
            node_group.links.remove(link)
        elif "Geometry" in clip_node.outputs and link.from_socket == clip_node.outputs["Geometry"]:
            node_group.links.remove(link)

    node_group.links.new(source_socket, clip_node.inputs["Geometry"])
    for socket in deduped:
        node_group.links.new(clip_node.outputs["Geometry"], socket)

    return True


def patch_after_group_input(node_group, clip_group):
    group_input = next(
        (node for node in node_group.nodes if node.bl_idname == "NodeGroupInput"),
        None,
    )
    if group_input is None or "Geometry" not in group_input.outputs:
        return False

    return insert_clip_after_socket(
        node_group,
        group_input.outputs["Geometry"],
        clip_group,
        location=(-340, 0),
    )


def patch_after_mesh_to_points(node_group, clip_group):
    mesh_to_points = next(
        (node for node in node_group.nodes if node.bl_idname == "GeometryNodeMeshToPoints"),
        None,
    )
    if mesh_to_points is None or "Points" not in mesh_to_points.outputs:
        return False

    return insert_clip_after_socket(
        node_group,
        mesh_to_points.outputs["Points"],
        clip_group,
        location=(mesh_to_points.location.x + 220, mesh_to_points.location.y),
    )


def patch_after_instance_on_points_with_realize(node_group, clip_group):
    instance_on_points = next(
        (node for node in node_group.nodes if node.bl_idname == "GeometryNodeInstanceOnPoints"),
        None,
    )
    if instance_on_points is None or "Instances" not in instance_on_points.outputs:
        return False

    realize_node = node_group.nodes.get(REALIZE_INSTANCES_NODE_NAME)
    if realize_node is None or realize_node.bl_idname != "GeometryNodeRealizeInstances":
        if realize_node is not None:
            node_group.nodes.remove(realize_node)
        realize_node = node_group.nodes.new("GeometryNodeRealizeInstances")
        realize_node.name = REALIZE_INSTANCES_NODE_NAME
    realize_node.location = (instance_on_points.location.x + 220, instance_on_points.location.y)

    clip_node = ensure_named_clip_node(
        node_group,
        POST_INSTANCE_CLIP_NODE_NAME,
        clip_group,
        location=(realize_node.location.x + 220, realize_node.location.y),
    )

    downstream_sockets = []
    for link in list(node_group.links):
        if link.from_socket == instance_on_points.outputs["Instances"]:
            downstream_sockets.append(link.to_socket)
            node_group.links.remove(link)
        elif link.from_socket == realize_node.outputs["Geometry"]:
            downstream_sockets.append(link.to_socket)
            node_group.links.remove(link)
        elif link.from_socket == clip_node.outputs["Geometry"]:
            downstream_sockets.append(link.to_socket)
            node_group.links.remove(link)

    deduped = []
    seen = set()
    for socket in downstream_sockets:
        key = (socket.node.name, socket.name)
        if key not in seen:
            seen.add(key)
            deduped.append(socket)

    node_group.links.new(instance_on_points.outputs["Instances"], realize_node.inputs["Geometry"])
    node_group.links.new(realize_node.outputs["Geometry"], clip_node.inputs["Geometry"])
    for socket in deduped:
        node_group.links.new(clip_node.outputs["Geometry"], socket)

    return True


def collect_target_node_groups(object_prefixes):
    node_groups = set()
    targets = []

    for obj in bpy.data.objects:
        if not any(obj.name.startswith(prefix) for prefix in object_prefixes):
            continue

        for modifier in obj.modifiers:
            if modifier.type != "NODES" or modifier.node_group is None:
                continue

            node_groups.add(modifier.node_group)
            targets.append((obj.name, modifier.node_group.name))

    return node_groups, sorted(targets)


def collect_named_node_groups(group_names):
    node_groups = set()
    targets = []

    for group_name in group_names:
        node_group = bpy.data.node_groups.get(group_name)
        if node_group is None:
            continue

        node_groups.add(node_group)
        targets.append(node_group.name)

    return node_groups, sorted(targets)


def unpatch_scene_groups(spec):
    point_groups, _point_targets = collect_target_node_groups(spec["point_object_prefixes"])
    mesh_groups, _mesh_targets = collect_target_node_groups(spec["ground_object_prefixes"])
    tree_object_groups, _tree_object_targets = collect_target_node_groups(spec["tree_object_prefixes"])
    tree_template_groups, _tree_template_targets = collect_named_node_groups(spec["tree_template_group_names"])

    all_groups = point_groups | mesh_groups | tree_object_groups | tree_template_groups
    changed = []
    for node_group in sorted(all_groups, key=lambda item: item.name):
        if unpatch_node_group(node_group):
            changed.append(node_group.name)
    return changed


def patch_scene_groups(spec, point_group, mesh_group, tree_group):
    if not ENABLE_CLIPBOX_PATCHING:
        print(f"Clipbox patching disabled for {spec['label']}")
        return [], [], []

    point_groups, point_targets = collect_target_node_groups(spec["point_object_prefixes"])
    mesh_groups, mesh_targets = collect_target_node_groups(spec["ground_object_prefixes"])
    tree_object_groups, tree_object_targets = collect_target_node_groups(spec["tree_object_prefixes"])
    tree_template_groups, tree_template_targets = collect_named_node_groups(spec["tree_template_group_names"])

    patched_points = []
    patched_meshes = []
    patched_trees = []

    print(f"Point-style targets: {point_targets}")
    print(f"Mesh-style targets: {mesh_targets}")
    print(f"Tree object targets: {tree_object_targets}")
    print(f"Tree template targets: {tree_template_targets}")

    for node_group in sorted(point_groups, key=lambda item: item.name):
        if patch_after_mesh_to_points(node_group, point_group):
            patched_points.append(node_group.name)

    for node_group in sorted(mesh_groups, key=lambda item: item.name):
        if patch_after_group_input(node_group, mesh_group):
            patched_meshes.append(node_group.name)

    all_tree_groups = tree_template_groups | tree_object_groups
    for node_group in sorted(all_tree_groups, key=lambda item: item.name):
        if patch_after_group_input(node_group, tree_group):
            patched_trees.append(node_group.name)

    return patched_points, patched_meshes, patched_trees


def main():
    if not runtime_flags.ENABLE_CLIPBOX_SETUP:
        changed_groups = []
        for spec in SITE_SPECS:
            changed_groups.extend(unpatch_scene_groups(spec))
        print(f"Clip box setup disabled by runtime flags; unpatched groups: {sorted(set(changed_groups))}")
        return

    print(f"Tree clip buffer XY: {TREE_CLIP_BUFFER_XY}")
    print(f"Tree clip buffer Z: {TREE_CLIP_BUFFER_Z}")

    for spec in SITE_SPECS:
        clip_box = ensure_clip_box(spec)
        point_group, mesh_group, tree_group = ensure_clip_groups(spec, clip_box)
        patched_points, patched_meshes, patched_trees = patch_scene_groups(spec, point_group, mesh_group, tree_group)

        print(f"\n{spec['label']} clip setup complete")
        print(f"Clip box: {clip_box.name}")
        print(f"Point-style groups patched: {patched_points}")
        print(f"Mesh-style groups patched: {patched_meshes}")
        print(f"Tree groups patched: {patched_trees}")


if __name__ == "__main__":
    main()
