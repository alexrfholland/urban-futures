"""Build the bV2 template world-source payload from canonical world PLYs.

Scope of this template setup stage:
- import site/building PLYs
- import road PLYs
- split roads into hires and lores by scale/size
- optionally build legacy point-only roads from highres road PLYs
- assign canonical object names, collection placement, pass indices
- rebuild the canonical world material / GN names with the working debug setup

This version keeps the simple `Col -> Emission` display path, but also
recreates the existing bV2 world AOV contract without importing the production
material. The AOV contract is hard-coded from inspection so the script remains
small and explicit.

Usage:
  blender --background bv2_debug_template.blend --python bV2_setup_template.py -- --reset trimmed-parade
  blender --background bv2_debug_template.blend --python bV2_setup_template.py -- --reset city trimmed-parade uni --output-blend D:\\path\\to\\copy.blend
  blender --background bv2_debug_template.blend --python bV2_setup_template.py -- --reset trimmed-parade --use-points
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np


REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "urban-futures":
    REPO_ROOT = REPO_ROOT.parent

PLY_DIR = REPO_ROOT / "_data-refactored" / "model_inputs" / "world" / "originals"
HIGHRES_PLY_DIR = REPO_ROOT / "_data-refactored" / "model_inputs" / "world" / "highres"
DEBUG_OUTPUT_DIR = REPO_ROOT / "_data-refactored" / "blenderv2" / "output" / "_debug"
INPUT_TEMPLATE_PATH = DEBUG_OUTPUT_DIR / "bv2_debug_template.blend"
DEFAULT_OUTPUT_BLEND = INPUT_TEMPLATE_PATH

WORLD_SOURCES_COLLECTION = "world_sources"
TEMPLATE_SCENE_NAME = "bV2_template"
TEMPLATE_VIEW_LAYER_NAME = "template_base"

DEBUG_MATERIAL_NAME = "v2WorldAOV"
DEBUG_POINT_GN_NAME = "v2WorldPoints"
DEBUG_CUBE_GN_NAME = "v2WorldCubes"

SCENE_AOV_SPECS = [
    ("proposal-decay", "VALUE"),
    ("proposal-release-control", "VALUE"),
    ("proposal-recruit", "VALUE"),
    ("proposal-colonise", "VALUE"),
    ("proposal-deploy-structure", "VALUE"),
    ("resource_none_mask", "VALUE"),
    ("resource_dead_branch_mask", "VALUE"),
    ("resource_peeling_bark_mask", "VALUE"),
    ("resource_perch_branch_mask", "VALUE"),
    ("resource_epiphyte_mask", "VALUE"),
    ("resource_fallen_log_mask", "VALUE"),
    ("resource_hollow_mask", "VALUE"),
    ("size", "VALUE"),
    ("control", "VALUE"),
    ("precolonial", "VALUE"),
    ("improvement", "VALUE"),
    ("canopy_resistance", "VALUE"),
    ("bioEnvelopeType", "VALUE"),
    ("intervention_bioenvelope_ply-int", "VALUE"),
    ("sim_Turns", "VALUE"),
    ("world_sim_turns", "VALUE"),
    ("world_sim_nodes", "VALUE"),
    ("world_design_bioenvelope", "VALUE"),
    ("world_design_bioenvelope_simple", "VALUE"),
    ("node_id", "VALUE"),
    ("instanceID", "VALUE"),
    ("source-year", "VALUE"),
    ("world_sim_matched", "VALUE"),
]

MATERIAL_AOV_ATTRIBUTE_MAP = [
    ("proposal-decay", "blender_proposal-decay"),
    ("proposal-release-control", "blender_proposal-release-control"),
    ("proposal-recruit", "blender_proposal-recruit"),
    ("proposal-colonise", "blender_proposal-colonise"),
    ("proposal-deploy-structure", "blender_proposal-deploy-structure"),
    ("world_sim_turns", "sim_Turns"),
    ("world_sim_nodes", "sim_Nodes"),
    ("world_design_bioenvelope", "scenario_bioEnvelope"),
    ("world_design_bioenvelope_simple", "scenario_bioEnvelopeSimple"),
    ("source-year", "source-year"),
    ("world_sim_matched", "sim_Matched"),
]

PASS_INDEX = {
    "buildings": 2,
    "roads": 1,
    "roads_hires": 1,
    "roads_lores": 1,
}

SITE_PLY_MAP = {
    "city": {
        "site_ply": "city-siteVoxels.ply",
        "road_ply": "city-roadVoxels.ply",
        "collection": "world_city",
    },
    "trimmed-parade": {
        "site_ply": "trimmed-parade-siteVoxels.ply",
        "road_ply": "trimmed-parade-roadVoxels.ply",
        "collection": "world_trimmed-parade",
    },
    "uni": {
        "site_ply": "uni-siteVoxels.ply",
        "road_ply": "uni-roadVoxels.ply",
        "collection": "world_uni",
    },
}

SITE_OBJECT_NAMES = {
    "city": {
        "buildings": "city_buildings_source",
        "roads": "city_roads_source",
        "roads_hires": "city_roads_source_hires",
        "roads_lores": "city_roads_source_lores",
    },
    "trimmed-parade": {
        "buildings": "trimmed-parade_buildings_source",
        "roads": "trimmed-parade_roads_source",
        "roads_hires": "trimmed-parade_roads_source_hires",
        "roads_lores": "trimmed-parade_roads_source_lores",
    },
    "uni": {
        "buildings": "uni_buildings_source",
        "roads": "uni_roads_source",
        "roads_hires": "uni_roads_source_hires",
        "roads_lores": "uni_roads_source_lores",
    },
}

EXPECTED_INPUT_COLLECTIONS = {
    "world_sources",
    "world_city",
    "world_trimmed-parade",
    "world_uni",
    "camera_sources",
}
EXPECTED_CAMERA_OBJECTS = {
    "city - camera - time slice - zoom",
    "city-yr180-hero-image",
    "parade - camera - time slice - zoom",
    "parade-hero-image",
    "uni - camera - time slice - zoom",
}
EXPECTED_PERSISTENT_MATERIALS = {
    "v2WorldAOV",
    "MINIMAL_RESOURCES",
    "Envelope",
}
EXPECTED_PERSISTENT_NODE_GROUPS = {
    "v2WorldPoints",
    "v2WorldCubes",
    "instance_template",
    "Envelope",
}
ALLOWED_CAMERA_COLLECTIONS = {
    "camera_sources",
    "camera_city_timeline",
    "camera_city_other",
    "camera_trimmed-parade_timeline",
    "camera_uni_timeline",
}

BUILD_MODE_SPLIT = "split"
BUILD_MODE_POINTS = "points"


def log(*args: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def verify_debug_input_template() -> None:
    opened_path = Path(bpy.data.filepath).resolve()
    if opened_path != INPUT_TEMPLATE_PATH.resolve():
        raise RuntimeError(
            f"Expected input template {INPUT_TEMPLATE_PATH}, got {opened_path}"
        )

    scene = bpy.data.scenes.get(TEMPLATE_SCENE_NAME)
    if scene is None:
        raise RuntimeError(f"Missing scene {TEMPLATE_SCENE_NAME!r} in input template")
    if bpy.context.view_layer.name != TEMPLATE_VIEW_LAYER_NAME:
        raise RuntimeError(
            f"Expected active view layer {TEMPLATE_VIEW_LAYER_NAME!r}, got {bpy.context.view_layer.name!r}"
        )

    collection_names = {collection.name for collection in bpy.data.collections}
    missing_collections = EXPECTED_INPUT_COLLECTIONS.difference(collection_names)
    if missing_collections:
        raise RuntimeError(
            f"Input template is missing required collections: {sorted(missing_collections)}"
        )

    camera_names = {obj.name for obj in bpy.data.objects if obj.type == "CAMERA"}
    missing_cameras = EXPECTED_CAMERA_OBJECTS.difference(camera_names)
    if missing_cameras:
        raise RuntimeError(
            f"Input template is missing required cameras: {sorted(missing_cameras)}"
        )

    allowed_world_object_names = set().union(*[set(names.values()) for names in SITE_OBJECT_NAMES.values()])
    for obj in bpy.data.objects:
        if obj.type != "CAMERA" and obj.name not in allowed_world_object_names:
            log("warning: ignoring unexpected object in input template", obj.name)

    material_names = {material.name for material in bpy.data.materials}
    missing_materials = EXPECTED_PERSISTENT_MATERIALS.difference(material_names)
    if missing_materials:
        raise RuntimeError(f"Input template is missing required materials: {sorted(missing_materials)}")

    node_group_names = {group.name for group in bpy.data.node_groups}
    missing_node_groups = EXPECTED_PERSISTENT_NODE_GROUPS.difference(node_group_names)
    if missing_node_groups:
        raise RuntimeError(f"Input template is missing required node groups: {sorted(missing_node_groups)}")

    extra_collections = collection_names.difference(EXPECTED_INPUT_COLLECTIONS | ALLOWED_CAMERA_COLLECTIONS)
    if extra_collections:
        log("warning: ignoring unexpected collections in input template", sorted(extra_collections))


def get_collection(name: str) -> bpy.types.Collection | None:
    return bpy.data.collections.get(name)


def ensure_collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    collection = get_collection(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        if parent is not None:
            parent.children.link(collection)
        else:
            bpy.context.scene.collection.children.link(collection)
    elif parent is not None:
        already_linked = any(child.name == name for child in parent.children)
        if not already_linked:
            parent.children.link(collection)
    return collection


def ensure_site_collection(site: str) -> bpy.types.Collection:
    site_collection_name = SITE_PLY_MAP[site]["collection"]
    site_collection = get_collection(site_collection_name)
    if site_collection is not None:
        return site_collection

    root = ensure_collection(WORLD_SOURCES_COLLECTION, bpy.context.scene.collection)
    return ensure_collection(site_collection_name, root)


def remove_object(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    mesh = obj.data if obj.type == "MESH" else None
    for modifier in list(obj.modifiers):
        obj.modifiers.remove(modifier)
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    log("  removed", name)


def remove_site_objects(site: str) -> None:
    for name in SITE_OBJECT_NAMES[site].values():
        remove_object(name)


def import_ply(filepath: Path) -> bpy.types.Object:
    bpy.ops.wm.ply_import(filepath=str(filepath))
    return bpy.context.selected_objects[0]


def extract_positions(mesh: bpy.types.Mesh) -> np.ndarray:
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    return coords.reshape((-1, 3))


def extract_point_attributes(mesh: bpy.types.Mesh) -> dict[str, dict[str, object]]:
    count = len(mesh.vertices)
    result: dict[str, dict[str, object]] = {}
    for attr in mesh.attributes:
        if attr.domain != "POINT" or attr.name == "position" or attr.name.startswith("."):
            continue
        if attr.data_type in {"FLOAT", "INT", "BOOLEAN"}:
            values = np.empty(count, dtype=np.float32 if attr.data_type == "FLOAT" else np.int32)
            attr.data.foreach_get("value", values)
        elif attr.data_type in {"FLOAT_COLOR", "BYTE_COLOR"}:
            values = np.empty(count * 4, dtype=np.float32).reshape((-1, 4))
            attr.data.foreach_get("color", values.reshape(-1))
        else:
            continue
        result[attr.name] = {"data_type": attr.data_type, "values": values}
    return result


def remove_attr_if_present(mesh: bpy.types.Mesh, name: str) -> None:
    attr = mesh.attributes.get(name)
    if attr is not None:
        mesh.attributes.remove(attr)


def ensure_imported_color(mesh: bpy.types.Mesh) -> None:
    source = mesh.attributes.get("Col")
    if source is None:
        raise RuntimeError(f"{mesh.name} is missing imported color attribute 'Col'")


def ensure_scene_aovs() -> None:
    scene = bpy.data.scenes.get(TEMPLATE_SCENE_NAME) or bpy.context.scene
    view_layer = scene.view_layers.get(TEMPLATE_VIEW_LAYER_NAME) or scene.view_layers[0]
    existing = {aov.name: aov for aov in view_layer.aovs}
    for aov_name, aov_type in SCENE_AOV_SPECS:
        aov = existing.get(aov_name)
        if aov is None:
            aov = view_layer.aovs.add()
            aov.name = aov_name
        aov.type = aov_type


def ensure_float_alias(mesh: bpy.types.Mesh, src_name: str, alias_name: str) -> None:
    source = mesh.attributes.get(src_name)
    if source is None:
        raise RuntimeError(f"{mesh.name} is missing float attribute {src_name!r}")
    values = np.empty(len(mesh.vertices), dtype=np.float32)
    source.data.foreach_get("value", values)
    remove_attr_if_present(mesh, alias_name)
    attr = mesh.attributes.new(name=alias_name, type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", values)


def ensure_debug_material() -> bpy.types.Material:
    material = bpy.data.materials.get(DEBUG_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(DEBUG_MATERIAL_NAME)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    attribute = nodes.new("ShaderNodeAttribute")
    attribute.location = (-240, 0)
    attribute.attribute_name = "Col"

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (0, 0)
    emission.inputs["Strength"].default_value = 1.0

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (220, 0)

    links.new(attribute.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    x_base = 620
    y_step = -180
    for index, (aov_name, attribute_name) in enumerate(MATERIAL_AOV_ATTRIBUTE_MAP):
        y = index * y_step

        attr = nodes.new("ShaderNodeAttribute")
        attr.location = (x_base, y)
        attr.attribute_name = attribute_name
        attr.label = f"AOV Attribute {aov_name}"

        aov = nodes.new("ShaderNodeOutputAOV")
        aov.location = (x_base + 260, y)
        aov.aov_name = aov_name
        aov.label = f"AOV Output {aov_name}"

        links.new(attr.outputs["Fac"], aov.inputs["Value"])
    return material


def ensure_debug_point_gn(material: bpy.types.Material) -> bpy.types.NodeTree:
    node_group = bpy.data.node_groups.get(DEBUG_POINT_GN_NAME)
    if node_group is None:
        node_group = bpy.data.node_groups.new(DEBUG_POINT_GN_NAME, "GeometryNodeTree")
    elif node_group.bl_idname != "GeometryNodeTree":
        bpy.data.node_groups.remove(node_group)
        node_group = bpy.data.node_groups.new(DEBUG_POINT_GN_NAME, "GeometryNodeTree")

    node_group.interface.clear()
    for node in list(node_group.nodes):
        node_group.nodes.remove(node)

    node_group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    node_group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    group_input = node_group.nodes.new("NodeGroupInput")
    group_input.location = (-360, 0)

    mesh_to_points = node_group.nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.location = (-140, 0)
    mesh_to_points.inputs["Radius"].default_value = 0.25

    set_material = node_group.nodes.new("GeometryNodeSetMaterial")
    set_material.location = (80, 0)
    set_material.inputs["Material"].default_value = material

    group_output = node_group.nodes.new("NodeGroupOutput")
    group_output.location = (300, 0)

    node_group.links.new(group_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    node_group.links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
    node_group.links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])
    return node_group


def ensure_debug_cube_gn(material: bpy.types.Material) -> bpy.types.NodeTree:
    node_group = bpy.data.node_groups.get(DEBUG_CUBE_GN_NAME)
    if node_group is None:
        node_group = bpy.data.node_groups.new(DEBUG_CUBE_GN_NAME, "GeometryNodeTree")
    elif node_group.bl_idname != "GeometryNodeTree":
        bpy.data.node_groups.remove(node_group)
        node_group = bpy.data.node_groups.new(DEBUG_CUBE_GN_NAME, "GeometryNodeTree")

    node_group.interface.clear()
    for node in list(node_group.nodes):
        node_group.nodes.remove(node)

    node_group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    node_group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    group_input = node_group.nodes.new("NodeGroupInput")
    group_input.location = (-620, 0)

    mesh_to_points = node_group.nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.location = (-400, 0)
    mesh_to_points.inputs["Radius"].default_value = 0.25

    cube = node_group.nodes.new("GeometryNodeMeshCube")
    cube.location = (-400, -220)
    cube.inputs["Size"].default_value = (1.0, 1.0, 0.25)

    instance_on_points = node_group.nodes.new("GeometryNodeInstanceOnPoints")
    instance_on_points.location = (-160, 0)

    realize = node_group.nodes.new("GeometryNodeRealizeInstances")
    realize.location = (60, 0)

    set_material = node_group.nodes.new("GeometryNodeSetMaterial")
    set_material.location = (280, 0)
    set_material.inputs["Material"].default_value = material

    group_output = node_group.nodes.new("NodeGroupOutput")
    group_output.location = (520, 0)

    node_group.links.new(group_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    node_group.links.new(mesh_to_points.outputs["Points"], instance_on_points.inputs["Points"])
    node_group.links.new(cube.outputs["Mesh"], instance_on_points.inputs["Instance"])
    node_group.links.new(instance_on_points.outputs["Instances"], realize.inputs["Geometry"])
    node_group.links.new(realize.outputs["Geometry"], set_material.inputs["Geometry"])
    node_group.links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])
    return node_group


def build_mesh(name: str, positions: np.ndarray, attrs: dict[str, dict[str, object]]) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(name)
    count = len(positions)
    if count:
        mesh.vertices.add(count)
        mesh.vertices.foreach_set("co", positions.astype(np.float32).reshape(-1))

    for attr_name, payload in attrs.items():
        data_type = str(payload["data_type"])
        values = payload["values"]
        attribute = mesh.attributes.new(name=attr_name, type=data_type, domain="POINT")
        if data_type in {"FLOAT_COLOR", "BYTE_COLOR"}:
            attribute.data.foreach_set("color", np.asarray(values, dtype=np.float32).reshape(-1))
        elif data_type == "BOOLEAN":
            attribute.data.foreach_set("value", np.asarray(values, dtype=bool))
        elif data_type == "INT":
            attribute.data.foreach_set("value", np.asarray(values, dtype=np.int32))
        else:
            attribute.data.foreach_set("value", np.asarray(values, dtype=np.float32))

    mesh.update()
    return mesh


def assign_object_state(
    obj: bpy.types.Object,
    *,
    target_collection: bpy.types.Collection,
    pass_index: int,
    material: bpy.types.Material,
    node_group: bpy.types.NodeTree,
    modifier_name: str,
) -> None:
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)
    target_collection.objects.link(obj)
    obj.pass_index = pass_index
    obj.hide_render = False
    obj.hide_viewport = False
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for modifier in list(obj.modifiers):
        obj.modifiers.remove(modifier)
    modifier = obj.modifiers.new(name=modifier_name, type="NODES")
    modifier.node_group = node_group


def build_point_world_object(
    *,
    filepath: Path,
    object_name: str,
    target_collection: bpy.types.Collection,
    pass_index: int,
    material: bpy.types.Material,
    point_gn: bpy.types.NodeTree,
) -> bpy.types.Object:
    obj = import_ply(filepath)
    obj.name = object_name
    obj.data.name = f"{object_name}_mesh"
    ensure_imported_color(obj.data)
    assign_object_state(
        obj,
        target_collection=target_collection,
        pass_index=pass_index,
        material=material,
        node_group=point_gn,
        modifier_name=DEBUG_POINT_GN_NAME,
    )
    return obj


def filter_attrs(attrs: dict[str, dict[str, object]], mask: np.ndarray) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, payload in attrs.items():
        result[name] = {
            "data_type": payload["data_type"],
            "values": np.asarray(payload["values"])[mask],
        }
    return result


def reset_site(site: str, *, build_mode: str = BUILD_MODE_SPLIT) -> dict[str, object]:
    log("resetting", site, f"mode={build_mode}")
    remove_site_objects(site)
    target_collection = ensure_site_collection(site)
    material = ensure_debug_material()
    point_gn = ensure_debug_point_gn(material)
    cube_gn = ensure_debug_cube_gn(material)

    names = SITE_OBJECT_NAMES[site]
    ply_map = SITE_PLY_MAP[site]

    # Buildings
    site_ply = PLY_DIR / ply_map["site_ply"]
    buildings = build_point_world_object(
        filepath=site_ply,
        object_name=names["buildings"],
        target_collection=target_collection,
        pass_index=PASS_INDEX["buildings"],
        material=material,
        point_gn=point_gn,
    )

    if build_mode == BUILD_MODE_POINTS:
        highres_road_ply = HIGHRES_PLY_DIR / ply_map["road_ply"]
        roads = build_point_world_object(
            filepath=highres_road_ply,
            object_name=names["roads"],
            target_collection=target_collection,
            pass_index=PASS_INDEX["roads"],
            material=material,
            point_gn=point_gn,
        )
        return {
            "site": site,
            "mode": build_mode,
            "collection": target_collection.name,
            "buildings": {"name": buildings.name, "verts": len(buildings.data.vertices)},
            "roads": {"name": roads.name, "verts": len(roads.data.vertices)},
        }

    # Roads
    road_ply = PLY_DIR / ply_map["road_ply"]
    imported_roads = import_ply(road_ply)
    ensure_imported_color(imported_roads.data)
    ensure_float_alias(imported_roads.data, "scale", "size")
    positions = extract_positions(imported_roads.data)
    attrs = extract_point_attributes(imported_roads.data)

    size_values = np.empty(len(imported_roads.data.vertices), dtype=np.float32)
    imported_roads.data.attributes["size"].data.foreach_get("value", size_values)
    hires_mask = size_values <= 0.75
    lores_mask = ~hires_mask

    road_mesh = imported_roads.data
    bpy.data.objects.remove(imported_roads, do_unlink=True)
    if road_mesh.users == 0:
        bpy.data.meshes.remove(road_mesh)

    hires_mesh = build_mesh(
        f"{names['roads_hires']}_mesh",
        positions[hires_mask],
        filter_attrs(attrs, hires_mask),
    )
    hires_obj = bpy.data.objects.new(names["roads_hires"], hires_mesh)
    assign_object_state(
        hires_obj,
        target_collection=target_collection,
        pass_index=PASS_INDEX["roads_hires"],
        material=material,
        node_group=point_gn,
        modifier_name=DEBUG_POINT_GN_NAME,
    )

    lores_mesh = build_mesh(
        f"{names['roads_lores']}_mesh",
        positions[lores_mask],
        filter_attrs(attrs, lores_mask),
    )
    lores_obj = bpy.data.objects.new(names["roads_lores"], lores_mesh)
    assign_object_state(
        lores_obj,
        target_collection=target_collection,
        pass_index=PASS_INDEX["roads_lores"],
        material=material,
        node_group=cube_gn,
        modifier_name=DEBUG_CUBE_GN_NAME,
    )

    return {
        "site": site,
        "mode": build_mode,
        "collection": target_collection.name,
        "buildings": {"name": buildings.name, "verts": len(buildings.data.vertices)},
        "roads_hires": {"name": hires_obj.name, "verts": len(hires_obj.data.vertices)},
        "roads_lores": {"name": lores_obj.name, "verts": len(lores_obj.data.vertices)},
    }


def verify_site(site: str, *, build_mode: str = BUILD_MODE_SPLIT) -> dict[str, object]:
    errors: list[str] = []
    names = SITE_OBJECT_NAMES[site]
    collection_name = SITE_PLY_MAP[site]["collection"]

    scene = bpy.data.scenes.get(TEMPLATE_SCENE_NAME)
    if scene is None:
        errors.append(f"missing scene {TEMPLATE_SCENE_NAME}")
    elif bpy.context.view_layer.name != TEMPLATE_VIEW_LAYER_NAME:
        pass

    if build_mode == BUILD_MODE_POINTS:
        expected = {
            "buildings": (names["buildings"], DEBUG_POINT_GN_NAME, PASS_INDEX["buildings"]),
            "roads": (names["roads"], DEBUG_POINT_GN_NAME, PASS_INDEX["roads"]),
        }
    else:
        expected = {
            "buildings": (names["buildings"], DEBUG_POINT_GN_NAME, PASS_INDEX["buildings"]),
            "roads_hires": (names["roads_hires"], DEBUG_POINT_GN_NAME, PASS_INDEX["roads_hires"]),
            "roads_lores": (names["roads_lores"], DEBUG_CUBE_GN_NAME, PASS_INDEX["roads_lores"]),
        }

    for role, (object_name, modifier_name, pass_index) in expected.items():
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            errors.append(f"{object_name}: missing")
            continue
        if obj.pass_index != pass_index:
            errors.append(f"{object_name}: pass_index={obj.pass_index} expected={pass_index}")
        if obj.hide_render:
            errors.append(f"{object_name}: hide_render=True")
        if obj.hide_viewport:
            errors.append(f"{object_name}: hide_viewport=True")
        if not obj.data.materials or obj.data.materials[0].name != DEBUG_MATERIAL_NAME:
            errors.append(f"{object_name}: material mismatch")
        modifier = obj.modifiers.get(modifier_name)
        if modifier is None or modifier.node_group is None or modifier.node_group.name != modifier_name:
            errors.append(f"{object_name}: missing modifier {modifier_name}")
        collections = [collection.name for collection in obj.users_collection]
        if collection_name not in collections:
            errors.append(f"{object_name}: wrong collection {collections}")
        if obj.data.attributes.get("Col") is None:
            errors.append(f"{object_name}: missing attr Col")
        if build_mode == BUILD_MODE_SPLIT and role != "buildings" and obj.data.attributes.get("size") is None:
            errors.append(f"{object_name}: missing attr size")

    return {"site": site, "ok": len(errors) == 0, "errors": errors}


def parse_args(argv: list[str]) -> tuple[list[str], Path | None, str]:
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    sites: list[str] = []
    output_blend: Path | None = None
    build_mode = BUILD_MODE_POINTS if "--use-points" in args else BUILD_MODE_SPLIT

    if "--reset" in args:
        index = args.index("--reset")
        for token in args[index + 1:]:
            if token.startswith("--"):
                break
            if token in SITE_PLY_MAP:
                sites.append(token)
            else:
                log("warning: unknown site", token)

    if "--output-blend" in args:
        index = args.index("--output-blend")
        if index + 1 >= len(args):
            raise RuntimeError("--output-blend requires a filepath")
        output_blend = Path(args[index + 1]).resolve()

    return sites, output_blend, build_mode


def main() -> None:
    sites, output_blend, build_mode = parse_args(sys.argv)
    if not sites:
        log("No sites specified. Usage: -- --reset city trimmed-parade uni [--output-blend path] [--use-points]")
        return

    verify_debug_input_template()
    ensure_scene_aovs()
    results = {}
    verification = {}
    for site in sites:
        results[site] = reset_site(site, build_mode=build_mode)

    for site in sites:
        verification[site] = verify_site(site, build_mode=build_mode)

    save_path = output_blend or DEFAULT_OUTPUT_BLEND
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
    log("saved", save_path)

    print("\n===RESET_REPORT_START===")
    print(json.dumps({"results": results, "verification": verification, "saved": str(save_path)}, indent=2))
    print("===RESET_REPORT_END===")


if __name__ == "__main__":
    main()
