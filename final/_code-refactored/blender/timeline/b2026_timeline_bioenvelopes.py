import importlib.util
import os
import sys
from pathlib import Path

import bpy
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract


TARGET_SITES = tuple(
    site.strip()
    for site in os.environ.get("B2026_TIMELINE_TARGET_SITES", "city,trimmed-parade,uni").split(",")
    if site.strip()
)
TARGET_SCENARIOS = tuple(
    scenario.strip()
    for scenario in os.environ.get("B2026_TIMELINE_TARGET_SCENARIOS", "positive,trending").split(",")
    if scenario.strip()
)
PASS_INDEX = 5
ENVELOPE_MATERIAL_NAME = "Envelope"
SITE_ENVELOPE_NODE_GROUPS = {
    "city": "Envelope",
    "trimmed-parade": "Envelope Parade",
    "uni": "Envelope",
    "street": "Envelope",
}
EMPTY_BASELINE_BIOENVELOPE_YEARS = {0}


def load_local_module(module_name: str, filename: str):
    file_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def delete_collection_tree(collection):
    for child in list(collection.children):
        delete_collection_tree(child)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def ensure_collection(name: str, parent=None):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)

    if parent is not None and collection.name not in parent.children:
        parent.children.link(collection)
    return collection


def reset_collection(name: str, parent=None):
    existing = bpy.data.collections.get(name)
    if existing is not None:
        delete_collection_tree(existing)
    return ensure_collection(name, parent=parent)


def new_geometry_socket(node_group, name, in_out):
    return node_group.interface.new_socket(
        name=name,
        in_out=in_out,
        socket_type="NodeSocketGeometry",
    )


def build_timeline_group(name: str, clip_group, translation):
    existing = bpy.data.node_groups.get(name)
    if existing is not None:
        bpy.data.node_groups.remove(existing)

    node_group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    new_geometry_socket(node_group, "Geometry", "INPUT")
    new_geometry_socket(node_group, "Geometry", "OUTPUT")

    nodes = node_group.nodes
    links = node_group.links

    group_input = nodes.new("NodeGroupInput")
    group_input.location = (-500, 0)

    clip_node = nodes.new("GeometryNodeGroup")
    clip_node.node_tree = clip_group
    clip_node.location = (-220, 0)

    transform_node = nodes.new("GeometryNodeTransform")
    transform_node.location = (40, 0)
    transform_node.inputs["Translation"].default_value = translation

    group_output = nodes.new("NodeGroupOutput")
    group_output.location = (280, 0)

    links.new(group_input.outputs["Geometry"], clip_node.inputs["Geometry"])
    links.new(clip_node.outputs["Geometry"], transform_node.inputs["Geometry"])
    links.new(transform_node.outputs["Geometry"], group_output.inputs["Geometry"])
    return node_group


def ensure_geometry_nodes_modifier(obj, modifier_name: str, node_group):
    modifier = obj.modifiers.get(modifier_name)
    if modifier is None:
        modifier = obj.modifiers.new(name=modifier_name, type="NODES")
    modifier.node_group = node_group
    return modifier


def prepend_geometry_nodes_modifier(obj, modifier_name: str, node_group):
    modifier = ensure_geometry_nodes_modifier(obj, modifier_name, node_group)
    current_index = list(obj.modifiers).index(modifier)
    if current_index != 0:
        obj.modifiers.move(current_index, 0)
    return modifier


def ensure_view_layer_aov(view_layer, aov_name, aov_type="VALUE"):
    for aov in view_layer.aovs:
        if aov.name == aov_name:
            return aov
    aov = view_layer.aovs.add()
    aov.name = aov_name
    aov.type = aov_type
    return aov


def ensure_envelope_aovs(scene):
    for view_layer in scene.view_layers:
        ensure_view_layer_aov(view_layer, "bioEnvelopeType")
        ensure_view_layer_aov(view_layer, "bioSimple")
        ensure_view_layer_aov(view_layer, "sim_Turns")


def import_ply_object(filepath: Path):
    existing_names = set(bpy.data.objects.keys())
    bpy.ops.wm.ply_import(filepath=str(filepath))
    for name in bpy.data.objects.keys():
        if name not in existing_names:
            obj = bpy.data.objects.get(name)
            if obj is not None and obj.type == "MESH":
                return obj
    raise RuntimeError(f"No mesh object was imported from {filepath}")


def duplicate_source_object(source_obj: bpy.types.Object, object_name: str):
    duplicate = source_obj.copy()
    duplicate.data = source_obj.data
    duplicate.name = object_name
    if getattr(duplicate, "data", None) is not None:
        duplicate.data = duplicate.data.copy()
        duplicate.data.name = f"{object_name}_mesh"
    for modifier in list(duplicate.modifiers):
        duplicate.modifiers.remove(modifier)
    return duplicate


def create_empty_placeholder_object(object_name: str):
    mesh = bpy.data.meshes.new(f"{object_name}_mesh")
    return bpy.data.objects.new(object_name, mesh)


def ensure_material_slot(obj):
    material = bpy.data.materials.get(ENVELOPE_MATERIAL_NAME)
    if material is None or obj.data is None:
        return
    if not obj.data.materials:
        obj.data.materials.append(material)
    else:
        obj.data.materials[0] = material


def ensure_year_vertex_attribute(obj, year_value: float):
    if obj.type != "MESH" or obj.data is None:
        return

    mesh = obj.data
    existing = mesh.attributes.get("year")
    if existing is not None and not getattr(existing, "is_required", False):
        mesh.attributes.remove(existing)

    attr = mesh.attributes.new(name="year", type="FLOAT", domain="POINT")
    values = np.full(len(mesh.vertices), float(year_value), dtype=np.float32)
    if len(values):
        attr.data.foreach_set("value", values)
    if hasattr(attr, "is_runtime_only"):
        attr.is_runtime_only = False


def build_site_scenario_bioenvelopes(site: str, scenario: str, clipbox_setup, timeline_layout):
    site_spec = timeline_layout.get_timeline_site_spec(site)
    if site_spec is None:
        raise ValueError(f"No timeline site spec for {site}")

    scene = bpy.data.scenes.get(site_spec["scene_name"])
    if scene is None:
        raise ValueError(f"Scene '{site_spec['scene_name']}' not found")

    top_role = "bio_positive" if scenario == "positive" else "bio_trending"
    parent_collection = bpy.data.collections.get(
        scene_contract.get_collection_name(site, top_role)
    )
    if parent_collection is None:
        raise ValueError(
            f"Collection '{scene_contract.get_collection_name(site, top_role)}' not found"
        )

    timeline_collection = reset_collection(
        f"Year_{site}_timeline_bioenvelope_{scenario}",
        parent=parent_collection,
    )
    manager_collection = bpy.data.collections.get(site_spec["manager_collection_name"])
    if manager_collection is None:
        raise ValueError(f"Collection '{site_spec['manager_collection_name']}' not found")

    legacy_name = scene_contract.get_collection_name(site, top_role, legacy=True)
    legacy_collection = bpy.data.collections.get(legacy_name)
    if legacy_collection is not None:
        legacy_collection.hide_render = True
        legacy_collection.hide_viewport = True

    node_group_name = SITE_ENVELOPE_NODE_GROUPS[site]
    envelope_group = bpy.data.node_groups.get(node_group_name)
    if envelope_group is None:
        raise ValueError(f"Geometry Nodes group '{node_group_name}' not found")

    created_names = []
    strips_by_year = {strip["year"]: strip for strip in site_spec["strips"]}
    for display_year in site_spec["timeline_years"]:
        position_year = timeline_layout.get_position_year(site, display_year)
        strip_spec = strips_by_year[position_year]
        year = display_year
        source_scenario, source_year = timeline_layout.resolve_source_asset_request(
            site,
            scenario,
            year,
        )
        source_obj = None
        ply_path = None

        try:
            ply_path = timeline_layout.resolve_bioenvelope_ply_path(site, scenario, year)
        except FileNotFoundError:
            if year not in EMPTY_BASELINE_BIOENVELOPE_YEARS:
                continue

        clip_box_name = f"TimelineStripBox__{site}__{strip_spec['label']}"
        clip_box = bpy.data.objects.get(clip_box_name)
        if clip_box is None:
            asset_site = timeline_layout.canonicalize_asset_site(site)
            if asset_site != site:
                clip_box_name = f"TimelineStripBox__{asset_site}__{strip_spec['label']}"
                clip_box = bpy.data.objects.get(clip_box_name)
        if clip_box is None:
            raise ValueError(f"Timeline strip box '{clip_box_name}' not found")

        clip_group = clipbox_setup.build_clip_group(
            f"Timeline Envelope Clip :: {site} :: {scenario} :: {strip_spec['label']}",
            clip_box,
            delete_domain="POINT",
        )
        timeline_group = build_timeline_group(
            f"Timeline Envelope Transform :: {site} :: {scenario} :: {strip_spec['label']}",
            clip_group,
            strip_spec["translate"],
        )

        object_name = f"{site}_{scenario}_envelope__yr{display_year}"
        if source_obj is not None:
            imported = duplicate_source_object(source_obj, object_name)
        elif ply_path is None:
            imported = create_empty_placeholder_object(object_name)
        else:
            imported = import_ply_object(ply_path)
            imported.name = object_name
            imported.data.name = imported.name
        imported.pass_index = PASS_INDEX
        imported.hide_render = False
        imported.hide_viewport = False
        ensure_year_vertex_attribute(imported, display_year)
        imported["timeline_year"] = display_year
        imported["source_scenario"] = source_scenario
        imported["source_timeline_year"] = source_year
        imported["position_timeline_year"] = position_year
        imported["timeline_label"] = f"yr{display_year}"
        imported["position_timeline_label"] = strip_spec["label"]

        timeline_collection.objects.link(imported)
        for collection in list(imported.users_collection):
            if collection != timeline_collection:
                collection.objects.unlink(imported)

        ensure_geometry_nodes_modifier(imported, "GeometryNodes", envelope_group)
        prepend_geometry_nodes_modifier(imported, "Timeline Clip Translate", timeline_group)
        ensure_material_slot(imported)
        created_names.append(imported.name)

    ensure_envelope_aovs(scene)
    print(
        f"TIMELINE_BIOENV site={site} scenario={scenario} "
        f"collection={timeline_collection.name} created={created_names}"
    )
    return created_names


def main():
    clipbox_setup = load_local_module(
        "b2026_clipbox_setup_timeline_bioenv_runtime",
        "b2026_timeline_clipbox_setup.py",
    )
    timeline_layout = load_local_module(
        "b2026_timeline_layout_timeline_bioenv_runtime",
        "b2026_timeline_layout.py",
    )
    active_test_mode = timeline_layout.get_active_timeline_test_mode()
    if active_test_mode is not None:
        print(f"Timeline test mode active: {active_test_mode}")

    built = []
    for site in TARGET_SITES:
        for scenario in TARGET_SCENARIOS:
            built.extend(build_site_scenario_bioenvelopes(site, scenario, clipbox_setup, timeline_layout))

    bpy.ops.wm.save_mainfile()
    print(f"Saved blend with timeline bioenvelopes ({len(built)} objects)")


if __name__ == "__main__":
    main()
