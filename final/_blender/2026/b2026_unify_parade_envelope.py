import bpy
import runpy
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "_futureSim_refactored"))

from _futureSim_refactored.paths import hook_bioenvelope_ply_path


PARADE_SCENE_NAME = "parade-senescent"
TARGET_COLLECTION_NAME = "Parade_envelope"
SOURCE_PLY_PATH = hook_bioenvelope_ply_path("trimmed-parade", "positive", 180, 1)
TARGET_OBJECT_NAME = "trimmed-parade_positive_1_envelope_scenarioYR180"
LEGACY_GROUND_OBJECTS = ("trimmed-parade_180_rewilded",)
ENVELOPE_NODE_GROUP_NAME = "Envelope"
PARADE_ENVELOPE_NODE_GROUP_NAME = "Envelope Parade"
ENVELOPE_MATERIAL_NAME = "Envelope"
PASS_INDEX = 5
SAVE_FILE = False


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


def unlink_from_other_collections(obj, keep_collection):
    for collection in list(obj.users_collection):
        if collection != keep_collection:
            collection.objects.unlink(obj)


def ensure_geometry_nodes_modifier(obj, node_group):
    modifier = obj.modifiers.get("GeometryNodes")
    if modifier is not None and modifier.type != "NODES":
        obj.modifiers.remove(modifier)
        modifier = None

    if modifier is None:
        modifier = obj.modifiers.new(name="GeometryNodes", type="NODES")

    modifier.node_group = node_group
    return modifier


def ensure_parade_envelope_group():
    base_group = bpy.data.node_groups.get(ENVELOPE_NODE_GROUP_NAME)
    if base_group is None:
        raise ValueError(f"Geometry Nodes group '{ENVELOPE_NODE_GROUP_NAME}' not found")

    existing = bpy.data.node_groups.get(PARADE_ENVELOPE_NODE_GROUP_NAME)
    if existing is not None:
        bpy.data.node_groups.remove(existing)

    duplicate = base_group.copy()
    duplicate.name = PARADE_ENVELOPE_NODE_GROUP_NAME
    return duplicate


def import_ply_object(filepath):
    existing_names = {obj.name for obj in bpy.data.objects}
    bpy.ops.wm.ply_import(filepath=str(filepath))
    new_objects = [
        obj for obj in bpy.data.objects
        if obj.name not in existing_names and obj.type == "MESH"
    ]
    if not new_objects:
        raise RuntimeError(f"No mesh object was imported from {filepath}")
    return new_objects[0]


def ensure_target_object(scene):
    collection = bpy.data.collections.get(TARGET_COLLECTION_NAME)
    if collection is None:
        raise ValueError(f"Collection '{TARGET_COLLECTION_NAME}' not found")

    if scene.collection.children.get(collection.name) is None:
        scene.collection.children.link(collection)

    existing = bpy.data.objects.get(TARGET_OBJECT_NAME)
    if existing is not None:
        return existing, collection

    imported = import_ply_object(SOURCE_PLY_PATH)
    imported.name = TARGET_OBJECT_NAME
    collection.objects.link(imported)
    unlink_from_other_collections(imported, collection)
    return imported, collection


def ensure_material_slot(obj):
    material = bpy.data.materials.get(ENVELOPE_MATERIAL_NAME)
    if material is None or obj.data is None:
        return

    if not obj.data.materials:
        obj.data.materials.append(material)
    else:
        obj.data.materials[0] = material


def hide_legacy_ground():
    hidden = []
    for name in LEGACY_GROUND_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        obj.hide_viewport = True
        obj.hide_render = True
        hidden.append(obj.name)
    return hidden


def main():
    scene = bpy.data.scenes.get(PARADE_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{PARADE_SCENE_NAME}' not found")

    if not SOURCE_PLY_PATH.exists():
        raise FileNotFoundError(f"Envelope PLY not found: {SOURCE_PLY_PATH}")

    envelope_group = ensure_parade_envelope_group()

    target_obj, collection = ensure_target_object(scene)
    target_obj.pass_index = PASS_INDEX
    target_obj.hide_viewport = False
    target_obj.hide_render = False
    ensure_geometry_nodes_modifier(target_obj, envelope_group)
    ensure_material_slot(target_obj)
    ensure_envelope_aovs(scene)

    hidden = hide_legacy_ground()

    runpy.run_path(
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/_blender/2026/b2026_compositor_aov_expose.py",
        run_name="__main__",
    )

    attrs = list(target_obj.data.attributes.keys()) if target_obj.data and hasattr(target_obj.data, "attributes") else []
    print(f"Unified parade envelope object: {target_obj.name}")
    print(f"Collection: {collection.name}")
    print(f"Modifier group: {target_obj.modifiers['GeometryNodes'].node_group.name}")
    print(f"Pass index: {target_obj.pass_index}")
    print(f"Attributes: {attrs}")
    print(f"Hidden legacy ground: {hidden}")

    if SAVE_FILE:
        bpy.ops.wm.save_mainfile()
        print("Saved current blend file")


if __name__ == "__main__":
    main()
