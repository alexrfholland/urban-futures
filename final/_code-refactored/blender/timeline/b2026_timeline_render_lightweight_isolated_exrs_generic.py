from __future__ import annotations

import importlib.util
from pathlib import Path
import hashlib
import os
import sys

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract


BLEND_SCENE_NAME = os.environ["B2026_SCENE_NAME"]
SITE_KEY = os.environ["B2026_SITE_KEY"]
CAMERA_NAME = os.environ["B2026_CAMERA_NAME"]
BUILD_MODE = os.environ.get("B2026_TIMELINE_BUILD_MODE", "timeline").strip().lower()
BIO_PREFIX = os.environ.get("B2026_BIO_PREFIX", f"{SITE_KEY}_")
OUTPUT_BASENAME = os.environ.get("B2026_OUTPUT_BASENAME", BLEND_SCENE_NAME)
OUTPUT_TAG = os.environ.get("B2026_OUTPUT_TAG", "8k")
OUTPUT_DIR = Path(os.environ["B2026_OUTPUT_DIR"])
SETUP_ONLY = os.environ.get("B2026_SETUP_ONLY", "0") == "1"
SAVE_MAINFILE = os.environ.get("B2026_SAVE_MAINFILE", "1") != "0"
MIST_WORLD_NAME = os.environ.get("B2026_MIST_WORLD_NAME", "debug_timeslice_world")
MIST_START = float(os.environ.get("B2026_MIST_START", "560"))
MIST_DEPTH = float(os.environ.get("B2026_MIST_DEPTH", "320"))
MIST_FALLOFF = os.environ.get("B2026_MIST_FALLOFF", "QUADRATIC")
CAMERA_SOURCE_BLEND = os.environ.get("B2026_CAMERA_SOURCE_BLEND", "").strip()
CAMERA_SOURCE_NAME = os.environ.get("B2026_CAMERA_SOURCE_NAME", CAMERA_NAME).strip()
MIST_SOURCE_BLEND = os.environ.get("B2026_MIST_SOURCE_BLEND", "").strip()
MIST_SOURCE_SCENE = os.environ.get("B2026_MIST_SOURCE_SCENE", "").strip()

WORLD_POINT_GROUP_NAMES = (
    "Background",
    "Background - Large pts",
    "Background.001",
    "Background - Large pts.001",
)
WORLD_CUBE_GROUP_NAMES = (
    "Background Cubes",
    "Background - Large pts Cubes",
    "Background.001 Cubes",
    "Background - Large pts.001 Cubes",
)
WORLD_POINT_SOURCE_MATERIALS = (
    "WORLD_AOV",
    "WORLD_POINT_AOV",
    "Material.007",
    "Material.001",
)
WORLD_AOV_MATERIAL_NAME = "WORLD_AOV"
TIMELINE_TARGET_VIEW_LAYERS = (
    "existing_condition",
    "pathway_state",
    "priority_state",
    "existing_condition_trending",
    "trending_state",
    "bioenvelope_positive",
    "bioenvelope_trending",
)
VIEW_LAYER_FILTER = tuple(
    name.strip()
    for name in os.environ.get("B2026_TARGET_VIEW_LAYERS", "").split(",")
    if name.strip()
)
FULL_RESOLUTION = (
    int(os.environ.get("B2026_RES_X", "7680")),
    int(os.environ.get("B2026_RES_Y", "4320")),
)
FULL_PERCENTAGE = int(os.environ.get("B2026_RES_PERCENT", "100"))
FULL_SAMPLES = int(os.environ.get("B2026_SAMPLES", "64"))
EXR_DEPTH = "16"
AOV_SPECS = (
    ("structure_id", "VALUE"),
    ("size", "VALUE"),
    ("control", "VALUE"),
    ("node_type", "VALUE"),
    ("tree_interventions", "VALUE"),
    ("tree_proposals", "VALUE"),
    ("proposal-decay", "VALUE"),
    ("proposal-release-control", "VALUE"),
    ("proposal-recruit", "VALUE"),
    ("proposal-colonise", "VALUE"),
    ("proposal-deploy-structure", "VALUE"),
    ("improvement", "VALUE"),
    ("canopy_resistance", "VALUE"),
    ("node_id", "VALUE"),
    ("instanceID", "VALUE"),
    ("instance_id", "VALUE"),
    ("precolonial", "VALUE"),
    ("bioEnvelopeType", "VALUE"),
    ("bioSimple", "VALUE"),
    ("sim_Turns", "VALUE"),
    ("isSenescent", "VALUE"),
    ("isTerminal", "VALUE"),
    ("resource", "VALUE"),
    ("resource_colour", "COLOR"),
    ("resource_tree_mask", "VALUE"),
    ("resource_none_mask", "VALUE"),
    ("resource_dead_branch_mask", "VALUE"),
    ("resource_peeling_bark_mask", "VALUE"),
    ("resource_perch_branch_mask", "VALUE"),
    ("resource_epiphyte_mask", "VALUE"),
    ("resource_fallen_log_mask", "VALUE"),
    ("resource_hollow_mask", "VALUE"),
    ("world_sim_turns", "VALUE"),
    ("world_sim_nodes", "VALUE"),
    ("world_design_bioenvelope", "VALUE"),
    ("world_design_bioenvelope_simple", "VALUE"),
    ("world_sim_matched", "VALUE"),
)
WORLD_POINT_AOV_SPECS = (
    ("world_sim_turns", "sim_Turns", "VALUE"),
    ("world_sim_nodes", "sim_Nodes", "VALUE"),
    ("world_design_bioenvelope", "scenario_bioEnvelope", "VALUE"),
    ("world_design_bioenvelope_simple", "scenario_bioEnvelopeSimple", "VALUE"),
    ("world_sim_matched", "sim_Matched", "VALUE"),
    ("proposal-decay", "blender_proposal-decay", "VALUE"),
    ("proposal-release-control", "blender_proposal-release-control", "VALUE"),
    ("proposal-recruit", "blender_proposal-recruit", "VALUE"),
    ("proposal-colonise", "blender_proposal-colonise", "VALUE"),
    ("proposal-deploy-structure", "blender_proposal-deploy-structure", "VALUE"),
)


def ensure_material(name: str):
    material = bpy.data.materials.get(name)
    if material is None:
        raise ValueError(f"Required material '{name}' was not found")
    return material


def refresh_minimal_resources_material():
    script_path = SCRIPT_DIR.parents[2] / "_blender" / "2026" / "MINIMAL_RESOURCES.py"
    spec = importlib.util.spec_from_file_location("b2026_minimal_resources_builder", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MINIMAL_RESOURCES builder from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError(f"{script_path} does not expose main()")
    module.main()


def ensure_world(name: str) -> bpy.types.World:
    world = bpy.data.worlds.get(name)
    if world is None:
        world = bpy.data.worlds.new(name)
    return world


def append_object_from_blend(blend_path: str, object_name: str):
    blend_file = Path(blend_path)
    if not blend_file.exists():
        raise FileNotFoundError(f"Blend file not found: {blend_file}")
    with bpy.data.libraries.load(str(blend_file), link=False) as (data_from, data_to):
        if object_name not in data_from.objects:
            raise ValueError(f"Object '{object_name}' not found in {blend_file}")
        data_to.objects = [object_name]
    appended = next((obj for obj in data_to.objects if obj is not None), None)
    if appended is None:
        raise ValueError(f"Object '{object_name}' could not be appended from {blend_file}")
    return appended


def append_camera_from_source(scene: bpy.types.Scene):
    existing = bpy.data.objects.get(CAMERA_NAME)
    if existing is not None:
        return existing
    if not CAMERA_SOURCE_BLEND:
        raise ValueError(f"Camera '{CAMERA_NAME}' not found and B2026_CAMERA_SOURCE_BLEND is not set")
    camera = append_object_from_blend(CAMERA_SOURCE_BLEND, CAMERA_SOURCE_NAME)
    if scene.collection.objects.get(camera.name) is None:
        scene.collection.objects.link(camera)
    if camera.name != CAMERA_NAME:
        camera.name = CAMERA_NAME
    return camera


def read_mist_settings_from_source_scene():
    if not MIST_SOURCE_BLEND or not MIST_SOURCE_SCENE:
        return {
            "use_mist": True,
            "start": MIST_START,
            "depth": MIST_DEPTH,
            "falloff": MIST_FALLOFF,
        }
    blend_file = Path(MIST_SOURCE_BLEND)
    if not blend_file.exists():
        raise FileNotFoundError(f"Blend file not found: {blend_file}")
    with bpy.data.libraries.load(str(blend_file), link=False) as (data_from, data_to):
        if MIST_SOURCE_SCENE not in data_from.scenes:
            raise ValueError(f"Scene '{MIST_SOURCE_SCENE}' not found in {blend_file}")
        data_to.scenes = [MIST_SOURCE_SCENE]
    source_scene = next((scene for scene in data_to.scenes if scene is not None), None)
    if source_scene is None:
        raise ValueError(f"Scene '{MIST_SOURCE_SCENE}' could not be appended from {blend_file}")
    source_world = source_scene.world
    if source_world is None:
        settings = {
            "use_mist": False,
            "start": MIST_START,
            "depth": MIST_DEPTH,
            "falloff": MIST_FALLOFF,
        }
    else:
        mist = source_world.mist_settings
        settings = {
            "use_mist": bool(mist.use_mist),
            "start": float(mist.start),
            "depth": float(mist.depth),
            "falloff": mist.falloff,
        }
    source_world = source_scene.world
    bpy.data.scenes.remove(source_scene)
    if source_world is not None and source_world.users == 0:
        bpy.data.worlds.remove(source_world)
    return settings


def ensure_scene_camera(scene: bpy.types.Scene):
    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None:
        camera = append_camera_from_source(scene)
    scene.camera = camera
    return camera


def ensure_scene_world_and_mist(scene: bpy.types.Scene):
    world = scene.world or ensure_world(MIST_WORLD_NAME)
    scene.world = world
    source_mist = read_mist_settings_from_source_scene()
    mist = world.mist_settings
    mist.use_mist = source_mist["use_mist"]
    mist.start = source_mist["start"]
    mist.depth = source_mist["depth"]
    mist.falloff = source_mist["falloff"]
    for view_layer in scene.view_layers:
        view_layer.use_pass_mist = True
    return world


def restore_instancer_materials():
    target_material = ensure_material("MINIMAL_RESOURCES")
    for node_group in bpy.data.node_groups:
        if not node_group.name.startswith(("tree_", "log_", "pole_", "instance_template")):
            continue
        for node in node_group.nodes:
            if node.bl_idname != "GeometryNodeSetMaterial" or "Material" not in node.inputs:
                continue
            node.inputs["Material"].default_value = target_material


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def ensure_world_point_material():
    material = bpy.data.materials.get(WORLD_AOV_MATERIAL_NAME)
    if material is None:
        source = None
        for name in WORLD_POINT_SOURCE_MATERIALS:
            source = bpy.data.materials.get(name)
            if source is not None:
                break
        if source is None:
            for node_group_name in WORLD_POINT_GROUP_NAMES + WORLD_CUBE_GROUP_NAMES:
                node_group = bpy.data.node_groups.get(node_group_name)
                if node_group is None:
                    continue
                for node in node_group.nodes:
                    if node.bl_idname != "GeometryNodeSetMaterial" or "Material" not in node.inputs:
                        continue
                    source = node.inputs["Material"].default_value
                    if source is not None:
                        break
                if source is not None:
                    break
        if source is None:
            raise ValueError("Could not find a source world point-cloud material.")
        material = source.copy()
        material.name = WORLD_AOV_MATERIAL_NAME

    node_tree = material.node_tree
    nodes = node_tree.nodes
    base_x = max((node.location.x for node in nodes), default=400.0) + 260.0
    base_y = max((node.location.y for node in nodes), default=200.0)

    for index, (aov_name, attr_name, aov_type) in enumerate(WORLD_POINT_AOV_SPECS):
        y = base_y - (index * 220.0)

        geometry_attr_name = f"World Geometry Attribute {aov_name}"
        geometry_attr = nodes.get(geometry_attr_name) or nodes.new("ShaderNodeAttribute")
        geometry_attr.name = geometry_attr_name
        geometry_attr.label = geometry_attr_name
        geometry_attr.location = (base_x - 520.0, y + 60.0)
        geometry_attr.attribute_name = attr_name
        if hasattr(geometry_attr, "attribute_type"):
            geometry_attr.attribute_type = "GEOMETRY"

        instancer_attr_name = f"World Instancer Attribute {aov_name}"
        instancer_attr = nodes.get(instancer_attr_name) or nodes.new("ShaderNodeAttribute")
        instancer_attr.name = instancer_attr_name
        instancer_attr.label = instancer_attr_name
        instancer_attr.location = (base_x - 520.0, y - 60.0)
        instancer_attr.attribute_name = attr_name
        if hasattr(instancer_attr, "attribute_type"):
            instancer_attr.attribute_type = "INSTANCER"

        aov_name_node = f"World Point AOV {aov_name}"
        aov_node = nodes.get(aov_name_node) or nodes.new("ShaderNodeOutputAOV")
        aov_node.name = aov_name_node
        aov_node.label = aov_name_node
        aov_node.location = (base_x, y)
        aov_node.aov_name = aov_name

        max_name = f"World Attribute Max {aov_name}"
        max_node = nodes.get(max_name) or nodes.new("ShaderNodeMath")
        max_node.name = max_name
        max_node.label = max_name
        max_node.location = (base_x - 160.0, y)
        max_node.operation = "MAXIMUM"
        ensure_link(node_tree, geometry_attr.outputs["Fac"], max_node.inputs[0])
        ensure_link(node_tree, instancer_attr.outputs["Fac"], max_node.inputs[1])
        ensure_link(node_tree, max_node.outputs["Value"], aov_node.inputs["Value"])

    return material


def restore_world_point_materials():
    target_material = ensure_world_point_material()
    for node_group_name in WORLD_POINT_GROUP_NAMES + WORLD_CUBE_GROUP_NAMES:
        node_group = bpy.data.node_groups.get(node_group_name)
        if node_group is None:
            continue
        for node in node_group.nodes:
            if node.bl_idname != "GeometryNodeSetMaterial" or "Material" not in node.inputs:
                continue
            node.inputs["Material"].default_value = target_material


def restore_bioenvelope_materials():
    target_material = ensure_material("Envelope")
    for obj in bpy.data.objects:
        if "_envelope__yr" not in obj.name or not obj.name.startswith(BIO_PREFIX):
            continue
        data = getattr(obj, "data", None)
        materials = getattr(data, "materials", None)
        if materials is None:
            continue
        if len(materials) == 0:
            materials.append(target_material)
            continue
        for index in range(len(materials)):
            materials[index] = target_material


def ensure_aov(view_layer: bpy.types.ViewLayer, name: str, aov_type: str):
    for aov in view_layer.aovs:
        if aov.name == name:
            aov.type = aov_type
            return aov
    aov = view_layer.aovs.add()
    aov.name = name
    aov.type = aov_type
    return aov


def ensure_target_layer_aovs(scene: bpy.types.Scene):
    ensure_standard_view_layers(scene)
    for layer_name in get_target_view_layers():
        view_layer = scene.view_layers.get(layer_name)
        if view_layer is None:
            continue
        if hasattr(view_layer, "use"):
            view_layer.use = True
        for aov_name, aov_type in AOV_SPECS:
            ensure_aov(view_layer, aov_name, aov_type)


def ensure_standard_view_layers(scene: bpy.types.Scene):
    layer_names = (
        scene_contract.SINGLE_STATE_VIEW_LAYERS
        if BUILD_MODE == "single_state"
        else scene_contract.STANDARD_VIEW_LAYERS
    )
    for layer_name in layer_names:
        if scene.view_layers.get(layer_name) is None:
            scene.view_layers.new(name=layer_name)


def get_target_view_layers():
    if BUILD_MODE == "single_state":
        return scene_contract.SINGLE_STATE_VIEW_LAYERS
    return TIMELINE_TARGET_VIEW_LAYERS


def ensure_render_passes(view_layer: bpy.types.ViewLayer):
    for attr in (
        "use_pass_combined",
        "use_pass_z",
        "use_pass_mist",
        "use_pass_normal",
        "use_pass_object_index",
        "use_pass_material_index",
        "use_pass_ambient_occlusion",
    ):
        if hasattr(view_layer, attr):
            setattr(view_layer, attr, True)


def set_collection_render_state(collection_names, hide_render):
    original_state = {}
    for collection_name in collection_names:
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            continue
        original_state[collection_name] = collection.hide_render
        collection.hide_render = hide_render
    return original_state


def restore_collection_render_state(state_by_name):
    for collection_name, hide_render in state_by_name.items():
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            collection.hide_render = hide_render


def set_object_render_state(object_names, hide_render):
    original_state = {}
    for object_name in object_names:
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            continue
        original_state[object_name] = obj.hide_render
        obj.hide_render = hide_render
    return original_state


def restore_object_render_state(state_by_name):
    for object_name, hide_render in state_by_name.items():
        obj = bpy.data.objects.get(object_name)
        if obj is not None:
            obj.hide_render = hide_render


def iter_collection_tree(collection):
    yield collection
    for child in collection.children:
        yield from iter_collection_tree(child)


def iter_layer_collection_tree(layer_collection):
    yield layer_collection
    for child in layer_collection.children:
        yield from iter_layer_collection_tree(child)


def get_layer_collection_by_name(view_layer: bpy.types.ViewLayer, collection_name: str):
    for layer_collection in iter_layer_collection_tree(view_layer.layer_collection):
        if layer_collection.collection.name == collection_name:
            return layer_collection
    return None


def clear_collection_tree_render_flags(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return
    for current in iter_collection_tree(collection):
        current.hide_render = False
        for obj in current.objects:
            obj.hide_render = False


def normalize_single_state_render_visibility(scene: bpy.types.Scene):
    if BUILD_MODE != "single_state":
        return

    collection_names = [
        scene_contract.get_single_state_top_level_name(SITE_KEY, role)
        for role in ("manager", "setup", "cameras", "positive", "priority", "trending")
    ]
    for collection_name in collection_names:
        clear_collection_tree_render_flags(collection_name)

    for obj in bpy.data.objects:
        if "__yr" in obj.name:
            obj.hide_render = False


def reset_view_layer_collection_excludes(view_layer: bpy.types.ViewLayer):
    for layer_collection in iter_layer_collection_tree(view_layer.layer_collection):
        layer_collection.exclude = False
        if hasattr(layer_collection, "holdout"):
            layer_collection.holdout = False
        if hasattr(layer_collection, "indirect_only"):
            layer_collection.indirect_only = False


def apply_single_state_view_layer_excludes(view_layer: bpy.types.ViewLayer, visibility: dict):
    reset_view_layer_collection_excludes(view_layer)

    for collection_name in visibility["hide_collections"]:
        layer_collection = get_layer_collection_by_name(view_layer, collection_name)
        if layer_collection is not None:
            layer_collection.exclude = True

    for collection_name in visibility["hide_child_collections"]:
        layer_collection = get_layer_collection_by_name(view_layer, collection_name)
        if layer_collection is not None:
            layer_collection.exclude = True


def get_single_state_layer_visibility(view_layer_name: str):
    contract = scene_contract.SITE_CONTRACTS[SITE_KEY]
    year = int(os.environ.get("B2026_SINGLE_STATE_YEAR", "180"))
    top = {
        role: scene_contract.get_single_state_top_level_name(SITE_KEY, role)
        for role in ("manager", "setup", "cameras", "positive", "priority", "trending")
    }
    positive_world = [
        scene_contract.get_single_state_world_object_name(object_name, year, "positive")
        for object_name in contract["world_objects"].values()
    ]
    trending_world = [
        scene_contract.get_single_state_world_object_name(object_name, year, "trending")
        for object_name in contract["world_objects"].values()
    ]
    setup_world = list(contract["world_objects"].values())
    positive_tree_log = [
        scene_contract.get_single_state_node_collection_name(
            SITE_KEY,
            node_type,
            year,
            "positive",
            collection_kind,
        )
        for node_type in ("tree", "log")
        for collection_kind in ("positions", "plyModels")
    ]
    positive_ply_models = [
        scene_contract.get_single_state_node_collection_name(
            SITE_KEY,
            node_type,
            year,
            "positive",
            "plyModels",
        )
        for node_type in ("tree", "log")
    ]
    positive_poles = [
        scene_contract.get_single_state_node_collection_name(
            SITE_KEY,
            "pole",
            year,
            "positive",
            collection_kind,
        )
        for collection_kind in ("positions", "plyModels")
    ]
    priority_tree_log = [
        scene_contract.get_single_state_node_collection_name(
            SITE_KEY,
            node_type,
            year,
            "positive",
            collection_kind,
            priority=True,
        )
        for node_type in ("tree", "log")
        for collection_kind in ("positions", "plyModels")
    ]
    priority_ply_models = [
        scene_contract.get_single_state_node_collection_name(
            SITE_KEY,
            node_type,
            year,
            "positive",
            "plyModels",
            priority=True,
        )
        for node_type in ("tree", "log")
    ]
    priority_poles = [
        scene_contract.get_single_state_node_collection_name(
            SITE_KEY,
            "pole",
            year,
            "positive",
            collection_kind,
            priority=True,
        )
        for collection_kind in ("positions", "plyModels")
    ]
    trending_tree_log = [
        scene_contract.get_single_state_node_collection_name(
            SITE_KEY,
            node_type,
            year,
            "trending",
            collection_kind,
        )
        for node_type in ("tree", "log")
        for collection_kind in ("positions", "plyModels")
    ]
    trending_ply_models = [
        scene_contract.get_single_state_node_collection_name(
            SITE_KEY,
            node_type,
            year,
            "trending",
            "plyModels",
        )
        for node_type in ("tree", "log")
    ]
    trending_poles = [
        scene_contract.get_single_state_node_collection_name(
            SITE_KEY,
            "pole",
            year,
            "trending",
            collection_kind,
        )
        for collection_kind in ("positions", "plyModels")
    ]
    positive_envelope = scene_contract.get_single_state_envelope_object_name(SITE_KEY, "positive", year)
    trending_envelope = scene_contract.get_single_state_envelope_object_name(SITE_KEY, "trending", year)

    hidden_top_levels = [top["manager"], top["setup"], top["cameras"], top["positive"], top["priority"], top["trending"]]
    visible_top_levels = []
    hidden_child_collections = []
    shown_objects = []
    hidden_objects = []

    if view_layer_name == "existing_condition":
        visible_top_levels = [top["positive"]]
        hidden_top_levels = [name for name in hidden_top_levels if name not in visible_top_levels]
        hidden_child_collections = [*positive_tree_log, *positive_poles]
        shown_objects = positive_world
        hidden_objects = [*setup_world, *trending_world, positive_envelope, trending_envelope]
    elif view_layer_name == "pathway_state":
        visible_top_levels = [top["positive"]]
        hidden_top_levels = [name for name in hidden_top_levels if name not in visible_top_levels]
        shown_objects = positive_world
        hidden_objects = [*setup_world, *trending_world, positive_envelope, trending_envelope]
        hidden_child_collections = positive_poles
    elif view_layer_name == "priority_state":
        visible_top_levels = [top["priority"]]
        hidden_top_levels = [name for name in hidden_top_levels if name not in visible_top_levels]
        hidden_child_collections = [*priority_poles, *priority_ply_models]
        hidden_objects = [*setup_world, *positive_world, *trending_world, positive_envelope, trending_envelope]
    elif view_layer_name == "trending_state":
        visible_top_levels = [top["trending"]]
        hidden_top_levels = [name for name in hidden_top_levels if name not in visible_top_levels]
        shown_objects = trending_world
        hidden_objects = [*setup_world, *positive_world, positive_envelope, trending_envelope]
        hidden_child_collections = trending_poles
    elif view_layer_name == "existing_condition_trending":
        visible_top_levels = [top["trending"]]
        hidden_top_levels = [name for name in hidden_top_levels if name not in visible_top_levels]
        shown_objects = trending_world
        hidden_objects = [*setup_world, *positive_world, positive_envelope, trending_envelope]
        hidden_child_collections = [*trending_tree_log, *trending_poles]
    elif view_layer_name == "bioenvelope_positive":
        visible_top_levels = [top["positive"]]
        hidden_top_levels = [name for name in hidden_top_levels if name not in visible_top_levels]
        hidden_child_collections = [*positive_tree_log, *positive_poles]
        shown_objects = [*positive_world, positive_envelope]
        hidden_objects = [*setup_world, *trending_world, trending_envelope]
    elif view_layer_name == "bioenvelope_trending":
        visible_top_levels = [top["trending"]]
        hidden_top_levels = [name for name in hidden_top_levels if name not in visible_top_levels]
        hidden_child_collections = [*trending_tree_log, *trending_poles]
        shown_objects = [*trending_world, trending_envelope]
        hidden_objects = [*setup_world, *positive_world, positive_envelope]
    else:
        visible_top_levels = [top["positive"]]
        hidden_top_levels = [name for name in hidden_top_levels if name not in visible_top_levels]
        shown_objects = positive_world
        hidden_objects = [*setup_world, *trending_world, positive_envelope, trending_envelope]

    # Extra safety: keep unrelated branch child collections hidden even if their top-level is visible elsewhere.
    if top["positive"] not in visible_top_levels:
        hidden_child_collections.extend([*positive_tree_log, *positive_poles])
    if top["priority"] not in visible_top_levels:
        hidden_child_collections.extend([*priority_tree_log, *priority_poles])
    if top["trending"] not in visible_top_levels:
        hidden_child_collections.extend([*trending_tree_log, *trending_poles])

    return {
        "show_collections": visible_top_levels,
        "hide_collections": hidden_top_levels,
        "hide_child_collections": hidden_child_collections,
        "show_objects": shown_objects,
        "hide_objects": hidden_objects,
    }


def build_scene_collection_toggles(view_layer_name: str):
    contract = scene_contract.SITE_CONTRACTS[SITE_KEY]
    legacy = contract["legacy"]
    top = contract["top_level"]

    cube_timeline_name = f"{legacy['base_cubes']}_Timeline"
    timeline_positive_bio = f"Year_{SITE_KEY}_timeline_bioenvelope_positive"
    timeline_trending_bio = f"Year_{SITE_KEY}_timeline_bioenvelope_trending"
    timeline_positive_world = scene_contract.get_timeline_world_collection_name(SITE_KEY, "positive")
    timeline_trending_world = scene_contract.get_timeline_world_collection_name(SITE_KEY, "trending")
    timeline_only_names = [
        legacy["timeline_base"],
        cube_timeline_name,
        legacy["timeline_positive"],
        legacy["timeline_priority"],
        legacy["timeline_trending"],
        timeline_positive_bio,
        timeline_trending_bio,
    ]
    single_state_common_hide = [
        top["base_cubes"],
        top["bio_trending"],
        *timeline_only_names,
    ]

    if BUILD_MODE == "single_state":
        if view_layer_name == "bioenvelope_positive":
            return [
                top["bio_positive"],
                legacy["bio_positive"],
            ], [
                top["base"],
                legacy["base"],
                top["positive"],
                top["priority"],
                top["trending"],
                legacy["positive"],
                legacy["priority"],
                legacy["trending"],
                legacy["bio_trending"],
                *single_state_common_hide,
            ]

        if view_layer_name == "priority_state":
            return [
                top["priority"],
                legacy["priority"],
            ], [
                top["base"],
                legacy["base"],
                top["positive"],
                top["trending"],
                top["bio_positive"],
                legacy["positive"],
                legacy["trending"],
                legacy["bio_positive"],
                legacy["bio_trending"],
                *single_state_common_hide,
            ]

        if view_layer_name == "trending_state":
            return [
                top["trending"],
                legacy["trending"],
            ], [
                top["base"],
                legacy["base"],
                top["positive"],
                top["priority"],
                top["bio_positive"],
                legacy["positive"],
                legacy["priority"],
                legacy["bio_positive"],
                legacy["bio_trending"],
                *single_state_common_hide,
            ]

        if view_layer_name == "existing_condition":
            return [
                top["base"],
                legacy["base"],
            ], [
                top["positive"],
                top["priority"],
                top["trending"],
                top["bio_positive"],
                legacy["positive"],
                legacy["priority"],
                legacy["trending"],
                legacy["bio_positive"],
                legacy["bio_trending"],
                *single_state_common_hide,
            ]

        return [
            top["positive"],
            legacy["positive"],
        ], [
            top["base"],
            legacy["base"],
            top["priority"],
            top["trending"],
            top["bio_positive"],
            legacy["priority"],
            legacy["trending"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            *single_state_common_hide,
        ]

    def timeline_show(*names):
        return [name for name in names if name]

    def timeline_hide(*names):
        return [name for name in names if name]

    if view_layer_name == "bioenvelope_positive":
        return timeline_show(
            top["base"],
            legacy["timeline_base"],
            timeline_positive_world,
            top["bio_positive"],
            timeline_positive_bio,
        ), timeline_hide(
            top["base_cubes"],
            top["positive"],
            top["priority"],
            top["trending"],
            top["bio_trending"],
            legacy["base"],
            legacy["base_cubes"],
            cube_timeline_name,
            timeline_trending_world,
            legacy["positive"],
            legacy["timeline_positive"],
            legacy["priority"],
            legacy["timeline_priority"],
            legacy["trending"],
            legacy["timeline_trending"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            timeline_trending_bio,
        )

    if view_layer_name == "bioenvelope_trending":
        return timeline_show(
            top["base"],
            legacy["timeline_base"],
            timeline_trending_world,
            top["bio_trending"],
            timeline_trending_bio,
        ), timeline_hide(
            top["base_cubes"],
            top["positive"],
            top["priority"],
            top["trending"],
            top["bio_positive"],
            legacy["base"],
            legacy["base_cubes"],
            cube_timeline_name,
            timeline_positive_world,
            legacy["positive"],
            legacy["timeline_positive"],
            legacy["priority"],
            legacy["timeline_priority"],
            legacy["trending"],
            legacy["timeline_trending"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            timeline_positive_bio,
        )

    if view_layer_name == "priority_state":
        return timeline_show(
            top["base"],
            legacy["timeline_base"],
            timeline_positive_world,
            top["priority"],
            legacy["timeline_priority"],
        ), timeline_hide(
            top["base_cubes"],
            top["positive"],
            top["trending"],
            top["bio_positive"],
            top["bio_trending"],
            legacy["base"],
            legacy["base_cubes"],
            cube_timeline_name,
            timeline_trending_world,
            legacy["positive"],
            legacy["timeline_positive"],
            legacy["trending"],
            legacy["timeline_trending"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            timeline_positive_bio,
            timeline_trending_bio,
        )

    if view_layer_name == "existing_condition_trending":
        return timeline_show(
            top["base"],
            legacy["timeline_base"],
            timeline_trending_world,
        ), timeline_hide(
            top["base_cubes"],
            top["positive"],
            top["priority"],
            top["trending"],
            top["bio_positive"],
            top["bio_trending"],
            legacy["base"],
            legacy["base_cubes"],
            cube_timeline_name,
            timeline_positive_world,
            legacy["positive"],
            legacy["timeline_positive"],
            legacy["priority"],
            legacy["timeline_priority"],
            legacy["trending"],
            legacy["timeline_trending"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            timeline_positive_bio,
            timeline_trending_bio,
        )

    if view_layer_name == "trending_state":
        return timeline_show(
            top["base"],
            legacy["timeline_base"],
            timeline_trending_world,
            top["trending"],
            legacy["timeline_trending"],
        ), timeline_hide(
            top["base_cubes"],
            top["positive"],
            top["priority"],
            top["bio_positive"],
            top["bio_trending"],
            legacy["base"],
            legacy["base_cubes"],
            cube_timeline_name,
            timeline_positive_world,
            legacy["positive"],
            legacy["timeline_positive"],
            legacy["priority"],
            legacy["timeline_priority"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            timeline_positive_bio,
            timeline_trending_bio,
        )

    if view_layer_name == "pathway_state":
        return timeline_show(
            top["base"],
            legacy["timeline_base"],
            timeline_positive_world,
            top["positive"],
            legacy["timeline_positive"],
        ), timeline_hide(
            top["base_cubes"],
            top["priority"],
            top["trending"],
            top["bio_positive"],
            top["bio_trending"],
            legacy["base"],
            legacy["base_cubes"],
            cube_timeline_name,
            timeline_trending_world,
            legacy["priority"],
            legacy["timeline_priority"],
            legacy["trending"],
            legacy["timeline_trending"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            timeline_positive_bio,
            timeline_trending_bio,
        )

    return timeline_show(
        top["base"],
        legacy["timeline_base"],
        timeline_positive_world,
    ), timeline_hide(
        top["base_cubes"],
        top["positive"],
        top["priority"],
        top["trending"],
        top["bio_positive"],
        top["bio_trending"],
        legacy["base"],
        legacy["base_cubes"],
        cube_timeline_name,
        timeline_trending_world,
        legacy["positive"],
        legacy["timeline_positive"],
        legacy["priority"],
        legacy["timeline_priority"],
        legacy["trending"],
        legacy["timeline_trending"],
        legacy["bio_positive"],
        legacy["bio_trending"],
        timeline_positive_bio,
        timeline_trending_bio,
    )


def clone_scene_for_layer(scene: bpy.types.Scene, view_layer_name: str):
    temp_scene = scene.copy()
    temp_scene.name = f"{scene.name}__{view_layer_name}__exr"
    temp_scene.camera = bpy.data.objects[CAMERA_NAME]
    temp_scene.render.use_compositing = True
    temp_scene.render.use_sequencer = False
    temp_scene.render.film_transparent = True
    temp_scene.render.resolution_x = FULL_RESOLUTION[0]
    temp_scene.render.resolution_y = FULL_RESOLUTION[1]
    temp_scene.render.resolution_percentage = FULL_PERCENTAGE
    temp_scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    temp_scene.render.image_settings.color_mode = "RGBA"
    temp_scene.render.image_settings.color_depth = EXR_DEPTH
    temp_scene.render.image_settings.exr_codec = "ZIP"
    temp_scene.view_settings.view_transform = "Standard"
    temp_scene.view_settings.look = "None"
    temp_scene.display_settings.display_device = "sRGB"
    temp_scene.sequencer_colorspace_settings.name = "sRGB"
    if hasattr(temp_scene, "cycles"):
        temp_scene.cycles.samples = FULL_SAMPLES
        temp_scene.cycles.preview_samples = FULL_SAMPLES

    target_layer = temp_scene.view_layers.get(view_layer_name)
    if target_layer is None:
        raise ValueError(f"View layer '{view_layer_name}' not found on copied scene '{temp_scene.name}'")
    if hasattr(target_layer, "use"):
        target_layer.use = True
    if hasattr(temp_scene.view_layers, "active"):
        temp_scene.view_layers.active = target_layer

    for layer in list(temp_scene.view_layers):
        if layer != target_layer:
            temp_scene.view_layers.remove(layer)

    target_layer = temp_scene.view_layers.get(view_layer_name)
    if target_layer is None:
        raise ValueError(f"Failed to isolate view layer '{view_layer_name}'")
    if hasattr(target_layer, "use"):
        target_layer.use = True
    ensure_render_passes(target_layer)
    return temp_scene


def clear_file_output_slots(output_node: bpy.types.Node):
    while len(output_node.inputs):
        output_node.file_slots.remove(output_node.inputs[0])


def configure_temp_scene_exr_output(temp_scene: bpy.types.Scene, view_layer_name: str, output_path: Path):
    temp_scene.use_nodes = True
    node_tree = temp_scene.node_tree
    node_tree.nodes.clear()

    render_node = node_tree.nodes.new("CompositorNodeRLayers")
    render_node.name = f"Temp Render Layers :: {view_layer_name}"
    render_node.scene = temp_scene
    render_node.layer = view_layer_name
    render_node.location = (-520.0, 0.0)

    output_node = node_tree.nodes.new("CompositorNodeOutputFile")
    output_node.name = f"Temp EXR Output :: {view_layer_name}"
    output_node.base_path = str(output_path.with_suffix(""))
    output_node.format.file_format = "OPEN_EXR_MULTILAYER"
    output_node.format.color_mode = "RGBA"
    output_node.format.color_depth = EXR_DEPTH
    output_node.format.exr_codec = "ZIP"
    output_node.location = (-40.0, 0.0)

    clear_file_output_slots(output_node)
    enabled_sockets = [socket for socket in render_node.outputs if getattr(socket, "enabled", True)]
    for socket in enabled_sockets:
        output_node.file_slots.new(socket.name)
    for socket in enabled_sockets:
        target_socket = output_node.inputs.get(socket.name)
        if target_socket is not None:
            ensure_link(node_tree, socket, target_socket)


def rename_temp_exr_output(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        output_path.parent.glob(f"{output_path.stem}*.exr"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise FileNotFoundError(f"No EXR output found for {output_path.stem}")
    rendered_path = candidates[-1]
    if output_path.exists():
        output_path.unlink()
    rendered_path.replace(output_path)


def render_isolated_exr(scene: bpy.types.Scene, view_layer_name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}_{view_layer_name}_{OUTPUT_TAG}.exr"

    temp_scene = clone_scene_for_layer(scene, view_layer_name)
    temp_scene.frame_set(1)
    configure_temp_scene_exr_output(temp_scene, view_layer_name, output_path)
    target_layer = temp_scene.view_layers.get(view_layer_name)
    if target_layer is None:
        raise ValueError(f"Temp scene '{temp_scene.name}' is missing view layer '{view_layer_name}'")
    if BUILD_MODE == "single_state":
        visibility = get_single_state_layer_visibility(view_layer_name)
        apply_single_state_view_layer_excludes(target_layer, visibility)
        shown_state = set_collection_render_state(visibility["show_collections"], hide_render=False)
        hidden_state = set_collection_render_state(visibility["hide_collections"], hide_render=True)
        hidden_child_state = set_collection_render_state(visibility["hide_child_collections"], hide_render=True)
        shown_object_state = set_object_render_state(visibility["show_objects"], hide_render=False)
        hidden_object_state = set_object_render_state(visibility["hide_objects"], hide_render=True)
    else:
        show_names, hide_names = build_scene_collection_toggles(view_layer_name)
        shown_state = set_collection_render_state(show_names, hide_render=False)
        hidden_state = set_collection_render_state(hide_names, hide_render=True)
        hidden_child_state = {}
        shown_object_state = {}
        hidden_object_state = {}

    try:
        with bpy.context.temp_override(scene=temp_scene, view_layer=target_layer):
            bpy.ops.render.render(scene=temp_scene.name, layer=view_layer_name, use_viewport=False)
        rename_temp_exr_output(output_path)
    finally:
        restore_collection_render_state(shown_state)
        restore_collection_render_state(hidden_state)
        restore_collection_render_state(hidden_child_state)
        restore_object_render_state(shown_object_state)
        restore_object_render_state(hidden_object_state)
        bpy.data.scenes.remove(temp_scene)

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"RENDER_DONE layer={view_layer_name} path={output_path} sha256={digest}")
    return output_path


def main():
    scene = bpy.data.scenes.get(BLEND_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{BLEND_SCENE_NAME}' not found in {bpy.data.filepath}")

    refresh_minimal_resources_material()
    camera = ensure_scene_camera(scene)
    world = ensure_scene_world_and_mist(scene)
    restore_instancer_materials()
    restore_world_point_materials()
    restore_bioenvelope_materials()
    normalize_single_state_render_visibility(scene)
    ensure_target_layer_aovs(scene)
    for view_layer in scene.view_layers:
        ensure_render_passes(view_layer)

    if SAVE_MAINFILE:
        bpy.ops.wm.save_mainfile()

    print(
        f"SETUP_READY scene={scene.name} camera={camera.name} world={world.name} "
        f"mist=({world.mist_settings.start},{world.mist_settings.depth},{world.mist_settings.falloff}) "
        f"setup_only={SETUP_ONLY}"
    )

    if SETUP_ONLY:
        return

    outputs = []
    target_layers = VIEW_LAYER_FILTER or get_target_view_layers()
    for view_layer_name in target_layers:
        if scene.view_layers.get(view_layer_name) is None:
            continue
        outputs.append(render_isolated_exr(scene, view_layer_name))

    print(f"Rendered {len(outputs)} isolated EXRs for {BLEND_SCENE_NAME}")


if __name__ == "__main__":
    main()
