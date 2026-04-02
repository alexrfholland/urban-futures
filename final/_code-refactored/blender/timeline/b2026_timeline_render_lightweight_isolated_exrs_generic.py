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
TARGET_VIEW_LAYERS = (
    "existing_condition",
    "pathway_state",
    "priority_state",
    "trending_state",
    "bioenvelope_positive",
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


def ensure_scene_camera(scene: bpy.types.Scene):
    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None:
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}")
    scene.camera = camera
    return camera


def ensure_scene_world_and_mist(scene: bpy.types.Scene):
    world = ensure_world(MIST_WORLD_NAME)
    scene.world = world
    mist = world.mist_settings
    mist.use_mist = True
    mist.start = MIST_START
    mist.depth = MIST_DEPTH
    mist.falloff = MIST_FALLOFF
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
    for layer_name in TARGET_VIEW_LAYERS:
        view_layer = scene.view_layers.get(layer_name)
        if view_layer is None:
            continue
        if hasattr(view_layer, "use"):
            view_layer.use = True
        for aov_name, aov_type in AOV_SPECS:
            ensure_aov(view_layer, aov_name, aov_type)


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


def build_scene_collection_toggles(view_layer_name: str):
    contract = scene_contract.SITE_CONTRACTS[SITE_KEY]
    legacy = contract["legacy"]
    top = contract["top_level"]

    cube_timeline_name = f"{legacy['base_cubes']}_Timeline"
    timeline_positive_bio = f"Year_{SITE_KEY}_timeline_bioenvelope_positive"
    timeline_trending_bio = f"Year_{SITE_KEY}_timeline_bioenvelope_trending"

    if view_layer_name == "bioenvelope_positive":
        return [top["base"], legacy["timeline_base"], timeline_positive_bio], [
            top["base_cubes"],
            top["positive"],
            top["priority"],
            top["trending"],
            top["bio_trending"],
            legacy["base"],
            legacy["base_cubes"],
            cube_timeline_name,
            legacy["positive"],
            legacy["timeline_positive"],
            legacy["priority"],
            legacy["timeline_priority"],
            legacy["trending"],
            legacy["timeline_trending"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            timeline_trending_bio,
        ]

    if view_layer_name == "priority_state":
        return [legacy["timeline_priority"]], [
            top["base"],
            top["base_cubes"],
            top["positive"],
            top["trending"],
            top["bio_positive"],
            top["bio_trending"],
            legacy["base"],
            legacy["timeline_base"],
            legacy["base_cubes"],
            cube_timeline_name,
            legacy["positive"],
            legacy["timeline_positive"],
            legacy["trending"],
            legacy["timeline_trending"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            timeline_positive_bio,
            timeline_trending_bio,
        ]

    if view_layer_name == "trending_state":
        return [legacy["timeline_trending"]], [
            top["base"],
            top["base_cubes"],
            top["positive"],
            top["priority"],
            top["bio_positive"],
            top["bio_trending"],
            legacy["base"],
            legacy["timeline_base"],
            legacy["base_cubes"],
            cube_timeline_name,
            legacy["positive"],
            legacy["timeline_positive"],
            legacy["priority"],
            legacy["timeline_priority"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            timeline_positive_bio,
            timeline_trending_bio,
        ]

    return [
        legacy["timeline_base"],
        legacy["timeline_positive"],
        legacy["timeline_priority"],
        legacy["timeline_trending"],
        timeline_positive_bio,
        timeline_trending_bio,
    ], [
        top["base_cubes"],
        legacy["base"],
        legacy["base_cubes"],
        cube_timeline_name,
        legacy["positive"],
        legacy["priority"],
        legacy["trending"],
        legacy["bio_positive"],
        legacy["bio_trending"],
    ]


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
    show_names, hide_names = build_scene_collection_toggles(view_layer_name)
    shown_state = set_collection_render_state(show_names, hide_render=False)
    hidden_state = set_collection_render_state(hide_names, hide_render=True)

    try:
        with bpy.context.temp_override(scene=temp_scene, view_layer=target_layer):
            bpy.ops.render.render(scene=temp_scene.name, layer=view_layer_name, use_viewport=False)
        rename_temp_exr_output(output_path)
    finally:
        restore_collection_render_state(shown_state)
        restore_collection_render_state(hidden_state)
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
    target_layers = VIEW_LAYER_FILTER or TARGET_VIEW_LAYERS
    for view_layer_name in target_layers:
        if scene.view_layers.get(view_layer_name) is None:
            continue
        outputs.append(render_isolated_exr(scene, view_layer_name))

    print(f"Rendered {len(outputs)} isolated EXRs for {BLEND_SCENE_NAME}")


if __name__ == "__main__":
    main()
