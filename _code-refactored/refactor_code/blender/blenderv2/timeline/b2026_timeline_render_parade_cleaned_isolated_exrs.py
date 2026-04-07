from __future__ import annotations

from pathlib import Path
import hashlib
import os
import sys

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


BLEND_SCENE_NAME = "parade"
CAMERA_NAME = "paraview_camera_parade"
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
OUTPUT_DIR = Path(
    os.environ.get(
        "B2026_OUTPUT_DIR",
        r"D:\2026 Arboreal Futures\data\renders\paraview\parade_lightweight_cleaned_exr_isolated",
    )
)
FULL_RESOLUTION = (
    int(os.environ.get("B2026_RES_X", "3840")),
    int(os.environ.get("B2026_RES_Y", "2160")),
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
)


def ensure_material(name: str):
    material = bpy.data.materials.get(name)
    if material is None:
        raise ValueError(f"Required material '{name}' was not found")
    return material


def restore_instancer_materials():
    target_material = ensure_material("MINIMAL_RESOURCES")
    updated = []
    for node_group in bpy.data.node_groups:
        if not node_group.name.startswith(("tree_", "log_", "pole_", "instance_template")):
            continue
        changed = False
        for node in node_group.nodes:
            if node.bl_idname != "GeometryNodeSetMaterial":
                continue
            if "Material" not in node.inputs:
                continue
            if node.inputs["Material"].default_value != target_material:
                node.inputs["Material"].default_value = target_material
                changed = True
        if changed:
            updated.append(node_group.name)
    print(f"Instancer node groups reset to MINIMAL_RESOURCES: {updated}")


def restore_bioenvelope_materials():
    target_material = ensure_material("Envelope")
    updated = []
    for obj in bpy.data.objects:
        if "_envelope__yr" not in obj.name:
            continue
        if not obj.name.startswith("trimmed-parade_"):
            continue
        data = getattr(obj, "data", None)
        materials = getattr(data, "materials", None)
        if materials is None:
            continue
        if len(materials) == 0:
            materials.append(target_material)
            updated.append(obj.name)
            continue
        changed = False
        for index in range(len(materials)):
            if materials[index] != target_material:
                materials[index] = target_material
                changed = True
        if changed:
            updated.append(obj.name)
    print(f"Bioenvelope objects reset to Envelope: {updated}")


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
            raise ValueError("Could not find a source world point-cloud material.")
        material = source.copy()
        material.name = WORLD_AOV_MATERIAL_NAME

    if not material.use_nodes or material.node_tree is None:
        raise ValueError(f"Material '{material.name}' does not have a node tree.")

    node_tree = material.node_tree
    nodes = node_tree.nodes
    base_x = max((node.location.x for node in nodes), default=400.0) + 260.0
    base_y = max((node.location.y for node in nodes), default=200.0)

    for index, (aov_name, attr_name, aov_type) in enumerate(WORLD_POINT_AOV_SPECS):
        y = base_y - (index * 220.0)

        geometry_attr_name = f"World Geometry Attribute {aov_name}"
        geometry_attr = nodes.get(geometry_attr_name)
        if geometry_attr is None or geometry_attr.bl_idname != "ShaderNodeAttribute":
            geometry_attr = nodes.new("ShaderNodeAttribute")
        geometry_attr.name = geometry_attr_name
        geometry_attr.label = geometry_attr_name
        geometry_attr.location = (base_x - 520.0, y + 60.0)
        geometry_attr.attribute_name = attr_name
        if hasattr(geometry_attr, "attribute_type"):
            geometry_attr.attribute_type = "GEOMETRY"

        instancer_attr_name = f"World Instancer Attribute {aov_name}"
        instancer_attr = nodes.get(instancer_attr_name)
        if instancer_attr is None or instancer_attr.bl_idname != "ShaderNodeAttribute":
            instancer_attr = nodes.new("ShaderNodeAttribute")
        instancer_attr.name = instancer_attr_name
        instancer_attr.label = instancer_attr_name
        instancer_attr.location = (base_x - 520.0, y - 60.0)
        instancer_attr.attribute_name = attr_name
        if hasattr(instancer_attr, "attribute_type"):
            instancer_attr.attribute_type = "INSTANCER"

        aov_name_node = f"World Point AOV {aov_name}"
        aov_node = nodes.get(aov_name_node)
        if aov_node is None or aov_node.bl_idname != "ShaderNodeOutputAOV":
            aov_node = nodes.new("ShaderNodeOutputAOV")
        aov_node.name = aov_name_node
        aov_node.label = aov_name_node
        aov_node.location = (base_x, y)
        aov_node.aov_name = aov_name

        if aov_type == "COLOR":
            mix_name = f"World Attribute Color Mix {aov_name}"
            mix_node = nodes.get(mix_name)
            if mix_node is None or mix_node.bl_idname != "ShaderNodeMix":
                mix_node = nodes.new("ShaderNodeMix")
            mix_node.name = mix_name
            mix_node.label = mix_name
            mix_node.location = (base_x - 160.0, y)
            if hasattr(mix_node, "data_type"):
                mix_node.data_type = "RGBA"
            mix_node.inputs["Factor"].default_value = 1.0
            ensure_link(node_tree, geometry_attr.outputs["Color"], mix_node.inputs["A"])
            ensure_link(node_tree, instancer_attr.outputs["Color"], mix_node.inputs["B"])
            ensure_link(node_tree, mix_node.outputs["Result"], aov_node.inputs["Color"])
        else:
            max_name = f"World Attribute Max {aov_name}"
            max_node = nodes.get(max_name)
            if max_node is None or max_node.bl_idname != "ShaderNodeMath":
                max_node = nodes.new("ShaderNodeMath")
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
    updated = []
    for node_group_name in WORLD_POINT_GROUP_NAMES + WORLD_CUBE_GROUP_NAMES:
        node_group = bpy.data.node_groups.get(node_group_name)
        if node_group is None:
            continue
        for node in node_group.nodes:
            if node.bl_idname != "GeometryNodeSetMaterial" or "Material" not in node.inputs:
                continue
            if node.inputs["Material"].default_value != target_material:
                node.inputs["Material"].default_value = target_material
                updated.append(node_group.name)
    print(f"World node groups reset to {WORLD_AOV_MATERIAL_NAME}: {sorted(set(updated))}")


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
    for layer_name in ("pathway_state", "existing_condition", "priority_state", "trending_state", "bioenvelope_positive"):
        view_layer = scene.view_layers.get(layer_name)
        if view_layer is None:
            continue
        for aov_name, aov_type in AOV_SPECS:
            ensure_aov(view_layer, aov_name, aov_type)


def ensure_render_passes(view_layer: bpy.types.ViewLayer):
    toggles = (
        "use_pass_combined",
        "use_pass_z",
        "use_pass_mist",
        "use_pass_normal",
        "use_pass_object_index",
        "use_pass_material_index",
        "use_pass_ambient_occlusion",
    )
    for attr in toggles:
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
    import b2026_timeline_scene_contract as scene_contract

    contract = scene_contract.SITE_CONTRACTS["trimmed-parade"]
    legacy = contract["legacy"]
    top = contract["top_level"]

    cube_timeline_name = f"{legacy['base_cubes']}_Timeline"
    timeline_positive_bio = f"Year_trimmed-parade_timeline_bioenvelope_positive"
    timeline_trending_bio = f"Year_trimmed-parade_timeline_bioenvelope_trending"

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
    temp_scene.render.use_compositing = False
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

    for layer in list(temp_scene.view_layers):
        if layer.name != view_layer_name:
            temp_scene.view_layers.remove(layer)

    target_layer = temp_scene.view_layers.get(view_layer_name)
    if target_layer is None:
        raise ValueError(f"Failed to isolate view layer '{view_layer_name}'")
    ensure_render_passes(target_layer)
    return temp_scene


def render_isolated_exr(scene: bpy.types.Scene, view_layer_name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{scene.name}_{CAMERA_NAME}_{view_layer_name}_4k_multilayer_isolated.exr"

    temp_scene = clone_scene_for_layer(scene, view_layer_name)
    temp_scene.render.filepath = str(output_path)
    show_names, hide_names = build_scene_collection_toggles(view_layer_name)
    shown_state = set_collection_render_state(show_names, hide_render=False)
    hidden_state = set_collection_render_state(hide_names, hide_render=True)

    try:
        bpy.ops.render.render(
            write_still=True,
            scene=temp_scene.name,
            layer=view_layer_name,
            use_viewport=False,
        )
    finally:
        restore_collection_render_state(shown_state)
        restore_collection_render_state(hidden_state)
        bpy.data.scenes.remove(temp_scene)

    if not output_path.exists():
        raise RuntimeError(f"Render did not produce {output_path}")

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"RENDER_DONE layer={view_layer_name} path={output_path} sha256={digest}")
    return output_path


def print_aov_summary(scene: bpy.types.Scene):
    for view_layer in scene.view_layers:
        ensure_render_passes(view_layer)
        aov_names = [(aov.name, aov.type) for aov in view_layer.aovs]
        print(f"AOVS layer={view_layer.name} values={aov_names}")


def main():
    scene = bpy.data.scenes.get(BLEND_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{BLEND_SCENE_NAME}' not found in {bpy.data.filepath}")
    if bpy.data.objects.get(CAMERA_NAME) is None:
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}")

    restore_instancer_materials()
    restore_world_point_materials()
    restore_bioenvelope_materials()
    ensure_target_layer_aovs(scene)
    print_aov_summary(scene)

    outputs = []
    for view_layer_name in TARGET_VIEW_LAYERS:
        outputs.append(render_isolated_exr(scene, view_layer_name))

    print(f"Rendered {len(outputs)} isolated parade multilayer EXRs")


if __name__ == "__main__":
    main()
