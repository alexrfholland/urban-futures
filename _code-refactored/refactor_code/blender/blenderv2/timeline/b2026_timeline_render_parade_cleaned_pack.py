from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import sys

import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract


BLEND_SCENE_NAME = "parade"
SITE_KEY = "trimmed-parade"
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
WORLD_AOV_MATERIAL_NAME = "WORLD_AOV"
TARGET_VIEW_LAYERS = (
    "existing_condition",
    "pathway_state",
    "priority_state",
    "trending_state",
    "bioenvelope_positive",
)
PREVIEW_SAMPLES = 16
PREVIEW_PERCENTAGE = 50
FULL_SAMPLES = 64
FULL_PERCENTAGE = 100
FULL_RESOLUTION = (3840, 2160)
PNG_DEPTH = "8"
EXR_DEPTH = "16"

PREVIEW_OUTPUT_DIR = Path(
    r"D:\2026 Arboreal Futures\data\renders\paraview\parade_lightweight_cleaned_previews"
)
FULL_OUTPUT_DIR = Path(
    r"D:\2026 Arboreal Futures\data\renders\paraview\parade_lightweight_cleaned_4k"
)
EXR_OUTPUT_DIR = Path(
    r"D:\2026 Arboreal Futures\data\renders\paraview\parade_lightweight_cleaned_exr"
)
RUN_PREVIEWS = False
RUN_4K = False
RUN_EXR = True


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


def load_local_module(module_name: str, filename: str):
    file_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_scene_collection_toggles(view_layer_name: str):
    contract = scene_contract.SITE_CONTRACTS[SITE_KEY]
    legacy = contract["legacy"]
    top = contract["top_level"]

    cube_timeline_name = f"{legacy['base_cubes']}_Timeline"
    timeline_positive_bio = f"Year_{SITE_KEY}_timeline_bioenvelope_positive"
    timeline_trending_bio = f"Year_{SITE_KEY}_timeline_bioenvelope_trending"

    if view_layer_name == "bioenvelope_positive":
        show_names = [top["base"], legacy["timeline_base"], timeline_positive_bio]
        hide_names = [
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
        return show_names, hide_names

    if view_layer_name == "priority_state":
        show_names = [
            legacy["timeline_priority"],
        ]
        hide_names = [
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
        return show_names, hide_names

    if view_layer_name == "trending_state":
        show_names = [
            legacy["timeline_trending"],
        ]
        hide_names = [
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
        return show_names, hide_names

    show_names = [
        legacy["timeline_base"],
        legacy["timeline_positive"],
        legacy["timeline_priority"],
        legacy["timeline_trending"],
        timeline_positive_bio,
        timeline_trending_bio,
    ]
    hide_names = [
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
    return show_names, hide_names


def configure_common_render_settings(scene: bpy.types.Scene):
    scene.camera = bpy.data.objects[CAMERA_NAME]
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.film_transparent = True
    scene.render.resolution_x = FULL_RESOLUTION[0]
    scene.render.resolution_y = FULL_RESOLUTION[1]
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"
    scene.sequencer_colorspace_settings.name = "sRGB"


def restore_production_materials(scene: bpy.types.Scene):
    envelope_material = bpy.data.materials.get("Envelope")
    tree_material = bpy.data.materials.get("MINIMAL_RESOURCES")
    world_point_material = bpy.data.materials.get(WORLD_AOV_MATERIAL_NAME)
    if envelope_material is None:
        raise ValueError("Material 'Envelope' was not found.")
    if tree_material is None:
        raise ValueError("Material 'MINIMAL_RESOURCES' was not found.")
    if world_point_material is not None:
        for node_group_name in WORLD_POINT_GROUP_NAMES + WORLD_CUBE_GROUP_NAMES:
            node_group = bpy.data.node_groups.get(node_group_name)
            if node_group is None:
                continue
            for node in node_group.nodes:
                if node.bl_idname == "GeometryNodeSetMaterial" and "Material" in node.inputs:
                    node.inputs["Material"].default_value = world_point_material

    for name in bpy.data.objects.keys():
        obj = bpy.data.objects[name]
        if obj.type != "MESH":
            continue
        if name.startswith("trimmed-parade_positive_envelope__yr") or name.startswith("trimmed-parade_trending_envelope__yr"):
            if len(obj.material_slots) == 0:
                obj.data.materials.append(envelope_material)
            else:
                obj.material_slots[0].material = envelope_material
        elif name.startswith("TreePositions_trimmed-parade_") or name.startswith("LogPositions_trimmed-parade_"):
            if len(obj.material_slots) > 0:
                obj.material_slots[0].material = tree_material
        else:
            continue


def render_view_layer(
    scene: bpy.types.Scene,
    view_layer_name: str,
    output_dir: Path,
    *,
    file_format: str,
    color_depth: str,
    resolution_percentage: int,
    samples: int,
    suffix: str,
):
    if scene.view_layers.get(view_layer_name) is None:
        raise ValueError(f"View layer '{view_layer_name}' not found in scene '{scene.name}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    extension = ".exr" if file_format == "OPEN_EXR_MULTILAYER" else ".png"
    output_path = output_dir / f"{scene.name}_{CAMERA_NAME}_{view_layer_name}_{suffix}{extension}"

    original_layer_use = {view_layer.name: view_layer.use for view_layer in scene.view_layers}
    original_percentage = scene.render.resolution_percentage
    original_filepath = scene.render.filepath
    original_format = scene.render.image_settings.file_format
    original_color_mode = scene.render.image_settings.color_mode
    original_color_depth = scene.render.image_settings.color_depth
    original_samples = getattr(scene.cycles, "samples", None) if hasattr(scene, "cycles") else None
    original_preview_samples = getattr(scene.cycles, "preview_samples", None) if hasattr(scene, "cycles") else None

    show_names, hide_names = build_scene_collection_toggles(view_layer_name)
    shown_state = set_collection_render_state(show_names, hide_render=False)
    hidden_state = set_collection_render_state(hide_names, hide_render=True)

    scene.render.filepath = str(output_path)
    scene.render.resolution_percentage = resolution_percentage
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = color_depth
    if file_format == "OPEN_EXR_MULTILAYER":
        scene.render.image_settings.exr_codec = "ZIP"

    if hasattr(scene, "cycles"):
        scene.cycles.samples = samples
        scene.cycles.preview_samples = samples

    for view_layer in scene.view_layers:
        view_layer.use = view_layer.name == view_layer_name

    try:
        bpy.ops.render.render(
            write_still=True,
            scene=scene.name,
            layer=view_layer_name,
            use_viewport=False,
        )
    finally:
        scene.render.resolution_percentage = original_percentage
        scene.render.filepath = original_filepath
        scene.render.image_settings.file_format = original_format
        scene.render.image_settings.color_mode = original_color_mode
        scene.render.image_settings.color_depth = original_color_depth
        if hasattr(scene, "cycles") and original_samples is not None:
            scene.cycles.samples = original_samples
            scene.cycles.preview_samples = original_preview_samples
        for view_layer in scene.view_layers:
            view_layer.use = original_layer_use.get(view_layer.name, True)
        restore_collection_render_state(shown_state)
        restore_collection_render_state(hidden_state)

    if not output_path.exists():
        raise RuntimeError(f"Render did not produce {output_path}")

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(
        f"RENDER_DONE scene={scene.name} layer={view_layer_name} format={file_format} "
        f"path={output_path} sha256={digest}"
    )
    return output_path


def print_aov_summary(scene: bpy.types.Scene):
    for view_layer in scene.view_layers:
        aov_names = [(aov.name, aov.type) for aov in view_layer.aovs]
        print(f"AOVS layer={view_layer.name} values={aov_names}")


def main():
    scene = bpy.data.scenes.get(BLEND_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{BLEND_SCENE_NAME}' not found in {bpy.data.filepath}")
    if bpy.data.objects.get(CAMERA_NAME) is None:
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}")

    configure_common_render_settings(scene)
    print_aov_summary(scene)
    restore_production_materials(scene)

    if RUN_PREVIEWS:
        for view_layer_name in TARGET_VIEW_LAYERS:
            render_view_layer(
                scene,
                view_layer_name,
                PREVIEW_OUTPUT_DIR,
                file_format="PNG",
                color_depth=PNG_DEPTH,
                resolution_percentage=PREVIEW_PERCENTAGE,
                samples=PREVIEW_SAMPLES,
                suffix="preview_rgba_standard",
            )

    if RUN_4K:
        for view_layer_name in TARGET_VIEW_LAYERS:
            render_view_layer(
                scene,
                view_layer_name,
                FULL_OUTPUT_DIR,
                file_format="PNG",
                color_depth=PNG_DEPTH,
                resolution_percentage=FULL_PERCENTAGE,
                samples=FULL_SAMPLES,
                suffix="4k_rgba_standard",
            )

    if RUN_EXR:
        exr_setup = load_local_module(
            "b2026_timeline_setup_view_layer_exr_outputs",
            "b2026_timeline_setup_view_layer_exr_outputs.py",
        )
        exr_setup.main()
        scene.render.use_compositing = True
        scene.render.use_sequencer = False
        scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = EXR_DEPTH
        scene.render.film_transparent = True
        scene.render.resolution_percentage = FULL_PERCENTAGE
        if hasattr(scene, "cycles"):
            scene.cycles.samples = FULL_SAMPLES
            scene.cycles.preview_samples = FULL_SAMPLES

        bpy.ops.render.render(
            write_still=True,
            scene=scene.name,
            use_viewport=False,
        )

    print("Parade cleaned preview, 4K PNG, and EXR render pack complete")


if __name__ == "__main__":
    main()
