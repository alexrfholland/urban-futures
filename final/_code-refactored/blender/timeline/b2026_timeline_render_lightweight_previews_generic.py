from __future__ import annotations

from pathlib import Path
import os
import sys

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract


SCENE_NAME = os.environ["B2026_SCENE_NAME"]
SITE_KEY = os.environ["B2026_SITE_KEY"]
CAMERA_NAME = os.environ["B2026_CAMERA_NAME"]
OUTPUT_DIR = Path(os.environ["B2026_OUTPUT_DIR"])
OUTPUT_BASENAME = os.environ.get("B2026_OUTPUT_BASENAME", SCENE_NAME)
PREVIEW_SAMPLES = int(os.environ.get("B2026_SAMPLES", "16"))
PREVIEW_PERCENTAGE = int(os.environ.get("B2026_RES_PERCENT", "50"))
TARGET_VIEW_LAYERS = (
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
    timeline_positive_world = scene_contract.get_timeline_world_collection_name(SITE_KEY, "positive")
    timeline_trending_world = scene_contract.get_timeline_world_collection_name(SITE_KEY, "trending")

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


def main():
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {bpy.data.filepath}")

    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scene.camera = camera
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"

    original_percentage = scene.render.resolution_percentage
    original_samples = getattr(scene.cycles, "samples", None) if hasattr(scene, "cycles") else None
    original_preview_samples = getattr(scene.cycles, "preview_samples", None) if hasattr(scene, "cycles") else None
    original_layer_use = {view_layer.name: view_layer.use for view_layer in scene.view_layers}

    target_layers = VIEW_LAYER_FILTER or TARGET_VIEW_LAYERS
    for view_layer_name in target_layers:
        if scene.view_layers.get(view_layer_name) is None:
            continue

        output_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}_{view_layer_name}_preview.png"
        scene.render.filepath = str(output_path)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"

        show_names, hide_names = build_scene_collection_toggles(view_layer_name)
        shown_state = set_collection_render_state(show_names, hide_render=False)
        hidden_state = set_collection_render_state(hide_names, hide_render=True)

        scene.render.resolution_percentage = PREVIEW_PERCENTAGE
        if hasattr(scene, "cycles"):
            scene.cycles.samples = PREVIEW_SAMPLES
            scene.cycles.preview_samples = PREVIEW_SAMPLES

        for view_layer in scene.view_layers:
            view_layer.use = view_layer.name == view_layer_name

        try:
            bpy.ops.render.render(write_still=True, scene=scene.name, layer=view_layer_name, use_viewport=False)
        finally:
            restore_collection_render_state(shown_state)
            restore_collection_render_state(hidden_state)

    scene.render.resolution_percentage = original_percentage
    if hasattr(scene, "cycles") and original_samples is not None:
        scene.cycles.samples = original_samples
        scene.cycles.preview_samples = original_preview_samples
    for view_layer in scene.view_layers:
        view_layer.use = original_layer_use.get(view_layer.name, True)

    print(f"Rendered previews for {SCENE_NAME} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
