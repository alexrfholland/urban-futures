from __future__ import annotations

import math
import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
SCENE_NAME = "city"
VIEW_LAYER_NAME = "pathway_state"
CAMERA_NAME = "WorldCam_3x"
SUN_NAME = "BaselineShadowSun"
OUTPUT_PATH = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "baseline_renders"
    / "city_baseline_pathway_preview_zoom3x_shadow_catcher.png"
)
GROUND_OBJECTS = {"city_highResRoad.001", "city_highResRoad.001_cubes"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def get_scene() -> bpy.types.Scene:
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' was not found")
    return scene


def ensure_sun(scene: bpy.types.Scene) -> None:
    sun_object = bpy.data.objects.get(SUN_NAME)
    if sun_object is None or sun_object.type != "LIGHT":
        light_data = bpy.data.lights.new(name=SUN_NAME, type="SUN")
        sun_object = bpy.data.objects.new(SUN_NAME, light_data)
        scene.collection.objects.link(sun_object)
    light_data = sun_object.data
    light_data.type = "SUN"
    light_data.energy = 2.5
    light_data.angle = math.radians(2.0)
    sun_object.rotation_euler = (
        math.radians(28.0),
        0.0,
        math.radians(144.0),
    )
    sun_object.hide_render = False
    sun_object.hide_viewport = False


def configure_shadow_catchers(scene: bpy.types.Scene) -> None:
    for obj in scene.objects:
        if obj.name in GROUND_OBJECTS:
            obj.hide_render = False
            obj.visible_camera = True
            obj.visible_shadow = True
            if hasattr(obj, "is_shadow_catcher"):
                obj.is_shadow_catcher = True
        elif obj.pass_index == 3:
            obj.hide_render = False
            obj.visible_camera = True
            obj.visible_shadow = True


def enable_only_target_view_layer(scene: bpy.types.Scene) -> None:
    for view_layer in scene.view_layers:
        view_layer.use = view_layer.name == VIEW_LAYER_NAME
    view_layer = scene.view_layers[VIEW_LAYER_NAME]
    view_layer.use_pass_shadow = True
    view_layer.use_pass_combined = True
    view_layer.cycles.use_pass_shadow_catcher = True


def configure_compositor(scene: bpy.types.Scene) -> None:
    scene.use_nodes = True
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    render_node = nodes.new("CompositorNodeRLayers")
    render_node.scene = scene
    render_node.layer = VIEW_LAYER_NAME
    render_node.location = (-400, 0)

    white_node = nodes.new("CompositorNodeRGB")
    white_node.location = (-400, 220)
    white_node.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)

    alpha_over = nodes.new("CompositorNodeAlphaOver")
    alpha_over.location = (-80, 80)
    alpha_over.use_premultiply = True

    composite = nodes.new("CompositorNodeComposite")
    composite.location = (220, 80)

    links.new(white_node.outputs[0], alpha_over.inputs[1])
    links.new(render_node.outputs["Shadow Catcher"], alpha_over.inputs[2])
    links.new(alpha_over.outputs[0], composite.inputs[0])


def configure_render(scene: bpy.types.Scene, output_path: Path) -> None:
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = env_int("B2026_RENDER_X", 1920)
    scene.render.resolution_y = env_int("B2026_RENDER_Y", 1080)
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    if hasattr(scene.view_settings, "look"):
        scene.view_settings.look = "None"
    scene.render.film_transparent = False
    scene.render.use_compositing = True
    scene.cycles.samples = env_int("B2026_CYCLES_SAMPLES", 64)
    scene.cycles.use_denoising = True
    output_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    scene = get_scene()
    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"Camera '{CAMERA_NAME}' was not found")

    ensure_sun(scene)
    configure_shadow_catchers(scene)
    enable_only_target_view_layer(scene)
    configure_compositor(scene)
    configure_render(scene, env_path("B2026_OUTPUT_PATH", OUTPUT_PATH))
    scene.camera = camera

    print(
        f"[baseline_zoom3x_shadow_catcher] Rendering {VIEW_LAYER_NAME} shadow catcher preview to {scene.render.filepath}",
        flush=True,
    )
    bpy.ops.render.render(write_still=True, scene=scene.name)


if __name__ == "__main__":
    main()
