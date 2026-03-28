from __future__ import annotations

import math
import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
SCENE_NAME = "city"
CAMERA_NAME = "WorldCam_3x"
SUN_NAME = "BaselineShadowSun"
OUTPUT_PATH = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "baseline_renders"
    / "city_baseline_pathway_preview_zoom3x_shadow_only.png"
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


def ensure_white_ground_material() -> bpy.types.Material:
    material = bpy.data.materials.get("BaselineShadowGround")
    if material is None:
        material = bpy.data.materials.new("BaselineShadowGround")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


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


def configure_world_fill(scene: bpy.types.Scene) -> None:
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise ValueError("World background node was not found")
    background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs[1].default_value = 0.0


def configure_visibility(scene: bpy.types.Scene) -> None:
    ground_material = ensure_white_ground_material()
    for obj in scene.objects:
        if obj.type in {"CAMERA", "LIGHT"}:
            continue

        obj.hide_render = False
        if obj.name in GROUND_OBJECTS or obj.pass_index == 1:
            obj.visible_camera = True
            obj.visible_shadow = True
            if obj.type == "MESH":
                obj.data.materials.clear()
                obj.data.materials.append(ground_material)
        elif obj.pass_index == 3:
            obj.visible_camera = False
            obj.visible_shadow = True
        else:
            obj.hide_render = True


def configure_render(scene: bpy.types.Scene, output_path: Path) -> None:
    scene.use_nodes = False
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
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = env_int("B2026_CYCLES_SAMPLES", 64)
        scene.cycles.use_denoising = True
    output_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    scene = get_scene()
    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"Camera '{CAMERA_NAME}' was not found")

    ensure_sun(scene)
    configure_world_fill(scene)
    configure_visibility(scene)
    configure_render(scene, env_path("B2026_OUTPUT_PATH", OUTPUT_PATH))
    scene.camera = camera
    bpy.ops.render.render(write_still=True, scene=scene.name)


if __name__ == "__main__":
    main()
