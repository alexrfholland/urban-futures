from __future__ import annotations

import math
import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
SCENE_NAME = "city"
CAMERA_NAME = "WorldCam_3x"
SUN_NAME = "BaselineShadowSun"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "baseline_renders"
    / "city_baseline_pathway_8k_zoom3x_shadow.png"
)


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


def mute_file_outputs(scene: bpy.types.Scene) -> int:
    if not scene.use_nodes or scene.node_tree is None:
        return 0
    muted = 0
    for node in scene.node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile" and not node.mute:
            node.mute = True
            muted += 1
    return muted


def ensure_sun(scene: bpy.types.Scene) -> bpy.types.Object:
    sun_object = bpy.data.objects.get(SUN_NAME)
    if sun_object is None or sun_object.type != "LIGHT":
        light_data = bpy.data.lights.new(name=SUN_NAME, type="SUN")
        sun_object = bpy.data.objects.new(SUN_NAME, light_data)
        scene.collection.objects.link(sun_object)
    else:
        light_data = sun_object.data

    light_data.type = "SUN"
    light_data.energy = 1.2
    light_data.angle = math.radians(6.0)
    sun_object.rotation_euler = (
        math.radians(52.0),
        0.0,
        math.radians(144.0),
    )
    sun_object.location = (0.0, 0.0, 0.0)
    sun_object.hide_render = False
    sun_object.hide_viewport = False
    return sun_object


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
    background.inputs[1].default_value = 0.35


def configure_render(scene: bpy.types.Scene, output_path: Path) -> None:
    scene.render.resolution_x = env_int("B2026_RENDER_X", 7680)
    scene.render.resolution_y = env_int("B2026_RENDER_Y", 4320)
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    if hasattr(scene.view_settings, "look"):
        scene.view_settings.look = "None"
    scene.render.film_transparent = True
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = env_int("B2026_CYCLES_SAMPLES", scene.cycles.samples)
        scene.cycles.use_denoising = True
    output_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    output_path = env_path("B2026_OUTPUT_PATH", DEFAULT_OUTPUT)
    scene = get_scene()
    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"Camera '{CAMERA_NAME}' was not found")

    muted = mute_file_outputs(scene)
    ensure_sun(scene)
    configure_world_fill(scene)
    configure_render(scene, output_path)
    scene.camera = camera

    print(
        f"[baseline_zoom3x_shadows] Rendering with {camera.name} to {output_path} (muted {muted} file outputs)",
        flush=True,
    )
    bpy.ops.render.render(write_still=True, scene=scene.name)
    bpy.ops.wm.save_mainfile()
    print("[baseline_zoom3x_shadows] Saved blend with shadow sun", flush=True)


if __name__ == "__main__":
    main()
