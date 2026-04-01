from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Matrix


# ONE-OFF REBUILD SCRIPT.
# This wipes the current file's scenes and rebuilds the stripped debug blend
# from the heavyweight source blends. Do not run this during normal camera
# iteration inside an already-prepared debug blend.


OUTPUT_BLEND = Path(
    r"D:\2026 Arboreal Futures\data\renders\timeslices\camera_tests\2026 futures timeslice debug camera framing.blend"
)

SITE_SPECS = [
    {
        "scene_name": "parade",
        "source_blend": Path(r"D:\2026 Arboreal Futures\data\2026 futures parade lightweight cleaned.blend"),
        "camera_name": "paraview_camera_parade__shared_fit_ground_bottom_height_parade",
        "point_object_names": [
            "trimmed-parade_base__timeline",
            "trimmed-parade_highResRoad__timeline",
        ],
    },
    {
        "scene_name": "city",
        "source_blend": Path(r"D:\2026 Arboreal Futures\data\2026 futures city lightweight cleaned.blend"),
        "camera_name": "paraview_camera_city__shared_fit_ground_bottom_height_parade",
        "point_object_names": [
            "city_buildings.001__timeline",
            "city_highResRoad.001__timeline",
        ],
    },
    {
        "scene_name": "street",
        "source_blend": Path(r"D:\2026 Arboreal Futures\data\2026 futures street lightweight cleaned.blend"),
        "camera_name": "paraview_camera_street__shared_fit_ground_bottom_height_parade",
        "point_object_names": [
            "uni_base__timeline",
            "uni_highResRoad__timeline",
        ],
    },
]


def remove_default_content() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scenes = list(bpy.data.scenes)
    if not scenes:
        raise ValueError("No scenes available to reuse.")

    scene_to_keep = scenes[0]
    for scene in scenes[1:]:
        bpy.data.scenes.remove(scene)
    scene_to_keep.name = "Scene"

    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def load_objects(source_blend: Path, object_names: list[str]) -> list[bpy.types.Object]:
    with bpy.data.libraries.load(str(source_blend), link=False) as (data_from, data_to):
        available = set(data_from.objects)
        missing = [name for name in object_names if name not in available]
        if missing:
            raise ValueError(f"Missing objects in {source_blend}: {missing}")
        data_to.objects = object_names
    loaded = []
    for obj in data_to.objects:
        if obj is None:
            raise ValueError(f"Failed to load object from {source_blend}")
        loaded.append(obj)
    return loaded


def repair_unparented_transform(obj: bpy.types.Object) -> None:
    if obj.parent is not None:
        return
    identity = Matrix.Identity(4)
    basis = obj.matrix_basis.copy()
    world = obj.matrix_world.copy()
    if world == identity and basis != identity:
        obj.matrix_world = basis


def build_scene_from_spec(spec: dict, reuse_first_scene: bool) -> bpy.types.Scene:
    if reuse_first_scene:
        scene = bpy.data.scenes["Scene"]
        scene.name = spec["scene_name"]
    else:
        scene = bpy.data.scenes.new(spec["scene_name"])

    scene.render.engine = "CYCLES"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"

    collection = bpy.data.collections.new(f"{spec['scene_name']}_debug_camera_framing")
    scene.collection.children.link(collection)

    object_names = [spec["camera_name"], *spec["point_object_names"]]
    objects = load_objects(spec["source_blend"], object_names)
    for obj in objects:
        collection.objects.link(obj)
        obj.hide_render = False
        obj.hide_viewport = False
        repair_unparented_transform(obj)

    camera = next(obj for obj in objects if obj.name == spec["camera_name"])
    scene.camera = camera
    return scene


def main() -> None:
    remove_default_content()

    first = True
    for spec in SITE_SPECS:
        build_scene_from_spec(spec, reuse_first_scene=first)
        first = False

    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))
    print(f"SAVED_DEBUG_BLEND {OUTPUT_BLEND}")


if __name__ == "__main__":
    main()
