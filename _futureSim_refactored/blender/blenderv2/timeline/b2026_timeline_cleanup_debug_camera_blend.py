from __future__ import annotations

from pathlib import Path

import bpy


BLEND_PATH = Path(
    r"D:\2026 Arboreal Futures\data\renders\timeslices\camera_tests\2026 futures timeslice debug camera framing v3.blend"
)

SITE_SPECS = [
    {
        "scene_name": "city",
        "original_camera_name": "city_good_timeslice_cam",
        "zoom_camera_name": "city_visible_zoom_test_cam",
        "active_collection_name": "city_debug_camera_framing",
        "archive_collection_name": "city_camera_archive",
        "original_display_name": "city - camera - time slice",
        "zoom_display_name": "city - camera - time slice - zoom",
    },
    {
        "scene_name": "parade",
        "original_camera_name": "parade_good_timeslice_cam",
        "zoom_camera_name": "parade_visible_zoom_test_cam",
        "active_collection_name": "parade_debug_camera_framing",
        "archive_collection_name": "parade_camera_archive",
        "original_display_name": "parade - camera - time slice",
        "zoom_display_name": "parade - camera - time slice - zoom",
    },
    {
        "scene_name": "street",
        "original_camera_name": "street_good_timeslice_cam",
        "zoom_camera_name": "street_visible_zoom_test_cam",
        "active_collection_name": "street_debug_camera_framing",
        "archive_collection_name": "street_camera_archive",
        "original_display_name": "street - camera - time slice",
        "zoom_display_name": "street - camera - time slice - zoom",
    },
]


def ensure_child_collection(parent: bpy.types.Collection, child_name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(child_name)
    if collection is None:
        collection = bpy.data.collections.new(child_name)
    if child_name not in parent.children:
        parent.children.link(collection)
    return collection


def relink_to_single_collection(obj: bpy.types.Object, target_collection: bpy.types.Collection) -> None:
    for collection in list(obj.users_collection):
        if collection != target_collection:
            collection.objects.unlink(obj)
    if obj.name not in target_collection.objects:
        target_collection.objects.link(obj)


def clean_site(spec: dict) -> None:
    scene = bpy.data.scenes[spec["scene_name"]]
    active_collection = bpy.data.collections[spec["active_collection_name"]]
    archive_collection = ensure_child_collection(scene.collection, spec["archive_collection_name"])

    original_camera = bpy.data.objects[spec["original_camera_name"]]
    zoom_camera = bpy.data.objects[spec["zoom_camera_name"]]

    keeper_names = {original_camera.name, zoom_camera.name}

    for obj in list(active_collection.objects):
        if obj.type != "CAMERA":
            continue
        if obj.name in keeper_names:
            continue
        relink_to_single_collection(obj, archive_collection)
        obj.hide_viewport = True
        obj.hide_render = True

    for obj in list(bpy.data.objects):
        if obj.type != "CAMERA":
            continue
        if not obj.name.startswith(f"{spec['scene_name']}_"):
            continue
        if obj.name in keeper_names:
            continue
        relink_to_single_collection(obj, archive_collection)
        obj.hide_viewport = True
        obj.hide_render = True

    relink_to_single_collection(original_camera, active_collection)
    relink_to_single_collection(zoom_camera, active_collection)

    original_camera.hide_viewport = False
    original_camera.hide_render = False
    zoom_camera.hide_viewport = False
    zoom_camera.hide_render = False

    original_camera.name = spec["original_display_name"]
    zoom_camera.name = spec["zoom_display_name"]
    original_camera.data.name = f"{spec['original_display_name']} data"
    zoom_camera.data.name = f"{spec['zoom_display_name']} data"

    scene.camera = zoom_camera

    print(
        f"CLEANED {scene.name} original={original_camera.name} "
        f"zoom={zoom_camera.name} archive={archive_collection.name}"
    )


def main() -> None:
    for spec in SITE_SPECS:
        clean_site(spec)

    if bpy.data.is_saved:
        bpy.ops.wm.save_mainfile()


if __name__ == "__main__":
    main()
