from __future__ import annotations

import bpy


SOURCE_BLEND = r"D:\2026 Arboreal Futures\data\2026 futures heroes6.blend"
TARGET_BLEND = r"D:\2026 Arboreal Futures\data\2026 futures street lightweight cleaned.blend"
SOURCE_SCENE_NAME = "uni"
TARGET_SCENE_NAME = "street"
SOURCE_CAMERA_NAME = "paraview_camera_uni"
TARGET_CAMERA_NAME = "paraview_camera_street"

COLLECTION_RENAMES = {
    "Uni_Base": "Street_Base",
    "Uni_Base-cubes": "Street_Base-cubes",
    "Uni_Manager": "Street_Manager",
    "Uni_Cameras": "Street_Cameras",
    "Uni_Bioenvelopes-positive": "Street_Bioenvelopes-positive",
    "Uni_Bioenvelopes-trending": "Street_Bioenvelopes-trending",
    "uni_positive": "street_positive",
    "uni_priority": "street_priority",
    "uni_trending": "street_trending",
}


def remove_non_target_cameras(target_name: str) -> None:
    for obj in list(bpy.data.objects):
        if obj.type != "CAMERA":
            continue
        if obj.name == target_name:
            continue
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data and data.users == 0:
            bpy.data.cameras.remove(data)


def remove_all_texts() -> None:
    for text in list(bpy.data.texts):
        bpy.data.texts.remove(text)


def rename_top_level_collections(scene: bpy.types.Scene) -> None:
    for collection in scene.collection.children:
        new_name = COLLECTION_RENAMES.get(collection.name)
        if new_name:
            collection.name = new_name


def purge_orphans() -> None:
    for _ in range(3):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def main() -> None:
    bpy.ops.wm.open_mainfile(filepath=SOURCE_BLEND)

    scene = bpy.data.scenes.get(SOURCE_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SOURCE_SCENE_NAME}' not found in {SOURCE_BLEND}")

    for other_scene in list(bpy.data.scenes):
        if other_scene != scene:
            bpy.data.scenes.remove(other_scene)

    scene.name = TARGET_SCENE_NAME
    rename_top_level_collections(scene)

    camera = bpy.data.objects.get(SOURCE_CAMERA_NAME)
    if camera is None:
        raise ValueError(f"Camera '{SOURCE_CAMERA_NAME}' not found in {SOURCE_BLEND}")
    camera.name = TARGET_CAMERA_NAME
    if camera.data:
        camera.data.name = TARGET_CAMERA_NAME
    scene.camera = camera

    remove_non_target_cameras(TARGET_CAMERA_NAME)
    remove_all_texts()
    purge_orphans()

    bpy.ops.wm.save_as_mainfile(filepath=TARGET_BLEND, compress=True)
    print(f"SAVED {TARGET_BLEND}")


if __name__ == "__main__":
    main()
