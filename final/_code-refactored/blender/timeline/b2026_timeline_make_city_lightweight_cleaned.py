from __future__ import annotations

import bpy


SOURCE_BLEND = r"D:\2026 Arboreal Futures\data\2026 futures city lightweight.blend"
TARGET_BLEND = r"D:\2026 Arboreal Futures\data\2026 futures city lightweight cleaned.blend"
SCENE_NAME = "city"
CAMERA_NAME = "paraview_camera_city"


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


def purge_orphans() -> None:
    for _ in range(3):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def main() -> None:
    bpy.ops.wm.open_mainfile(filepath=SOURCE_BLEND)

    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {SOURCE_BLEND}")

    for other_scene in list(bpy.data.scenes):
        if other_scene != scene:
            bpy.data.scenes.remove(other_scene)

    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None:
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {SOURCE_BLEND}")
    scene.camera = camera

    remove_non_target_cameras(CAMERA_NAME)
    remove_all_texts()
    purge_orphans()

    bpy.ops.wm.save_as_mainfile(filepath=TARGET_BLEND, compress=True)
    print(f"SAVED {TARGET_BLEND}")


if __name__ == "__main__":
    main()
