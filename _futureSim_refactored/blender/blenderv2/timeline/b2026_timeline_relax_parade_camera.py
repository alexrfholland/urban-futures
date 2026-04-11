from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Vector


SCENE_NAME = "parade"
CAMERA_NAME = "paraview_camera_parade"
RAW_PARAVIEW_POSITION = Vector((-710.5998866124906, 155.04839940955577, 780.0399301944501))
BACKOFF_FRACTION = 0.1


def main():
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {bpy.data.filepath}")
    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None:
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}")

    current = camera.location.copy()
    target = current.lerp(RAW_PARAVIEW_POSITION, BACKOFF_FRACTION)
    camera.location = target
    bpy.context.view_layer.update()
    bpy.ops.wm.save_mainfile()
    print(f"RELAXED_CAMERA from={tuple(round(v, 6) for v in current)} to={tuple(round(v, 6) for v in target)}")


if __name__ == "__main__":
    main()
