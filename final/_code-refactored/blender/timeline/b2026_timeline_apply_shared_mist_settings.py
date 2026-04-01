from __future__ import annotations

from pathlib import Path

import bpy


BLEND_PATH = Path(
    r"D:\2026 Arboreal Futures\data\renders\timeslices\camera_tests\2026 futures timeslice debug camera framing v3.blend"
)

WORLD_NAME = "debug_timeslice_world"
MIST_START = 560.0
MIST_DEPTH = 320.0
MIST_FALLOFF = "QUADRATIC"


def ensure_world(name: str) -> bpy.types.World:
    world = bpy.data.worlds.get(name)
    if world is None:
        world = bpy.data.worlds.new(name)
    return world


def main() -> None:
    world = ensure_world(WORLD_NAME)
    mist = world.mist_settings
    mist.use_mist = True
    mist.start = MIST_START
    mist.depth = MIST_DEPTH
    mist.falloff = MIST_FALLOFF

    for scene in bpy.data.scenes:
        scene.world = world
        for view_layer in scene.view_layers:
            view_layer.use_pass_mist = True
        print(
            f"MIST_APPLIED scene={scene.name} world={world.name} "
            f"start={mist.start} depth={mist.depth} falloff={mist.falloff}"
        )

    if bpy.data.is_saved:
        bpy.ops.wm.save_mainfile()


if __name__ == "__main__":
    main()
