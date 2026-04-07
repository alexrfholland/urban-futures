from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract


SCENE_NAME = os.environ["B2026_SCENE_NAME"]
SITE_KEY = os.environ["B2026_SITE_KEY"]


def find_layer_collection(layer_collection: bpy.types.LayerCollection, target_name: str):
    if layer_collection.name == target_name or layer_collection.collection.name == target_name:
        return layer_collection
    for child in layer_collection.children:
        found = find_layer_collection(child, target_name)
        if found is not None:
            return found
    return None


def main() -> None:
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {bpy.data.filepath}")

    contract = scene_contract.SITE_CONTRACTS[SITE_KEY]
    top = contract["top_level"]
    legacy = contract["legacy"]

    cube_collection_names = {
        top["base_cubes"],
        legacy["base_cubes"],
        f"{legacy['base_cubes']}_Timeline",
    }

    for collection_name in cube_collection_names:
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            collection.hide_render = True
            collection.hide_viewport = True

    for view_layer in scene.view_layers:
        for collection_name in cube_collection_names:
            layer_collection = find_layer_collection(view_layer.layer_collection, collection_name)
            if layer_collection is not None:
                layer_collection.exclude = True
                layer_collection.hide_viewport = True

    for obj in bpy.data.objects:
        if "_cubes" in obj.name or obj.name.endswith("Cubes"):
            obj.hide_render = True
            obj.hide_viewport = True

    bpy.ops.wm.save_mainfile()
    print(f"Disabled cubes for scene={SCENE_NAME} site={SITE_KEY}")


if __name__ == "__main__":
    main()
