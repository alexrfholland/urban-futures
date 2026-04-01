from __future__ import annotations

import os
from typing import Iterable

import bpy


SCENE_NAME = os.environ["B2026_SCENE_NAME"]
CAMERA_NAME = os.environ["B2026_CAMERA_NAME"]
VIEW_LAYER_NAME = os.environ.get("B2026_VIEW_LAYER_NAME", "debug_camera_framing")
DEBUG_COLLECTION_NAME = os.environ.get("B2026_DEBUG_COLLECTION_NAME", f"{SCENE_NAME}_{VIEW_LAYER_NAME}")
POINT_OBJECT_NAMES = [
    name.strip()
    for name in os.environ["B2026_POINT_OBJECT_NAMES"].split(";")
    if name.strip()
]


def walk_layer_collections(layer_collection: bpy.types.LayerCollection) -> Iterable[bpy.types.LayerCollection]:
    yield layer_collection
    for child in layer_collection.children:
        yield from walk_layer_collections(child)


def find_layer_collections_with_object(
    layer_collection: bpy.types.LayerCollection,
    object_name: str,
    path: list[bpy.types.LayerCollection] | None = None,
) -> list[list[bpy.types.LayerCollection]]:
    if path is None:
        path = []
    current_path = path + [layer_collection]
    matches: list[list[bpy.types.LayerCollection]] = []
    if any(obj.name == object_name for obj in layer_collection.collection.objects):
        matches.append(current_path)
    for child in layer_collection.children:
        matches.extend(find_layer_collections_with_object(child, object_name, current_path))
    return matches


def find_layer_collections_by_collection_name(
    layer_collection: bpy.types.LayerCollection,
    collection_name: str,
    path: list[bpy.types.LayerCollection] | None = None,
) -> list[list[bpy.types.LayerCollection]]:
    if path is None:
        path = []
    current_path = path + [layer_collection]
    matches: list[list[bpy.types.LayerCollection]] = []
    if layer_collection.collection.name == collection_name:
        matches.append(current_path)
    for child in layer_collection.children:
        matches.extend(find_layer_collections_by_collection_name(child, collection_name, current_path))
    return matches


def ensure_view_layer(scene: bpy.types.Scene, view_layer_name: str) -> bpy.types.ViewLayer:
    view_layer = scene.view_layers.get(view_layer_name)
    if view_layer is None:
        view_layer = scene.view_layers.new(name=view_layer_name)
    return view_layer


def ensure_debug_collection(scene: bpy.types.Scene, collection_name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        collection = bpy.data.collections.new(collection_name)
    if collection.name not in scene.collection.children:
        scene.collection.children.link(collection)
    return collection


def main() -> None:
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {bpy.data.filepath}.")

    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}.")

    target_object_names = [*POINT_OBJECT_NAMES, CAMERA_NAME]
    missing = [name for name in target_object_names if bpy.data.objects.get(name) is None]
    if missing:
        raise ValueError(f"Missing required objects in {bpy.data.filepath}: {missing}")

    debug_collection = ensure_debug_collection(scene, DEBUG_COLLECTION_NAME)
    target_name_set = set(target_object_names)
    for obj in list(debug_collection.objects):
        if obj.name not in target_name_set:
            debug_collection.objects.unlink(obj)
    for object_name in target_object_names:
        obj = bpy.data.objects[object_name]
        if obj.name not in debug_collection.objects:
            debug_collection.objects.link(obj)

    view_layer = ensure_view_layer(scene, VIEW_LAYER_NAME)

    # Start with everything excluded from this debug layer.
    for layer_collection in walk_layer_collections(view_layer.layer_collection):
        if layer_collection is view_layer.layer_collection:
            layer_collection.exclude = False
            continue
        layer_collection.exclude = True
        layer_collection.holdout = False
        layer_collection.indirect_only = False

    included_paths = find_layer_collections_by_collection_name(view_layer.layer_collection, DEBUG_COLLECTION_NAME)
    if not included_paths:
        raise ValueError(
            f"Debug collection '{DEBUG_COLLECTION_NAME}' is not linked into the layer collection tree for scene '{scene.name}'."
        )

    for path in included_paths:
        for layer_collection in path:
            layer_collection.exclude = False
            layer_collection.holdout = False
            layer_collection.indirect_only = False

    scene.camera = camera

    renderable = []
    for layer_collection in walk_layer_collections(view_layer.layer_collection):
        if layer_collection.exclude:
            continue
        for obj in layer_collection.collection.objects:
            if obj.hide_render:
                continue
            if obj.name not in renderable:
                renderable.append(obj.name)

    print(f"DEBUG_VIEW_LAYER {VIEW_LAYER_NAME}")
    print(f"  scene={scene.name}")
    print(f"  camera={camera.name}")
    print(f"  debug_collection={debug_collection.name}")
    print(f"  expected={target_object_names}")
    print(f"  renderable={sorted(renderable)}")

    if bpy.data.is_saved:
        bpy.ops.wm.save_mainfile()


if __name__ == "__main__":
    main()
