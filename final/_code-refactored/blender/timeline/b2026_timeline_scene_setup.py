from __future__ import annotations

from pathlib import Path
import sys

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract


def iter_layer_collection_tree(layer_collection):
    yield layer_collection
    for child in layer_collection.children:
        yield from iter_layer_collection_tree(child)


def get_layer_collection_by_name(view_layer: bpy.types.ViewLayer, collection_name: str):
    for layer_collection in iter_layer_collection_tree(view_layer.layer_collection):
        if layer_collection.collection.name == collection_name:
            return layer_collection
    return None


def reset_view_layer_excludes(view_layer: bpy.types.ViewLayer) -> None:
    for layer_collection in iter_layer_collection_tree(view_layer.layer_collection):
        layer_collection.exclude = False
        if hasattr(layer_collection, "holdout"):
            layer_collection.holdout = False
        if hasattr(layer_collection, "indirect_only"):
            layer_collection.indirect_only = False


def set_excluded(view_layer: bpy.types.ViewLayer, collection_names, excluded: bool = True) -> None:
    for collection_name in collection_names:
        layer_collection = get_layer_collection_by_name(view_layer, collection_name)
        if layer_collection is not None:
            layer_collection.exclude = excluded


def ensure_view_layers(
    scene: bpy.types.Scene,
    layer_names: tuple[str, ...] | list[str] | None = None,
    *,
    remove_layers: tuple[str, ...] | list[str] = (),
) -> None:
    wanted = tuple(layer_names or scene_contract.STANDARD_VIEW_LAYERS)
    for layer_name in wanted:
        if scene.view_layers.get(layer_name) is None:
            scene.view_layers.new(name=layer_name)
    for layer_name in remove_layers:
        view_layer = scene.view_layers.get(layer_name)
        if view_layer is not None:
            scene.view_layers.remove(view_layer)


def ensure_collection(parent: bpy.types.Collection, name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if parent.children.get(collection.name) is None:
        parent.children.link(collection)
    return collection


def ensure_root_collection(scene: bpy.types.Scene, name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if scene.collection.children.get(collection.name) is None:
        scene.collection.children.link(collection)
    return collection


def rename_root_collection(scene: bpy.types.Scene, old_name: str, new_name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(old_name)
    if collection is None:
        collection = bpy.data.collections.get(new_name)
    if collection is None:
        collection = bpy.data.collections.new(new_name)
    collection.name = new_name
    if scene.collection.children.get(collection.name) is None:
        scene.collection.children.link(collection)
    return collection


def unlink_root_collection_if_present(scene: bpy.types.Scene, name: str) -> None:
    collection = bpy.data.collections.get(name)
    if collection is None:
        return
    if scene.collection.children.get(collection.name) is not None:
        scene.collection.children.unlink(collection)


def relink_object_to_collection(obj: bpy.types.Object, target_collection: bpy.types.Collection) -> None:
    if target_collection.objects.get(obj.name) is None:
        target_collection.objects.link(obj)
    for collection in list(obj.users_collection):
        if collection != target_collection:
            collection.objects.unlink(obj)


def ensure_single_state_shell(scene: bpy.types.Scene, site: str) -> dict[str, bpy.types.Collection]:
    collections = {
        role: ensure_root_collection(scene, scene_contract.get_single_state_top_level_name(site, role))
        for role in scene_contract.SINGLE_STATE_TOP_LEVEL_ROLES
    }
    return collections


def ensure_timeline_shell(scene: bpy.types.Scene, site: str) -> dict[str, bpy.types.Collection]:
    contract = scene_contract.SITE_CONTRACTS[site]
    top = contract["top_level"]
    legacy = contract["legacy"]

    # Reuse the single-state shell roots when present so camera/base objects stay linked.
    manager = rename_root_collection(
        scene,
        scene_contract.get_single_state_top_level_name(site, "manager"),
        top["manager"],
    )
    base = rename_root_collection(
        scene,
        scene_contract.get_single_state_top_level_name(site, "setup"),
        top["base"],
    )
    cameras = rename_root_collection(
        scene,
        scene_contract.get_single_state_top_level_name(site, "cameras"),
        top["cameras"],
    )

    collections = {
        "manager": manager,
        "base": base,
        "cameras": cameras,
        "base_cubes": ensure_root_collection(scene, top["base_cubes"]),
        "bio_positive": ensure_root_collection(scene, top["bio_positive"]),
        "bio_trending": ensure_root_collection(scene, top["bio_trending"]),
        "positive": ensure_root_collection(scene, top["positive"]),
        "priority": ensure_root_collection(scene, top["priority"]),
        "trending": ensure_root_collection(scene, top["trending"]),
    }

    ensure_collection(base, legacy["base"])
    timeline_base = ensure_collection(base, legacy["timeline_base"])
    ensure_collection(collections["base_cubes"], legacy["base_cubes"])
    ensure_collection(collections["base_cubes"], f"{legacy['base_cubes']}_Timeline")
    ensure_collection(cameras, legacy["cameras"])
    ensure_collection(collections["bio_positive"], legacy["bio_positive"])
    ensure_collection(collections["bio_positive"], f"Year_{site}_timeline_bioenvelope_positive")
    ensure_collection(collections["bio_trending"], legacy["bio_trending"])
    ensure_collection(collections["bio_trending"], f"Year_{site}_timeline_bioenvelope_trending")
    ensure_collection(collections["positive"], legacy["timeline_positive"])
    ensure_collection(collections["priority"], legacy["timeline_priority"])
    ensure_collection(collections["trending"], legacy["timeline_trending"])
    ensure_collection(timeline_base, scene_contract.get_timeline_world_collection_name(site, "positive"))
    ensure_collection(timeline_base, scene_contract.get_timeline_world_collection_name(site, "trending"))

    base_world = bpy.data.collections.get(legacy["base"])
    if base_world is not None:
        for object_name in contract["world_objects"].values():
            obj = bpy.data.objects.get(object_name)
            if obj is not None:
                relink_object_to_collection(obj, base_world)

    # The timeline shell is now the source of truth; unlink leftover single-state roots.
    for role in scene_contract.SINGLE_STATE_TOP_LEVEL_ROLES:
        single_state_name = scene_contract.get_single_state_top_level_name(site, role)
        if single_state_name in {top["manager"], top["base"], top["cameras"], top["positive"], top["priority"], top["trending"]}:
            continue
        unlink_root_collection_if_present(scene, single_state_name)

    return collections
