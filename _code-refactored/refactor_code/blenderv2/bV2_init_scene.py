"""Initialize a clean working scene from the bV2 template.

This module handles scene structure only:
- create a fresh working scene using template scene settings
- create canonical collection shell
- clone site source world objects and cameras from the template
- create canonical view layers
- register canonical AOVs
- clear compositor/output-path state

It intentionally does not build instancers, bioenvelopes, or world outputs.
"""

from __future__ import annotations

import os
import time
from typing import Iterable

import bpy

try:
    from .bV2_scene_contract import (
        TEMPLATE_SCENE_NAME,
        TEMPLATE_ROOT_COLLECTIONS,
        TOP_LEVEL_COLLECTIONS,
        get_aov_names,
        get_alternate_camera_names,
        get_source_world_objects,
        get_timeline_camera_name,
        get_view_layer_names,
        get_view_layer_semantics,
        get_working_collection_tree,
        make_scene_name,
    )
except ImportError:
    # Allow direct Blender execution via `blender -P path\\to\\bV2_init_scene.py`.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bV2_scene_contract import (  # type: ignore
        TEMPLATE_SCENE_NAME,
        TEMPLATE_ROOT_COLLECTIONS,
        TOP_LEVEL_COLLECTIONS,
        get_aov_names,
        get_alternate_camera_names,
        get_source_world_objects,
        get_timeline_camera_name,
        get_view_layer_names,
        get_view_layer_semantics,
        get_working_collection_tree,
        make_scene_name,
    )


LOG_PATH = os.environ.get("BV2_LOG_PATH", "").strip()
ENABLE_UI_WORKSPACE_CLEANUP = os.environ.get("BV2_ENABLE_UI_WORKSPACE_CLEANUP", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def log(*args) -> None:
    message = " ".join(str(arg) for arg in args)
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")


def use_template_scene(target_scene_name: str) -> bpy.types.Scene:
    """Use the template scene in place as the working scene.

    The build opens the template blend and saves a new output blend, so it is
    safe and simpler to mutate the template scene in memory instead of trying
    to keep a second scene alive through the save.
    """

    template_scene = bpy.data.scenes.get(TEMPLATE_SCENE_NAME)
    if template_scene is None:
        raise RuntimeError(f"Template scene {TEMPLATE_SCENE_NAME!r} was not found in the open blend")

    template_scene.name = target_scene_name
    return template_scene


def ensure_top_level_collections(scene: bpy.types.Scene, names: Iterable[str]) -> dict[str, bpy.types.Collection]:
    """Ensure canonical top-level collections exist and are linked to the scene root."""

    root = scene.collection
    resolved: dict[str, bpy.types.Collection] = {}
    for name in names:
        collection = bpy.data.collections.new(f"{scene.name}::{name}")
        collection["bV2_role"] = name
        root.children.link(collection)
        resolved[name] = collection
    return resolved


def ensure_second_level_collections(
    top_level: dict[str, bpy.types.Collection],
) -> dict[str, bpy.types.Collection]:
    """Create the canonical second-level collection shell."""

    resolved: dict[str, bpy.types.Collection] = {}
    for parent_name, child_names in get_working_collection_tree().items():
        parent = top_level[parent_name]
        for child_name in child_names:
            collection = bpy.data.collections.new(f"{parent.name}::{child_name}")
            collection["bV2_role"] = child_name
            parent.children.link(collection)
            resolved[child_name] = collection
    return resolved


def clone_object_to_collection(
    source_name: str,
    target_collection: bpy.types.Collection,
    copied_name: str | None = None,
) -> bpy.types.Object:
    """Clone an object and its data into a target collection."""

    source = bpy.data.objects.get(source_name)
    if source is None:
        raise RuntimeError(f"Required template object {source_name!r} was not found")

    cloned = source.copy()
    if source.data is not None:
        cloned.data = source.data.copy()
    target_name = copied_name or source_name
    cloned["bV2_target_name"] = target_name
    cloned.name = target_name
    if cloned.data is not None:
        cloned.data.name = target_name
    target_collection.objects.link(cloned)
    return cloned


def remove_collection_tree(collection: bpy.types.Collection) -> None:
    """Recursively remove a collection tree and its contained objects."""

    for child in list(collection.children):
        remove_collection_tree(child)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def populate_site_sources(
    site: str,
    mode: str,
    second_level: dict[str, bpy.types.Collection],
    cameras_collection: bpy.types.Collection,
    camera_name: str | None = None,
) -> tuple[dict[str, bpy.types.Object], dict[str, bpy.types.Object]]:
    """Clone required site source assets into the working scene shell."""

    cloned_world: dict[str, bpy.types.Object] = {}
    cloned_cameras: dict[str, bpy.types.Object] = {}

    for role, object_name in get_source_world_objects(site).items():
        cloned_world[role] = clone_object_to_collection(
            source_name=object_name,
            target_collection=second_level["world_sources"],
        )

    timeline_camera_name = get_timeline_camera_name(site)
    requested_camera_name = camera_name or timeline_camera_name
    keep_camera_names = {requested_camera_name}
    if mode == "single_state":
        keep_camera_names.add(timeline_camera_name)
    for alternate_name in get_alternate_camera_names(site):
        if camera_name == alternate_name:
            keep_camera_names.add(alternate_name)

    for keep_name in sorted(keep_camera_names):
        cloned_cameras[keep_name] = clone_object_to_collection(
            source_name=keep_name,
            target_collection=cameras_collection,
        )

    return cloned_world, cloned_cameras


def ensure_view_layers(scene: bpy.types.Scene) -> None:
    """Reset the scene to the canonical ordered view-layer set."""

    desired = list(get_view_layer_names())
    while len(scene.view_layers) > 1:
        scene.view_layers.remove(scene.view_layers[-1])
    if not scene.view_layers:
        scene.view_layers.new(name=desired[0])
    scene.view_layers[0].name = desired[0]
    existing_names = {scene.view_layers[0].name}
    for name in desired[1:]:
        if name not in existing_names:
            scene.view_layers.new(name=name)
            existing_names.add(name)


def reset_layer_aovs(view_layer: bpy.types.ViewLayer) -> None:
    """Clear and repopulate a view layer's AOV registry using the canonical order."""

    while view_layer.aovs:
        view_layer.aovs.remove(view_layer.aovs[0])
    for aov_name in get_aov_names():
        aov = view_layer.aovs.add()
        aov.name = aov_name
        aov.type = "VALUE"


def apply_neutral_scene_state(scene: bpy.types.Scene) -> None:
    """Clear inherited render/compositor state from the template copy."""

    scene.render.filepath = ""
    scene.camera = None
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    scene.use_nodes = False
    if scene.node_tree is not None:
        for node in list(scene.node_tree.nodes):
            scene.node_tree.nodes.remove(node)


def prune_workspaces(keep_name: str = "Layout") -> None:
    """Delete non-essential workspaces so saved files reopen in scene view."""

    window = bpy.context.window
    if window is None:
        return

    keep_workspace = bpy.data.workspaces.get(keep_name)
    if keep_workspace is None:
        keep_workspace = bpy.data.workspaces[0] if bpy.data.workspaces else None
    if keep_workspace is None:
        return

    while True:
        removable = [workspace for workspace in bpy.data.workspaces if workspace.name != keep_workspace.name]
        if not removable:
            break
        window.workspace = removable[0]
        bpy.ops.workspace.delete()

    window.workspace = keep_workspace


def configure_viewport_camera_defaults(scene: bpy.types.Scene) -> None:
    """Configure saved UI data so the file opens in camera view in 3D space."""

    if scene.camera is None:
        return

    for workspace in bpy.data.workspaces:
        for screen in workspace.screens:
            for area in screen.areas:
                for space in area.spaces:
                    if space.type != "VIEW_3D":
                        continue
                    space.region_3d.view_perspective = "CAMERA"
                    if hasattr(space, "use_local_camera"):
                        space.use_local_camera = False
                    if hasattr(space, "camera"):
                        space.camera = scene.camera
                    if hasattr(space, "shading"):
                        space.shading.type = "MATERIAL"


def remove_template_asset_collections(scene: bpy.types.Scene) -> None:
    """Remove the template-only asset collections from the working scene/file."""

    for collection_name in TEMPLATE_ROOT_COLLECTIONS:
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            continue
        if collection in scene.collection.children[:]:
            scene.collection.children.unlink(collection)
        remove_collection_tree(collection)


def normalize_cloned_asset_names(objects: Iterable[bpy.types.Object]) -> None:
    """Rename cloned objects/data back to their canonical target names."""

    for obj in objects:
        target_name = obj.get("bV2_target_name")
        if not target_name:
            continue
        obj.name = target_name
        if obj.data is not None:
            obj.data.name = target_name


def normalize_object_visibility(objects: Iterable[bpy.types.Object]) -> None:
    """Reset object-level viewport/render visibility to a known baseline."""

    for obj in objects:
        obj.hide_viewport = False
        obj.hide_render = False
        try:
            obj.hide_set(False)
        except Exception:
            pass


def _walk_layer_collections(layer_collection: bpy.types.LayerCollection) -> Iterable[bpy.types.LayerCollection]:
    yield layer_collection
    for child in layer_collection.children:
        yield from _walk_layer_collections(child)


def apply_view_layer_semantics(scene: bpy.types.Scene) -> None:
    """Apply the semantic view-layer contract to Blender layer collections.

    The scene shell uses second-level collection roles that match the semantic
    contract tags directly, so this function can wire excludes deterministically.
    """

    always_excluded = {"world_sources", "helpers"}
    role_like_collections = {
        "world_positive_attributes",
        "world_trending_attributes",
        "positive_instances",
        "positive_priority_instances",
        "trending_instances",
        "bioenvelope_positive",
        "bioenvelope_trending",
    }

    for view_layer in scene.view_layers:
        allowed_roles = set(get_view_layer_semantics(view_layer.name))
        for layer_collection in _walk_layer_collections(view_layer.layer_collection):
            role_name = str(layer_collection.collection.get("bV2_role", layer_collection.name.split("::")[-1]))
            layer_collection.hide_viewport = False
            if role_name in always_excluded:
                layer_collection.exclude = True
            elif role_name in role_like_collections:
                layer_collection.exclude = role_name not in allowed_roles
            else:
                layer_collection.exclude = False


def write_scene_metadata(
    scene: bpy.types.Scene,
    site: str,
    mode: str,
    year: int | str | None = None,
) -> None:
    """Store basic build metadata on the scene for downstream scripts."""

    scene["bV2_site"] = site
    scene["bV2_mode"] = mode
    scene["bV2_year"] = "" if year is None else str(year)


def init_scene(
    site: str,
    mode: str,
    year: int | str | None = None,
    camera_name: str | None = None,
) -> bpy.types.Scene:
    """Build and return a clean working scene for the target site and mode."""

    scene_name = make_scene_name(site=site, mode=mode, year=year)
    log("INIT_SCENE", "scene_name=", scene_name, "site=", site, "mode=", mode, "year=", year, "camera=", camera_name)

    log("INIT_USE_TEMPLATE_SCENE_START", TEMPLATE_SCENE_NAME, "->", scene_name)
    scene = use_template_scene(scene_name)
    log("INIT_USE_TEMPLATE_SCENE_DONE", "active_scene=", scene.name)

    log("INIT_TOP_LEVEL_COLLECTIONS_START", ",".join(TOP_LEVEL_COLLECTIONS))
    top_level = ensure_top_level_collections(scene, TOP_LEVEL_COLLECTIONS)
    log("INIT_TOP_LEVEL_COLLECTIONS_DONE", "count=", len(top_level), "names=", ",".join(sorted(top_level.keys())))

    log("INIT_SECOND_LEVEL_COLLECTIONS_START")
    second_level = ensure_second_level_collections(top_level)
    log(
        "INIT_SECOND_LEVEL_COLLECTIONS_DONE",
        "count=",
        len(second_level),
        "names=",
        ",".join(sorted(second_level.keys())),
    )

    log("INIT_POPULATE_SITE_SOURCES_START", "site=", site, "mode=", mode)
    cloned_world, cloned_cameras = populate_site_sources(
        site=site,
        mode=mode,
        second_level=second_level,
        cameras_collection=top_level["cameras"],
        camera_name=camera_name,
    )
    log(
        "INIT_POPULATE_SITE_SOURCES_DONE",
        "world=",
        ",".join(sorted(obj.name for obj in cloned_world.values())),
        "cameras=",
        ",".join(sorted(obj.name for obj in cloned_cameras.values())),
    )

    log("INIT_VIEW_LAYERS_START")
    ensure_view_layers(scene)
    log("INIT_VIEW_LAYERS_DONE", "names=", ",".join(view_layer.name for view_layer in scene.view_layers))

    log("INIT_AOV_RESET_START", "layer_count=", len(scene.view_layers), "aov_count=", len(tuple(get_aov_names())))
    for view_layer in scene.view_layers:
        log("INIT_AOV_RESET_LAYER_START", view_layer.name)
        reset_layer_aovs(view_layer)
        log("INIT_AOV_RESET_LAYER_DONE", view_layer.name, "count=", len(view_layer.aovs))

    log("INIT_NEUTRAL_SCENE_STATE_START")
    apply_neutral_scene_state(scene)
    log("INIT_NEUTRAL_SCENE_STATE_DONE")

    log("INIT_REMOVE_TEMPLATE_COLLECTIONS_START", ",".join(TEMPLATE_ROOT_COLLECTIONS))
    remove_template_asset_collections(scene)
    log("INIT_REMOVE_TEMPLATE_COLLECTIONS_DONE")

    log("INIT_NORMALIZE_NAMES_START")
    normalize_cloned_asset_names([*cloned_world.values()])
    normalize_cloned_asset_names([*cloned_cameras.values()])
    log("INIT_NORMALIZE_NAMES_DONE")

    log("INIT_NORMALIZE_VISIBILITY_START")
    normalize_object_visibility([*cloned_world.values(), *cloned_cameras.values()])
    log("INIT_NORMALIZE_VISIBILITY_DONE")

    log("INIT_WRITE_METADATA_START")
    write_scene_metadata(scene, site=site, mode=mode, year=year)
    log("INIT_WRITE_METADATA_DONE")

    active_camera_name = camera_name or get_timeline_camera_name(site)
    log("INIT_ASSIGN_CAMERA_START", active_camera_name)
    scene.camera = cloned_cameras.get(active_camera_name)
    log("INIT_ASSIGN_CAMERA_DONE", "scene_camera=", scene.camera.name if scene.camera else "None")

    if ENABLE_UI_WORKSPACE_CLEANUP:
        log("INIT_PRUNE_WORKSPACES_START")
        prune_workspaces("Layout")
        log("INIT_PRUNE_WORKSPACES_DONE", "workspaces=", ",".join(workspace.name for workspace in bpy.data.workspaces))

        log("INIT_CONFIGURE_VIEWPORT_CAMERA_START")
        configure_viewport_camera_defaults(scene)
        log("INIT_CONFIGURE_VIEWPORT_CAMERA_DONE")
    else:
        log("INIT_UI_WORKSPACE_STEPS_SKIPPED")

    log("INIT_APPLY_VIEW_LAYER_SEMANTICS_START")
    apply_view_layer_semantics(scene)
    log("INIT_APPLY_VIEW_LAYER_SEMANTICS_DONE")
    log("INIT_SCENE_DONE", scene.name)
    return scene
