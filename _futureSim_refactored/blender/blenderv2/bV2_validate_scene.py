"""Lightweight contract validation for bV2 scenes.

This validation is intentionally structural only. It does not render, does not
evaluate heavy geometry, and should stay fast enough to use in headless smoke
tests without hanging the pipeline.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import bpy

try:
    from .bV2_scene_contract import (
        MATERIAL_NAMES,
        get_aov_names,
        get_expected_camera_names,
        get_source_world_objects,
        get_view_layer_names,
        get_working_collection_tree,
        make_bioenvelope_object_name,
        make_world_object_name,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bV2_scene_contract import (  # type: ignore
        MATERIAL_NAMES,
        get_aov_names,
        get_expected_camera_names,
        get_source_world_objects,
        get_view_layer_names,
        get_working_collection_tree,
        make_bioenvelope_object_name,
        make_world_object_name,
    )


LOG_PATH = os.environ.get("BV2_LOG_PATH", "").strip()
SOURCE_YEAR_ATTRIBUTE = "source-year"
STATE_ROLES = ("positive_instances", "positive_priority_instances", "trending_instances")


def log(*args) -> None:
    message = " ".join(str(arg) for arg in args)
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")


def get_active_scene() -> bpy.types.Scene:
    scene = bpy.context.scene
    if scene is None:
        raise RuntimeError("No active Blender scene")
    return scene


def find_collection_by_role(scene: bpy.types.Scene, role: str) -> bpy.types.Collection | None:
    queue = [scene.collection]
    while queue:
        collection = queue.pop(0)
        role_name = str(collection.get("bV2_role", collection.name.split("::")[-1]))
        if role_name == role:
            return collection
        queue.extend(collection.children)
    return None


def mesh_has_attribute(obj: bpy.types.Object, attribute_name: str) -> bool:
    if obj.type != "MESH" or obj.data is None:
        return False
    mesh = obj.data
    attributes = getattr(mesh, "attributes", None)
    if attributes is None:
        return False
    return attributes.get(attribute_name) is not None


def object_uses_material(obj: bpy.types.Object, material_name: str) -> bool:
    if obj.type != "MESH" or obj.data is None:
        return False
    materials = getattr(obj.data, "materials", None)
    if materials is None:
        return False
    return any(material is not None and material.name == material_name for material in materials)


def validate_scene(scene: bpy.types.Scene | None = None, *, strict: bool = True) -> dict[str, object]:
    scene = scene or get_active_scene()
    errors: list[str] = []
    warnings: list[str] = []

    site = str(scene.get("bV2_site", "")).strip()
    mode = str(scene.get("bV2_mode", "")).strip()
    year_raw = str(scene.get("bV2_year", "")).strip()
    year = int(year_raw) if year_raw else None
    if not site:
        errors.append("Scene is missing bV2_site metadata")
    if not mode:
        errors.append("Scene is missing bV2_mode metadata")

    expected_camera_names = tuple(get_expected_camera_names(site, mode, year)) if site and mode else ()
    if scene.camera is None:
        errors.append("Scene has no active camera")
    elif expected_camera_names and scene.camera.name not in expected_camera_names:
        warnings.append(
            f"Scene camera is {scene.camera.name!r}; expected one of {expected_camera_names!r}"
        )

    is_baseline = mode == "baseline"

    expected_layers = list(get_view_layer_names(mode))
    actual_layers = [view_layer.name for view_layer in scene.view_layers]
    for layer_name in expected_layers:
        if layer_name not in actual_layers:
            errors.append(f"Missing view layer {layer_name!r}")
    extra_layers = [name for name in actual_layers if name not in expected_layers]
    if extra_layers:
        warnings.append(f"Extra view layers present: {', '.join(extra_layers)}")

    required_aovs = set(get_aov_names())
    for view_layer in scene.view_layers:
        actual_aovs = {aov.name for aov in view_layer.aovs}
        missing = sorted(required_aovs - actual_aovs)
        if missing:
            errors.append(f"View layer {view_layer.name!r} is missing AOVs: {', '.join(missing)}")

    expected_roles = set(get_working_collection_tree(mode).keys())
    for child_roles in get_working_collection_tree(mode).values():
        expected_roles.update(child_roles)
    for role in sorted(expected_roles):
        if find_collection_by_role(scene, role) is None:
            errors.append(f"Missing collection role {role!r}")

    if not bool(scene.get("bV2_instancers_built", False)):
        errors.append("Scene flag bV2_instancers_built is not set")
    if not is_baseline and not bool(scene.get("bV2_bioenvelopes_built", False)):
        errors.append("Scene flag bV2_bioenvelopes_built is not set")
    if not bool(scene.get("bV2_world_attributes_built", False)):
        errors.append("Scene flag bV2_world_attributes_built is not set")

    validate_states = ("positive",) if is_baseline else ("positive", "trending")

    if is_baseline:
        world_kinds = ("terrain",)
    else:
        world_kinds = tuple(get_source_world_objects(site).keys()) if site else ()

    for state in validate_states:
        for kind in world_kinds:
            object_name = make_world_object_name(kind, site, mode, state, year)
            obj = bpy.data.objects.get(object_name)
            if obj is None:
                errors.append(f"Missing world object {object_name!r}")
                continue
            if not mesh_has_attribute(obj, SOURCE_YEAR_ATTRIBUTE):
                errors.append(f"World object {object_name!r} is missing {SOURCE_YEAR_ATTRIBUTE!r}")
            if not object_uses_material(obj, MATERIAL_NAMES["world"]):
                warnings.append(f"World object {object_name!r} is not using {MATERIAL_NAMES['world']!r}")

    if not is_baseline:
        for state in validate_states:
            object_name = make_bioenvelope_object_name(site, mode, state, year) if site and mode else ""
            obj = bpy.data.objects.get(object_name) if object_name else None
            if obj is None:
                errors.append(f"Missing bioenvelope object {object_name!r}")
                continue
            if not mesh_has_attribute(obj, SOURCE_YEAR_ATTRIBUTE):
                errors.append(f"Bioenvelope {object_name!r} is missing {SOURCE_YEAR_ATTRIBUTE!r}")
            if obj.type == "MESH" and len(obj.data.vertices) == 0:
                warnings.append(f"Bioenvelope {object_name!r} is empty")
            if not object_uses_material(obj, MATERIAL_NAMES["bioenvelope"]):
                warnings.append(f"Bioenvelope {object_name!r} is not using {MATERIAL_NAMES['bioenvelope']!r}")

    for role in STATE_ROLES:
        collection = find_collection_by_role(scene, role)
        if collection is None:
            continue
        if not collection.objects:
            warnings.append(f"Instancer collection {role!r} is empty")
            continue
        for obj in collection.objects:
            if obj.type != "MESH":
                continue
            if "_positions_" in obj.name and not mesh_has_attribute(obj, SOURCE_YEAR_ATTRIBUTE):
                errors.append(f"Instancer point cloud {obj.name!r} is missing {SOURCE_YEAR_ATTRIBUTE!r}")

    summary = {
        "scene": scene.name,
        "site": site,
        "mode": mode,
        "year": year,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    log("VALIDATE_SCENE", summary)
    if strict and errors:
        raise RuntimeError("bV2 validation failed:\n- " + "\n- ".join(errors))
    return summary


def main() -> None:
    strict = os.environ.get("BV2_VALIDATE_STRICT", "1").strip().lower() not in {"0", "false", "no"}
    validate_scene(strict=strict)


if __name__ == "__main__":
    main()
