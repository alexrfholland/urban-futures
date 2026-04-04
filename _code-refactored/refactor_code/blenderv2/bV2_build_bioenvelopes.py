"""Build bV2 bioenvelopes for the active scene.

This builder follows the same scene-runtime pattern as the instancer stage:
- read site/mode/year from active scene metadata
- resolve local bundle inputs from the shared bV2 path config
- build one positive and one trending bioenvelope object
- for timeline mode, translate each source year into the visual strip layout
- preserve per-point provenance via `source-year`
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable

import bpy
import bmesh
import numpy as np

try:
    from .bV2_scene_contract import (
        GLOBAL_RULES,
        MATERIAL_NAMES,
        NODE_GROUP_NAMES,
        get_mode_year_token,
        make_bioenvelope_object_name,
    )
    from .bV2_paths import iter_blender_input_roots
except ImportError:
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from bV2_scene_contract import (  # type: ignore
        GLOBAL_RULES,
        MATERIAL_NAMES,
        NODE_GROUP_NAMES,
        get_mode_year_token,
        make_bioenvelope_object_name,
    )
    from bV2_paths import iter_blender_input_roots  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_DATA_ROOT = REPO_ROOT / "data" / "revised" / "final"

TIMELINE_YEARS = (0, 10, 30, 60, 180)
EMPTY_BASELINE_BIOENVELOPE_YEARS = {0}
BIOENVELOPE_PASS_INDEX = 5
SOURCE_YEAR_DEFAULT = int(GLOBAL_RULES["source_year_initial_value"])
TIMELINE_OFFSET_STEP = 5.0
STATE_TO_COLLECTION_ROLE = {
    "positive": "bioenvelope_positive",
    "trending": "bioenvelope_trending",
}
ASSET_SITE_ALIASES = {"street": "uni"}
VISUAL_STRIP_POSITION_OVERRIDES = {
    "city": {
        0: 180,
        10: 60,
        30: 30,
        60: 10,
        180: 0,
    },
}
LOG_PATH = os.environ.get("BV2_LOG_PATH", "").strip()
ENVELOPE_AOV_ATTRIBUTE_NAMES = (
    ("source-year", "source-year"),
    ("proposal-decay", "blender_proposal-decay"),
    ("proposal-release-control", "blender_proposal-release-control"),
    ("proposal-recruit", "blender_proposal-recruit"),
    ("proposal-colonise", "blender_proposal-colonise"),
    ("proposal-deploy-structure", "blender_proposal-deploy-structure"),
)


def log(*args) -> None:
    message = " ".join(str(arg) for arg in args)
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")


def get_bioenvelope_display_material_name() -> str:
    mode = (
        os.environ.get("BV2_BIOENVELOPE_DISPLAY_MODE", "").strip().lower()
        or os.environ.get("BV2_DISPLAY_MODE", "").strip().lower()
    )
    if mode in {"source_year", "source-year", "debug_source_years", "debug-source-years"}:
        return MATERIAL_NAMES["debug_source_years"]
    return MATERIAL_NAMES["bioenvelope"]


def cumulative_timeline_translate(axis: str, offset_index: int) -> tuple[float, float, float]:
    distance = float(offset_index) * TIMELINE_OFFSET_STEP
    if axis == "x":
        return (distance, 0.0, 0.0)
    if axis == "y":
        return (0.0, -distance, 0.0)
    raise ValueError(f"Unsupported timeline offset axis: {axis!r}")


TIMELINE_SITE_SPECS = {
    "trimmed-parade": {
        "box_length": (280.875, 112.0, 50.0),
        "strips": {
            0: {"label": "yr0", "box_position": (-89.26000000000931, 279.06, 42.0), "translate": cumulative_timeline_translate("y", 0)},
            10: {"label": "yr10", "box_position": (-89.26000000000931, 166.86, 42.0), "translate": cumulative_timeline_translate("y", 1)},
            30: {"label": "yr30", "box_position": (-89.26000000000931, 54.66, 42.0), "translate": cumulative_timeline_translate("y", 2)},
            60: {"label": "yr60", "box_position": (-89.26000000000931, -57.54, 42.0), "translate": cumulative_timeline_translate("y", 3)},
            180: {"label": "yr180", "box_position": (-89.26000000000931, -169.74, 42.0), "translate": cumulative_timeline_translate("y", 4)},
        },
    },
    "city": {
        "box_length": (281.0, 112.0, 209.0),
        "strips": {
            0: {"label": "yr0", "box_position": (-75.8041194739053, 97.4443366928, 23.5), "translate": cumulative_timeline_translate("y", 0)},
            10: {"label": "yr10", "box_position": (-75.8041194739053, -14.5556633072, 23.5), "translate": cumulative_timeline_translate("y", 1)},
            30: {"label": "yr30", "box_position": (-75.8041194739053, -126.5556633072, 23.5), "translate": cumulative_timeline_translate("y", 2)},
            60: {"label": "yr60", "box_position": (-75.8041194739053, -238.5556633072, 23.5), "translate": cumulative_timeline_translate("y", 3)},
            180: {"label": "yr180", "box_position": (-75.8041194739053, -350.5556633071974, 23.5), "translate": cumulative_timeline_translate("y", 4)},
        },
    },
    "uni": {
        "box_length": (112.2, 281.0, 77.0),
        "strips": {
            0: {"label": "yr0", "box_position": (-300.29, -88.6363, 32.5), "translate": cumulative_timeline_translate("x", 0)},
            10: {"label": "yr10", "box_position": (-188.09, -88.6363, 32.5), "translate": cumulative_timeline_translate("x", 1)},
            30: {"label": "yr30", "box_position": (-75.89, -88.6363, 32.5), "translate": cumulative_timeline_translate("x", 2)},
            60: {"label": "yr60", "box_position": (36.31, -88.6363, 32.5), "translate": cumulative_timeline_translate("x", 3)},
            180: {"label": "yr180", "box_position": (148.51, -88.6363, 32.5), "translate": cumulative_timeline_translate("x", 4)},
        },
    },
}


def canonicalize_asset_site(site: str) -> str:
    return ASSET_SITE_ALIASES.get(site, site)


def iter_existing_bundle_roots() -> Iterable[Path]:
    seen: set[Path] = set()
    for candidate in iter_blender_input_roots():
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def get_active_scene() -> bpy.types.Scene:
    scene = bpy.context.scene
    if scene is None:
        raise RuntimeError("No active Blender scene")
    return scene


def get_runtime_config(scene: bpy.types.Scene | None = None) -> dict[str, object]:
    scene = scene or get_active_scene()
    site = str(scene.get("bV2_site", "")).strip()
    mode = str(scene.get("bV2_mode", "")).strip()
    year_raw = str(scene.get("bV2_year", "")).strip()
    year = int(year_raw) if year_raw else None
    if not site or not mode:
        raise RuntimeError("Scene is missing bV2 runtime metadata (`bV2_site`, `bV2_mode`)")
    return {
        "scene": scene,
        "site": site,
        "mode": mode,
        "year": year,
        "year_token": get_mode_year_token(mode, year),
    }


def get_active_years(mode: str, year: int | None) -> tuple[int, ...]:
    if mode == "timeline":
        return TIMELINE_YEARS
    if year is None:
        raise ValueError("single_state bioenvelope build requires a year")
    return (int(year),)


def get_position_year(site: str, display_year: int) -> int:
    return VISUAL_STRIP_POSITION_OVERRIDES.get(site, {}).get(display_year, display_year)


def get_timeline_strip_spec(site: str, display_year: int) -> dict:
    site_spec = TIMELINE_SITE_SPECS.get(site)
    if site_spec is None:
        raise ValueError(f"No timeline strip spec configured for site '{site}'")
    position_year = get_position_year(site, display_year)
    strip_spec = site_spec["strips"].get(position_year)
    if strip_spec is None:
        raise ValueError(
            f"No timeline strip defined for site='{site}' year={display_year} "
            f"(position year {position_year})"
        )
    return strip_spec


def get_timeline_translate(site: str, display_year: int) -> tuple[float, float, float]:
    return tuple(get_timeline_strip_spec(site, display_year)["translate"])


def resolve_bioenvelope_ply_path(site: str, scenario: str, year: int) -> Path:
    asset_site = canonicalize_asset_site(site)
    bundle_name = f"{asset_site}_{scenario}_1_envelope_scenarioYR{year}.ply"
    for root in iter_existing_bundle_roots():
        for relative in (
            Path("bioenvelopes") / asset_site / bundle_name,
            Path("envelopes") / asset_site / bundle_name,
            Path(asset_site) / "bioenvelopes" / bundle_name,
            Path(asset_site) / "envelopes" / bundle_name,
        ):
            candidate = root / relative
            if candidate.exists():
                return candidate

    candidate = REPO_DATA_ROOT / asset_site / "ply" / bundle_name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve bioenvelope PLY for site={site}, scenario={scenario}, year={year}"
    )


def find_collection_by_role(scene: bpy.types.Scene, role: str) -> bpy.types.Collection:
    queue = list(scene.collection.children)
    while queue:
        collection = queue.pop(0)
        role_name = str(collection.get("bV2_role", collection.name.split("::")[-1]))
        if role_name == role:
            return collection
        queue.extend(collection.children)
    raise RuntimeError(f"Could not find collection for role {role!r} in scene {scene.name!r}")


def remove_collection_contents(collection: bpy.types.Collection) -> None:
    for child in list(collection.children):
        remove_collection_contents(child)
        collection.children.unlink(child)
        bpy.data.collections.remove(child)
    for obj in list(collection.objects):
        mesh = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def import_ply_object(filepath: Path) -> bpy.types.Object:
    before_names = set(bpy.data.objects.keys())
    windows = list(bpy.context.window_manager.windows) if bpy.context.window_manager is not None else []
    if windows:
        window = windows[0]
        screen = window.screen
        area = next((area for area in screen.areas if area.type == "VIEW_3D"), None)
        if area is None and screen.areas:
            area = screen.areas[0]
        region = None
        if area is not None:
            region = next((region for region in area.regions if region.type == "WINDOW"), None)
        override = {"window": window, "screen": screen}
        if area is not None:
            override["area"] = area
        if region is not None:
            override["region"] = region
        with bpy.context.temp_override(**override):
            bpy.ops.wm.ply_import(filepath=str(filepath))
    else:
        bpy.ops.wm.ply_import(filepath=str(filepath))
    new_names = [name for name in bpy.data.objects.keys() if name not in before_names]
    for name in new_names:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type == "MESH":
            return obj
    raise RuntimeError(f"No mesh object was imported from {filepath}")


def build_window_view3d_override(
    *,
    window: bpy.types.Window | None,
    scene: bpy.types.Scene | None,
    view_layer: bpy.types.ViewLayer | None,
) -> dict[str, object]:
    override: dict[str, object] = {}
    if window is None:
        return override
    screen = window.screen
    area = next((area for area in screen.areas if area.type == "VIEW_3D"), None)
    if area is None and screen.areas:
        area = screen.areas[0]
    region = None
    space = None
    if area is not None:
        region = next((region for region in area.regions if region.type == "WINDOW"), None)
        space = next((space for space in area.spaces if space.type == "VIEW_3D"), None)
    override["window"] = window
    override["screen"] = screen
    if scene is not None:
        override["scene"] = scene
    if view_layer is not None:
        override["view_layer"] = view_layer
    if area is not None:
        override["area"] = area
    if region is not None:
        override["region"] = region
    if space is not None:
        override["space_data"] = space
    return override


def relink_object_to_collection(obj: bpy.types.Object, target_collection: bpy.types.Collection) -> None:
    if target_collection.objects.get(obj.name) is None:
        target_collection.objects.link(obj)
    for collection in list(obj.users_collection):
        if collection != target_collection:
            collection.objects.unlink(obj)


def ensure_material_slot(obj: bpy.types.Object, material_name: str) -> None:
    material = bpy.data.materials.get(material_name)
    if material is None or obj.type != "MESH" or obj.data is None:
        return
    mesh = obj.data
    if len(mesh.materials) == 0:
        mesh.materials.append(material)
    else:
        for index in range(len(mesh.materials)):
            mesh.materials[index] = material


def ensure_envelope_material_aovs() -> None:
    material_name = MATERIAL_NAMES["bioenvelope"]
    material = bpy.data.materials.get(material_name)
    if material is None or not material.use_nodes or material.node_tree is None:
        raise RuntimeError(f"Could not find required node-based bioenvelope material {material_name!r}")

    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    existing_aovs = {
        node.aov_name: node
        for node in nodes
        if node.type == "OUTPUT_AOV" and getattr(node, "aov_name", "")
    }
    base_x = 260.0
    base_y = -260.0
    step_y = -160.0

    for index, (aov_name, attribute_name) in enumerate(ENVELOPE_AOV_ATTRIBUTE_NAMES):
        aov_node = existing_aovs.get(aov_name)
        if aov_node is None:
            aov_node = nodes.new("ShaderNodeOutputAOV")
            aov_node.aov_name = aov_name
            aov_node.name = f"AOV {aov_name}"

        attribute_node = next(
            (
                node
                for node in nodes
                if node.type == "ATTRIBUTE" and getattr(node, "attribute_name", "") == attribute_name
            ),
            None,
        )
        if attribute_node is None:
            attribute_node = nodes.new("ShaderNodeAttribute")
            attribute_node.attribute_name = attribute_name
            attribute_node.name = f"Attribute {attribute_name}"

        y = base_y + index * step_y
        attribute_node.location = (base_x - 220.0, y)
        aov_node.location = (base_x + 80.0, y)
        value_input = aov_node.inputs.get("Value")
        if value_input is None:
            continue
        for link in list(value_input.links):
            links.remove(link)
        links.new(attribute_node.outputs["Fac"], value_input)


def cleanup_modifier_node_group(modifier: bpy.types.Modifier | None) -> None:
    if modifier is None:
        return
    node_group = getattr(modifier, "node_group", None)
    if node_group is None:
        return
    if node_group.name == NODE_GROUP_NAMES["bioenvelope"]:
        return
    if node_group.users == 1:
        bpy.data.node_groups.remove(node_group)


def ensure_source_year_attribute(obj: bpy.types.Object, year_value: int) -> None:
    if obj.type != "MESH" or obj.data is None:
        return
    mesh = obj.data
    existing = mesh.attributes.get("source-year")
    if existing is not None and not getattr(existing, "is_required", False):
        mesh.attributes.remove(existing)
    attr = mesh.attributes.new(name="source-year", type="INT", domain="POINT")
    values = np.full(len(attr.data), int(year_value), dtype=np.int32)
    if len(values):
        attr.data.foreach_set("value", values)
    if hasattr(attr, "is_runtime_only"):
        attr.is_runtime_only = False
    mesh.update()


def trim_mesh_object_to_timeline_strip(obj: bpy.types.Object, site: str, display_year: int) -> None:
    if obj.type != "MESH" or obj.data is None:
        return
    site_spec = TIMELINE_SITE_SPECS.get(site)
    if site_spec is None:
        raise ValueError(f"No timeline site spec configured for site '{site}'")
    strip_spec = get_timeline_strip_spec(site, display_year)
    mins = np.asarray(strip_spec["box_position"], dtype=np.float64)
    lengths = np.asarray(site_spec["box_length"], dtype=np.float64)
    maxs = mins + lengths
    epsilon = 1e-6

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    if bm.faces:
        faces_to_delete = []
        for face in bm.faces:
            keep_face = True
            for vert in face.verts:
                co = vert.co
                if (
                    co.x < mins[0] - epsilon
                    or co.x > maxs[0] + epsilon
                    or co.y < mins[1] - epsilon
                    or co.y > maxs[1] + epsilon
                    or co.z < mins[2] - epsilon
                    or co.z > maxs[2] + epsilon
                ):
                    keep_face = False
                    break
            if not keep_face:
                faces_to_delete.append(face)
        if faces_to_delete:
            bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")

    verts_to_delete = [
        vert
        for vert in bm.verts
        if (
            vert.co.x < mins[0] - epsilon
            or vert.co.x > maxs[0] + epsilon
            or vert.co.y < mins[1] - epsilon
            or vert.co.y > maxs[1] + epsilon
            or vert.co.z < mins[2] - epsilon
            or vert.co.z > maxs[2] + epsilon
        )
    ]
    if verts_to_delete:
        bmesh.ops.delete(bm, geom=verts_to_delete, context="VERTS")

    loose_verts = [vert for vert in bm.verts if not vert.link_faces and not vert.link_edges]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def ensure_envelope_modifier(obj: bpy.types.Object) -> None:
    template_name = NODE_GROUP_NAMES["bioenvelope"]
    template_group = bpy.data.node_groups.get(template_name)
    if template_group is None:
        raise RuntimeError(f"Could not find required node group {template_name!r}")

    ensure_envelope_material_aovs()
    display_material = bpy.data.materials.get(get_bioenvelope_display_material_name())
    if display_material is None:
        raise RuntimeError(f"Could not find bioenvelope display material {get_bioenvelope_display_material_name()!r}")

    existing = obj.modifiers.get("bV2_bioenvelope")
    cleanup_modifier_node_group(existing)
    if existing is not None:
        obj.modifiers.remove(existing)

    node_group = template_group.copy()
    node_group.name = f"{obj.name}__bio_gn"
    for node in node_group.nodes:
        if node.type == "SET_MATERIAL" and "Material" in node.inputs:
            node.inputs["Material"].default_value = display_material

    modifier = obj.modifiers.new(name="bV2_bioenvelope", type="NODES")
    modifier.node_group = node_group


def create_empty_mesh_object(name: str, target_collection: bpy.types.Collection) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    target_collection.objects.link(obj)
    return obj


def join_mesh_objects(
    objects: list[bpy.types.Object],
    final_name: str,
    *,
    scene: bpy.types.Scene | None = None,
    window: bpy.types.Window | None = None,
    view_layer: bpy.types.ViewLayer | None = None,
) -> bpy.types.Object:
    if not objects:
        raise ValueError("join_mesh_objects requires at least one object")
    if len(objects) == 1:
        result = objects[0]
        result.name = final_name
        if result.data is not None:
            result.data.name = final_name
        return result

    scene = scene or bpy.context.scene
    active = objects[0]
    target_view_layer = view_layer or getattr(bpy.context, "view_layer", None)

    previous_scene = window.scene if window is not None else None
    previous_view_layer = window.view_layer if window is not None else None
    if window is not None and scene is not None:
        window.scene = scene
    if window is not None and target_view_layer is not None:
        window.view_layer = target_view_layer

    override = build_window_view3d_override(window=window, scene=scene, view_layer=target_view_layer)
    if scene is not None:
        override["scene"] = scene
    if target_view_layer is not None:
        override["view_layer"] = target_view_layer
        override["active_object"] = active
        override["selected_objects"] = objects
        override["selected_editable_objects"] = objects

    try:
        with bpy.context.temp_override(**override):
            active_context_object = target_view_layer.objects.active if target_view_layer is not None else None
            if scene is not None:
                for obj in scene.objects:
                    try:
                        obj.select_set(False)
                    except Exception:
                        continue
            for obj in objects:
                obj.select_set(True)
            if target_view_layer is not None:
                target_view_layer.objects.active = active
            if active_context_object is not None and getattr(active_context_object, "mode", "OBJECT") != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.join()
    finally:
        if window is not None and previous_scene is not None:
            window.scene = previous_scene
        if window is not None and previous_view_layer is not None:
            window.view_layer = previous_view_layer

    active.name = final_name
    if active.data is not None:
        active.data.name = final_name
    return active


def configure_bioenvelope_object(
    obj: bpy.types.Object,
    *,
    site: str,
    mode: str,
    state: str,
    year: int | None,
    source_years: tuple[int, ...],
) -> None:
    obj.pass_index = BIOENVELOPE_PASS_INDEX
    obj.hide_viewport = False
    obj.hide_render = False
    try:
        obj.hide_set(False)
    except Exception:
        pass
    obj["bV2_site"] = site
    obj["bV2_mode"] = mode
    obj["bV2_state"] = state
    obj["bV2_year_token"] = get_mode_year_token(mode, year)
    obj["bV2_source_years"] = ",".join(str(value) for value in source_years)
    ensure_envelope_modifier(obj)
    ensure_material_slot(obj, get_bioenvelope_display_material_name())


def build_state_bioenvelope(
    scene: bpy.types.Scene,
    *,
    site: str,
    mode: str,
    year: int | None,
    state: str,
) -> dict[str, object]:
    collection = find_collection_by_role(scene, STATE_TO_COLLECTION_ROLE[state])
    remove_collection_contents(collection)

    final_name = make_bioenvelope_object_name(site, mode, state, year)
    imported_objects: list[bpy.types.Object] = []
    imported_years: list[int] = []
    skipped_years: list[int] = []

    for display_year in get_active_years(mode, year):
        try:
            ply_path = resolve_bioenvelope_ply_path(site, state, display_year)
        except FileNotFoundError:
            skipped_years.append(display_year)
            if mode == "timeline" and display_year in EMPTY_BASELINE_BIOENVELOPE_YEARS:
                log("BIOENV_TIMELINE_EMPTY_BASELINE", "site=", site, "state=", state, "year=", display_year)
                continue
            log("BIOENV_MISSING_SOURCE", "site=", site, "state=", state, "year=", display_year)
            continue

        log("BIOENV_IMPORT_START", "site=", site, "state=", state, "year=", display_year, "path=", ply_path)
        obj = import_ply_object(ply_path)
        obj.name = f"{final_name}__yr{display_year}"
        if obj.data is not None:
            obj.data.name = f"{obj.name}_mesh"
        relink_object_to_collection(obj, collection)
        if mode == "timeline":
            trim_mesh_object_to_timeline_strip(obj, site, display_year)
        ensure_source_year_attribute(obj, display_year)
        if mode == "timeline":
            obj.location = get_timeline_translate(site, display_year)
        imported_objects.append(obj)
        imported_years.append(display_year)
        log("BIOENV_IMPORT_DONE", "site=", site, "state=", state, "year=", display_year, "object=", obj.name)

    if imported_objects:
        windows = list(bpy.context.window_manager.windows) if bpy.context.window_manager is not None else []
        window = windows[0] if windows else None
        previous_view_layer = window.view_layer if window is not None else None
        target_view_layer = scene.view_layers.get(STATE_TO_COLLECTION_ROLE[state])
        try:
            final_object = join_mesh_objects(
                imported_objects,
                final_name,
                scene=scene,
                window=window,
                view_layer=target_view_layer,
            )
        finally:
            if window is not None and previous_view_layer is not None:
                window.view_layer = previous_view_layer
    else:
        log("BIOENV_NO_SOURCE_GEOMETRY", "site=", site, "state=", state, "mode=", mode, "year=", year)
        final_object = create_empty_mesh_object(final_name, collection)
        ensure_source_year_attribute(final_object, SOURCE_YEAR_DEFAULT)

    configure_bioenvelope_object(
        final_object,
        site=site,
        mode=mode,
        state=state,
        year=year,
        source_years=tuple(imported_years),
    )

    result = {
        "state": state,
        "object": final_object.name,
        "vertex_count": int(len(final_object.data.vertices)) if final_object.type == "MESH" and final_object.data else 0,
        "source_years": tuple(imported_years),
        "missing_years": tuple(skipped_years),
    }
    log("BIOENV_STATE_DONE", result)
    return result


def build_bioenvelopes(
    scene: bpy.types.Scene | None = None,
    *,
    site: str | None = None,
    mode: str | None = None,
    year: int | None = None,
) -> dict[str, object]:
    scene = scene or get_active_scene()
    runtime = get_runtime_config(scene)
    site = site or str(runtime["site"])
    mode = mode or str(runtime["mode"])
    if year is None:
        year = runtime["year"]  # type: ignore[assignment]

    log("BIOENV_BUILD_START", "scene=", scene.name, "site=", site, "mode=", mode, "year=", year)
    results = [
        build_state_bioenvelope(scene, site=site, mode=mode, year=year, state="positive"),
        build_state_bioenvelope(scene, site=site, mode=mode, year=year, state="trending"),
    ]
    scene["bV2_bioenvelopes_built"] = True
    summary = {
        "scene": scene.name,
        "site": site,
        "mode": mode,
        "year": year,
        "results": results,
    }
    log("BIOENV_BUILD_DONE", summary)
    return summary


def main() -> None:
    build_bioenvelopes()


if __name__ == "__main__":
    main()
