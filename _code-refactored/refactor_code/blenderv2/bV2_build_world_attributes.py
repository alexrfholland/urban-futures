"""Build bV2 world attribute objects for the active scene.

This stage:
- duplicates the per-site source roads/buildings objects
- transfers world point attributes from the final assessed VTKs
- keeps the canonical shared world material / GN path
- preserves source-year provenance on all renderable world geometry
- builds one positive and one trending world result
- in timeline mode, trims each year to the strip region before combining
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable

import bpy
import numpy as np
from mathutils import Matrix


REPO_ROOT = Path(__file__).resolve().parents[3]


def inject_repo_venv_site_packages() -> None:
    candidates = [
        REPO_ROOT / ".venv" / "Lib" / "site-packages",
    ]
    candidates.extend((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))
    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
except ImportError:
    inject_repo_venv_site_packages()
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore


try:
    from .bV2_scene_contract import (
        GLOBAL_RULES,
        MATERIAL_NAMES,
        NODE_GROUP_NAMES,
        get_mode_year_token,
        get_source_world_objects,
        make_world_object_name,
    )
    from .bV2_paths import iter_blender_input_roots
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bV2_scene_contract import (  # type: ignore
        GLOBAL_RULES,
        MATERIAL_NAMES,
        NODE_GROUP_NAMES,
        get_mode_year_token,
        get_source_world_objects,
        make_world_object_name,
    )
    from bV2_paths import iter_blender_input_roots  # type: ignore


LOG_PATH = os.environ.get("BV2_LOG_PATH", "").strip()
SAVE_MAINFILE = os.environ.get("BV2_SAVE_MAINFILE", "1").strip() != "0"
TIMELINE_YEARS = (0, 10, 30, 60, 180)
TIMELINE_OFFSET_STEP = 5.0
SOURCE_YEAR_DEFAULT = int(GLOBAL_RULES["source_year_initial_value"])
ENABLE_LOCAL_NEIGHBOR_FALLBACK = True
LOCAL_NEIGHBOR_OFFSETS = np.array(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=np.int64,
)
CHUNK_SIZE = 1_000_000
DATA_BUNDLE_ROOT_ENV_NAMES = (
    "BV2_DATA_BUNDLE_ROOTS",
    "BV2_DATA_BUNDLE_ROOT",
    "B2026_DATA_BUNDLE_ROOTS",
    "B2026_DATA_BUNDLE_ROOT",
)

ATTR_TURNS = "sim_Turns"
ATTR_NODES = "sim_Nodes"
ATTR_BIO = "scenario_bioEnvelope"
ATTR_BIO_SIMPLE = "scenario_bioEnvelopeSimple"
ATTR_MATCHED = "sim_Matched"
ATTR_SOURCE_YEAR = "source-year"
PROPOSAL_ATTRIBUTE_NAMES = (
    "blender_proposal-decay",
    "blender_proposal-release-control",
    "blender_proposal-recruit",
    "blender_proposal-colonise",
    "blender_proposal-deploy-structure",
)
TRANSFERRED_ATTRIBUTE_NAMES = (
    ATTR_TURNS,
    ATTR_NODES,
    ATTR_BIO,
    ATTR_BIO_SIMPLE,
    ATTR_MATCHED,
    ATTR_SOURCE_YEAR,
    *PROPOSAL_ATTRIBUTE_NAMES,
)
REPLACED_WORLD_ATTRIBUTE_NAMES = {
    *TRANSFERRED_ATTRIBUTE_NAMES,
}
STATE_TO_COLLECTION_ROLE = {
    "positive": "world_positive_attributes",
    "trending": "world_trending_attributes",
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
BIO_ENVELOPE_MAP = {
    "none": 0,
    "exoskeleton": 1,
    "brownRoof": 2,
    "otherGround": 3,
    "node-rewilded": 4,
    "rewilded": 4,
    "footprint-depaved": 5,
    "livingFacade": 6,
    "greenRoof": 7,
}
BIO_ENVELOPE_SIMPLE_MAP = {
    "none": 0,
    "brownRoof": 2,
    "livingFacade": 3,
    "greenRoof": 4,
}


def log(*args) -> None:
    message = " ".join(str(arg) for arg in args)
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")


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
    for env_name in DATA_BUNDLE_ROOT_ENV_NAMES:
        raw_value = os.environ.get(env_name, "").strip()
        if not raw_value:
            continue
        for raw_path in raw_value.split(os.pathsep):
            candidate = Path(raw_path.strip())
            if not raw_path.strip() or candidate in seen:
                continue
            if candidate.exists():
                seen.add(candidate)
                yield candidate
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
        raise ValueError("single_state world build requires a year")
    return (int(year),)


def get_timeline_strip_spec(site: str, display_year: int) -> dict:
    site_spec = TIMELINE_SITE_SPECS.get(site)
    if site_spec is None:
        raise ValueError(f"No timeline strip spec configured for site '{site}'")
    position_year = VISUAL_STRIP_POSITION_OVERRIDES.get(site, {}).get(display_year, display_year)
    strip_spec = site_spec["strips"].get(position_year)
    if strip_spec is None:
        raise ValueError(
            f"No timeline strip defined for site='{site}' year={display_year} "
            f"(position year {position_year})"
        )
    return strip_spec


def get_timeline_translate(site: str, display_year: int) -> np.ndarray:
    return np.asarray(get_timeline_strip_spec(site, display_year)["translate"], dtype=np.float64)


def clip_mask_for_points(points: np.ndarray, strip_spec: dict, site_spec: dict) -> np.ndarray:
    mins = np.asarray(strip_spec["box_position"], dtype=np.float64)
    lengths = np.asarray(site_spec["box_length"], dtype=np.float64)
    maxs = mins + lengths
    epsilon = 1e-6
    return np.all((points >= mins - epsilon) & (points <= maxs + epsilon), axis=1)


def resolve_vtk_path(site: str, scenario: str, year: int) -> Path:
    asset_site = canonicalize_asset_site(site)
    bundle_name = f"{asset_site}_{scenario}_1_yr{year}_state_with_indicators.vtk"
    for root in iter_existing_bundle_roots():
        candidate = root / "vtks" / asset_site / bundle_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve VTK for site={site}, scenario={scenario}, year={year}"
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


def mesh_vertex_world_positions(obj: bpy.types.Object) -> np.ndarray:
    vertex_count = len(obj.data.vertices)
    if vertex_count == 0:
        return np.empty((0, 3), dtype=np.float64)

    coords = np.empty(vertex_count * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", coords)
    coords = coords.reshape((-1, 3)).astype(np.float64, copy=False)

    matrix_world = np.asarray(obj.matrix_world, dtype=np.float64)
    rotation_scale = matrix_world[:3, :3]
    translation = matrix_world[:3, 3]
    return (coords @ rotation_scale.T) + translation


def extract_point_attribute_payloads(mesh: bpy.types.Mesh) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    point_count = len(mesh.vertices)
    for attribute in mesh.attributes:
        if attribute.domain != "POINT":
            continue
        if getattr(attribute, "is_required", False):
            continue
        if attribute.name == "position" or attribute.name.startswith("."):
            continue
        if attribute.name in REPLACED_WORLD_ATTRIBUTE_NAMES:
            continue

        if attribute.data_type in {"FLOAT", "INT", "BOOLEAN"}:
            field_name = "value"
            dtype = np.int32 if attribute.data_type in {"INT", "BOOLEAN"} else np.float32
            values = np.empty(point_count, dtype=dtype)
        elif attribute.data_type == "FLOAT_VECTOR":
            field_name = "vector"
            values = np.empty(point_count * 3, dtype=np.float32).reshape((-1, 3))
        elif attribute.data_type in {"FLOAT_COLOR", "BYTE_COLOR"}:
            field_name = "color"
            values = np.empty(point_count * 4, dtype=np.float32).reshape((-1, 4))
        else:
            continue

        flat_target = values.reshape(-1) if getattr(values, "ndim", 1) > 1 else values
        attribute.data.foreach_get(field_name, flat_target)
        payloads[attribute.name] = {
            "data_type": attribute.data_type,
            "field_name": field_name,
            "values": values,
        }
    return payloads


def apply_point_attribute_payloads(mesh: bpy.types.Mesh, payloads: dict[str, dict]) -> None:
    for name, payload in payloads.items():
        existing = mesh.attributes.get(name)
        if existing is not None and not getattr(existing, "is_required", False):
            mesh.attributes.remove(existing)
        attribute = mesh.attributes.new(name=name, type=payload["data_type"], domain="POINT")
        values = payload["values"]
        flat_values = values.reshape(-1) if getattr(values, "ndim", 1) > 1 else values
        if payload["data_type"] in {"INT", "BOOLEAN"}:
            typed_values = np.ascontiguousarray(flat_values, dtype=np.int32)
        else:
            typed_values = np.ascontiguousarray(flat_values, dtype=np.float32)
        attribute.data.foreach_set(payload["field_name"], typed_values)
        if hasattr(attribute, "is_runtime_only"):
            attribute.is_runtime_only = False


def payload_from_int_values(values: np.ndarray) -> dict[str, object]:
    return {
        "data_type": "INT",
        "field_name": "value",
        "values": np.ascontiguousarray(values, dtype=np.int32),
    }


def build_mesh_from_points(mesh_name: str, points: np.ndarray, payloads: dict[str, dict]) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(mesh_name)
    if len(points):
        mesh.vertices.add(len(points))
        mesh.vertices.foreach_set("co", np.ascontiguousarray(points.reshape(-1), dtype=np.float32))
    apply_point_attribute_payloads(mesh, payloads)
    mesh.update()
    return mesh


def copy_material_slots(source_obj: bpy.types.Object, target_mesh: bpy.types.Mesh) -> None:
    if getattr(source_obj, "data", None) is None:
        return
    for material in source_obj.data.materials:
        target_mesh.materials.append(material)


def ensure_world_material_slot(obj: bpy.types.Object) -> None:
    material = bpy.data.materials.get(MATERIAL_NAMES["world"])
    if material is None or obj.type != "MESH" or obj.data is None:
        return
    mesh = obj.data
    if len(mesh.materials) == 0:
        mesh.materials.append(material)
    else:
        for index in range(len(mesh.materials)):
            mesh.materials[index] = material


def cleanup_modifier_node_group(modifier: bpy.types.Modifier | None) -> None:
    if modifier is None:
        return
    node_group = getattr(modifier, "node_group", None)
    if node_group is None:
        return
    if node_group.name == NODE_GROUP_NAMES["world"]:
        return
    if node_group.users == 1:
        bpy.data.node_groups.remove(node_group)


def ensure_world_modifier(obj: bpy.types.Object) -> None:
    template_name = NODE_GROUP_NAMES["world"]
    template_group = bpy.data.node_groups.get(template_name)
    if template_group is None:
        raise RuntimeError(f"Could not find required node group {template_name!r}")

    material = bpy.data.materials.get(MATERIAL_NAMES["world"])
    if material is None:
        raise RuntimeError(f"Could not find required world material {MATERIAL_NAMES['world']!r}")

    existing = obj.modifiers.get("bV2_world")
    if existing is None:
        existing = next((modifier for modifier in obj.modifiers if modifier.type == "NODES"), None)
    cleanup_modifier_node_group(existing)
    if existing is not None:
        obj.modifiers.remove(existing)

    node_group = template_group.copy()
    node_group.name = f"{obj.name}__world_gn"
    for node in node_group.nodes:
        if node.type == "SET_MATERIAL" and "Material" in node.inputs:
            node.inputs["Material"].default_value = material

    modifier = obj.modifiers.new(name="bV2_world", type="NODES")
    modifier.node_group = node_group


def remove_existing_object(name: str) -> None:
    existing = bpy.data.objects.get(name)
    if existing is None:
        return
    old_mesh = existing.data if existing.type == "MESH" else None
    for modifier in list(existing.modifiers):
        cleanup_modifier_node_group(modifier)
    bpy.data.objects.remove(existing, do_unlink=True)
    if old_mesh is not None and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def read_vtk_polydata(vtk_path: Path):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(vtk_path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    data = reader.GetOutput()
    if data is None or data.GetNumberOfPoints() == 0:
        raise ValueError(f"Failed to read point data from {vtk_path}")
    return data


def vtk_string_values(point_data, name: str, count: int) -> list[str]:
    array = point_data.GetAbstractArray(name)
    if array is None:
        raise KeyError(name)
    return [str(array.GetValue(i)) for i in range(count)]


def build_vtk_lookup(vtk_path: Path) -> dict[str, np.ndarray]:
    poly = read_vtk_polydata(vtk_path)
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64, copy=False)
    origin = points[0].astype(np.float64)
    grid_indices = np.rint(points - origin).astype(np.int32)
    index_min = grid_indices.min(axis=0)
    index_max = grid_indices.max(axis=0)
    dims = (index_max - index_min + 1).astype(np.int64)
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    shifted = (grid_indices - index_min).astype(np.int64)
    keys = shifted @ strides
    order = np.argsort(keys, kind="mergesort")

    point_data = poly.GetPointData()
    sim_turns = vtk_to_numpy(point_data.GetArray(ATTR_TURNS)).astype(np.int32, copy=False)
    sim_nodes = vtk_to_numpy(point_data.GetArray(ATTR_NODES)).astype(np.int32, copy=False)

    bio_raw = np.asarray(vtk_string_values(point_data, "scenario_bioEnvelope", poly.GetNumberOfPoints()), dtype=str)
    bio_raw[bio_raw == "nan"] = "none"
    bio_envelope = np.array([BIO_ENVELOPE_MAP.get(value, 0) for value in bio_raw], dtype=np.int32)
    bio_envelope_simple = np.array([BIO_ENVELOPE_SIMPLE_MAP.get(value, 0) for value in bio_raw], dtype=np.int32)

    proposal_arrays: dict[str, np.ndarray] = {}
    for attr_name in PROPOSAL_ATTRIBUTE_NAMES:
        array = point_data.GetArray(attr_name)
        if array is None:
            raise KeyError(f"VTK is missing required direct Blender proposal array {attr_name!r}: {vtk_path}")
        proposal_arrays[attr_name] = vtk_to_numpy(array).astype(np.int32, copy=False)

    cache = {
        "origin": origin,
        "index_min": index_min.astype(np.int32),
        "index_max": index_max.astype(np.int32),
        "strides": strides,
        "sorted_keys": keys[order].astype(np.int64),
        "sim_turns_sorted": sim_turns[order],
        "sim_nodes_sorted": sim_nodes[order],
        "bio_envelope_sorted": bio_envelope[order],
        "bio_envelope_simple_sorted": bio_envelope_simple[order],
    }
    for attr_name, values in proposal_arrays.items():
        cache[f"{attr_name}_sorted"] = values[order].astype(np.int32, copy=False)
    return cache


def local_neighbor_lookup(
    world_points: np.ndarray,
    origin: np.ndarray,
    index_min: np.ndarray,
    index_max: np.ndarray,
    strides: np.ndarray,
    sorted_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(world_points) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)

    base_indices = np.floor(world_points - origin).astype(np.int64)
    candidate_indices = base_indices[:, None, :] + LOCAL_NEIGHBOR_OFFSETS[None, :, :]
    in_bounds = np.all((candidate_indices >= index_min) & (candidate_indices <= index_max), axis=2)

    shifted = candidate_indices - index_min
    candidate_keys = shifted @ strides
    flat_keys = candidate_keys.reshape(-1)
    flat_in_bounds = in_bounds.reshape(-1)
    safe_keys = flat_keys.copy()
    safe_keys[~flat_in_bounds] = 0

    flat_insertion = np.searchsorted(sorted_keys, safe_keys, side="left")
    flat_found = flat_in_bounds & (flat_insertion < len(sorted_keys)) & (sorted_keys[flat_insertion] == safe_keys)

    n_points = len(world_points)
    candidate_centers = origin + candidate_indices.astype(np.float64)
    distances_sq = np.sum((candidate_centers - world_points[:, None, :]) ** 2, axis=2)
    found_matrix = flat_found.reshape(n_points, -1)
    insertion_matrix = flat_insertion.reshape(n_points, -1)
    distances_sq[~found_matrix] = np.inf

    best_neighbor = np.argmin(distances_sq, axis=1)
    has_match = np.isfinite(distances_sq[np.arange(n_points), best_neighbor])
    best_lookup = insertion_matrix[np.arange(n_points), best_neighbor]
    return has_match, best_lookup.astype(np.int64, copy=False)


def transfer_lookup_to_points(
    world_points: np.ndarray,
    cache: dict[str, np.ndarray],
    *,
    source_year: int,
) -> dict[str, np.ndarray]:
    total = len(world_points)
    origin = cache["origin"]
    index_min = cache["index_min"].astype(np.int64)
    index_max = cache["index_max"].astype(np.int64)
    strides = cache["strides"].astype(np.int64)
    sorted_keys = cache["sorted_keys"].astype(np.int64)
    sim_turns_sorted = cache["sim_turns_sorted"].astype(np.int32)
    sim_nodes_sorted = cache["sim_nodes_sorted"].astype(np.int32)
    bio_envelope_sorted = cache["bio_envelope_sorted"].astype(np.int32)
    bio_envelope_simple_sorted = cache["bio_envelope_simple_sorted"].astype(np.int32)
    proposal_sorted_arrays = {
        attr_name: cache[f"{attr_name}_sorted"].astype(np.int32)
        for attr_name in PROPOSAL_ATTRIBUTE_NAMES
    }

    out_turns = np.full(total, -1, dtype=np.int32)
    out_nodes = np.full(total, -1, dtype=np.int32)
    out_bio = np.zeros(total, dtype=np.int32)
    out_bio_simple = np.zeros(total, dtype=np.int32)
    out_match = np.zeros(total, dtype=np.int32)
    out_source_year = np.full(total, int(source_year), dtype=np.int32)
    out_proposals = {
        attr_name: np.zeros(total, dtype=np.int32)
        for attr_name in PROPOSAL_ATTRIBUTE_NAMES
    }

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        chunk_points = world_points[start:end]
        if len(chunk_points) == 0:
            continue

        target_indices = np.rint(chunk_points - origin).astype(np.int64)
        in_bounds = np.all((target_indices >= index_min) & (target_indices <= index_max), axis=1)
        if not np.any(in_bounds):
            continue

        valid_positions = np.flatnonzero(in_bounds)
        valid_indices = target_indices[in_bounds] - index_min
        keys = valid_indices @ strides

        insertion = np.searchsorted(sorted_keys, keys, side="left")
        found = (insertion < len(sorted_keys)) & (sorted_keys[insertion] == keys)
        if np.any(found):
            matched_positions = valid_positions[found] + start
            matched_lookup = insertion[found]
            out_turns[matched_positions] = sim_turns_sorted[matched_lookup]
            out_nodes[matched_positions] = sim_nodes_sorted[matched_lookup]
            out_bio[matched_positions] = bio_envelope_sorted[matched_lookup]
            out_bio_simple[matched_positions] = bio_envelope_simple_sorted[matched_lookup]
            out_match[matched_positions] = 1
            for attr_name, sorted_values in proposal_sorted_arrays.items():
                out_proposals[attr_name][matched_positions] = sorted_values[matched_lookup]

        if ENABLE_LOCAL_NEIGHBOR_FALLBACK:
            fallback_points = chunk_points[in_bounds][~found]
            fallback_positions = valid_positions[~found] + start
            fallback_has_match, fallback_lookup = local_neighbor_lookup(
                fallback_points,
                origin=origin,
                index_min=index_min,
                index_max=index_max,
                strides=strides,
                sorted_keys=sorted_keys,
            )
            if np.any(fallback_has_match):
                matched_positions = fallback_positions[fallback_has_match]
                matched_lookup = fallback_lookup[fallback_has_match]
                out_turns[matched_positions] = sim_turns_sorted[matched_lookup]
                out_nodes[matched_positions] = sim_nodes_sorted[matched_lookup]
                out_bio[matched_positions] = bio_envelope_sorted[matched_lookup]
                out_bio_simple[matched_positions] = bio_envelope_simple_sorted[matched_lookup]
                out_match[matched_positions] = 1
                for attr_name, sorted_values in proposal_sorted_arrays.items():
                    out_proposals[attr_name][matched_positions] = sorted_values[matched_lookup]

    transferred = {
        ATTR_TURNS: out_turns,
        ATTR_NODES: out_nodes,
        ATTR_BIO: out_bio,
        ATTR_BIO_SIMPLE: out_bio_simple,
        ATTR_MATCHED: out_match,
        ATTR_SOURCE_YEAR: out_source_year,
    }
    transferred.update(out_proposals)
    return transferred


def merge_source_and_transferred_payloads(
    source_payloads: dict[str, dict],
    transferred: dict[str, np.ndarray],
) -> dict[str, dict]:
    combined = {
        name: {
            "data_type": payload["data_type"],
            "field_name": payload["field_name"],
            "values": np.array(payload["values"], copy=True),
        }
        for name, payload in source_payloads.items()
    }
    for name, values in transferred.items():
        combined[name] = payload_from_int_values(values)
    return combined


def empty_source_payload_copy(source_payloads: dict[str, dict]) -> dict[str, dict]:
    empty_payloads: dict[str, dict] = {}
    for name, payload in source_payloads.items():
        values = payload["values"]
        shape = (0,) + values.shape[1:]
        empty_payloads[name] = {
            "data_type": payload["data_type"],
            "field_name": payload["field_name"],
            "values": np.empty(shape, dtype=values.dtype),
        }
    for name in TRANSFERRED_ATTRIBUTE_NAMES:
        empty_payloads[name] = payload_from_int_values(np.empty((0,), dtype=np.int32))
    return empty_payloads


def build_single_state_payload(
    source_obj: bpy.types.Object,
    cache: dict[str, np.ndarray],
    *,
    source_year: int,
) -> tuple[np.ndarray, dict[str, dict]]:
    world_points = mesh_vertex_world_positions(source_obj)
    source_payloads = extract_point_attribute_payloads(source_obj.data)
    transferred = transfer_lookup_to_points(world_points, cache, source_year=source_year)
    payloads = merge_source_and_transferred_payloads(source_payloads, transferred)
    return world_points.astype(np.float32, copy=False), payloads


def build_combined_timeline_payload(
    source_obj: bpy.types.Object,
    *,
    site: str,
    scenario: str,
) -> tuple[np.ndarray, dict[str, dict]]:
    site_spec = TIMELINE_SITE_SPECS[site]
    world_points = mesh_vertex_world_positions(source_obj)
    source_payloads = extract_point_attribute_payloads(source_obj.data)

    point_chunks: list[np.ndarray] = []
    attribute_chunks: dict[str, list[np.ndarray]] = {name: [] for name in source_payloads}
    transferred_chunks: dict[str, list[np.ndarray]] = {name: [] for name in TRANSFERRED_ATTRIBUTE_NAMES}
    vtk_cache_by_year: dict[int, dict[str, np.ndarray]] = {}

    for display_year in TIMELINE_YEARS:
        strip_spec = get_timeline_strip_spec(site, display_year)
        mask = clip_mask_for_points(world_points, strip_spec, site_spec)
        point_count = int(np.count_nonzero(mask))
        if point_count == 0:
            log("WORLD_TIMELINE_SLICE_EMPTY", site, scenario, source_obj.name, f"yr{display_year}")
            continue

        if display_year not in vtk_cache_by_year:
            vtk_cache_by_year[display_year] = build_vtk_lookup(
                resolve_vtk_path(site, scenario, display_year)
            )

        slice_points = world_points[mask]
        transferred = transfer_lookup_to_points(
            slice_points,
            vtk_cache_by_year[display_year],
            source_year=display_year,
        )
        translated_points = slice_points + get_timeline_translate(site, display_year)
        point_chunks.append(translated_points.astype(np.float32, copy=False))

        for name, payload in source_payloads.items():
            attribute_chunks[name].append(np.array(payload["values"][mask], copy=True))
        for name in TRANSFERRED_ATTRIBUTE_NAMES:
            transferred_chunks[name].append(transferred[name])

        log(
            "WORLD_TIMELINE_SLICE",
            site,
            scenario,
            source_obj.name,
            f"yr{display_year}",
            f"points={point_count}",
            f"matched={int(transferred[ATTR_MATCHED].sum())}",
        )

    if not point_chunks:
        return np.empty((0, 3), dtype=np.float32), empty_source_payload_copy(source_payloads)

    payloads: dict[str, dict] = {}
    for name, payload in source_payloads.items():
        payloads[name] = {
            "data_type": payload["data_type"],
            "field_name": payload["field_name"],
            "values": np.concatenate(attribute_chunks[name], axis=0),
        }
    for name in TRANSFERRED_ATTRIBUTE_NAMES:
        payloads[name] = payload_from_int_values(np.concatenate(transferred_chunks[name], axis=0))

    combined_points = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
    return combined_points, payloads


def build_world_object(
    source_obj: bpy.types.Object,
    target_collection: bpy.types.Collection,
    target_name: str,
    points: np.ndarray,
    payloads: dict[str, dict],
) -> bpy.types.Object:
    remove_existing_object(target_name)

    rebuilt = source_obj.copy()
    rebuilt.animation_data_clear()
    rebuilt.name = target_name
    rebuilt.matrix_world = Matrix.Identity(4)
    rebuilt.hide_render = False
    rebuilt.hide_viewport = False
    rebuilt.show_name = True

    for modifier in list(rebuilt.modifiers):
        cleanup_modifier_node_group(modifier)
        rebuilt.modifiers.remove(modifier)

    new_mesh = build_mesh_from_points(f"{target_name}_mesh", points, payloads)
    copy_material_slots(source_obj, new_mesh)
    rebuilt.data = new_mesh

    target_collection.objects.link(rebuilt)
    for collection in list(rebuilt.users_collection):
        if collection != target_collection:
            collection.objects.unlink(rebuilt)

    ensure_world_material_slot(rebuilt)
    ensure_world_modifier(rebuilt)
    return rebuilt


def build_world_attributes(scene: bpy.types.Scene | None = None) -> dict[str, list[str]]:
    config = get_runtime_config(scene)
    scene = config["scene"]
    site = str(config["site"])
    mode = str(config["mode"])
    year = config["year"]

    log("WORLD_BUILD_START", scene.name, site, mode, year if year is not None else "timeline")

    source_world_objects = get_source_world_objects(site)
    target_collections = {
        state: find_collection_by_role(scene, STATE_TO_COLLECTION_ROLE[state])
        for state in STATE_TO_COLLECTION_ROLE
    }
    for collection in target_collections.values():
        remove_collection_contents(collection)

    created: dict[str, list[str]] = {state: [] for state in STATE_TO_COLLECTION_ROLE}

    if mode == "single_state":
        active_year = get_active_years(mode, year)[0]
        vtk_caches = {
            state: build_vtk_lookup(resolve_vtk_path(site, state, active_year))
            for state in STATE_TO_COLLECTION_ROLE
        }
        for state, target_collection in target_collections.items():
            for kind, source_name in source_world_objects.items():
                source_obj = bpy.data.objects.get(source_name)
                if source_obj is None:
                    raise RuntimeError(f"Missing source world object {source_name!r} for site {site!r}")
                points, payloads = build_single_state_payload(
                    source_obj,
                    vtk_caches[state],
                    source_year=active_year,
                )
                target_name = make_world_object_name(kind, site, mode, state, year)
                obj = build_world_object(source_obj, target_collection, target_name, points, payloads)
                created[state].append(obj.name)
                log(
                    "WORLD_OBJECT_DONE",
                    scene.name,
                    obj.name,
                    f"points={len(points)}",
                    f"matched={int(payloads[ATTR_MATCHED]['values'].sum())}",
                )
    else:
        for state, target_collection in target_collections.items():
            for kind, source_name in source_world_objects.items():
                source_obj = bpy.data.objects.get(source_name)
                if source_obj is None:
                    raise RuntimeError(f"Missing source world object {source_name!r} for site {site!r}")
                points, payloads = build_combined_timeline_payload(
                    source_obj,
                    site=site,
                    scenario=state,
                )
                target_name = make_world_object_name(kind, site, mode, state, year)
                obj = build_world_object(source_obj, target_collection, target_name, points, payloads)
                created[state].append(obj.name)
                log(
                    "WORLD_OBJECT_DONE",
                    scene.name,
                    obj.name,
                    f"points={len(points)}",
                    f"matched={int(payloads[ATTR_MATCHED]['values'].sum())}",
                )

    scene["bV2_world_attributes_built"] = True
    log("WORLD_BUILD_DONE", scene.name, created)
    return created


def main() -> None:
    build_world_attributes()
    if SAVE_MAINFILE:
        bpy.ops.wm.save_mainfile()
        log("WORLD_SAVE_DONE")


if __name__ == "__main__":
    main()
