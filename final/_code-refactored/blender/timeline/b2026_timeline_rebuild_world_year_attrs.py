from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy
import numpy as np
import vtk
from mathutils import Matrix
from vtk.util.numpy_support import vtk_to_numpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_layout as timeline_layout
import b2026_timeline_scene_contract as scene_contract


SITE_KEY = os.environ["B2026_SITE_KEY"]
SCENE_NAME = os.environ.get("B2026_SCENE_NAME") or scene_contract.SITE_CONTRACTS[SITE_KEY]["scene_name"]
WORLD_SCENARIO = os.environ.get("B2026_WORLD_SCENARIO", "positive")
SAVE_MAINFILE = os.environ.get("B2026_SAVE_MAINFILE", "1") != "0"
ENABLE_LOCAL_NEIGHBOR_FALLBACK = True
CHUNK_SIZE = 1_000_000

ATTR_TURNS = "sim_Turns"
ATTR_NODES = "sim_Nodes"
ATTR_BIO = "scenario_bioEnvelope"
ATTR_BIO_SIMPLE = "scenario_bioEnvelopeSimple"
ATTR_MATCHED = "sim_Matched"

BIO_ENVELOPE_MAP = {
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
    "brownRoof": 2,
    "livingFacade": 3,
    "greenRoof": 4,
}

LOCAL_NEIGHBOR_OFFSETS = np.array(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=np.int64,
)

REPLACED_WORLD_ATTRIBUTE_NAMES = {
    ATTR_TURNS,
    ATTR_NODES,
    ATTR_BIO,
    ATTR_BIO_SIMPLE,
    ATTR_MATCHED,
    "timeline_year",
}


def log(message: str) -> None:
    print(f"[timeline_world_year_attrs] {message}")


def find_scene_child_collection(scene: bpy.types.Scene, suffix: str) -> bpy.types.Collection | None:
    target = suffix.lower()
    for collection in scene.collection.children:
        if collection.name.lower().endswith(target):
            return collection
    return None


def mesh_vertex_world_positions(obj: bpy.types.Object) -> np.ndarray:
    vertex_count = len(obj.data.vertices)
    if vertex_count == 0:
        return np.empty((0, 3), dtype=np.float32)

    coords = np.empty(vertex_count * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", coords)
    coords = coords.reshape((-1, 3)).astype(np.float64, copy=False)

    matrix_world = np.asarray(obj.matrix_world, dtype=np.float64)
    rotation_scale = matrix_world[:3, :3]
    translation = matrix_world[:3, 3]
    return (coords @ rotation_scale.T) + translation


def extract_point_attribute_payloads(mesh: bpy.types.Mesh) -> dict[str, dict]:
    payloads = {}
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
            value_length = point_count
            field_name = "value"
            dtype = np.int32 if attribute.data_type in {"INT", "BOOLEAN"} else np.float32
            values = np.empty(value_length, dtype=dtype)
        elif attribute.data_type == "FLOAT_VECTOR":
            value_length = point_count * 3
            field_name = "vector"
            values = np.empty(value_length, dtype=np.float32).reshape((-1, 3))
        elif attribute.data_type in {"FLOAT_COLOR", "BYTE_COLOR"}:
            value_length = point_count * 4
            field_name = "color"
            values = np.empty(value_length, dtype=np.float32).reshape((-1, 4))
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
        attribute = mesh.attributes.get(name)
        if attribute is not None:
            mesh.attributes.remove(attribute)
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


def ensure_int_attribute(mesh: bpy.types.Mesh, name: str):
    existing = mesh.attributes.get(name)
    if existing is not None:
        mesh.attributes.remove(existing)
    return mesh.attributes.new(name=name, type="INT", domain="POINT")


def assign_int_attribute(mesh: bpy.types.Mesh, name: str, values: np.ndarray) -> None:
    attr = ensure_int_attribute(mesh, name)
    attr.data.foreach_set("value", np.ascontiguousarray(values, dtype=np.int32))


def get_float_attribute_values(mesh: bpy.types.Mesh, name: str) -> np.ndarray | None:
    attr = mesh.attributes.get(name)
    if attr is None or attr.domain != "POINT" or attr.data_type not in {"FLOAT", "INT"}:
        return None
    values = np.empty(len(mesh.vertices), dtype=np.float32)
    attr.data.foreach_get("value", values)
    return values


def build_mesh_from_points(mesh_name: str, points: np.ndarray, payloads: dict[str, dict]) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(mesh_name)
    if len(points):
        mesh.vertices.add(len(points))
        mesh.vertices.foreach_set("co", points.reshape(-1))
    apply_point_attribute_payloads(mesh, payloads)
    mesh.update()
    return mesh


def copy_material_slots(source_obj: bpy.types.Object, target_mesh: bpy.types.Mesh) -> None:
    if getattr(source_obj, "data", None) is None:
        return
    for material in source_obj.data.materials:
        target_mesh.materials.append(material)


def replace_timeline_object(
    source_obj: bpy.types.Object,
    target_collection: bpy.types.Collection,
    target_name: str,
    points: np.ndarray,
    payloads: dict[str, dict],
):
    existing = bpy.data.objects.get(target_name)
    if existing is not None:
        old_mesh = existing.data if existing.type == "MESH" else None
        bpy.data.objects.remove(existing, do_unlink=True)
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

    duplicate = source_obj.copy()
    duplicate.animation_data_clear()
    duplicate.name = target_name
    duplicate.hide_render = False
    duplicate.hide_viewport = False
    duplicate.show_name = True
    duplicate.matrix_world = Matrix.Identity(4)
    duplicate["timeline_mode"] = "composite"
    duplicate["timeline_strip_count"] = 5

    new_mesh = build_mesh_from_points(f"{target_name}_mesh", points, payloads)
    copy_material_slots(source_obj, new_mesh)
    duplicate.data = new_mesh

    target_collection.objects.link(duplicate)
    for collection in list(duplicate.users_collection):
        if collection != target_collection:
            collection.objects.unlink(duplicate)
    return duplicate


def ensure_strip_box(site: str, strip_spec: dict, site_spec: dict, manager_collection):
    name = f"TimelineStripBox__{site}__{strip_spec['label']}"
    existing = bpy.data.objects.get(name)
    if existing is not None:
        return existing

    lengths = site_spec["box_length"]
    location = (
        strip_spec["box_position"][0] + lengths[0] / 2.0,
        strip_spec["box_position"][1] + lengths[1] / 2.0,
        strip_spec["box_position"][2] + lengths[2] / 2.0,
    )
    scale = (lengths[0] / 2.0, lengths[1] / 2.0, lengths[2] / 2.0)

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
    clip_box = bpy.context.active_object
    clip_box.name = name
    clip_box.scale = scale
    clip_box.display_type = "BOUNDS"
    clip_box.hide_render = True
    clip_box.show_name = True
    clip_box.show_in_front = True

    if clip_box.name not in manager_collection.objects:
        manager_collection.objects.link(clip_box)
    for collection in list(clip_box.users_collection):
        if collection != manager_collection:
            collection.objects.unlink(clip_box)
    return clip_box


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
    bio_raw = vtk_string_values(point_data, "scenario_bioEnvelope", poly.GetNumberOfPoints())
    bio_envelope = np.array([BIO_ENVELOPE_MAP.get(value, 0) for value in bio_raw], dtype=np.int32)
    bio_envelope_simple = np.array(
        [BIO_ENVELOPE_SIMPLE_MAP.get(value, 1 if value else 0) for value in bio_raw],
        dtype=np.int32,
    )

    return {
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


def transfer_lookup_to_points(world_points: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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

    out_turns = np.full(total, -1, dtype=np.int32)
    out_nodes = np.full(total, -1, dtype=np.int32)
    out_bio = np.zeros(total, dtype=np.int32)
    out_bio_simple = np.zeros(total, dtype=np.int32)
    out_match = np.zeros(total, dtype=np.int32)

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

    return {
        ATTR_TURNS: out_turns,
        ATTR_NODES: out_nodes,
        ATTR_BIO: out_bio,
        ATTR_BIO_SIMPLE: out_bio_simple,
        ATTR_MATCHED: out_match,
    }


def refresh_existing_timeline_object(obj: bpy.types.Object, site_spec: dict, scenario: str) -> None:
    world_points = mesh_vertex_world_positions(obj)
    timeline_years = get_float_attribute_values(obj.data, "timeline_year")
    if timeline_years is None:
        raise ValueError(f"{obj.name} is missing required point attribute 'timeline_year'")

    strip_by_year = {strip["year"]: strip for strip in site_spec["strips"]}
    out_turns = np.full(len(world_points), -1, dtype=np.int32)
    out_nodes = np.full(len(world_points), -1, dtype=np.int32)
    out_bio = np.zeros(len(world_points), dtype=np.int32)
    out_bio_simple = np.zeros(len(world_points), dtype=np.int32)
    out_match = np.zeros(len(world_points), dtype=np.int32)

    vtk_cache_by_year = {}

    for year in site_spec["timeline_years"]:
        mask = np.isclose(timeline_years, float(year))
        if not np.any(mask):
            continue
        position_year = timeline_layout.get_position_year(SITE_KEY, year)
        strip_spec = strip_by_year[position_year]
        translated_points = world_points[mask]
        original_points = translated_points - np.asarray(strip_spec["translate"], dtype=np.float64)
        if year not in vtk_cache_by_year:
            vtk_path = timeline_layout.resolve_state_vtk_path(SITE_KEY, scenario, year)
            vtk_cache_by_year[year] = build_vtk_lookup(vtk_path)
        transferred = transfer_lookup_to_points(original_points, vtk_cache_by_year[year])
        out_turns[mask] = transferred[ATTR_TURNS]
        out_nodes[mask] = transferred[ATTR_NODES]
        out_bio[mask] = transferred[ATTR_BIO]
        out_bio_simple[mask] = transferred[ATTR_BIO_SIMPLE]
        out_match[mask] = transferred[ATTR_MATCHED]

    assign_int_attribute(obj.data, ATTR_TURNS, out_turns)
    assign_int_attribute(obj.data, ATTR_NODES, out_nodes)
    assign_int_attribute(obj.data, ATTR_BIO, out_bio)
    assign_int_attribute(obj.data, ATTR_BIO_SIMPLE, out_bio_simple)
    assign_int_attribute(obj.data, ATTR_MATCHED, out_match)


def build_combined_timeline_payload(source_obj, site_spec: dict, scenario: str) -> tuple[np.ndarray, dict[str, dict]]:
    world_points = mesh_vertex_world_positions(source_obj)
    source_payloads = extract_point_attribute_payloads(source_obj.data)

    point_chunks = []
    attribute_chunks = {name: [] for name in source_payloads}
    world_attr_chunks = {
        ATTR_TURNS: [],
        ATTR_NODES: [],
        ATTR_BIO: [],
        ATTR_BIO_SIMPLE: [],
        ATTR_MATCHED: [],
        "timeline_year": [],
    }

    vtk_cache_by_year = {}

    strips_by_year = {strip["year"]: strip for strip in site_spec["strips"]}
    for display_year in site_spec["timeline_years"]:
        position_year = timeline_layout.get_position_year(SITE_KEY, display_year)
        strip_spec = strips_by_year[position_year]
        year = display_year
        vtk_path = timeline_layout.resolve_state_vtk_path(SITE_KEY, scenario, year)
        if year not in vtk_cache_by_year:
            vtk_cache_by_year[year] = build_vtk_lookup(vtk_path)

        mask = timeline_layout.clip_mask_for_points(world_points, strip_spec, site_spec)
        if not np.any(mask):
            continue

        slice_points = world_points[mask]
        transferred = transfer_lookup_to_points(slice_points, vtk_cache_by_year[year])

        point_chunks.append(timeline_layout.translate_points(slice_points, strip_spec))
        world_attr_chunks["timeline_year"].append(
            np.full(np.count_nonzero(mask), year, dtype=np.float32)
        )
        for name, payload in source_payloads.items():
            world_attr_chunks.setdefault(name, None)
            attribute_chunks[name].append(payload["values"][mask])
        for name, values in transferred.items():
            world_attr_chunks[name].append(values)

    if not point_chunks:
        empty_attrs = {}
        for name, payload in source_payloads.items():
            values = payload["values"]
            empty_shape = (0,) + values.shape[1:]
            empty_attrs[name] = {
                "data_type": payload["data_type"],
                "field_name": payload["field_name"],
                "values": np.empty(empty_shape, dtype=values.dtype if hasattr(values, "dtype") else np.float32),
            }
        for name in (ATTR_TURNS, ATTR_NODES, ATTR_BIO, ATTR_BIO_SIMPLE, ATTR_MATCHED):
            empty_attrs[name] = {"data_type": "INT", "field_name": "value", "values": np.empty((0,), dtype=np.int32)}
        empty_attrs["timeline_year"] = {"data_type": "FLOAT", "field_name": "value", "values": np.empty((0,), dtype=np.float32)}
        return np.empty((0, 3), dtype=np.float32), empty_attrs

    combined_points = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
    combined_attrs = {}
    for name, payload in source_payloads.items():
        combined_attrs[name] = {
            "data_type": payload["data_type"],
            "field_name": payload["field_name"],
            "values": np.concatenate(attribute_chunks[name], axis=0),
        }

    for name in (ATTR_TURNS, ATTR_NODES, ATTR_BIO, ATTR_BIO_SIMPLE, ATTR_MATCHED):
        combined_attrs[name] = {
            "data_type": "INT",
            "field_name": "value",
            "values": np.concatenate(world_attr_chunks[name], axis=0).astype(np.int32, copy=False),
        }
    combined_attrs["timeline_year"] = {
        "data_type": "FLOAT",
        "field_name": "value",
        "values": np.concatenate(world_attr_chunks["timeline_year"], axis=0).astype(np.float32, copy=False),
    }
    return combined_points, combined_attrs


def rebuild_site_world() -> None:
    site_spec = timeline_layout.get_timeline_site_spec(SITE_KEY)
    if site_spec is None:
        raise ValueError(f"No timeline site spec for {SITE_KEY}")

    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found")

    manager_collection = bpy.data.collections.get(site_spec["manager_collection_name"])
    if manager_collection is None:
        manager_collection = bpy.data.collections.get(scene_contract.get_collection_name(SITE_KEY, "manager"))
    if manager_collection is None:
        manager_collection = find_scene_child_collection(scene, "_manager") or find_scene_child_collection(scene, "manager")
    if manager_collection is None:
        raise ValueError("Manager collection not found")

    timeline_collection = bpy.data.collections.get(site_spec["world_collection_name"])
    if timeline_collection is None:
        timeline_collection = find_scene_child_collection(scene, "_world_timeline")
    if timeline_collection is None:
        raise ValueError(f"Timeline base collection '{site_spec['world_collection_name']}' not found")

    cube_collection_name = scene_contract.get_collection_name(SITE_KEY, "base_cubes", legacy=True) + "_Timeline"
    timeline_cube_collection = bpy.data.collections.get(cube_collection_name)
    if timeline_cube_collection is None:
        timeline_cube_collection = find_scene_child_collection(scene, "_worldcubes_timeline")

    for strip_spec in site_spec["strips"]:
        ensure_strip_box(SITE_KEY, strip_spec, site_spec, manager_collection)

    created_points = []
    created_cubes = []
    for object_name in site_spec["source_world_objects"]:
        source_obj = bpy.data.objects.get(object_name)
        timeline_name = f"{object_name}__timeline"
        existing_timeline = bpy.data.objects.get(timeline_name)

        if source_obj is not None:
            combined_points, combined_payloads = build_combined_timeline_payload(source_obj, site_spec, WORLD_SCENARIO)
            created_points.append(
                replace_timeline_object(
                    source_obj,
                    timeline_collection,
                    timeline_name,
                    combined_points,
                    combined_payloads,
                ).name
            )

            cube_source = bpy.data.objects.get(f"{object_name}_cubes")
            if cube_source is not None and timeline_cube_collection is not None:
                created_cubes.append(
                    replace_timeline_object(
                        cube_source,
                        timeline_cube_collection,
                        f"{object_name}_cubes__timeline",
                        combined_points,
                        combined_payloads,
                    ).name
                )
            continue

        if existing_timeline is not None:
            refresh_existing_timeline_object(existing_timeline, site_spec, WORLD_SCENARIO)
            created_points.append(existing_timeline.name)
        else:
            log(f"missing source and timeline world object: {object_name}")

        existing_timeline_cube = bpy.data.objects.get(f"{object_name}_cubes__timeline")
        if existing_timeline_cube is not None:
            refresh_existing_timeline_object(existing_timeline_cube, site_spec, WORLD_SCENARIO)
            created_cubes.append(existing_timeline_cube.name)

    log(
        f"site={SITE_KEY} scene={SCENE_NAME} scenario={WORLD_SCENARIO} "
        f"created_points={created_points} created_cubes={created_cubes}"
    )


def main() -> None:
    rebuild_site_world()
    if SAVE_MAINFILE:
        bpy.ops.wm.save_mainfile()
        log("Saved mainfile")


if __name__ == "__main__":
    main()
