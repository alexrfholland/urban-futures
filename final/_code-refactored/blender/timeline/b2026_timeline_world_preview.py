import importlib.util
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_SITES = ("trimmed-parade", "city", "uni")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract


def load_local_module(module_name: str, filename: str):
    file_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def delete_collection_tree(collection):
    for child in list(collection.children):
        delete_collection_tree(child)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def ensure_collection(name: str, parent=None):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)

    target_parent = parent or bpy.context.scene.collection
    if collection.name not in target_parent.children:
        target_parent.children.link(collection)
    return collection


def reset_collection(name: str, parent=None):
    existing = bpy.data.collections.get(name)
    if existing is not None:
        delete_collection_tree(existing)
    return ensure_collection(name, parent=parent)


def ensure_strip_box(site: str, strip_spec: dict, site_spec: dict, manager_collection):
    name = f"TimelineStripBox__{site}__{strip_spec['label']}"
    existing = bpy.data.objects.get(name)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)

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

        if attribute.data_type in {"FLOAT", "INT", "BOOLEAN"}:
            value_length = point_count
            value_shape = ()
            field_name = "value"
        elif attribute.data_type == "FLOAT_VECTOR":
            value_length = point_count * 3
            value_shape = (3,)
            field_name = "vector"
        elif attribute.data_type in {"FLOAT_COLOR", "BYTE_COLOR"}:
            value_length = point_count * 4
            value_shape = (4,)
            field_name = "color"
        else:
            continue

        values = np.empty(value_length, dtype=np.float32)
        attribute.data.foreach_get(field_name, values)
        if value_shape:
            values = values.reshape((-1, value_shape[0]))

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
        attribute = mesh.attributes.new(
            name=name,
            type=payload["data_type"],
            domain="POINT",
        )
        values = payload["values"]
        flat_values = values.reshape(-1) if getattr(values, "ndim", 1) > 1 else values
        if payload["data_type"] in {"INT", "BOOLEAN"}:
            typed_values = np.ascontiguousarray(flat_values, dtype=np.int32)
        else:
            typed_values = np.ascontiguousarray(flat_values, dtype=np.float32)
        attribute.data.foreach_set(payload["field_name"], typed_values)
        if hasattr(attribute, "is_runtime_only"):
            attribute.is_runtime_only = False


def build_combined_timeline_payload(source_obj, site_spec: dict, timeline_layout):
    world_points = mesh_vertex_world_positions(source_obj)
    source_payloads = extract_point_attribute_payloads(source_obj.data)

    point_chunks = []
    attribute_chunks = {name: [] for name in source_payloads}
    timeline_year_chunks = []

    for strip_spec in site_spec["strips"]:
        mask = timeline_layout.clip_mask_for_points(world_points, strip_spec, site_spec)
        if not np.any(mask):
            continue

        point_chunks.append(timeline_layout.translate_points(world_points[mask], strip_spec))
        timeline_year_chunks.append(
            np.full(np.count_nonzero(mask), strip_spec["year"], dtype=np.float32)
        )
        for name, payload in source_payloads.items():
            attribute_chunks[name].append(payload["values"][mask])

    if point_chunks:
        combined_points = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
        combined_attrs = {}
        for name, payload in source_payloads.items():
            values = np.concatenate(attribute_chunks[name], axis=0)
            combined_attrs[name] = {
                "data_type": payload["data_type"],
                "field_name": payload["field_name"],
                "values": values,
            }
        combined_attrs["timeline_year"] = {
            "data_type": "FLOAT",
            "field_name": "value",
            "values": np.concatenate(timeline_year_chunks, axis=0),
        }
        return combined_points, combined_attrs

    empty_attrs = {}
    for name, payload in source_payloads.items():
        values = payload["values"]
        empty_shape = (0,) + values.shape[1:]
        empty_attrs[name] = {
            "data_type": payload["data_type"],
            "field_name": payload["field_name"],
            "values": np.empty(empty_shape, dtype=np.float32),
        }
    empty_attrs["timeline_year"] = {
        "data_type": "FLOAT",
        "field_name": "value",
        "values": np.empty((0,), dtype=np.float32),
    }
    return np.empty((0, 3), dtype=np.float32), empty_attrs


def build_mesh_from_points(mesh_name: str, points: np.ndarray, payloads: dict[str, dict]) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(mesh_name)
    vertex_count = len(points)
    if vertex_count:
        mesh.vertices.add(vertex_count)
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


def build_site_timeline_world(site: str, clipbox_setup, timeline_layout):
    site_spec = timeline_layout.get_timeline_site_spec(site)
    if site_spec is None:
        raise ValueError(f"No timeline site spec for {site}")

    scene = bpy.data.scenes.get(site_spec["scene_name"])
    if scene is None:
        raise ValueError(f"Scene '{site_spec['scene_name']}' not found")

    manager_collection = bpy.data.collections.get(site_spec["manager_collection_name"])
    if manager_collection is None:
        raise ValueError(f"Collection '{site_spec['manager_collection_name']}' not found")

    base_parent = bpy.data.collections.get(
        scene_contract.get_collection_name(site, "base")
    )
    if base_parent is None:
        raise ValueError(
            f"Collection '{scene_contract.get_collection_name(site, 'base')}' not found"
        )
    timeline_collection = reset_collection(
        site_spec["world_collection_name"],
        parent=base_parent,
    )
    legacy_cube_collection_name = (
        scene_contract.get_collection_name(site, "base_cubes", legacy=True) + "_Timeline"
    )
    cube_parent = bpy.data.collections.get(
        scene_contract.get_collection_name(site, "base_cubes")
    )
    if cube_parent is None:
        raise ValueError(
            f"Collection '{scene_contract.get_collection_name(site, 'base_cubes')}' not found"
        )
    timeline_cube_collection = reset_collection(
        legacy_cube_collection_name,
        parent=cube_parent,
    )

    created_objects = []
    created_cube_objects = []
    for strip_spec in site_spec["strips"]:
        ensure_strip_box(site, strip_spec, site_spec, manager_collection)

    for object_name in site_spec["source_world_objects"]:
        source_obj = bpy.data.objects.get(object_name)
        if source_obj is None:
            print(f"WARNING missing source world object: {object_name}")
            continue

        combined_points, combined_payloads = build_combined_timeline_payload(
            source_obj,
            site_spec,
            timeline_layout,
        )
        created_objects.append(
            replace_timeline_object(
                source_obj,
                timeline_collection,
                f"{object_name}__timeline",
                combined_points,
                combined_payloads,
            ).name
        )

        cube_source = bpy.data.objects.get(f"{object_name}_cubes")
        if cube_source is not None:
            created_cube_objects.append(
                replace_timeline_object(
                    cube_source,
                    timeline_cube_collection,
                    f"{object_name}_cubes__timeline",
                    combined_points,
                    combined_payloads,
                ).name
            )

    print(
        f"TIMELINE_WORLD site={site} collection={timeline_collection.name} "
        f"created_points={len(created_objects)} created_cubes={len(created_cube_objects)}"
    )
    return created_objects


def main():
    clipbox_setup = load_local_module(
        "b2026_clipbox_setup_timeline_world_runtime",
        "b2026_timeline_clipbox_setup.py",
    )
    timeline_layout = load_local_module(
        "b2026_timeline_layout_timeline_world_runtime",
        "b2026_timeline_layout.py",
    )

    for site in TARGET_SITES:
        build_site_timeline_world(site, clipbox_setup, timeline_layout)

    bpy.ops.wm.save_mainfile()
    print("Saved blend with timeline world preview collections")


if __name__ == "__main__":
    main()
