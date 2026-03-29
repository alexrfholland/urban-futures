from pathlib import Path
import math
import re
import sys

import bpy
from mathutils import Vector


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

from refactor_code.paths import hook_log_ply_library_dir, hook_tree_ply_library_dir

DEFAULT_LOG_FILEPATH = REPO_ROOT / "data/blender/2026/logs/instancer_city_180_trending_20260323_024211.log"
TREE_PLY_FOLDER = hook_tree_ply_library_dir()
LOG_PLY_FOLDER = hook_log_ply_library_dir()
TARGET_MATERIAL_NAME = "MINIMAL_RESOURCES"

GRID_COLLECTION_PREFIX = "ModelGrid"
GRID_SPACING_X = 6.0
GRID_SPACING_Y = 8.0
ROW_SIZE = 8
APPLY_ROTATION_X_DEGREES = 0.0
LABEL_SIZE = 0.45
LABEL_Y_OFFSET = 1.5
LABEL_Z_OFFSET = 0.02


def parse_scalar_metadata(lines):
    metadata = {}
    for line in lines:
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key in {"run_id", "site", "scenario", "year"}:
            metadata[key] = value.strip()
    return metadata


def collect_section_rows(lines, section_name):
    rows = []
    in_section = False
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped == f"{section_name}:":
            in_section = True
            continue
        if not in_section:
            continue
        if not stripped.strip():
            break
        if stripped.lstrip().startswith("resolved_filename"):
            continue
        if stripped.endswith(":") and not stripped.startswith(" "):
            break
        rows.append(stripped)
    return rows


def parse_resolved_model_counts(lines, section_name):
    entries = []
    for row in collect_section_rows(lines, section_name):
        match = re.match(r"^\s*(?P<filename>\S+)\s+(?P<count>\d+)\s*$", row)
        if not match:
            continue
        entries.append(
            {
                "filename": match.group("filename"),
                "count": int(match.group("count")),
            }
        )
    return entries


def resolve_ply_path(filename, node_type):
    base_folder = TREE_PLY_FOLDER if node_type == "tree" else LOG_PLY_FOLDER
    return base_folder / filename


def delete_collection_tree(collection):
    for child in list(collection.children):
        delete_collection_tree(child)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for parent in list(bpy.data.collections):
        if parent.children.get(collection.name) is not None:
            parent.children.unlink(collection)
    bpy.data.collections.remove(collection)


def ensure_clean_collection(name, parent_collection):
    existing = bpy.data.collections.get(name)
    if existing is not None:
        delete_collection_tree(existing)
    collection = bpy.data.collections.new(name)
    parent_collection.children.link(collection)
    return collection


def import_ply_object(filepath):
    before_names = set(bpy.data.objects.keys())
    bpy.ops.wm.ply_import(filepath=str(filepath))
    new_names = [name for name in bpy.data.objects.keys() if name not in before_names]
    new_objects = [bpy.data.objects[name] for name in new_names]
    mesh_objects = [obj for obj in new_objects if obj.type == "MESH"]
    if mesh_objects:
        return mesh_objects[0]
    if new_objects:
        return new_objects[0]
    return None


def assign_material(obj, material_name):
    if obj is None or obj.type != "MESH":
        return
    material = bpy.data.materials.get(material_name)
    if material is None:
        raise ValueError(f"Material '{material_name}' was not found.")
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def dimensions_from_world_bbox(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mins = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maxs = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return maxs - mins


def world_bbox(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mins = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maxs = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return mins, maxs


def layout_objects_in_grid(objects):
    if not objects:
        return

    col_count = max(1, ROW_SIZE)
    cell_width = max((dimensions_from_world_bbox(obj).x for obj in objects), default=1.0) + GRID_SPACING_X
    cell_depth = max((dimensions_from_world_bbox(obj).y for obj in objects), default=1.0) + GRID_SPACING_Y

    for index, obj in enumerate(objects):
        row = index // col_count
        col = index % col_count
        obj.location = Vector((col * cell_width, -row * cell_depth, 0.0))

        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_z = min(v.z for v in bbox)
        obj.location.z -= min_z


def format_tree_label(filename):
    stem = Path(filename).stem
    match = re.match(
        r"^precolonial\.(True|False)_size\.([^_]+)_control\.([^_]+)_id\.(\d+)$",
        stem,
    )
    if not match:
        return stem
    epoch = "Precolonial" if match.group(1) == "True" else "Colonial"
    size = match.group(2)
    control = match.group(3)
    tree_id = match.group(4)
    return f"{epoch}\n{size}\n{control}\ntree {tree_id}"


def format_log_label(filename):
    stem = Path(filename).stem
    match = re.match(r"^size\.([^\.]+)\.log\.(\d+)$", stem)
    if not match:
        return stem
    size = match.group(1)
    log_id = match.group(2)
    return f"Log\n{size}\nid {log_id}"


def create_text_label(section_collection, target_obj, node_type, filename):
    curve = bpy.data.curves.new(
        name=f"{target_obj.name}__label_curve",
        type="FONT",
    )
    curve.body = format_tree_label(filename) if node_type == "tree" else format_log_label(filename)
    curve.size = LABEL_SIZE
    curve.align_x = "CENTER"
    curve.align_y = "TOP"

    label = bpy.data.objects.new(f"{target_obj.name}__label", curve)
    section_collection.objects.link(label)

    mins, maxs = world_bbox(target_obj)
    label.location = Vector((
        (mins.x + maxs.x) / 2.0,
        mins.y - LABEL_Y_OFFSET,
        0.0 + LABEL_Z_OFFSET,
    ))
    label.rotation_euler = (0.0, 0.0, 0.0)
    return label


def build_grid_from_log(log_path):
    lines = log_path.read_text(encoding="utf-8").splitlines()
    metadata = parse_scalar_metadata(lines)
    tree_entries = parse_resolved_model_counts(lines, "tree_counts_by_resolved_model")
    log_entries = parse_resolved_model_counts(lines, "log_counts_by_resolved_model")

    if not tree_entries and not log_entries:
        raise ValueError(f"No tree/log model sections found in {log_path}")

    run_id = metadata.get("run_id", log_path.stem)
    root_collection_name = f"{GRID_COLLECTION_PREFIX}_{run_id}"
    scene = bpy.context.scene
    root_collection = ensure_clean_collection(root_collection_name, scene.collection)

    imported_objects = []
    imported_entries = []

    for node_type, entries in (("tree", tree_entries), ("log", log_entries)):
        if not entries:
            continue
        section_collection = bpy.data.collections.new(f"{node_type}_{run_id}")
        root_collection.children.link(section_collection)

        for entry in entries:
            filepath = resolve_ply_path(entry["filename"], node_type)
            if not filepath.exists():
                print(f"Missing {node_type} PLY referenced by log: {filepath}")
                continue

            obj = import_ply_object(filepath)
            if obj is None:
                print(f"Failed to import {filepath}")
                continue

            obj.name = f"{node_type}__{filepath.stem}"
            for collection in list(obj.users_collection):
                collection.objects.unlink(obj)
            section_collection.objects.link(obj)
            assign_material(obj, TARGET_MATERIAL_NAME)
            obj.rotation_euler.x = math.radians(APPLY_ROTATION_X_DEGREES)
            imported_objects.append(obj)
            imported_entries.append((section_collection, obj, node_type, entry["filename"]))

    layout_objects_in_grid(imported_objects)

    for section_collection, obj, node_type, filename in imported_entries:
        create_text_label(section_collection, obj, node_type, filename)

    print(f"Grid collection: {root_collection.name}")
    print(f"Imported trees: {len(tree_entries)}")
    print(f"Imported logs: {len(log_entries)}")
    return {
        "collection_name": root_collection.name,
        "tree_count": len(tree_entries),
        "log_count": len(log_entries),
    }


def main():
    log_path = Path(DEFAULT_LOG_FILEPATH)
    if not log_path.exists():
        raise ValueError(f"Log file not found: {log_path}")
    result = build_grid_from_log(log_path)
    print(result)


if __name__ == "__main__":
    main()
