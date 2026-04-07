from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


PROPOSALS_MATERIAL_NAME = "PROPOSALS"
WORLD_POINT_GROUP_NAMES = (
    "Background",
    "Background - Large pts",
    "Background.001",
    "Background - Large pts.001",
)
WORLD_CUBE_GROUP_NAMES = (
    "Background Cubes",
    "Background - Large pts Cubes",
    "Background.001 Cubes",
    "Background - Large pts.001 Cubes",
)


def load_builder():
    script_path = SCRIPT_DIR.parents[2] / "_blender" / "2026" / "PROPOSALS.py"
    spec = importlib.util.spec_from_file_location("b2026_proposals_builder", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PROPOSALS builder from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_material():
    module = load_builder()
    if not hasattr(module, "build_material"):
        raise AttributeError("PROPOSALS builder does not expose build_material()")
    return module.build_material()


def apply_to_instancers(material: bpy.types.Material) -> list[str]:
    updated = []
    for node_group in bpy.data.node_groups:
        if not node_group.name.startswith(("tree_", "log_", "pole_", "instance_template")):
            continue
        changed = False
        for node in node_group.nodes:
            if node.bl_idname != "GeometryNodeSetMaterial" or "Material" not in node.inputs:
                continue
            if node.inputs["Material"].default_value != material:
                node.inputs["Material"].default_value = material
                changed = True
        if changed:
            updated.append(node_group.name)
    return sorted(set(updated))


def apply_to_world_point_groups(material: bpy.types.Material) -> list[str]:
    updated = []
    for node_group_name in WORLD_POINT_GROUP_NAMES + WORLD_CUBE_GROUP_NAMES:
        node_group = bpy.data.node_groups.get(node_group_name)
        if node_group is None:
            continue
        changed = False
        for node in node_group.nodes:
            if node.bl_idname != "GeometryNodeSetMaterial" or "Material" not in node.inputs:
                continue
            if node.inputs["Material"].default_value != material:
                node.inputs["Material"].default_value = material
                changed = True
        if changed:
            updated.append(node_group_name)
    return sorted(set(updated))


def apply_to_direct_material_slots(material: bpy.types.Material) -> list[str]:
    updated = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or getattr(obj, "data", None) is None:
            continue
        if not (
            obj.name.startswith(("TreePositions_", "LogPositions_"))
            or "_buildings.001__yr" in obj.name
            or "_highResRoad.001__yr" in obj.name
            or obj.name.endswith("_buildings.001")
            or obj.name.endswith("_highResRoad.001")
        ):
            continue
        materials = obj.data.materials
        if len(materials) == 0:
            materials.append(material)
            updated.append(obj.name)
            continue
        if materials[0] != material:
            materials[0] = material
            updated.append(obj.name)
    return sorted(set(updated))


def main():
    material = ensure_material()
    instancer_groups = apply_to_instancers(material)
    world_groups = apply_to_world_point_groups(material)
    direct_slots = apply_to_direct_material_slots(material)
    print(f"Applied {PROPOSALS_MATERIAL_NAME} to instancer groups: {instancer_groups}")
    print(f"Applied {PROPOSALS_MATERIAL_NAME} to world groups: {world_groups}")
    print(f"Applied {PROPOSALS_MATERIAL_NAME} to direct material slots: {direct_slots}")


if __name__ == "__main__":
    main()
