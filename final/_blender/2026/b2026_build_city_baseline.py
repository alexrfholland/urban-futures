from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import bpy
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

from refactor_code.paths import hook_baseline_terrain_ply_path, hook_baseline_trees_csv_path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_SCENE_NAME = os.environ.get("B2026_BASELINE_SOURCE_SCENE", "city")
SOURCE_SCENE_BACKUP_NAME = os.environ.get("B2026_BASELINE_SOURCE_BACKUP_SCENE", "city_source")
BASELINE_SCENE_NAME = os.environ.get("B2026_BASELINE_SCENE_NAME", "city")
BASELINE_TERRAIN_PLY = Path(
    os.environ.get(
        "B2026_BASELINE_TERRAIN_PLY",
        str(hook_baseline_terrain_ply_path("city", 1)),
    )
)
BASELINE_TREE_CSV = Path(
    os.environ.get(
        "B2026_BASELINE_TREE_CSV",
        str(hook_baseline_trees_csv_path("city")),
    )
)
BASELINE_RENDER_PATH = Path(
    os.environ.get(
        "B2026_BASELINE_RENDER_PATH",
        str(REPO_ROOT / "data" / "blender" / "2026" / "baseline_renders" / "city_baseline_pathway.png"),
    )
)
SAVE_MAINFILE = os.environ.get("B2026_BASELINE_SAVE_MAINFILE", "1") != "0"
RENDER_STILL = os.environ.get("B2026_BASELINE_RENDER", "1") != "0"
MUTE_FILE_OUTPUTS = os.environ.get("B2026_BASELINE_MUTE_FILE_OUTPUTS", "1") != "0"
WORLD_CUBE_SIZE = (1.0, 1.0, 0.25)
BASELINE_TREE_COLLECTION = "Year_city_180_positive"
BASELINE_PRIORITY_COLLECTION = f"{BASELINE_TREE_COLLECTION}_priority"
BASELINE_PATHWAY_VIEW_LAYER = "pathway_state"
BASELINE_EXR_VIEW_LAYERS = {
    "pathway_state",
    "existing_condition",
    "city_priority",
}
BASELINE_DISABLED_VIEW_LAYERS = {
    "trending_state",
    "city_bioenvelope",
}
REMOVE_ROOT_COLLECTIONS = (
    "city_envelope",
    "Envelope_trending",
    "Year_city_180_trending",
)
KEEP_ROOT_COLLECTIONS = {
    "City_World",
    "City_Manager",
    "City_Camera",
    "City_Camera_Archive",
    "City_WorldCubes",
    BASELINE_TREE_COLLECTION,
    BASELINE_PRIORITY_COLLECTION,
}
BASELINE_TERRAIN_GROUP_MATERIALS = {
    "Background - Large pts": "BASELINE_GROUND_WHITE",
    "Background - Large pts.001": "BASELINE_GROUND_WHITE",
    "Background - Large pts.001 Baseline Terrain Cubes": "BASELINE_GROUND_WHITE",
}


def log(message: str) -> None:
    print(f"[city_baseline] {message}", flush=True)


def load_local_module(module_name: str, filename: str):
    script_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remove_object_if_present(name: str) -> bool:
    obj = bpy.data.objects.get(name)
    if obj is None:
        return False
    bpy.data.objects.remove(obj, do_unlink=True)
    return True


def remove_collection_from_scene(scene: bpy.types.Scene, collection_name: str) -> None:
    collection = scene.collection.children.get(collection_name)
    if collection is not None:
        scene.collection.children.unlink(collection)
        log(f"Unlinked root collection from scene: {collection_name}")


def remove_collection_tree_everywhere(collection_name: str) -> None:
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return

    for child in list(collection.children):
        remove_collection_tree_everywhere(child.name)

    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for scene in bpy.data.scenes:
        if scene.collection.children.get(collection_name) is not None:
            scene.collection.children.unlink(collection)

    for parent in bpy.data.collections:
        if parent.children.get(collection_name) is not None:
            parent.children.unlink(collection)

    bpy.data.collections.remove(collection)
    log(f"Deleted collection tree: {collection_name}")


def ensure_scene_copy() -> bpy.types.Scene:
    source_scene = bpy.data.scenes.get(SOURCE_SCENE_NAME)
    if source_scene is None:
        raise ValueError(f"Source scene '{SOURCE_SCENE_NAME}' was not found")

    if SOURCE_SCENE_BACKUP_NAME in bpy.data.scenes and BASELINE_SCENE_NAME in bpy.data.scenes:
        baseline_scene = bpy.data.scenes.get(BASELINE_SCENE_NAME)
        if baseline_scene is None:
            raise ValueError("Baseline scene lookup failed")
        return baseline_scene

    baseline_scene = source_scene.copy()
    source_scene.name = SOURCE_SCENE_BACKUP_NAME
    baseline_scene.name = BASELINE_SCENE_NAME
    baseline_scene.camera = source_scene.camera
    log(f"Created scene copy '{BASELINE_SCENE_NAME}' from '{SOURCE_SCENE_BACKUP_NAME}'")
    return baseline_scene


def import_ply_object(filepath: Path) -> bpy.types.Object:
    before_names = set(bpy.data.objects.keys())
    bpy.ops.wm.ply_import(filepath=str(filepath))
    new_names = [name for name in bpy.data.objects.keys() if name not in before_names]
    new_objects = [bpy.data.objects[name] for name in new_names]
    mesh_objects = [obj for obj in new_objects if obj.type == "MESH"]
    if mesh_objects:
        return mesh_objects[0]
    if new_objects:
        return new_objects[0]
    raise RuntimeError(f"PLY import produced no objects: {filepath}")


def replace_terrain_mesh() -> bpy.types.Object:
    if not BASELINE_TERRAIN_PLY.exists():
        raise FileNotFoundError(f"Baseline terrain PLY not found: {BASELINE_TERRAIN_PLY}")

    target_obj = bpy.data.objects.get("city_highResRoad.001")
    if target_obj is None:
        raise ValueError("Expected object 'city_highResRoad.001' was not found")

    imported = import_ply_object(BASELINE_TERRAIN_PLY)
    target_obj.data = imported.data
    target_obj.data.name = "city_highResRoad.001_baseline_mesh"
    target_obj.location = imported.location
    target_obj.rotation_euler = imported.rotation_euler
    target_obj.scale = imported.scale
    target_obj.pass_index = 1
    bpy.data.objects.remove(imported, do_unlink=True)
    log(f"Replaced terrain mesh on {target_obj.name} from {BASELINE_TERRAIN_PLY.name}")
    return target_obj


def rebuild_terrain_world_cubes(terrain_obj: bpy.types.Object) -> None:
    world_cubes = load_local_module("b2026_world_cubes_runtime", "b2026_world_cubes.py")
    existing_duplicate = bpy.data.objects.get("city_highResRoad.001_cubes")
    if existing_duplicate is not None:
        bpy.data.objects.remove(existing_duplicate, do_unlink=True)
        log("Removed existing terrain cube duplicate")

    baseline_group_name = "Background - Large pts.001 Baseline Terrain Cubes"
    existing_group = bpy.data.node_groups.get(baseline_group_name)
    if existing_group is not None:
        bpy.data.node_groups.remove(existing_group)
        log(f"Removed old node group: {baseline_group_name}")

    original_suffix = world_cubes.GROUP_SUFFIX
    original_voxel_size = world_cubes.VOXEL_SIZE
    try:
        world_cubes.GROUP_SUFFIX = " Baseline Terrain Cubes"
        world_cubes.VOXEL_SIZE = WORLD_CUBE_SIZE
        replacement_group = world_cubes.build_cube_group("Background - Large pts.001")
        if replacement_group is None:
            raise RuntimeError("Failed to build baseline terrain cube node group")
        duplicate_name = world_cubes.duplicate_with_cube_group(
            terrain_obj,
            replacement_group,
            "City_WorldCubes",
        )
        duplicate = bpy.data.objects.get(duplicate_name)
        if duplicate is not None:
            duplicate.pass_index = 1
            duplicate.hide_render = False
            duplicate.hide_viewport = False
            duplicate.hide_select = False
        log(f"Rebuilt terrain cubes with voxel size {WORLD_CUBE_SIZE}: {duplicate_name}")
    finally:
        world_cubes.GROUP_SUFFIX = original_suffix
        world_cubes.VOXEL_SIZE = original_voxel_size


def build_baseline_tree_distribution(scene: bpy.types.Scene) -> None:
    if not BASELINE_TREE_CSV.exists():
        raise FileNotFoundError(f"Baseline tree CSV not found: {BASELINE_TREE_CSV}")

    inst = load_local_module("b2026_instancer_runtime", "b2026_instancer.py")
    inst.TARGET_SCENE_NAME = scene.name
    inst.SITE = "city"
    inst.SCENARIO = "positive"
    inst.SCENARIOS_TO_BUILD = ("positive",)
    inst.USE_3D_CURSOR_FILTER = False
    inst.USE_CAMERA_VIEW_FILTER = False
    inst.DISTANCE_UNITS = 10000
    inst.WRITE_RUN_LOG = True
    inst.refresh_site_paths()
    inst.BASE_PATH = str(BASELINE_TREE_CSV.parent)
    inst.CSV_FILENAME = BASELINE_TREE_CSV.name
    inst.CSV_FILEPATH = str(BASELINE_TREE_CSV)

    inst.cleanup_existing_run_artifacts()

    df = pd.read_csv(BASELINE_TREE_CSV)
    df["nodeType"] = "tree"
    df, size_override_summary = inst.randomize_trending_tree_sizes(df)
    df, tree_id_remap_summary = inst.remap_tree_ids_to_available_models(df, inst.PLY_FOLDER)
    log(
        "Baseline tree CSV rows="
        f"{len(df)}, remapped={tree_id_remap_summary['remapped_count']}, "
        f"already_valid={tree_id_remap_summary['already_valid_count']}"
    )

    year_collection = bpy.data.collections.new(BASELINE_TREE_COLLECTION)
    scene.collection.children.link(year_collection)
    tree_results = inst.process_collection(df.copy(), inst.PLY_FOLDER, "tree", year_collection)
    if tree_results[0] is None:
        raise RuntimeError("Baseline tree instancing did not produce a point cloud")
    force_collection_tree_pass_index(year_collection, 3)

    baseline_priority_sizes = set(inst.PRIORITY_TREE_SIZES) | {"large"}
    priority_tree_df = df[
        df["size"].astype(str).str.lower().isin(baseline_priority_sizes)
    ].copy()
    if len(priority_tree_df) > 0 and scene.view_layers.get("city_priority") is not None:
        priority_collection = bpy.data.collections.new(BASELINE_PRIORITY_COLLECTION)
        scene.collection.children.link(priority_collection)
        priority_results = inst.process_collection(
            priority_tree_df,
            inst.PLY_FOLDER,
            "tree",
            priority_collection,
            variant_suffix="priority",
        )
        if priority_results[0] is None:
            raise RuntimeError("Baseline priority tree instancing did not produce a point cloud")
        force_collection_tree_pass_index(priority_collection, 3)
        log(
            "Built baseline priority tree distribution in collection: "
            f"{BASELINE_PRIORITY_COLLECTION} ({len(priority_tree_df)} rows)"
        )
    else:
        log("Skipped baseline priority tree distribution (no qualifying rows or no city_priority layer)")

    run_context = {
        "input_rows": int(len(df)),
        "filtered_rows": int(len(df)),
        "tree_rows": int(len(df)),
        "pole_rows": 0,
        "log_rows": 0,
    }
    tree_summary = tree_results[2] if tree_results is not None and len(tree_results) > 2 else None
    log_path = inst.write_run_log(
        run_context,
        tree_summary,
        None,
        None,
        size_override_summary,
        tree_id_remap_summary,
    )
    if log_path is not None:
        log(f"Wrote baseline instancer run log: {log_path}")

    inst.configure_scenario_view_layer_visibility(scene)
    log(f"Built baseline tree distribution in collection: {BASELINE_TREE_COLLECTION}")


def remove_baseline_buildings() -> None:
    for name in ("city_buildings.001_cubes", "city_buildings.001"):
        if remove_object_if_present(name):
            log(f"Removed baseline carry-over object: {name}")


def iter_collection_tree_objects(collection: bpy.types.Collection):
    for obj in collection.objects:
        yield obj
    for child in collection.children:
        yield from iter_collection_tree_objects(child)


def force_collection_tree_pass_index(collection: bpy.types.Collection, pass_index: int) -> None:
    updated = 0
    for obj in iter_collection_tree_objects(collection):
        if obj.pass_index != pass_index:
            obj.pass_index = pass_index
            updated += 1
    log(
        f"Forced pass_index={pass_index} for collection tree '{collection.name}' "
        f"({updated} objects updated)"
    )


def ensure_geometry_group_material(group_name: str, material_name: str) -> None:
    group = bpy.data.node_groups.get(group_name)
    material = bpy.data.materials.get(material_name)
    if group is None or material is None:
        log(
            f"Skipped ground material repair for group='{group_name}' "
            f"material='{material_name}'"
        )
        return

    updated = 0
    for node in group.nodes:
        if node.bl_idname != "GeometryNodeSetMaterial":
            continue
        if node.inputs["Material"].default_value != material:
            node.inputs["Material"].default_value = material
            updated += 1

    log(
        f"Repaired geometry-node material on '{group_name}' using '{material_name}' "
        f"({updated} Set Material nodes updated)"
    )


def ensure_baseline_ground_material() -> bpy.types.Material:
    material = bpy.data.materials.get("BASELINE_GROUND_WHITE")
    if material is None:
        material = bpy.data.materials.new("BASELINE_GROUND_WHITE")
    material.use_nodes = True
    material.use_fake_user = True
    node_tree = material.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    output = node_tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (240.0, 0.0)
    emission = node_tree.nodes.new("ShaderNodeEmission")
    emission.location = (0.0, 0.0)
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    log("Ensured baseline ground material: BASELINE_GROUND_WHITE")
    return material


def repair_baseline_ground_materials() -> None:
    ensure_baseline_ground_material()
    for group_name, material_name in BASELINE_TERRAIN_GROUP_MATERIALS.items():
        ensure_geometry_group_material(group_name, material_name)


def repair_baseline_resource_material() -> None:
    minimal_resources = load_local_module("minimal_resources_runtime", "MINIMAL_RESOURCES.py")
    minimal_resources.main()
    log("Rebuilt MINIMAL_RESOURCES with baseline resource palette")


def ensure_baseline_terrain_visibility() -> None:
    source_obj = bpy.data.objects.get("city_highResRoad.001")
    if source_obj is not None:
        source_obj.hide_render = True
        source_obj.pass_index = 1

    cube_obj = bpy.data.objects.get("city_highResRoad.001_cubes")
    if cube_obj is not None:
        cube_obj.hide_render = False
        cube_obj.hide_viewport = False
        cube_obj.hide_select = False
        cube_obj.pass_index = 1
        log("Enabled baseline terrain cube render object: city_highResRoad.001_cubes")


def repair_city_clip_groups() -> None:
    clipbox_setup = load_local_module("b2026_clipbox_setup_runtime", "b2026_clipbox_setup.py")
    city_spec = next(spec for spec in clipbox_setup.SITE_SPECS if spec["label"] == "city")
    clip_box = clipbox_setup.ensure_clip_box(city_spec)
    point_group, mesh_group, tree_group = clipbox_setup.ensure_clip_groups(city_spec, clip_box)
    patched_points, patched_meshes, patched_trees = clipbox_setup.patch_scene_groups(
        city_spec,
        point_group,
        mesh_group,
        tree_group,
    )
    log(
        "Repaired city clip groups: "
        f"points={patched_points}, meshes={patched_meshes}, trees={patched_trees}"
    )


def register_camera_clipboxes() -> None:
    camera_clipboxes = load_local_module(
        "b2026_camera_clipboxes_runtime",
        "b2026_camera_clipboxes.py",
    )
    camera_clipboxes.register()
    log("Registered camera clipbox proxies and sync handlers")


def prune_scene_collections(scene: bpy.types.Scene) -> None:
    for collection_name in REMOVE_ROOT_COLLECTIONS:
        remove_collection_tree_everywhere(collection_name)

    for child in list(scene.collection.children):
        if child.name not in KEEP_ROOT_COLLECTIONS:
            scene.collection.children.unlink(child)
            log(f"Pruned root collection from baseline scene: {child.name}")


def retarget_render_layers(scene: bpy.types.Scene) -> None:
    if not scene.use_nodes or scene.node_tree is None:
        return

    muted_file_outputs = 0
    for node in scene.node_tree.nodes:
        if node.bl_idname != "CompositorNodeRLayers":
            if MUTE_FILE_OUTPUTS and node.bl_idname == "CompositorNodeOutputFile":
                node.mute = True
                muted_file_outputs += 1
            continue
        node.scene = scene
        if node.layer not in scene.view_layers:
            node.layer = scene.view_layers[0].name
    log(
        "Retargeted compositor Render Layers nodes to baseline scene"
        + (f" and muted {muted_file_outputs} file outputs" if MUTE_FILE_OUTPUTS else "")
    )


def configure_render(scene: bpy.types.Scene) -> None:
    scene.camera = bpy.data.objects.get("WorldCam") or scene.camera
    scene.render.filepath = str(BASELINE_RENDER_PATH)
    scene.render.image_settings.file_format = "PNG"
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    if hasattr(scene.view_settings, "look"):
        scene.view_settings.look = "None"
    BASELINE_RENDER_PATH.parent.mkdir(parents=True, exist_ok=True)
    log(f"Configured render output: {BASELINE_RENDER_PATH}")


def configure_baseline_exr_outputs(scene: bpy.types.Scene) -> None:
    exr_setup = load_local_module(
        "b2026_setup_view_layer_exr_outputs_runtime",
        "b2026_setup_view_layer_exr_outputs.py",
    )
    output_dir = exr_setup.build_outputs(scene)

    for node in scene.node_tree.nodes:
        if node.bl_idname != "CompositorNodeOutputFile":
            continue
        if not node.name.startswith(exr_setup.OUTPUT_NODE_PREFIX):
            node.mute = True
            continue
        view_layer_name = node.name.removeprefix(exr_setup.OUTPUT_NODE_PREFIX)
        node.mute = view_layer_name not in BASELINE_EXR_VIEW_LAYERS

    log(
        "Configured baseline EXR outputs in "
        f"{output_dir} for {sorted(BASELINE_EXR_VIEW_LAYERS)}"
    )


def enable_baseline_view_layers(scene: bpy.types.Scene) -> None:
    for view_layer in scene.view_layers:
        view_layer.use = view_layer.name not in BASELINE_DISABLED_VIEW_LAYERS

    log(
        "Baseline render view layers enabled="
        f"{sorted(view_layer.name for view_layer in scene.view_layers if view_layer.use)}; "
        f"disabled={sorted(BASELINE_DISABLED_VIEW_LAYERS)}"
    )


def main() -> None:
    baseline_scene = ensure_scene_copy()
    terrain_obj = replace_terrain_mesh()
    repair_baseline_ground_materials()
    rebuild_terrain_world_cubes(terrain_obj)
    repair_baseline_ground_materials()
    build_baseline_tree_distribution(baseline_scene)
    repair_baseline_resource_material()
    repair_city_clip_groups()
    register_camera_clipboxes()
    remove_baseline_buildings()
    ensure_baseline_terrain_visibility()
    prune_scene_collections(baseline_scene)
    retarget_render_layers(baseline_scene)
    configure_baseline_exr_outputs(baseline_scene)
    enable_baseline_view_layers(baseline_scene)
    configure_render(baseline_scene)

    if SAVE_MAINFILE:
        bpy.ops.wm.save_mainfile()
        log(f"Saved blend: {bpy.data.filepath}")

    if RENDER_STILL:
        bpy.ops.render.render(write_still=True, scene=baseline_scene.name)
        log(f"Rendered still: {BASELINE_RENDER_PATH}")


if __name__ == "__main__":
    main()
