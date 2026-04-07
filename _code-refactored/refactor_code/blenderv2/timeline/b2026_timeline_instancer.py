import bpy
import pandas as pd
import numpy as np
import os
import glob
import hashlib
import importlib.util
import sys
import builtins
from datetime import datetime
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_CODE_ROOT = REPO_ROOT / "_code-refactored"
if str(SHARED_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_CODE_ROOT))

from refactor_code.blender.proposal_framebuffers import (
    DEFAULT_OUTPUT_COLUMNS as PROPOSAL_FRAMEBUFFER_OUTPUT_COLUMNS,
    build_blender_proposal_framebuffer_columns,
)


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)

# Constants
#SITE = 'uni'
SITE = 'trimmed-parade'
AUTO_SITE_FROM_SCENE = True
SCENARIO = 'trending'
SCENARIOS_TO_BUILD = ('positive', 'trending')
SCENARIO_VIEW_LAYER_MAP = {
    'pathway_state': {
        'main_roles': ('positive',),
    },
    'priority_state': {
        'main_roles': (),
        'priority_roles': ('priority',),
    },
    'city_priority': {
        'main_roles': (),
        'priority_roles': ('priority',),
    },
    'trending_state': {
        'main_roles': ('trending',),
    },
}
YEAR = 180
TARGET_SCENE_NAME = os.environ.get("B2026_TARGET_SCENE_NAME") or None
PASS_INDEX = 11
DISTANCE_UNITS = 200
USE_3D_CURSOR_FILTER = True
USE_CAMERA_VIEW_FILTER = False
REMAP_TREE_IDS_TO_AVAILABLE_MODELS = True
TREE_ID_REMAP_SEED = 20260320
TRENDING_RANDOM_SMALL_TREES_ENABLED = True
TRENDING_RANDOM_SMALL_TREES_PROPORTION = 30
TRENDING_RANDOM_SMALL_TREES_SEED = 20260320
WRITE_RUN_LOG = True
RUN_LOG_DIR = os.environ.get(
    "B2026_RUN_LOG_DIR",
    r"D:\2026 Arboreal Futures\data\blender\2026\logs",
)
POLE_BYPASS = True
POLE_FALLBACK_PLY = 'artificial_precolonial.False_size.snag_control.improved-tree_id.10.ply'
HIDE_IMPORTED_MODEL_OBJECTS = True
FALLEN_LOG_INT_RESOURCE = 6
AUTO_RUN_CLIPBOX_SETUP = False
AUTO_RUN_CAMERA_CLIPBOXES = False
FOCUS_NEW_POINT_CLOUD_OBJECT = False
PRIORITY_TREE_SIZES = ('senescing', 'snag', 'fallen')
IGNORED_SIZE_NAMES = ('decayed', 'other')
TIMELINE_MODE = True
TIMELINE_ACTIVE_YEARS = (0, 10, 30, 60, 180)
MERGED_TIMELINE_CSV_OUTPUT_DIR = Path(r'D:\2026 Arboreal Futures\data\debug\timeline_csv')
SCRIPT_DIRECTORY_FALLBACK = Path(__file__).resolve().parent
RESOURCE_BINARY_ATTRIBUTE_NAMES = (
    'resource_hollow',
    'resource_epiphyte',
    'resource_dead branch',
    'resource_perch branch',
    'resource_peeling bark',
    'resource_fallen log',
    'resource_other',
)
PROPOSAL_FRAMEBUFFER_ATTRIBUTE_NAMES = tuple(PROPOSAL_FRAMEBUFFER_OUTPUT_COLUMNS.values())
# Paths
PLY_FOLDER = str(REPO_ROOT / "_data-refactored" / "model-inputs" / "tree_library_exports" / "treeMeshesPly")
LOG_FOLDER = str(REPO_ROOT / "_data-refactored" / "model-inputs" / "tree_library_exports" / "logMeshesPly")
BASE_PATH = str(REPO_ROOT / "_data-refactored" / "v3engine_outputs" / "feature-locations" / SITE)

#if SITE == 'trimmed-parade':
#    BASE_PATH = f'/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/{SITE}/initial'


CSV_FILENAME = f'{SITE}_{SCENARIO}_1_nodeDF_{YEAR}.csv'
CSV_FILEPATH = os.path.join(BASE_PATH, CSV_FILENAME)

print(f'CSV_FILEPATH IS: {CSV_FILEPATH}')


# Helper Functions
def infer_site_from_scene_name(scene_name):
    scene_contract = get_scene_contract_module()
    return scene_contract.infer_site_from_scene_name(scene_name)


def get_build_mode(site_name=None):
    timeline_layout = get_timeline_layout_module()
    target_site = site_name or SITE
    if TIMELINE_MODE and timeline_layout.timeline_mode_supported(target_site):
        return timeline_layout.get_build_mode()
    return "timeline"


def single_state_mode_active(site_name=None):
    return get_build_mode(site_name) == "single_state"


def get_active_build_years(site_name=None):
    timeline_layout = get_timeline_layout_module()
    target_site = site_name or SITE
    if TIMELINE_MODE and timeline_layout.timeline_mode_supported(target_site):
        return tuple(timeline_layout.get_active_years(target_site))
    return (YEAR,)


def get_primary_build_year(site_name=None):
    return int(get_active_build_years(site_name)[-1])


def get_csv_filename_for_scenario(scenario_name):
    return f'{SITE}_{scenario_name}_1_nodeDF_{get_primary_build_year()}.csv'


def get_csv_filepath_for_scenario(scenario_name):
    if TIMELINE_MODE and get_timeline_layout_module().timeline_mode_supported(SITE):
        timeline_layout = get_timeline_layout_module()
        return str(
            timeline_layout.resolve_feature_csv_path(
                SITE,
                scenario_name,
                get_primary_build_year(),
            )
        )
    return os.path.join(BASE_PATH, get_csv_filename_for_scenario(scenario_name))


def refresh_site_paths():
    global PLY_FOLDER, LOG_FOLDER, BASE_PATH, CSV_FILENAME, CSV_FILEPATH
    timeline_layout = get_timeline_layout_module()
    PLY_FOLDER = str(timeline_layout.resolve_tree_ply_folder())
    LOG_FOLDER = str(timeline_layout.resolve_log_ply_folder())
    BASE_PATH = str(REPO_ROOT / "_data-refactored" / "v3engine_outputs" / "feature-locations" / SITE)
    CSV_FILENAME = get_csv_filename_for_scenario(SCENARIO)
    CSV_FILEPATH = get_csv_filepath_for_scenario(SCENARIO)


def configure_site_from_scene(scene):
    global SITE
    if not AUTO_SITE_FROM_SCENE:
        refresh_site_paths()
        return

    inferred_site = infer_site_from_scene_name(getattr(scene, "name", ""))
    if inferred_site:
        SITE = inferred_site

    refresh_site_paths()


def get_available_scenarios():
    if TIMELINE_MODE and get_timeline_layout_module().timeline_mode_supported(SITE):
        timeline_layout = get_timeline_layout_module()
        available = []
        for scenario_name in SCENARIOS_TO_BUILD:
            try:
                timeline_layout.build_site_dataframe(
                    SITE,
                    scenario_name,
                    get_active_build_years(SITE),
                )
            except FileNotFoundError:
                continue
            available.append(scenario_name)
        if available:
            return tuple(available)
        return (SCENARIO,)

    available = tuple(
        scenario_name
        for scenario_name in SCENARIOS_TO_BUILD
        if os.path.exists(get_csv_filepath_for_scenario(scenario_name))
    )
    if available:
        return available
    return (SCENARIO,)


def get_run_id(scenario_name=None):
    scenario_name = scenario_name or SCENARIO
    if single_state_mode_active():
        return f"{SITE}_yr{get_primary_build_year()}_{scenario_name}"
    if site_uses_timeline_mode():
        return f"{SITE}_timeline_{scenario_name}"
    return f"{SITE}_{YEAR}_{scenario_name}"


def get_run_collection_name(scenario_name=None):
    if single_state_mode_active():
        scene_contract = get_scene_contract_module()
        scenario_name = scenario_name or SCENARIO
        role = {
            'positive': 'positive',
            'trending': 'trending',
        }.get(scenario_name, 'positive')
        return scene_contract.get_single_state_top_level_name(SITE, role)
    return f"Year_{get_run_id(scenario_name)}"


def get_priority_run_collection_name(scenario_name=None):
    if single_state_mode_active():
        scene_contract = get_scene_contract_module()
        return scene_contract.get_single_state_top_level_name(SITE, 'priority')
    return f"{get_run_collection_name(scenario_name)}_priority"


def get_run_parent_collection_name(scenario_name=None):
    scene_contract = get_scene_contract_module()
    scenario_name = scenario_name or SCENARIO
    if single_state_mode_active():
        role = {
            'positive': 'positive',
            'trending': 'trending',
        }.get(scenario_name, 'positive')
        return scene_contract.get_single_state_top_level_name(SITE, role)
    role = {
        'positive': 'positive',
        'trending': 'trending',
    }.get(scenario_name, 'positive')
    return scene_contract.get_collection_name(SITE, role)


def get_priority_parent_collection_name():
    scene_contract = get_scene_contract_module()
    if single_state_mode_active():
        return scene_contract.get_single_state_top_level_name(SITE, 'priority')
    return scene_contract.get_collection_name(SITE, 'priority')


def ensure_child_collection(parent_collection, child_collection):
    if parent_collection.children.get(child_collection.name) is None:
        parent_collection.children.link(child_collection)


def ensure_single_state_collections_linked(scene):
    if not single_state_mode_active():
        return

    scene_contract = get_scene_contract_module()
    for role in ('manager', 'setup', 'cameras', 'positive', 'priority', 'trending'):
        ensure_run_parent_collection(
            scene,
            scene_contract.get_single_state_top_level_name(SITE, role),
        )


def ensure_run_parent_collection(scene, collection_name):
    parent = bpy.data.collections.get(collection_name)
    if parent is None:
        parent = bpy.data.collections.new(collection_name)
    if scene.collection.children.get(parent.name) is None:
        scene.collection.children.link(parent)
    return parent


def find_layer_collection(layer_collection, collection_name):
    if layer_collection.collection.name == collection_name:
        return layer_collection
    for child in layer_collection.children:
        found = find_layer_collection(child, collection_name)
        if found is not None:
            return found
    return None


def set_collection_exclude(view_layer, collection_name, excluded):
    layer_collection = find_layer_collection(view_layer.layer_collection, collection_name)
    if layer_collection is None:
        return False
    layer_collection.exclude = excluded
    return True


def configure_priority_view_layer_visibility(scene, priority_scenario_name=None):
    priority_layers = [
        scene.view_layers.get(layer_name)
        for layer_name in ('priority_state', 'city_priority')
        if scene.view_layers.get(layer_name) is not None
    ]
    if not priority_layers:
        return False

    for priority_view_layer in priority_layers:
        set_collection_exclude(
            priority_view_layer,
            get_priority_parent_collection_name(),
            False,
        )
        for scenario_name in ('positive', 'trending'):
            set_collection_exclude(
                priority_view_layer,
                get_run_parent_collection_name(scenario_name),
                True,
            )

    print("Configured priority view layers to isolate the priority collection")
    return True


def configure_scenario_view_layer_visibility(scene):
    available_scenarios = get_available_scenarios()
    configured = False
    scene_contract = get_scene_contract_module()
    site_contract = scene_contract.SITE_CONTRACTS.get(SITE, {})
    legacy = site_contract.get('legacy', {})
    timeline_positive_bio = f"Year_{SITE}_timeline_bioenvelope_positive"
    timeline_trending_bio = f"Year_{SITE}_timeline_bioenvelope_trending"
    for view_layer in scene.view_layers:
        visibility_spec = SCENARIO_VIEW_LAYER_MAP.get(
            view_layer.name,
            {'main_roles': (), 'priority_roles': ()},
        )
        main_visible = set(visibility_spec.get('main_roles', ()))
        priority_visible = set(visibility_spec.get('priority_roles', ()))
        if view_layer.name in SCENARIO_VIEW_LAYER_MAP:
            configured = True
        for candidate in ('positive', 'trending'):
            set_collection_exclude(
                view_layer,
                get_run_parent_collection_name(candidate),
                candidate not in main_visible,
            )
        set_collection_exclude(
            view_layer,
            get_priority_parent_collection_name(),
            'priority' not in priority_visible,
        )
        if single_state_mode_active():
            for role in ('manager', 'setup', 'cameras', 'positive', 'priority', 'trending'):
                collection_name = scene_contract.get_single_state_top_level_name(SITE, role)
                set_collection_exclude(view_layer, collection_name, False)
            for collection_name in (
                legacy.get('timeline_base'),
                legacy.get('timeline_positive'),
                legacy.get('timeline_priority'),
                legacy.get('timeline_trending'),
                timeline_positive_bio,
                timeline_trending_bio,
            ):
                if collection_name:
                    set_collection_exclude(view_layer, collection_name, True)
            for collection_name in (
                legacy.get('base'),
                legacy.get('base_cubes'),
                legacy.get('bio_positive'),
                legacy.get('bio_trending'),
            ):
                if collection_name:
                    set_collection_exclude(view_layer, collection_name, True)

    if configured:
        print(
            "Configured scenario visibility:"
            " pathway_state -> positive,"
            " priority_state/city_priority -> priority,"
            " trending_state -> trending"
        )
    return configured


def get_script_directory():
    file_path = globals().get("__file__")
    if file_path:
        return Path(file_path).resolve().parent
    return SCRIPT_DIRECTORY_FALLBACK


def load_local_script_module(module_name, filename):
    script_path = get_script_directory() / filename
    if not script_path.exists():
        raise FileNotFoundError(f"Required follow-up script was not found: {script_path}")

    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        existing_path = Path(getattr(existing_module, "__file__", "")).resolve()
        if existing_path == script_path.resolve():
            return existing_module, script_path
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, script_path


def get_timeline_layout_module():
    module, _script_path = load_local_script_module(
        "b2026_timeline_layout",
        "b2026_timeline_layout.py",
    )
    return module


def get_scene_contract_module():
    module, _script_path = load_local_script_module(
        "b2026_timeline_scene_contract",
        "b2026_timeline_scene_contract.py",
    )
    return module


def site_uses_timeline_mode(site_name=None):
    timeline_layout = get_timeline_layout_module()
    target_site = site_name or SITE
    return (
        TIMELINE_MODE
        and timeline_layout.timeline_mode_supported(target_site)
        and not single_state_mode_active(target_site)
    )


def run_clipbox_followup_scripts():
    if not AUTO_RUN_CLIPBOX_SETUP and not AUTO_RUN_CAMERA_CLIPBOXES:
        return

    print("\nRunning clipbox follow-up scripts...")

    if AUTO_RUN_CLIPBOX_SETUP:
        clipbox_module, clipbox_path = load_local_script_module(
            "b2026_timeline_clipbox_setup",
            "b2026_timeline_clipbox_setup.py",
        )
        if not hasattr(clipbox_module, "main"):
            raise AttributeError(f"{clipbox_path} does not expose main()")
        clipbox_module.main()
        print(f"Ran clipbox setup: {clipbox_path}")

    if AUTO_RUN_CAMERA_CLIPBOXES:
        camera_module, camera_path = load_local_script_module(
            "b2026_timeline_camera_clipboxes",
            "b2026_timeline_camera_clipboxes.py",
        )
        if not hasattr(camera_module, "register"):
            raise AttributeError(f"{camera_path} does not expose register()")
        camera_module.register()
        if hasattr(camera_module, "sync_all_scene_clipboxes"):
            camera_module.sync_all_scene_clipboxes()
        print(f"Ran camera clipboxes: {camera_path}")


def unlink_collection(collection):
    for scene in bpy.data.scenes:
        if scene.collection.children.get(collection.name):
            scene.collection.children.unlink(collection)

    for parent in bpy.data.collections:
        if parent.children.get(collection.name):
            parent.children.unlink(collection)


def delete_collection_tree(collection):
    for child in list(collection.children):
        delete_collection_tree(child)

    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    unlink_collection(collection)
    bpy.data.collections.remove(collection)


def iter_collection_objects_recursive(collection):
    for obj in list(collection.objects):
        yield obj
    for child in list(collection.children):
        yield from iter_collection_objects_recursive(child)


def require_target_scene():
    if TARGET_SCENE_NAME:
        scene = bpy.data.scenes.get(TARGET_SCENE_NAME)
        if scene is None:
            raise ValueError(f"Scene '{TARGET_SCENE_NAME}' was not found.")
        return scene

    scene = bpy.context.scene
    if scene is None:
        raise ValueError("No active scene found")
    return scene


def remove_node_group_if_present(name):
    node_group = bpy.data.node_groups.get(name)
    if node_group:
        bpy.data.node_groups.remove(node_group)


def get_node_run_id(node_type, variant_suffix=None, scenario_name=None):
    if single_state_mode_active():
        scenario_name = scenario_name or SCENARIO
        priority = variant_suffix == "priority"
        return get_scene_contract_module().get_single_state_node_collection_base(
            SITE,
            node_type,
            get_primary_build_year(),
            scenario_name,
            priority=priority,
        )
    base = f"{node_type}_{get_run_id(scenario_name)}"
    if variant_suffix:
        return f"{base}_{variant_suffix}"
    return base


def get_point_cloud_object_name(node_type, variant_suffix=None, scenario_name=None):
    if single_state_mode_active():
        scenario_name = scenario_name or SCENARIO
        suffix = f"{scenario_name}_priority" if variant_suffix == "priority" else scenario_name
        base = f"{node_type.capitalize()}Positions_{SITE}_yr{get_primary_build_year()}_{suffix}"
        return base
    base = f"{node_type.capitalize()}Positions_{get_run_id(scenario_name)}"
    if variant_suffix:
        return f"{base}_{variant_suffix}"
    return base


def get_import_target_name(instance_index, filename_stem, variant_suffix=None, scenario_name=None):
    base = f"instanceID.{instance_index}_{get_run_id(scenario_name)}"
    if variant_suffix:
        base = f"{base}_{variant_suffix}"
    return f"{base}_{filename_stem}"


def focus_object_if_visible_in_active_view_layer(obj):
    if not FOCUS_NEW_POINT_CLOUD_OBJECT:
        return

    view_layer = bpy.context.view_layer
    if view_layer is None:
        return

    if view_layer.objects.get(obj.name) is None:
        return

    try:
        bpy.ops.object.select_all(action='DESELECT')
    except RuntimeError:
        return

    try:
        obj.select_set(True)
        view_layer.objects.active = obj
    except RuntimeError:
        return


def ensure_geometry_nodes_modifier(obj, modifier_name, node_group):
    geo_nodes = obj.modifiers.get(modifier_name)
    if geo_nodes and geo_nodes.type != 'NODES':
        obj.modifiers.remove(geo_nodes)
        geo_nodes = None

    if geo_nodes is None:
        geo_nodes = obj.modifiers.new(name=modifier_name, type='NODES')

    geo_nodes.node_group = node_group

    focus_object_if_visible_in_active_view_layer(obj)
    return geo_nodes


def configure_geometry_node_group(node_group, obj):
    if hasattr(node_group, "is_modifier"):
        node_group.is_modifier = True
    if hasattr(node_group, "is_tool"):
        node_group.is_tool = False
    if hasattr(node_group, "is_mode_object"):
        node_group.is_mode_object = True
    if obj.type == 'MESH' and hasattr(node_group, "is_type_mesh"):
        node_group.is_type_mesh = True


def configure_imported_model_visibility(obj):
    if not HIDE_IMPORTED_MODEL_OBJECTS:
        return

    obj.hide_viewport = True
    obj.hide_render = True
    obj.hide_select = True
    obj.display_type = 'BOUNDS'


def remove_object_if_present(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        return False
    bpy.data.objects.remove(obj, do_unlink=True)
    return True


def remove_collection_if_present(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        return False
    delete_collection_tree(collection)
    return True


def collection_contains_site_objects(collection):
    site_token = SITE.lower()
    run_token = get_run_id().lower()
    for obj in iter_collection_objects_recursive(collection):
        name = obj.name.lower()
        if site_token in name or run_token in name:
            return True
    return False


def cleanup_existing_run_artifacts():
    node_types = ("tree", "pole", "log")
    scenario_names = tuple(dict.fromkeys((*SCENARIOS_TO_BUILD, SCENARIO)))

    collection_names = set()
    object_names = set()
    node_group_names = set()

    for scenario_name in scenario_names:
        run_id = get_run_id(scenario_name)
        if not single_state_mode_active():
            collection_names.add(get_run_collection_name(scenario_name))
            collection_names.add(get_priority_run_collection_name(scenario_name))
        node_group_names.add(run_id)

        for node_type in node_types:
            node_run_id = get_node_run_id(node_type, scenario_name=scenario_name)
            collection_names.add(f"{node_run_id}_positions")
            collection_names.add(f"{node_run_id}_plyModels")
            object_names.add(get_point_cloud_object_name(node_type, scenario_name=scenario_name))
            node_group_names.add(node_run_id)

            legacy_collection_prefix = f"{node_type}_{get_primary_build_year()}_{scenario_name}"
            for suffix in ("_positions", "_plyModels"):
                legacy_name = f"{legacy_collection_prefix}{suffix}"
                legacy_collection = bpy.data.collections.get(legacy_name)
                if legacy_collection and collection_contains_site_objects(legacy_collection):
                    collection_names.add(legacy_name)

        for node_type in ("tree", "log"):
            priority_run_id = get_node_run_id(node_type, "priority", scenario_name=scenario_name)
            collection_names.add(f"{priority_run_id}_positions")
            collection_names.add(f"{priority_run_id}_plyModels")
            object_names.add(get_point_cloud_object_name(node_type, "priority", scenario_name=scenario_name))
            node_group_names.add(priority_run_id)

    for object_name in sorted(object_names):
        if remove_object_if_present(object_name):
            print(f"Removed existing object: {object_name}")

    for collection_name in sorted(collection_names):
        if remove_collection_if_present(collection_name):
            print(f"Removed existing collection tree: {collection_name}")

    for node_group_name in sorted(node_group_names):
        if bpy.data.node_groups.get(node_group_name):
            remove_node_group_if_present(node_group_name)
            print(f"Removed existing node group: {node_group_name}")


def cleanup_scene():
    """Clean up any existing data to ensure fresh start."""
    cleanup_existing_run_artifacts()

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def convert_control(value):
    control_map = {
        'street-tree': 1,
        'park-tree': 2,
        'reserve-tree': 3,
        'improved-tree': 4,
    }
    return pd.Series(value).str.lower().map(control_map).fillna(-1)


def convert_precolonial(value):
    normalized = pd.Series(value).fillna('').astype(str).str.strip().str.lower()
    precolonial_map = {
        'false': 1,
        '0': 1,
        'no': 1,
        'true': 2,
        '1': 2,
        'yes': 2,
    }
    return normalized.map(precolonial_map).fillna(-1)


def convert_size(value):
    size_map = {
        'small': 1,
        'medium': 2,
        'large': 3,
        'senescing': 4,
        'snag': 5,
        'fallen': 6,
    }
    return pd.Series(value).str.lower().map(size_map).fillna(-1)


def convert_node_type(value):
    node_type_map = {
        'tree': 0,
        'pole': 1,
        'log': 2,
    }
    return pd.Series(value).astype(str).str.lower().map(node_type_map).fillna(-1)


def convert_tree_interventions(value):
    normalized = pd.Series(value).fillna('').astype(str).str.strip().str.lower()
    intervention_map = {
        'none': 0,
        'paved': 0,
        'exoskeleton': 1,
        '': 2,
        'footprint-depaved': 2,
        'node-rewilded': 3,
        'rewilded': 3,
    }
    return normalized.map(intervention_map).fillna(-1)


def convert_tree_proposals(value):
    proposal_map = {
        'age-in-place': 2,
        'senescent': 1,
        'replace': 0,
    }
    normalized = pd.Series(value).fillna('').astype(str).str.strip().str.upper()
    mapped = normalized.str.lower().map(proposal_map)
    # Preserve missing/blank as -1 so they are not conflated with REPLACE.
    mapped[normalized.eq('')] = -1
    return mapped.fillna(-1)


def convert_improvement(value):
    normalized = pd.Series(value).fillna('').astype(str).str.strip().str.lower()
    truthy = {'true', 'yes', '1', 'y'}
    falsy = {'false', 'no', '0', 'n', ''}
    result = pd.Series(np.full(len(normalized), -1, dtype=np.int32), index=normalized.index)
    result[normalized.isin(truthy)] = 1
    result[normalized.isin(falsy)] = 0
    return result.fillna(-1)


def get_series_or_default(df, column_name, default_value):
    if column_name in df.columns:
        return df[column_name]
    return pd.Series([default_value] * len(df), index=df.index)


def stable_hash_index(parts, modulo):
    if modulo <= 0:
        return 0
    payload = "|".join(str(part) for part in parts).encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], 'big') % modulo


def choose_stable_match_with_reuse_avoidance(matches, needed, usage_key, used_filenames_by_bucket):
    if len(matches) == 0:
        return None, 0, False

    base_idx = stable_hash_index(
        (
            needed.get('precolonial', ''),
            needed.get('size', ''),
            needed.get('control', ''),
            needed.get('id', 0),
        ),
        len(matches),
    )

    if usage_key is None:
        return matches.iloc[base_idx], base_idx, False

    used_filenames = used_filenames_by_bucket.setdefault(usage_key, set())
    filenames = matches['filename'].tolist()

    for offset in range(len(matches)):
        idx = (base_idx + offset) % len(matches)
        filename = filenames[idx]
        if filename not in used_filenames:
            used_filenames.add(filename)
            return matches.iloc[idx], idx, offset != 0

    return matches.iloc[base_idx], base_idx, False


def normalize_small_tree_proportion(value):
    if value <= 0:
        return 0.0
    if value > 1.0:
        return min(value / 100.0, 1.0)
    return float(value)


def build_available_tree_id_map(ply_folder):
    tree_id_map = {}
    files = [
        filename
        for filename in os.listdir(ply_folder)
        if filename.endswith('.ply') and not filename.startswith('artificial_')
    ]

    for filename in files:
        stem = filename[:-4]
        try:
            precolonial_part, size_part, _control_part, id_part = stem.split('_')
        except ValueError:
            continue

        precolonial = precolonial_part.split('.')[1]
        size = size_part.split('.')[1].lower()
        tree_id = int(id_part.split('.')[1])
        tree_id_map.setdefault((precolonial, size), set()).add(tree_id)

    return {
        key: tuple(sorted(values))
        for key, values in tree_id_map.items()
    }


def remap_tree_ids_to_available_models(df, ply_folder):
    summary = {
        'enabled': bool(REMAP_TREE_IDS_TO_AVAILABLE_MODELS),
        'seed': TREE_ID_REMAP_SEED,
        'already_valid_count': 0,
        'remapped_count': 0,
        'missing_map_count': 0,
    }

    if not REMAP_TREE_IDS_TO_AVAILABLE_MODELS:
        return df, summary

    available_tree_ids = build_available_tree_id_map(ply_folder)
    result = df.copy()
    remapped_ids = []

    for row in result.itertuples():
        if getattr(row, 'nodeType', 'tree') != 'tree':
            remapped_ids.append(getattr(row, 'tree_id', -1))
            continue

        precolonial = str(row.precolonial)
        size = str(row.size).lower()
        available_ids = available_tree_ids.get((precolonial, size))

        try:
            original_id = int(row.tree_id)
        except (TypeError, ValueError):
            original_id = -1

        if not available_ids:
            remapped_ids.append(original_id)
            summary['missing_map_count'] += 1
            continue

        if original_id in available_ids:
            remapped_ids.append(original_id)
            summary['already_valid_count'] += 1
            continue

        chosen_id = available_ids[
            stable_hash_index(
                (
                    TREE_ID_REMAP_SEED,
                    getattr(row, 'structureID', row.Index),
                    getattr(row, 'x', 0.0),
                    getattr(row, 'y', 0.0),
                    precolonial,
                    size,
                    original_id,
                ),
                len(available_ids),
            )
        ]
        remapped_ids.append(chosen_id)
        summary['remapped_count'] += 1

    result['tree_id'] = np.array(remapped_ids, dtype=np.int32)

    print(
        "\nTree ID remap:"
        f" seed={TREE_ID_REMAP_SEED},"
        f" already_valid={summary['already_valid_count']},"
        f" remapped={summary['remapped_count']},"
        f" missing_map={summary['missing_map_count']}"
    )
    return result, summary


def select_spread_indices(df, candidate_indices, count, rng):
    if count <= 0:
        return np.array([], dtype=candidate_indices.dtype)

    if count >= len(candidate_indices):
        return candidate_indices.copy()

    coords = df.loc[candidate_indices, ['x', 'y']].to_numpy(dtype=np.float32)
    selected_positions = []
    selected_mask = np.zeros(len(candidate_indices), dtype=bool)

    first_position = int(rng.integers(len(candidate_indices)))
    selected_positions.append(first_position)
    selected_mask[first_position] = True

    min_dist_sq = np.sum((coords - coords[first_position]) ** 2, axis=1)
    min_dist_sq[selected_mask] = -1.0

    while len(selected_positions) < count:
        remaining_positions = np.flatnonzero(~selected_mask)
        if len(remaining_positions) == 0:
            break

        pool_size = min(len(remaining_positions), max(16, min(64, count * 4)))
        candidate_pool = rng.choice(remaining_positions, size=pool_size, replace=False)
        best_position = int(candidate_pool[np.argmax(min_dist_sq[candidate_pool])])

        selected_positions.append(best_position)
        selected_mask[best_position] = True

        dist_sq = np.sum((coords - coords[best_position]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        min_dist_sq[selected_mask] = -1.0

    return candidate_indices[np.array(selected_positions, dtype=np.int32)]


def randomize_trending_tree_sizes(df):
    requested_proportion = normalize_small_tree_proportion(TRENDING_RANDOM_SMALL_TREES_PROPORTION)
    summary = {
        'enabled': bool(TRENDING_RANDOM_SMALL_TREES_ENABLED),
        'seed': TRENDING_RANDOM_SMALL_TREES_SEED,
        'requested_proportion': float(TRENDING_RANDOM_SMALL_TREES_PROPORTION),
        'normalized_proportion': requested_proportion,
        'converted_count': 0,
        'converted_large': 0,
        'converted_medium': 0,
        'candidate_count': 0,
        'selection_method': 'spatially_balanced_random',
    }

    if SCENARIO != 'trending' or not TRENDING_RANDOM_SMALL_TREES_ENABLED:
        return df, summary

    if requested_proportion <= 0:
        print("\nTrending size override enabled, but proportion is 0; skipping")
        return df, summary

    candidate_mask = (
        df['nodeType'].eq('tree') &
        df['size'].astype(str).str.lower().isin(['large', 'medium'])
    )
    candidate_indices = df.index[candidate_mask].to_numpy()
    summary['candidate_count'] = int(len(candidate_indices))

    if len(candidate_indices) == 0:
        print("\nTrending size override found no medium/large tree candidates")
        return df, summary

    count = min(int(round(len(candidate_indices) * requested_proportion)), len(candidate_indices))
    if count <= 0:
        print("\nTrending size override rounded down to 0 selected trees; skipping")
        return df, summary

    rng = np.random.default_rng(TRENDING_RANDOM_SMALL_TREES_SEED)
    selected_indices = select_spread_indices(df, candidate_indices, count, rng)

    result = df.copy()
    original_sizes = result.loc[selected_indices, 'size'].astype(str).str.lower()
    converted_large = int((original_sizes == 'large').sum())
    converted_medium = int((original_sizes == 'medium').sum())
    result.loc[selected_indices, 'size'] = 'small'
    summary['converted_count'] = count
    summary['converted_large'] = converted_large
    summary['converted_medium'] = converted_medium

    print(
        "\nTrending size override:"
        f" seed={TRENDING_RANDOM_SMALL_TREES_SEED},"
        f" proportion={requested_proportion:.4f},"
        f" converted {count} trees to small"
        f" ({converted_medium} medium, {converted_large} large)"
        " using a spatially balanced random sample"
    )
    return result, summary


def build_collection_summary(df, unique_filenames, node_objects, node_type, resource_binary_fallback_models=None):
    resolved_model_counts = (
        df.groupby('resolved_filename', dropna=False)
        .size()
        .reset_index(name='count')
        .sort_values(['count', 'resolved_filename'], ascending=[False, True])
        .reset_index(drop=True)
    )

    summary = {
        'node_type': node_type,
        'total_instances': int(len(df)),
        'unique_requested_templates': int(len(unique_filenames)),
        'unique_resolved_models': int(df['resolved_filename'].nunique(dropna=True)),
        'imported_model_objects': int(len(node_objects)),
        'resolved_model_counts': resolved_model_counts,
        'resource_binary_fallback_models': sorted(set(resource_binary_fallback_models or [])),
    }

    if node_type == 'tree':
        tree_breakdown = (
            df.groupby(['size', 'precolonial', 'tree_id', 'resolved_filename'], dropna=False)
            .size()
            .reset_index(name='count')
            .sort_values(
                ['count', 'size', 'precolonial', 'tree_id', 'resolved_filename'],
                ascending=[False, True, True, True, True],
            )
            .reset_index(drop=True)
        )
        summary['tree_breakdown'] = tree_breakdown
        preview_columns = [
            column_name
            for column_name in ['x', 'y', 'z', 'structureID', 'tree_id', 'size', 'control', 'resolved_filename']
            if column_name in df.columns
        ]
        summary['requested_position_preview'] = df[preview_columns].head(50).reset_index(drop=True)

    return summary


def format_dataframe_for_log(df):
    if df is None or df.empty:
        return "(none)"
    return df.to_string(index=False)


def get_filter_mode_name():
    if site_uses_timeline_mode():
        return 'timeline_full_strip_layout'
    if USE_CAMERA_VIEW_FILTER:
        return 'camera_view'
    if USE_3D_CURSOR_FILTER:
        return '3d_cursor_square'
    return 'camera_square'


def extend_log_with_collection_summary(lines, summary):
    node_type = summary['node_type']
    lines.extend([
        f"{node_type}_generation_summary:",
        f"  total_instances: {summary['total_instances']}",
        f"  unique_requested_templates: {summary['unique_requested_templates']}",
        f"  unique_resolved_models: {summary['unique_resolved_models']}",
        f"  imported_model_objects: {summary['imported_model_objects']}",
    ])

    if node_type == 'tree':
        lines.extend([
            f"  resource_binary_fallback_model_count: {len(summary.get('resource_binary_fallback_models', []))}",
            "",
            "tree_breakdown_by_size_precolonial_tree_id_model:",
            format_dataframe_for_log(summary.get('tree_breakdown')),
            "",
            "tree_counts_by_resolved_model:",
            format_dataframe_for_log(summary.get('resolved_model_counts')),
            "",
            "tree_requested_positions_preview_first_50:",
            format_dataframe_for_log(summary.get('requested_position_preview')),
            "",
            "tree_point_cloud_vertices_preview_first_50:",
            format_dataframe_for_log(summary.get('point_cloud_vertex_preview')),
            "",
            "tree_resource_binary_fallback_models:",
            "\n".join(summary.get('resource_binary_fallback_models', [])) or "(none)",
        ])
        return

    lines.extend([
        "",
        f"{node_type}_counts_by_resolved_model:",
        format_dataframe_for_log(summary.get('resolved_model_counts')),
    ])


def write_run_log(run_context, tree_summary, pole_summary, log_summary, size_override_summary, tree_id_remap_summary):
    if not WRITE_RUN_LOG:
        return None

    log_dir = Path(RUN_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone()
    timestamp_slug = timestamp.strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f"instancer_{get_run_id()}_{timestamp_slug}.log"

    lines = [
        f"timestamp: {timestamp.isoformat()}",
        f"run_id: {get_run_id()}",
        f"site: {SITE}",
        f"scenario: {SCENARIO}",
        f"year: {YEAR}",
        f"timeline_mode: {site_uses_timeline_mode()}",
        f"timeline_years: {run_context.get('timeline_years', [])}",
        f"csv_filepath: {CSV_FILEPATH}",
        f"source_csv_paths: {run_context.get('source_csv_paths', [])}",
        f"filter_mode: {get_filter_mode_name()}",
        f"distance_units: {DISTANCE_UNITS}",
        f"input_rows: {run_context['input_rows']}",
        f"filtered_rows: {run_context['filtered_rows']}",
        f"tree_rows: {run_context['tree_rows']}",
        f"pole_rows: {run_context['pole_rows']}",
        f"log_rows: {run_context['log_rows']}",
        "",
        "tree_id_remap:",
        f"  enabled: {tree_id_remap_summary['enabled']}",
        f"  seed: {tree_id_remap_summary['seed']}",
        f"  already_valid_count: {tree_id_remap_summary['already_valid_count']}",
        f"  remapped_count: {tree_id_remap_summary['remapped_count']}",
        f"  missing_map_count: {tree_id_remap_summary['missing_map_count']}",
        "",
        "trending_random_small_trees:",
        f"  enabled: {size_override_summary['enabled']}",
        f"  requested_proportion: {size_override_summary['requested_proportion']}",
        f"  normalized_proportion: {size_override_summary['normalized_proportion']}",
        f"  seed: {size_override_summary['seed']}",
        f"  candidate_count: {size_override_summary['candidate_count']}",
        f"  converted_count: {size_override_summary['converted_count']}",
        f"  converted_medium: {size_override_summary['converted_medium']}",
        f"  converted_large: {size_override_summary['converted_large']}",
        f"  selection_method: {size_override_summary['selection_method']}",
        "",
    ]

    if tree_summary is None:
        lines.extend([
            "tree_generation_summary:",
            "  (no trees generated)",
        ])
    else:
        extend_log_with_collection_summary(lines, tree_summary)

    lines.append("")
    if pole_summary is None:
        lines.extend([
            "pole_generation_summary:",
            "  (no poles generated)",
        ])
    else:
        extend_log_with_collection_summary(lines, pole_summary)

    lines.append("")
    if log_summary is None:
        lines.extend([
            "log_generation_summary:",
            "  (no logs generated)",
        ])
    else:
        extend_log_with_collection_summary(lines, log_summary)

    log_path.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f"\nRun log written to: {log_path}")
    return log_path


def save_merged_tree_log_csv(df, scenario_name):
    if not site_uses_timeline_mode():
        return None

    merged = df[df['nodeType'].isin(['tree', 'log'])].copy()
    if merged.empty:
        return None

    MERGED_TIMELINE_CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MERGED_TIMELINE_CSV_OUTPUT_DIR / f"{SITE}_{scenario_name}_merged_tree_log_timeline.csv"
    merged.to_csv(output_path, index=False)
    print(f"Saved merged tree/log timeline CSV: {output_path}")
    return output_path


def find_camera_by_pass_index(pass_index):
    """Find the first camera with the specified pass index."""
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA' and obj.pass_index == pass_index:
            return obj
    raise ValueError(f"No object found with pass index {pass_index}")


def filter_by_camera_view(df, scene, camera_obj):
    mask = []
    depths = []

    for x, y, z in df[['x', 'y', 'z']].to_numpy():
        camera_view = world_to_camera_view(scene, camera_obj, Vector((x, y, z)))
        is_visible = (
            0.0 <= camera_view.x <= 1.0 and
            0.0 <= camera_view.y <= 1.0 and
            camera_view.z >= 0.0
        )
        mask.append(is_visible)
        depths.append(camera_view.z)

    visible_mask = np.array(mask, dtype=bool)
    result = df[visible_mask].copy()
    result['camera_view_depth'] = np.array(depths, dtype=np.float32)[visible_mask]
    return result


def parse_ply_filenames(filenames, node_type='tree'):
    """Vectorized parsing of multiple PLY filenames based on node type."""
    df = pd.DataFrame({'filename': filenames})

    try:
        if node_type in ['tree', 'pole']:
            parts = df['filename'].str[:-4].str.split('_', expand=True)
            result = pd.DataFrame({
                'precolonial': parts[0].str.split('.').str[1],
                'size': parts[1].str.split('.').str[1],
                'control': parts[2].str.split('.').str[1],
                'id': parts[3].str.split('.').str[1].astype(int),
                'filename': df['filename'],
            })
            print("\nDEBUG - ID parts:")
            print(parts[3])
        else:
            parts = df['filename'].str[:-4].str.split('.', expand=True)
            result = pd.DataFrame({
                'size': parts[1],
                'id': parts[3].astype(int),
                'filename': df['filename'],
            })
        return result
    except Exception as e:
        print(f"Warning: Error parsing filenames: {str(e)}")
        print("Parts shape:", parts.shape)
        print("Column 3 (should be IDs):", parts[3].head())
        return pd.DataFrame()


def import_ply_object(filepath):
    before_names = set(bpy.data.objects.keys())
    bpy.ops.wm.ply_import(filepath=filepath)
    new_names = [name for name in bpy.data.objects.keys() if name not in before_names]
    new_objects = [bpy.data.objects[name] for name in new_names]
    mesh_objects = [obj for obj in new_objects if obj.type == 'MESH']
    if mesh_objects:
        return mesh_objects[0]
    if new_objects:
        return new_objects[0]
    return None


def shift_int_resource_on_imported_models(node_objects, delta, label):
    """Shift imported model int_resource values by a constant amount."""
    print(f"\nShifting int_resource by {delta} for {label} imported models...")

    updated_objects = 0
    updated_values = 0
    for obj in node_objects.values():
        mesh = getattr(obj, "data", None)
        if mesh is None or not hasattr(mesh, "attributes"):
            continue

        attr = mesh.attributes.get("int_resource")
        if attr is None:
            continue

        values = np.empty(len(attr.data), dtype=np.float32)
        attr.data.foreach_get("value", values)
        values = np.rint(values).astype(np.int32)
        values += int(delta)
        attr.data.foreach_set("value", values if getattr(attr, "data_type", "") == 'INT' else values.astype(np.float32))
        mesh.update()

        updated_objects += 1
        updated_values += len(values)

    print(
        f"Updated {updated_values} int_resource values across "
        f"{updated_objects} imported {label} models"
    )


def zero_int_resource_ones_on_imported_models(node_objects):
    """Change imported model int_resource values from 2 to 1."""
    print("\nTrending scenario: changing int_resource values from 2 to 1...")

    updated_objects = 0
    updated_values = 0

    for obj in node_objects.values():
        mesh = getattr(obj, "data", None)
        if mesh is None or not hasattr(mesh, "attributes"):
            continue

        attr = mesh.attributes.get("int_resource")
        if attr is None:
            continue

        values = np.empty(len(attr.data), dtype=np.float32)
        attr.data.foreach_get("value", values)
        values = np.rint(values).astype(np.int32)
        mask = values == 2
        if not np.any(mask):
            continue

        values[mask] = 1
        attr.data.foreach_set("value", values if getattr(attr, "data_type", "") == 'INT' else values.astype(np.float32))
        mesh.update()

        updated_objects += 1
        updated_values += int(mask.sum())

    print(f"Updated {updated_values} int_resource values across {updated_objects} imported models")


def set_int_resource_on_imported_models(node_objects, target_value, label):
    """Set all int_resource values on imported models to a single enum value."""
    print(f"\nSetting int_resource to {target_value} for {label} imported models...")

    updated_objects = 0
    updated_values = 0
    target_int = int(target_value)

    for obj in node_objects.values():
        mesh = getattr(obj, "data", None)
        if mesh is None or not hasattr(mesh, "attributes"):
            continue

        attr = mesh.attributes.get("int_resource")
        if attr is None:
            attr = mesh.attributes.new(name="int_resource", type="INT", domain="POINT")

        values = np.full(len(attr.data), target_int, dtype=np.int32)
        attr.data.foreach_set("value", values if getattr(attr, "data_type", "") == 'INT' else values.astype(np.float32))
        mesh.update()

        updated_objects += 1
        updated_values += len(values)

    print(
        f"Updated {updated_values} int_resource values across "
        f"{updated_objects} imported {label} models"
    )


def filename_represents_fallen_tree(filename):
    return "_size.fallen_" in filename


# Process Collection Function
def process_collection(df, ply_folder, node_type, year_collection, variant_suffix=None):
    """Main function to process a collection of nodes."""
    print(f"\nStarting process_collection for {node_type}")
    print(f"Number of {node_type}s to process: {len(df)}")
    print(f"Using PLY folder: {ply_folder}")

    if len(df) == 0:
        print(f"No {node_type}s to process in dataframe")
        return None, {}, None

    print(f"\nProcessing {node_type} collection...")

    node_run_id = get_node_run_id(node_type, variant_suffix)
    positions_name = f"{node_run_id}_positions"
    models_name = f"{node_run_id}_plyModels"

    for coll_name in [positions_name, models_name]:
        existing_collection = year_collection.children.get(coll_name)
        if existing_collection:
            print(f"Cleaning up existing {coll_name} collection...")
            for obj in existing_collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(existing_collection)

    positions_collection = bpy.data.collections.new(positions_name)
    models_collection = bpy.data.collections.new(models_name)
    year_collection.children.link(positions_collection)
    year_collection.children.link(models_collection)

    print(f"Created collections: {positions_name} and {models_name}")

    print("Creating filenames...")
    if node_type in ['tree', 'pole']:
        df['filename'] = (
            'precolonial.' + df['precolonial'].astype(str).str.capitalize() +
            '_size.' + df['size'].astype(str) +
            '_control.' + df['control'].astype(str) +
            '_id.' + df['tree_id'].astype(str) +
            '.ply'
        )
    else:
        df['filename'] = (
            'size.' + df['size'].astype(str) +
            '.log.' + df['tree_id'].astype(str) +
            '.ply'
        )

    unique_filenames = df['filename'].unique()
    fallen_tree_keys = set()
    print(f"\nFound {len(unique_filenames)} unique {node_type} type combinations")

    print("Scanning PLY files...")
    available_plys = pd.Series([
        f for f in os.listdir(ply_folder)
        if f.endswith('.ply') and not f.startswith('artificial_')
    ])
    print(f"Found {len(available_plys)} usable PLY files")

    fallback_map = {}
    if node_type == 'pole' and POLE_BYPASS:
        print(f"\n⚡ POLE BYPASS ENABLED")
        print(f"   Using {POLE_FALLBACK_PLY} for all poles")
        print(f"   Number of unique filenames: {len(unique_filenames)}")
        print(f"   Unique filenames: {unique_filenames}")
        fallback_map = {filename: POLE_FALLBACK_PLY for filename in unique_filenames}
        print(f"   Created fallback map with {len(fallback_map)} entries")
        available_templates = pd.DataFrame(columns=['precolonial', 'size', 'control', 'id', 'filename'])
        needed_templates = pd.DataFrame(columns=['precolonial', 'size', 'control', 'id', 'filename'])
    else:
        available_templates = parse_ply_filenames(available_plys, node_type)
        needed_templates = parse_ply_filenames(unique_filenames, node_type)

        print("\nPreprocessing size mappings...")
        if 'size' in needed_templates.columns:
            artificial_mask = needed_templates['size'] == 'artificial'
            if artificial_mask.any():
                print(f"Converting {artificial_mask.sum()} 'artificial' sizes to 'snag' for template search")
                needed_templates.loc[artificial_mask, 'size'] = 'snag'

        def get_random_index(row, max_val):
            return stable_hash_index(
                (
                    row.get('precolonial', ''),
                    row.get('size', ''),
                    row.get('control', ''),
                    row.get('id', 0),
                ),
                max_val,
            )

        print("\nCreating fallback mapping...")
        used_filenames_by_bucket = {}
        for idx, needed in needed_templates.iterrows():
            print(f"\nProcessing template {idx + 1}/{len(needed_templates)}:")
            print(
                f"Searching for: precolonial={needed.get('precolonial', 'N/A')}, "
                f"size={needed.get('size', 'N/A')}, "
                f"control={needed.get('control', 'N/A')}, "
                f"id={needed.get('id', 'N/A')}"
            )

            if node_type in ['tree', 'pole']:
                exact_match_mask = (
                    (available_templates['precolonial'] == needed['precolonial']) &
                    (available_templates['size'] == needed['size']) &
                    (available_templates['control'] == needed['control']) &
                    (available_templates['id'] == needed['id'])
                )
                best_match_mask = (
                    (available_templates['precolonial'] == needed['precolonial']) &
                    (available_templates['size'] == needed['size']) &
                    (available_templates['control'] == needed['control'])
                )
                precolonial_size_match_mask = (
                    (available_templates['precolonial'] == needed['precolonial']) &
                    (available_templates['size'] == needed['size'])
                )
            else:
                exact_match_mask = (
                    (available_templates['size'] == needed['size']) &
                    (available_templates['id'] == needed['id'])
                )
                best_match_mask = available_templates['size'] == needed['size']
                precolonial_size_match_mask = available_templates['size'] == needed['size']

            size_match_mask = available_templates['size'] == needed['size']

            if exact_match_mask.any():
                matches = available_templates[exact_match_mask]
                match = matches.iloc[0]
                fallback_map[needed['filename']] = match['filename']
                print("✓ Exact match found!")
                print(f"   Using: {match['filename']}")
            elif best_match_mask.any():
                matches = available_templates[best_match_mask]
                usage_key = (
                    'best_match',
                    needed.get('precolonial', ''),
                    needed.get('size', ''),
                    needed.get('control', ''),
                )
                match, random_idx, reused_alternate = choose_stable_match_with_reuse_avoidance(
                    matches,
                    needed,
                    usage_key,
                    used_filenames_by_bucket,
                )
                fallback_map[needed['filename']] = match['filename']
                print("↳ No exact match, falling back to best match")
                print(f"   Found {len(matches)} matching templates")
                print(f"   Selected index {random_idx}: {match['filename']}")
                if reused_alternate:
                    print("   Adjusted selection to avoid reusing an earlier fallback in this bucket")
            elif precolonial_size_match_mask.any():
                matches = available_templates[precolonial_size_match_mask]
                usage_key = (
                    'precolonial_size',
                    needed.get('precolonial', ''),
                    needed.get('size', ''),
                )
                match, random_idx, reused_alternate = choose_stable_match_with_reuse_avoidance(
                    matches,
                    needed,
                    usage_key,
                    used_filenames_by_bucket,
                )
                fallback_map[needed['filename']] = match['filename']
                print("↳ No control match, falling back to same precolonial + size")
                print(f"   Found {len(matches)} precolonial/size-matching templates")
                print(f"   Selected index {random_idx}: {match['filename']}")
                if reused_alternate:
                    print("   Adjusted selection to avoid reusing an earlier fallback in this bucket")
            elif size_match_mask.any():
                matches = available_templates[size_match_mask]
                usage_key = (
                    'size_only',
                    needed.get('size', ''),
                )
                match, random_idx, reused_alternate = choose_stable_match_with_reuse_avoidance(
                    matches,
                    needed,
                    usage_key,
                    used_filenames_by_bucket,
                )
                fallback_map[needed['filename']] = match['filename']
                print("↳ No same-precolonial match, falling back to size-only match")
                print(f"   Found {len(matches)} size-matching templates")
                print(f"   Selected index {random_idx}: {match['filename']}")
                if reused_alternate:
                    print("   Adjusted selection to avoid reusing an earlier fallback in this bucket")
            else:
                random_idx = get_random_index(needed, len(available_templates))
                match = available_templates.iloc[random_idx]
                fallback_map[needed['filename']] = match['filename']
                print("⚠ No good matches found, using random template")
                print(f"   Selecting from {len(available_templates)} available templates")
                print(f"   Selected index {random_idx}: {match['filename']}")

    instance_map = pd.Series(range(len(unique_filenames)), index=sorted(unique_filenames))
    df['instanceID'] = df['filename'].map(instance_map)

    print(f"\nImporting {len(instance_map)} PLY files...")
    node_objects = {}
    run_id = get_run_id()

    for i, orig_filename in enumerate(sorted(unique_filenames)):
        actual_filename = fallback_map.get(orig_filename, orig_filename)
        filepath = os.path.join(ply_folder, actual_filename)

        if os.path.exists(filepath):
            print(f"Importing {actual_filename} as instance {i}")

            target_name = get_import_target_name(i, actual_filename[:-4], variant_suffix)
            for obj in list(bpy.data.objects):
                if obj.name.startswith(target_name):
                    if obj.users_collection:
                        for collection in list(obj.users_collection):
                            collection.objects.unlink(obj)
                    bpy.data.objects.remove(obj)

            node = import_ply_object(filepath)

            if node is None:
                print(f"Warning: Could not find imported object for {actual_filename}")
                continue

            node.name = target_name
            if getattr(node, "data", None) is not None:
                node.data.name = f"{target_name}_mesh"

            if node.users_collection:
                for collection in list(node.users_collection):
                    collection.objects.unlink(node)
            models_collection.objects.link(node)

            node.pass_index = 3
            if node_type == 'tree':
                if filename_represents_fallen_tree(actual_filename):
                    fallen_tree_keys.add(i)

            configure_imported_model_visibility(node)
            node_objects[i] = node
        else:
            print(f"Warning: File not found: {filepath}")

    print(f"Imported {len(node_objects)} unique models")
    if node_type in {'tree', 'log'}:
        print(f"Preserving imported resource_* binary attributes from PLYs for {node_type} models")

    shift_int_resource_on_imported_models(node_objects, 1, node_type)

    if node_type == 'tree':
        if fallen_tree_keys:
            fallen_tree_objects = {
                key: node_objects[key]
                for key in fallen_tree_keys
                if key in node_objects
            }
            if fallen_tree_objects:
                # Disabled: preserve imported tree int_resource values for fallen trees.
                pass
    elif node_type == 'log':
        # Disabled: preserve imported log int_resource values.
        pass

    df['resolved_filename'] = df['filename'].map(lambda x: fallback_map.get(x, x))
    collection_summary = build_collection_summary(
        df,
        unique_filenames,
        node_objects,
        node_type,
        resource_binary_fallback_models=None,
    )

    df['model_index'] = df['filename'].map(lambda x: list(sorted(unique_filenames)).index(x))

    print("\nCreating point cloud...")
    points = df[['x', 'y', 'z']].to_numpy()
    point_cloud_name = get_point_cloud_object_name(node_type, variant_suffix)
    mesh = bpy.data.meshes.new(point_cloud_name)
    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()

    point_cloud = bpy.data.objects.new(point_cloud_name, mesh)
    point_cloud.pass_index = 3
    positions_collection.objects.link(point_cloud)

    print(f"Created point cloud with {len(points)} locations")

    if node_type == 'tree':
        vertex_preview_count = min(50, len(mesh.vertices))
        point_cloud_vertex_preview = pd.DataFrame(
            [
                {
                    'vertex_index': index,
                    'x': float(mesh.vertices[index].co.x),
                    'y': float(mesh.vertices[index].co.y),
                    'z': float(mesh.vertices[index].co.z),
                }
                for index in range(vertex_preview_count)
            ]
        )
        collection_summary['point_cloud_vertex_preview'] = point_cloud_vertex_preview

    print("\nAdding attributes...")
    node_type_series = get_series_or_default(df, 'nodeType', node_type)
    tree_intervention_series = get_series_or_default(df, 'rewilded', '')
    tree_proposal_series = get_series_or_default(df, 'action', '')
    improvement_series = get_series_or_default(df, 'Improvement', '')
    canopy_resistance_series = get_series_or_default(df, 'CanopyResistance', -1.0)
    node_id_series = get_series_or_default(df, 'nodeID', -1)

    base_attr_types = {
        'rotation': ('FLOAT', 'value', df['rotateZ'].to_numpy()),
        'tree_type': ('INT', 'value', df['tree_id'].to_numpy(dtype=np.int32)),
        'structure_id': ('INT', 'value', df['structureID'].to_numpy(dtype=np.int32)),
        'year': (
            'FLOAT',
            'value',
            pd.to_numeric(get_series_or_default(df, 'year', YEAR), errors='coerce').fillna(float(YEAR)).to_numpy(dtype=np.float32),
        ),
        'precolonial': ('INT', 'value', convert_precolonial(df['precolonial']).to_numpy(dtype=np.int32)),
        'size': ('INT', 'value', convert_size(df['size']).to_numpy(dtype=np.int32)),
        'instanceID': ('INT', 'value', df['model_index'].to_numpy(dtype=np.int32)),
        'node_type': ('INT', 'value', convert_node_type(node_type_series).to_numpy(dtype=np.int32)),
        'tree_interventions': (
            'INT',
            'value',
            convert_tree_interventions(tree_intervention_series).to_numpy(dtype=np.int32),
        ),
        'tree_proposals': (
            'INT',
            'value',
            convert_tree_proposals(tree_proposal_series).to_numpy(dtype=np.int32),
        ),
        'improvement': ('INT', 'value', convert_improvement(improvement_series).to_numpy(dtype=np.int32)),
        'canopy_resistance': (
            'FLOAT',
            'value',
            pd.to_numeric(canopy_resistance_series, errors='coerce').fillna(-1.0).to_numpy(dtype=np.float32),
        ),
        'node_id': (
            'INT',
            'value',
            pd.to_numeric(node_id_series, errors='coerce').fillna(-1).to_numpy(dtype=np.int32),
        ),
    }
    for attr_name in PROPOSAL_FRAMEBUFFER_ATTRIBUTE_NAMES:
        base_attr_types[attr_name] = (
            'INT',
            'value',
            pd.to_numeric(get_series_or_default(df, attr_name, 0), errors='coerce').fillna(0).to_numpy(dtype=np.int32),
        )

    if node_type in ['tree', 'pole']:
        base_attr_types['control'] = ('INT', 'value', convert_control(df['control']).to_numpy(dtype=np.int32))
        base_attr_types['life_expectancy'] = ('INT', 'value', df['useful_life_expectancy'].to_numpy(dtype=np.int32))
    else:
        base_attr_types['logMass'] = ('FLOAT', 'value', df['logMass'].to_numpy())

    for attr_name, (attr_type, value_type, data) in base_attr_types.items():
        print(f"Adding {attr_name} attribute...")
        attr = mesh.attributes.new(name=attr_name, type=attr_type, domain='POINT')
        attr.data.foreach_set(value_type, data)
        if hasattr(attr, 'is_runtime_only'):
            attr.is_runtime_only = False

    print(f"\nSuccessfully created {node_type} instance system!")
    print(f"- {len(node_objects)} unique meshes")
    print(f"- {len(df)} instance locations")
    print(f"- {len(base_attr_types)} attributes")

    print("\nSetting up geometry nodes...")
    template_name = "instance_template"
    if template_name in bpy.data.node_groups:
        print(f"Found template node group: {template_name}")

        try:
            node_group_name = node_run_id
            modifier_name = node_group_name

            remove_node_group_if_present(node_group_name)

            new_node_group = bpy.data.node_groups[template_name].copy()
            new_node_group.name = node_group_name
            configure_geometry_node_group(new_node_group, point_cloud)
            print(f"Created new node group: {new_node_group.name}")

            geo_nodes = ensure_geometry_nodes_modifier(
                point_cloud,
                modifier_name,
                new_node_group,
            )
            print("Applied geometry nodes modifier to point cloud")

            for node in new_node_group.nodes:
                if node.type == 'COLLECTION_INFO':
                    if hasattr(node, "inputs") and "Collection" in node.inputs:
                        node.inputs["Collection"].default_value = models_collection
                        print(f"Updated Collection Info node to use {models_collection.name}")
                        break

            print("Geometry nodes setup complete")
            return point_cloud, node_objects, collection_summary

        except Exception as e:
            print(f"Error in geometry nodes setup: {str(e)}")
            return None, {}, collection_summary

    print(f"Warning: Could not find template node group '{template_name}'")
    return None, {}, collection_summary


def apply_site_specific_csv_fixes(df):
    adjusted = df.copy()

    if SITE == 'trimmed-parade':
        adjusted['nodeType'] = 'tree'
        excluded_ids = {253, 255, 256, 691, 696}
        if site_uses_timeline_mode() and 'timeline_year' in adjusted.columns:
            adjusted = adjusted[
                ~(
                    adjusted['timeline_year'].eq(10)
                    & adjusted['structureID'].isin(excluded_ids)
                )
            ]
        elif YEAR == 10:
            adjusted = adjusted[~adjusted['structureID'].isin(excluded_ids)]

    return adjusted


def drop_ignored_size_rows(df):
    if 'size' not in df.columns:
        return df, 0

    normalized_sizes = df['size'].fillna('').astype(str).str.strip().str.lower()
    ignore_mask = normalized_sizes.isin(IGNORED_SIZE_NAMES)
    ignored_count = int(ignore_mask.sum())
    if ignored_count == 0:
        return df, 0

    filtered = df.loc[~ignore_mask].copy()
    ignored_labels = sorted(set(normalized_sizes.loc[ignore_mask]))
    print(
        f"\nIgnoring {ignored_count} rows with unsupported sizes: "
        f"{', '.join(ignored_labels)}"
    )
    return filtered, ignored_count


def ensure_proposal_framebuffer_columns(df):
    proposal_columns = build_blender_proposal_framebuffer_columns(df)
    adjusted = df.copy()
    for attr_name in PROPOSAL_FRAMEBUFFER_ATTRIBUTE_NAMES:
        adjusted[attr_name] = proposal_columns[attr_name].to_numpy(dtype=np.int32)
    return adjusted


def load_scenario_dataframe(scenario_name):
    if TIMELINE_MODE and get_timeline_layout_module().timeline_mode_supported(SITE):
        timeline_layout = get_timeline_layout_module()
        df, source_paths, used_strips = timeline_layout.build_site_dataframe(
            SITE,
            scenario_name,
            get_active_build_years(SITE),
        )
        df = apply_site_specific_csv_fixes(df)
        df = ensure_proposal_framebuffer_columns(df)
        return df, source_paths, used_strips

    df = pd.read_csv(CSV_FILEPATH)
    df = apply_site_specific_csv_fixes(df)
    df = ensure_proposal_framebuffer_columns(df)
    return df, [Path(CSV_FILEPATH)], []


def run_scenario(scene, scenario_name):
    global SCENARIO
    SCENARIO = scenario_name
    refresh_site_paths()

    print(f"\nStarting instance system creation for scenario '{SCENARIO}'...")
    print(f"CSV_FILEPATH IS: {CSV_FILEPATH}")
    print(f"File exists: {os.path.exists(CSV_FILEPATH)}")

    print("\nReading CSV file...")
    df, source_csv_paths, used_timeline_strips = load_scenario_dataframe(SCENARIO)
    df, ignored_size_row_count = drop_ignored_size_rows(df)

    df, size_override_summary = randomize_trending_tree_sizes(df)
    df, tree_id_remap_summary = remap_tree_ids_to_available_models(df, PLY_FOLDER)

    print("\nCSV Contents Summary:")
    print(f"Total rows: {len(df)}")
    print("Unique nodeTypes:", df['nodeType'].unique())
    print("Counts by nodeType:")
    print(df['nodeType'].value_counts())

    if single_state_mode_active():
        year_collection = ensure_run_parent_collection(
            scene,
            get_run_parent_collection_name(scenario_name),
        )
    else:
        year_collection_name = get_run_collection_name()
        year_collection = bpy.data.collections.new(year_collection_name)
        run_parent = ensure_run_parent_collection(
            scene,
            get_run_parent_collection_name(scenario_name),
        )
        ensure_child_collection(run_parent, year_collection)

    if site_uses_timeline_mode():
        df_filtered = df.copy()
        print(f"Timeline mode active; keeping all {len(df_filtered)} rows after strip layout")
    elif USE_CAMERA_VIEW_FILTER:
        camera_obj = find_camera_by_pass_index(PASS_INDEX)
        print(f"Using camera view for filtering: {camera_obj.name} (pass index {PASS_INDEX})")
        df_filtered = filter_by_camera_view(df, scene, camera_obj)
        print(f"Filtered {len(df)} objects to {len(df_filtered)} within camera view")
    else:
        if USE_3D_CURSOR_FILTER:
            filter_origin = scene.cursor.location
            origin_label = "3D Cursor"
        else:
            camera_obj = find_camera_by_pass_index(PASS_INDEX)
            filter_origin = camera_obj.location
            origin_label = f"camera {camera_obj.name}"

        origin_x = filter_origin.x
        origin_y = filter_origin.y

        print(f"Using {origin_label} position for filtering: x={origin_x:.2f}, y={origin_y:.2f}")

        df['distance_to_origin'] = np.sqrt(
            np.square(df['x'] - origin_x) +
            np.square(df['y'] - origin_y)
        )

        mask = (
            (df['x'].between(origin_x - DISTANCE_UNITS, origin_x + DISTANCE_UNITS)) &
            (df['y'].between(origin_y - DISTANCE_UNITS, origin_y + DISTANCE_UNITS))
        )
        df_filtered = df[mask]
        print(f"Filtered {len(df)} objects to {len(df_filtered)} within distance threshold")

    treeDF = df_filtered[df_filtered['nodeType'] == 'tree'].copy()
    poleDF = df_filtered[df_filtered['nodeType'] == 'pole'].copy()
    logDF = df_filtered[df_filtered['nodeType'] == 'log'].copy()
    if len(logDF) > 0:
        logDF.loc[:, 'size'] = 'fallen'
        df_filtered.loc[df_filtered['nodeType'] == 'log', 'size'] = 'fallen'
    merged_tree_log_csv = save_merged_tree_log_csv(df_filtered, scenario_name)

    print(f"\nFound {len(treeDF)} trees, {len(poleDF)} poles, and {len(logDF)} logs to process")

    tree_results = None
    priority_tree_results = None
    priority_log_results = None
    pole_results = None
    log_results = None

    if len(treeDF) > 0:
        print("\nProcessing trees...")
        tree_results = process_collection(treeDF, PLY_FOLDER, 'tree', year_collection)
        if tree_results[0]:
            print(f"Successfully processed {len(tree_results[1])} tree types")
    else:
        print("No trees to process")

    priority_layer_names = ('priority_state', 'city_priority')
    scene_contract = get_scene_contract_module()
    site_contract = scene_contract.SITE_CONTRACTS.get(SITE, {})
    has_priority_target = (
        site_contract.get('build_priority_branch', False)
        and any(scene.view_layers.get(name) is not None for name in priority_layer_names)
    )
    should_build_priority = has_priority_target and scenario_name == 'positive'
    if not has_priority_target:
        print("\nNo priority view layer found; skipping priority tree instance branch")
    elif has_priority_target and scenario_name != 'positive':
        print(f"\nSkipping priority branch for scenario '{scenario_name}' (positive only)")

    if should_build_priority:
        priority_treeDF = treeDF[
            treeDF['size'].astype(str).str.lower().isin(PRIORITY_TREE_SIZES)
        ].copy()
        print(f"\nFound {len(priority_treeDF)} city priority trees (sizes 4/5/6)")

        priority_collection = None
        if len(priority_treeDF) > 0 or len(logDF) > 0:
            if single_state_mode_active():
                priority_collection = ensure_run_parent_collection(
                    scene,
                    get_priority_parent_collection_name(),
                )
            else:
                priority_collection = bpy.data.collections.new(get_priority_run_collection_name())
                priority_parent = ensure_run_parent_collection(
                    scene,
                    get_priority_parent_collection_name(),
                )
                ensure_child_collection(priority_parent, priority_collection)

        if len(priority_treeDF) > 0:
            print("\nProcessing priority trees...")
            priority_tree_results = process_collection(
                priority_treeDF,
                PLY_FOLDER,
                'tree',
                priority_collection,
                variant_suffix='priority',
            )
            if priority_tree_results[0]:
                print(f"Successfully processed {len(priority_tree_results[1])} priority tree types")

        if len(logDF) > 0 and priority_collection is not None:
            print("\nProcessing priority logs...")
            priority_log_results = process_collection(
                logDF.copy(),
                LOG_FOLDER,
                'log',
                priority_collection,
                variant_suffix='priority',
            )
            if priority_log_results[0]:
                print(f"Successfully processed {len(priority_log_results[1])} priority log types")

    if len(poleDF) > 0:
        print("\nProcessing poles...")
        pole_results = process_collection(poleDF, PLY_FOLDER, 'pole', year_collection)
        if pole_results[0]:
            print(f"Successfully processed {len(pole_results[1])} pole types")
    else:
        print("No poles to process")

    if len(logDF) > 0:
        print("\nProcessing logs...")
        log_results = process_collection(logDF, LOG_FOLDER, 'log', year_collection)
        if log_results[0]:
            print(f"Successfully processed {len(log_results[1])} log types")
    else:
        print("No logs to process")

    run_context = {
        'source_csv_paths': [str(path) for path in source_csv_paths],
        'timeline_years': [strip['year'] for strip in used_timeline_strips] or list(get_active_build_years(SITE)),
        'merged_tree_log_csv': str(merged_tree_log_csv) if merged_tree_log_csv else None,
        'build_mode': get_build_mode(SITE),
        'ignored_size_rows': int(ignored_size_row_count),
        'input_rows': int(len(df)),
        'filtered_rows': int(len(df_filtered)),
        'tree_rows': int(len(treeDF)),
        'pole_rows': int(len(poleDF)),
        'log_rows': int(len(logDF)),
    }
    tree_summary = tree_results[2] if tree_results is not None and len(tree_results) > 2 else None
    pole_summary = pole_results[2] if pole_results is not None and len(pole_results) > 2 else None
    log_summary = log_results[2] if log_results is not None and len(log_results) > 2 else None
    write_run_log(run_context, tree_summary, pole_summary, log_summary, size_override_summary, tree_id_remap_summary)
    return {
        'scenario': scenario_name,
        'tree_results': tree_results,
        'priority_tree_results': priority_tree_results,
        'priority_log_results': priority_log_results,
        'pole_results': pole_results,
        'log_results': log_results,
    }


def main():
    scene = require_target_scene()
    configure_site_from_scene(scene)
    timeline_layout = get_timeline_layout_module()
    active_test_mode = timeline_layout.get_active_timeline_test_mode()
    if active_test_mode is not None:
        print(f"Timeline test mode active: {active_test_mode}")
    print(f"Timeline build mode: {get_build_mode(SITE)} (site={SITE}, years={get_active_build_years(SITE)})")
    cleanup_scene()

    available_scenarios = get_available_scenarios()
    print(f"\nScenarios to build: {available_scenarios}")

    scenario_results = []
    for scenario_name in available_scenarios:
        scenario_results.append(run_scenario(scene, scenario_name))

    ensure_single_state_collections_linked(scene)
    configure_scenario_view_layer_visibility(scene)
    run_clipbox_followup_scripts()
    bpy.ops.wm.save_mainfile()

    print("\nAll processing complete!")
    return scenario_results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")
