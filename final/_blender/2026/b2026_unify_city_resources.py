import importlib.util
from pathlib import Path

import bpy


SITE = "city"
SCENARIO = "positive"
YEAR = 180
TARGET_SCENE_NAME = "city-site"
DISTANCE_UNITS = 200
USE_3D_CURSOR_FILTER = True
USE_CAMERA_VIEW_FILTER = False
WRITE_RUN_LOG = True

CITY_VIEW_LAYER_AOVS = (
    ("structure_id", "VALUE"),
    ("resource", "VALUE"),
    ("size", "VALUE"),
    ("control", "VALUE"),
    ("node_type", "VALUE"),
    ("tree_interventions", "VALUE"),
    ("tree_proposals", "VALUE"),
    ("improvement", "VALUE"),
    ("canopy_resistance", "VALUE"),
    ("node_id", "VALUE"),
    ("instance_id", "VALUE"),
    ("isSenescent", "VALUE"),
    ("resource_colour", "COLOR"),
    ("isTerminal", "VALUE"),
    ("bioEnvelopeType", "VALUE"),
    ("bioSimple", "VALUE"),
    ("sim_Turns", "VALUE"),
)

LEGACY_CITY_OBJECT_PREFIXES = (
    "TreePositions.",
    "LogPositions.",
    "precolonial.",
    "size.",
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
BLEND_PATH = ROOT_DIR / "data" / "blender" / "2026" / "2026 hero tests 2_city-site.blend"


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_view_layer_aov(view_layer, name, aov_type):
    for existing in view_layer.aovs:
        if existing.name == name:
            existing.type = aov_type
            return existing

    aov = view_layer.aovs.add()
    aov.name = name
    aov.type = aov_type
    return aov


def ensure_city_scene_aovs():
    scene = bpy.data.scenes.get(TARGET_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{TARGET_SCENE_NAME}' was not found.")

    for view_layer in scene.view_layers:
        for aov_name, aov_type in CITY_VIEW_LAYER_AOVS:
            ensure_view_layer_aov(view_layer, aov_name, aov_type)


def configure_instancer(instancer):
    instancer.SITE = SITE
    instancer.SCENARIO = SCENARIO
    instancer.YEAR = YEAR
    instancer.TARGET_SCENE_NAME = TARGET_SCENE_NAME
    instancer.DISTANCE_UNITS = DISTANCE_UNITS
    instancer.USE_3D_CURSOR_FILTER = USE_3D_CURSOR_FILTER
    instancer.USE_CAMERA_VIEW_FILTER = USE_CAMERA_VIEW_FILTER
    instancer.WRITE_RUN_LOG = WRITE_RUN_LOG
    instancer.BASE_PATH = str(ROOT_DIR / "data" / "revised" / "final" / SITE)
    instancer.CSV_FILENAME = f"{SITE}_{SCENARIO}_1_nodeDF_{YEAR}.csv"
    instancer.CSV_FILEPATH = str(Path(instancer.BASE_PATH) / instancer.CSV_FILENAME)


def hide_object(obj):
    obj.hide_viewport = True
    obj.hide_render = True
    obj.hide_select = True


def hide_collection_objects(collection):
    for obj in collection.objects:
        hide_object(obj)
    for child in collection.children:
        hide_collection_objects(child)


def hide_legacy_city_content():
    legacy_city_collection = bpy.data.collections.get("City_Year_180")
    if legacy_city_collection is not None:
        hide_collection_objects(legacy_city_collection)

    scene = bpy.data.scenes.get(TARGET_SCENE_NAME)
    if scene is None:
        return

    for obj in scene.objects:
        if obj.name.startswith("TreePositions_city_") or obj.name.startswith("LogPositions_city_"):
            continue
        if obj.name.startswith(LEGACY_CITY_OBJECT_PREFIXES):
            hide_object(obj)


def run_tree_aov_setup():
    module = load_module(
        "b2026_tree_aov_setup_runtime",
        SCRIPT_DIR / "b2026_tree_aov_setup.py",
    )
    module.TARGET_SCENE_NAME = TARGET_SCENE_NAME
    module.TARGET_VIEW_LAYER_NAME = None
    module.main()


def run_clipbox_setup():
    module = load_module(
        "b2026_clipbox_setup_runtime",
        SCRIPT_DIR / "b2026_clipbox_setup.py",
    )
    module.main()


def run_compositor_aov_expose():
    module = load_module(
        "b2026_compositor_aov_expose_runtime",
        SCRIPT_DIR / "b2026_compositor_aov_expose.py",
    )
    module.main()


def main():
    instancer = load_module(
        "b2026_instancer_runtime",
        SCRIPT_DIR / "b2026_instancer.py",
    )
    configure_instancer(instancer)
    instancer.main()

    ensure_city_scene_aovs()
    run_tree_aov_setup()
    hide_legacy_city_content()
    run_clipbox_setup()
    run_compositor_aov_expose()

    bpy.ops.wm.save_mainfile(filepath=str(BLEND_PATH))
    print(f"Saved {BLEND_PATH}")


if __name__ == "__main__":
    main()
