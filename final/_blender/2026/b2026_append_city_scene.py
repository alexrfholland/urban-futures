import bpy
import os


CITY_BLEND_PATH = "/Users/alexholland/Data/2024/2024 - Arboreal Futures/blender/City.blend"
SOURCE_SCENE_NAME = "Scene.001"
TARGET_SCENE_NAME = "city-site"
SAVE_FILE = False


TOP_LEVEL_COLLECTION_RENAMES = {
    "Ground": "city_envelope",
    "World": "City_World",
    "Year_180": "City_Year_180",
    "Year_60": "City_Year_60",
    "Manager": "City_Manager",
    "Camera": "City_Camera",
}

PASS_INDEX_RULES = (
    ("city_highResRoad", 1),
    ("city_buildings", 2),
    ("city_1_envelope_scenarioYR180", 5),
)


def make_unique_name(name, existing_names):
    if name not in existing_names:
        return name

    index = 1
    while f"{name}.{index:03d}" in existing_names:
        index += 1
    return f"{name}.{index:03d}"


def base_collection_key(collection_name):
    if collection_name.startswith("Ground"):
        return "Ground"
    if collection_name.startswith("World"):
        return "World"
    if collection_name.startswith("Year_180"):
        return "Year_180"
    if collection_name.startswith("Year_60"):
        return "Year_60"
    if collection_name.startswith("Manager"):
        return "Manager"
    if collection_name.startswith("Camera"):
        return "Camera"
    return None


def append_scene_from_blend():
    if not os.path.exists(CITY_BLEND_PATH):
        raise FileNotFoundError(f"City blend not found: {CITY_BLEND_PATH}")

    if TARGET_SCENE_NAME in bpy.data.scenes:
        raise ValueError(
            f"Scene '{TARGET_SCENE_NAME}' already exists. Rename or delete it before importing again."
        )

    existing_scene_names = {scene.name for scene in bpy.data.scenes}

    with bpy.data.libraries.load(CITY_BLEND_PATH, link=False) as (data_from, data_to):
        if SOURCE_SCENE_NAME not in data_from.scenes:
            raise ValueError(
                f"Scene '{SOURCE_SCENE_NAME}' not found in {CITY_BLEND_PATH}. "
                f"Available scenes: {data_from.scenes}"
            )
        data_to.scenes = [SOURCE_SCENE_NAME]

    imported_scenes = [scene for scene in bpy.data.scenes if scene.name not in existing_scene_names]
    if len(imported_scenes) != 1:
        raise RuntimeError(f"Expected 1 imported scene, found {len(imported_scenes)}")

    city_scene = imported_scenes[0]
    city_scene.name = TARGET_SCENE_NAME
    return city_scene


def rename_top_level_collections(scene):
    existing_collection_names = {collection.name for collection in bpy.data.collections}

    for child in list(scene.collection.children):
        base_key = base_collection_key(child.name)
        if not base_key:
            continue

        target_name = TOP_LEVEL_COLLECTION_RENAMES[base_key]
        existing_collection_names.discard(child.name)
        child.name = make_unique_name(target_name, existing_collection_names)
        existing_collection_names.add(child.name)


def print_scene_summary(scene):
    print(f"Imported scene: {scene.name}")
    print("Top-level collections:")
    for child in scene.collection.children:
        print(f"  - {child.name}")

    print("View layers:")
    for view_layer in scene.view_layers:
        print(f"  - {view_layer.name}")


def apply_city_pass_index_contract():
    for obj in bpy.data.objects:
        for prefix, pass_index in PASS_INDEX_RULES:
            if obj.name.startswith(prefix):
                obj.pass_index = pass_index
                break


def main():
    city_scene = append_scene_from_blend()
    rename_top_level_collections(city_scene)
    apply_city_pass_index_contract()
    print_scene_summary(city_scene)

    if SAVE_FILE:
        bpy.ops.wm.save_mainfile()
        print("Saved current blend file")


if __name__ == "__main__":
    main()
