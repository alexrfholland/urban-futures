import bpy


TARGET_SCENE_NAME = "city"
TARGET_VIEW_LAYER_NAME = "city_priority"
YEAR = 180
SCENARIO = "positive"
PRIORITY_COLLECTION_NAME = f"Year_city_{YEAR}_{SCENARIO}_priority"
KEEP_TOP_LEVEL_COLLECTION_NAMES = (
    "City_Manager",
    "City_Camera",
    "City_Camera_Archive",
    PRIORITY_COLLECTION_NAME,
)


def link_collection_to_scene(scene, collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return False
    if scene.collection.children.get(collection.name) is None:
        scene.collection.children.link(collection)
    return True


def set_branch_exclude(layer_collection, excluded):
    layer_collection.exclude = excluded
    for child in layer_collection.children:
        set_branch_exclude(child, excluded)


def configure_view_layer(view_layer):
    root = view_layer.layer_collection
    for child in root.children:
        keep = child.collection.name in KEEP_TOP_LEVEL_COLLECTION_NAMES
        set_branch_exclude(child, not keep)


def main():
    scene = bpy.data.scenes.get(TARGET_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{TARGET_SCENE_NAME}' not found.")

    view_layer = scene.view_layers.get(TARGET_VIEW_LAYER_NAME)
    if view_layer is None:
        raise ValueError(
            f"View layer '{TARGET_VIEW_LAYER_NAME}' not found in scene '{scene.name}'."
        )

    linked = []
    missing = []
    for collection_name in KEEP_TOP_LEVEL_COLLECTION_NAMES:
        if link_collection_to_scene(scene, collection_name):
            linked.append(collection_name)
        else:
            missing.append(collection_name)

    configure_view_layer(view_layer)

    print(
        f"Configured scene '{scene.name}' view layer '{view_layer.name}' "
        f"to keep only: {linked}"
    )
    if missing:
        print(f"Missing collections not linked: {missing}")


if __name__ == "__main__":
    main()
