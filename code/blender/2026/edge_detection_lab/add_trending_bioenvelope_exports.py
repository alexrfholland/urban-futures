import bpy
from pathlib import Path


TARGET_SCENE = "City"
TRENDING_NODE_NAME = "City EXR :: trending_state"
SOURCE_OUTPUT_NAME = "bioEnvelopeType"
REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_DIR = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"
OUTPUT_DIR = REPO_ROOT / "data" / "blender" / "2026" / "trending_bioenvelope_bioenvelopetype"
PALETTE_GROUP_NAME = "WORLD_BIOENVELOPE_PALETTE"
EXR_NODE_TO_FILE = {
    "City EXR :: pathway_state": "city-pathway_state.exr",
    "City EXR :: existing_condition": "city-existing_condition.exr",
    "City EXR :: city_priority": "city-city_priority.exr",
    "City EXR :: city_bioenvelope": "city-city_bioenvelope.exr",
    "City EXR :: trending_state": "city-trending_state.exr",
}

COLOURED_DEFAULTS = {
    "0 Unmatched": (0.0, 0.0, 0.0, 0.0),
    "1 Exoskeleton": (0.85, 0.39, 0.55, 1.0),
    "2 BrownRoof": (0.72, 0.48, 0.22, 1.0),
    "3 OtherGround": (0.46, 0.64, 0.77, 1.0),
    "4 Rewilded": (0.36, 0.72, 0.34, 1.0),
    "5 FootprintDepaved": (0.92, 0.75, 0.33, 1.0),
    "6 LivingFacade": (0.30, 0.66, 0.60, 1.0),
    "7 GreenRoof": (0.55, 0.80, 0.31, 1.0),
}

PALETTE_NAMES = (
    "0 Unmatched",
    "1 Exoskeleton",
    "2 BrownRoof",
    "3 OtherGround",
    "4 Rewilded",
    "5 FootprintDepaved",
    "6 LivingFacade",
    "7 GreenRoof",
)

EXPORT_SPECS = (
    {"group_output": "Image", "slug": "full-image", "socket_name": "Image"},
    {"group_output": "1 Exoskeleton", "slug": "exoskeleton", "socket_name": "1 Exoskeleton"},
    {"group_output": "2 BrownRoof", "slug": "brownroof", "socket_name": "2 BrownRoof"},
    {"group_output": "3 OtherGround", "slug": "otherground", "socket_name": "3 OtherGround"},
    {"group_output": "4 Rewilded", "slug": "rewilded", "socket_name": "4 Rewilded"},
    {"group_output": "5 FootprintDepaved", "slug": "footprintdepaved", "socket_name": "5 FootprintDepaved"},
    {"group_output": "6 LivingFacade", "slug": "livingfacade", "socket_name": "6 LivingFacade"},
    {"group_output": "7 GreenRoof", "slug": "greenroof", "socket_name": "7 GreenRoof"},
)


def ensure_link(node_tree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def ensure_frame(node_tree, name, label, location):
    node = node_tree.nodes.get(name)
    if node is None or node.bl_idname != "NodeFrame":
        node = node_tree.nodes.new("NodeFrame")
    node.name = name
    node.label = label
    node.location = location
    return node


def ensure_group_node(node_tree, node_name, node_group, location, parent):
    node = node_tree.nodes.get(node_name)
    if node is None or node.bl_idname != "CompositorNodeGroup":
        node = node_tree.nodes.new("CompositorNodeGroup")
    node.name = node_name
    node.label = node_name
    node.node_tree = node_group
    node.location = location
    node.parent = parent
    return node


def ensure_reroute(node_tree, name, label, location, parent, input_socket):
    node = node_tree.nodes.get(name)
    if node is None or node.bl_idname != "NodeReroute":
        node = node_tree.nodes.new("NodeReroute")
    node.name = name
    node.label = label
    node.location = location
    node.parent = parent
    ensure_link(node_tree, input_socket, node.inputs[0])
    return node


def set_group_defaults(group_node):
    for input_name, value in COLOURED_DEFAULTS.items():
        socket = group_node.inputs.get(input_name)
        if socket is not None:
            socket.default_value = value


def require_scene():
    scene = bpy.data.scenes.get(TARGET_SCENE)
    if scene is None:
        raise ValueError(f"Scene '{TARGET_SCENE}' was not found.")
    if not scene.use_nodes or scene.node_tree is None:
        raise ValueError(f"Scene '{TARGET_SCENE}' does not have a compositor node tree.")
    return scene


def require_palette_group():
    group = bpy.data.node_groups.get(PALETTE_GROUP_NAME)
    if group is None:
        raise ValueError(f"Node group '{PALETTE_GROUP_NAME}' was not found.")
    return group


def refresh_exr_images(node_tree):
    refreshed = []
    for node_name, filename in EXR_NODE_TO_FILE.items():
        node = node_tree.nodes.get(node_name)
        if node is None or node.image is None:
            continue

        filepath = EXR_DIR / filename
        if not filepath.exists():
            raise ValueError(f"Expected EXR was not found: {filepath}")

        node.image.filepath = str(filepath)
        node.image.reload()
        refreshed.append((node_name, str(filepath)))
    return refreshed


def require_source_socket(node_tree):
    node = node_tree.nodes.get(TRENDING_NODE_NAME)
    if node is None:
        raise ValueError(f"Node '{TRENDING_NODE_NAME}' was not found.")
    socket = node.outputs.get(SOURCE_OUTPUT_NAME)
    if socket is None:
        raise ValueError(f"Output '{SOURCE_OUTPUT_NAME}' was not found on '{TRENDING_NODE_NAME}'.")
    return socket


def ensure_file_output(node_tree, parent):
    node = node_tree.nodes.get("trending_bioenvelope_file_output")
    if node is None or node.bl_idname != "CompositorNodeOutputFile":
        if node is not None:
            node_tree.nodes.remove(node)
        node = node_tree.nodes.new("CompositorNodeOutputFile")
    node.name = "trending_bioenvelope_file_output"
    node.label = "trending_bioenvelope_file_output"
    node.location = (420.0, 0.0)
    node.parent = parent
    node.base_path = str(OUTPUT_DIR)
    node.format.file_format = "PNG"
    node.format.color_mode = "RGBA"
    node.format.color_depth = "8"
    while len(node.file_slots) > 0:
        node.file_slots.remove(node.inputs[0])
    for spec in EXPORT_SPECS:
        node.file_slots.new(spec["socket_name"])
        node.file_slots[-1].path = f"trending_bioenvelope_{spec['slug']}_"
    return node


def patch_scene(scene):
    node_tree = scene.node_tree
    source_socket = require_source_socket(node_tree)
    palette_group = require_palette_group()

    colour_frame = ensure_frame(
        node_tree,
        "TrendingBioEnvelope::Colour",
        "trending_bioenvelope_coloured",
        (16050.0, -5200.0),
    )
    group_node = ensure_group_node(
        node_tree,
        "trending_bioenvelope_coloured",
        palette_group,
        (0.0, 0.0),
        colour_frame,
    )
    set_group_defaults(group_node)
    ensure_link(node_tree, source_socket, group_node.inputs["WorldBioEnvelope"])

    ensure_reroute(
        node_tree,
        "trending_bioenvelope_full-image_output",
        "trending_bioenvelope_full-image_output",
        (340.0, -180.0),
        colour_frame,
        group_node.outputs["Image"],
    )

    export_frame = ensure_frame(
        node_tree,
        "TrendingBioEnvelope::Exports",
        "trending_bioenvelope_exports",
        (16620.0, -5200.0),
    )
    file_output = ensure_file_output(node_tree, export_frame)

    for index, spec in enumerate(EXPORT_SPECS):
        reroute = ensure_reroute(
            node_tree,
            f"trending_bioenvelope_{spec['slug']}_output",
            f"trending_bioenvelope_{spec['slug']}_output",
            (120.0, -(index * 140.0)),
            export_frame,
            group_node.outputs[spec["group_output"]],
        )
        ensure_link(node_tree, reroute.outputs[0], file_output.inputs[index])

    return file_output


def rename_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    renamed = []
    for spec in EXPORT_SPECS:
        frame_path = OUTPUT_DIR / f"trending_bioenvelope_{spec['slug']}_0000.png"
        final_path = OUTPUT_DIR / f"trending_bioenvelope_{spec['slug']}.png"
        if frame_path.exists():
            if final_path.exists():
                final_path.unlink()
            frame_path.replace(final_path)
            renamed.append(final_path)
    return renamed


def render_only_new_output(scene, active_output_node):
    original_mute = {}
    original_filepath = scene.render.filepath
    temp_render_prefix = OUTPUT_DIR / "_discard_render"
    for node in scene.node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile":
            original_mute[node.name] = node.mute
            node.mute = node.name != active_output_node.name
    try:
        scene.render.filepath = str(temp_render_prefix)
        bpy.ops.render.render(write_still=True)
    finally:
        scene.render.filepath = original_filepath
        for node_name, mute in original_mute.items():
            node = scene.node_tree.nodes.get(node_name)
            if node is not None:
                node.mute = mute
        for path in OUTPUT_DIR.glob("_discard_render*"):
            if path.is_file():
                path.unlink()


def main():
    scene = require_scene()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    refreshed = refresh_exr_images(scene.node_tree)
    file_output = patch_scene(scene)
    bpy.ops.wm.save_mainfile()
    render_only_new_output(scene, file_output)
    renamed = rename_outputs()
    bpy.ops.wm.save_mainfile()
    print(f"Saved blend: {bpy.data.filepath}")
    for node_name, filepath in refreshed:
        print(f"EXR {node_name} -> {filepath}")
    for path in renamed:
        print(f"OUTPUT {path}")


if __name__ == "__main__":
    main()
