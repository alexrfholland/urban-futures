import os
from pathlib import Path

import bpy


OUTPUT_DIR = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/snapshot_validation")
CITY_OUTPUT = OUTPUT_DIR / "city_snapshot_4k.png"
PARADE_OUTPUT = OUTPUT_DIR / "parade_snapshot_4k.png"
PATHWAY_ENV = "B2026_PATHWAY_EXR"
EXISTING_ENV = "B2026_EXISTING_EXR"
PARADE_PATHWAY_ENV = "B2026_PARADE_PATHWAY_EXR"
PARADE_EXISTING_ENV = "B2026_PARADE_EXISTING_EXR"
SNAPSHOT_PREFIX = "Snapshot EXR :: "


def log(message: str) -> None:
    print(f"[snapshot_pair] {message}")


def snapshot_scene() -> bpy.types.Scene:
    if not bpy.data.scenes:
        raise RuntimeError("No scenes were found in the snapshot blend.")
    return bpy.data.scenes[0]


def snapshot_image_nodes(node_tree: bpy.types.NodeTree) -> dict[str, bpy.types.Node]:
    nodes = {}
    for node in node_tree.nodes:
        if node.bl_idname != "CompositorNodeImage":
            continue
        label = node.label or node.name
        if not label.startswith(SNAPSHOT_PREFIX):
            continue
        layer = label.removeprefix(SNAPSHOT_PREFIX).strip()
        nodes[layer] = node
    return nodes


def set_snapshot_images(scene: bpy.types.Scene, pathway_exr: Path, existing_exr: Path) -> None:
    nodes = snapshot_image_nodes(scene.node_tree)
    mapping = {
        "pathway_state": pathway_exr,
        "existing_condition": existing_exr,
    }
    for layer, path in mapping.items():
        node = nodes.get(layer)
        if node is None:
            raise RuntimeError(f"Snapshot image node for '{layer}' was not found.")
        node.image = bpy.data.images.load(str(path), check_existing=True)


def render_scene(scene: bpy.types.Scene, output_path: Path) -> None:
    render = scene.render
    image_settings = render.image_settings
    saved = {
        "filepath": render.filepath,
        "file_format": image_settings.file_format,
        "color_mode": image_settings.color_mode,
    }
    try:
        render.filepath = str(output_path)
        image_settings.file_format = "PNG"
        image_settings.color_mode = "RGBA"
        log(f"Rendering snapshot compositor to {output_path}")
        bpy.ops.render.render(write_still=True, scene=scene.name)
    finally:
        render.filepath = saved["filepath"]
        image_settings.file_format = saved["file_format"]
        image_settings.color_mode = saved["color_mode"]


def required_path(env_name: str) -> Path:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable {env_name}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = snapshot_scene()

    set_snapshot_images(
        scene,
        required_path(PATHWAY_ENV),
        required_path(EXISTING_ENV),
    )
    render_scene(scene, CITY_OUTPUT)

    set_snapshot_images(
        scene,
        required_path(PARADE_PATHWAY_ENV),
        required_path(PARADE_EXISTING_ENV),
    )
    render_scene(scene, PARADE_OUTPUT)


if __name__ == "__main__":
    main()
