import json
import os
from pathlib import Path

import bpy


RUN_DIR = Path(os.environ["B2026_RUN_DIR"])
SCENE_NAME = os.environ["B2026_SCENE_NAME"]
PATHWAY_EXR = Path(os.environ["B2026_PATHWAY_EXR"])
EXISTING_EXR = Path(os.environ["B2026_EXISTING_EXR"])
OUTPUT_PNG = RUN_DIR / f"{SCENE_NAME}_snapshot.png"
SNAPSHOT_PREFIX = "Snapshot EXR :: "


def snapshot_scene() -> bpy.types.Scene:
    if not bpy.data.scenes:
        raise RuntimeError("No scenes were found in the snapshot blend.")
    return bpy.data.scenes[0]


def image_nodes(node_tree: bpy.types.NodeTree) -> dict[str, bpy.types.Node]:
    found = {}
    for node in node_tree.nodes:
        if node.bl_idname != "CompositorNodeImage":
            continue
        label = node.label or node.name
        if not label.startswith(SNAPSHOT_PREFIX):
            continue
        layer_name = label.removeprefix(SNAPSHOT_PREFIX).strip()
        found[layer_name] = node
    return found


def main() -> None:
    scene = snapshot_scene()
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 512
    scene.cycles.preview_samples = 64
    scene.render.film_transparent = True
    scene.render.resolution_x = 3840
    scene.render.resolution_y = 2160
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(OUTPUT_PNG)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    nodes = image_nodes(scene.node_tree)
    nodes["pathway_state"].image = bpy.data.images.load(str(PATHWAY_EXR), check_existing=True)
    nodes["existing_condition"].image = bpy.data.images.load(str(EXISTING_EXR), check_existing=True)

    bpy.ops.render.render(write_still=True, scene=scene.name)

    manifest_path = RUN_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("snapshot_pngs", {})[SCENE_NAME] = str(OUTPUT_PNG)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
