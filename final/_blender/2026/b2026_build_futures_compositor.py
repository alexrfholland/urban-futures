import bpy
import os
from pathlib import Path


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


SOURCE_BLEND_PATH = Path(
    env_str(
        "B2026_SOURCE_BLEND_PATH",
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes.blend",
    )
)
SOURCE_SCENE_NAME = env_str("B2026_SOURCE_SCENE_NAME", "city")
SOURCE_SNAPSHOT_BLEND_PATH = env_str("B2026_SOURCE_SNAPSHOT_BLEND_PATH", "")
OUTPUT_SCENE_NAME = env_str("B2026_OUTPUT_SCENE_NAME", "futures_compositing_pipeline")
OUTPUT_BLEND_PATH = Path(
    env_str(
        "B2026_OUTPUT_BLEND_PATH",
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/initial snapshot/futures_compositing_pipeline.blend",
    )
)


def log(message: str) -> None:
    print(f"[futures_compositor] {message}")


def latest_snapshot_blend() -> Path:
    root = SOURCE_BLEND_PATH.parent / f"{SOURCE_BLEND_PATH.stem}_snapshots"
    if not root.exists():
        raise FileNotFoundError(f"Snapshot root folder not found: {root}")

    snapshot_name = f"{slugify(SOURCE_SCENE_NAME)}_snapshot.blend"
    archive_dirs = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    for folder in archive_dirs:
        candidate = folder / snapshot_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No archived snapshot blend named '{snapshot_name}' found under {root}")


def resolve_source_snapshot_blend() -> Path:
    if SOURCE_SNAPSHOT_BLEND_PATH.strip():
        path = Path(SOURCE_SNAPSHOT_BLEND_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Explicit source snapshot blend not found: {path}")
        return path
    return latest_snapshot_blend()


def import_snapshot_scene(snapshot_blend_path: Path) -> bpy.types.Scene:
    scene_name = f"{SOURCE_SCENE_NAME} Snapshot"
    with bpy.data.libraries.load(str(snapshot_blend_path), link=False) as (data_from, data_to):
        scene_names = [name for name in data_from.scenes if name == scene_name]
        if not scene_names:
            raise ValueError(f"No scene named '{scene_name}' found in {snapshot_blend_path}")
        data_to.scenes = scene_names

    imported_scene = next((scene for scene in data_to.scenes if scene is not None), None)
    if imported_scene is None:
        raise ValueError(f"Failed to import '{scene_name}' from {snapshot_blend_path}")
    return imported_scene


def collect_datablocks_for_export(scene: bpy.types.Scene) -> set:
    data_blocks = {scene}
    visited_trees = set()

    def visit_tree(node_tree: bpy.types.NodeTree) -> None:
        if node_tree in visited_trees:
            return
        visited_trees.add(node_tree)
        data_blocks.add(node_tree)
        for node in node_tree.nodes:
            node_group = getattr(node, "node_tree", None)
            if node_group is not None:
                visit_tree(node_group)
            image = getattr(node, "image", None)
            if image is not None:
                data_blocks.add(image)

    visit_tree(scene.node_tree)
    return data_blocks


def export_output_blend(scene: bpy.types.Scene) -> None:
    OUTPUT_BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_BLEND_PATH.exists():
        OUTPUT_BLEND_PATH.unlink()
    bpy.data.libraries.write(str(OUTPUT_BLEND_PATH), collect_datablocks_for_export(scene))


def main() -> None:
    snapshot_blend_path = resolve_source_snapshot_blend()
    imported_scene = import_snapshot_scene(snapshot_blend_path)
    imported_scene.name = OUTPUT_SCENE_NAME
    export_output_blend(imported_scene)
    log(f"Source snapshot blend: {snapshot_blend_path}")
    log(f"Wrote reusable compositor blend: {OUTPUT_BLEND_PATH}")


if __name__ == "__main__":
    main()
