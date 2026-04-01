from __future__ import annotations

from pathlib import Path
import bpy


SOURCE_BLEND = Path(r"D:\2026 Arboreal Futures\data\2026 futures heroes6.blend")
OUTPUT_DIR = Path(r"D:\2026 Arboreal Futures\data")

TARGETS = {
    "city": {
        "source_scene": "city",
        "output_blend": OUTPUT_DIR / "2026 futures city lightweight.blend",
        "camera_name": "paraview_camera_city",
        "view_layers": (
            "pathway_state",
            "existing_condition",
            "city_priority",
            "city_bioenvelope",
            "trending_state",
        ),
    },
    "street": {
        "source_scene": "parade",
        "output_blend": OUTPUT_DIR / "2026 futures street lightweight.blend",
        "camera_name": "paraview_camera_parade",
        "view_layers": (
            "pathway_state",
            "existing_condition",
            "priority_state",
            "bioenvelope_positive",
            "bioenvelope_trending",
            "trending_state",
        ),
    },
}


def ensure_render_rgba(scene: bpy.types.Scene) -> None:
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"


def remove_all_text_blocks() -> None:
    for text in list(bpy.data.texts):
        bpy.data.texts.remove(text)


def remove_other_scenes(target_scene_name: str) -> None:
    for scene in list(bpy.data.scenes):
        if scene.name != target_scene_name:
            bpy.data.scenes.remove(scene)


def assert_view_layers(scene: bpy.types.Scene, expected_layers: tuple[str, ...]) -> None:
    missing = [layer for layer in expected_layers if scene.view_layers.get(layer) is None]
    if missing:
        raise RuntimeError(
            f"Scene '{scene.name}' is missing required view layers: {', '.join(missing)}"
        )


def save_minimal_blend(target_name: str) -> Path:
    spec = TARGETS[target_name]
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE_BLEND))

    scene = bpy.data.scenes.get(spec["source_scene"])
    if scene is None:
        raise RuntimeError(f"Scene '{spec['source_scene']}' was not found in {SOURCE_BLEND}")

    assert_view_layers(scene, spec["view_layers"])
    if bpy.data.objects.get(spec["camera_name"]) is None:
        raise RuntimeError(f"Camera '{spec['camera_name']}' was not found in {SOURCE_BLEND}")

    if target_name != spec["source_scene"]:
        scene.name = target_name
    ensure_render_rgba(scene)
    remove_other_scenes(scene.name)
    remove_all_text_blocks()

    output_blend = spec["output_blend"]
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    return output_blend


def main() -> None:
    for target_name in TARGETS:
        output_blend = save_minimal_blend(target_name)
        print(f"Saved minimal blend: {output_blend}")


if __name__ == "__main__":
    main()
