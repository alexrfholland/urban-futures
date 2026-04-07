from __future__ import annotations

import hashlib
import os
from pathlib import Path

import bpy


BLEND_SPECS = {
    "city": {
        "blend": Path(r"D:\2026 Arboreal Futures\data\2026 futures city lightweight.blend"),
        "camera": "paraview_camera_city",
        "view_layers": (
            "pathway_state",
            "existing_condition",
            "city_priority",
            "city_bioenvelope",
            "trending_state",
        ),
        "output_dir": Path(r"D:\2026 Arboreal Futures\data\renders\paraview\city_lightweight_tests"),
    },
    "street": {
        "blend": Path(r"D:\2026 Arboreal Futures\data\2026 futures street lightweight.blend"),
        "camera": "paraview_camera_parade",
        "view_layers": (
            "pathway_state",
            "existing_condition",
            "priority_state",
            "bioenvelope_positive",
            "bioenvelope_trending",
            "trending_state",
        ),
        "output_dir": Path(r"D:\2026 Arboreal Futures\data\renders\paraview\street_lightweight_tests"),
    },
}

PREVIEW_PERCENTAGE = 25
PREVIEW_SAMPLES = 8


def ensure_render_rgba(scene: bpy.types.Scene) -> None:
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.use_compositing = False
    scene.render.use_sequencer = False


def render_view_layer(scene: bpy.types.Scene, camera_name: str, output_dir: Path, view_layer_name: str) -> Path:
    camera = bpy.data.objects.get(camera_name)
    if camera is None or camera.type != "CAMERA":
        raise RuntimeError(f"Camera '{camera_name}' was not found")
    if scene.view_layers.get(view_layer_name) is None:
        raise RuntimeError(f"View layer '{view_layer_name}' was not found in scene '{scene.name}'")

    original_camera = scene.camera
    original_engine = scene.render.engine
    original_percentage = scene.render.resolution_percentage
    original_samples = scene.cycles.samples if hasattr(scene, "cycles") else None
    original_preview_samples = scene.cycles.preview_samples if hasattr(scene, "cycles") else None
    original_layer_use = {layer.name: layer.use for layer in scene.view_layers}

    output_dir.mkdir(parents=True, exist_ok=True)
    scene.camera = camera
    ensure_render_rgba(scene)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_percentage = PREVIEW_PERCENTAGE
    if hasattr(scene, "eevee"):
        eevee = scene.eevee
        for attr in ("taa_render_samples", "taa_samples"):
            if hasattr(eevee, attr):
                setattr(eevee, attr, PREVIEW_SAMPLES)
                break

    print(
        f"Rendering {scene.name} / {view_layer_name} at "
        f"{PREVIEW_PERCENTAGE}% and {PREVIEW_SAMPLES} samples"
    )
    for layer in scene.view_layers:
        layer.use = layer.name == view_layer_name

    output_prefix = "city" if camera_name.endswith("_city") else "street"
    output_path = output_dir / f"{output_prefix}_{camera_name}_{view_layer_name}_transparent.png"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)

    if not output_path.exists():
        raise RuntimeError(f"Render did not produce {output_path}")

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"Rendered {output_path} sha256={digest}")

    scene.camera = original_camera
    scene.render.engine = original_engine
    scene.render.resolution_percentage = original_percentage
    if hasattr(scene, "cycles") and original_samples is not None:
        scene.cycles.samples = original_samples
        scene.cycles.preview_samples = original_preview_samples
    for layer in scene.view_layers:
        layer.use = original_layer_use.get(layer.name, True)

    return output_path


def main() -> None:
    requested = os.environ.get("B2026_CITY_STREET_TARGETS", "").strip()
    targets = tuple(
        target for target in (requested.split(",") if requested else BLEND_SPECS.keys()) if target
    )
    for label in targets:
        spec = BLEND_SPECS[label]
        bpy.ops.wm.open_mainfile(filepath=str(spec["blend"]))
        scene = bpy.data.scenes[0]
        if scene.name not in ("city", "parade"):
            raise RuntimeError(f"Unexpected scene loaded from {spec['blend']}: {scene.name}")
        for view_layer_name in spec["view_layers"]:
            render_view_layer(scene, spec["camera"], spec["output_dir"], view_layer_name)
        print(f"Rendered previews for {label}")


if __name__ == "__main__":
    main()
