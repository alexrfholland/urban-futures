import hashlib
import importlib.util
import sys
from pathlib import Path

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract
import b2026_timeline_runtime_flags as runtime_flags


OUTPUT_DIR = Path(r"D:\2026 Arboreal Futures\data\renders\paraview\timeline_pointcloud")
TARGET_VIEW_LAYERS = (
    "existing_condition",
    "pathway_state",
    "priority_state",
    "trending_state",
    "bioenvelope_positive",
)
PREVIEW_SAMPLES = 16
PREVIEW_PERCENTAGE = 50

RENDER_SPECS = {
    "city": {
        "site": "city",
        "camera": "paraview_camera_city",
    },
    "parade": {
        "site": "trimmed-parade",
        "camera": "paraview_camera_parade",
    },
    "uni": {
        "site": "uni",
        "camera": "paraview_camera_uni",
    },
}


def load_local_module(module_name: str, filename: str):
    file_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def set_collection_render_state(collection_names, hide_render):
    original_state = {}
    for collection_name in collection_names:
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            continue
        original_state[collection_name] = collection.hide_render
        collection.hide_render = hide_render
    return original_state


def restore_collection_render_state(state_by_name):
    for collection_name, hide_render in state_by_name.items():
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            collection.hide_render = hide_render


def build_scene_collection_toggles(site: str, view_layer_name: str):
    contract = scene_contract.SITE_CONTRACTS[site]
    legacy = contract["legacy"]
    top = contract["top_level"]

    cube_timeline_name = f"{legacy['base_cubes']}_Timeline"
    timeline_positive_bio = f"Year_{site}_timeline_bioenvelope_positive"
    timeline_trending_bio = f"Year_{site}_timeline_bioenvelope_trending"

    if view_layer_name == "bioenvelope_positive":
        show_names = [
            timeline_positive_bio,
        ]
        hide_names = [
            top["base"],
            top["base_cubes"],
            top["positive"],
            top["priority"],
            top["trending"],
            top["bio_trending"],
            legacy["base"],
            legacy["timeline_base"],
            legacy["base_cubes"],
            cube_timeline_name,
            legacy["positive"],
            legacy["timeline_positive"],
            legacy["priority"],
            legacy["timeline_priority"],
            legacy["trending"],
            legacy["timeline_trending"],
            legacy["bio_positive"],
            legacy["bio_trending"],
            timeline_trending_bio,
        ]
        return show_names, hide_names

    show_names = [
        legacy["timeline_base"],
        legacy["timeline_positive"],
        legacy["timeline_priority"],
        legacy["timeline_trending"],
        timeline_positive_bio,
        timeline_trending_bio,
    ]
    hide_names = [
        top["base_cubes"],
        legacy["base"],
        legacy["base_cubes"],
        cube_timeline_name,
        legacy["positive"],
        legacy["priority"],
        legacy["trending"],
        legacy["bio_positive"],
        legacy["bio_trending"],
    ]
    return show_names, hide_names


def render_scene_view_layer(scene_name: str, spec: dict, view_layer_name: str, clipbox_module):
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' was not found")

    camera = bpy.data.objects.get(spec["camera"])
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"Camera '{spec['camera']}' was not found")

    if scene.view_layers.get(view_layer_name) is None:
        raise ValueError(f"View layer '{view_layer_name}' not found in scene '{scene_name}'")

    scene.camera = camera
    if clipbox_module is not None and hasattr(clipbox_module, "stamp_camera_proxy_from_live_clip"):
        clipbox_module.stamp_camera_proxy_from_live_clip(scene, camera)
    if clipbox_module is not None and hasattr(clipbox_module, "sync_scene_clipbox"):
        clipbox_module.sync_scene_clipbox(scene)

    scene.render.use_compositing = False
    scene.render.use_sequencer = False

    original_percentage = scene.render.resolution_percentage
    original_samples = getattr(scene.cycles, "samples", None) if hasattr(scene, "cycles") else None
    original_preview_samples = getattr(scene.cycles, "preview_samples", None) if hasattr(scene, "cycles") else None
    original_layer_use = {view_layer.name: view_layer.use for view_layer in scene.view_layers}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{scene_name}_{camera.name}_{view_layer_name}_timeline_pointcloud.png"
    scene.render.filepath = str(output_path)

    show_names, hide_names = build_scene_collection_toggles(spec["site"], view_layer_name)
    shown_state = set_collection_render_state(show_names, hide_render=False)
    hidden_state = set_collection_render_state(hide_names, hide_render=True)

    scene.render.resolution_percentage = PREVIEW_PERCENTAGE
    if hasattr(scene, "cycles"):
        scene.cycles.samples = PREVIEW_SAMPLES
        scene.cycles.preview_samples = PREVIEW_SAMPLES

    for view_layer in scene.view_layers:
        view_layer.use = view_layer.name == view_layer_name

    try:
        bpy.ops.render.render(
            write_still=True,
            scene=scene.name,
            layer=view_layer_name,
            use_viewport=False,
        )
    finally:
        scene.render.resolution_percentage = original_percentage
        if hasattr(scene, "cycles") and original_samples is not None:
            scene.cycles.samples = original_samples
            scene.cycles.preview_samples = original_preview_samples
        for view_layer in scene.view_layers:
            view_layer.use = original_layer_use.get(view_layer.name, True)
        restore_collection_render_state(shown_state)
        restore_collection_render_state(hidden_state)

    if not output_path.exists():
        raise RuntimeError(f"Render did not produce {output_path}")

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(
        f"RENDER_DONE scene={scene_name} layer={view_layer_name} "
        f"path={output_path} sha256={digest}"
    )
    return output_path


def main():
    print(f"Working blend: {bpy.data.filepath}")

    clipbox_setup = load_local_module(
        "b2026_clipbox_setup_timeline_render_runtime",
        "b2026_timeline_clipbox_setup.py",
    )
    clipbox_setup.main()

    camera_clipboxes = None
    if runtime_flags.any_clipbox_scripts_enabled():
        clipbox_setup.main()

        camera_clipboxes = load_local_module(
            "b2026_camera_clipboxes_timeline_render_runtime",
            "b2026_timeline_camera_clipboxes.py",
        )
        camera_clipboxes.register()
        if hasattr(camera_clipboxes, "sync_all_scene_clipboxes"):
            camera_clipboxes.sync_all_scene_clipboxes()

        clipbox_setup.main()

    outputs = []
    for scene_name, spec in RENDER_SPECS.items():
        for view_layer_name in TARGET_VIEW_LAYERS:
            outputs.append(
                render_scene_view_layer(
                    scene_name,
                    spec,
                    view_layer_name,
                    camera_clipboxes,
                )
            )

    print(f"Rendered {len(outputs)} timeline pointcloud previews")


if __name__ == "__main__":
    main()
