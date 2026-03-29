from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
CODE_ROOT = REPO_ROOT / "code" / "blender" / "2026" / "edge_detection_lab"
DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "outputs" / "edge_lab_output_suite_baseline_20260329"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def preferred_exr(root: Path, canonical: str, legacy: str) -> Path:
    canonical_path = root / canonical
    legacy_path = root / legacy
    if canonical_path.exists():
        return canonical_path
    return legacy_path


OUTPUT_BLEND_PATH = env_path(
    "EDGE_LAB_COMBINED_BLEND_PATH",
    DATA_ROOT / "edge_lab_output_suite_combined.blend",
)
OUTPUT_ROOT = env_path("EDGE_LAB_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
EXR_ROOT = env_path(
    "EDGE_LAB_EXR_ROOT",
    REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6_baseline-city",
)
PATHWAY_EXR = env_path("EDGE_LAB_PATHWAY_EXR", preferred_exr(EXR_ROOT, "pathway_state.exr", "city-pathway_state.exr"))
PRIORITY_EXR = env_path("EDGE_LAB_PRIORITY_EXR", preferred_exr(EXR_ROOT, "priority.exr", "city-city_priority.exr"))
EXISTING_EXR = env_path("EDGE_LAB_EXISTING_EXR", preferred_exr(EXR_ROOT, "existing_condition.exr", "city-existing_condition.exr"))
TRENDING_EXR = env_path("EDGE_LAB_TRENDING_EXR", preferred_exr(EXR_ROOT, "trending_state.exr", "city-trending_state.exr"))
MIST_SOURCE_BLEND = env_path("EDGE_LAB_MIST_SOURCE_BLEND", OUTPUT_ROOT / "blends" / "outlines_mist.blend")


SCENE_BUILDERS = (
    ("AO", CODE_ROOT / "render_exr_ao_v2_blender.py", OUTPUT_ROOT / "ao"),
    ("Normals", CODE_ROOT / "render_exr_normals_v2_blender.py", OUTPUT_ROOT / "normals"),
    ("Resources", CODE_ROOT / "render_exr_arboreal_resource_fills_v1_blender.py", OUTPUT_ROOT / "resources"),
    ("DepthOutliner", CODE_ROOT / "render_exr_arboreal_depth_outliner_baseline_blender.py", OUTPUT_ROOT / "depth_outliner"),
)


def log(message: str) -> None:
    print(f"[build_edge_lab_combined_compositor] {message}")


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem.replace(".", "_"), module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clear_state() -> None:
    scratch = bpy.data.scenes[0]
    scratch.name = "Scratch"
    for scene in list(bpy.data.scenes)[1:]:
        bpy.data.scenes.remove(scene)
    for world in list(bpy.data.worlds):
        bpy.data.worlds.remove(world)


def create_scene(name: str) -> bpy.types.Scene:
    scene = bpy.data.scenes.new(name)
    scene.use_nodes = True
    return scene


def normalize_scene_color_management(scene: bpy.types.Scene) -> None:
    try:
        scene.display_settings.display_device = "sRGB"
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def configure_module(module, scene_output_dir: Path) -> None:
    module.OUTPUT_DIR = scene_output_dir
    module.BLEND_PATH = OUTPUT_BLEND_PATH
    if hasattr(module, "EXR_ROOT"):
        module.EXR_ROOT = EXR_ROOT
    if hasattr(module, "PATHWAY_EXR"):
        module.PATHWAY_EXR = PATHWAY_EXR
    if hasattr(module, "PRIORITY_EXR"):
        module.PRIORITY_EXR = PRIORITY_EXR
    if hasattr(module, "EXISTING_EXR"):
        module.EXISTING_EXR = EXISTING_EXR
    if hasattr(module, "TRENDING_EXR"):
        module.TRENDING_EXR = TRENDING_EXR


def build_scene_from_module(scene_name: str, module_path: Path, scene_output_dir: Path) -> None:
    module = load_module(module_path)
    scene = create_scene(scene_name)
    configure_module(module, scene_output_dir)
    module.build_scene(scene)
    normalize_scene_color_management(scene)
    scene["edge_lab_builder"] = module_path.name
    scene["edge_lab_output_dir"] = str(scene_output_dir)
    log(f"Built {scene_name} from {module_path.name}")


def append_mist_scene() -> None:
    if not MIST_SOURCE_BLEND.exists():
        scene = create_scene("MistOutlines")
        scene["edge_lab_builder"] = "pending_mist_refactor"
        log("Created placeholder MistOutlines scene; source blend not found.")
        return
    with bpy.data.libraries.load(str(MIST_SOURCE_BLEND), link=False) as (data_from, data_to):
        if "Scene" not in data_from.scenes:
            raise ValueError(f"'Scene' not found in {MIST_SOURCE_BLEND}")
        data_to.scenes = ["Scene"]
    appended_scene = data_to.scenes[0]
    if appended_scene is None:
        raise RuntimeError(f"Failed to append scene from {MIST_SOURCE_BLEND}")
    appended_scene.name = "MistOutlines"
    normalize_scene_color_management(appended_scene)
    appended_scene["edge_lab_builder"] = "mist_source_blend"
    appended_scene["edge_lab_output_dir"] = str(OUTPUT_ROOT / "outlines_mist")
    log(f"Appended MistOutlines from {MIST_SOURCE_BLEND}")


def order_scenes(scene_names: list[str]) -> None:
    window = bpy.context.window
    for name in scene_names:
        scene = bpy.data.scenes.get(name)
        if scene is not None:
            window.scene = scene


def purge_unused_data() -> None:
    for _ in range(5):
        result = bpy.ops.outliner.orphans_purge(
            do_local_ids=True,
            do_linked_ids=True,
            do_recursive=True,
        )
        if result != {"FINISHED"}:
            break


def main() -> None:
    clear_state()
    built_names: list[str] = []
    for scene_name, module_path, scene_output_dir in SCENE_BUILDERS:
        build_scene_from_module(scene_name, module_path, scene_output_dir)
        built_names.append(scene_name)

    append_mist_scene()
    built_names.append("MistOutlines")

    for scene in list(bpy.data.scenes):
        if scene.name not in set(built_names + ["Scratch"]):
            bpy.data.scenes.remove(scene)

    order_scenes(built_names)

    scratch = bpy.data.scenes.get("Scratch")
    if scratch is not None and len(bpy.data.scenes) > 1:
        bpy.data.scenes.remove(scratch)

    if built_names:
        bpy.context.window.scene = bpy.data.scenes[built_names[0]]

    purge_unused_data()

    OUTPUT_BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND_PATH))
    log(f"Saved {OUTPUT_BLEND_PATH}")


if __name__ == "__main__":
    main()
