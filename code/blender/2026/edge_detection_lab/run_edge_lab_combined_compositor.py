from __future__ import annotations

import importlib.util
import os
import shutil
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


COMBINED_BLEND_PATH = env_path(
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


SCENE_BUILDERS = (
    ("AO", CODE_ROOT / "render_exr_ao_v2_blender.py", OUTPUT_ROOT / "ao"),
    ("Normals", CODE_ROOT / "render_exr_normals_v2_blender.py", OUTPUT_ROOT / "normals"),
    ("Resources", CODE_ROOT / "render_exr_arboreal_resource_fills_v1_blender.py", OUTPUT_ROOT / "resources"),
    ("DepthOutliner", CODE_ROOT / "render_exr_arboreal_depth_outliner_baseline_blender.py", OUTPUT_ROOT / "depth_outliner"),
)


def log(message: str) -> None:
    print(f"[run_edge_lab_combined_compositor] {message}")


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem.replace(".", "_"), module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_module(module, scene_output_dir: Path) -> None:
    module.OUTPUT_DIR = scene_output_dir
    module.BLEND_PATH = COMBINED_BLEND_PATH
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


def render_family_scene(scene_name: str, module_path: Path, scene_output_dir: Path) -> None:
    scene = bpy.data.scenes[scene_name]
    module = load_module(module_path)
    configure_module(module, scene_output_dir)
    rendered_paths = module.build_scene(scene)
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=False, scene=scene.name)
    for path in rendered_paths:
        module.rename_output(path)
    if scene_name == "DepthOutliner":
        node_tree = scene.node_tree
        module.render_socket_to_png(
            scene,
            node_tree,
            node_tree.nodes["pathway_edge_masked"].outputs["Image"],
            scene_output_dir / "pathway_depth_outliner.png",
        )
        module.render_socket_to_png(
            scene,
            node_tree,
            node_tree.nodes["priority_edge_masked"].outputs["Image"],
            scene_output_dir / "priority_depth_outliner.png",
        )
        for leftover in scene_output_dir.glob("*_0001.png"):
            leftover.unlink()
    log(f"Rendered {scene_name}")


def render_mist_scene() -> None:
    scene = bpy.data.scenes["MistOutlines"]
    bpy.context.window.scene = scene
    module = load_module(CODE_ROOT / "render_exr_arboreal_mist_variants_v2_blender.py")
    module.EXR_ROOT = EXR_ROOT
    module.OUTPUT_ROOT = OUTPUT_ROOT / "outlines_mist"
    module.BLEND_PATH = COMBINED_BLEND_PATH
    module.PATHWAY_EXR = PATHWAY_EXR
    module.PRIORITY_EXR = PRIORITY_EXR
    module.TRENDING_EXR = TRENDING_EXR
    module.PREP_ROOT = module.OUTPUT_ROOT / "_prep"
    module.VARIANT_PRESET = "kirschsizes"
    module.RENDER_MODE = "edges_only"
    module.VARIANT_SPECS = module.KIRSCHSIZE_VARIANTS
    module.WORKFLOW_ID = "blender_exr_arboreal_mist_kirschsizes_v1"
    module.WORKFLOW_NOTES = "Canonical mist-based arboreal outline export."

    prepared = module.build_prep_stage(scene)
    for scene_key, (visible_path, mist_visible_path) in prepared.items():
        if scene_key == "trending" and not TRENDING_EXR.exists():
            continue
        for spec in module.VARIANT_SPECS:
            module.build_variant_stage(scene, scene_key, spec, visible_path, mist_visible_path, module.OUTPUT_ROOT)
    if module.PREP_ROOT.exists():
        shutil.rmtree(module.PREP_ROOT)
    log("Rendered MistOutlines")


def main() -> None:
    if not COMBINED_BLEND_PATH.exists():
        raise FileNotFoundError(COMBINED_BLEND_PATH)
    bpy.ops.wm.open_mainfile(filepath=str(COMBINED_BLEND_PATH))

    for scene_name, module_path, scene_output_dir in SCENE_BUILDERS:
        render_family_scene(scene_name, module_path, scene_output_dir)
    render_mist_scene()

    bpy.ops.wm.save_as_mainfile(filepath=str(COMBINED_BLEND_PATH))
    log(f"Saved {COMBINED_BLEND_PATH}")


if __name__ == "__main__":
    main()
