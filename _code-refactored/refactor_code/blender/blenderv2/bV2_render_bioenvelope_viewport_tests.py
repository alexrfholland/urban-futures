"""Render viewport previews for bV2 instancer + bioenvelope debug verification.

This utility:
- opens the bV2 template with UI disabled
- builds one target scene shell
- builds instancers in `debug-source-years` display mode
- builds bioenvelopes in `debug-source-years` display mode
- saves a viewport render image for every view layer using the scene camera

It is intended to be run in Blender GUI mode, one process per case.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import bpy


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
CODE_ROOT = next(parent for parent in SCRIPT_PATH.parents if parent.name == "_code-refactored")
REPO_ROOT = CODE_ROOT.parent
TEMPLATE_BLEND = REPO_ROOT / "_data-refactored" / "blenderv2" / "bV2_template.blend"
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
SITE_LABELS = {
    "trimmed-parade": "parade",
}


def inject_repo_venv_site_packages() -> None:
    candidates = [
        REPO_ROOT / ".venv" / "Lib" / "site-packages",
    ]
    candidates.extend((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))
    for candidate in candidates:
        if candidate.exists():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return
    raise RuntimeError("Could not find repo .venv site-packages")


def load_local_module(module_name: str, filename: str):
    file_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable {name}")
    return value


def get_optional_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def get_runtime_case() -> dict[str, object]:
    site = get_required_env("BV2_TEST_SITE")
    mode = get_required_env("BV2_TEST_MODE")
    year_raw = os.environ.get("BV2_TEST_YEAR", "").strip()
    year = int(year_raw) if year_raw else None
    case_output_root = Path(get_required_env("BV2_TEST_CASE_OUTPUT_ROOT"))
    case_output_root.mkdir(parents=True, exist_ok=True)
    source_blend_raw = get_optional_env("BV2_TEST_SOURCE_BLEND")
    site_label = SITE_LABELS.get(site, site)
    if mode == "timeline":
        case_tag = f"{site_label}_timeline"
    else:
        case_tag = f"{site_label}_single-state_yr{year}"
    return {
        "site": site,
        "mode": mode,
        "year": year,
        "case_output_root": case_output_root,
        "case_tag": case_tag,
        "source_blend": Path(source_blend_raw) if source_blend_raw else None,
    }


def log(*args) -> None:
    print("[BV2_VIEWPORT_TEST]", *args, flush=True)


def open_blend(filepath: Path) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(filepath), load_ui=False)


def find_view3d_context():
    window = bpy.context.window_manager.windows[0]
    screen = window.screen
    area = next(area for area in screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    space = next(space for space in area.spaces if space.type == "VIEW_3D")
    return window, screen, area, region, space


def configure_viewport_for_material_preview(scene: bpy.types.Scene) -> tuple:
    window, screen, area, region, space = find_view3d_context()
    window.scene = scene
    if scene.camera is not None:
        space.region_3d.view_perspective = "CAMERA"
    space.shading.type = "MATERIAL"
    if hasattr(space, "overlay"):
        space.overlay.show_overlays = False
    if hasattr(space.shading, "use_scene_lights"):
        space.shading.use_scene_lights = True
    if hasattr(space.shading, "use_scene_world"):
        space.shading.use_scene_world = True
    if hasattr(space, "show_gizmo"):
        space.show_gizmo = False
    return window, screen, area, region, space


def save_scene_copy(scene: bpy.types.Scene, case_output_root: Path, case_tag: str) -> Path:
    blend_path = case_output_root / f"{case_tag}__instancers_bioenv_debug.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True)
    return blend_path


def render_all_view_layers(scene: bpy.types.Scene, case_output_root: Path, case_tag: str) -> list[Path]:
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    window, screen, area, region, space = configure_viewport_for_material_preview(scene)

    outputs: list[Path] = []
    for view_layer in scene.view_layers:
        output_path = case_output_root / f"{case_tag}__{view_layer.name}.png"
        window.view_layer = view_layer
        scene.render.filepath = str(output_path)
        with bpy.context.temp_override(
            window=window,
            screen=screen,
            area=area,
            region=region,
            scene=scene,
            view_layer=view_layer,
            space_data=space,
        ):
            bpy.ops.render.opengl(write_still=True, view_context=True)
        outputs.append(output_path)
        log("rendered", view_layer.name, "->", output_path)
    return outputs


def write_manifest(
    scene: bpy.types.Scene,
    case_output_root: Path,
    case_tag: str,
    rendered_paths: list[Path],
    blend_path: Path,
) -> None:
    manifest_path = case_output_root / f"{case_tag}__manifest.txt"
    lines = [
        f"case={case_tag}",
        f"scene={scene.name}",
        f"camera={scene.camera.name if scene.camera else ''}",
        f"site={scene.get('bV2_site', '')}",
        f"mode={scene.get('bV2_mode', '')}",
        f"year={scene.get('bV2_year', '')}",
        f"blend={blend_path}",
        "renders:",
    ]
    lines.extend(str(path) for path in rendered_paths)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_runtime_scene(site: str, mode: str, year: int | None) -> bpy.types.Scene:
    for scene in bpy.data.scenes:
        if str(scene.get("bV2_site", "")).strip() != site:
            continue
        if str(scene.get("bV2_mode", "")).strip() != mode:
            continue
        scene_year_raw = str(scene.get("bV2_year", "")).strip()
        scene_year = int(scene_year_raw) if scene_year_raw else None
        if scene_year == year:
            return scene
    if bpy.context.scene is None:
        raise RuntimeError("No active scene after opening source blend")
    return bpy.context.scene


def apply_debug_source_years_to_instancers(scene: bpy.types.Scene) -> None:
    debug_material = bpy.data.materials.get("debug-source-years")
    if debug_material is None:
        raise RuntimeError("Could not find required debug-source-years material")

    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if obj.get("bV2_source_ply") or "_models_" in obj.name:
            mesh = obj.data
            if mesh is None:
                continue
            if len(mesh.materials) == 0:
                mesh.materials.append(debug_material)
            else:
                for index in range(len(mesh.materials)):
                    mesh.materials[index] = debug_material

        if "_positions_" not in obj.name:
            continue
        for modifier in obj.modifiers:
            if modifier.type != "NODES":
                continue
            node_group = getattr(modifier, "node_group", None)
            if node_group is None:
                continue
            for node in node_group.nodes:
                if node.type == "SET_MATERIAL" and "Material" in node.inputs:
                    node.inputs["Material"].default_value = debug_material


def run_case() -> None:
    os.environ["BV2_INSTANCER_DISPLAY_MODE"] = "debug-source-years"
    os.environ["BV2_BIOENVELOPE_DISPLAY_MODE"] = "debug-source-years"

    runtime = get_runtime_case()
    build_bioenvelopes = load_local_module("bV2_build_bioenvelopes_viewport_test", "bV2_build_bioenvelopes.py")
    source_blend = runtime["source_blend"]
    if source_blend is not None:
        open_blend(source_blend)
        scene = find_runtime_scene(runtime["site"], runtime["mode"], runtime["year"])
    else:
        inject_repo_venv_site_packages()
        open_blend(TEMPLATE_BLEND)
        init_scene = load_local_module("bV2_init_scene_viewport_test", "bV2_init_scene.py")
        build_instancers = load_local_module("bV2_build_instancers_viewport_test", "bV2_build_instancers.py")
        scene = init_scene.init_scene(
            runtime["site"],
            runtime["mode"],
            runtime["year"],
        )
        build_instancers.build_instancers(scene)

    window = bpy.context.window_manager.windows[0]
    window.scene = scene
    apply_debug_source_years_to_instancers(scene)
    build_bioenvelopes.build_bioenvelopes(scene)

    blend_path = save_scene_copy(scene, runtime["case_output_root"], runtime["case_tag"])
    rendered_paths = render_all_view_layers(scene, runtime["case_output_root"], runtime["case_tag"])
    write_manifest(scene, runtime["case_output_root"], runtime["case_tag"], rendered_paths, blend_path)
    log("case complete", scene.name, "renders=", len(rendered_paths))
    os._exit(0)


def timer_entry():
    try:
        run_case()
    except Exception as exc:
        log("FAILED", type(exc).__name__, exc)
        import traceback

        traceback.print_exc()
        os._exit(1)
    return None


bpy.app.timers.register(timer_entry, first_interval=0.5)
