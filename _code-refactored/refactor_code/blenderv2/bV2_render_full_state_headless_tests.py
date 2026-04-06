"""Headless full-state renders for bV2 test blends.

This utility:
- opens an existing bV2 test blend in headless mode
- ensures instancers, world, and bioenvelopes use their live materials
- builds bioenvelopes into the scene if needed
- renders one PNG per view layer using the scene camera
- writes flat 1080p outputs into one case folder
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import bpy


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[3]
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
SITE_LABELS = {
    "trimmed-parade": "parade",
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


def get_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable {name}")
    return value


def get_runtime_case() -> dict[str, object]:
    site = get_required_env("BV2_TEST_SITE")
    mode = get_required_env("BV2_TEST_MODE")
    year_raw = os.environ.get("BV2_TEST_YEAR", "").strip()
    year = int(year_raw) if year_raw else None
    source_blend = Path(get_required_env("BV2_TEST_SOURCE_BLEND"))
    output_root = Path(get_required_env("BV2_TEST_CASE_OUTPUT_ROOT"))
    output_root.mkdir(parents=True, exist_ok=True)

    site_label = SITE_LABELS.get(site, site)
    if mode == "timeline":
        case_tag = f"{site_label}_timeline"
    else:
        case_tag = f"{site_label}_single-state_yr{year}"

    return {
        "site": site,
        "mode": mode,
        "year": year,
        "source_blend": source_blend,
        "output_root": output_root,
        "case_tag": case_tag,
    }


def log(*args) -> None:
    print("[BV2_FULL_HEADLESS]", *args, flush=True)


def open_blend(filepath: Path) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(filepath), load_ui=False)


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


def find_collection_by_role(scene: bpy.types.Scene, role: str) -> bpy.types.Collection:
    queue = list(scene.collection.children)
    while queue:
        collection = queue.pop(0)
        role_name = str(collection.get("bV2_role", collection.name.split("::")[-1]))
        if role_name == role:
            return collection
        queue.extend(collection.children)
    raise RuntimeError(f"Could not find collection for role {role!r}")


def set_object_material(obj: bpy.types.Object, material_name: str) -> None:
    material = bpy.data.materials.get(material_name)
    if material is None or obj.type != "MESH":
        return
    mesh = obj.data
    if mesh is None:
        return
    if len(mesh.materials) == 0:
        mesh.materials.append(material)
    else:
        for index in range(len(mesh.materials)):
            mesh.materials[index] = material


def set_modifier_material(obj: bpy.types.Object, material_name: str) -> None:
    material = bpy.data.materials.get(material_name)
    if material is None:
        return
    for modifier in obj.modifiers:
        if modifier.type != "NODES":
            continue
        node_group = getattr(modifier, "node_group", None)
        if node_group is None:
            continue
        for node in node_group.nodes:
            if node.type == "SET_MATERIAL" and "Material" in node.inputs:
                node.inputs["Material"].default_value = material


def apply_live_materials_to_world(scene: bpy.types.Scene) -> None:
    material_name = "v2WorldAOV"
    for role in ("world_positive_attributes", "world_trending_attributes"):
        collection = find_collection_by_role(scene, role)
        for obj in collection.objects:
            if obj.type != "MESH":
                continue
            set_object_material(obj, material_name)
            set_modifier_material(obj, material_name)


def apply_live_materials_to_instancers(scene: bpy.types.Scene) -> None:
    material_name = "MINIMAL_RESOURCES"
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if obj.get("bV2_source_ply") or "_models_" in obj.name or obj.name.startswith("instanceID."):
            set_object_material(obj, material_name)
        if "_positions_" in obj.name:
            set_modifier_material(obj, material_name)


def apply_live_materials_to_bioenvelopes(scene: bpy.types.Scene) -> None:
    material_name = "Envelope"
    for role in ("bioenvelope_positive", "bioenvelope_trending"):
        collection = find_collection_by_role(scene, role)
        for obj in collection.objects:
            if obj.type != "MESH":
                continue
            set_object_material(obj, material_name)
            set_modifier_material(obj, material_name)


def configure_headless_cycles(scene: bpy.types.Scene) -> None:
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        scene.cycles.preview_samples = 1
        try:
            scene.cycles.device = "CPU"
        except Exception:
            pass


def save_scene_copy(scene: bpy.types.Scene, output_root: Path, case_tag: str) -> Path:
    blend_path = output_root / f"{case_tag}__full_live.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True)
    return blend_path


def render_all_view_layers(scene: bpy.types.Scene, output_root: Path, case_tag: str) -> list[Path]:
    outputs: list[Path] = []
    for view_layer in scene.view_layers:
        scene.render.filepath = str(output_root / f"{case_tag}__{view_layer.name}.png")
        bpy.ops.render.render(write_still=True, scene=scene.name, layer=view_layer.name)
        outputs.append(Path(scene.render.filepath))
        log("rendered", view_layer.name, "->", scene.render.filepath)
    return outputs


def write_manifest(
    scene: bpy.types.Scene,
    output_root: Path,
    case_tag: str,
    rendered_paths: list[Path],
    blend_path: Path,
) -> None:
    manifest_path = output_root / f"{case_tag}__manifest.txt"
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


def run_case() -> None:
    os.environ.pop("BV2_INSTANCER_DISPLAY_MODE", None)
    os.environ.pop("BV2_BIOENVELOPE_DISPLAY_MODE", None)
    os.environ.pop("BV2_DISPLAY_MODE", None)

    runtime = get_runtime_case()
    build_bioenvelopes = load_local_module("bV2_build_bioenvelopes_full_headless", "bV2_build_bioenvelopes.py")

    open_blend(runtime["source_blend"])
    scene = find_runtime_scene(runtime["site"], runtime["mode"], runtime["year"])

    build_bioenvelopes.build_bioenvelopes(scene)
    apply_live_materials_to_world(scene)
    apply_live_materials_to_instancers(scene)
    apply_live_materials_to_bioenvelopes(scene)
    configure_headless_cycles(scene)

    blend_path = save_scene_copy(scene, runtime["output_root"], runtime["case_tag"])
    rendered_paths = render_all_view_layers(scene, runtime["output_root"], runtime["case_tag"])
    write_manifest(scene, runtime["output_root"], runtime["case_tag"], rendered_paths, blend_path)
    log("case complete", scene.name, "renders=", len(rendered_paths))


if __name__ == "__main__":
    try:
        run_case()
    except Exception as exc:
        log("FAILED", type(exc).__name__, exc)
        import traceback

        traceback.print_exc()
        raise
