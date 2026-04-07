from __future__ import annotations

from pathlib import Path
import os
import sys

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_build_template_from_single_state as timeline_template
import b2026_timeline_scene_setup as timeline_scene_setup
import b2026_unified_scene_contract as unified_contract


def detect_site() -> str:
    raw = os.environ.get("B2026_SITE_KEY", "").strip()
    if raw in unified_contract.SITE_CONTRACTS:
        return raw
    scene_name = os.environ.get("B2026_SCENE_NAME", "").strip()
    inferred = unified_contract.infer_site_from_scene_name(scene_name)
    if inferred is not None:
        return inferred
    return "city"


def detect_scene_name(site: str) -> str:
    return os.environ.get("B2026_SCENE_NAME", "").strip() or unified_contract.SITE_CONTRACTS[site]["scene_name"]


def default_output_path(source_blend: Path, build_mode: str) -> Path:
    stem = source_blend.stem
    if build_mode == "timeline":
        candidate_stem = stem.replace("single-state template", "timeline template").strip()
        if candidate_stem == stem:
            candidate_stem = f"{stem} timeline template"
        return source_blend.with_name(f"{candidate_stem}.blend")

    candidate_stem = stem.replace("single-state template", "single-state shell").strip()
    if candidate_stem == stem:
        candidate_stem = f"{stem} single-state shell"
    return source_blend.with_name(f"{candidate_stem}.blend")


def get_source_blend(site: str) -> Path:
    return timeline_template.get_source_blend(site)


def get_output_blend(source_blend: Path, build_mode: str) -> Path:
    env_path = os.environ.get("B2026_OUTPUT_BLEND", "").strip()
    if env_path:
        return Path(env_path)
    return default_output_path(source_blend, build_mode)


def get_reference_blend(site: str) -> Path:
    return timeline_template.get_reference_blend(site)


def prepare_single_state_template(scene: bpy.types.Scene, site: str, reference_blend: Path) -> None:
    timeline_scene_setup.ensure_single_state_shell(scene, site)
    timeline_scene_setup.ensure_view_layers(
        scene,
        unified_contract.STANDARD_VIEW_LAYERS,
        remove_layers=unified_contract.LEGACY_TIMELINE_ALIAS_VIEW_LAYERS,
    )

    material_names = ["WORLD_AOV", "Envelope"]
    if site == "trimmed-parade":
        material_names.append("Envelope Parade")
    timeline_template.append_materials(reference_blend, material_names)

    contract = unified_contract.SITE_CONTRACTS[site]
    for object_name in contract["world_objects"].values():
        timeline_template.set_single_material(object_name, "WORLD_AOV")

    world = bpy.data.worlds.get("debug_timeslice_world")
    if world is not None:
        scene.world = world

    timeline_template.ensure_timeslice_cameras(scene, site)


def main() -> None:
    build_mode = unified_contract.get_build_mode()
    site = detect_site()
    scene_name = detect_scene_name(site)
    source_blend = get_source_blend(site)
    output_blend = get_output_blend(source_blend, build_mode)
    reference_blend = get_reference_blend(site)

    bpy.ops.wm.open_mainfile(filepath=str(source_blend))
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' not found in {source_blend}")

    if build_mode == "timeline":
        timeline_template.prepare_timeline_template(scene, site, reference_blend)
    else:
        prepare_single_state_template(scene, site, reference_blend)

    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    print(f"[unified-template] mode={build_mode} site={site} source={source_blend}")
    print(f"[unified-template] saved={output_blend}")


if __name__ == "__main__":
    main()
