from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract
import b2026_timeline_scene_setup as scene_setup


VALID_SITES = {"city", "trimmed-parade", "uni", "street"}
SITE_ALIASES = {
    "parade": "trimmed-parade",
    "street": "uni",
}


def normalize_site_name(raw_name: str) -> str:
    site = raw_name.strip().lower()
    site = SITE_ALIASES.get(site, site)
    if site not in VALID_SITES:
        raise ValueError(
            f"Unsupported site '{raw_name}'. Expected one of: {', '.join(sorted(VALID_SITES))}"
        )
    return site


def detect_scene_name() -> str:
    scene = bpy.context.scene
    if scene is None:
        raise ValueError("No active Blender scene is available.")
    return scene.name


def detect_site_name(scene_name: str) -> str:
    normalized = scene_name.strip().lower()
    return normalize_site_name(normalized)


def set_default_env(var_name: str, value: str) -> None:
    os.environ[var_name] = os.environ.get(var_name, value)


def configure_environment() -> tuple[str, str, int]:
    scene_name = os.environ.get("B2026_TARGET_SCENE_NAME", "").strip() or detect_scene_name()
    site_name = os.environ.get("B2026_TIMELINE_TARGET_SITES", "").strip()
    if not site_name:
        site_name = os.environ.get("B2026_SITE_KEY", "").strip()
    if not site_name:
        site_name = detect_site_name(scene_name)
    site_name = normalize_site_name(site_name)

    year = int(os.environ.get("B2026_SINGLE_STATE_YEAR", "180"))

    os.environ["B2026_TIMELINE_BUILD_MODE"] = "single_state"
    os.environ["B2026_SINGLE_STATE_YEAR"] = str(year)
    os.environ["B2026_TARGET_SCENE_NAME"] = scene_name
    os.environ["B2026_SCENE_NAME"] = scene_name
    os.environ["B2026_SITE_KEY"] = site_name
    os.environ["B2026_TIMELINE_TARGET_SITES"] = site_name

    return scene_name, site_name, year


def run_script(script_name: str) -> None:
    script_path = SCRIPT_DIR / script_name
    print(f"[single-state] running {script_path.name}")
    runpy.run_path(str(script_path), run_name="__main__")


def ensure_standard_view_layers(scene: bpy.types.Scene) -> None:
    scene_setup.ensure_view_layers(scene, scene_contract.SINGLE_STATE_VIEW_LAYERS)


def apply_single_state_view_layers(scene: bpy.types.Scene, site_name: str, year: int) -> None:
    ensure_standard_view_layers(scene)
    top = {
        role: scene_contract.get_single_state_top_level_name(site_name, role)
        for role in ("manager", "setup", "cameras", "positive", "priority", "trending")
    }
    positive_tree_log = [
        scene_contract.get_single_state_node_collection_name(site_name, node_type, year, "positive", collection_kind)
        for node_type in ("tree", "log")
        for collection_kind in ("positions", "plyModels")
    ]
    positive_ply_models = [
        scene_contract.get_single_state_node_collection_name(site_name, node_type, year, "positive", "plyModels")
        for node_type in ("tree", "log")
    ]
    priority_tree_log = [
        scene_contract.get_single_state_node_collection_name(
            site_name, node_type, year, "positive", collection_kind, priority=True
        )
        for node_type in ("tree", "log")
        for collection_kind in ("positions", "plyModels")
    ]
    priority_ply_models = [
        scene_contract.get_single_state_node_collection_name(
            site_name, node_type, year, "positive", "plyModels", priority=True
        )
        for node_type in ("tree", "log")
    ]
    trending_tree_log = [
        scene_contract.get_single_state_node_collection_name(site_name, node_type, year, "trending", collection_kind)
        for node_type in ("tree", "log")
        for collection_kind in ("positions", "plyModels")
    ]
    trending_ply_models = [
        scene_contract.get_single_state_node_collection_name(site_name, node_type, year, "trending", "plyModels")
        for node_type in ("tree", "log")
    ]
    positive_envelope_collection = scene_contract.get_single_state_envelope_collection_name(site_name, "positive")
    trending_envelope_collection = scene_contract.get_single_state_envelope_collection_name(site_name, "trending")

    all_top = [top["manager"], top["setup"], top["cameras"], top["positive"], top["priority"], top["trending"]]

    per_layer = {
        "existing_condition": {
            "hide_top": [top["manager"], top["setup"], top["cameras"], top["priority"], top["trending"]],
            "hide_children": [*positive_tree_log, positive_envelope_collection],
        },
        "pathway_state": {
            "hide_top": [top["manager"], top["setup"], top["cameras"], top["priority"], top["trending"]],
            "hide_children": [positive_envelope_collection],
        },
        "priority_state": {
            "hide_top": [top["manager"], top["setup"], top["cameras"], top["positive"], top["trending"]],
            "hide_children": priority_ply_models,
        },
        "trending_state": {
            "hide_top": [top["manager"], top["setup"], top["cameras"], top["positive"], top["priority"]],
            "hide_children": [trending_envelope_collection],
        },
        "existing_condition_trending": {
            "hide_top": [top["manager"], top["setup"], top["cameras"], top["positive"], top["priority"]],
            "hide_children": [*trending_tree_log, trending_envelope_collection],
        },
        "bioenvelope_positive": {
            "hide_top": [top["manager"], top["setup"], top["cameras"], top["priority"], top["trending"]],
            "hide_children": positive_tree_log,
        },
        "bioenvelope_trending": {
            "hide_top": [top["manager"], top["setup"], top["cameras"], top["positive"], top["priority"]],
            "hide_children": trending_tree_log,
        },
    }

    for view_layer in scene.view_layers:
        scene_setup.reset_view_layer_excludes(view_layer)
        spec = per_layer.get(view_layer.name)
        if spec is None:
            scene_setup.set_excluded(view_layer, all_top, excluded=False)
            continue
        scene_setup.set_excluded(view_layer, spec["hide_top"], excluded=True)
        scene_setup.set_excluded(view_layer, spec["hide_children"], excluded=True)
        # Extra safety for unrelated tree/log branches.
        if view_layer.name not in {"pathway_state", "existing_condition", "bioenvelope_positive"}:
            scene_setup.set_excluded(view_layer, positive_tree_log, excluded=True)
        if view_layer.name in {"pathway_state", "existing_condition", "bioenvelope_positive"}:
            scene_setup.set_excluded(view_layer, positive_ply_models, excluded=False)
        if view_layer.name != "priority_state":
            scene_setup.set_excluded(view_layer, priority_tree_log, excluded=True)
        if view_layer.name == "priority_state":
            scene_setup.set_excluded(view_layer, priority_ply_models, excluded=True)
        if view_layer.name not in {"trending_state", "bioenvelope_trending"}:
            scene_setup.set_excluded(view_layer, trending_tree_log, excluded=True)
        if view_layer.name in {"trending_state", "bioenvelope_trending"}:
            scene_setup.set_excluded(view_layer, trending_ply_models, excluded=False)
        if view_layer.name == "existing_condition_trending":
            scene_setup.set_excluded(view_layer, trending_ply_models, excluded=True)
        if view_layer.name != "bioenvelope_positive":
            scene_setup.set_excluded(view_layer, [positive_envelope_collection], excluded=True)
        if view_layer.name != "bioenvelope_trending":
            scene_setup.set_excluded(view_layer, [trending_envelope_collection], excluded=True)


def main() -> None:
    scene_name, site_name, year = configure_environment()
    print(
        f"[single-state] scene={scene_name} site={site_name} year={year} "
        f"blend={bpy.data.filepath or '<unsaved>'}"
    )

    # Run in build order so generated collections/objects are present before attrs/materials.
    run_script("b2026_timeline_instancer.py")
    run_script("b2026_timeline_bioenvelopes.py")
    run_script("b2026_timeline_rebuild_world_year_attrs.py")

    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' not found after single-state generation")
    apply_single_state_view_layers(scene, site_name, year)

    bpy.ops.wm.save_mainfile()
    print("[single-state] complete and blend saved")


if __name__ == "__main__":
    main()
