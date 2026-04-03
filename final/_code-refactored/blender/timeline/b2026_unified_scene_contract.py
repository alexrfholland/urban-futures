from __future__ import annotations

from pathlib import Path
import os
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as timeline_contract


STANDARD_VIEW_LAYERS = timeline_contract.STANDARD_VIEW_LAYERS
SINGLE_STATE_VIEW_LAYERS = timeline_contract.SINGLE_STATE_VIEW_LAYERS
SITE_CONTRACTS = timeline_contract.SITE_CONTRACTS
SINGLE_STATE_TOP_LEVEL_ROLES = timeline_contract.SINGLE_STATE_TOP_LEVEL_ROLES
LEGACY_TIMELINE_ALIAS_VIEW_LAYERS = (
    "city_priority",
    "city_bioenvelope",
)
VALID_BUILD_MODES = ("timeline", "single_state")
PUBLIC_SCRIPT_FAMILY = (
    "b2026_unified_scene_contract.py",
    "b2026_unified_scene_setup.py",
    "b2026_unified_build_template.py",
    "b2026_unified_build_instancers.py",
    "b2026_unified_build_bioenvelopes.py",
    "b2026_unified_build_world.py",
    "b2026_unified_setup_render.py",
    "b2026_unified_render_exrs.py",
    "b2026_unified_render_workbench_view_layers.py",
    "b2026_unified_build_scene.py",
    "b2026_unified_validate_scene.py",
)
VIEW_LAYER_SEMANTICS = {
    "existing_condition": {
        "world": "positive",
        "instancers": None,
        "envelope": None,
    },
    "pathway_state": {
        "world": "positive",
        "instancers": "positive",
        "envelope": None,
    },
    "priority_state": {
        "world": None,
        "instancers": "priority",
        "envelope": None,
    },
    "existing_condition_trending": {
        "world": "trending",
        "instancers": None,
        "envelope": None,
    },
    "trending_state": {
        "world": "trending",
        "instancers": "trending",
        "envelope": None,
    },
    "bioenvelope_positive": {
        "world": "positive",
        "instancers": None,
        "envelope": "positive",
    },
    "bioenvelope_trending": {
        "world": "trending",
        "instancers": None,
        "envelope": "trending",
    },
}


infer_site_from_scene_name = timeline_contract.infer_site_from_scene_name
get_site_contract = timeline_contract.get_site_contract
get_collection_name = timeline_contract.get_collection_name
get_priority_view_layer_names = timeline_contract.get_priority_view_layer_names
get_positive_bioenvelope_view_layer_names = timeline_contract.get_positive_bioenvelope_view_layer_names
get_single_state_top_level_name = timeline_contract.get_single_state_top_level_name
get_single_state_world_object_name = timeline_contract.get_single_state_world_object_name
get_single_state_node_collection_base = timeline_contract.get_single_state_node_collection_base
get_single_state_node_collection_name = timeline_contract.get_single_state_node_collection_name
get_single_state_envelope_object_name = timeline_contract.get_single_state_envelope_object_name
get_single_state_envelope_collection_name = timeline_contract.get_single_state_envelope_collection_name
get_timeline_world_collection_name = timeline_contract.get_timeline_world_collection_name
get_timeline_world_object_name = timeline_contract.get_timeline_world_object_name


def get_build_mode(default: str = "timeline") -> str:
    mode = os.environ.get("B2026_TIMELINE_BUILD_MODE", default).strip().lower()
    if mode not in VALID_BUILD_MODES:
        raise ValueError(
            f"Unknown B2026_TIMELINE_BUILD_MODE '{mode}'. "
            f"Expected one of: {', '.join(VALID_BUILD_MODES)}"
        )
    return mode


def get_single_state_collection_contract(site: str, year: int) -> dict[str, object]:
    top = {
        role: get_single_state_top_level_name(site, role)
        for role in SINGLE_STATE_TOP_LEVEL_ROLES
    }
    positive_tree_log = tuple(
        get_single_state_node_collection_name(site, node_type, year, "positive", collection_kind)
        for node_type in ("tree", "log")
        for collection_kind in ("positions", "plyModels")
    )
    positive_ply_models = tuple(
        get_single_state_node_collection_name(site, node_type, year, "positive", "plyModels")
        for node_type in ("tree", "log")
    )
    priority_tree_log = tuple(
        get_single_state_node_collection_name(
            site,
            node_type,
            year,
            "positive",
            collection_kind,
            priority=True,
        )
        for node_type in ("tree", "log")
        for collection_kind in ("positions", "plyModels")
    )
    priority_ply_models = tuple(
        get_single_state_node_collection_name(
            site,
            node_type,
            year,
            "positive",
            "plyModels",
            priority=True,
        )
        for node_type in ("tree", "log")
    )
    trending_tree_log = tuple(
        get_single_state_node_collection_name(site, node_type, year, "trending", collection_kind)
        for node_type in ("tree", "log")
        for collection_kind in ("positions", "plyModels")
    )
    trending_ply_models = tuple(
        get_single_state_node_collection_name(site, node_type, year, "trending", "plyModels")
        for node_type in ("tree", "log")
    )
    return {
        "top": top,
        "all_top": tuple(top.values()),
        "positive_tree_log": positive_tree_log,
        "positive_ply_models": positive_ply_models,
        "priority_tree_log": priority_tree_log,
        "priority_ply_models": priority_ply_models,
        "trending_tree_log": trending_tree_log,
        "trending_ply_models": trending_ply_models,
        "positive_envelope_collection": get_single_state_envelope_collection_name(site, "positive"),
        "trending_envelope_collection": get_single_state_envelope_collection_name(site, "trending"),
    }


def get_single_state_view_layer_expectations(site: str, year: int) -> dict[str, dict[str, bool]]:
    names = get_single_state_collection_contract(site, year)
    candidate_names = {
        *names["all_top"],
        *names["positive_tree_log"],
        *names["priority_tree_log"],
        *names["trending_tree_log"],
        names["positive_envelope_collection"],
        names["trending_envelope_collection"],
    }

    per_layer = {
        "existing_condition": {
            "hide_top": [
                names["top"]["manager"],
                names["top"]["setup"],
                names["top"]["cameras"],
                names["top"]["priority"],
                names["top"]["trending"],
            ],
            "hide_children": [*names["positive_tree_log"], names["positive_envelope_collection"]],
        },
        "pathway_state": {
            "hide_top": [
                names["top"]["manager"],
                names["top"]["setup"],
                names["top"]["cameras"],
                names["top"]["priority"],
                names["top"]["trending"],
            ],
            "hide_children": [names["positive_envelope_collection"]],
        },
        "priority_state": {
            "hide_top": [
                names["top"]["manager"],
                names["top"]["setup"],
                names["top"]["cameras"],
                names["top"]["positive"],
                names["top"]["trending"],
            ],
            "hide_children": [*names["priority_ply_models"]],
        },
        "existing_condition_trending": {
            "hide_top": [
                names["top"]["manager"],
                names["top"]["setup"],
                names["top"]["cameras"],
                names["top"]["positive"],
                names["top"]["priority"],
            ],
            "hide_children": [*names["trending_tree_log"], names["trending_envelope_collection"]],
        },
        "bioenvelope_positive": {
            "hide_top": [
                names["top"]["manager"],
                names["top"]["setup"],
                names["top"]["cameras"],
                names["top"]["priority"],
                names["top"]["trending"],
            ],
            "hide_children": [*names["positive_tree_log"]],
        },
        "bioenvelope_trending": {
            "hide_top": [
                names["top"]["manager"],
                names["top"]["setup"],
                names["top"]["cameras"],
                names["top"]["positive"],
                names["top"]["priority"],
            ],
            "hide_children": [*names["trending_tree_log"]],
        },
        "trending_state": {
            "hide_top": [
                names["top"]["manager"],
                names["top"]["setup"],
                names["top"]["cameras"],
                names["top"]["positive"],
                names["top"]["priority"],
            ],
            "hide_children": [names["trending_envelope_collection"]],
        },
    }

    expected_by_layer: dict[str, dict[str, bool]] = {}
    for view_layer_name in STANDARD_VIEW_LAYERS:
        expected = {name: False for name in candidate_names}
        spec = per_layer[view_layer_name]
        for collection_name in spec["hide_top"]:
            expected[collection_name] = True
        for collection_name in spec["hide_children"]:
            expected[collection_name] = True

        if view_layer_name not in {"pathway_state", "existing_condition", "bioenvelope_positive"}:
            for collection_name in names["positive_tree_log"]:
                expected[collection_name] = True
        if view_layer_name in {"pathway_state", "existing_condition", "bioenvelope_positive"}:
            for collection_name in names["positive_ply_models"]:
                expected[collection_name] = False
        if view_layer_name != "priority_state":
            for collection_name in names["priority_tree_log"]:
                expected[collection_name] = True
        if view_layer_name == "priority_state":
            for collection_name in names["priority_ply_models"]:
                expected[collection_name] = True
        if view_layer_name not in {"trending_state", "bioenvelope_trending"}:
            for collection_name in names["trending_tree_log"]:
                expected[collection_name] = True
        if view_layer_name in {"trending_state", "bioenvelope_trending"}:
            for collection_name in names["trending_ply_models"]:
                expected[collection_name] = False
        if view_layer_name == "existing_condition_trending":
            for collection_name in names["trending_ply_models"]:
                expected[collection_name] = True
        if view_layer_name != "bioenvelope_positive":
            expected[names["positive_envelope_collection"]] = True
        if view_layer_name != "bioenvelope_trending":
            expected[names["trending_envelope_collection"]] = True

        expected_by_layer[view_layer_name] = expected
    return expected_by_layer


def get_timeline_collection_contract(site: str) -> dict[str, object]:
    contract = SITE_CONTRACTS[site]
    top = contract["top_level"]
    legacy = contract["legacy"]
    timeline_cube = f"{legacy['base_cubes']}_Timeline"
    timeline_positive_bio = f"Year_{site}_timeline_bioenvelope_positive"
    timeline_trending_bio = f"Year_{site}_timeline_bioenvelope_trending"
    timeline_positive_world = get_timeline_world_collection_name(site, "positive")
    timeline_trending_world = get_timeline_world_collection_name(site, "trending")
    child_defaults = {
        legacy["base"]: True,
        legacy["timeline_base"]: False,
        timeline_positive_world: True,
        timeline_trending_world: True,
        legacy["base_cubes"]: True,
        timeline_cube: True,
        legacy["bio_positive"]: True,
        timeline_positive_bio: True,
        legacy["bio_trending"]: True,
        timeline_trending_bio: True,
        legacy["timeline_positive"]: True,
        legacy["timeline_priority"]: True,
        legacy["timeline_trending"]: True,
    }
    return {
        "top": top,
        "legacy": legacy,
        "timeline_cube": timeline_cube,
        "timeline_positive_bio": timeline_positive_bio,
        "timeline_trending_bio": timeline_trending_bio,
        "timeline_positive_world": timeline_positive_world,
        "timeline_trending_world": timeline_trending_world,
        "all_top": (
            top["manager"],
            top["base"],
            top["base_cubes"],
            top["cameras"],
            top["bio_positive"],
            top["bio_trending"],
            top["positive"],
            top["priority"],
            top["trending"],
        ),
        "child_defaults": child_defaults,
    }


def get_timeline_view_layer_expectations(site: str) -> dict[str, dict[str, bool]]:
    names = get_timeline_collection_contract(site)
    top = names["top"]
    legacy = names["legacy"]

    visible_by_layer = {
        "existing_condition": {
            top["base"],
            legacy["timeline_base"],
            names["timeline_positive_world"],
        },
        "pathway_state": {
            top["base"],
            legacy["timeline_base"],
            names["timeline_positive_world"],
            top["positive"],
            legacy["timeline_positive"],
        },
        "priority_state": {
            top["priority"],
            legacy["timeline_priority"],
        },
        "existing_condition_trending": {
            top["base"],
            legacy["timeline_base"],
            names["timeline_trending_world"],
        },
        "trending_state": {
            top["base"],
            legacy["timeline_base"],
            names["timeline_trending_world"],
            top["trending"],
            legacy["timeline_trending"],
        },
        "bioenvelope_positive": {
            top["base"],
            legacy["timeline_base"],
            names["timeline_positive_world"],
            top["bio_positive"],
            names["timeline_positive_bio"],
        },
        "bioenvelope_trending": {
            top["base"],
            legacy["timeline_base"],
            names["timeline_trending_world"],
            top["bio_trending"],
            names["timeline_trending_bio"],
        },
    }

    candidate_names = {
        *names["all_top"],
        *names["child_defaults"].keys(),
    }
    expected_by_layer: dict[str, dict[str, bool]] = {}
    for view_layer_name in STANDARD_VIEW_LAYERS:
        expected = {name: False for name in candidate_names}
        visible = visible_by_layer[view_layer_name]
        for collection_name in names["all_top"]:
            expected[collection_name] = collection_name not in visible
        for collection_name, default_exclude in names["child_defaults"].items():
            if collection_name in visible:
                expected[collection_name] = False
            else:
                expected[collection_name] = default_exclude
        expected_by_layer[view_layer_name] = expected
    return expected_by_layer


def get_expected_timeline_world_objects(site: str) -> tuple[str, ...]:
    contract = SITE_CONTRACTS[site]
    return tuple(
        get_timeline_world_object_name(source_object_name, scenario)
        for scenario in ("positive", "trending")
        for source_object_name in contract["world_objects"].values()
    )


def get_expected_single_state_world_objects(site: str, year: int) -> tuple[str, ...]:
    contract = SITE_CONTRACTS[site]
    return tuple(
        get_single_state_world_object_name(source_object_name, year, scenario)
        for scenario in ("positive", "trending")
        for source_object_name in contract["world_objects"].values()
    )


def get_expected_instancer_specs(
    site: str,
    *,
    build_mode: str,
    year: int | None = None,
) -> tuple[dict[str, object], ...]:
    specs: list[dict[str, object]] = []

    for node_type in ("tree", "log"):
        for scenario in ("positive", "trending"):
            priority_values = (False, True) if scenario == "positive" else (False,)
            for priority in priority_values:
                if build_mode == "single_state":
                    if year is None:
                        raise ValueError("Single-state instancer specs require a year")
                    suffix = f"{scenario}_priority" if priority else scenario
                    base = get_single_state_node_collection_base(
                        site,
                        node_type,
                        year,
                        scenario,
                        priority=priority,
                    )
                    point_object = f"{node_type.capitalize()}Positions_{site}_yr{year}_{suffix}"
                else:
                    base = f"{node_type}_{site}_timeline_{scenario}"
                    point_object = f"{node_type.capitalize()}Positions_{site}_timeline_{scenario}"
                    if priority:
                        base = f"{base}_priority"
                        point_object = f"{point_object}_priority"

                specs.append(
                    {
                        "node_type": node_type,
                        "scenario": scenario,
                        "priority": priority,
                        "point_object": point_object,
                        "positions_collection": f"{base}_positions",
                        "models_collection": f"{base}_plyModels",
                        "node_group": base,
                        "modifier_name": base,
                    }
                )

    return tuple(specs)
