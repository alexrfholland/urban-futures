"""Canonical scene/schema contract for the Blender v2 pipeline.

This module is intentionally pure Python. It defines names, stable ordering,
and helper functions that other Blender scripts can rely on.
"""

from __future__ import annotations

import os
from typing import Final


SUPPORTED_SITES: Final[tuple[str, ...]] = ("city", "trimmed-parade", "uni")
SUPPORTED_MODES: Final[tuple[str, ...]] = ("single_state", "timeline", "baseline")

TEMPLATE_SCENE_NAME: Final[str] = "bV2_template"
TEMPLATE_VIEW_LAYER_NAME: Final[str] = "template_base"
TEMPLATE_BLEND_NAME: Final[str] = "bV2_template.blend"
TEMPLATE_ROOT_COLLECTIONS: Final[tuple[str, ...]] = ("world_sources", "camera_sources")

MATERIAL_NAMES: Final[dict[str, str]] = {
    "world": "v2WorldAOV",
    "instancers": "MINIMAL_RESOURCES",
    "bioenvelope": "Envelope",
    "debug_source_years": "debug-source-years",
    "proposals_preview": "PROPOSALS",
}

NODE_GROUP_NAMES: Final[dict[str, str]] = {
    "world": "v2WorldPoints",
    "world_cubes": "v2WorldCubes",
    "instancers": "instance_template",
    "bioenvelope": "Envelope",
}

TOP_LEVEL_COLLECTIONS: Final[tuple[str, ...]] = (
    "cameras",
    "world",
    "instancers",
    "bioenvelopes",
    "build",
)

WORKING_COLLECTION_TREE: Final[dict[str, tuple[str, ...]]] = {
    "cameras": (),
    "world": (
        "world_sources",
        "world_positive_attributes",
        "world_trending_attributes",
    ),
    "instancers": (
        "positive_instances",
        "positive_priority_instances",
        "trending_instances",
    ),
    "bioenvelopes": (
        "bioenvelope_positive",
        "bioenvelope_trending",
    ),
    "build": ("helpers",),
}

BASELINE_COLLECTION_TREE: Final[dict[str, tuple[str, ...]]] = {
    "cameras": (),
    "world": (
        "world_sources",
        "world_positive_attributes",
    ),
    "instancers": (
        "positive_instances",
        "positive_priority_instances",
    ),
    "bioenvelopes": (),
    "build": ("helpers",),
}

VIEW_LAYER_NAMES: Final[tuple[str, ...]] = (
    "existing_condition_positive",
    "existing_condition_trending",
    "positive_state",
    "positive_priority_state",
    "trending_state",
    "bioenvelope_positive",
    "bioenvelope_trending",
)

BASELINE_VIEW_LAYER_NAMES: Final[tuple[str, ...]] = (
    "existing_condition_positive",
    "positive_state",
    "positive_priority_state",
)

AOV_NAMES: Final[tuple[str, ...]] = (
    "proposal-decay",
    "proposal-release-control",
    "proposal-recruit",
    "proposal-colonise",
    "proposal-deploy-structure",
    "resource_none_mask",
    "resource_dead_branch_mask",
    "resource_peeling_bark_mask",
    "resource_perch_branch_mask",
    "resource_epiphyte_mask",
    "resource_fallen_log_mask",
    "resource_hollow_mask",
    "size",
    "control",
    "precolonial",
    "improvement",
    "canopy_resistance",
    "bioEnvelopeType",
    "intervention_bioenvelope_ply-int",
    "sim_Turns",
    "world_sim_turns",
    "world_sim_nodes",
    "world_design_bioenvelope",
    "world_design_bioenvelope_simple",
    "node_id",
    "instanceID",
    "source-year",
    "world_sim_matched",
)

# Semantic view-layer contents. These are asset roles, not Blender collection
# names. Scene-building code can resolve these roles to real collections/objects.
VIEW_LAYER_SEMANTICS: Final[dict[str, tuple[str, ...]]] = {
    "existing_condition_positive": ("world_positive_attributes",),
    "existing_condition_trending": ("world_trending_attributes",),
    "positive_state": (
        "positive_instances",
        "world_positive_attributes",
        "bioenvelope_positive",
    ),
    "positive_priority_state": (
        "positive_priority_instances",
        "world_positive_attributes",
    ),
    "trending_state": (
        "trending_instances",
        "world_trending_attributes",
        "bioenvelope_trending",
    ),
    "bioenvelope_positive": (
        "world_positive_attributes",
        "bioenvelope_positive",
    ),
    "bioenvelope_trending": (
        "world_trending_attributes",
        "bioenvelope_trending",
    ),
}

STATE_TOKENS: Final[tuple[str, ...]] = (
    "positive",
    "positive_priority",
    "trending",
)

SITE_CONTRACTS: Final[dict[str, dict[str, object]]] = {
    "city": {
        "world_sources": {
            "buildings": "city_buildings_source",
            "roads_hires": "city_roads_source_hires",
            "roads_lores": "city_roads_source_lores",
        },
        "world_source_aliases": {
            "buildings": ("city_buildings_source",),
            "roads": ("city_roads_source", "city_roads_source_hires"),
            "roads_cubes": ("city_roads_source_lores",),
        },
        "cameras": {
            "timeline": "city - camera - time slice - zoom",
            "hero": "city-yr180-hero-image",
        },
        "instancer_families": ("trees", "logs"),
    },
    "trimmed-parade": {
        "world_sources": {
            "buildings": "trimmed-parade_buildings_source",
            "roads_hires": "trimmed-parade_roads_source_hires",
            "roads_lores": "trimmed-parade_roads_source_lores",
        },
        "world_source_aliases": {
            "buildings": ("trimmed-parade_buildings_source",),
            "roads": ("trimmed-parade_roads_source", "trimmed-parade_roads_source_hires"),
            "roads_cubes": ("trimmed-parade_roads_source_lores",),
        },
        "cameras": {
            "timeline": "parade - camera - time slice - zoom",
            "hero": "parade-hero-image",
        },
        "instancer_families": ("trees", "logs"),
    },
    "uni": {
        "world_sources": {
            "buildings": "uni_buildings_source",
            "roads_hires": "uni_roads_source_hires",
            "roads_lores": "uni_roads_source_lores",
        },
        "world_source_aliases": {
            "buildings": ("uni_buildings_source",),
            "roads": ("uni_roads_source", "uni_roads_source_hires"),
            "roads_cubes": ("uni_roads_source_lores",),
        },
        "cameras": {
            "timeline": "uni - camera - time slice - zoom",
        },
        "instancer_families": ("trees", "logs", "poles"),
    },
}

REQUIRED_WORLD_SOURCE_ROLES: Final[tuple[str, ...]] = ("buildings", "roads")
OPTIONAL_WORLD_SOURCE_ROLES: Final[tuple[str, ...]] = ("roads_cubes",)
WORLD_SOURCE_RUNTIME_MODES: Final[tuple[str, ...]] = ("cubes", "points")

GLOBAL_RULES: Final[dict[str, object]] = {
    "canonical_scenario_branches": ("positive", "trending"),
    "priority_derived_from": "positive",
    "all_renderable_geometry_has_source_year": True,
    "source_year_initial_value": -1,
    "world_material": MATERIAL_NAMES["world"],
    "world_node_group": NODE_GROUP_NAMES["world"],
    "instancer_material": MATERIAL_NAMES["instancers"],
    "debug_source_years_material": MATERIAL_NAMES["debug_source_years"],
    "bioenvelope_material": MATERIAL_NAMES["bioenvelope"],
}

TIMELINE_MIST_PROFILE: Final[dict[str, object]] = {
    "profile": "timeline_shared",
    "use_mist": True,
    "start": 560.0,
    "depth": 320.0,
    "falloff": "QUADRATIC",
}

CITY_SINGLE_STATE_HERO_MIST_PROFILE: Final[dict[str, object]] = {
    "profile": "city_single_state_hero",
    "use_mist": False,
    "start": 60.0,
    "depth": 420.0,
    "falloff": "LINEAR",
}


def _assert_supported(value: str, supported: tuple[str, ...], kind: str) -> str:
    if value not in supported:
        supported_list = ", ".join(supported)
        raise ValueError(f"Unsupported {kind}: {value!r}. Expected one of: {supported_list}")
    return value


def ensure_supported_site(site: str) -> str:
    """Return the site if valid, otherwise raise."""

    return _assert_supported(site, SUPPORTED_SITES, "site")


def ensure_supported_mode(mode: str) -> str:
    """Return the mode if valid, otherwise raise."""

    return _assert_supported(mode, SUPPORTED_MODES, "mode")


def get_site_contract(site: str) -> dict[str, object]:
    """Return the canonical contract for a site."""

    return SITE_CONTRACTS[ensure_supported_site(site)]


def get_mode_year_token(mode: str, year: int | str | None = None) -> str:
    """Return the canonical year token used in generated names."""

    mode = ensure_supported_mode(mode)
    if mode == "timeline":
        return "timeline"
    if mode == "baseline" and year is None:
        return "baseline"
    if year is None:
        raise ValueError(f"{mode} mode requires a year")
    if isinstance(year, str) and year.startswith("yr"):
        return year
    return f"yr{int(year)}"


def make_scene_name(site: str, mode: str, year: int | str | None = None) -> str:
    """Return the canonical built-scene name."""

    site = ensure_supported_site(site)
    mode = ensure_supported_mode(mode)
    if mode == "timeline":
        return f"bV2_{site}_timeline"
    if mode == "baseline":
        if year is None:
            return f"bV2_{site}_baseline"
        return f"bV2_{site}_baseline_{get_mode_year_token(mode, year)}"
    return f"bV2_{site}_single_state_{get_mode_year_token(mode, year)}"


def make_position_object_name(
    family: str,
    site: str,
    mode: str,
    state: str,
    year: int | str | None = None,
) -> str:
    """Return a canonical positions object name."""

    return f"{family}_positions_{ensure_supported_site(site)}_{get_mode_year_token(mode, year)}_{state}"


def make_models_collection_name(
    family: str,
    site: str,
    mode: str,
    state: str,
    year: int | str | None = None,
) -> str:
    """Return a canonical model collection name."""

    return f"{family}_models_{ensure_supported_site(site)}_{get_mode_year_token(mode, year)}_{state}"


def make_world_object_name(
    kind: str,
    site: str,
    mode: str,
    state: str,
    year: int | str | None = None,
) -> str:
    """Return a canonical rebuilt world object name."""

    return f"{kind}_{ensure_supported_site(site)}_{get_mode_year_token(mode, year)}_{state}"


def make_bioenvelope_object_name(
    site: str,
    mode: str,
    state: str,
    year: int | str | None = None,
) -> str:
    """Return a canonical bioenvelope object name."""

    return f"bioenvelope_{ensure_supported_site(site)}_{get_mode_year_token(mode, year)}_{state}"


def get_view_layer_names(mode: str | None = None) -> tuple[str, ...]:
    """Return the canonical ordered view-layer names."""

    if mode == "baseline":
        return BASELINE_VIEW_LAYER_NAMES
    return VIEW_LAYER_NAMES


def get_working_collection_tree(mode: str | None = None) -> dict[str, tuple[str, ...]]:
    """Return the canonical top-level and second-level working collections."""

    if mode == "baseline":
        return BASELINE_COLLECTION_TREE
    return WORKING_COLLECTION_TREE


def get_view_layer_semantics(view_layer_name: str) -> tuple[str, ...]:
    """Return semantic content tags for a view layer."""

    return VIEW_LAYER_SEMANTICS[view_layer_name]


def get_aov_names() -> tuple[str, ...]:
    """Return the canonical ordered AOV names."""

    return AOV_NAMES


def get_source_world_objects(site: str) -> dict[str, str]:
    """Return source world object names for a site."""

    return dict(get_site_contract(site)["world_sources"])


def get_source_world_object_aliases(site: str) -> dict[str, tuple[str, ...]]:
    """Return canonical world-source roles mapped to acceptable template object names."""

    aliases = get_site_contract(site).get("world_source_aliases")
    if aliases is None:
        return {
            str(role): (str(name),)
            for role, name in dict(get_site_contract(site)["world_sources"]).items()
        }
    return {str(role): tuple(names) for role, names in dict(aliases).items()}


def get_world_source_runtime_mode() -> str:
    """Return the runtime world-source mode requested by the build environment."""

    raw = os.environ.get("BV2_POINTSORCUBES", "").strip().lower()
    if raw in {"", "points", "point"}:
        return "points"
    if raw in {"cubes", "split"}:
        return "cubes"
    supported = ", ".join(WORLD_SOURCE_RUNTIME_MODES)
    raise RuntimeError(
        f"Unsupported BV2_POINTSORCUBES={raw!r}. Expected one of: points, point, {supported}, split"
    )


def resolve_source_world_object_names(site: str, available_names: set[str]) -> dict[str, str]:
    """Resolve required/optional world sources against the available object names."""

    runtime_mode = get_world_source_runtime_mode()
    resolved: dict[str, str] = {}
    missing_required: list[str] = []
    for role, candidates in get_source_world_object_aliases(site).items():
        if runtime_mode == "points" and role == "roads_cubes":
            continue
        match = next((name for name in candidates if name in available_names), None)
        if match is not None:
            resolved[role] = match
        elif role in REQUIRED_WORLD_SOURCE_ROLES:
            missing_required.append(f"{role}: {candidates}")
    if missing_required:
        raise RuntimeError(
            "Missing required world source objects for "
            f"{site!r}: {', '.join(missing_required)}"
        )
    return resolved


def get_timeline_camera_name(site: str) -> str:
    """Return the canonical timeline camera for a site."""

    return str(get_site_contract(site)["cameras"]["timeline"])


def get_default_camera_name(
    site: str,
    mode: str,
    year: int | str | None = None,
) -> str:
    """Return the default active camera for a built scene."""

    site = ensure_supported_site(site)
    mode = ensure_supported_mode(mode)
    cameras = dict(get_site_contract(site)["cameras"])
    if mode == "single_state" and "hero" in cameras:
        return str(cameras["hero"])
    return str(cameras["timeline"])


def get_expected_camera_names(
    site: str,
    mode: str,
    year: int | str | None = None,
) -> tuple[str, ...]:
    """Return the acceptable camera names for validation."""

    default_name = get_default_camera_name(site, mode, year)
    timeline_name = get_timeline_camera_name(site)
    if default_name == timeline_name:
        return (default_name,)
    return (default_name, timeline_name)


def get_alternate_camera_names(site: str) -> tuple[str, ...]:
    """Return any non-timeline cameras for a site."""

    cameras = dict(get_site_contract(site)["cameras"])
    return tuple(name for role, name in cameras.items() if role != "timeline")


def get_default_mist_profile(
    site: str,
    mode: str,
    year: int | str | None = None,
    camera_name: str | None = None,
) -> dict[str, object]:
    """Return the default mist profile for a site/mode/camera combination."""

    site = ensure_supported_site(site)
    mode = ensure_supported_mode(mode)
    cameras = dict(get_site_contract(site)["cameras"])
    resolved_camera_name = str(camera_name or get_default_camera_name(site, mode, year))
    hero_name = str(cameras.get("hero", "")).strip()
    if hero_name and resolved_camera_name == hero_name:
        return dict(CITY_SINGLE_STATE_HERO_MIST_PROFILE)
    return dict(TIMELINE_MIST_PROFILE)


def get_instancer_families(site: str) -> tuple[str, ...]:
    """Return valid instancer families for a site."""

    return tuple(get_site_contract(site)["instancer_families"])
