from __future__ import annotations


STANDARD_VIEW_LAYERS = (
    "pathway_state",
    "existing_condition",
    "priority_state",
    "existing_condition_trending",
    "bioenvelope_positive",
    "bioenvelope_trending",
    "trending_state",
)


SINGLE_STATE_VIEW_LAYERS = STANDARD_VIEW_LAYERS


SITE_CONTRACTS = {
    "city": {
        "scene_name": "city",
        "label": "City",
        "top_level": {
            "base": "City_Base",
            "base_cubes": "City_Base-cubes",
            "manager": "City_Manager",
            "cameras": "City_Cameras",
            "bio_positive": "City_Bioenvelopes-positive",
            "bio_trending": "City_Bioenvelopes-trending",
            "positive": "city_positive",
            "priority": "city_priority",
            "trending": "city_trending",
        },
        "legacy": {
            "base": "City_World",
            "base_cubes": "City_WorldCubes",
            "manager": "City_Manager",
            "cameras": "City_Camera",
            "camera_archive": "City_Camera_Archive",
            "bio_positive": "city_envelope",
            "bio_trending": "Envelope_trending",
            "timeline_base": "City_World_Timeline",
            "positive": "Year_city_180_positive",
            "priority": "Year_city_180_positive_priority",
            "trending": "Year_city_180_trending",
            "timeline_positive": "Year_city_timeline_positive",
            "timeline_priority": "Year_city_timeline_positive_priority",
            "timeline_trending": "Year_city_timeline_trending",
        },
        "view_layer_aliases": {
            "priority": ("priority_state", "city_priority"),
            "bio_positive": ("bioenvelope_positive", "city_bioenvelope"),
        },
        "build_priority_branch": True,
        "live_clip_object": "City_ClipBox",
        "camera_proxy_prefix": "CityClipProxy__",
        "camera_subcollection_prefix": "CityCamera__",
        "world_objects": {
            "base": "city_buildings.001",
            "road": "city_highResRoad.001",
        },
        "world_prefixes": {
            "base": ("city_buildings",),
            "road": ("city_highResRoad",),
        },
    },
    "trimmed-parade": {
        "scene_name": "parade",
        "label": "Parade",
        "top_level": {
            "base": "Parade_Base",
            "base_cubes": "Parade_Base-cubes",
            "manager": "Parade_Manager",
            "cameras": "Parade_Cameras",
            "bio_positive": "Parade_Bioenvelopes-positive",
            "bio_trending": "Parade_Bioenvelopes-trending",
            "positive": "trimmed-parade_positive",
            "priority": "trimmed-parade_priority",
            "trending": "trimmed-parade_trending",
        },
        "legacy": {
            "base": "Parade_World",
            "base_cubes": "Parade_WorldCubes",
            "manager": "Parade_Manager",
            "cameras": "Parade_Camera",
            "camera_archive": "Parade_Camera_Archive",
            "bio_positive": "Parade_envelope",
            "bio_trending": "Parade_envelope_trending",
            "timeline_base": "Parade_World_Timeline",
            "positive": "Year_trimmed-parade_180_positive",
            "priority": "Year_trimmed-parade_180_positive_priority",
            "trending": "Year_trimmed-parade_180_trending",
            "timeline_positive": "Year_trimmed-parade_timeline_positive",
            "timeline_priority": "Year_trimmed-parade_timeline_priority",
            "timeline_trending": "Year_trimmed-parade_timeline_trending",
        },
        "view_layer_aliases": {
            "priority": ("priority_state",),
            "bio_positive": ("bioenvelope_positive",),
        },
        "build_priority_branch": True,
        "live_clip_object": "ClipBox",
        "camera_proxy_prefix": "ParadeClipProxy__",
        "camera_subcollection_prefix": "ParadeCamera__",
        "world_objects": {
            "base": "trimmed-parade_base",
            "road": "trimmed-parade_highResRoad",
        },
        "world_prefixes": {
            "base": ("trimmed-parade_base",),
            "road": ("trimmed-parade_highResRoad",),
        },
    },
    "uni": {
        "scene_name": "uni",
        "label": "Uni",
        "top_level": {
            "base": "Uni_Base",
            "base_cubes": "Uni_Base-cubes",
            "manager": "Uni_Manager",
            "cameras": "Uni_Cameras",
            "bio_positive": "Uni_Bioenvelopes-positive",
            "bio_trending": "Uni_Bioenvelopes-trending",
            "positive": "uni_positive",
            "priority": "uni_priority",
            "trending": "uni_trending",
        },
        "legacy": {
            "base": "Uni_World",
            "base_cubes": "Uni_WorldCubes",
            "manager": "Uni_Manager",
            "cameras": "Uni_Camera",
            "camera_archive": "Uni_Camera_Archive",
            "bio_positive": "Uni_envelope",
            "bio_trending": "Uni_envelope_trending",
            "timeline_base": "Uni_World_Timeline",
            "positive": "Year_uni_180_positive",
            "priority": "Year_uni_180_positive_priority",
            "trending": "Year_uni_180_trending",
            "timeline_positive": "Year_uni_timeline_positive",
            "timeline_priority": "Year_uni_timeline_priority",
            "timeline_trending": "Year_uni_timeline_trending",
        },
        "view_layer_aliases": {
            "priority": ("priority_state",),
            "bio_positive": ("bioenvelope_positive",),
        },
        "build_priority_branch": True,
        "live_clip_object": "Uni_ClipBox",
        "camera_proxy_prefix": "UniClipProxy__",
        "camera_subcollection_prefix": "UniCamera__",
        "world_objects": {
            "base": "uni_base",
            "road": "uni_highResRoad",
        },
        "world_prefixes": {
            "base": ("uni_base",),
            "road": ("uni_highResRoad",),
        },
    },
    "street": {
        "scene_name": "street",
        "label": "Street",
        "top_level": {
            "base": "Street_Base",
            "base_cubes": "Street_Base-cubes",
            "manager": "Street_Manager",
            "cameras": "Street_Cameras",
            "bio_positive": "Street_Bioenvelopes-positive",
            "bio_trending": "Street_Bioenvelopes-trending",
            "positive": "street_positive",
            "priority": "street_priority",
            "trending": "street_trending",
        },
        "legacy": {
            "base": "Uni_World",
            "base_cubes": "Uni_WorldCubes",
            "manager": "Street_Manager",
            "cameras": "Uni_Camera",
            "camera_archive": "Uni_Camera_Archive",
            "bio_positive": "Uni_envelope",
            "bio_trending": "Uni_envelope_trending",
            "timeline_base": "Uni_World_Timeline",
            "positive": "Year_uni_180_positive",
            "priority": "Year_uni_180_positive_priority",
            "trending": "Year_uni_180_trending",
            "timeline_positive": "Year_uni_timeline_positive",
            "timeline_priority": "Year_uni_timeline_positive_priority",
            "timeline_trending": "Year_uni_timeline_trending",
        },
        "view_layer_aliases": {
            "priority": ("priority_state",),
            "bio_positive": ("bioenvelope_positive",),
        },
        "build_priority_branch": True,
        "live_clip_object": "Uni_ClipBox",
        "camera_proxy_prefix": "UniClipProxy__",
        "camera_subcollection_prefix": "UniCamera__",
        "world_objects": {
            "base": "uni_base",
            "road": "uni_highResRoad",
        },
        "world_prefixes": {
            "base": ("uni_base",),
            "road": ("uni_highResRoad",),
        },
    },
}


def infer_site_from_scene_name(scene_name: str | None) -> str | None:
    normalized = (scene_name or "").strip().lower()
    if normalized.startswith("city"):
        return "city"
    if normalized.startswith("parade") or "trimmed-parade" in normalized:
        return "trimmed-parade"
    if normalized.startswith("uni"):
        return "uni"
    if normalized.startswith("street"):
        return "street"
    return None


def get_site_contract(site_or_scene_name: str | None) -> dict | None:
    if site_or_scene_name in SITE_CONTRACTS:
        return SITE_CONTRACTS[site_or_scene_name]

    inferred = infer_site_from_scene_name(site_or_scene_name)
    if inferred is None:
        return None
    return SITE_CONTRACTS[inferred]


def get_collection_name(site: str, role: str, *, legacy: bool = False) -> str:
    contract = SITE_CONTRACTS[site]
    key = "legacy" if legacy else "top_level"
    return contract[key][role]


def get_priority_view_layer_names(scene_name: str | None) -> tuple[str, ...]:
    contract = get_site_contract(scene_name)
    if contract is None:
        return ("priority_state",)
    return tuple(contract["view_layer_aliases"].get("priority", ("priority_state",)))


def get_positive_bioenvelope_view_layer_names(scene_name: str | None) -> tuple[str, ...]:
    contract = get_site_contract(scene_name)
    if contract is None:
        return ("bioenvelope_positive",)
    return tuple(contract["view_layer_aliases"].get("bio_positive", ("bioenvelope_positive",)))


SINGLE_STATE_TOP_LEVEL_ROLES = (
    "manager",
    "setup",
    "cameras",
    "positive",
    "priority",
    "trending",
)


def get_single_state_top_level_name(site: str, role: str) -> str:
    if role not in SINGLE_STATE_TOP_LEVEL_ROLES:
        raise KeyError(f"Unknown single-state top-level role '{role}'")
    return f"{site}_{role}"


def get_single_state_world_object_name(source_object_name: str, year: int, scenario: str) -> str:
    return f"{source_object_name}__yr{year}_{scenario}_state"


def get_single_state_node_collection_base(
    site: str,
    node_type: str,
    year: int,
    scenario: str,
    *,
    priority: bool = False,
) -> str:
    suffix = f"{scenario}_priority" if priority else scenario
    return f"{node_type}_{site}_yr{year}_{suffix}"


def get_single_state_node_collection_name(
    site: str,
    node_type: str,
    year: int,
    scenario: str,
    collection_kind: str,
    *,
    priority: bool = False,
) -> str:
    if collection_kind not in {"positions", "plyModels"}:
        raise KeyError(f"Unknown node collection kind '{collection_kind}'")
    base = get_single_state_node_collection_base(
        site,
        node_type,
        year,
        scenario,
        priority=priority,
    )
    return f"{base}_{collection_kind}"


def get_single_state_envelope_object_name(site: str, scenario: str, year: int) -> str:
    return f"{site}_{scenario}_envelope__yr{year}"


def get_single_state_envelope_collection_name(site: str, scenario: str) -> str:
    return f"{site}_{scenario}_envelope"


def get_timeline_world_collection_name(site: str, scenario: str) -> str:
    return f"Year_{site}_timeline_world_{scenario}"


def get_timeline_world_object_name(source_object_name: str, scenario: str) -> str:
    return f"{source_object_name}__timeline_{scenario}_state"
