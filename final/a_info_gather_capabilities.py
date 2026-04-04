"""
a_infov3_gather_capabilities.py
===============================
Defines capability indicators and extracts them from scenario VTK files.

USAGE:
    python a_infov3_gather_capabilities.py --site trimmed-parade --scenario positive --year 60

This script:
    1. Defines capability indicators (EDIT SECTION 1 BELOW)
    2. Loads a scenario VTK file
    3. Creates boolean indicator layers for each capability
    4. Saves the VTK with indicator layers added

The indicator definitions are at the TOP of this file for easy editing.
"""

import numpy as np
import pandas as pd
import pyvista as pv
import sys
from pathlib import Path
from scipy.spatial import cKDTree
import a_helper_functions

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

from refactor_code.paths import (
    engine_output_root,
    engine_output_baseline_action_counts_path,
    engine_output_baseline_indicator_counts_path,
    engine_output_baseline_state_vtk_path,
    engine_output_state_action_counts_path,
    engine_output_state_indicator_counts_path,
    engine_output_state_vtk_path,
    normalize_output_mode,
    refactor_statistics_root,
    scenario_baseline_combined_vtk_path,
    scenario_output_root,
)
from refactor_code.blender.proposal_framebuffers_vtk import build_blender_proposal_framebuffer_arrays
from refactor_code.scenario import params_v3

SCRIPT_DIR = Path(__file__).parent

# Default assessed years for the current v3 stack.
DEFAULT_YEARS = params_v3.generate_timesteps(interval=30)  # [0, 10, 30, 60, 90, 120, 150, 180]


def format_voxel_size(voxel_size):
    numeric = float(voxel_size)
    return str(int(numeric)) if numeric.is_integer() else str(voxel_size)


# =============================================================================
# SECTION 1: CAPABILITY INDICATOR DEFINITIONS
# =============================================================================
# 
# Edit this section to change what capabilities are measured.
#
# Each indicator has:
#   - id: Unique identifier in format Persona.Capability.Indicator
#   - persona: Bird, Lizard, or Tree
#   - capability: self (sustain), others (connect), generations (persist)
#   - label: Human-readable description for graphs
#   - query: What voxels to count (see QUERY SYNTAX below)
#
# Optional distance filter (for spatial relationships):
#   - distance_from: What to measure distance from ('canopy-feature')
#   - distance_type: 'within' or 'outside'
#   - distance_meters: Distance in meters
#
# QUERY SYNTAX:
#   - 'field > 0'           : Voxels where field is positive
#   - 'field == value'      : Voxels where field equals value
#   - 'ground_not_paved'    : Ground voxels not on paved surfaces
#
# DISTANCE_FROM OPTIONS:
#   - 'canopy-feature': Trees and fallen logs (forest_size != nan OR stat_fallen log > 0)
#
# CURRENT INDICATORS:
# ┌─────────────────────────────────┬───────────────────────────────────────────────────────────────┐
# │ ID                              │ Query                                                         │
# ├─────────────────────────────────┼───────────────────────────────────────────────────────────────┤
# │ Bird.self.peeling               │ stat_peeling bark > 0                                         │
# │ Bird.others.perch               │ stat_perch branch > 0                                         │
# │ Bird.generations.hollow         │ stat_hollow > 0                                               │
# ├─────────────────────────────────┼───────────────────────────────────────────────────────────────┤
# │ Lizard.self.grass               │ search_bioavailable == low-vegetation                         │
# │ Lizard.self.dead                │ stat_dead branch > 0                                          │
# │ Lizard.self.epiphyte            │ stat_epiphyte > 0                                             │
# │ Lizard.others.notpaved          │ ground_not_paved                                              │
# │ Lizard.generations.nurse-log    │ stat_fallen log > 0                                           │
# │ Lizard.generations.fallen-tree  │ forest_size in fallen|decayed                                 │
# ├─────────────────────────────────┼───────────────────────────────────────────────────────────────┤
# │ Tree.self.senescent             │ forest_size in senescing|snag|fallen|decayed                  │
# │ Tree.others.notpaved            │ ground_not_paved + within 50m canopy + ground_only            │
# │ Tree.generations.grassland      │ low-vegetation + within 20m canopy + ground_only              │
# └─────────────────────────────────┴───────────────────────────────────────────────────────────────┘
#
# ground_only: Excludes building voxels (facade, green roof, brown roof)
#
# =============================================================================

INDICATORS = [
    # ----- BIRD CAPABILITIES -----
    {
        'id': 'Bird.self.peeling',
        'persona': 'Bird',
        'capability': 'self',
        'label': 'Peeling bark volume',
        'query': 'stat_peeling bark > 0',
    },
    {
        'id': 'Bird.others.perch',
        'persona': 'Bird',
        'capability': 'others',
        'label': 'Perchable canopy volume',
        'query': 'stat_perch branch > 0',
    },
    {
        'id': 'Bird.generations.hollow',
        'persona': 'Bird',
        'capability': 'generations',
        'label': 'Hollow count',
        'query': 'stat_hollow > 0',
    },
    
    # ----- LIZARD CAPABILITIES -----
    {
        'id': 'Lizard.self.grass',
        'persona': 'Lizard',
        'capability': 'self',
        'label': 'Ground cover area',
        'query': 'search_bioavailable == low-vegetation',
    },
    {
        'id': 'Lizard.self.dead',
        'persona': 'Lizard',
        'capability': 'self',
        'label': 'Dead branch volume',
        'query': 'stat_dead branch > 0',
    },
    {
        'id': 'Lizard.self.epiphyte',
        'persona': 'Lizard',
        'capability': 'self',
        'label': 'Epiphyte count',
        'query': 'stat_epiphyte > 0',
    },
    {
        'id': 'Lizard.others.notpaved',
        'persona': 'Lizard',
        'capability': 'others',
        'label': 'Non-paved surface area',
        'query': 'ground_not_paved',
    },
    {
        'id': 'Lizard.generations.nurse-log',
        'persona': 'Lizard',
        'capability': 'generations',
        'label': 'Nurse log volume',
        'query': 'stat_fallen log > 0',
    },
    {
        'id': 'Lizard.generations.fallen-tree',
        'persona': 'Lizard',
        'capability': 'generations',
        'label': 'Fallen tree volume',
        'query': 'forest_size in fallen|decayed',
    },
    
    # ----- TREE CAPABILITIES -----
    {
        'id': 'Tree.self.senescent',
        'persona': 'Tree',
        'capability': 'self',
        'label': 'Late-life tree and deadwood volume',
        'query': 'forest_size in senescing|snag|fallen|decayed',
    },
    {
        'id': 'Tree.others.notpaved',
        'persona': 'Tree',
        'capability': 'others',
        'label': 'Soil near canopy features',
        'query': 'ground_not_paved',
        'distance_from': 'canopy-feature',
        'distance_type': 'within',
        'distance_meters': 50,
        'ground_only': True,  # Exclude facades and roofs
    },
    {
        'id': 'Tree.generations.grassland',
        'persona': 'Tree',
        'capability': 'generations',
        'label': 'Grassland for recruitment',
        'query': 'search_bioavailable == low-vegetation',
        'distance_from': 'canopy-feature',
        'distance_type': 'within',
        'distance_meters': 20,
        'ground_only': True,  # Exclude facades and roofs
    },
]


# =============================================================================
# SECTION 2: SUPPORT ACTION DEFINITIONS
# =============================================================================
#
# Support actions track what urban interventions contribute to each capability.
# These are counted separately and broken down by urban element or control level.
#
# BREAKDOWN TYPES:
#   - 'control_level': Count by tree management (high/medium/low control)
#   - 'urban_element': Count by urban surface type (parking, roadway, etc.)
#   - 'artificial': Count only non-precolonial (installed) elements
#
# =============================================================================

SUPPORT_ACTIONS = {
    # Bird support actions - track by canopy control level
    'Bird.self.peeling': {'breakdown': 'control_level', 'also_count': 'artificial'},
    'Bird.others.perch': {'breakdown': 'control_level'},
    'Bird.generations.hollow': {'breakdown': 'control_level', 'also_count': 'artificial'},
    
    # Lizard support actions - track by urban element conversion
    'Lizard.self.grass': {'breakdown': 'urban_element'},
    'Lizard.self.dead': {'breakdown': 'control_level'},
    'Lizard.self.epiphyte': {'breakdown': 'control_level', 'also_count': 'artificial'},
    'Lizard.others.notpaved': {'breakdown': 'urban_element'},
    'Lizard.generations.nurse-log': {'breakdown': 'urban_element'},
    'Lizard.generations.fallen-tree': {'breakdown': 'urban_element'},
    
    # Tree support actions - track by rewilding status
    'Tree.self.senescent': {'breakdown': 'rewilding_status'},
    'Tree.others.notpaved': {'breakdown': 'urban_element'},
    'Tree.generations.grassland': {'breakdown': 'urban_element'},
}

# Control levels for canopy breakdown
CONTROL_LEVELS = {
    'high': ['street-tree'],
    'medium': ['park-tree'],
    'low': ['reserve-tree', 'improved-tree'],
}

# Urban element types for ground breakdown
URBAN_ELEMENTS = [
    'open space', 'green roof', 'brown roof', 'facade',
    'roadway', 'busy roadway', 'existing conversion',
    'other street potential', 'parking', 'none'
]

# Rewilding status types
REWILDING_TYPES = ['footprint-depaved', 'exoskeleton', 'node-rewilded', 'none']
PROPOSAL_LABEL_DTYPE = "<U64"
COLONISE_PROPOSAL_VALUES = {
    "brownroof",
    "greenroof",
    "livingfacade",
    "footprint-depaved",
    "node-rewilded",
    "otherground",
    "rewilded",
}
COLONISE_REWILD_VALUES = {"node-rewilded", "footprint-depaved", "rewilded"}
COLONISE_ENRICH_VALUES = {"greenroof"}
COLONISE_ROUGHEN_VALUES = {"brownroof", "livingfacade"}
DECAY_BUFFER_VALUES = {"node-rewilded", "footprint-depaved"}
RECRUIT_BUFFER_VALUES = {"node-rewilded", "footprint-depaved"}
RECRUIT_REWILD_VALUES = {"otherground", "rewilded"}


# =============================================================================
# SECTION 3: MASK CREATION FUNCTIONS
# =============================================================================

def get_branch_mask(polydata):
    """Voxels in tree canopy (arboreal)."""
    if 'search_bioavailable' not in polydata.point_data:
        return np.zeros(polydata.n_points, dtype=bool)
    return polydata.point_data['search_bioavailable'] == 'arboreal'


def get_ground_mask(polydata):
    """Voxels at ground level (low-vegetation or open space)."""
    if 'search_bioavailable' not in polydata.point_data:
        return np.zeros(polydata.n_points, dtype=bool)
    bio = polydata.point_data['search_bioavailable']
    return (bio == 'low-vegetation') | (bio == 'open space')


def get_paved_mask(polydata):
    """Voxels on paved surfaces."""
    if 'search_urban_elements' not in polydata.point_data:
        return np.zeros(polydata.n_points, dtype=bool)
    urban = polydata.point_data['search_urban_elements']
    return (urban == 'roadway') | (urban == 'busy roadway') | (urban == 'parking')


def get_distance_reference_mask(polydata, distance_from):
    """
    Get mask for reference points to measure distance from.
    
    Args:
        polydata: The VTK polydata
        distance_from: Type of reference ('canopy-feature')
    
    Returns:
        Boolean mask of reference points
    """
    mask = np.zeros(polydata.n_points, dtype=bool)
    
    if distance_from == 'canopy-feature':
        # Trees (any forest_size that isn't empty)
        if 'forest_size' in polydata.point_data:
            fs = polydata.point_data['forest_size']
            if fs.dtype.kind in ['U', 'S']:
                mask |= (fs != 'nan') & (fs != 'none') & (fs != '')
        
        # Fallen logs
        if 'stat_fallen log' in polydata.point_data:
            fl = polydata.point_data['stat_fallen log']
            if np.issubdtype(fl.dtype, np.number):
                mask |= fl > 0
    else:
        raise ValueError(f"Unknown distance_from type: {distance_from}")
    
    return mask


def get_points_within_distance(polydata, reference_mask, distance):
    """Find points within distance of reference points using KDTree."""
    if not np.any(reference_mask):
        return np.zeros(polydata.n_points, dtype=bool)
    
    reference_points = polydata.points[reference_mask]
    tree = cKDTree(reference_points)
    distances, _ = tree.query(polydata.points, k=1)
    return distances <= distance


def get_building_mask(polydata):
    """
    Get mask for building voxels (facades and roofs).
    These are identified by search_urban_elements being 'facade', 'green roof', or 'brown roof'.
    """
    building_types = ['facade', 'green roof', 'brown roof']
    
    if 'search_urban_elements' in polydata.point_data:
        urban = polydata.point_data['search_urban_elements']
        mask = np.zeros(polydata.n_points, dtype=bool)
        for btype in building_types:
            mask |= (urban == btype)
        return mask
    
    # If search_urban_elements doesn't exist, return all False (no buildings detected)
    return np.zeros(polydata.n_points, dtype=bool)


def _normalize_str_array(values):
    return np.char.lower(np.asarray(values).astype(str))


def _coerce_bool_array(values):
    array = np.asarray(values)
    if array.dtype == bool:
        return array
    if np.issubdtype(array.dtype, np.number):
        numeric = np.nan_to_num(array.astype(float), nan=0.0)
        return numeric != 0

    normalized = _normalize_str_array(array)
    truthy = {"true", "1", "yes", "y", "t"}
    return np.isin(normalized, list(truthy))


def _empty_proposal_labels(size):
    return np.full(size, "none", dtype=PROPOSAL_LABEL_DTYPE)


def _empty_v3_decisions(size):
    return np.full(size, "not-assessed", dtype=PROPOSAL_LABEL_DTYPE)


def _empty_v3_interventions(size):
    return np.full(size, "none", dtype=PROPOSAL_LABEL_DTYPE)


def _assign_proposal_labels(labels, opportunity_mask, intervention_masks, proposal_name):
    labels[opportunity_mask] = f"{proposal_name}-other"
    for intervention_name, mask in intervention_masks:
        labels[mask] = f"{proposal_name}_{intervention_name}"
    return labels


def _string_point_data(polydata, name, default):
    if name not in polydata.point_data:
        return np.full(polydata.n_points, default, dtype=PROPOSAL_LABEL_DTYPE)
    values = np.asarray(polydata.point_data[name]).astype(str)
    cleaned = values.copy()
    cleaned[np.isin(_normalize_str_array(cleaned), ["", "nan"])] = default
    return cleaned.astype(PROPOSAL_LABEL_DTYPE)


def add_proposal_point_data(polydata):
    """Add manuscript proposal/intervention categorical layers to the augmented VTK."""
    n_points = polydata.n_points
    required_arrays = {
        "scenario_rewilded",
        "scenario_bioEnvelope",
        "scenario_outputs",
        "search_bioavailable",
        "forest_control",
        "forest_size",
        "forest_precolonial",
        "indicator_Bird_self_peeling",
        "indicator_Tree_generations_grassland",
    }
    missing_arrays = sorted(array for array in required_arrays if array not in polydata.point_data)
    if missing_arrays:
        print(f"Skipping proposal point-data derivation; missing arrays: {missing_arrays}")
        for proposal_name in [
            "proposal_decay",
            "proposal_recruit",
            "proposal_release_control",
            "proposal_colonise",
            "proposal_deploy_structure",
        ]:
            polydata.point_data[proposal_name] = _empty_proposal_labels(n_points)
        return polydata

    scenario_rewilded_lower = _normalize_str_array(polydata.point_data["scenario_rewilded"])
    scenario_bio_envelope_lower = _normalize_str_array(polydata.point_data["scenario_bioEnvelope"])
    scenario_outputs_lower = _normalize_str_array(polydata.point_data["scenario_outputs"])
    search_bioavailable_lower = _normalize_str_array(polydata.point_data["search_bioavailable"])
    forest_control_lower = _normalize_str_array(polydata.point_data["forest_control"])
    forest_size_lower = _normalize_str_array(polydata.point_data["forest_size"])

    forest_precolonial = _coerce_bool_array(polydata.point_data["forest_precolonial"])
    recruit_indicator = _coerce_bool_array(polydata.point_data["indicator_Tree_generations_grassland"])
    peeling_indicator = _coerce_bool_array(polydata.point_data["indicator_Bird_self_peeling"])

    recruit_opportunity = get_points_within_distance(
        polydata,
        get_distance_reference_mask(polydata, "canopy-feature"),
        20.0,
    ) & (~get_building_mask(polydata))

    proposal_decay = _assign_proposal_labels(
        _empty_proposal_labels(n_points),
        np.isin(
            scenario_rewilded_lower,
            ["exoskeleton", "footprint-depaved", "node-rewilded", "rewilded"],
        ),
        [
            ("buffer-feature", np.isin(scenario_bio_envelope_lower, list(DECAY_BUFFER_VALUES))),
            ("brace-feature", scenario_bio_envelope_lower == "exoskeleton"),
        ],
        "decay",
    )

    proposal_recruit = _assign_proposal_labels(
        _empty_proposal_labels(n_points),
        recruit_opportunity,
        [
            (
                "buffer-feature",
                recruit_indicator & np.isin(scenario_bio_envelope_lower, list(RECRUIT_BUFFER_VALUES)),
            ),
            (
                "rewild-ground",
                recruit_indicator & np.isin(scenario_bio_envelope_lower, list(RECRUIT_REWILD_VALUES)),
            ),
        ],
        "recruit",
    )

    release_opportunity = search_bioavailable_lower == "arboreal"
    proposal_release_control = _assign_proposal_labels(
        _empty_proposal_labels(n_points),
        release_opportunity,
        [
            (
                "eliminate-pruning",
                release_opportunity
                & np.isin(
                    forest_control_lower,
                    ["reserve-tree", "reserve tree", "improved-tree", "improved tree"],
                ),
            ),
            (
                "reduce-pruning",
                release_opportunity & np.isin(forest_control_lower, ["park-tree", "park tree"]),
            ),
        ],
        "release-control",
    )

    colonise_opportunity = np.isin(scenario_outputs_lower, list(COLONISE_PROPOSAL_VALUES))
    proposal_colonise = _assign_proposal_labels(
        _empty_proposal_labels(n_points),
        colonise_opportunity,
        [
            ("rewild-ground", colonise_opportunity & np.isin(scenario_outputs_lower, list(COLONISE_REWILD_VALUES))),
            ("enrich-envelope", colonise_opportunity & np.isin(scenario_outputs_lower, list(COLONISE_ENRICH_VALUES))),
            ("roughen-envelope", colonise_opportunity & np.isin(scenario_outputs_lower, list(COLONISE_ROUGHEN_VALUES))),
        ],
        "colonise",
    )

    upgrade_mask = (~forest_precolonial) & peeling_indicator
    adapt_mask = (forest_size_lower == "artificial") & (~forest_precolonial) & (~upgrade_mask)
    proposal_deploy_structure = _assign_proposal_labels(
        _empty_proposal_labels(n_points),
        upgrade_mask | adapt_mask,
        [
            ("upgrade-feature", upgrade_mask),
            ("adapt-utility-pole", adapt_mask),
        ],
        "deploy-structure",
    )

    polydata.point_data["proposal_decay"] = proposal_decay
    polydata.point_data["proposal_recruit"] = proposal_recruit
    polydata.point_data["proposal_release_control"] = proposal_release_control
    polydata.point_data["proposal_colonise"] = proposal_colonise
    polydata.point_data["proposal_deploy_structure"] = proposal_deploy_structure
    return polydata


def ensure_v3_proposal_point_data(polydata):
    n_points = polydata.n_points
    proposal_decay = _empty_v3_decisions(n_points)
    proposal_release_control = _empty_v3_decisions(n_points)
    proposal_colonise = np.full(n_points, "proposal-colonise_rejected", dtype=PROPOSAL_LABEL_DTYPE)
    proposal_recruit = _empty_v3_decisions(n_points)
    proposal_deploy_structure = _empty_v3_decisions(n_points)

    decay_intervention = _empty_v3_interventions(n_points)
    release_control_intervention = _empty_v3_interventions(n_points)
    colonise_intervention = _empty_v3_interventions(n_points)
    recruit_intervention = _empty_v3_interventions(n_points)
    deploy_structure_intervention = _empty_v3_interventions(n_points)

    scenario_bio_envelope_lower = _normalize_str_array(_string_point_data(polydata, "scenario_bioEnvelope", "none"))
    scenario_outputs_lower = _normalize_str_array(_string_point_data(polydata, "scenario_outputs", "none"))
    search_bioavailable_lower = _normalize_str_array(_string_point_data(polydata, "search_bioavailable", "none"))
    forest_size_lower = _normalize_str_array(_string_point_data(polydata, "forest_size", "none"))
    forest_control_lower = _normalize_str_array(_string_point_data(polydata, "forest_control", "none"))

    forest_precolonial = (
        _coerce_bool_array(polydata.point_data["forest_precolonial"])
        if "forest_precolonial" in polydata.point_data
        else np.zeros(n_points, dtype=bool)
    )
    peeling_indicator = (
        _coerce_bool_array(polydata.point_data["indicator_Bird_self_peeling"])
        if "indicator_Bird_self_peeling" in polydata.point_data
        else np.zeros(n_points, dtype=bool)
    )
    recruit_indicator = (
        _coerce_bool_array(polydata.point_data["indicator_Tree_generations_grassland"])
        if "indicator_Tree_generations_grassland" in polydata.point_data
        else np.zeros(n_points, dtype=bool)
    )
    recruit_planting_mask = (
        np.asarray(polydata.point_data["scenario_rewildingPlantings"]).astype(float) >= 0
        if "scenario_rewildingPlantings" in polydata.point_data
        else np.zeros(n_points, dtype=bool)
    )

    forest_decay_decision = _string_point_data(polydata, "forest_proposal-decay_decision", "not-assessed")
    forest_decay_intervention = _string_point_data(polydata, "forest_proposal-decay_intervention", "none")
    forest_release_decision = _string_point_data(polydata, "forest_proposal-release-control_decision", "not-assessed")
    forest_release_intervention = _string_point_data(polydata, "forest_proposal-release-control_intervention", "none")
    forest_deploy_decision = _string_point_data(polydata, "forest_proposal-deploy-structure_decision", "not-assessed")
    forest_deploy_intervention = _string_point_data(polydata, "forest_proposal-deploy-structure_intervention", "none")

    forest_decay_decision_lower = _normalize_str_array(forest_decay_decision)
    forest_release_decision_lower = _normalize_str_array(forest_release_decision)
    forest_deploy_decision_lower = _normalize_str_array(forest_deploy_decision)

    forest_decay_mask = ~np.isin(forest_decay_decision_lower, ["", "nan", "not-assessed"])
    proposal_decay[forest_decay_mask] = forest_decay_decision[forest_decay_mask]
    decay_intervention[forest_decay_mask] = forest_decay_intervention[forest_decay_mask]
    decay_buffer_mask = np.isin(scenario_bio_envelope_lower, list(DECAY_BUFFER_VALUES))
    decay_brace_mask = scenario_bio_envelope_lower == "exoskeleton"
    proposal_decay[decay_buffer_mask | decay_brace_mask] = "proposal-decay_accepted"
    decay_intervention[decay_buffer_mask] = "buffer-feature"
    decay_intervention[decay_brace_mask] = "brace-feature"

    release_opportunity = search_bioavailable_lower == "arboreal"
    forest_release_mask = ~np.isin(forest_release_decision_lower, ["", "nan", "not-assessed"])
    proposal_release_control[forest_release_mask] = forest_release_decision[forest_release_mask]
    release_control_intervention[forest_release_mask] = forest_release_intervention[forest_release_mask]
    proposal_release_control[release_opportunity & (release_control_intervention == "none")] = "proposal-release-control_rejected"
    release_control_intervention[
        release_opportunity
        & np.isin(forest_control_lower, ["park-tree", "park tree"])
        & (release_control_intervention == "none")
    ] = "reduce-pruning"
    release_control_intervention[
        release_opportunity
        & np.isin(forest_control_lower, ["reserve-tree", "reserve tree", "improved-tree", "improved tree"])
        & (release_control_intervention == "none")
    ] = "eliminate-pruning"
    proposal_release_control[release_opportunity & (release_control_intervention != "none")] = "proposal-release-control_accepted"

    colonise_opportunity = np.isin(scenario_outputs_lower, list(COLONISE_PROPOSAL_VALUES))
    proposal_colonise[colonise_opportunity] = "proposal-colonise_accepted"
    colonise_intervention[np.isin(scenario_outputs_lower, list(COLONISE_REWILD_VALUES))] = "rewild-ground"
    colonise_intervention[np.isin(scenario_outputs_lower, list(COLONISE_ENRICH_VALUES))] = "enrich-envelope"
    colonise_intervention[np.isin(scenario_outputs_lower, list(COLONISE_ROUGHEN_VALUES))] = "roughen-envelope"

    recruit_buffer_opportunity = get_points_within_distance(
        polydata,
        get_distance_reference_mask(polydata, "canopy-feature"),
        20.0,
    ) & (~get_building_mask(polydata))
    recruit_consideration_mask = recruit_buffer_opportunity | recruit_planting_mask
    proposal_recruit[recruit_consideration_mask] = "proposal-recruit_rejected"
    recruit_intervention[recruit_indicator & np.isin(scenario_bio_envelope_lower, list(RECRUIT_BUFFER_VALUES))] = "buffer-feature"
    recruit_intervention[recruit_indicator & np.isin(scenario_bio_envelope_lower, list(RECRUIT_REWILD_VALUES))] = "rewild-ground"
    recruit_acceptance_mask = np.isin(recruit_intervention, ["buffer-feature", "rewild-ground"])
    proposal_recruit[recruit_acceptance_mask] = "proposal-recruit_accepted"

    forest_deploy_mask = ~np.isin(forest_deploy_decision_lower, ["", "nan", "not-assessed"])
    proposal_deploy_structure[forest_deploy_mask] = forest_deploy_decision[forest_deploy_mask]
    deploy_structure_intervention[forest_deploy_mask] = forest_deploy_intervention[forest_deploy_mask]
    adapt_mask = (forest_size_lower == "artificial") & (~forest_precolonial) & (deploy_structure_intervention == "none")
    proposal_deploy_structure[adapt_mask] = "proposal-deploy-structure_accepted"
    deploy_structure_intervention[adapt_mask] = "adapt-utility-pole"
    upgrade_mask = (~forest_precolonial) & peeling_indicator & (deploy_structure_intervention == "none")
    proposal_deploy_structure[upgrade_mask] = "proposal-deploy-structure_accepted"
    deploy_structure_intervention[upgrade_mask] = "upgrade-feature"

    polydata.point_data["proposal_decayV3"] = proposal_decay
    polydata.point_data["proposal_release_controlV3"] = proposal_release_control
    polydata.point_data["proposal_coloniseV3"] = proposal_colonise
    polydata.point_data["proposal_recruitV3"] = proposal_recruit
    polydata.point_data["proposal_deploy_structureV3"] = proposal_deploy_structure
    polydata.point_data["proposal_decayV3_intervention"] = decay_intervention
    polydata.point_data["proposal_release_controlV3_intervention"] = release_control_intervention
    polydata.point_data["proposal_coloniseV3_intervention"] = colonise_intervention
    polydata.point_data["proposal_recruitV3_intervention"] = recruit_intervention
    polydata.point_data["proposal_deploy_structureV3_intervention"] = deploy_structure_intervention
    blender_point_arrays = build_blender_proposal_framebuffer_arrays(polydata.point_data)
    for name, values in blender_point_arrays.items():
        polydata.point_data[name] = values
    return polydata


# =============================================================================
# SECTION 4: INDICATOR EVALUATION
# =============================================================================

def evaluate_query(polydata, query):
    """
    Evaluate a query string and return a boolean mask.
    
    Query syntax:
        'field > 0'        -> Voxels where field is positive
        'field == value'   -> Voxels where field equals value
        'ground_not_paved' -> Ground voxels not on paved surfaces
    """
    n_points = polydata.n_points
    
    # Query: 'field > 0'
    if ' > 0' in query:
        field = query.replace(' > 0', '').strip()
        if field not in polydata.point_data:
            return np.zeros(n_points, dtype=bool)
        data = polydata.point_data[field]
        if np.issubdtype(data.dtype, np.number):
            return data > 0
        else:
            return (data != 'none') & (data != '') & (data != 'nan')
    
    # Query: 'field == value'
    if ' == ' in query:
        field, value = query.split(' == ')
        field = field.strip()
        value = value.strip()
        if field not in polydata.point_data:
            return np.zeros(n_points, dtype=bool)
        return polydata.point_data[field] == value
    
    # Query: 'field in value1|value2|...'
    if ' in ' in query:
        field, values = query.split(' in ', 1)
        field = field.strip()
        if field not in polydata.point_data:
            return np.zeros(n_points, dtype=bool)
        value_list = [value.strip() for value in values.split('|') if value.strip()]
        return np.isin(_normalize_str_array(polydata.point_data[field]), [value.lower() for value in value_list])

    # Query: 'ground_not_paved'
    if query == 'ground_not_paved':
        ground = get_ground_mask(polydata)
        paved = get_paved_mask(polydata)
        return ground & ~paved
    
    raise ValueError(f"Unknown query syntax: {query}")


def apply_distance_filter(polydata, mask, indicator):
    """
    Apply distance filter to a mask if the indicator has distance settings.
    
    Args:
        polydata: The VTK polydata
        mask: Boolean mask from evaluate_query()
        indicator: The indicator dict (may contain distance_from, distance_type, distance_meters)
    
    Returns:
        Filtered boolean mask
    """
    if 'distance_from' not in indicator:
        return mask
    
    distance_from = indicator['distance_from']
    distance_type = indicator['distance_type']
    distance_meters = indicator['distance_meters']
    
    # Get reference points
    reference_mask = get_distance_reference_mask(polydata, distance_from)
    
    # Find points within distance of reference
    within_distance = get_points_within_distance(polydata, reference_mask, distance_meters)
    
    # Apply filter based on type
    if distance_type == 'within':
        return mask & within_distance
    elif distance_type == 'outside':
        return mask & ~within_distance
    else:
        raise ValueError(f"Unknown distance_type: {distance_type}")


def apply_indicators(polydata):
    """Apply all indicator definitions to create boolean layers in polydata."""
    print(f"\nApplying {len(INDICATORS)} indicators...")
    
    results = {}
    
    # Pre-compute building mask once if any indicator needs ground_only
    building_mask = None
    
    for indicator in INDICATORS:
        ind_id = indicator['id']
        layer_name = f"indicator_{ind_id.replace('.', '_')}"
        
        # Evaluate the base query
        mask = evaluate_query(polydata, indicator['query'])
        
        # Apply distance filter if specified
        mask = apply_distance_filter(polydata, mask, indicator)
        
        # Apply ground_only filter if specified (exclude building voxels)
        if indicator.get('ground_only', False):
            if building_mask is None:
                building_mask = get_building_mask(polydata)
            mask = mask & ~building_mask
        
        polydata.point_data[layer_name] = mask
        
        count = np.sum(mask)
        pct = 100.0 * count / polydata.n_points
        results[ind_id] = count
        
        print(f"  {ind_id}: {count:,} voxels ({pct:.2f}%)")
    
    return polydata, results


# =============================================================================
# SECTION 5: SUPPORT ACTION COUNTING
# =============================================================================

def count_support_actions(polydata, indicator_id, indicator_mask):
    """
    Count support actions for a single indicator.
    
    Returns a list of dicts with action breakdowns.
    """
    if indicator_id not in SUPPORT_ACTIONS:
        return []
    
    config = SUPPORT_ACTIONS[indicator_id]
    breakdown = config['breakdown']
    records = []
    
    # Count by control level (for canopy indicators)
    if breakdown == 'control_level' and 'forest_control' in polydata.point_data:
        control = polydata.point_data['forest_control']
        for level_name, control_types in CONTROL_LEVELS.items():
            level_mask = np.zeros(polydata.n_points, dtype=bool)
            for ct in control_types:
                level_mask |= (control == ct)
            count = np.sum(indicator_mask & level_mask)
            records.append({
                'indicator_id': indicator_id,
                'action_type': 'control_level',
                'action_value': level_name,
                'count': count
            })
    
    # Count by urban element (for ground indicators)
    if breakdown == 'urban_element' and 'search_urban_elements' in polydata.point_data:
        urban = polydata.point_data['search_urban_elements']
        for element in URBAN_ELEMENTS:
            element_mask = (urban == element)
            count = np.sum(indicator_mask & element_mask)
            records.append({
                'indicator_id': indicator_id,
                'action_type': 'urban_element',
                'action_value': element,
                'count': count
            })
    
    # Count by rewilding status
    if breakdown == 'rewilding_status' and 'scenario_rewilded' in polydata.point_data:
        rewilded = polydata.point_data['scenario_rewilded']
        for rwild_type in REWILDING_TYPES:
            rwild_mask = (rewilded == rwild_type)
            count = np.sum(indicator_mask & rwild_mask)
            records.append({
                'indicator_id': indicator_id,
                'action_type': 'rewilding_status',
                'action_value': rwild_type,
                'count': count
            })

    # Bird actions: artificial structures deployed
    if indicator_id.startswith('Bird.') and 'forest_size' in polydata.point_data:
        forest_size = polydata.point_data['forest_size']
        artificial_mask = (forest_size == 'artificial')
        count = np.sum(indicator_mask & artificial_mask)
        records.append({
            'indicator_id': indicator_id,
            'action_type': 'artificial_structures_deployed',
            'action_value': 'artificial',
            'count': count
        })
    
    # Also count artificial (non-precolonial) if specified
    if config.get('also_count') == 'artificial' and 'forest_precolonial' in polydata.point_data:
        precol = polydata.point_data['forest_precolonial']
        if precol.dtype == bool:
            artificial_mask = ~precol
        else:
            artificial_mask = (precol == False) | (precol == 'False') | (precol == 0)
        count = np.sum(indicator_mask & artificial_mask)
        records.append({
            'indicator_id': indicator_id,
            'action_type': 'artificial',
            'action_value': 'installed',
            'count': count
        })
    
    return records


def gather_all_support_actions(polydata):
    """Gather support action counts for all indicators."""
    print("\nGathering support action counts...")
    
    all_records = []
    
    for indicator in INDICATORS:
        ind_id = indicator['id']
        layer_name = f"indicator_{ind_id.replace('.', '_')}"
        
        if layer_name not in polydata.point_data:
            continue
        
        indicator_mask = polydata.point_data[layer_name]
        records = count_support_actions(polydata, ind_id, indicator_mask)
        all_records.extend(records)
    
    return all_records


# =============================================================================
# SECTION 6: FILE PROCESSING
# =============================================================================

def get_vtk_path(site, scenario, year, voxel_size=1, output_mode=None):
    """Construct path to the best available VTK for compatibility processing."""
    data_dir = scenario_output_root(output_mode)
    voxel = format_voxel_size(voxel_size)

    if scenario == 'baseline':
        final_path = engine_output_baseline_state_vtk_path(site, voxel_size, output_mode)
        if final_path.exists():
            return final_path
        urban_features_path = scenario_baseline_combined_vtk_path(site, voxel_size, output_mode).with_name(
            f'{site}_baseline_combined_{voxel}_urban_features.vtk'
        )
        if urban_features_path.exists():
            return urban_features_path
        combined_path = scenario_baseline_combined_vtk_path(site, voxel_size, output_mode)
        if combined_path.exists():
            return combined_path
        raise FileNotFoundError(f"No baseline VTK found for {site}")

    final_path = engine_output_state_vtk_path(site, scenario, year, voxel_size, output_mode)
    if final_path.exists():
        return final_path

    base = data_dir / site
    candidates = [
        base / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features.vtk",
        base / f"{site}_{voxel}_{scenario}_scenarioYR{year}_urban_features.vtk",
        base / f"{site}_{scenario}_{voxel}_scenarioYR{year}.vtk",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No VTK found for {site}/{scenario}/year{year}")


def _build_indicator_action_frames(polydata, results, site, scenario, year, voxel_size):
    indicator_records = []
    for indicator in INDICATORS:
        ind_id = indicator['id']
        indicator_records.append({
            'site': site,
            'scenario': scenario,
            'year': year,
            'indicator_id': ind_id,
            'persona': indicator['persona'],
            'capability': indicator['capability'],
            'label': indicator['label'],
            'count': results.get(ind_id, 0),
            'voxel_size': voxel_size,
        })

    action_records = gather_all_support_actions(polydata)
    for record in action_records:
        record.update({
            'site': site,
            'scenario': scenario,
            'year': year,
            'voxel_size': voxel_size,
        })

    return pd.DataFrame(indicator_records), pd.DataFrame(action_records)


def _write_per_state_stats(indicator_df, action_df, site, scenario, year, voxel_size=1, output_mode=None):
    if scenario == 'baseline':
        indicator_path = engine_output_baseline_indicator_counts_path(site, voxel_size, output_mode)
        action_path = engine_output_baseline_action_counts_path(site, voxel_size, output_mode)
    else:
        indicator_path = engine_output_state_indicator_counts_path(site, scenario, year, voxel_size, output_mode)
        action_path = engine_output_state_action_counts_path(site, scenario, year, voxel_size, output_mode)

    if not indicator_df.empty:
        indicator_df.to_csv(indicator_path, index=False)
    if not action_df.empty:
        action_df.to_csv(action_path, index=False)


def merge_site_stats(site, scenarios=None, years=None, voxel_size=1, include_baseline=True, output_mode=None):
    if scenarios is None:
        scenarios = ['positive', 'trending']
    if years is None:
        years = DEFAULT_YEARS

    indicator_frames = []
    action_frames = []
    baseline_counts = {}

    if include_baseline:
        baseline_indicator_path = engine_output_baseline_indicator_counts_path(site, voxel_size, output_mode)
        baseline_action_path = engine_output_baseline_action_counts_path(site, voxel_size, output_mode)
        if baseline_indicator_path.exists():
            baseline_indicator_df = pd.read_csv(baseline_indicator_path)
            indicator_frames.append(baseline_indicator_df)
            for _, row in baseline_indicator_df.iterrows():
                baseline_counts[row['indicator_id']] = row['count']
        if baseline_action_path.exists():
            action_frames.append(pd.read_csv(baseline_action_path))

    for scenario in scenarios:
        for year in years:
            indicator_path = engine_output_state_indicator_counts_path(site, scenario, year, voxel_size, output_mode)
            action_path = engine_output_state_action_counts_path(site, scenario, year, voxel_size, output_mode)
            if indicator_path.exists():
                indicator_frames.append(pd.read_csv(indicator_path))
            if action_path.exists():
                action_frames.append(pd.read_csv(action_path))

    indicator_df = pd.concat(indicator_frames, ignore_index=True) if indicator_frames else pd.DataFrame()
    action_df = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()

    if not indicator_df.empty and baseline_counts:
        def calc_pct(row):
            baseline = baseline_counts.get(row['indicator_id'], 0)
            if baseline > 0:
                return round(row['count'] / baseline * 100, 1)
            return None
        indicator_df['pct_of_baseline'] = indicator_df.apply(calc_pct, axis=1)

    csv_dir = refactor_statistics_root(output_mode) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    if not indicator_df.empty:
        indicator_df.to_csv(csv_dir / f'{site}_{voxel_size}_indicator_counts.csv', index=False)
    if not action_df.empty:
        action_df.to_csv(csv_dir / f'{site}_{voxel_size}_action_counts.csv', index=False)

    return indicator_df, action_df


def process_polydata(polydata, site, scenario, year, voxel_size=1, save_vtk=True, save_stats=True, output_mode=None):
    """
    Process a single in-memory polydata: apply indicators, gather counts, and optionally save.
    
    Returns:
        indicator_df: DataFrame of indicator counts
        action_df: DataFrame of support action counts
        polydata: The processed polydata (with indicator/proposal layers)
    """
    print(f"\n{'='*60}")
    print(f"Processing in-memory state: {site} / {scenario} / yr{year}")
    print(f"Loaded {polydata.n_points:,} points")
    print(f"{'='*60}")

    polydata, results = apply_indicators(polydata)
    polydata = add_proposal_point_data(polydata)
    polydata = ensure_v3_proposal_point_data(polydata)

    indicator_df, action_df = _build_indicator_action_frames(
        polydata, results, site, scenario, year, voxel_size
    )

    if save_stats:
        _write_per_state_stats(indicator_df, action_df, site, scenario, year, voxel_size, output_mode)

    if not a_helper_functions.export_all_pointdata_variables():
        polydata = a_helper_functions.drop_polydata_point_arrays_if_present(
            polydata,
            a_helper_functions.LEAN_EXPORT_POINTDATA_DROP_ARRAYS,
        )
    
    if save_vtk:
        if scenario == 'baseline':
            output_path = engine_output_baseline_state_vtk_path(site, voxel_size, output_mode)
        else:
            output_path = engine_output_state_vtk_path(site, scenario, year, voxel_size, output_mode)
        polydata.save(str(output_path))
        print(f"\nSaved: {output_path}")

    return indicator_df, action_df, polydata


def process_vtk(vtk_path, site, scenario, year, voxel_size=1, save_vtk=True, save_stats=True, output_mode=None):
    """
    Compatibility wrapper that loads a VTK from disk and passes it to process_polydata.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {vtk_path.name}")
    print(f"{'='*60}")
    polydata = pv.read(str(vtk_path))
    print(f"Loaded {polydata.n_points:,} points")
    return process_polydata(
        polydata,
        site,
        scenario,
        year,
        voxel_size=voxel_size,
        save_vtk=save_vtk,
        save_stats=save_stats,
        output_mode=output_mode,
    )


def process_site(
    site,
    scenarios=None,
    years=None,
    voxel_size=1,
    save_vtk=True,
    include_baseline=True,
    output_mode=None,
):
    """
    Process all VTK files for a site, including baseline.
    
    Returns:
        all_indicator_counts: Combined DataFrame of indicator counts (with pct_of_baseline)
        all_action_counts: Combined DataFrame of action counts
    """
    if scenarios is None:
        scenarios = ['positive', 'trending']
    if years is None:
        years = DEFAULT_YEARS
    
    print(f"\n{'#'*60}")
    print(f"# PROCESSING SITE: {site}")
    print(f"# Scenarios: {scenarios}")
    print(f"# Years: {years}")
    print(f"# Include baseline: {include_baseline}")
    print(f"{'#'*60}")
    
    all_indicator_frames = []
    all_action_frames = []

    # Process baseline first (scenario='baseline', year=-180)
    if include_baseline:
        try:
            vtk_path = get_vtk_path(site, 'baseline', -180, voxel_size, output_mode)
            indicator_df, action_df, _ = process_vtk(
                vtk_path, site, 'baseline', -180, voxel_size, save_vtk, False, output_mode
            )
            if not indicator_df.empty:
                all_indicator_frames.append(indicator_df)
            if not action_df.empty:
                all_action_frames.append(action_df)
        except FileNotFoundError as e:
            print(f"Skipping baseline: {e}")
    
    # Process each scenario and year
    for scenario in scenarios:
        for year in years:
            try:
                vtk_path = get_vtk_path(site, scenario, year, voxel_size, output_mode)
                indicator_df, action_df, _ = process_vtk(
                    vtk_path, site, scenario, year, voxel_size, save_vtk, False, output_mode
                )
                if not indicator_df.empty:
                    all_indicator_frames.append(indicator_df)
                if not action_df.empty:
                    all_action_frames.append(action_df)
            except FileNotFoundError as e:
                print(f"Skipping: {e}")

    if not all_indicator_frames and not all_action_frames:
        return pd.DataFrame(), pd.DataFrame()

    per_state_dir = engine_output_root(output_mode) / "stats" / "per-state" / site
    per_state_dir.mkdir(parents=True, exist_ok=True)

    if include_baseline and all_indicator_frames:
        baseline_indicator_df = next((df for df in all_indicator_frames if not df.empty and (df["scenario"] == "baseline").all()), None)
        baseline_action_df = next((df for df in all_action_frames if not df.empty and (df["scenario"] == "baseline").all()), None)
        if baseline_indicator_df is not None:
            _write_per_state_stats(baseline_indicator_df, baseline_action_df if baseline_action_df is not None else pd.DataFrame(), site, 'baseline', -180, voxel_size, output_mode)

    for df in all_indicator_frames:
        if df.empty or (df["scenario"] == "baseline").all():
            continue
        first = df.iloc[0]
        matching_action_df = next(
            (
                action_df
                for action_df in all_action_frames
                if not action_df.empty
                and action_df.iloc[0]["scenario"] == first["scenario"]
                and int(action_df.iloc[0]["year"]) == int(first["year"])
            ),
            pd.DataFrame(),
        )
        _write_per_state_stats(df, matching_action_df, site, first["scenario"], int(first["year"]), voxel_size, output_mode)

    return merge_site_stats(site, scenarios=scenarios, years=years, voxel_size=voxel_size, include_baseline=include_baseline, output_mode=output_mode)


# =============================================================================
# SECTION 7: MAIN ENTRY POINT
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Apply capability indicators to scenario VTK files'
    )
    parser.add_argument('--site', type=str, default='trimmed-parade')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Single scenario, or omit for all')
    parser.add_argument('--years', type=str, default=None,
                        help='Comma-separated assessed years, for example 0,10,30,60,90,120,150,180')
    parser.add_argument('--year', type=int, default=None,
                        help='Single year, or omit for all')
    parser.add_argument('--interval', type=int, default=None,
                        help='Sub-timestep interval (e.g., 30 for years 90, 120, 150)')
    parser.add_argument('--voxel-size', type=float, default=1)
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save VTK files')
    parser.add_argument(
        '--output-mode',
        type=str,
        default=None,
        choices=['canonical', 'validation'],
        help='Output root mode for reading scenario files and writing engine outputs.',
    )
    
    args = parser.parse_args()
    
    scenarios = [args.scenario] if args.scenario else None
    
    # Handle years with optional interval
    if args.years:
        years = [int(item.strip()) for item in args.years.split(',') if item.strip()]
    elif args.year is not None:
        years = [args.year]
    elif args.interval is not None:
        years = params_v3.generate_timesteps(interval=args.interval)
        print(f"Generated timesteps with interval {args.interval}: {years}")
    else:
        years = None  # Uses DEFAULT_YEARS
    
    output_mode = normalize_output_mode(args.output_mode)

    indicator_df, action_df = process_site(
        args.site,
        scenarios,
        years,
        args.voxel_size,
        save_vtk=not args.no_save,
        output_mode=output_mode,
    )
    
    # Save CSVs to csv subfolder
    csv_dir = refactor_statistics_root(output_mode) / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    if not indicator_df.empty:
        indicator_path = csv_dir / f'{args.site}_{args.voxel_size}_indicator_counts.csv'
        indicator_df.to_csv(indicator_path, index=False)
        print(f"\nSaved: {indicator_path}")
    
    if not action_df.empty:
        action_path = csv_dir / f'{args.site}_{args.voxel_size}_action_counts.csv'
        action_df.to_csv(action_path, index=False)
        print(f"Saved: {action_path}")


if __name__ == '__main__':
    main()
