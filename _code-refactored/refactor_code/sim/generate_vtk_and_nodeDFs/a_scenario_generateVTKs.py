


#==============================================================================
# IMPORTS
#==============================================================================
import os
import pandas as pd
import xarray as xr
import numpy as np
import pyvista as pv
import sys
from pathlib import Path
from scipy.spatial import cKDTree

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_code-refactored")
REPO_ROOT = CODE_ROOT.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.sim.generate_interim_state_data.engine_v3 import calculate_under_node_treatment_status
from refactor_code.sim.setup import params_v3
from refactor_code.sim.setup.constants import (
    COLONISE_FULL_ENVELOPE,
    COLONISE_FULL_GROUND,
    COLONISE_PARTIAL_ENVELOPE,
    DECAY_FULL,
    DECAY_PARTIAL,
    DEPLOY_FULL_POLE,
    DEPLOY_FULL_UPGRADE,
    RECRUIT_FULL,
    RECRUIT_PARTIAL,
    RELEASECONTROL_FULL,
    RELEASECONTROL_PARTIAL,
)
from refactor_code.sim.voxel import voxel_a_helper_functions as a_helper_functions
from refactor_code.sim.voxel import voxel_a_voxeliser as a_voxeliser

from refactor_code.paths import (
    engine_output_nodedf_path,
    scenario_log_df_path,
    scenario_node_df_path,
    scenario_state_vtk_path,
    scenario_tree_df_path,
    scenario_pole_df_path,
)
from refactor_code.blender.bexport.proposal_framebuffers import build_blender_proposal_framebuffer_columns
from refactor_code.blender.bexport.proposal_framebuffers_vtk import build_blender_proposal_framebuffer_arrays

VTK_PROPOSAL_LABEL_DTYPE = "<U64"
BUILDING_URBAN_VALUES = {"facade", "green roof", "brown roof"}

# --- Proposal intervention string constants (imported from constants.py) ------
PROPOSAL_DECAY_BUFFER_INTERVENTION = DECAY_FULL
PROPOSAL_DECAY_BRACE_INTERVENTION = DECAY_PARTIAL
PROPOSAL_RELEASE_CONTROL_REDUCE_INTERVENTION = RELEASECONTROL_PARTIAL
PROPOSAL_RELEASE_CONTROL_ELIMINATE_INTERVENTION = RELEASECONTROL_FULL
PROPOSAL_COLONISE_REWILD_INTERVENTION = COLONISE_FULL_GROUND
PROPOSAL_COLONISE_ENRICH_INTERVENTION = COLONISE_FULL_ENVELOPE
PROPOSAL_COLONISE_ROUGHEN_INTERVENTION = COLONISE_PARTIAL_ENVELOPE
PROPOSAL_RECRUIT_BUFFER_INTERVENTION = RECRUIT_PARTIAL
PROPOSAL_RECRUIT_REWILD_INTERVENTION = RECRUIT_FULL
PROPOSAL_DEPLOY_STRUCTURE_ADAPT_INTERVENTION = DEPLOY_FULL_POLE
PROPOSAL_DEPLOY_STRUCTURE_UPGRADE_INTERVENTION = DEPLOY_FULL_UPGRADE

# --- Proposal intervention value sets ----------------------------------------
# scenario_bioEnvelope values that trigger each decay intervention
PROPOSAL_DECAY_BUFFER_INTERVENTION_VALUES = {"node-rewilded", "footprint-depaved"}
PROPOSAL_DECAY_BRACE_INTERVENTION_VALUES = {"exoskeleton"}

# scenario_outputs values that trigger colonise acceptance / interventions
PROPOSAL_COLONISE_OPPORTUNITY_VALUES = {
    "brownroof",
    "greenroof",
    "livingfacade",
    "footprint-depaved",
    "node-rewilded",
    "otherground",
    "rewilded",
}
PROPOSAL_COLONISE_REWILD_INTERVENTION_VALUES = {"node-rewilded", "footprint-depaved", "rewilded"}
PROPOSAL_COLONISE_ENRICH_INTERVENTION_VALUES = {"greenroof"}
PROPOSAL_COLONISE_ROUGHEN_INTERVENTION_VALUES = {"brownroof", "livingfacade"}

# scenario_bioEnvelope values that trigger recruit interventions
PROPOSAL_RECRUIT_BUFFER_INTERVENTION_VALUES = {"footprint-depaved"}
PROPOSAL_RECRUIT_REWILD_INTERVENTION_VALUES = {"node-rewilded", "otherground", "rewilded"}
PROPOSAL_RECRUIT_DISTANCE_M = 20.0

# forest_control values that trigger release-control interventions
PROPOSAL_RELEASE_CONTROL_REDUCE_INTERVENTION_VALUES = {"park-tree", "park tree"}
PROPOSAL_RELEASE_CONTROL_ELIMINATE_INTERVENTION_VALUES = {"reserve-tree", "reserve tree", "improved-tree", "improved tree"}


def _normalize_str_array(values):
    return np.char.lower(np.asarray(values).astype(str))


def _empty_proposal_labels(size):
    return np.full(size, "none", dtype=VTK_PROPOSAL_LABEL_DTYPE)


def _assign_proposal_labels(labels, opportunity_mask, intervention_masks, proposal_name):
    labels[opportunity_mask] = f"{proposal_name}-other"
    for intervention_name, mask in intervention_masks:
        labels[mask] = f"{proposal_name}_{intervention_name}"
    return labels


def _empty_v3_decisions(size):
    return np.full(size, "not-assessed", dtype=VTK_PROPOSAL_LABEL_DTYPE)


def _empty_v3_interventions(size):
    return np.full(size, "none", dtype=VTK_PROPOSAL_LABEL_DTYPE)


def _coerce_bool_array(values):
    array = np.asarray(values)
    if array.dtype == bool:
        return array
    if np.issubdtype(array.dtype, np.number):
        return np.nan_to_num(array.astype(float), nan=0.0) != 0
    normalized = _normalize_str_array(array)
    return np.isin(normalized, ["true", "1", "yes", "y", "t"])


def _string_ds_values(ds, name, default):
    if name not in ds.variables:
        return np.full(ds.sizes["voxel"], default, dtype=VTK_PROPOSAL_LABEL_DTYPE)
    values = np.asarray(ds[name].values).astype(str)
    if values.shape[0] != ds.sizes["voxel"]:
        return np.full(ds.sizes["voxel"], default, dtype=VTK_PROPOSAL_LABEL_DTYPE)
    cleaned = values.copy()
    cleaned[np.isin(_normalize_str_array(cleaned), ["", "nan"])] = default
    return cleaned.astype(VTK_PROPOSAL_LABEL_DTYPE)


def _points_within_distance(points, source_mask, distance_m):
    if points.size == 0 or not np.any(source_mask):
        return np.zeros(len(points), dtype=bool)
    tree = cKDTree(points[source_mask])
    distances, _ = tree.query(points, k=1, distance_upper_bound=distance_m)
    return np.isfinite(distances)


def _build_building_mask(search_urban_elements):
    search_urban_elements_lower = _normalize_str_array(search_urban_elements)
    return np.isin(search_urban_elements_lower, list(BUILDING_URBAN_VALUES))


def create_proposal_point_data(ds):
    required_vars = {
        "forest_control",
        "forest_precolonial",
        "forest_size",
        "scenario_bioEnvelope",
        "scenario_outputs",
        "scenario_under-node-treatment",
        "search_bioavailable",
        "search_urban_elements",
        "indicator_Bird_self_peeling",
        "indicator_Tree_generations_grassland",
    }
    missing_vars = sorted(var for var in required_vars if var not in ds.variables)
    if missing_vars:
        print(f"Skipping proposal point-data creation; missing arrays: {missing_vars}")
        return ds

    voxel_count = ds.sizes["voxel"]
    scenario_under_node_treatment_lower = _normalize_str_array(ds["scenario_under-node-treatment"].values)
    scenario_bio_envelope_lower = _normalize_str_array(ds["scenario_bioEnvelope"].values)
    scenario_outputs_lower = _normalize_str_array(ds["scenario_outputs"].values)
    forest_control_lower = _normalize_str_array(ds["forest_control"].values)
    forest_size_lower = _normalize_str_array(ds["forest_size"].values)
    search_bioavailable_lower = _normalize_str_array(ds["search_bioavailable"].values)
    search_urban_elements = ds["search_urban_elements"].values
    forest_precolonial = np.asarray(ds["forest_precolonial"].values, dtype=bool)
    peeling_indicator = np.asarray(ds["indicator_Bird_self_peeling"].values, dtype=bool)
    recruit_indicator = np.asarray(ds["indicator_Tree_generations_grassland"].values, dtype=bool)

    points = np.vstack(
        (
            ds["centroid_x"].values,
            ds["centroid_y"].values,
            ds["centroid_z"].values,
        )
    ).T

    canopy_feature_mask = ~np.isin(forest_size_lower, ["", "nan", "none"])
    if "stat_fallen log" in ds.variables:
        fallen_log = np.asarray(ds["stat_fallen log"].values)
        if np.issubdtype(fallen_log.dtype, np.number):
            canopy_feature_mask |= fallen_log > 0
    recruit_opportunity = _points_within_distance(points, canopy_feature_mask, PROPOSAL_RECRUIT_DISTANCE_M) & (
        ~_build_building_mask(search_urban_elements)
    )

    proposal_decay = _assign_proposal_labels(
        _empty_proposal_labels(voxel_count),
        np.isin(
            scenario_under_node_treatment_lower,
            list(PROPOSAL_DECAY_BUFFER_INTERVENTION_VALUES | PROPOSAL_DECAY_BRACE_INTERVENTION_VALUES | {"rewilded"}),
        ),
        [
            (PROPOSAL_DECAY_BUFFER_INTERVENTION, np.isin(scenario_bio_envelope_lower, list(PROPOSAL_DECAY_BUFFER_INTERVENTION_VALUES))),
            (PROPOSAL_DECAY_BRACE_INTERVENTION, np.isin(scenario_bio_envelope_lower, list(PROPOSAL_DECAY_BRACE_INTERVENTION_VALUES))),
        ],
        "decay",
    )

    proposal_recruit = _assign_proposal_labels(
        _empty_proposal_labels(voxel_count),
        recruit_opportunity,
        [
            (
                PROPOSAL_RECRUIT_BUFFER_INTERVENTION,
                recruit_indicator & np.isin(scenario_bio_envelope_lower, list(PROPOSAL_RECRUIT_BUFFER_INTERVENTION_VALUES)),
            ),
            (
                PROPOSAL_RECRUIT_REWILD_INTERVENTION,
                recruit_indicator & np.isin(scenario_bio_envelope_lower, list(PROPOSAL_RECRUIT_REWILD_INTERVENTION_VALUES)),
            ),
        ],
        "recruit",
    )

    release_opportunity = search_bioavailable_lower == "arboreal"
    proposal_release_control = _assign_proposal_labels(
        _empty_proposal_labels(voxel_count),
        release_opportunity,
        [
            (
                PROPOSAL_RELEASE_CONTROL_ELIMINATE_INTERVENTION,
                release_opportunity
                & np.isin(forest_control_lower, list(PROPOSAL_RELEASE_CONTROL_ELIMINATE_INTERVENTION_VALUES)),
            ),
            (
                PROPOSAL_RELEASE_CONTROL_REDUCE_INTERVENTION,
                release_opportunity & np.isin(forest_control_lower, list(PROPOSAL_RELEASE_CONTROL_REDUCE_INTERVENTION_VALUES)),
            ),
        ],
        "release-control",
    )

    colonise_opportunity = np.isin(scenario_outputs_lower, list(PROPOSAL_COLONISE_OPPORTUNITY_VALUES))
    proposal_colonise = _assign_proposal_labels(
        _empty_proposal_labels(voxel_count),
        colonise_opportunity,
        [
            (PROPOSAL_COLONISE_REWILD_INTERVENTION, colonise_opportunity & np.isin(scenario_outputs_lower, list(PROPOSAL_COLONISE_REWILD_INTERVENTION_VALUES))),
            (PROPOSAL_COLONISE_ENRICH_INTERVENTION, colonise_opportunity & np.isin(scenario_outputs_lower, list(PROPOSAL_COLONISE_ENRICH_INTERVENTION_VALUES))),
            (PROPOSAL_COLONISE_ROUGHEN_INTERVENTION, colonise_opportunity & np.isin(scenario_outputs_lower, list(PROPOSAL_COLONISE_ROUGHEN_INTERVENTION_VALUES))),
        ],
        "colonise",
    )

    upgrade_mask = (~forest_precolonial) & peeling_indicator
    adapt_mask = (forest_size_lower == "artificial") & (~forest_precolonial) & (~upgrade_mask)
    proposal_deploy_structure = _assign_proposal_labels(
        _empty_proposal_labels(voxel_count),
        upgrade_mask | adapt_mask,
        [
            (PROPOSAL_DEPLOY_STRUCTURE_UPGRADE_INTERVENTION, upgrade_mask),
            (PROPOSAL_DEPLOY_STRUCTURE_ADAPT_INTERVENTION, adapt_mask),
        ],
        "deploy-structure",
    )

    ds["proposal_decay"] = xr.DataArray(proposal_decay, dims="voxel")
    ds["proposal_recruit"] = xr.DataArray(proposal_recruit, dims="voxel")
    ds["proposal_release_control"] = xr.DataArray(proposal_release_control, dims="voxel")
    ds["proposal_colonise"] = xr.DataArray(proposal_colonise, dims="voxel")
    ds["proposal_deploy_structure"] = xr.DataArray(proposal_deploy_structure, dims="voxel")
    return ds


def create_v3_proposal_point_data(ds):
    voxel_count = ds.sizes["voxel"]
    proposal_decay = _empty_v3_decisions(voxel_count)
    proposal_release_control = _empty_v3_decisions(voxel_count)
    proposal_colonise = np.full(voxel_count, "proposal-colonise_rejected", dtype=VTK_PROPOSAL_LABEL_DTYPE)
    proposal_recruit = _empty_v3_decisions(voxel_count)
    proposal_deploy_structure = _empty_v3_decisions(voxel_count)

    decay_intervention = _empty_v3_interventions(voxel_count)
    release_control_intervention = _empty_v3_interventions(voxel_count)
    colonise_intervention = _empty_v3_interventions(voxel_count)
    recruit_intervention = _empty_v3_interventions(voxel_count)
    deploy_structure_intervention = _empty_v3_interventions(voxel_count)

    scenario_bio_envelope_lower = _normalize_str_array(_string_ds_values(ds, "scenario_bioEnvelope", "none"))
    scenario_outputs_lower = _normalize_str_array(_string_ds_values(ds, "scenario_outputs", "none"))
    search_bioavailable_lower = _normalize_str_array(_string_ds_values(ds, "search_bioavailable", "none"))
    search_urban_elements = _string_ds_values(ds, "search_urban_elements", "none")
    search_urban_elements_lower = _normalize_str_array(search_urban_elements)
    forest_size_lower = _normalize_str_array(_string_ds_values(ds, "forest_size", "none"))
    forest_control_lower = _normalize_str_array(_string_ds_values(ds, "forest_control", "none"))

    forest_precolonial = (
        _coerce_bool_array(ds["forest_precolonial"].values)
        if "forest_precolonial" in ds.variables
        else np.zeros(voxel_count, dtype=bool)
    )
    peeling_indicator = (
        _coerce_bool_array(ds["indicator_Bird_self_peeling"].values)
        if "indicator_Bird_self_peeling" in ds.variables
        else np.zeros(voxel_count, dtype=bool)
    )
    recruit_indicator = (
        _coerce_bool_array(ds["indicator_Tree_generations_grassland"].values)
        if "indicator_Tree_generations_grassland" in ds.variables
        else np.zeros(voxel_count, dtype=bool)
    )

    points = np.vstack(
        (
            ds["centroid_x"].values,
            ds["centroid_y"].values,
            ds["centroid_z"].values,
        )
    ).T

    canopy_feature_mask = ~np.isin(forest_size_lower, ["", "nan", "none"])
    if "stat_fallen log" in ds.variables:
        fallen_log = np.asarray(ds["stat_fallen log"].values)
        if np.issubdtype(fallen_log.dtype, np.number):
            canopy_feature_mask |= fallen_log > 0

    building_mask = np.isin(search_urban_elements_lower, list(BUILDING_URBAN_VALUES))
    recruit_buffer_opportunity = _points_within_distance(points, canopy_feature_mask, PROPOSAL_RECRUIT_DISTANCE_M) & (~building_mask)
    recruit_planting_mask = (
        np.asarray(ds["scenario_rewildingPlantings"].values) >= 0
        if "scenario_rewildingPlantings" in ds.variables
        else np.zeros(voxel_count, dtype=bool)
    )

    forest_decay_decision = _string_ds_values(ds, "forest_proposal-decay_decision", "not-assessed")
    forest_decay_intervention = _string_ds_values(ds, "forest_proposal-decay_intervention", "none")
    forest_release_decision = _string_ds_values(ds, "forest_proposal-release-control_decision", "not-assessed")
    forest_release_intervention = _string_ds_values(ds, "forest_proposal-release-control_intervention", "none")
    forest_deploy_decision = _string_ds_values(ds, "forest_proposal-deploy-structure_decision", "not-assessed")
    forest_deploy_intervention = _string_ds_values(ds, "forest_proposal-deploy-structure_intervention", "none")

    forest_decay_decision_lower = _normalize_str_array(forest_decay_decision)
    forest_decay_intervention_lower = _normalize_str_array(forest_decay_intervention)
    forest_release_decision_lower = _normalize_str_array(forest_release_decision)
    forest_release_intervention_lower = _normalize_str_array(forest_release_intervention)
    forest_deploy_decision_lower = _normalize_str_array(forest_deploy_decision)
    forest_deploy_intervention_lower = _normalize_str_array(forest_deploy_intervention)

    forest_decay_mask = ~np.isin(forest_decay_decision_lower, ["", "nan", "not-assessed"])
    proposal_decay[forest_decay_mask] = forest_decay_decision[forest_decay_mask]
    decay_intervention[forest_decay_mask] = forest_decay_intervention[forest_decay_mask]

    decay_buffer_mask = np.isin(scenario_bio_envelope_lower, list(PROPOSAL_DECAY_BUFFER_INTERVENTION_VALUES))
    decay_brace_mask = np.isin(scenario_bio_envelope_lower, list(PROPOSAL_DECAY_BRACE_INTERVENTION_VALUES))
    proposal_decay[decay_buffer_mask | decay_brace_mask] = "proposal-decay_accepted"
    decay_intervention[decay_buffer_mask] = PROPOSAL_DECAY_BUFFER_INTERVENTION
    decay_intervention[decay_brace_mask] = PROPOSAL_DECAY_BRACE_INTERVENTION

    release_opportunity = search_bioavailable_lower == "arboreal"
    forest_release_mask = ~np.isin(forest_release_decision_lower, ["", "nan", "not-assessed"])
    proposal_release_control[forest_release_mask] = forest_release_decision[forest_release_mask]
    release_control_intervention[forest_release_mask] = forest_release_intervention[forest_release_mask]
    proposal_release_control[release_opportunity & (release_control_intervention == "none")] = "proposal-release-control_rejected"
    release_control_intervention[
        release_opportunity
        & np.isin(forest_control_lower, list(PROPOSAL_RELEASE_CONTROL_REDUCE_INTERVENTION_VALUES))
        & (release_control_intervention == "none")
    ] = PROPOSAL_RELEASE_CONTROL_REDUCE_INTERVENTION
    release_control_intervention[
        release_opportunity
        & np.isin(forest_control_lower, list(PROPOSAL_RELEASE_CONTROL_ELIMINATE_INTERVENTION_VALUES))
        & (release_control_intervention == "none")
    ] = PROPOSAL_RELEASE_CONTROL_ELIMINATE_INTERVENTION
    proposal_release_control[release_opportunity & (release_control_intervention != "none")] = "proposal-release-control_accepted"

    colonise_opportunity = np.isin(scenario_outputs_lower, list(PROPOSAL_COLONISE_OPPORTUNITY_VALUES))
    proposal_colonise[colonise_opportunity] = "proposal-colonise_accepted"
    colonise_intervention[np.isin(scenario_outputs_lower, list(PROPOSAL_COLONISE_REWILD_INTERVENTION_VALUES))] = PROPOSAL_COLONISE_REWILD_INTERVENTION
    colonise_intervention[np.isin(scenario_outputs_lower, list(PROPOSAL_COLONISE_ENRICH_INTERVENTION_VALUES))] = PROPOSAL_COLONISE_ENRICH_INTERVENTION
    colonise_intervention[np.isin(scenario_outputs_lower, list(PROPOSAL_COLONISE_ROUGHEN_INTERVENTION_VALUES))] = PROPOSAL_COLONISE_ROUGHEN_INTERVENTION

    recruit_consideration_mask = recruit_buffer_opportunity | recruit_planting_mask
    proposal_recruit[recruit_consideration_mask] = "proposal-recruit_rejected"
    recruit_intervention[recruit_indicator & np.isin(scenario_bio_envelope_lower, list(PROPOSAL_RECRUIT_BUFFER_INTERVENTION_VALUES))] = PROPOSAL_RECRUIT_BUFFER_INTERVENTION
    recruit_intervention[recruit_indicator & np.isin(scenario_bio_envelope_lower, list(PROPOSAL_RECRUIT_REWILD_INTERVENTION_VALUES))] = PROPOSAL_RECRUIT_REWILD_INTERVENTION
    recruit_acceptance_mask = np.isin(recruit_intervention, [PROPOSAL_RECRUIT_BUFFER_INTERVENTION, PROPOSAL_RECRUIT_REWILD_INTERVENTION])
    proposal_recruit[recruit_acceptance_mask] = "proposal-recruit_accepted"

    forest_deploy_mask = ~np.isin(forest_deploy_decision_lower, ["", "nan", "not-assessed"])
    proposal_deploy_structure[forest_deploy_mask] = forest_deploy_decision[forest_deploy_mask]
    deploy_structure_intervention[forest_deploy_mask] = forest_deploy_intervention[forest_deploy_mask]
    adapt_mask = (forest_size_lower == "artificial") & (~forest_precolonial) & (deploy_structure_intervention == "none")
    proposal_deploy_structure[adapt_mask] = "proposal-deploy-structure_accepted"
    deploy_structure_intervention[adapt_mask] = PROPOSAL_DEPLOY_STRUCTURE_ADAPT_INTERVENTION
    upgrade_mask = (~forest_precolonial) & peeling_indicator & (deploy_structure_intervention == "none")
    proposal_deploy_structure[upgrade_mask] = "proposal-deploy-structure_accepted"
    deploy_structure_intervention[upgrade_mask] = PROPOSAL_DEPLOY_STRUCTURE_UPGRADE_INTERVENTION

    ds["proposal_decayV4"] = xr.DataArray(proposal_decay, dims="voxel")
    ds["proposal_release_controlV4"] = xr.DataArray(proposal_release_control, dims="voxel")
    ds["proposal_coloniseV4"] = xr.DataArray(proposal_colonise, dims="voxel")
    ds["proposal_recruitV4"] = xr.DataArray(proposal_recruit, dims="voxel")
    ds["proposal_deploy_structureV4"] = xr.DataArray(proposal_deploy_structure, dims="voxel")

    ds["proposal_decayV4_intervention"] = xr.DataArray(decay_intervention, dims="voxel")
    ds["proposal_release_controlV4_intervention"] = xr.DataArray(release_control_intervention, dims="voxel")
    ds["proposal_coloniseV4_intervention"] = xr.DataArray(colonise_intervention, dims="voxel")
    ds["proposal_recruitV4_intervention"] = xr.DataArray(recruit_intervention, dims="voxel")
    ds["proposal_deploy_structureV4_intervention"] = xr.DataArray(deploy_structure_intervention, dims="voxel")

    blender_point_arrays = build_blender_proposal_framebuffer_arrays(
        {
            "proposal_decayV3": proposal_decay,
            "proposal_release_controlV3": proposal_release_control,
            "proposal_coloniseV3": proposal_colonise,
            "proposal_recruitV3": proposal_recruit,
            "proposal_deploy_structureV3": proposal_deploy_structure,
            "proposal_decayV3_intervention": decay_intervention,
            "proposal_release_controlV3_intervention": release_control_intervention,
            "proposal_coloniseV3_intervention": colonise_intervention,
            "proposal_recruitV3_intervention": recruit_intervention,
            "proposal_deploy_structureV3_intervention": deploy_structure_intervention,
        }
    )
    for name, values in blender_point_arrays.items():
        ds[name] = xr.DataArray(values, dims="voxel")

    return ds


#==============================================================================
# XARRAY PROCESSING FUNCTIONS
#==============================================================================
def create_under_node_treatment_variable(ds, df):
    """
    Updates the 'scenario_under-node-treatment' variable in the xarray dataset based on the dataframe values.
    Matches are made based on NodeID. Non-matching NodeIDs are ignored.
    
    Parameters:
    ds (xarray.Dataset): The xarray dataset containing voxel and node information.
    df (pandas.DataFrame): The dataframe containing NodeID and under-node treatment scenarios.
    
    Returns:
    xarray.Dataset: The updated dataset with the 'scenario_under-node-treatment' variable modified.
    
    Variables created/modified:
    - ds['scenario_under-node-treatment']: Categorical variable indicating under-node treatment status for each voxel
    """
    #--------------------------------------------------------------------------
    # STEP 1: PREPARE INPUT DATA
    # Standardize 'None' values to lowercase 'none' for consistency
    #--------------------------------------------------------------------------
    # Replace 'None' with 'none' in the DataFrame
    df['under-node-treatment'] = df['under-node-treatment'].replace('None', 'none')
    
    #--------------------------------------------------------------------------
    # STEP 2: INITIALIZE SCENARIO_REWILDED VARIABLE
    # Create the variable if it doesn't exist with default 'none' values
    #--------------------------------------------------------------------------
    if 'scenario_under-node-treatment' not in ds.variables:
        # Use object dtype for variable-length strings
        ds = ds.assign(**{"scenario_under-node-treatment": ('voxel', np.array(['none'] * ds.dims['voxel'], dtype='O'))})
    
    #--------------------------------------------------------------------------
    # STEP 3: EXTRACT REFERENCE ARRAYS
    # Get the arrays needed for node matching
    #--------------------------------------------------------------------------
    canopy_id = ds['node_CanopyID'].values
    sim_nodes = ds['sim_Nodes'].values
    
    #--------------------------------------------------------------------------
    # STEP 4: UPDATE SCENARIO_REWILDED BASED ON NODE MATCHING
    # Match voxels to nodes and update their rewilding status
    #--------------------------------------------------------------------------
    skipped_invalid_node_ids = 0
    for idx, row in df.iterrows():
        treatment_value = row['under-node-treatment']
        if treatment_value not in ['exoskeleton', 'footprint-depaved', 'node-rewilded']:
            continue

        raw_node_id = row.get('NodeID', np.nan)
        if pd.isna(raw_node_id):
            skipped_invalid_node_ids += 1
            continue

        try:
            node_id = int(float(raw_node_id))
        except (TypeError, ValueError):
            skipped_invalid_node_ids += 1
            continue

        # Rewild-ground recruits currently carry NodeID = -1.
        # Never map those placeholder ids back onto voxel node arrays, because
        # the background voxel fields also use -1 and that floods whole scenes.
        if node_id < 0:
            skipped_invalid_node_ids += 1
            continue

        if treatment_value in ['exoskeleton', 'footprint-depaved']:
            # Match using 'node_CanopyID'
            mask = (canopy_id == node_id)
        else:
            # Match using 'sim_Nodes'
            mask = (sim_nodes == node_id)

        # Update 'scenario_under-node-treatment' for matching voxels
        ds['scenario_under-node-treatment'].values[mask] = treatment_value

    if skipped_invalid_node_ids:
        print(f"Skipped {skipped_invalid_node_ids} under-node treatment rows with invalid NodeID values")

    #--------------------------------------------------------------------------
    # STEP 5: ASSIGN NON-NODE BASED REWILDING STATUS
    # For points that are scenario_rewildingEnabled but have scenario_under-node-treatment = 'none'
    #--------------------------------------------------------------------------
    # Check if scenario_rewildingEnabled exists
    if 'scenario_rewildingEnabled' in ds.variables:
        print("scenario_rewildingEnabled exists in the dataset")
        
        # Print some statistics about scenario_rewildingEnabled
        enabled_count = (ds['scenario_rewildingEnabled'] >= 0).sum().item()
        print(f"Number of voxels with scenario_rewildingEnabled >= 0: {enabled_count}")
        
        # Create mask for points that are enabled for rewilding but don't have a specific type
        generic_rewilding_mask = (
            (ds['scenario_rewildingEnabled'] >= 0)
            & (ds['scenario_under-node-treatment'] == 'none')
        )
        
        # Print count of voxels that match the mask
        mask_count = generic_rewilding_mask.sum().item()
        print(f"Number of voxels that match the generic rewilding mask: {mask_count}")
        
        # Assign 'rewilded' category to these points
        ds['scenario_under-node-treatment'].values[generic_rewilding_mask] = 'rewilded'
        
        # Print count of generic rewilded points
        print(f'Number of rewilded points: {generic_rewilding_mask.sum().item()}')
    else:
        print("WARNING: scenario_rewildingEnabled does not exist in the dataset")

    #--------------------------------------------------------------------------
    # STEP 6: PRINT STATISTICS
    # Output counts of rewilding categories for verification
    #--------------------------------------------------------------------------
    # Print all unique values and counts for df['under-node-treatment'] using pandas
    print("Column under-node-treatment values and counts in dataframe:")
    print(df['under-node-treatment'].value_counts())

    # Print all unique variable values and counts for scenario_under-node-treatment
    unique_values, counts = np.unique(ds['scenario_under-node-treatment'], return_counts=True)
    print("Column scenario_under-node-treatment values and counts in xarray dataset:")
    for value, count in zip(unique_values, counts):
        print(f"scenario_under-node-treatment value: {value}, count: {count}")
    
    return ds

def create_bioEnvelope_catagories(ds, params):
    """
    Updates the xarray dataset with bio-envelope categories based on simulation parameters.
    
    Parameters:
    ds (xarray.Dataset): The xarray dataset containing voxel information.
    params (dict): Dictionary of scenario parameters including thresholds.
    
    Returns:
    xarray.Dataset: The updated dataset with bio-envelope variables.
    
    Variables created/modified:
    - ds['bioMask']: Boolean mask indicating voxels eligible for bio-envelope
    - ds['scenario_bioEnvelope']: Categorical variable indicating bio-envelope type for each voxel
    """
    #--------------------------------------------------------------------------
    # STEP 1: CREATE BIO-ENVELOPE ELIGIBILITY MASK
    # Determine which voxels are eligible for bio-envelope based on simulation parameters
    # Note: Parameters are pre-interpolated for sub-timesteps
    #--------------------------------------------------------------------------
    turnThreshold = params['sim_TurnsThreshold']
    resistanceThreshold = params.get('sim_averageResistance', 0)

    bioMask = (
        (ds['sim_Turns'] <= turnThreshold)
        & (ds['sim_averageResistance'] <= resistanceThreshold)
        & (ds['sim_Turns'] >= 0)
    )
    ds['bioMask'] = bioMask

    #--------------------------------------------------------------------------
    # STEP 2: ASSIGN BIO-ENVELOPE CATEGORIES
    # Update bio-envelope categories based on building elements and bioMask
    #--------------------------------------------------------------------------
    # Note: scenario_bioEnvelope is already initialized as a copy of scenario_under-node-treatment in generate_vtk
    
    # Assign 'otherGround' to bio-envelope-eligible voxels that remain unlabeled
    otherground_mask = bioMask & (ds['scenario_bioEnvelope'] == 'none')
    ds['scenario_bioEnvelope'].loc[otherground_mask] = 'otherGround'

    # Assign 'livingFacade' to scenario_bioEnvelope where site_building_element == 'facade' and bioMask is True
    ds['scenario_bioEnvelope'].loc[(ds['site_building_element'] == 'facade') & bioMask] = 'livingFacade'

    # Assign 'greenRoof' to explicit green-roof voxels
    ds['scenario_bioEnvelope'].loc[(ds['envelope_roofType'] == 'green roof') & bioMask] = 'greenRoof'

    # Treat all remaining roof voxels as brown-roof opportunity
    ds['scenario_bioEnvelope'].loc[
        (ds['site_building_element'] == 'roof') & (ds['envelope_roofType'] != 'green roof') & bioMask
    ] = 'brownRoof'

    """poly = a_helper_functions.convert_xarray_into_polydata(ds)
    poly.plot(scalars='scenario_bioEnvelope', cmap='Set1')"""

    #--------------------------------------------------------------------------
    # STEP 3: PRINT STATISTICS
    # Output counts of bio-envelope categories for verification
    #--------------------------------------------------------------------------
    unique_values, counts = np.unique(ds['scenario_bioEnvelope'], return_counts=True)
    print('column scenario_bioEnvelope values and counts in xarray dataset:')
    for value, count in zip(unique_values, counts):
        print(f'scenario_bioEnvelope value: {value}, count: {count}')

    return ds

def assign_v4_proposals_under_canopy_and_nodes(ds, df):
    """V4: assign decay proposals to under-canopy voxels by matching each
    node's voxels via node_CanopyID. Only touches voxels still marked
    not-assessed (not already set by node broadcast).

    Decay is the only proposal that needs node matching here because we must
    check the node's actual decay decision (release-control trees also set
    under-node-treatment but should not trigger decay proposals).

    Colonise and recruit are assigned from scenario_bioEnvelope instead."""
    canopy_id = ds['node_CanopyID'].values
    n_assigned = 0

    for _, row in df.iterrows():
        treatment = row.get('under-node-treatment', 'paved')
        if treatment not in ('exoskeleton', 'footprint-depaved', 'node-rewilded'):
            continue

        raw_node_id = row.get('NodeID', np.nan)
        if pd.isna(raw_node_id):
            continue
        try:
            node_id = int(float(raw_node_id))
        except (TypeError, ValueError):
            continue
        if node_id < 0:
            continue

        decay_decision = str(row.get('proposal-decay_decision', 'not-assessed'))
        if decay_decision == 'not-assessed':
            continue

        canopy_mask = canopy_id == node_id
        if not canopy_mask.any():
            continue

        decay_intervention = str(row.get('proposal-decay_intervention', 'none'))
        unset = np.asarray(ds['proposal_decayV4'].values).astype('<U64') == 'not-assessed'
        target = canopy_mask & unset
        if target.any():
            ds['proposal_decayV4'].values[target] = decay_decision
            ds['proposal_decayV4_intervention'].values[target] = decay_intervention
            n_assigned += int(target.sum())

    if n_assigned > 0:
        print(f"V4 decay under-canopy: assigned {n_assigned} voxels")

    return ds


def assign_v4_proposals_from_bioenvelope(ds):
    """Assign colonise and recruit V4 proposals based on scenario_bioEnvelope.
    Only touches voxels still marked not-assessed.

    Colonise:
        node-rewilded, footprint-depaved, rewilded, otherground → rewild-ground
        greenroof → enrich-envelope
        brownroof, livingfacade → roughen-envelope

    Recruit:
        node-rewilded, footprint-depaved → buffer-feature
        otherground, rewilded → rewild-ground
    """
    if 'scenario_bioEnvelope' not in ds.variables:
        return ds

    bio_envelope = _normalize_str_array(ds['scenario_bioEnvelope'].values)

    # --- colonise ---
    if 'proposal_coloniseV4' in ds.variables:
        colonise_decision = np.asarray(ds['proposal_coloniseV4'].values).astype('<U64')
        colonise_intervention = np.asarray(ds['proposal_coloniseV4_intervention'].values).astype('<U64')
        unset = colonise_decision == 'not-assessed'

        colonise_rewild = unset & np.isin(bio_envelope, ["node-rewilded", "footprint-depaved", "rewilded", "otherground"])
        colonise_enrich = unset & np.isin(bio_envelope, ["greenroof"])
        colonise_roughen = unset & np.isin(bio_envelope, ["brownroof", "livingfacade"])

        colonise_decision[colonise_rewild | colonise_enrich | colonise_roughen] = "proposal-colonise_accepted"
        colonise_intervention[colonise_rewild] = COLONISE_FULL_GROUND
        colonise_intervention[colonise_enrich] = COLONISE_FULL_ENVELOPE
        colonise_intervention[colonise_roughen] = COLONISE_PARTIAL_ENVELOPE

        ds['proposal_coloniseV4'].values[:] = colonise_decision
        ds['proposal_coloniseV4_intervention'].values[:] = colonise_intervention
        print(f"V4 colonise from bioenvelope: {int(colonise_rewild.sum())} rewild-ground, {int(colonise_enrich.sum())} enrich-envelope, {int(colonise_roughen.sum())} roughen-envelope")

    # --- recruit ---
    if 'proposal_recruitV4' in ds.variables:
        recruit_decision = np.asarray(ds['proposal_recruitV4'].values).astype('<U64')
        recruit_intervention = np.asarray(ds['proposal_recruitV4_intervention'].values).astype('<U64')
        unset = recruit_decision == 'not-assessed'

        recruitInterventionPARTIAL = unset & np.isin(bio_envelope, list(PROPOSAL_RECRUIT_BUFFER_INTERVENTION_VALUES))
        recruitInterventionFULL = unset & np.isin(bio_envelope, list(PROPOSAL_RECRUIT_REWILD_INTERVENTION_VALUES))

        recruit_decision[recruitInterventionPARTIAL | recruitInterventionFULL] = "proposal-recruit_accepted"
        recruit_intervention[recruitInterventionPARTIAL] = RECRUIT_PARTIAL
        recruit_intervention[recruitInterventionFULL] = RECRUIT_FULL

        # Reject recruit-accepted voxels within 1.5m of existing trees
        forest_size = _normalize_str_array(ds['forest_size'].values) if 'forest_size' in ds.variables else None
        if forest_size is not None:
            tree_mask = np.isin(forest_size, ["medium", "large", "senescing"])
            if tree_mask.any():
                points = np.vstack((ds['centroid_x'].values, ds['centroid_y'].values, ds['centroid_z'].values)).T
                tree_kdtree = cKDTree(points[tree_mask])
                distances, _ = tree_kdtree.query(points, distance_upper_bound=1.5)
                too_close = np.isfinite(distances) & (recruit_decision == "proposal-recruit_accepted")
                recruit_decision[too_close] = "proposal-recruit_rejected"
                recruit_intervention[too_close] = "none"
                n_rejected = int(too_close.sum())
                if n_rejected > 0:
                    print(f"V4 recruit: rejected {n_rejected} voxels within 1.5m of trees")

        ds['proposal_recruitV4'].values[:] = recruit_decision
        ds['proposal_recruitV4_intervention'].values[:] = recruit_intervention
        print(f"V4 recruit from bioenvelope: {int(recruitInterventionFULL.sum())} FULL (rewild-larger-patch), {int(recruitInterventionPARTIAL.sum())} PARTIAL (rewild-smaller-patch)")

    return ds


def finalDSprocessing(ds):
    """
    Creates updated resource variables in the xarray dataset for final analysis.
    
    Parameters:
    ds (xarray.Dataset): The xarray dataset containing voxel and resource information.
    
    Returns:
    xarray.Dataset: The updated dataset with additional resource variables.
    
    Variables created/modified:
    - ds['updatedResource_elevatedDeadBranches']: Combined dead branches for senescing trees
    - ds['updatedResource_groundDeadBranches']: Combined resources for fallen trees
    - ds['maskforTrees']: Boolean mask identifying voxels with tree resources
    - ds['maskForRewilding']: Boolean mask identifying voxels with rewilding status
    - ds['scenario_outputs']: Combined categorical variable for visualization
    """
    #--------------------------------------------------------------------------
    # STEP 1: CREATE ELEVATED DEAD BRANCHES RESOURCE
    # Combines dead branch and other resources for senescing trees
    #--------------------------------------------------------------------------
    # Initialize with a copy of the dead branch resource
    ds['updatedResource_elevatedDeadBranches'] = ds['resource_dead branch'].copy()
    
    # Get mask for 'forest_size' == 'senescing'
    mask_senescing = ds['forest_size'] == 'senescing'
    
    # Update 'updatedResource_elevatedDeadBranches' for senescing trees
    ds['updatedResource_elevatedDeadBranches'].loc[mask_senescing] = ds['resource_dead branch'].loc[mask_senescing] + ds['resource_other'].loc[mask_senescing]

    #--------------------------------------------------------------------------
    # STEP 2: CREATE GROUND DEAD BRANCHES RESOURCE
    # Combines multiple resources for fallen trees
    #--------------------------------------------------------------------------
    # Initialize with a copy of the fallen log resource
    ds['updatedResource_groundDeadBranches'] = ds['resource_fallen log'].copy()
    
    # Get mask for ground deadwood phases
    mask_fallen = (ds['forest_size'] == 'fallen') | (ds['forest_size'] == 'decayed')
    
    # Update 'updatedResource_groundDeadBranches' for fallen trees
    ds['updatedResource_groundDeadBranches'].loc[mask_fallen] = (
        ds['resource_dead branch'].loc[mask_fallen] + 
        ds['resource_other'].loc[mask_fallen] + 
        ds['resource_peeling bark'].loc[mask_fallen] + 
        ds['resource_fallen log'].loc[mask_fallen] + 
        ds['resource_perch branch'].loc[mask_fallen]
    )

    #--------------------------------------------------------------------------
    # STEP 3: CREATE TREE RESOURCE MASK
    # Identifies voxels that contain any tree resources (except leaf litter)
    #--------------------------------------------------------------------------
    # Initialize mask as all False
    maskforTrees = np.zeros(ds.dims['voxel'], dtype=bool)
    
    # Loop through resource variables and update the mask
    for var_name in ds.variables:
        if var_name.startswith('resource_') and var_name != 'resource_leaf litter':
            # Create a boolean mask where the values are greater than 0
            resource_mask = ds[var_name].values > 0
            
            # Combine the mask with the current mask using logical OR
            maskforTrees = np.logical_or(maskforTrees, resource_mask)
    
    # Add the mask to the dataset
    ds['maskforTrees'] = xr.DataArray(maskforTrees, dims='voxel')
    
    #--------------------------------------------------------------------------
    # STEP 4: CREATE REWILDING MASK
    # Identifies voxels that have been rewilded (not 'none') and don't have tree resources
    #--------------------------------------------------------------------------
    # Create a mask for rewilded voxels that don't overlap with tree resources
    maskForRewilding = (ds['scenario_bioEnvelope'].values != 'none') & (~maskforTrees)
    ds['maskForRewilding'] = xr.DataArray(maskForRewilding, dims='voxel')
    
    # Print statistics about the masks
    print(f"Total voxels: {ds.dims['voxel']}")
    print(f"Tree voxels: {maskforTrees.sum()}")
    print(f"Rewilded voxels (non-overlapping): {maskForRewilding.sum()}")
    
    #--------------------------------------------------------------------------
    # STEP 5: CREATE COMBINED SCENARIO OUTPUT VARIABLE
    # Combines tree size and rewilding status into a single categorical variable
    #--------------------------------------------------------------------------
    # Initialize with object dtype
    scenario_outputs = np.full(ds.dims['voxel'], 'none', dtype='O')
    
    # For rewilded voxels, use the appropriate rewilding status
    # Check if scenario_bioEnvelope exists and use it instead of scenario_under-node-treatment
    if 'scenario_bioEnvelope' in ds.variables:
        print("Using scenario_bioEnvelope for rewilded voxels in scenario_outputs")
        scenario_outputs[maskForRewilding] = ds['scenario_bioEnvelope'].values[maskForRewilding]
    else:
        print("Using scenario_under-node-treatment for rewilded voxels in scenario_outputs")
        scenario_outputs[maskForRewilding] = ds['scenario_under-node-treatment'].values[maskForRewilding]
    
    # For tree voxels, use the forest size
    scenario_outputs[maskforTrees] = ds['forest_size'].values[maskforTrees]
    
    # Add the combined variable to the dataset
    ds['scenario_outputs'] = xr.DataArray(scenario_outputs, dims='voxel')
    
    # Print unique values and counts for scenario_outputs
    unique_values, counts = np.unique(ds['scenario_outputs'], return_counts=True)
    print('Column scenario_outputs values and counts in xarray dataset:')
    for value, count in zip(unique_values, counts):
        print(f'scenario_outputs value: {value}, count: {count}')

    return ds

#==============================================================================
# POLYDATA PROCESSING FUNCTIONS
#==============================================================================
def process_polydata(polydata):
    """
    Process the polydata to verify and print statistics about the variables.
    The main variables are now created in the xarray dataset before conversion.
    
    Parameters:
    polydata (pyvista.PolyData): The polydata object converted from xarray
    
    Returns:
    pyvista.PolyData: The processed polydata
    
    Variables used (already created in xarray):
    - maskforTrees: Boolean mask identifying voxels with tree resources
    - maskForRewilding: Boolean mask identifying voxels with rewilding status
    - scenario_outputs: Combined categorical variable for visualization
    """
    # Print unique values and counts for scenario_outputs
    if 'scenario_outputs' in polydata.point_data:
        print(f'unique values and counts for scenario_outputs in polydata: {pd.Series(polydata.point_data["scenario_outputs"]).value_counts()}')
    

    maskforTrees = polydata.point_data['maskforTrees']
    maskForRewilding = polydata.point_data['maskForRewilding']
    
    # Print statistics about the masks
    print(f"Total points in polydata: {polydata.n_points}")
    print(f"Tree points: {maskforTrees.sum()}")
    print(f"Rewilded points (non-overlapping): {maskForRewilding.sum()}")
    
    #--------------------------------------------------------------------------
    # STEP 4: CREATE COMBINED SCENARIO OUTPUT VARIABLE
    # Combines tree size and rewilding status into a single categorical variable
    #--------------------------------------------------------------------------
    # Initialize with object dtype
    scenario_outputs = np.full(polydata.n_points, 'none', dtype='O')
    
    # For rewilded voxels, use the appropriate rewilding status
    # Check if scenario_bioEnvelope exists and use it instead of scenario_under-node-treatment
    if 'scenario_bioEnvelope' in polydata.point_data:
        print("Using scenario_bioEnvelope for rewilded voxels in scenario_outputs")
        scenario_outputs[maskForRewilding] = polydata.point_data['scenario_bioEnvelope'][maskForRewilding]
    else:
        print("Using scenario_under-node-treatment for rewilded voxels in scenario_outputs")
        scenario_outputs[maskForRewilding] = polydata.point_data['scenario_under-node-treatment'][maskForRewilding]
    
    # For tree voxels, use the forest size
    scenario_outputs[maskforTrees] = polydata.point_data['forest_size'][maskforTrees]
    
    # Print unique values and counts for scenario_outputs
    print(f'unique values and counts for scenario_outputs: {pd.Series(scenario_outputs).value_counts()}')
    polydata.point_data['scenario_outputs'] = scenario_outputs
    
    print(f'unique values and counts for scenario_outputs in polydata: {pd.Series(polydata.point_data["scenario_outputs"]).value_counts()}')
    
    return polydata

#==============================================================================
# VISUALIZATION FUNCTIONS
#==============================================================================
def plot_scenario_under_node_treatment(polydata, treeDF, years_passed, site):
    """
    Creates a visualization of the scenario with trees, rewilding, and site voxels.
    
    Parameters:
    polydata (pyvista.PolyData): The polydata object with all variables
    treeDF (pd.DataFrame): Tree dataframe for labeling
    years_passed (int): Years passed for title
    site (str): Site name for title
    
    Uses variables:
    - maskforTrees: Boolean mask identifying voxels with tree resources
    - maskForRewilding: Boolean mask identifying voxels with rewilding status
    """
    # Get masks from polydata (created in xarray)
    maskforTrees = polydata.point_data['maskforTrees']
    maskForRewilding = polydata.point_data['maskForRewilding']

    # Extract different polydata subsets based on the masks
    
    treePoly = polydata.extract_points(maskforTrees)  # Points where forest_tree_id is not NaN
    designActionPoly = polydata.extract_points(maskForRewilding)

    # Extract site voxels (neither trees nor rewilding)
    siteMask = ~(maskforTrees | maskForRewilding)
    sitePoly = polydata.extract_points(siteMask)




    # Print all point_data variables in polydata
    print(f'point_data variables in polydata: {sitePoly.point_data.keys()}')

    # Create the plotter
    plotter = pv.Plotter()
    # Add title to the plotter
    plotter.add_text(f"Scenario at {site} after {years_passed} years", position="upper_edge", font_size=16, color='black')

    # Label trees
    label_trees(treeDF, plotter)

    # Add 'none' points as white
    # Add site points if they exist
    if sitePoly.n_points > 0:
        plotter.add_mesh(sitePoly, color='white')
    else:
        print("No site points to visualize")
    
    # Add tree points if they exist
    if treePoly.n_points > 0:
        plotter.add_mesh(treePoly, scalars='forest_size', cmap='Set1')
    else:
        print("No tree points to visualize")
    
    # Add rewilding/design action points if they exist
    if designActionPoly.n_points > 0:
        plotter.add_mesh(designActionPoly, scalars='scenario_bioEnvelope', cmap='Set2', show_scalar_bar=True)
    else:
        print("No rewilding/design action points to visualize")
    plotter.enable_eye_dome_lighting()

    plotter.show()
    plotter.close()

def label_trees(df, plotter):
    # Prepare points and labels from the filtered subset
    TARGET_SIZES_FOR_LABELS = ['large', 'senescing', 'snag', 'fallen', 'decayed']
    label_df = df[df['size'].isin(TARGET_SIZES_FOR_LABELS)]
    label_points = label_df[['x', 'y', 'z']].values
    label_points[:,2] = label_points[:,2] + 10  # Give some elevation to the labels so they are easier to see
    labels = label_df['size'].astype(str).tolist()
    
    # Add the labels to the plotter
    plotter.add_point_scalar_labels(
        points=label_points,
        labels=labels,
        fmt='%s',              # Since 'size' is categorical, no formatting is needed
        preamble='Size: ',
        font_size=20,          # Adjust font size as needed
        text_color='black',    # Choose a contrasting color for visibility
        shadow=True,           # Add shadow for better readability
        render_points_as_spheres=False,  # Optional: Customize label rendering
        point_size=10          # Size of the points associated with labels if rendered
    )

#==============================================================================
# UTILITY FUNCTIONS
#==============================================================================
def print_simulation_statistics(df, year, site):
    print(f"Simulation Summary for Year: {year}, Site: {site}")
    
    # Print total number of trees
    total_trees = len(df)
    print(f"Total number of trees: {total_trees}")
    
    # Print unique values and their counts for the 'size' column
    print("\nUnique values and their counts for 'size':")
    print(df['size'].value_counts())

    # Print unique values and their counts for the 'action' column
    print("\nUnique values and their counts for 'action':")
    print(df['action'].value_counts())

    # Print unique values and their counts for the 'under-node-treatment' column
    print("\nUnique values and their counts for 'under-node-treatment':")
    print(df['under-node-treatment'].value_counts())

    print(f"Trees planted: {df[df['isNewTree'] == True].shape[0]}")
    
    print("\nEnd of simulation statistics.\n")

#==============================================================================
# MAIN PROCESSING FUNCTIONS
#==============================================================================
def generate_vtk(
    site,
    scenario,
    year,
    voxel_size,
    ds,
    treeDF,
    logDF=None,
    poleDF=None,
    enable_visualization=False,
    return_polydata=False,
    save_raw_vtk=False,
):
    """
    Generates a VTK file for a given site, scenario, and year.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    scenario (str): Scenario type ('positive', 'trending')
    year (int): Year to simulate
    voxel_size (int): Voxel size
    ds (xarray.Dataset): Xarray dataset
    treeDF (pd.DataFrame): Tree dataframe
    logDF (pd.DataFrame, optional): Log dataframe
    poleDF (pd.DataFrame, optional): Pole dataframe
    enable_visualization (bool, optional): Whether to show visualization
    
    Returns:
    xarray.Dataset: Updated dataset
    """
    ds = ds.copy(deep=True)

    # Get scenario parameters with interpolation for sub-timesteps
    params = params_v3.get_params_for_year(site, scenario, year)
    params["absolute_year"] = year
    params["previous_year"] = max(0, year - 30)
    params["step_years"] = max(0, params["absolute_year"] - params["previous_year"])

    print(f'Generating VTK for {site}, {scenario}, year {year}')
    
    #--------------------------------------------------------------------------
    # STEP 1: UPDATE XARRAY WITH SCENARIO DATA
    # Variables created/modified:
    # - ds['scenario_under-node-treatment']: Under-node treatment status for each voxel
    # - ds['scenario_bioEnvelope']: Bio-envelope status for each voxel (if logs/poles exist)
    # - ds['bioMask']: Boolean mask for bio-envelope eligibility
    #--------------------------------------------------------------------------
    # First, build the under-node-treatment variables that feed the later VTK layers.
    print('Ensuring rewilding variables are created in xarray')
    _, ds = calculate_under_node_treatment_status(treeDF, ds, params)
    
    # Now integrate node-based rewilding results into xarray
    print('Integrating node-based rewilding results into xarray')
    ds = create_under_node_treatment_variable(ds, treeDF)

    # Always initialize scenario_bioEnvelope as a copy of scenario_under-node-treatment
    print("Initializing scenario_bioEnvelope as a copy of scenario_under-node-treatment")
    ds['scenario_bioEnvelope'] = xr.DataArray(
        data=np.array(ds['scenario_under-node-treatment'].values, dtype='O'),
        dims='voxel'
    )
    
    # Update bioEnvelope categories if logs or poles exist
    print('Updating bioEnvelope voxel categories')
    if logDF is not None or poleDF is not None:
        ds = create_bioEnvelope_catagories(ds, params)
    else:
        print("No logs or poles found, using scenario_under-node-treatment values for scenario_bioEnvelope")

    unique_values, counts = np.unique(ds['scenario_under-node-treatment'], return_counts=True)
    print(f"Unique values and counts for scenario_under-node-treatment: {unique_values}, {counts}")
    
    unique_values, counts = np.unique(ds['scenario_bioEnvelope'], return_counts=True)
    print(f'Unique values and counts for scenario_bioEnvelope: {unique_values}, {counts}')

    #--------------------------------------------------------------------------
    # STEP 2: PREPARE VALID POINTS FOR RESOURCE VOXELIZATION
    # Variables created:
    # - validpointsMask: Boolean mask for voxels to include in resource calculation
    # - valid_points: Array of 3D coordinates for valid voxels
    #--------------------------------------------------------------------------
    # Prepare valid points for resource voxelization
    validpointsMask = ds['scenario_bioEnvelope'].values != 'none'

    # Extract valid points as a numpy array
    valid_points = np.array([
        ds['centroid_x'].values[validpointsMask],
        ds['centroid_y'].values[validpointsMask],
        ds['centroid_z'].values[validpointsMask]
    ]).T

    #--------------------------------------------------------------------------
    # STEP 3: INTEGRATE RESOURCES INTO XARRAY
    # Variables created/modified:
    # - Multiple resource variables in ds (resource_*)
    # - forest_* variables in ds
    # - proposal_*V4 variables in ds (pre-seeded from node-level proposal data)
    # - combinedDF_scenario: Combined dataframe of all nodes with resources
    #--------------------------------------------------------------------------

    # Pre-initialize all 10 V4 proposal arrays on ds.
    voxel_count = ds.sizes["voxel"]

    # Node-broadcast proposals: tree-owned voxels get overwritten with real
    # node proposal values during voxel integration below.
    _v4_node_broadcast = {
        "proposal_decayV4": "not-assessed",
        "proposal_decayV4_intervention": "none",
        "proposal_release_controlV4": "not-assessed",
        "proposal_release_controlV4_intervention": "none",
        "proposal_deploy_structureV4": "not-assessed",
        "proposal_deploy_structureV4_intervention": "none",
    }
    for var, default in _v4_node_broadcast.items():
        ds[var] = xr.DataArray(np.full(voxel_count, default, dtype=object), dims="voxel")

    # Derived proposals: these do not depend on node df broadcast and are
    # populated later from scenario/search/indicator context.
    _v4_derived = {
        "proposal_recruitV4": "not-assessed",
        "proposal_recruitV4_intervention": "none",
        "proposal_coloniseV4": "not-assessed",
        "proposal_coloniseV4_intervention": "none",
    }
    for var, default in _v4_derived.items():
        ds[var] = xr.DataArray(np.full(voxel_count, default, dtype=object), dims="voxel")

    templateResolution = 1
    ds, combinedDF_scenario = a_voxeliser.integrate_resources_into_xarray(ds, treeDF, templateResolution, logDF, poleDF, valid_points)

    # V4 proposal assignment — category 1A/1B: under-canopy and under-node voxels
    ds = assign_v4_proposals_under_canopy_and_nodes(ds, treeDF)
    # V4 proposal assignment — category 2: non-node rewilding and colonise voxels
    ds = assign_v4_proposals_from_bioenvelope(ds)

    #--------------------------------------------------------------------------
    # STEP 4: FINAL XARRAY PROCESSING
    # Variables created:
    # - ds['updatedResource_elevatedDeadBranches']: Combined dead branches for senescing trees
    # - ds['updatedResource_groundDeadBranches']: Combined resources for fallen trees
    # - ds['maskforTrees']: Boolean mask identifying voxels with tree resources
    # - ds['maskForRewilding']: Boolean mask identifying voxels with rewilding status
    # - ds['scenario_outputs']: Combined categorical variable for visualization
    #--------------------------------------------------------------------------
    ds = finalDSprocessing(ds)
    # Commented out: V4 proposal assignment now handled by
    # assign_v4_proposals_under_canopy_and_nodes, assign_v4_proposals_from_bioenvelope,
    # and assign_v4_proposals_per_voxel (in a_info_gather_capabilities).
    # ds = create_proposal_point_data(ds)
    # Commented out: this ran before search_bioavailable existed (stage 5),
    # producing incomplete proposals. The late path in
    # ensure_v3_proposal_point_data(polydata) now handles full computation
    # with search_bioavailable available.
    # ds = create_v3_proposal_point_data(ds)

    blender_framebuffer_columns = build_blender_proposal_framebuffer_columns(combinedDF_scenario)
    for column_name in blender_framebuffer_columns.columns:
        combinedDF_scenario[column_name] = blender_framebuffer_columns[column_name]

    # Save combinedDF_scenario to csv
    print(f'Saving {year} combinedDF_scenario to csv')
    legacy_node_df_path = scenario_node_df_path(site, scenario, year, voxel_size)
    combinedDF_scenario.to_csv(legacy_node_df_path, index=False)

    refactored_node_df_path = engine_output_nodedf_path(site, scenario, year, voxel_size)
    combinedDF_scenario.to_csv(refactored_node_df_path, index=False)
    print(f'Saved refactored nodeDF to {refactored_node_df_path}')
    
    # Print statistics
    print_simulation_statistics(combinedDF_scenario, year, site)

    #--------------------------------------------------------------------------
    # STEP 5: CONVERT XARRAY TO POLYDATA AND SAVE VTK
    # All variables are now created in the xarray dataset and transferred to polydata
    #--------------------------------------------------------------------------
    # Convert to polydata and process
    polydata = a_helper_functions.convert_xarray_into_polydata(ds)
    polydata = process_polydata(polydata)
    if not a_helper_functions.export_all_pointdata_variables():
        polydata = a_helper_functions.drop_polydata_point_arrays_if_present(
            polydata,
            a_helper_functions.LEAN_EXPORT_POINTDATA_DROP_ARRAYS,
        )
    
    vtk_file = None
    if save_raw_vtk:
        vtk_file = scenario_state_vtk_path(site, scenario, year, voxel_size)
        polydata.save(vtk_file)
        print(f'Saved VTK file to {vtk_file}')

    #--------------------------------------------------------------------------
    # STEP 6: OPTIONAL VISUALIZATION
    # Creates visualization with:
    # - Tree voxels colored by forest_size
    # - Rewilding voxels colored by scenario_under-node-treatment
    # - Site voxels in white
    # - Tree labels for large, senescing, snag, and fallen trees
    #--------------------------------------------------------------------------
    if enable_visualization:
        plot_scenario_under_node_treatment(polydata, treeDF, year, site)
    
    if return_polydata:
        return (str(vtk_file) if vtk_file is not None else None), polydata
    return str(vtk_file) if vtk_file is not None else None

def load_scenario_dataframes(site, scenario, year, voxel_size):
    """
    Loads scenario dataframes for a given site, scenario, and year.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    scenario (str): Scenario type ('positive', 'trending')
    year (int): Year to load
    voxel_size (int): Voxel size
    
    Returns:
    tuple: (treeDF, logDF, poleDF) - Loaded dataframes
    """
    # Load tree dataframe
    tree_file = scenario_tree_df_path(site, scenario, year, voxel_size)
    if os.path.exists(tree_file):
        treeDF = pd.read_csv(tree_file)
        print(f'Loaded tree dataframe from {tree_file}')
    else:
        print(f'Tree dataframe file not found: {tree_file}')
        return None, None, None
    
    # Load log dataframe
    log_file = scenario_log_df_path(site, scenario, year, voxel_size)
    if os.path.exists(log_file):
        logDF = pd.read_csv(log_file)
        print(f'Loaded log dataframe from {log_file}')
    else:
        logDF = None
        print(f'Log dataframe file not found: {log_file}')
    
    # Load pole dataframe
    pole_file = scenario_pole_df_path(site, scenario, year, voxel_size)
    if os.path.exists(pole_file):
        poleDF = pd.read_csv(pole_file)
        print(f'Loaded pole dataframe from {pole_file}')
    else:
        poleDF = None
        print(f'Pole dataframe file not found: {pole_file}')
    
    return treeDF, logDF, poleDF

#==============================================================================
# MAIN EXECUTION
#==============================================================================
if __name__ == "__main__":
    from refactor_code.sim.setup import a_scenario_initialiseDS
    
    # Define default sites and scenarios
    default_sites = ['trimmed-parade', 'city', 'uni']
    default_scenarios = ['positive', 'trending']
    default_voxel_size = 1
    
    # Get user input for site
    sites_input = input(f"Enter site(s) to process (comma-separated) or press Enter for default {default_sites}: ")
    sites = sites_input.split(',') if sites_input else default_sites
    sites = [site.strip() for site in sites]
    
    # Get user input for scenarios
    scenarios_input = input(f"Enter scenario(s) to process (comma-separated) or press Enter for default {default_scenarios}: ")
    scenarios = scenarios_input.split(',') if scenarios_input else default_scenarios
    scenarios = [scenario.strip() for scenario in scenarios]
    
    # Get user input for voxel size
    voxel_size_input = input(f"Enter voxel size (default {default_voxel_size}): ")
    voxel_size = int(voxel_size_input) if voxel_size_input else default_voxel_size
    
    # Get user input for years
    all_years = [0, 10, 30, 60, 180]
    years_input = input(f"Enter years to process (comma-separated) or press Enter for default {all_years}: ")
    years = [int(year) for year in years_input.split(',')] if years_input else all_years
    
    # Get user input for visualization
    vis_input = input("Enable visualization? (yes/no, default no): ")
    enable_visualization = vis_input.lower() in ['yes', 'y', 'true', '1']
    
    # Generate VTKs for each site, scenario, and year
    for site in sites:
        print(f"\n===== Processing {site} =====\n")
        # Initialize dataset
        possibility_space_ds = a_scenario_initialiseDS.initialize_dataset(site, voxel_size)
        
        for scenario in scenarios:
            print(f"\n--- Processing {scenario} scenario ---\n")
            
            for year in years:
                print(f"\n- Processing year {year} -\n")
                
                # Load scenario dataframes
                treeDF, logDF, poleDF = load_scenario_dataframes(site, scenario, year, voxel_size)
                
                if treeDF is not None:
                    # Generate VTK
                    generate_vtk(
                        site,
                        scenario,
                        year,
                        voxel_size,
                        possibility_space_ds,
                        treeDF,
                        logDF,
                        poleDF,
                        enable_visualization,
                    )
                else:
                    print(f"Skipping VTK generation for {site}, {scenario}, year {year} - dataframes not found")
                
    print("\nAll VTK generation completed.")
