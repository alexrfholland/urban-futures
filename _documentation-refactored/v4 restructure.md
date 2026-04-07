# V4 Restructure

Checkpoint before this restructure:

- commit: `e0af588`
- message: `pre v4-restructure`

Current v4 branch start:

- branch: `engine-v4`
- first v4 commit base: `65f1691`

## Purpose

This pass moves the active v3 simulation, baseline, tree-library, Blender-export,
and Blender v2 code into `_code-refactored` and consolidates active input/output
path routing under `_data-refactored`.

The goal is clarity, not compatibility.

After this pass:

- the live v3 stack is in `_code-refactored/refactor_code`
- active tree-template inputs stay in `_data-refactored/model-inputs`
- active generated-state roots stay in `_data-refactored/model-outputs`
- old `final/` paths are historical references only
- no compatibility wrappers are kept as the live entrypoints

## Main Moves

### Sim And Outputs

Old root:

- `final/`

New roots:

- `_code-refactored/refactor_code/sim/run/`
- `_code-refactored/refactor_code/sim/setup/`
- `_code-refactored/refactor_code/sim/generate_interim_state_data/`
- `_code-refactored/refactor_code/sim/generate_vtk_and_nodeDFs/`
- `_code-refactored/refactor_code/sim/baseline/`
- `_code-refactored/refactor_code/sim/voxel/`
- `_code-refactored/refactor_code/outputs/stats/`
- `_code-refactored/refactor_code/outputs/report/`

Moved live files:

- `final/run_full_v3_batch.py` -> `_code-refactored/refactor_code/sim/run/run_full_v3_batch.py`
- `final/run_saved_v3_vtks.py` -> `_code-refactored/refactor_code/sim/run/run_saved_v3_vtks.py`
- `final/run_all_simulations.py` -> `_code-refactored/refactor_code/sim/run/run_all_simulations.py`
- `final/a_scenario_initialiseDS.py` -> `_code-refactored/refactor_code/sim/setup/a_scenario_initialiseDS.py`
- `final/a_scenario_runscenario.py` -> `_code-refactored/refactor_code/sim/generate_interim_state_data/a_scenario_runscenario.py`
- `final/a_scenario_generateVTKs.py` -> `_code-refactored/refactor_code/sim/generate_vtk_and_nodeDFs/a_scenario_generateVTKs.py`
- `final/a_scenario_urban_elements_count.py` -> `_code-refactored/refactor_code/sim/generate_vtk_and_nodeDFs/a_scenario_urban_elements_count.py`
- `final/a_scenario_get_baselines.py` -> `_code-refactored/refactor_code/sim/baseline/a_scenario_get_baselines.py`
- `final/a_scenario_manager.py` -> `_code-refactored/refactor_code/sim/run/a_scenario_manager.py`
- `final/a_info_gather_capabilities.py` -> `_code-refactored/refactor_code/sim/generate_vtk_and_nodeDFs/a_info_gather_capabilities.py`
- `final/a_info_proposal_interventions.py` -> `_code-refactored/refactor_code/outputs/report/a_info_proposal_interventions.py`
- `final/a_info_pathway_tracking_graphs.py` -> `_code-refactored/refactor_code/outputs/report/a_info_pathway_tracking_graphs.py`
- `final/a_info_output_capabilities.py` -> `_code-refactored/refactor_code/outputs/stats/a_info_output_capabilities.py`
- `final/a_info_debuglog.py` -> `_code-refactored/refactor_code/outputs/stats/a_info_debuglog.py`
- `final/refresh_indicator_csvs_from_baseline.py` -> `_code-refactored/refactor_code/outputs/stats/refresh_indicator_csvs_from_baseline.py`
- `final/a_voxeliser.py` -> `_code-refactored/refactor_code/sim/voxel/voxel_a_voxeliser.py`
- `final/a_rotate_resource_structures.py` -> `_code-refactored/refactor_code/sim/voxel/voxel_a_rotate_resource_structures.py`
- `final/a_helper_functions.py` -> `_code-refactored/refactor_code/sim/voxel/voxel_a_helper_functions.py`
- `final/f_SiteCoordinates.py` -> `_code-refactored/refactor_code/sim/voxel/voxel_f_SiteCoordinates.py`

Current note:

- `tree_processing` now lives in `_code-refactored/refactor_code/input_processing/tree_processing/`
- the dedicated cleanup pass for that area still remains to be done

### Tree Library And Template Runtime

Old roots:

- `final/a_resource_distributor_dataframes.py`
- `final/tree_processing/`

New root:

- `_code-refactored/refactor_code/input_processing/tree_processing/`

Moved live files:

- `final/a_resource_distributor_dataframes.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/a_resource_distributor_dataframes.py`
- `final/tree_processing/aa_tree_helper_functions.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/aa_tree_helper_functions.py`
- `final/tree_processing/aa_io.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/aa_io.py`
- `final/tree_processing/adTree_AssignResources.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/adTree_AssignResources.py`
- `final/tree_processing/combine_edit_individual_trees.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/combine_edit_individual_trees.py`
- `final/tree_processing/combine_resource_treeMeshGenerator.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/combine_resource_treeMeshGenerator.py`
- `final/tree_processing/combined_generateResourceDict.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/combined_generateResourceDict.py`
- `final/tree_processing/combined_redoSnags.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/combined_redoSnags.py`
- `final/tree_processing/combined_tree_manager.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/combined_tree_manager.py`
- `final/tree_processing/combined_voxelise_dfs.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/combined_voxelise_dfs.py`
- `final/tree_processing/b_generate_utility_pole_and_artificial_tree.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/b_generate_utility_pole_and_artificial_tree.py`
- `final/tree_processing/a_log_mesh_generator.py` -> `_code-refactored/refactor_code/input_processing/tree_processing/a_log_mesh_generator.py`

### Tree Library Export Data

Old local working roots:

- `data/revised/final/treeMeshes`
- `data/revised/final/treeMeshesPly`
- `data/revised/final/logMeshes`
- `data/revised/final/logMeshesPly`

New local working root:

- `_data-refactored/model-inputs/tree_library_exports/`

Live folders:

- `treeMeshes`
- `treeMeshesPly`
- `logMeshes`
- `logMeshesPly`

### Blender

Old roots:

- `_code-refactored/refactor_code/blender/`
- `final/_blender/`

New roots:

- `_code-refactored/refactor_code/blender/bexport/`
- `_code-refactored/refactor_code/blender/blenderv2/`

Moved live files:

- `_code-refactored/refactor_code/blender/b_generate_rewilded_envelopes.py` -> `_code-refactored/refactor_code/blender/bexport/export_rewilded_envelopes.py`
- `_code-refactored/refactor_code/blender/b_generate_rewilded_ground.py` -> `_code-refactored/refactor_code/blender/bexport/export_rewilded_ground.py`
- `_code-refactored/refactor_code/blender/b_rewilded_surface_shell.py` -> `_code-refactored/refactor_code/blender/bexport/rewilded_surface_shell.py`
- `final/_blender/a_vtk_to_ply.py` -> `_code-refactored/refactor_code/blender/bexport/vtk_to_ply.py`
- `final/_blender/a_export_to_ply.py` -> `_code-refactored/refactor_code/blender/bexport/export_to_ply.py`
- `final/_blender/b_extract_scene.py` -> `_code-refactored/refactor_code/blender/bexport/extract_scene.py`
- `final/_blender/b_tree_system_importer.py` -> `_code-refactored/refactor_code/blender/bexport/tree_system_importer.py`
- `final/f_vtk_to_ply_surfaces.py` -> `_code-refactored/refactor_code/blender/bexport/bexport_f_vtk_to_ply_surfaces.py`

Moved live folders:

- `final/_code-refactored/blender/timeline/` -> `_code-refactored/refactor_code/blender/blenderv2/timeline/`
- `final/_code-refactored/blender/city_street/` -> `_code-refactored/refactor_code/blender/blenderv2/city_street/`

## Paths Cleanup

`_code-refactored/refactor_code/paths.py` now groups live path resolution into:

1. versioning
2. site information
3. baselines
4. tree library
5. model outputs
6. assessed/postprocessed outputs
7. statistics
8. blender

New live helpers include:

- `canonical_version_roots(...)`
- `generated_state_run_root(...)`
- `simv3_run_root(...)`
- `mediaflux_versioned_destination(...)`
- `site_subset_dataset_path(...)`
- `site_rewilding_voxel_array_path(...)`
- `site_tree_locations_path(...)`
- `site_pole_locations_path(...)`
- `site_log_locations_path(...)`
- `tree_template_root(...)`
- `tree_mesh_vtk_dir(...)`
- `tree_mesh_ply_dir(...)`
- `log_mesh_vtk_dir(...)`
- `log_mesh_ply_dir(...)`
- `scenario_visualization_dir(...)`
- `statistics_csv_dir(...)`
- `statistics_debug_dir(...)`

The canonical v3 roots are now the default source of truth in `paths.py`.

## Current V4 Structure

The current live code structure after this pass is:

- `input_processing/tree_processing`
- `sim/run`
- `sim/setup`
- `sim/generate_interim_state_data`
- `sim/generate_vtk_and_nodeDFs`
- `sim/voxel`
- `sim/baseline`
- `outputs/stats`
- `outputs/report`
- `blender/bexport`
- `blender/blenderv2`

Meaning:

- `input_processing/tree_processing`
  - tree-template and tree-library preparation code
  - still the next cleanup target
- `sim/run`
  - entrypoints and orchestration scripts
- `sim/setup`
  - scenario-ready dataset loading and normalization
- `sim/generate_interim_state_data`
  - scenario engine and interim state-table generation
- `sim/generate_vtk_and_nodeDFs`
  - integrated `nodeDF` generation and final enriched VTK generation
- `sim/voxel`
  - low-level voxel/resource/rotation/helper machinery used by `sim/generate_vtk_and_nodeDFs`
- `sim/baseline`
  - baseline generation branch
- `outputs/stats`
  - post-run compiled summaries, merged counts, comparisons, validation tables
- `outputs/report`
  - graphs and reporting outputs
- `blender/bexport`
  - Blender export-prep code
- `blender/blenderv2`
  - Blender v2 runtime/render code

## Current Sim Generation Flow And Proposal Handoff

This section keeps the current runtime flow and the proposal handoff in one
place.

### 1. Batch Entrypoint

Script:

- `_code-refactored/refactor_code/sim/run/run_full_v3_batch.py`

Main functions:

- `_prepare_subset_dataset(site, voxel_size, write_cache=True)`
- `run_site_scenario(site, scenario, years, ...)`

Inputs:

- CLI selectors:
  - `--sites`
  - `--scenarios`
  - `--years`
  - `--voxel-size`
  - `--node-only`
  - `--vtk-only`
- env vars:
  - `TREE_TEMPLATE_ROOT`
  - `REFACTOR_RUN_OUTPUT_ROOT`

Hands off:

- `possibility_space_ds`
- `treeDF`
- `logDF`
- `poleDF`

### 2. Setup

Script:

- `_code-refactored/refactor_code/sim/setup/a_scenario_initialiseDS.py`

Functions:

- `initialize_dataset(site, voxel_size, write_cache=True)`
- `load_node_dataframes(site, voxel_size)`
- `PreprocessData(treeDF, ds, extraTreeDF=None)`
- `further_xarray_processing(ds)`
- `log_processing(logDF, ds)`
- `pole_processing(poleDF, extraPoleDF, ds)`

Creates:

- base `possibility_space_ds`
- base `treeDF`
- base `logDF`
- base `poleDF`

### 3. Interim Scenario Engine

Scripts:

- `_code-refactored/refactor_code/sim/generate_interim_state_data/a_scenario_runscenario.py`
- `_code-refactored/refactor_code/sim/generate_interim_state_data/engine_v3.py`

Functions:

- `run_scenario(...)`
- `run_timestep(...)`
- `_run_single_pulse(...)`

Creates:

- `treeDF_scenario`
- `logDF_scenario`
- `poleDF_scenario`

Creates canonical node-level proposal fields:

Decisions

- `df[proposal-decay_decision]`
- `df[proposal-release-control_decision]`
- `df[proposal-recruit_decision]`
- `df[proposal-colonise_decision]`
- `df[proposal-deploy-structure_decision]`

Interventions

- `df[proposal-decay_intervention]`
- `df[proposal-release-control_intervention]`
- `df[proposal-recruit_intervention]`
- `df[proposal-colonise_intervention]`
- `df[proposal-deploy-structure_intervention]`

Also creates the node-level state those depend on:

- `df[under-node-treatment]`
- `df[control]`
- `df[control_reached]`
- `df[lifecycle_state]`

### 4. VTK Build And Integrated nodeDF

Script:

- `_code-refactored/refactor_code/sim/generate_vtk_and_nodeDFs/a_scenario_generateVTKs.py`

Main functions:

- `generate_vtk(site, scenario, year, voxel_size, ds, treeDF, logDF=None, poleDF=None, ...)`
- `create_under_node_treatment_variable(ds, df)`

Dependency script:

- `_code-refactored/refactor_code/sim/voxel/voxel_a_voxeliser.py`
  - `integrate_resources_into_xarray(...)`

Inputs:

- `possibility_space_ds.copy(deep=True)`
- `treeDF_scenario`
- `logDF_scenario`
- `poleDF_scenario`

Creates and assigns:

- `ds[scenario_rewildingEnabled]`
  - from `ds[sim_Turns]`, `ds[site_building_element]`, and the current year
- `ds[scenario_rewildingPlantings]`
  - from enabled rewilding space minus proximity to active tree positions from `df`
- `ds[scenario_under-node-treatment]`
  - by mapping `df[under-node-treatment]` into voxel regions using:
    - `ds[node_CanopyID] == df[NodeID]` for `exoskeleton` and `footprint-depaved`
    - `ds[sim_Nodes] == df[NodeID]` for `node-rewilded`
  - then filling remaining enabled rewilding space as generic `rewilded`
- `ds[scenario_bioEnvelope]`
  - first as a copy of `ds[scenario_under-node-treatment]`
  - then updated for logs and poles

Then assigns tree/log/pole-level dataframe data to owned voxels per template
ownership during voxel integration.

Current live names created there are broadcast `forest_*` fields such as:

- `ds[forest_size]`
- `ds[forest_control]`
- `ds[forest_precolonial]`

As part of the v4 proposal cleanup, proposal / intervention point-data should
not be broadcast as `forest_proposal-*`.

Instead, during this same owned-voxel broadcast step, the canonical node-level
proposal / intervention values should be mapped directly onto the temporary V4
VTK proposal arrays.

For the simple tree/log/pole template ownership case, this means:

- `df[proposal-decay_decision]` -> update `ds[proposal_decayV4]`
- `df[proposal-decay_intervention]` -> update `ds[proposal_decayV4_intervention]`
- `df[proposal-release-control_decision]` -> update `ds[proposal_release_controlV4]`
- `df[proposal-release-control_intervention]` -> update `ds[proposal_release_controlV4_intervention]`
- `df[proposal-deploy-structure_decision]` -> update `ds[proposal_deploy_structureV4]`
- `df[proposal-deploy-structure_intervention]` -> update `ds[proposal_deploy_structureV4_intervention]`

That means:

- each tree template / log template / pole template voxel assigned to a row
  should receive that row's canonical dataframe proposal / intervention value
- `forest_*` stays for non-proposal broadcast fields such as size, control, and
  precolonial
- proposal broadcast moves directly into the temporary `V4` proposal arrays

This stage also creates:

- `combinedDF_scenario`

What the integrated `nodeDF` is:

- it is not a copy of `treeDF`
- it is the combined active export table returned from voxel integration
- it includes tree, log, and pole rows together
- it reflects the active enabled rows used in the VTK/resource build
- it is the table that then receives:
  - `blender_proposal-decay`
  - `blender_proposal-release-control`
  - `blender_proposal-recruit`
  - `blender_proposal-colonise`
  - `blender_proposal-deploy-structure`

This stage writes:

- integrated final `nodeDF`
  - `output/feature-locations/{site}/{site}_{scenario}_1_nodeDF_yr{year}.csv`

At the end of this stage:

- `ds` is converted to in-memory polydata
- from this point onward, point-data should be described as `vtk[...]`

For the planned temporary `v4` VTK proposal layer, initialize:

- `vtk[proposal_decayV4] = "not-assessed"`
- `vtk[proposal_decayV4_intervention] = "none"`
- `vtk[proposal_release_controlV4] = "not-assessed"`
- `vtk[proposal_release_controlV4_intervention] = "none"`
- `vtk[proposal_recruitV4] = "not-assessed"`
- `vtk[proposal_recruitV4_intervention] = "none"`
- `vtk[proposal_coloniseV4] = "not-assessed"`
- `vtk[proposal_coloniseV4_intervention] = "none"`
- `vtk[proposal_deploy_structureV4] = "not-assessed"`
- `vtk[proposal_deploy_structureV4_intervention] = "none"`

### 5. Urban / Search Enrichment

Script:

- `_code-refactored/refactor_code/sim/generate_vtk_and_nodeDFs/a_scenario_urban_elements_count.py`

Function:

- `process_scenario_polydata(...)`

Input:

- in-memory `polydata`

Creates:

- `vtk[search_bioavailable]`
- `vtk[search_urban_elements]`

### 6. Indicators And Final Proposal / Render Fields

Script:

- `_code-refactored/refactor_code/sim/generate_vtk_and_nodeDFs/a_info_gather_capabilities.py`

Functions:

- `apply_indicators(polydata)`
- `add_proposal_point_data(polydata)`
- `ensure_v3_proposal_point_data(polydata)`
- `process_polydata(...)`

Input:

- enriched in-memory `polydata`

Creates:

- `vtk[indicator_*]`

Current canonical proposal naming is different between the integrated `nodeDF`
and the final VTK.

Canonical `nodeDF` proposal / intervention schema:

- `df[proposal-decay_decision]`
- `df[proposal-decay_intervention]`
- `df[proposal-release-control_decision]`
- `df[proposal-release-control_intervention]`
- `df[proposal-recruit_decision]`
- `df[proposal-recruit_intervention]`
- `df[proposal-colonise_decision]`
- `df[proposal-colonise_intervention]`
- `df[proposal-deploy-structure_decision]`
- `df[proposal-deploy-structure_intervention]`

Canonical VTK proposal / intervention schema:

- `vtk[proposal_decayV3]`
- `vtk[proposal_decayV3_intervention]`
- `vtk[proposal_release_controlV3]`
- `vtk[proposal_release_controlV3_intervention]`
- `vtk[proposal_recruitV3]`
- `vtk[proposal_recruitV3_intervention]`
- `vtk[proposal_coloniseV3]`
- `vtk[proposal_coloniseV3_intervention]`
- `vtk[proposal_deploy_structureV3]`
- `vtk[proposal_deploy_structureV3_intervention]`

This means the current accepted v3 stack still has two canonical proposal
schemas:

- `nodeDF` uses the hyphenated `proposal-..._decision` /
  `proposal-..._intervention` convention
- VTK uses the underscore `proposal_*V3` / `proposal_*V3_intervention`
  convention

Current VTK `V3` assignment logic:

- the same `proposal_*V3` logic currently exists in two places:
  - early on `ds[...]` in `create_v3_proposal_point_data(ds)`
  - late on `vtk[...]` in `ensure_v3_proposal_point_data(polydata)`
- those two functions currently use the same assignment rules

`vtk[proposal_decayV3]` and `vtk[proposal_decayV3_intervention]`

- start as `not-assessed` / `none`
- copy from:
  - `vtk[forest_proposal-decay_decision]`
  - `vtk[forest_proposal-decay_intervention]`
  where those are present
- then assign under-canopy decay voxels through `vtk[scenario_bioEnvelope]`:
  - `node-rewilded` or `footprint-depaved` ->
    `proposal-decay_accepted` / `buffer-feature`
  - `exoskeleton` ->
    `proposal-decay_accepted` / `brace-feature`

`vtk[proposal_release_controlV3]` and
`vtk[proposal_release_controlV3_intervention]`

- start as `not-assessed` / `none`
- copy from:
  - `vtk[forest_proposal-release-control_decision]`
  - `vtk[forest_proposal-release-control_intervention]`
  where those are present
- then use `vtk[search_bioavailable] == arboreal` as the release-control
  opportunity mask
- for opportunity voxels that still have no intervention assigned:
  - `vtk[forest_control] in {park-tree}` ->
    `proposal-release-control_accepted` / `reduce-pruning`
  - `vtk[forest_control] in {reserve-tree, improved-tree}` ->
    `proposal-release-control_accepted` / `eliminate-pruning`
  - remaining opportunity voxels stay
    `proposal-release-control_rejected` / `none`

`vtk[proposal_coloniseV3]` and `vtk[proposal_coloniseV3_intervention]`

- start as `proposal-colonise_rejected` / `none`
- use `vtk[scenario_outputs]` as the source field
- accept where `vtk[scenario_outputs]` is one of:
  - `brownroof`
  - `greenroof`
  - `livingfacade`
  - `footprint-depaved`
  - `node-rewilded`
  - `otherground`
  - `rewilded`
- intervention mapping:
  - `node-rewilded`, `footprint-depaved`, `rewilded` -> `rewild-ground`
  - `greenroof` -> `enrich-envelope`
  - `brownroof`, `livingfacade` -> `roughen-envelope`

`vtk[proposal_recruitV3]` and `vtk[proposal_recruitV3_intervention]`

- start as `not-assessed` / `none`
- create a recruit consideration mask from:
  - voxels within 20 m of a canopy feature and not on building-valued
    `vtk[search_urban_elements]`
  - plus voxels where `vtk[scenario_rewildingPlantings] >= 0`
- consideration voxels are first marked:
  - `proposal-recruit_rejected` / `none`
- then accept from:
  - `vtk[indicator_Tree_generations_grassland] == True`
  - together with `vtk[scenario_bioEnvelope]`
- intervention mapping:
  - `node-rewilded`, `footprint-depaved` -> `buffer-feature`
  - `otherground`, `rewilded` -> `rewild-ground`

`vtk[proposal_deploy_structureV3]` and
`vtk[proposal_deploy_structureV3_intervention]`

- start as `not-assessed` / `none`
- copy from:
  - `vtk[forest_proposal-deploy-structure_decision]`
  - `vtk[forest_proposal-deploy-structure_intervention]`
  where those are present
- then fill remaining unassigned opportunity voxels from:
  - `vtk[forest_size] == artificial` and `vtk[forest_precolonial] == False` ->
    `proposal-deploy-structure_accepted` / `adapt-utility-pole`
  - `vtk[indicator_Bird_self_peeling] == True` and
    `vtk[forest_precolonial] == False` ->
    `proposal-deploy-structure_accepted` / `upgrade-feature`

Blender mapping comes last.

- for `nodeDF`, Blender framebuffers are derived from the canonical nodeDF
  decision / intervention pairs
- for VTK, Blender framebuffers are currently derived from the canonical
  `proposal_*V3` point-data pairs
- Blender framebuffer arrays are derived outputs, not the proposal source of
  truth

Older manuscript / hack VTK proposal labels are still present in parts of the
pipeline but are no longer part of the live intended schema:

- `vtk[proposal_decay]`
- `vtk[proposal_release_control]`
- `vtk[proposal_recruit]`
- `vtk[proposal_colonise]`
- `vtk[proposal_deploy_structure]`

Those older non-`V3` VTK proposal label arrays should be treated as legacy-only
and dropped in v4 cleanup work.

This stage writes:

- final enriched VTK
  - `output/vtks/{site}/{site}_{scenario}_1_yr{year}_state_with_indicators.vtk`

## Naming Rule For Active Legacy Dependencies

When active legacy dependency scripts are moved into the live structure:

- preserve their original stems
- add a grouping prefix rather than replacing the stem with an unrelated new name

Examples:

- `a_voxeliser.py` -> `voxel_a_voxeliser.py`
- `a_rotate_resource_structures.py` -> `voxel_a_rotate_resource_structures.py`
- `a_helper_functions.py` -> `voxel_a_helper_functions.py`
- `f_SiteCoordinates.py` -> `voxel_f_SiteCoordinates.py`

This rule applies not only to scripts still living under `final/`, but also to any
other dependency-heavy scripts that are still actively called by live code.

## Rename Safety Rule

Before renaming any dependency-heavy script in the restructure:

- confirm which legacy code still calls the current refactored code
- confirm which refactored code still calls legacy code
- confirm the full import/call chain for the script being renamed

Do not rename these scripts based on filename cleanup alone.

For active dependency-heavy files, rename only after verifying the current call sites.

## Validation

Smoke validation was rerun on the moved v4 paths with explicit:

- `TREE_TEMPLATE_ROOT`
- `REFACTOR_RUN_OUTPUT_ROOT`

Scratch root used:

- `_data-refactored/model-outputs/generated-states/v4-smoke`

Smoke passes completed:

- `uv run python _code-refactored/refactor_code/sim/run/run_full_v3_batch.py --node-only --multiple-agent --sites trimmed-parade --scenarios positive --years 0 --voxel-size 1`
- `uv run python _code-refactored/refactor_code/sim/run/run_full_v3_batch.py --vtk-only --multiple-agent --sites trimmed-parade --scenarios positive --years 0 --voxel-size 1`

Observed result:

- both passes completed successfully on the scratch root
- final smoke artifacts include:
  - `temp/interim-data/trimmed-parade/trimmed-parade_positive_1_treeDF_0.csv`
  - `output/feature-locations/trimmed-parade/trimmed-parade_positive_1_nodeDF_yr0.csv`
  - `output/vtks/trimmed-parade/trimmed-parade_positive_1_yr0_state_with_indicators.vtk`
- current remaining console output was limited to existing pandas/xarray `FutureWarning` messages, not path/import failures

## Rule After This Pass

- edit live v3 code in `_code-refactored/refactor_code/...`
- edit active templates and exports in `_data-refactored/...`
- do not treat `final/` as the live location for the active v3 stack
- if an old historical script path breaks, inspect commit `e0af588`
