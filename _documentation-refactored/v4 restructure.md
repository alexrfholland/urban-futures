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
