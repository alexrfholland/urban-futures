# V4 Restructure

Checkpoint before this restructure:

- commit: `e0af588`
- message: `pre v4-restructure`

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

### Scenario Runtime

Old root:

- `final/`

New root:

- `_code-refactored/refactor_code/scenario/runtime/`

Moved live files:

- `final/run_full_v3_batch.py` -> `_code-refactored/refactor_code/scenario/runtime/run_full_v3_batch.py`
- `final/run_saved_v3_vtks.py` -> `_code-refactored/refactor_code/scenario/runtime/run_saved_v3_vtks.py`
- `final/run_all_simulations.py` -> `_code-refactored/refactor_code/scenario/runtime/run_all_simulations.py`
- `final/a_scenario_initialiseDS.py` -> `_code-refactored/refactor_code/scenario/runtime/a_scenario_initialiseDS.py`
- `final/a_scenario_runscenario.py` -> `_code-refactored/refactor_code/scenario/runtime/a_scenario_runscenario.py`
- `final/a_scenario_generateVTKs.py` -> `_code-refactored/refactor_code/scenario/runtime/a_scenario_generateVTKs.py`
- `final/a_scenario_urban_elements_count.py` -> `_code-refactored/refactor_code/scenario/runtime/a_scenario_urban_elements_count.py`
- `final/a_scenario_get_baselines.py` -> `_code-refactored/refactor_code/scenario/runtime/a_scenario_get_baselines.py`
- `final/a_scenario_manager.py` -> `_code-refactored/refactor_code/scenario/runtime/a_scenario_manager.py`
- `final/a_info_gather_capabilities.py` -> `_code-refactored/refactor_code/scenario/runtime/a_info_gather_capabilities.py`
- `final/a_info_proposal_interventions.py` -> `_code-refactored/refactor_code/scenario/runtime/a_info_proposal_interventions.py`
- `final/a_info_pathway_tracking_graphs.py` -> `_code-refactored/refactor_code/scenario/runtime/a_info_pathway_tracking_graphs.py`
- `final/a_info_output_capabilities.py` -> `_code-refactored/refactor_code/scenario/runtime/a_info_output_capabilities.py`
- `final/a_info_debuglog.py` -> `_code-refactored/refactor_code/scenario/runtime/a_info_debuglog.py`
- `final/refresh_indicator_csvs_from_baseline.py` -> `_code-refactored/refactor_code/scenario/runtime/refresh_indicator_csvs_from_baseline.py`

### Tree Library And Template Runtime

Old roots:

- `final/a_resource_distributor_dataframes.py`
- `final/tree_processing/`

New root:

- `_code-refactored/refactor_code/tree_processing/`

Moved live files:

- `final/a_resource_distributor_dataframes.py` -> `_code-refactored/refactor_code/tree_processing/a_resource_distributor_dataframes.py`
- `final/tree_processing/aa_tree_helper_functions.py` -> `_code-refactored/refactor_code/tree_processing/aa_tree_helper_functions.py`
- `final/tree_processing/aa_io.py` -> `_code-refactored/refactor_code/tree_processing/aa_io.py`
- `final/tree_processing/adTree_AssignResources.py` -> `_code-refactored/refactor_code/tree_processing/adTree_AssignResources.py`
- `final/tree_processing/combine_edit_individual_trees.py` -> `_code-refactored/refactor_code/tree_processing/combine_edit_individual_trees.py`
- `final/tree_processing/combine_resource_treeMeshGenerator.py` -> `_code-refactored/refactor_code/tree_processing/combine_resource_treeMeshGenerator.py`
- `final/tree_processing/combined_generateResourceDict.py` -> `_code-refactored/refactor_code/tree_processing/combined_generateResourceDict.py`
- `final/tree_processing/combined_redoSnags.py` -> `_code-refactored/refactor_code/tree_processing/combined_redoSnags.py`
- `final/tree_processing/combined_tree_manager.py` -> `_code-refactored/refactor_code/tree_processing/combined_tree_manager.py`
- `final/tree_processing/combined_voxelise_dfs.py` -> `_code-refactored/refactor_code/tree_processing/combined_voxelise_dfs.py`
- `final/tree_processing/b_generate_utility_pole_and_artificial_tree.py` -> `_code-refactored/refactor_code/tree_processing/b_generate_utility_pole_and_artificial_tree.py`
- `final/tree_processing/a_log_mesh_generator.py` -> `_code-refactored/refactor_code/tree_processing/a_log_mesh_generator.py`

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

### Blender Export Prep

Old roots:

- `_code-refactored/refactor_code/blender/`
- `final/_blender/`

New root:

- `_code-refactored/refactor_code/blender_export/`

Moved live files:

- `_code-refactored/refactor_code/blender/b_generate_rewilded_envelopes.py` -> `_code-refactored/refactor_code/blender_export/export_rewilded_envelopes.py`
- `_code-refactored/refactor_code/blender/b_generate_rewilded_ground.py` -> `_code-refactored/refactor_code/blender_export/export_rewilded_ground.py`
- `_code-refactored/refactor_code/blender/b_rewilded_surface_shell.py` -> `_code-refactored/refactor_code/blender_export/rewilded_surface_shell.py`
- `final/_blender/a_vtk_to_ply.py` -> `_code-refactored/refactor_code/blender_export/vtk_to_ply.py`
- `final/_blender/a_export_to_ply.py` -> `_code-refactored/refactor_code/blender_export/export_to_ply.py`
- `final/_blender/b_extract_scene.py` -> `_code-refactored/refactor_code/blender_export/extract_scene.py`
- `final/_blender/b_tree_system_importer.py` -> `_code-refactored/refactor_code/blender_export/tree_system_importer.py`

### Blender V2

Old root:

- `final/_code-refactored/blender/`

New root:

- `_code-refactored/refactor_code/blenderv2/`

Moved live folders:

- `final/_code-refactored/blender/timeline/` -> `_code-refactored/refactor_code/blenderv2/timeline/`
- `final/_code-refactored/blender/city_street/` -> `_code-refactored/refactor_code/blenderv2/city_street/`

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

## Rule After This Pass

- edit live v3 code in `_code-refactored/refactor_code/...`
- edit active templates and exports in `_data-refactored/...`
- do not treat `final/` as the live location for the active v3 stack
- if an old historical script path breaks, inspect commit `e0af588`
