# Scenario Engine V3 Overview

This note is the shortest route into the current v3 stack.

Use it first, then follow the linked documents for detail.

## Core Links

- status: [_documentation-refactored/scenario_engine_v3_status.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v3_status.md)
- validation: [_documentation-refactored/scenario_engine_v3_validation.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v3_validation.md)
- point-data schema: [_documentation-refactored/scenario_engine_v3_pointdata_schema.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v3_pointdata_schema.md)
- render schema: [_documentation-refactored/scenario_engine_v3_render_schema.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v3_render_schema.md)
- workflow rules: [AGENTS.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/AGENTS.md)

## Where Files Live

Current unified candidate root example:

- [_data-refactored/model-outputs/generated-states/simv3-5](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5)

The run is split into three buckets:

- [temp/interim-data](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/temp/interim-data)
  - interim `treeDF`
  - interim `logDF`
  - interim `poleDF`
- [temp/validation](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/temp/validation)
  - run metadata
  - timing logs
  - validation renders
- [output](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output)
  - [vtks](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/vtks)
  - [feature-locations](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/feature-locations)
  - [stats](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/stats)
  - [baselines](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/baselines)

## Settings

Required environment variables for candidate runs:

- `REFACTOR_RUN_OUTPUT_ROOT`
- `TREE_TEMPLATE_ROOT`

Approved template root:

- [_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

Current standard voxel size:

- `1`

## Flow

Main batch runner:

- [_code-refactored/refactor_code/sim/run/run_full_v3_batch.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/sim/run/run_full_v3_batch.py)

The normal flow is:

1. `--node-only`
   - writes interim `treeDF`
   - writes `logDF` and `poleDF` when the site carries those resources
2. `--vtk-only`
   - loads the saved interim CSVs and `possibility_space_ds`
   - builds one in-memory polydata per state
   - mutates it through search-layer and indicator/proposal enrichment
   - writes the final `state_with_indicators.vtk`
   - writes the final integrated `nodeDF`
3. `--baselines-only`
   - generates baseline outputs separately
   - writes the final baseline `state_with_indicators.vtk`
4. `--compile-stats-only`
   - reads final `state_with_indicators.vtk`
   - writes per-state stats
   - writes merged site-level stats

The important simplification is:

- no intermediate `urban_features.vtk` is required in the normal candidate path
- the final VTK artifact is `state_with_indicators.vtk`

## Dependencies

The active v3 stack now lives directly under `_code-refactored/`.

Core runtime modules:

- [_code-refactored/refactor_code/sim/run/run_full_v3_batch.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/sim/run/run_full_v3_batch.py)
  - main batch entrypoint
- [_code-refactored/refactor_code/sim/setup/a_scenario_initialiseDS.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/sim/setup/a_scenario_initialiseDS.py)
  - dataset and source-data preparation
- [_code-refactored/refactor_code/sim/generate_state/a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/sim/generate_state/a_scenario_runscenario.py)
  - scenario runner around the v3 engine
- [_code-refactored/refactor_code/sim/generate_vtk/a_scenario_generateVTKs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/sim/generate_vtk/a_scenario_generateVTKs.py)
  - integrated `nodeDF` generation and base polydata build
- [_code-refactored/refactor_code/sim/generate_vtk/a_scenario_urban_elements_count.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/sim/generate_vtk/a_scenario_urban_elements_count.py)
  - urban-feature point-data enrichment
- [_code-refactored/refactor_code/sim/generate_vtk/a_info_gather_capabilities.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/sim/generate_vtk/a_info_gather_capabilities.py)
  - indicator and proposal enrichment, final `state_with_indicators.vtk`, and stats

Shared refactored modules:

- [_code-refactored/refactor_code/sim/generate_state/engine_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/sim/generate_state/engine_v3.py)
  - canonical v3 simulation logic
- [_code-refactored/refactor_code/sim/setup/params_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/sim/setup/params_v3.py)
  - canonical v3 parameters and timesteps
- [_code-refactored/refactor_code/sim/baseline/baseline_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/sim/baseline/baseline_v3.py)
  - baseline generation core
- [_code-refactored/refactor_code/paths.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/paths.py)
  - unified run-root routing
- [_code-refactored/refactor_code/outputs/report/render_forest_size_views.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/outputs/report/render_forest_size_views.py)
  - validation render views
- [_code-refactored/refactor_code/outputs/report/render_custom_proposal_schema_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/outputs/report/render_custom_proposal_schema_v3.py)
  - custom proposal-schema renders
  - writes:
    - `engine3-proposals_interventions`
    - `engine3-proposals`

So the practical rule is:

- run the batch through `_code-refactored/refactor_code/sim/run/run_full_v3_batch.py`
- treat `_code-refactored/` as the only live code surface for the active v3 stack
- treat old `final/` references in historical notes as pre-v4 locations

## Outputs

Production-ready for Blender / rendering:

- final [state_with_indicators.vtk](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/vtks)
- final integrated [nodeDF](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/feature-locations)

Statistics:

- per-state stats: [output/stats/per-state](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/stats/per-state)
- merged site stats: [output/stats/csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/stats/csv)
- pathway comparison: [output/stats/comparison_pathways](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/stats/comparison_pathways)

Validation / visualisation:

- render outputs and timing logs in [temp/validation](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/temp/validation)
- default visualisation set:
  - `engine3-proposals_interventions_with-legend`
  - `engine3-proposals`
- non-default / opt-in visualisation variants:
  - `engine3-proposals_interventions`
  - `engine3-proposals_with-legend`
  - the older validation views from `render_forest_size_views.py`
- render scripts:
  - [_code-refactored/refactor_code/outputs/report/render_forest_size_views.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/outputs/report/render_forest_size_views.py)
  - [_code-refactored/refactor_code/outputs/report/render_custom_proposal_schema_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/outputs/report/render_custom_proposal_schema_v3.py)

## Comparison And Validation Flow

Use the flow below in order.

1. Generate interim scenario state.
   - Run `--node-only`.
   - This produces `treeDF`, plus `logDF` and `poleDF` where those site resources exist.

2. Generate final state outputs.
   - Run `--vtk-only`.
   - This loads the interim CSVs and `possibility_space_ds`, builds one polydata per state in memory, enriches it, then writes:
     - final `state_with_indicators.vtk`
     - final integrated `nodeDF`

3. Generate baselines.
   - Run `--baselines-only` or `--regenerate-baselines`.
   - This writes the baseline comparison state for each site, including the final baseline `state_with_indicators.vtk`.

4. Compile statistics.
   - Run `--compile-stats-only`.
   - This reads final `state_with_indicators.vtk` files and writes:
     - per-state stats
     - merged site-level stats

5. Build pathway comparisons.
   - Use the merged stats in `output/stats/csv`.
   - Write comparison tables and explanatory markdown into `output/stats/comparison_pathways`.
   - Current example:
     - [comparison_capability-indicatorsYr180.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/stats/comparison_pathways/comparison_capability-indicatorsYr180.md)

6. Run validation renders.
   - Use the final `state_with_indicators.vtk` files, not interim CSVs.
   - Write render outputs and logs into `temp/validation`.
   - Default visual QA should use:
     - `engine3-proposals_interventions_with-legend`
     - `engine3-proposals`
   - Other render variants are opt-in.

The important split is:

- production state outputs:
  - `output/vtks`
  - `output/feature-locations`
- statistics and comparisons:
  - `output/stats`
- validation and visual QA:
  - `temp/validation`

So comparison is a postprocess on final VTK-derived stats, and validation rendering is a separate visual QA layer on the same final VTKs.

## Current Full Candidate Run

Current verified run:

- [_data-refactored/model-outputs/generated-states/simv3-5](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5)

Verified counts:

- `48` pathway `state_with_indicators.vtk`
- `3` baseline `state_with_indicators.vtk`
- `48` integrated `nodeDF` CSVs
- `102` per-state stats CSVs
- `6` merged site stats CSVs
