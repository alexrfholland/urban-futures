# Scenario Engine V3 Naming Cleanup

Checkpoint before this rename pass:

- commit: `70a0c95`
- message: `pre v3 name changes for key sim engine variables`

## Purpose

This rename pass removes the most confusing legacy v2/v3 names from the active
v3 stack.

Priority:

1. the live engine should use the current proposal terminology
2. saved nodeDF and VTK outputs should expose the current schema names
3. the code should not maintain parallel live aliases through the timestep logic

## Important Rule

This document tracks field and function names.

It does **not** rename semantic category values such as:

- `node-rewilded`
- `footprint-depaved`
- `rewilded`
- `exoskeleton`

Those still exist as category values inside the renamed fields.

## Implemented Renames

### Functions

- `determine_lifecycle_decisions(...)` -> `determine_proposal_decay(...)`
- `apply_senescence_states(...)` -> `apply_proposal_decay_accepted_lifecycle_changes(...)`
- `handle_replace_trees(...)` -> `apply_proposal_decay_rejected_changes(...)`
- `assign_decay_support(...)` -> `assign_decay_interventions(...)`

### Internal Variables

- `senesceChance` -> `proposal_decay_chance`
- `senesceRoll` -> `proposal_decay_roll`
- `subsetDS` / `subset_ds` -> `possibility_space_ds`

Note:

- the instruction document used `possibility-space_ds`
- Python identifiers cannot use `-`
- the implemented code name is therefore `possibility_space_ds`

### Parameter Keys

Implemented in [params_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/params_v3.py):

- `senescingThreshold` -> `lifecycle_senescing_ramp_start`
- `ageInPlaceThreshold` -> `minimal-tree-support-threshold`
- `plantThreshold` -> `moderate-tree-support-threshold`
- `rewildThreshold` -> `maximum-tree-support-threshold`

These are dictionary keys, so the hyphenated names from the instruction
document are used directly.

### Tree / Node Dataframe Schema

- `rewilded` -> `under-node-treatment`
- `lifecycle_decision` -> `proposal-decay_decision`
- `control_realized` -> `control_reached`
- `pruning_target` -> `proposal-release-control_intervention`
- `pruning_target_years` -> `proposal-release-control_target_years`
- `autonomy_years` -> `proposal-release-control_years`

### Xarray / VTK Schema

- `scenario_rewilded` -> `scenario_under-node-treatment`

## Active File Surface Updated

Core engine / schema:

- [_code-refactored/refactor_code/scenario/engine_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v3.py)
- [_code-refactored/refactor_code/scenario/params_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/params_v3.py)
- [_code-refactored/refactor_code/scenario/baseline_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/baseline_v3.py)
- [_code-refactored/refactor_code/scenario/validation.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/validation.py)

Scenario prep / batch / VTK:

- [final/a_scenario_initialiseDS.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_initialiseDS.py)
- [final/run_full_v3_batch.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/run_full_v3_batch.py)
- [final/a_scenario_generateVTKs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_generateVTKs.py)

Stats / proposal readers:

- [final/a_info_gather_capabilities.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_gather_capabilities.py)
- [final/a_info_proposal_interventions.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_proposal_interventions.py)

Direct VTK / Blender helpers:

- [_code-refactored/refactor_code/blender/b_generate_rewilded_ground.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/b_generate_rewilded_ground.py)
- [_code-refactored/refactor_code/blender/b_rewilded_surface_shell.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/b_rewilded_surface_shell.py)
- [_code-refactored/refactor_code/scenario/inspect_vtk.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/inspect_vtk.py)

## Deliberate Non-Goals

This pass does not attempt to rename:

- category values like `rewilded`
- `scenario_rewildingEnabled`
- `scenario_rewildingPlantings`
- legacy v2 / archived scripts outside the active v3 run path

## Compatibility Position

The active engine logic now uses the renamed fields directly.

There is no parallel dual-schema state through the live timestep logic.

Legacy CSVs carrying the old column names are no longer normalized on read.
They should be regenerated.

## Validation Requirement After This Change

Any fresh regenerated nodeDF output should expose:

- `under-node-treatment`
- `proposal-decay_decision`
- `control_reached`
- `proposal-release-control_intervention`
- `proposal-release-control_target_years`
- `proposal-release-control_years`

Any fresh regenerated VTK output should expose:

- `scenario_under-node-treatment`
- `scenario_bioEnvelope`

and should no longer rely on:

- `scenario_rewilded`
