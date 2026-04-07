# V4 Restructure Handoff

Use this together with:

- `_documentation-refactored/v4 restructure.md`
- `AGENTS.md`

## Current Baseline

- branch: `engine-v4`
- pre-v4 checkpoint: `e0af588`
- v4 branch start: `65f1691`

## What Is Now Done

- the v4 branch was created from `engine-v3` at `65f1691`
- active sim code is now grouped under:
  - `sim/run`
  - `sim/setup`
  - `sim/generate_state`
  - `sim/generate_vtk`
  - `sim/baseline`
  - `sim/voxel`
  - `outputs/stats`
  - `outputs/report`
- active Blender export prep now lives under `_code-refactored/refactor_code/blender/bexport`
- active Blender v2 code now lives under `_code-refactored/refactor_code/blender/blenderv2`
- active legacy sim helpers were moved out of `final/` into:
  - `sim/voxel/voxel_a_voxeliser.py`
  - `sim/voxel/voxel_a_rotate_resource_structures.py`
  - `sim/voxel/voxel_a_helper_functions.py`
  - `sim/voxel/voxel_f_SiteCoordinates.py`
- active Blender export helper was moved out of `final/` into:
  - `blender/bexport/bexport_f_vtk_to_ply_surfaces.py`
- imports were updated so the live sim runtime no longer depends on `final/` for active execution
- `tree_processing` was intentionally left in `_code-refactored/refactor_code/tree_processing`

## Confirmed Runtime Facts

### `--node-only` vs `--vtk-only`

- `--node-only` writes interim scenario tables only:
  - `treeDF`
  - `logDF`
  - `poleDF`
- `--node-only` does not write the integrated final `nodeDF`
- `--vtk-only` loads saved `treeDF` / `logDF` / `poleDF`
- `--vtk-only` writes:
  - final integrated `nodeDF`
  - final enriched `state_with_indicators.vtk`

Confirmed in:

- `_code-refactored/refactor_code/sim/run/run_full_v3_batch.py`
- `_code-refactored/refactor_code/sim/generate_vtk/a_scenario_generateVTKs.py`
- `_code-refactored/refactor_code/sim/generate_state/engine_v3.py`
- `AGENTS.md`

### `nodeDF` vs `treeDF` / `logDF` / `poleDF`

The integrated `nodeDF` is not just a copy of the interim state tables.

It:

- combines tree, log, and pole rows
- includes integration/materialization fields not present in `treeDF`
- includes downstream export columns

### Recruits and `NodeID`

Current canonical v3 behavior still allows new planted trees to have:

- `isNewTree == True`
- `NodeID == -1`

## Validation Completed

Scratch smoke root used:

- `_data-refactored/model-outputs/generated-states/v4-smoke`

Explicit env vars used:

- `TREE_TEMPLATE_ROOT`
- `REFACTOR_RUN_OUTPUT_ROOT`

Validated commands:

1. `uv run python _code-refactored/refactor_code/sim/run/run_full_v3_batch.py --node-only --multiple-agent --sites trimmed-parade --scenarios positive --years 0 --voxel-size 1`
2. `uv run python _code-refactored/refactor_code/sim/run/run_full_v3_batch.py --vtk-only --multiple-agent --sites trimmed-parade --scenarios positive --years 0 --voxel-size 1`

Observed result:

- both smoke commands completed successfully
- scratch outputs were written under:
  - `temp/interim-data`
  - `temp/validation`
  - `output/feature-locations`
  - `output/vtks`
- current remaining console output was limited to existing pandas/xarray `FutureWarning` messages

## Current Stable Layout

- `sim/run`
- `sim/setup`
- `sim/generate_state`
- `sim/generate_vtk`
- `sim/baseline`
- `sim/voxel`
- `tree_processing`
- `outputs/stats`
- `outputs/report`
- `blender/bexport`
- `blender/blenderv2`

## Next Steps

1. Do the dedicated `tree_processing` cleanup pass.
2. Simplify the tree-variant / template structure now that the sim and Blender runtime paths are stable.
3. Keep updating docs that still mention `scenario/` or the old Blender folders.
4. Keep using `uv` with explicit `TREE_TEMPLATE_ROOT` and `REFACTOR_RUN_OUTPUT_ROOT` for any further validation runs.

## Recommended Prompt For Next Session

- focus on `tree_processing`
- keep the current `sim/*`, `outputs/*`, and `blender/*` structure stable unless there is a strong reason to move it again
- continue preferring direct moves over wrappers or duplicate live paths
