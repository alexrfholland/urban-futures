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
  - `sim/generate_interim_state_data`
  - `sim/generate_vtk_and_nodeDFs`
  - `sim/baseline`
  - `sim/voxel`
  - `outputs/stats`
  - `outputs/report`
- active Blender export prep now lives under `_futureSim_refactored/blender/bexport`
- active Blender v2 code now lives under `_futureSim_refactored/blender/blenderv2`
- active legacy sim helpers were moved out of `final/` into:
  - `sim/voxel/voxel_a_voxeliser.py`
  - `sim/voxel/voxel_a_rotate_resource_structures.py`
  - `sim/voxel/voxel_a_helper_functions.py`
  - `sim/voxel/voxel_f_SiteCoordinates.py`
- active Blender export helper was moved out of `final/` into:
  - `blender/bexport/bexport_f_vtk_to_ply_surfaces.py`
- imports were updated so the live sim runtime no longer depends on `final/` for active execution
- `tree_processing` now lives in `_futureSim_refactored/input_processing/tree_processing`
- three recruit zone masks (`scenario_nodeRewildRecruitZone`, `scenario_underCanopyRecruitZone`, `scenario_rewildGroundRecruitZone`) computed centrally in `calculate_under_node_treatment_status()`
- `recruit_mechanism` now uses specific values (`node-rewild`, `under-canopy`, `ground`) instead of the previous `node` / `buffer-feature` / `rewild-ground`
- recruit telemetry CSV with per-pulse zone/density/occupancy tracking
- persistent run log at `_data-refactored/run_log.csv` with fallback in `paths.py`
- debug recruit renderer with 9 diagnostic layers (6 per-variable + 3 composite zone layers)
- proposal renderer updated to v4 hybrid views (`render_proposal_v4.py`)

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

- `_futureSim_refactored/sim/run/run_full_v3_batch.py`
- `_futureSim_refactored/sim/generate_vtk_and_nodeDFs/a_scenario_generateVTKs.py`
- `_futureSim_refactored/sim/generate_interim_state_data/engine_v3.py`
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

### Three Recruitment Types (v4)

The engine now distinguishes three recruitment mechanisms via `recruit_mechanism` on the treeDF:

| `recruit_mechanism` | Zone mask | Allocation | Mortality |
|---|---|---|---|
| `node-rewild` | `scenario_nodeRewildRecruitZone` | Voxel-mask from `sim_Nodes` matching node-rewilded tree NodeIDs | Reserve (0.03) |
| `under-canopy` | `scenario_underCanopyRecruitZone` | Voxel-mask from `node_CanopyID` matching footprint-depaved/exoskeleton trees | Urban (0.06) |
| `ground` | `scenario_rewildGroundRecruitZone` | Depaved ground filtered by `ground_filter_mode` (default: node-exclusion) | Reserve (0.03) |

All three zone masks are computed together in `calculate_under_node_treatment_status()`. Convention: `>= 0` = active (year enabled), `-1` = inactive.

Previously these were `buffer-feature` and `rewild-ground`. The v4 split uses specific `recruit_mechanism` values (`node-rewild`, `under-canopy`, `ground`) eliminating the need for compound checks.

### Recruit Telemetry

Each pulse writes per-type rows to `{site}_{scenario}_recruit_telemetry.csv` with fields including `zone_voxel_count`, `density_per_pulse`, `filled`, `fallback_used`, `fallback_count`. After all years, `log_run_stats()` prints per-year and per-pulse summary tables.

### Run Log

Persistent CSV at `_data-refactored/run_log.csv` (columns: timestamp, name, output_root, description). Appended by the batch runner on every run. `_futureSim_refactored.paths.refactor_run_output_root()` falls back to the last logged root when `REFACTOR_RUN_OUTPUT_ROOT` is not set.

### Debug Recruit Renderer

`outputs/report/render_debug_recruit.py` — renders 9 diagnostic layers per VTK:
- 6 per-variable layers (recruit_isNewTree, recruit_hasbeenReplanted, recruit_mechanism, recruit_year, recruit_mortality_rate, recruit_mortality_cohort)
- 3 composite zone layers (ground_recruitment, node_rewild_recruitment, under_canopy_recruitment) each showing zone voxels in one colour and recruited tree canopies in another

## Validation Completed

Scratch smoke root used:

- `_data-refactored/model-outputs/generated-states/v4-smoke`

Explicit env vars used:

- `TREE_TEMPLATE_ROOT`
- `REFACTOR_RUN_OUTPUT_ROOT`

Validated commands:

1. `uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py --node-only --multiple-agent --sites trimmed-parade --scenarios positive --years 0 --voxel-size 1`
2. `uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py --vtk-only --multiple-agent --sites trimmed-parade --scenarios positive --years 0 --voxel-size 1`

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
- `sim/generate_interim_state_data`
- `sim/generate_vtk_and_nodeDFs`
- `sim/baseline`
- `sim/voxel`
- `input_processing/tree_processing`
- `outputs/stats`
- `outputs/report`
- `blender/bexport`
- `blender/blenderv2`

## Next Steps

1. Do the dedicated `input_processing/tree_processing` cleanup pass.
2. Simplify the tree-variant / template structure now that the sim and Blender runtime paths are stable.
3. Keep updating docs that still mention `scenario/` or the old Blender folders.
4. Keep using `uv` with explicit `TREE_TEMPLATE_ROOT` and `REFACTOR_RUN_OUTPUT_ROOT` for any further validation runs.

## Recommended Prompt For Next Session

- focus on `input_processing/tree_processing`
- keep the current `sim/*`, `outputs/*`, and `blender/*` structure stable unless there is a strong reason to move it again
- continue preferring direct moves over wrappers or duplicate live paths
