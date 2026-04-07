# Blender Output Hooks

This note defines the Blender-facing endpoints for the limited refactor.

## Planned Refactored Root

All Blender-facing endpoints move under:

- `_data-refactored/final-hooks/`

Current first-pass folders:

- `vtks/{site}/`
- `feature-locations/{site}/`
- `world/{site}/`
- `bioenvelopes/{site}/`
- `feature-libraries/treePlyLibrary/`
- `feature-libraries/logPlyLibrary/`
- `baselines/{site}/`

Current first-pass naming rules:

- scenario and baseline state VTKs use the cleaned `state_with_indicators` naming under `vtks/{site}/`
- state `nodeDF` tables move to `feature-locations/{site}/`
- world files use `buildings` and `road`
- bioenvelope files keep their current filenames
- baseline support files keep their current filenames inside `baselines/{site}/`
- `ground_scenarioYR{year}.ply` is out of scope for the refactored hook set
- hook ids follow the refactored folder names: `VTKS`, `FEATURE_LOCATIONS`, `FEATURE_LIBRARIES`, `WORLD`, `BIOENVELOPES`, `BASELINES`

## Hook Set

| Hook ID | Blender-facing endpoint | Current path pattern | Planned refactored path pattern | Main Blender consumer(s) | AGENTS ref | Pipeline ref | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `HOOK_VTKS_STATE-LATEST` | latest assessed state VTK | `data/revised/final/output/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features_with_indicators.vtk` | `_data-refactored/final-hooks/vtks/{site}/{site}_{scenario}_{voxel_size}_yr{year}_state_with_indicators.vtk` | `b2026_transfer_vtk_sim_layers.py`, `b_generate_rewilded_envelopes.py`, `b_generate_rewilded_ground.py` | [`2.1`](./AGENTS.md#21-state-and-baseline-products) | `2.3-INFO-1` | This is the latest per-state VTK. |
| `HOOK_VTKS_BASELINE-STATE` | baseline assessed state VTK | `data/revised/final-v3/baselines/{site}_baseline_combined_{voxel_size}_urban_features.vtk` | `_data-refactored/v3engine_outputs/vtks/{site}/{site}_baseline_{voxel_size}_state_with_indicators.vtk` | `a_info_gather_capabilities.py` | [`2.1`](./AGENTS.md#21-state-and-baseline-products) | `2.5-BASELINE-1` and `2.3-INFO-1` | This is the active baseline state VTK used by the assessment pass. |
| `HOOK_FEATURE_LOCATIONS_NODEDF` | instancer node table | `_data-refactored/v3engine_outputs/feature-locations/{site}/{site}_{scenario}_{voxel_size}_nodeDF_yr{year}.csv` | `_data-refactored/v3engine_outputs/feature-locations/{site}/{site}_{scenario}_{voxel_size}_nodeDF_yr{year}.csv` | `b2026_instancer.py` | [`2.1`](./AGENTS.md#21-state-and-baseline-products) | `2.2-SCENARIO-3` | This is now the live refactored instancer table. |
| `HOOK_FEATURE_LIBRARIES_TREE-PLY` | tree and pole template PLY library | `_data-refactored/model-inputs/tree_library_exports/treeMeshesPly/*.ply` | `_data-refactored/model-inputs/tree_library_exports/treeMeshesPly/*.ply` | `b2026_instancer.py` | [`2.2`](./AGENTS.md#22-tree--log--pole-ply-libraries) | `3` | In the 2026 instancer, this folder also carries the pole / artificial-support templates. |
| `HOOK_FEATURE_LIBRARIES_LOG-PLY` | log template PLY library | `_data-refactored/model-inputs/tree_library_exports/logMeshesPly/*.ply` | `_data-refactored/model-inputs/tree_library_exports/logMeshesPly/*.ply` | `b2026_instancer.py` | [`2.2`](./AGENTS.md#22-tree--log--pole-ply-libraries) | `3` | Keep the current log-template rules. |
| `HOOK_WORLD_BUILDINGS-PLY` | base world buildings PLY | `data/revised/final/{site}/{site}_buildings.ply` | `_data-refactored/final-hooks/world/{site}/{site}_buildings.ply` | scene imports, `b2026_world_cubes.py` | [`2.3`](./AGENTS.md#23-base-world-plys) | `4-WORLDPLY-3` | Shared site-level endpoint. |
| `HOOK_WORLD_ROAD-PLY` | base world road PLY | `data/revised/final/{site}/{site}_highResRoad.ply` | `_data-refactored/final-hooks/world/{site}/{site}_road.ply` | scene imports, `b2026_world_cubes.py` | [`2.3`](./AGENTS.md#23-base-world-plys) | `4-WORLDPLY-3` | Rename `highResRoad` to `road` in the refactored hook layer. |
| `HOOK_BIOENVELOPES-PLY` | bioenvelope shell PLY | `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.ply` | `_data-refactored/final-hooks/bioenvelopes/{site}/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.ply` | scene imports, `b2026_clipbox_setup.py` | [`2.4`](./AGENTS.md#24-envelope-plys) | `2.4-ENVELOPE-1` and `5` | Keep the current bioenvelope filename pattern for the first pass. |
| `HOOK_BASELINES_TREE-CSV` | baseline tree CSV | `data/revised/final-v3/baselines/{site}_baseline_trees.csv` | `_data-refactored/v3engine_outputs/baselines/{site}/{site}_baseline_trees.csv` | `b2026_build_city_baseline.py` | [`2.1`](./AGENTS.md#21-state-and-baseline-products) | `2.5-BASELINE-1.1` | Baseline-specific consumer. |
| `HOOK_BASELINES_TERRAIN-PLY` | baseline terrain PLY | `data/revised/final/baselines/{site}_baseline_terrain_{voxel_size}.ply` | `_data-refactored/final-hooks/baselines/{site}/{site}_baseline_terrain_{voxel_size}.ply` | `b2026_build_city_baseline.py` | [`2.1`](./AGENTS.md#21-state-and-baseline-products) | `2.5-BASELINE-1` | Baseline-specific terrain surface. |

## Current Refactor Boundary

For the first pass, the limited refactor should only change the hooks above.

- Do change the root and cleaned hook layout for these files.
- Do not move the wider upstream simulation outputs yet.
- Do not add the superseded ground shell to the refactored Blender bundle.
