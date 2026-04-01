# Required Files Audit

This audit is based on:

- [`AGENTS.md`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/AGENTS.md)
- [`KEY_SCRIPTS.md`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/KEY_SCRIPTS.md)
- the main 2026 scripts used with the current production blend

Date checked: `2026-03-29`

## Short Answer

For the current city/parade 2026 Blender workflow, the project expects a repo-local `data/` tree under:

- `D:\2026 Arboreal Futures\urban-futures\data`

The minimum important paths are:

- `data/blender/2026/2026 futures heroes6.blend`
- `data/revised/final/treeMeshesPly`
- `data/revised/final/logMeshesPly`
- `data/revised/final/{site}/{site}_{scenario}_1_nodeDF_{year}.csv`
- `data/revised/final/{site}/{site}_{scenario}_1_scenarioYR{year}.vtk`

Additional files are required depending on whether we are doing:

- world cube rebuilds
- envelope import
- baseline scene generation
- EXR output/compositor generation

## Canonical Locations From The Agent Guide

The agent guide defines these as the canonical data families:

- production blend files:
  - `data/blender/2026/2026 futures heroes6.blend`
  - `data/blender/2026/2026 futures heroes6_baseline.blend`
- tree and pole PLY library:
  - `data/revised/final/treeMeshesPly`
- log PLY library:
  - `data/revised/final/logMeshesPly`
- per-site state folders:
  - `data/revised/final/city`
  - `data/revised/final/trimmed-parade`
  - `data/revised/final/uni`
- baseline products:
  - `data/revised/final/baselines/...`

## Files Required By Workflow

### 1. Heavy Production Blend

Needed to open and use the main heavy scene:

- `data/blender/2026/2026 futures heroes6.blend`

Optional but documented:

- `data/blender/2026/2026 futures heroes6_baseline.blend`

## 2. Instancer Inputs

The instancer is the main dependency driver.

Per [`b2026_instancer.py`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/b2026_instancer.py), it expects:

- tree/pole PLY library:
  - `data/revised/final/treeMeshesPly`
- log PLY library:
  - `data/revised/final/logMeshesPly`
- per-site CSV folder:
  - `data/revised/final/{site}`

The actual CSV filename pattern is:

- `data/revised/final/{site}/{site}_{scenario}_1_nodeDF_{year}.csv`

Examples for the current scripts:

- `data/revised/final/city/city_positive_1_nodeDF_180.csv`
- `data/revised/final/city/city_trending_1_nodeDF_180.csv`
- `data/revised/final/trimmed-parade/trimmed-parade_positive_1_nodeDF_180.csv`
- `data/revised/final/trimmed-parade/trimmed-parade_trending_1_nodeDF_180.csv`

Important behavior:

- if both scenario CSVs exist, the instancer will build both `positive` and `trending`
- for city, `positive` also drives the `city_priority` branch
- `treeMeshesPly` also needs to contain the pole templates used for `nodeType == pole`

### 3. World Attribute Transfer Inputs

Per [`b2026_transfer_vtk_sim_layers.py`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/b2026_transfer_vtk_sim_layers.py), world/base meshes expect per-state VTKs at:

- `data/revised/final/trimmed-parade/trimmed-parade_{scenario}_1_scenarioYR{year}.vtk`
- `data/revised/final/city/city_{scenario}_1_scenarioYR{year}.vtk`

Examples:

- `data/revised/final/city/city_positive_1_scenarioYR180.vtk`
- `data/revised/final/city/city_trending_1_scenarioYR180.vtk`

This script also expects:

- a Python environment at `.venv`
- specifically `.venv/bin/python` in the current script, which is Unix-shaped and will need adjustment on Windows

The script writes cache/report outputs under:

- `data/blender/2026/vtk_sim_layer_cache`
- `data/blender/2026/vtk_year180_point_data_layers.md`

### 4. Base World Geometry Inputs

Per [`AGENTS.md`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/AGENTS.md), the base-world extraction step ultimately produces site-level world PLYs:

- `{site}_buildings.ply`
- `{site}_highResRoad.ply`

These are generated from:

- `data/revised/{site}-siteVoxels-masked.vtk`
- `data/revised/{site}-roadVoxels-coloured.vtk`

The heavy blend appears to already contain imported world objects such as:

- `city_buildings`
- `city_highResRoad`
- `city_buildings.001`
- `city_highResRoad.001`

So these source files are required when rebuilding or refreshing the world, not necessarily for merely opening the existing blend.

### 5. Envelope Inputs

Per the guide, envelope PLYs are expected at:

- `data/revised/final/{site}/ply/{site}_{scenario}_1_envelope_scenarioYR{year}.ply`
- `data/revised/final/{site}/ply/{site}_{scenario}_1_ground_scenarioYR{year}.ply`

Confirmed directly in [`b2026_unify_parade_envelope.py`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/b2026_unify_parade_envelope.py):

- `data/revised/final/trimmed-parade/ply/trimmed-parade_positive_1_envelope_scenarioYR180.ply`

The city scene contract expects envelope collections such as:

- `city_envelope`
- `Envelope_trending`

### 6. Baseline Inputs

Per [`b2026_build_city_baseline.py`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/b2026_build_city_baseline.py), baseline generation expects:

- `data/revised/final/baselines/city_baseline_terrain_1.ply`
- `data/revised/final/baselines/city_baseline_trees.csv`

And the guide also documents related baseline files:

- `data/revised/final/baselines/{site}_baseline_resources_{voxel_size}.vtk`
- `data/revised/final/baselines/{site}_baseline_terrain_{voxel_size}.vtk`
- `data/revised/final/baselines/{site}_baseline_combined_{voxel_size}.vtk`

Baseline render outputs are written to:

- `data/blender/2026/baseline_renders/...`

### 7. EXR Output And Compositor Inputs

Per [`b2026_setup_view_layer_exr_outputs.py`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/b2026_setup_view_layer_exr_outputs.py), EXRs are written beside the saved source blend:

- folder pattern:
  - `data/blender/2026/{blend-stem}-{scene-name}/`
- EXR pattern:
  - `{scene-name}-{view-layer-name}.exr`

Per [`b2026_build_city_exr_compositor.py`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/b2026_build_city_exr_compositor.py), the compositor expects:

- source blend:
  - `data/blender/2026/2026 futures heroes6.blend`
- source scene:
  - `city`
- EXRs in:
  - `data/blender/2026/2026 futures heroes6-city/`

Expected EXR examples:

- `data/blender/2026/2026 futures heroes6-city/city-pathway_state.exr`
- `data/blender/2026/2026 futures heroes6-city/city-existing_condition.exr`
- `data/blender/2026/2026 futures heroes6-city/city-city_priority.exr`
- `data/blender/2026/2026 futures heroes6-city/city-city_bioenvelope.exr`
- `data/blender/2026/2026 futures heroes6-city/city-trending_state.exr`

## Files You Have Already Given Me

Currently visible outside the repo in `D:\2026 Arboreal Futures\data`:

- `2026 futures heroes6.blend`
- `treeMeshesPly`
- `logMeshesPLY`
- `treeMeshes`
- `logMeshes`
- `tree_VTKpts`

These correspond most clearly to:

- `2026 futures heroes6.blend` -> `data/blender/2026/2026 futures heroes6.blend`
- `treeMeshesPly` -> `data/revised/final/treeMeshesPly`
- `logMeshesPLY` -> `data/revised/final/logMeshesPly`

The following still appear missing from the copied set, based on the script contract:

- all per-state `nodeDF` CSVs
- all per-state scenario VTKs
- baseline CSV/PLY/VTK files
- site world source VTKs or exported world PLYs
- envelope PLYs
- baseline blend
- EXR output folders

## Important Path Reality Check

There is a mismatch between the documentation and some live scripts:

- newer scripts use repo-relative `REPO_ROOT / data / ...` style paths
- some important scripts, especially [`b2026_instancer.py`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/b2026_instancer.py), still contain hard-coded macOS paths under:
  - `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/...`

That means the files are not only required logically; they must either:

- exist under the repo-local `data/...` tree and the scripts be patched to use it
- or be mirrored in whatever path layout those hard-coded scripts still expect

## Practical Minimum To Get The Main Blend Running

If the goal is to get `2026 futures heroes6.blend` working with the main current city workflow, the minimum file set is:

- `data/blender/2026/2026 futures heroes6.blend`
- `data/revised/final/treeMeshesPly/*.ply`
- `data/revised/final/logMeshesPly/*.ply`
- `data/revised/final/city/city_positive_1_nodeDF_180.csv`
- `data/revised/final/city/city_trending_1_nodeDF_180.csv`

If you also want world simulation attributes and full render pipeline support, add:

- `data/revised/final/city/city_positive_1_scenarioYR180.vtk`
- `data/revised/final/city/city_trending_1_scenarioYR180.vtk`
- `data/revised/final/city/ply/city_positive_1_envelope_scenarioYR180.ply`
- `data/revised/final/city/ply/city_trending_1_envelope_scenarioYR180.ply`

If you want baseline generation too, add:

- `data/revised/final/baselines/city_baseline_trees.csv`
- `data/revised/final/baselines/city_baseline_terrain_1.ply`
