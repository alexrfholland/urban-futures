# Blender Data Workflow

This file maps the Blender-facing data endpoints back through the scripts that generate them and the earlier project files they depend on.

## 1. Blend Endpoints

Active heavy blends:

- `data/blender/2026/2026 futures heroes6.blend`
- `data/blender/2026/2026 futures heroes6_baseline.blend`

Main external data endpoints used by the heavy Blender scene:

- base world PLYs: `data/revised/final/{site}/{site}_buildings.ply`, `data/revised/final/{site}/{site}_highResRoad.ply`
- state instancer CSVs: `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_nodeDF_{year}.csv`
- state VTKs: `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`
- tree / pole PLY library: `data/revised/final/treeMeshesPly/*.ply`
- log PLY library: `data/revised/final/logMeshesPly/*.ply`
- envelope PLYs: `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.ply`
- baseline products: `data/revised/final/baselines/*`

## 2. Data Types

### 2.1. State And Baseline Products

Blender-facing endpoints:

- state instancer CSV: `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_nodeDF_{year}.csv`
- state VTK: `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`
- baseline trees CSV: `data/revised/final/baselines/{site}_baseline_trees.csv`
- baseline resource VTK: `data/revised/final/baselines/{site}_baseline_resources_{voxel_size}.vtk`
- baseline terrain VTK: `data/revised/final/baselines/{site}_baseline_terrain_{voxel_size}.vtk`
- baseline combined VTK: `data/revised/final/baselines/{site}_baseline_combined_{voxel_size}.vtk`

State product chain:

1. `final/a_scenario_generateVTKs.py`
   - reads `data/revised/final/{site}/{site}_{voxel_size}_subsetForScenarios.nc`
   - reads `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv`
   - reads `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_logDF_{year}.csv`
   - reads `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_poleDF_{year}.csv`
   - writes `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_nodeDF_{year}.csv`
   - writes `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`
2. `final/a_scenario_runscenario.py`
   - reads the initialized tree / log / pole dataframes in memory
   - writes `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv`
   - writes `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_logDF_{year}.csv`
   - writes `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_poleDF_{year}.csv`
3. `final/a_scenario_initialiseDS.py`
   - reads `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_RewildingNodes.nc`
   - reads `data/revised/final/{site}/{site}_{voxel_size}_treeDF.csv`
   - reads `data/revised/final/{site}/{site}_{voxel_size}_poleDF.csv`
   - reads `data/revised/final/{site}/{site}_{voxel_size}_logDF.csv`
   - optionally reads `data/revised/final/{site}/{site}-extraPoleDF.csv`
   - optionally reads `data/revised/final/{site}/{site}-extraTreeDF.csv`
   - writes `data/revised/final/{site}/{site}_{voxel_size}_subsetForScenarios.nc`
4. `final/a_manager.py`
   - writes `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_initial.nc`
   - writes `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withTreesAndPoleLocations.nc`
   - writes `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withLogs.nc`
   - writes `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withResistance.nc`
   - writes `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withResistanceSUBSET.nc`
   - writes `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_RewildingNodes.nc`
   - writes updated `treeDF`, `poleDF`, `logDF`, and `node_mappingsDF` CSVs into `data/revised/final/{site}/`
5. `final/f_further_processing.py`
   - reads `data/revised/{site}-siteVoxels.vtk`
   - reads `data/revised/{site}-roadVoxels.vtk`
   - reads `data/revised/{site}-treeVoxels.vtk`
   - writes `data/revised/final/{site}-siteVoxels-masked.vtk`
   - writes `data/revised/final/{site}-roadVoxels-coloured.vtk`
   - writes `data/revised/final/{site}-treeVoxels-coloured.vtk`
   - writes `data/revised/final/{site}-canopyVoxels.vtk`
6. `final/f_manager.py`
   - reads `data/revised/csv/site_locations.csv`
   - reads `data/revised/experimental/DevelopmentActivityModel-trimmed-metric.glb`
   - reads photo-mesh VTKs from `data/revised/obj/_converted/*.vtk`
   - reads photo-mesh metadata from `data/revised/obj/_converted/converted_mesh_centroids.csv`
   - reads contour shapefile `data/revised/shapefiles/contours/EL_CONTOUR_1TO5M.shp`
   - reads urban forest records from the City of Melbourne urban forest API
   - reads road segment records from the City of Melbourne road segments API
   - writes `data/revised/{site}-tree-locations.csv`
   - writes `data/revised/{site}-treeVoxels.vtk`
   - writes `data/revised/{site}-roadVoxels.vtk`
   - writes `data/revised/{site}-siteVoxels.vtk`

Important upstream side inputs in the state chain:

- `data/csvs/pre-colonial-plant-list.csv`
- `data/revised/trees/logLibraryStats.csv`
- `data/revised/shapefiles/pylons/pylons.shp`
- `data/revised/shapefiles/pylons/streetlights.shp`

Baseline product chain:

1. `final/a_scenario_get_baselines.py`
   - reads `data/csvs/tree-baseline-density.csv`
   - reads `data/revised/final/{site}/{site}_{voxel_size}_subsetForScenarios.nc`
   - reads `data/revised/final/{site}-roadVoxels-coloured.vtk`
   - calls `final/a_resource_distributor_dataframes.py`
   - writes `data/revised/final/baselines/{site}_baseline_trees.csv`
   - writes `data/revised/final/baselines/{site}_baseline_resources_{voxel_size}.vtk`
   - writes `data/revised/final/baselines/{site}_baseline_terrain_{voxel_size}.vtk`
   - writes `data/revised/final/baselines/{site}_baseline_combined_{voxel_size}.vtk`
2. `final/a_resource_distributor_dataframes.py`
   - reads `data/revised/trees/{voxel_size}_voxel_tree_dict.pkl`
   - expands baseline tree rows into resource point clouds before baseline VTK export
3. the baseline branch still depends on the same site voxel / road voxel chain listed above

Notes:

- the state VTK is also read directly by `final/_blender/2026/b2026_transfer_vtk_sim_layers.py`
- the baseline scene path is separate, but the baseline products above are the baseline Blender-side data endpoints

### 2.2. Tree / Log / Pole PLY Libraries

Blender-facing endpoints:

- tree / pole library: `data/revised/final/treeMeshesPly/*.ply`
- log library: `data/revised/final/logMeshesPly/*.ply`

Immediate Blender-facing generator:

1. `final/_blender/a_vtk_to_ply.py`
   - reads `data/revised/final/treeMeshes/*.vtk`
   - writes `data/revised/final/treeMeshesPly/*.ply`
   - reads `data/revised/final/logMeshes/*.vtk`
   - writes `data/revised/final/logMeshesPly/*.ply`

Tree mesh chain:

1. `final/tree_processing/combine_resource_treeMeshGenerator.py`
   - reads `_data-refactored/model-inputs/tree_libraries/base/trees/template-library.overrides-applied.pkl` or `_data-refactored/model-inputs/tree_libraries/base/trees/template-library.selected-overrides.pkl`
   - writes `data/revised/final/treeMeshes/*.vtk`
2. `final/tree_processing/combined_tree_manager.py`
   - reads `data/revised/trees/elm_tree_dict.pkl`
   - reads `data/revised/trees/updated_tree_dict.pkl`
   - reads `data/revised/trees/resource_dicDF.csv`
   - reads `data/revised/lidar scans/elm/adtree/processedGraph/*.graphml`
   - writes `_data-refactored/model-inputs/tree_libraries/base/trees/template-library.overrides-applied.pkl`
   - writes `_data-refactored/model-inputs/tree_libraries/base/trees/template-library.selected-overrides.pkl`
3. `final/tree_processing/combined_generateResourceDict.py`
   - reads `data/csvs/lerouxdata-update.csv`
   - writes `data/revised/trees/resource_dicDF.csv`
4. legacy deeper tree-template provenance noted in `combined_tree_manager.py`
   - `data/revised/revised_tree_dict.pkl`
   - `data/treeOutputs/adjusted_tree_templates.pkl`
   - `data/treeOutputs/fallen_trees_dict.pkl`
   - these older files are described there as coming from `modules/treeBake_recreateLogs.py` and `modules/treeBake_treeAging.py`

Log mesh chain:

1. `final/tree_processing/a_log_mesh_generator.py`
   - reads `data/treeOutputs/logLibrary.pkl`
   - writes `data/revised/final/logMeshes/*.vtk`
2. `final/_blender/a_vtk_to_ply.py`
   - writes `data/revised/final/logMeshesPly/*.ply`
3. `modules/treeBake_recreateLogs.py`
   - writes `data/treeOutputs/logLibrary.pkl`

Pole / artificial support chain:

- actual pole node positions come from `final/a_urban_forest_parser.py`, which reads:
  - `data/revised/shapefiles/pylons/pylons.shp`
  - `data/revised/shapefiles/pylons/streetlights.shp`
- artificial pole / snag template assets are added by `final/tree_processing/b_generate_utility_pole_and_artificial_tree.py`, which:
  - reads `data/revised/utlity poles/utility_pole.ply`
  - writes `data/revised/final/treeMeshes/*.vtk`
  - writes `data/revised/final/treeMeshesPly/*.ply`

### 2.3. Base World PLYs

Blender-facing endpoints:

- `data/revised/final/{site}/{site}_buildings.ply`
- `data/revised/final/{site}/{site}_highResRoad.ply`

Direct generator:

1. `final/_blender/b_extract_scene.py`
   - reads `data/revised/{site}-siteVoxels-masked.vtk`
   - reads `data/revised/{site}-roadVoxels-coloured.vtk`
   - writes `data/revised/final/{site}/{site}_buildings.ply`
   - writes `data/revised/final/{site}/{site}_highResRoad.ply`

Upstream chain:

1. `final/f_further_processing.py`
   - reads `data/revised/{site}-siteVoxels.vtk`
   - reads `data/revised/{site}-roadVoxels.vtk`
   - writes masked / coloured world VTKs
2. `final/f_manager.py`
   - reads `data/revised/csv/site_locations.csv`
   - reads `data/revised/experimental/DevelopmentActivityModel-trimmed-metric.glb`
   - reads photo-mesh VTKs from `data/revised/obj/_converted/*.vtk`
   - reads `data/revised/obj/_converted/converted_mesh_centroids.csv`
   - reads contour shapefile `data/revised/shapefiles/contours/EL_CONTOUR_1TO5M.shp`
   - reads the City of Melbourne urban forest API
   - reads the City of Melbourne road segments API
   - writes the original site / road VTKs used by `f_further_processing.py`

Current path mismatch to note:

- `b_extract_scene.py` currently reads `data/revised/{site}-siteVoxels-masked.vtk` and `data/revised/{site}-roadVoxels-coloured.vtk`
- `f_further_processing.py` currently writes those files to `data/revised/final/{site}-siteVoxels-masked.vtk` and `data/revised/final/{site}-roadVoxels-coloured.vtk`
- this is exactly the kind of path contract that should be normalized during refactoring

### 2.4. Envelope PLYs

Blender-facing endpoints:

- envelope shell PLY: `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.ply`
- envelope surface VTK: `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.vtk`

Direct generator:

1. `final/_blender/b_generate_rewilded_envelopes.py`
   - reads `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`
   - reads `data/revised/{site}-siteVoxels-masked.vtk`
   - reads `data/revised/{site}-roadVoxels-coloured.vtk`
   - writes `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.ply`
   - writes `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.vtk`

Envelope upstream chain:

1. state VTK branch
   - comes from the state product chain in section 2.1
2. reference world branch
   - the normals and reference surfaces come from the base-world voxel files in section 2.3

Related companion output:

- `final/_blender/b_generate_rewilded_ground.py` uses the same state VTK branch and writes:
  - `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_ground_scenarioYR{year}.ply`
  - `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_ground_scenarioYR{year}.vtk`

Current path mismatch to note:

- `b_generate_rewilded_envelopes.py` currently reads `data/revised/{site}-siteVoxels-masked.vtk` and `data/revised/{site}-roadVoxels-coloured.vtk`
- `f_further_processing.py` currently writes those files to `data/revised/final/`

## 3. Refactor Notes

The cleanest directory refactor for this Blender pipeline would start by normalizing these contracts before moving files:

- one canonical path helper for `data/revised/` vs `data/revised/final/`
- one canonical description of Blender endpoints per data type
- wrappers or aliases for old paths while scripts are migrated
- special attention to the tree-template branch, because it already mixes current `final/tree_processing/*` scripts with older `modules/treeBake_*` outputs
