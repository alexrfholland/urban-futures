# Data Pipeline Script Summaries

This file gives one structured entry per script or key step in the Blender data pipeline, using the same four fields for each step: inputs, description of processing, outputs, and chain notes where provenance breaks.

## 2. State And Baseline Products

Section note:
- Use the step codes below for cross-reference.
- The envelope branch in `2.4` diverges from the state branch after `2.2-SCENARIO-3`.

### 2.1. World Preparation

`2.1-WORLD-1. Build raw site voxel datasets`
Script:
- `final/f_manager.py`
Inputs:
- `site locations [csv]` - `data/revised/csv/site_locations.csv`
- `development activity model [glb]` - `data/revised/experimental/DevelopmentActivityModel-trimmed-metric.glb`
- `converted context meshes [vtk]` - `data/revised/obj/_converted/*.vtk`
- `context centroids [csv]` - `data/revised/obj/_converted/converted_mesh_centroids.csv`
- `contours [shapefile]` - `data/revised/shapefiles/contours/EL_CONTOUR_1TO5M.shp`
- `urban forest records [api]` - City of Melbourne urban forest API
- `road segment records [api]` - City of Melbourne road segment API
Description of processing:
- Crops terrain, roads, buildings, photo meshes, and urban-forest points to the site and writes the first per-site voxel datasets.
Outputs:
- `tree locations [csv]` - `data/revised/{site}-tree-locations.csv`
- `tree voxels [vtk]` - `data/revised/{site}-treeVoxels.vtk`
- `road voxels [vtk]` - `data/revised/{site}-roadVoxels.vtk`
- `site voxels [vtk]` - `data/revised/{site}-siteVoxels.vtk`

`2.1-WORLD-2. Clean the world voxel datasets`
Script:
- `final/f_further_processing.py`
Inputs:
- `site voxels [vtk]` - `data/revised/{site}-siteVoxels.vtk`
- `road voxels [vtk]` - `data/revised/{site}-roadVoxels.vtk`
- `tree voxels [vtk]` - `data/revised/{site}-treeVoxels.vtk`
Description of processing:
- Filters the raw site voxels to buildings, adds roof and canopy fields, recenters the geometry, and writes the cleaned world VTKs used by both simulation and envelope export.
Outputs:
- `masked site voxels [vtk]` - `data/revised/final/{site}-siteVoxels-masked.vtk`
- `coloured road voxels [vtk]` - `data/revised/final/{site}-roadVoxels-coloured.vtk`
- `coloured tree voxels [vtk]` - `data/revised/final/{site}-treeVoxels-coloured.vtk`
- `canopy voxels [vtk]` - `data/revised/final/{site}-canopyVoxels.vtk`
Chain note:
- These cleaned world files are later picked up by `2.2-SITE-1` and, via cross-reference, by `2.4-ENVELOPE-1`.

### 2.2. Simulation Pipeline

`2.2-SITE-1. Build the base simulation datasets and node tables`
Script:
- `final/a_manager.py`
Inputs:
- `masked site voxels [vtk]` - `data/revised/final/{site}-siteVoxels-masked.vtk`
- `coloured road voxels [vtk]` - `data/revised/final/{site}-roadVoxels-coloured.vtk`
- `tree locations [csv]` - `data/revised/{site}-tree-locations.csv`
- `log library stats [csv]` - `data/revised/trees/logLibraryStats.csv`
- `pole shapefiles [shapefile]` - `data/revised/shapefiles/pylons/pylons.shp`
- `streetlight shapefiles [shapefile]` - `data/revised/shapefiles/pylons/streetlights.shp`
Description of processing:
- Orchestrates the site-to-simulation build by voxelizing the cleaned world, attaching trees and poles, distributing roof logs, computing resistance, and generating rewilding nodes.
Outputs:
- `initial site array [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_initial.nc`
- `site array with trees and poles [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withTreesAndPoleLocations.nc`
- `site array with logs [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withLogs.nc`
- `site array with resistance [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withResistance.nc`
- `site array resistance subset [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withResistanceSUBSET.nc`
- `site array with rewilding nodes [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_RewildingNodes.nc`
- `treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_treeDF.csv`
- `poleDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_poleDF.csv`
- `logDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_logDF.csv`
- `node mappings [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_node_mappingsDF.csv`
Optional reuse:
- If `a_manager.py` runs with `stage == 'resistance'`, it skips `2.2-SITE-1.1` to `2.2-SITE-1.3` and reloads `voxelArray_withLogs.nc`, `treeDF.csv`, `poleDF.csv`, and `logDF.csv` before continuing at `2.2-SITE-1.4`.

`2.2-SITE-1.1. Create the initial site xarray`
Script:
- `final/a_manager.py` via `a_voxeliser.voxelize_polydata_and_create_xarray`
Inputs:
- `2.1-WORLD-2` cleaned world VTKs
Description of processing:
- Voxelizes the cleaned site world into the first xarray dataset for simulation.
Outputs:
- `initial site array [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_initial.nc`

`2.2-SITE-1.2. Attach trees and poles`
Script:
- `final/a_manager.py` via `a_urban_forest_parser.get_resource_dataframe`
Inputs:
- `initial site array [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_initial.nc`
- `tree locations [csv]` - `data/revised/{site}-tree-locations.csv`
- `pole shapefiles [shapefile]` - `data/revised/shapefiles/pylons/pylons.shp`
- `streetlight shapefiles [shapefile]` - `data/revised/shapefiles/pylons/streetlights.shp`
Description of processing:
- Adds tree and pole nodes to the site array and writes the first node tables.
Outputs:
- `site array with trees and poles [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withTreesAndPoleLocations.nc`
- `treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_treeDF.csv`
- `poleDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_poleDF.csv`

`2.2-SITE-1.3. Distribute roof logs`
Script:
- `final/a_manager.py` via `a_logDistributor.process_roof_logs`
Inputs:
- `site array with trees and poles [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withTreesAndPoleLocations.nc`
- `log library stats [csv]` - `data/revised/trees/logLibraryStats.csv`
Description of processing:
- Places roof-log structures into the site array and writes the log table and grouped roof metadata.
Outputs:
- `site array with logs [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withLogs.nc`
- `logDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_logDF.csv`
- `grouped roof info [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_grouped_roof_info.csv`

`2.2-SITE-1.4. Compute resistance`
Script:
- `final/a_manager.py` via `a_create_resistance_grid.get_resistance`
Inputs:
- `site array with logs [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withLogs.nc`
- `treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_treeDF.csv`
- `poleDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_poleDF.csv`
- `logDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_logDF.csv`
Description of processing:
- Computes resistance fields for the full site array and writes both the full and subset resistance datasets.
Outputs:
- `site array with resistance [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withResistance.nc`
- `site array resistance subset [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withResistanceSUBSET.nc`

`2.2-SITE-1.5. Generate rewilding nodes`
Script:
- `final/a_manager.py` via `a_rewilding.GetRewildingNodes`
Inputs:
- `site array with resistance [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withResistance.nc`
- `updated treeDF [dataframe]` - in-memory result from `2.2-SITE-1.4`
- `updated poleDF [dataframe]` - in-memory result from `2.2-SITE-1.4`
- `updated logDF [dataframe]` - in-memory result from `2.2-SITE-1.4`
Description of processing:
- Writes rewilding-node fields back into the site array and persists the final base simulation tables.
Outputs:
- `site array with rewilding nodes [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_RewildingNodes.nc`
- `treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_treeDF.csv`
- `poleDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_poleDF.csv`
- `logDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_logDF.csv`
Chain note:
- If `a_manager.py` runs with `stage == 'resistance'`, it skips `2.2-SITE-1.1` to `2.2-SITE-1.3` and reloads `voxelArray_withLogs.nc`, `treeDF.csv`, `poleDF.csv`, and `logDF.csv` before continuing at `2.2-SITE-1.4`.

#### Scenario Manager

`2.2-SCENARIO-1. Prepare the scenario-ready dataset`
Script:
- `final/a_scenario_manager.py`
Inputs:
- `cellular-automata-rewilding [xarray]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_RewildingNodes.nc`
- `treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_treeDF.csv`
- `poleDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_poleDF.csv`
- `logDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_logDF.csv`
- `extraTreeDF [optional dataframe csv]` - `data/revised/final/{site}/{site}-extraTreeDF.csv`
- `extraPoleDF [optional dataframe csv]` - `data/revised/final/{site}/{site}-extraPoleDF.csv`
Description of processing:
- Uses the scenario manager to load the base simulation products, subset the xarray, normalize the node tables, inject optional extra nodes, and prepare the scenario-ready dataset.
Outputs:
- `subset dataset [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_subsetForScenarios.nc`
- `initial scenario poly [vtk]` - `data/revised/final/{site}/{site}_{voxel_size}_scenarioInitialPoly.vtk`

`2.2-SCENARIO-1.1. Load and subset the rewilding-node dataset`
Script:
- `final/a_scenario_manager.py` via `a_scenario_initialiseDS.initialize_dataset`
Inputs:
- `site array with rewilding nodes [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_RewildingNodes.nc`
Description of processing:
- Loads the rewilding-node dataset, selects the scenario fields, and writes the subset dataset.
Outputs:
- `subset dataset [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_subsetForScenarios.nc`

`2.2-SCENARIO-1.2. Load base and optional node tables`
Script:
- `final/a_scenario_manager.py` via `a_scenario_initialiseDS.load_node_dataframes` and `load_extra_node_dataframes`
Inputs:
- `treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_treeDF.csv`
- `poleDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_poleDF.csv`
- `logDF [dataframe csv]` - `data/revised/final/{site}/{site}_{voxel_size}_logDF.csv`
- `extraTreeDF [optional dataframe csv]` - `data/revised/final/{site}/{site}-extraTreeDF.csv`
- `extraPoleDF [optional dataframe csv]` - `data/revised/final/{site}/{site}-extraPoleDF.csv`
Description of processing:
- Loads the persisted node tables and discovers optional extra tree and pole tables when they exist.
Outputs:
- `treeDF [dataframe]` - in-memory dataset for scenario initialization
- `poleDF [dataframe]` - in-memory dataset for scenario initialization
- `logDF [dataframe]` - in-memory dataset for scenario initialization
- `extraTreeDF [optional dataframe]` - in-memory dataset for scenario initialization
- `extraPoleDF [optional dataframe]` - in-memory dataset for scenario initialization

`2.2-SCENARIO-1.3. Normalize node tables for scenario use`
Script:
- `final/a_scenario_manager.py` via `a_scenario_initialiseDS.PreprocessData`, `further_xarray_processing`, `log_processing`, and `pole_processing`
Inputs:
- `subset dataset [xarray]` - in-memory result from `2.2-SCENARIO-1.1`
- `treeDF [dataframe]` - in-memory result from `2.2-SCENARIO-1.2`
- `poleDF [dataframe]` - in-memory result from `2.2-SCENARIO-1.2`
- `logDF [dataframe]` - in-memory result from `2.2-SCENARIO-1.2`
- `extraTreeDF [optional dataframe]` - in-memory result from `2.2-SCENARIO-1.2`
- `extraPoleDF [optional dataframe]` - in-memory result from `2.2-SCENARIO-1.2`
Description of processing:
- Normalizes the tree, pole, and log tables against the subset dataset, computes `sim_averageResistance`, and writes the initial scenario polydata.
Outputs:
- `initial scenario poly [vtk]` - `data/revised/final/{site}/{site}_{voxel_size}_scenarioInitialPoly.vtk`
- `initialized treeDF [dataframe]` - in-memory dataset for scenario runs
- `initialized poleDF [dataframe]` - in-memory dataset for scenario runs
- `initialized logDF [dataframe]` - in-memory dataset for scenario runs

`2.2-SCENARIO-2. Run or reuse yearly scenario tables`
Script:
- `final/a_scenario_manager.py`
Inputs:
- `subset dataset [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_subsetForScenarios.nc`
- `initialized treeDF [dataframe]` - in-memory result from `2.2-SCENARIO-1.3`
- `initialized poleDF [dataframe]` - in-memory result from `2.2-SCENARIO-1.3`
- `initialized logDF [dataframe]` - in-memory result from `2.2-SCENARIO-1.3`
- `scenario parameters [python config]` - `final/a_scenario_params.py`
Description of processing:
- Either runs the yearly scenario simulation or reuses previously written yearly dataframes when `skip_scenario` is enabled and all required files already exist.
Outputs:
- `scenario treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv`
- `scenario logDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_logDF_{year}.csv`
- `scenario poleDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_poleDF_{year}.csv`
Optional reuse:
- If `skip_scenario` is enabled and all required yearly CSVs already exist, `a_scenario_manager.py` reloads them instead of re-running `2.2-SCENARIO-2.1`.

`2.2-SCENARIO-2.1. Run the yearly scenario simulation`
Script:
- `final/a_scenario_manager.py` via `a_scenario_runscenario.run_scenario`
Inputs:
- `subset dataset [xarray]` - in-memory result from `2.2-SCENARIO-1`
- `initialized treeDF [dataframe]` - in-memory result from `2.2-SCENARIO-1.3`
- `initialized poleDF [dataframe]` - in-memory result from `2.2-SCENARIO-1.3`
- `initialized logDF [dataframe]` - in-memory result from `2.2-SCENARIO-1.3`
- `scenario parameters [python config]` - `final/a_scenario_params.py`
Description of processing:
- Applies the year-specific scenario logic to tree, log, and pole nodes and writes the yearly state tables.
Outputs:
- `scenario treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv`
- `scenario logDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_logDF_{year}.csv`
- `scenario poleDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_poleDF_{year}.csv`
Chain note:
- If `skip_scenario` is enabled and all yearly CSVs already exist, `a_scenario_manager.py` reuses them through `a_scenario_generateVTKs.load_scenario_dataframes` instead of re-running this substep.

`2.2-SCENARIO-3. Materialize the yearly scenario VTKs`
Script:
- `final/a_scenario_manager.py`
Inputs:
- `subset dataset [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_subsetForScenarios.nc`
- `scenario treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv`
- `scenario logDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_logDF_{year}.csv`
- `scenario poleDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_poleDF_{year}.csv`
Description of processing:
- Projects the yearly scenario tables back into the voxel grid, integrates resources, writes the yearly node table, and exports the state VTK that later branches reuse.
Outputs:
- `scenario nodeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_nodeDF_{year}.csv`
- `scenario state [vtk]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`
Chain note:
- This is the last shared upstream step before the envelope branch diverges.
- The downstream assessment branch continues through `2.2-SCENARIO-3.1` and `2.3-INFO-1` to `2.3-INFO-4`.

`2.2-SCENARIO-3.1. Add urban feature search layers`
Script:
- `final/a_scenario_urban_elements_count.py`
Inputs:
- `scenario state [vtk]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`
- `site array with logs [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_voxelArray_withLogs.nc`
Description of processing:
- Reads the classic state VTK, transfers site and envelope context onto it, derives the search layers used for assessment, and writes the urban-features VTK.
Outputs:
- `urban features state [vtk]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk`
Chain note:
- This step adds `FEATURES-*` lookup fields plus `search_bioavailable`, `search_design_action`, and `search_urban_elements`.

### 2.3. Assessment / Info Scripts

`2.3-INFO-1. Indicator Outcomes`
Script:
- `final/a_info_gather_capabilities.py`
- `final/a_info_output_capabilities.py`
Inputs:
- `urban features state [vtk]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk`
- `baseline urban features state [optional vtk]` - `data/revised/final/baselines/{site}_baseline_combined_{voxel_size}_urban_features.vtk`
Description of processing:
- `a_info_gather_capabilities.py` reads the urban-features VTK, applies the capability queries, writes one boolean indicator layer per outcome into a new VTK, and writes the per-site indicator and support-action CSVs.
- `a_info_output_capabilities.py` reads those per-site CSVs and writes the combined all-sites and pathway-summary CSVs; it does not write a VTK.
Outputs:
- `a_info_gather_capabilities.py` outputs:
- `indicator outcomes state [vtk]` - `data/revised/final/output/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features_with_indicators.vtk`
- `indicator counts [dataframe csv]` - `data/revised/final/output/csv/{site}_{voxel_size}_indicator_counts.csv`
- `action counts [dataframe csv]` - `data/revised/final/output/csv/{site}_{voxel_size}_action_counts.csv`
- `a_info_output_capabilities.py` outputs:
- `all sites indicator counts [dataframe csv]` - `data/revised/final/output/csv/all_sites_{voxel_size}_indicator_counts.csv`
- `all sites action counts [dataframe csv]` - `data/revised/final/output/csv/all_sites_{voxel_size}_action_counts.csv`
- `all sites proposal opportunities [dataframe csv]` - `data/revised/final/output/csv/all_sites_{voxel_size}_proposal_opportunities.csv`
- `all sites proposal interventions [dataframe csv]` - `data/revised/final/output/csv/all_sites_{voxel_size}_proposal_interventions.csv`
- `totals by site, persona, pathway [dataframe csv]` - `data/revised/final/output/csv/totals_by_site_persona_pathway_{voxel_size}.csv`
- `totals by pathway and persona [dataframe csv]` - `data/revised/final/output/csv/totals_by_pathway_persona_{voxel_size}.csv`
- `totals by pathway [dataframe csv]` - `data/revised/final/output/csv/totals_by_pathway_{voxel_size}.csv`
Chain note:
- The source of truth for the capability definitions lives in `a_info_gather_capabilities.py`.
- `a_info_gather_capabilities.py` is the only script in this step that writes a VTK.
- The VTK layer names use `indicator_{Persona}_{capability}_{indicator}` rather than the dotted ids in code.

Capability layers written into `*_urban_features_with_indicators.vtk`:

| Persona | Capability | Indicator label | VTK array name |
| --- | --- | --- | --- |
| Bird | `self` | Peeling bark volume | `indicator_Bird_self_peeling` |
| Bird | `others` | Perchable canopy volume | `indicator_Bird_others_perch` |
| Bird | `generations` | Hollow count | `indicator_Bird_generations_hollow` |
| Lizard | `self` | Ground cover area | `indicator_Lizard_self_grass` |
| Lizard | `self` | Dead branch volume | `indicator_Lizard_self_dead` |
| Lizard | `self` | Epiphyte count | `indicator_Lizard_self_epiphyte` |
| Lizard | `others` | Non-paved surface area | `indicator_Lizard_others_notpaved` |
| Lizard | `generations` | Nurse log volume | `indicator_Lizard_generations_nurse-log` |
| Lizard | `generations` | Fallen tree volume | `indicator_Lizard_generations_fallen-tree` |
| Tree | `self` | Senescing tree volume | `indicator_Tree_self_senescent` |
| Tree | `others` | Soil near canopy features | `indicator_Tree_others_notpaved` |
| Tree | `generations` | Grassland for recruitment | `indicator_Tree_generations_grassland` |

`2.3-INFO-2. Proposal Opportunities`
Script:
- `final/a_info_proposal_interventions.py`
Inputs:
- `indicator outcomes state [vtk]` - `data/revised/final/output/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features_with_indicators.vtk`
- `scenario treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv`
Description of processing:
- Reads the assessed scenario state and yearly tree table, then counts where each manuscript proposal becomes possible under the current code logic.
Outputs:
- `proposal opportunities [dataframe csv]` - `data/revised/final/output/csv/{site}_{voxel_size}_proposal_opportunities.csv`
Chain note:
- The manuscript defines five proposals: `Deploy Structure`, `Decay`, `Recruit`, `Colonise`, and `Release Control`.
- The current script computes `Decay`, `Colonise`, and `Release Control`, and still writes stub rows for `Deploy Structure` and `Recruit`.
- This step does not read `nodeDF`; it reads `treeDF_{year}.csv` and the assessed VTK.

Proposal opportunity tracking:

| Proposal | What gets counted as an opportunity to make this proposal | Current tracking | Node DF name | Node DF column | Node DF field/value used | VTK name | VTK point data attribute name | VTK point data attribute field/value used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Deploy Structure` | stub row only | stub only | none | none | none | none | none | none |
| `Decay` | existing non-new trees marked for aging in place or senescence, plus related rewilded opportunity voxels | computed | `treeDF_{year}.csv` | `isNewTree`, `action` | `isNewTree == False`; `action == AGE-IN-PLACE` or `SENESCENT` | `*_urban_features_with_indicators.vtk` | `scenario_rewilded` | `exoskeleton`, `footprint-depaved`, `node-rewilded`, `rewilded` |
| `Recruit` | stub row only | stub only | none | none | none | none | none | none |
| `Colonise` | candidate roof, facade, and rewilded-ground voxels | computed | none | none | none | `*_urban_features_with_indicators.vtk` | `scenario_outputs` | `brownRoof`, `greenRoof`, `livingFacade`, `footprint-depaved`, `node-rewilded`, `otherGround`, `rewilded` |
| `Release Control` | arboreal canopy voxels | computed | none | none | none | `*_urban_features_with_indicators.vtk` | `search_bioavailable` | `arboreal` |

`2.3-INFO-3. Community Response`
Script:
- `final/a_info_proposal_interventions.py`
Inputs:
- `indicator outcomes state [vtk]` - `data/revised/final/output/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features_with_indicators.vtk`
- `scenario treeDF [dataframe csv]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv`
Description of processing:
- Reads the same proposal state and counts which proposal space receives full support or partial support under the current code mapping; refused space is inferred as the remaining opportunity space not assigned to a support row.
Outputs:
- `proposal interventions [dataframe csv]` - `data/revised/final/output/csv/{site}_{voxel_size}_proposal_interventions.csv`
- `proposal qc [dataframe csv]` - `data/revised/final/output/csv/{site}_{voxel_size}_proposal_qc.csv`
Chain note:
- The manuscript intervention set is `Buffer Feature`, `Brace Feature`, `Rewild Ground`, `Adapt Utility Pole`, `Upgrade Feature`, `Enrich Envelope`, and `Roughen Envelope`.
- The current script still uses older support labels for some proposals, so the table below distinguishes manuscript language from the measured code labels.
- This step also does not read `nodeDF`; it measures support from `treeDF_{year}.csv` and the assessed VTK.

Community-response tracking:

| Proposal | Intervention family | Support level | Current code measurement label | Node DF name | Node DF column | Node DF field/value used | VTK name | VTK point data attribute name | VTK point data attribute field/value used | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Deploy Structure` | `Adapt utility pole` | full | not yet computed | none | none | none | none | none | none |  |
| `Deploy Structure` | `Upgrade feature` | full | not yet computed | none | none | none | none | none | none |  |
| `Decay` | `Buffer feature` | full | `Buffer` | `treeDF_{year}.csv` | `rewilded` | `node-rewilded`, `footprint-depaved` | `*_urban_features_with_indicators.vtk` | `scenario_rewilded` | `node-rewilded`, `footprint-depaved`, `rewilded` |  |
| `Decay` | `Brace feature` | partial | `Brace` | `treeDF_{year}.csv` | `rewilded` | `exoskeleton` | `*_urban_features_with_indicators.vtk` | `scenario_rewilded` | `exoskeleton` |  |
| `Recruit` | `Buffer feature` | full | not yet computed | none | none | none | none | none | none |  |
| `Recruit` | `Rewild ground` | full | not yet computed | none | none | none | none | none | none |  |
| `Colonise` | `Rewild ground` | full | `Connect (full)` | none | none | none | `*_urban_features_with_indicators.vtk` | `scenario_outputs` | `node-rewilded`, `rewilded` |  |
| `Colonise` | `Enrich envelope` | full | `Connect (full)` | none | none | none | `*_urban_features_with_indicators.vtk` | `scenario_outputs` | `greenRoof` |  |
| `Colonise` | `Rewild ground` | partial in current code | `Connect (partial)` | none | none | none | `*_urban_features_with_indicators.vtk` | `scenario_outputs` | `footprint-depaved` |  |
| `Colonise` | `Roughen envelope` | partial | `Connect (partial)` | none | none | none | `*_urban_features_with_indicators.vtk` | `scenario_outputs` | `brownRoof`, `livingFacade` |  |
| `Release Control` | `Buffer feature` | full | `Eliminate pruning` | none | none | none | `*_urban_features_with_indicators.vtk` | `search_bioavailable`, `forest_control` | `arboreal`; `reserve-tree`, `improved-tree` |  |
| `Release Control` | `Brace feature` | partial | `Reduce pruning` | none | none | none | `*_urban_features_with_indicators.vtk` | `search_bioavailable`, `forest_control` | `arboreal`; `park-tree` |  |

`2.3-INFO-4. Implementation`
Script:
- `final/a_info_gather_capabilities.py` - defines the implementation logic, counts support actions for each indicator outcome, and writes the per-site `action_counts.csv`.
- `final/a_info_output_capabilities.py` - reads the per-site `action_counts.csv` files and writes the combined all-sites action table.
Inputs:
- `urban features state [vtk]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk`
Description of processing:
- `a_info_gather_capabilities.py` reads the indicator-supporting voxels, counts which implementation conditions support each one, and writes the per-site support-action table.
- `a_info_output_capabilities.py` combines those per-site tables into a single all-sites summary; it does not define or recompute the logic.
Outputs:
- `a_info_gather_capabilities.py` outputs:
- `action counts [dataframe csv]` - `data/revised/final/output/csv/{site}_{voxel_size}_action_counts.csv`
- `a_info_output_capabilities.py` outputs:
- `all sites action counts [dataframe csv]` - `data/revised/final/output/csv/all_sites_{voxel_size}_action_counts.csv`
Chain note:
- This is the support-action layer used to describe implementation.
- The action-count logic is code-defined in `a_info_gather_capabilities.py` through `SUPPORT_ACTIONS`, `CONTROL_LEVELS`, `URBAN_ELEMENTS`, and `REWILDING_TYPES`.

Implementation logic types:

| Logic type | Source field | What it measures | Values written into `action_counts.csv` |
| --- | --- | --- | --- |
| `control_level` | `forest_control` | how much indicator-supporting canopy sits under each management level | `high`, `medium`, `low` |
| `urban_element` | `search_urban_elements` | how much indicator-supporting space sits on each urban surface or conversion type | `open space`, `green roof`, `brown roof`, `facade`, `roadway`, `busy roadway`, `existing conversion`, `other street potential`, `parking`, `none` |
| `rewilding_status` | `scenario_rewilded` | how much indicator-supporting tree space sits in each rewilding state | `footprint-depaved`, `exoskeleton`, `node-rewilded`, `none` |
| `artificial_structures_deployed` | `forest_size` | how much bird-supporting outcome is provided by artificial structures | `artificial` |
| `artificial` | `forest_precolonial` | how much indicator-supporting outcome is provided by installed non-precolonial elements | `installed` |

Capability-to-implementation mapping:

| Indicator outcome | Main implementation logic | Extra implementation counts |
| --- | --- | --- |
| `Bird.self.peeling` | `control_level` | `artificial_structures_deployed`, `artificial` |
| `Bird.others.perch` | `control_level` | `artificial_structures_deployed` |
| `Bird.generations.hollow` | `control_level` | `artificial_structures_deployed`, `artificial` |
| `Lizard.self.grass` | `urban_element` | none |
| `Lizard.self.dead` | `control_level` | none |
| `Lizard.self.epiphyte` | `control_level` | `artificial` |
| `Lizard.others.notpaved` | `urban_element` | none |
| `Lizard.generations.nurse-log` | `urban_element` | none |
| `Lizard.generations.fallen-tree` | `urban_element` | none |
| `Tree.self.senescent` | `rewilding_status` | none |
| `Tree.others.notpaved` | `urban_element` | none |
| `Tree.generations.grassland` | `urban_element` | none |

#### Graph Outputs

Section note:
- These scripts sit downstream of the assessment tables.
- `2.3-INFO-1` is the main upstream source, because the graph scripts read `indicator_counts.csv` rather than VTKs.

`2.3-GRAPH-1. Stream graphs`
Script:
- `final/a_info_graphs.py`
Inputs:
- `data/revised/final/output/csv/{site}_{voxel_size}_indicator_counts.csv`
Description of processing:
- Reads the indicator counts, smooths the scenario trajectories against baseline, and writes stacked stream graphs for the indicator outcomes over time.
Outputs:
- `data/revised/final/output/plots/stream_graph_*.html`
- `data/revised/final/output/plots/stream_graph_*.png`
Chain note:
- This is the active graph script still kept in `final/`.

`2.3-GRAPH-2. Performance bubbles`
Script:
- `final/SS/a_info_performance_bubbles.py`
Inputs:
- preferred `data/revised/final/output/csv/all_sites_{voxel_size}_indicator_counts.csv`
- fallback `data/revised/final/output/csv/{site}_{voxel_size}_indicator_counts.csv`
Description of processing:
- Reads the indicator counts, compares positive against trending through time, and writes bubble charts sized by positive performance and positioned by relative performance.
Outputs:
- `data/revised/final/output/plots/performance_bubbles_*.html`
- `data/revised/final/output/plots/performance_bubbles_*.png`
Chain note:
- Moved to `final/SS/` because it is not currently part of the core active workflow.

`2.3-GRAPH-3. Legacy capability plots`
Script:
- `final/SS/a_info_capability_plots.R`
Inputs:
- `*_capabilities_by_timestep.csv`
Description of processing:
- Reads an older wide capability table format and writes simple persona and capability line plots.
Outputs:
- `plots/*_all_capabilities.png`
- `plots/*_{persona}_capabilities.png`
Chain note:
- Moved to `final/SS/` because it depends on an older input pattern that is not part of the current active chain.

### 2.4. Envelope Branch

`2.4-ENVELOPE-1. Build the envelope shell from the yearly scenario VTK`
Script:
- `final/_blender/b_generate_rewilded_envelopes.py`
Inputs:
- `scenario state [vtk]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`
- `masked site voxels [vtk]` - `data/revised/{site}-siteVoxels-masked.vtk`
- `coloured road voxels [vtk]` - `data/revised/{site}-roadVoxels-coloured.vtk`
Description of processing:
- Selects bio-envelope voxels from the yearly scenario VTK, borrows normals from the cleaned site and road reference clouds, builds the envelope shell, and exports Blender-ready envelope geometry.
Outputs:
- `envelope shell points [ply]` - `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.ply`
- `envelope surface [vtk]` - `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.vtk`
Cross-reference:
- This branch diverges after `2.2-SCENARIO-3`.
- It reuses the cleaned world VTKs from `2.1-WORLD-2`; it does not rerun `2.1-WORLD-1`, `2.1-WORLD-2`, or `2.2-SITE-1`.
Chain note:
- The script still reads `data/revised/{site}-siteVoxels-masked.vtk` and `data/revised/{site}-roadVoxels-coloured.vtk`, while `2.1-WORLD-2` now writes those files under `data/revised/final/`.

`2.4-GROUND-1. Build the companion rewilded ground shell`
Script:
- `final/_blender/b_generate_rewilded_ground.py`
Inputs:
- `scenario state [vtk]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`
Description of processing:
- Selects rewilded ground voxels from the same yearly scenario VTK, extracts a shell surface, and exports the companion ground geometry.
Outputs:
- `ground shell points [ply]` - `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_ground_scenarioYR{year}.ply`
- `ground surface [vtk]` - `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_ground_scenarioYR{year}.vtk`
Cross-reference:
- This branch also diverges after `2.2-SCENARIO-3`.

### 2.5. Baseline Branch

`2.5-BASELINE-1. Generate or reuse the site baseline`
Script:
- `final/a_scenario_manager.py`
- `final/a_scenario_get_baselines.py`
Inputs:
- `baseline densities [csv]` - `data/csvs/tree-baseline-density.csv`
- `subset dataset [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_subsetForScenarios.nc`
- `road voxels [vtk]` - `data/revised/final/{site}-roadVoxels-coloured.vtk`
- `combined voxel templates [pickle dataframe]` - `data/revised/trees/{voxel_size}_combined_voxel_templateDF.pkl`
Description of processing:
- Either reuses an existing combined baseline VTK or builds a new baseline cohort from the scenario-ready site, expands resources onto those trees, and packages the baseline as separate and combined outputs.
Outputs:
- `baseline trees [dataframe csv]` - `data/revised/final/baselines/{site}_baseline_trees.csv`
- `baseline resources [vtk]` - `data/revised/final/baselines/{site}_baseline_resources_{voxel_size}.vtk`
- `baseline terrain [vtk]` - `data/revised/final/baselines/{site}_baseline_terrain_{voxel_size}.vtk`
- `baseline combined [vtk]` - `data/revised/final/baselines/{site}_baseline_combined_{voxel_size}.vtk`
Optional reuse:
- If `a_scenario_manager.py` runs `process_baseline(..., check=True)` and the combined baseline VTK already exists, it returns that file without rerunning `2.5-BASELINE-1.1` to `2.5-BASELINE-1.3`.

`2.5-BASELINE-1.1. Build the baseline cohort and assign positions`
Script:
- `final/a_scenario_get_baselines.py`
Inputs:
- `baseline densities [csv]` - `data/csvs/tree-baseline-density.csv`
- `subset dataset [xarray netcdf]` - `data/revised/final/{site}/{site}_{voxel_size}_subsetForScenarios.nc`
- `road voxels [vtk]` - `data/revised/final/{site}-roadVoxels-coloured.vtk`
Description of processing:
- Converts the density table into a baseline tree cohort, derives size and senescence states, finds ground voxels, and places the baseline trees into the site.
Outputs:
- `baseline tree cohort [dataframe]` - in-memory dataset for baseline resource expansion

`2.5-BASELINE-1.2. Expand resources onto the baseline cohort`
Script:
- `final/a_scenario_get_baselines.py` via `a_resource_distributor_dataframes.process_all_trees`
Inputs:
- `baseline tree cohort [dataframe]` - in-memory result from `2.5-BASELINE-1.1`
- `combined voxel templates [pickle dataframe]` - `data/revised/trees/{voxel_size}_combined_voxel_templateDF.pkl`
Description of processing:
- Queries the combined voxel template library, expands resource structures onto the placed baseline trees, and rotates the resulting structures into site position.
Outputs:
- `baseline resourceDF [dataframe]` - in-memory dataset for VTK export
Chain note:
- Older scripts still point at `data/revised/trees/{voxel_size}_voxel_tree_dict.pkl`, but the active baseline path goes through `{voxel_size}_combined_voxel_templateDF.pkl`.

`2.5-BASELINE-1.3. Export the baseline VTK products`
Script:
- `final/a_scenario_get_baselines.py`
Inputs:
- `baseline tree cohort [dataframe]` - in-memory result from `2.5-BASELINE-1.1`
- `baseline resourceDF [dataframe]` - in-memory result from `2.5-BASELINE-1.2`
Description of processing:
- Builds terrain and combined polydata from the placed baseline trees and resource structures, then writes the final baseline exports.
Outputs:
- `baseline trees [dataframe csv]` - `data/revised/final/baselines/{site}_baseline_trees.csv`
- `baseline resources [vtk]` - `data/revised/final/baselines/{site}_baseline_resources_{voxel_size}.vtk`
- `baseline terrain [vtk]` - `data/revised/final/baselines/{site}_baseline_terrain_{voxel_size}.vtk`
- `baseline combined [vtk]` - `data/revised/final/baselines/{site}_baseline_combined_{voxel_size}.vtk`
Chain note:
- If `a_scenario_manager.py` runs `process_baseline(..., check=True)` and the combined baseline VTK already exists, it reuses that file instead of running `2.5-BASELINE-1.1` to `2.5-BASELINE-1.3`.

## 3. Tree / Log / Pole PLY Libraries

### Resource tables

Step 1. `csv converters/leroux.py`
Inputs:
- `data/csvs/lerouxdata-long.csv`
Description of processing:
- Maps the long-form Le Roux survey table onto the project resource labels and writes the normalized survey table used downstream.
Outputs:
- `data/csvs/lerouxdata-update.csv`
Chain note:
- No direct writer found for `data/csvs/lerouxdata-long.csv`; the chain breaks at this external source table.

### Legacy treeBake template branch

Step 2. `final/tree_processing/combined_generateResourceDict.py`
Inputs:
- `data/csvs/lerouxdata-update.csv`
Description of processing:
- Computes the newer per-condition resource matrix from the normalized Le Roux table and writes the CSV resource dictionary for the mixed Blender-era branch.
Outputs:
- `data/revised/trees/resource_dicDF.csv`

Step 3. `final/tree_processing/unchanged/generateResourceDict.py`
Inputs:
- `data/csvs/lerouxdata-update.csv`
Description of processing:
- Builds the older tuple-keyed resource dictionary from the same normalized Le Roux table and writes the JSON resource table for the legacy treeBake branch.
Outputs:
- `data/treeOutputs/tree_resources.json`

Step 4. `modules/treeBake_eucs`
Inputs:
- `data/treeInputs/trunks/orig/*.csv`
Description of processing:
- Voxelizes raw eucalypt cylinder CSVs, carries source attributes onto voxel centroids, and writes the initial eucalypt template library.
Outputs:
- `data/treeInputs/trunks/initial_templates/*.csv`
Chain note:
- No in-repo script directly writes `data/treeInputs/trunks/orig/*.csv`; `modules/treeBake_cylinders.py` documents the expected schema but reads that folder rather than producing it.

Step 5. `modules/treeBake_elms.py`
Inputs:
- `data/treeInputs/trunks-elms/orig/csvs-orig/branch/*.csv`
- `data/treeInputs/trunks-elms/orig/csvs-orig/cylinder/*.csv`
- `data/treeInputs/trunks-elms/orig/trunk_point_clouds/*.txt`
Description of processing:
- Clusters raw elm branch graphs, aligns them to cylinder tables and trunk point clouds, and writes the initial elm point-template CSVs plus clustered intermediates.
Outputs:
- `data/treeInputs/trunks-elms/orig/csvs-clustered/branch/*.csv`
- `data/treeInputs/trunks-elms/orig/csvs-clustered/cylinder/*.csv`
- `data/treeInputs/trunks-elms/orig/initial_templates/*.csv`
- `data/treeInputs/trunks-elms/initial templates/*.csv`
Chain note:
- No in-repo script directly writes `data/treeInputs/trunks-elms/orig/csvs-orig/*.csv` or `data/treeInputs/trunks-elms/orig/trunk_point_clouds/*.txt`; the chain enters the repo at those raw exports.

Step 6. `modules/treeBake_combined_refactored.py`
Inputs:
- `data/treeInputs/trunks/initial_templates/*.csv`
- `data/treeInputs/trunks-elms/initial templates/*.csv`
- `data/treeInputs/leaves/*.csv`
Description of processing:
- Normalizes the eucalypt and elm template CSVs into a common coordinate and topology schema, appends canopy clusters from the leaf CSVs, and writes the processed trunk libraries that the legacy resource assigner consumes.
Outputs:
- `data/treeInputs/trunks/processed/*.csv`
- `data/treeInputs/trunks-elms/processed/*.csv`
Chain note:
- No in-repo script directly writes `data/treeInputs/leaves/*.csv`; the legacy canopy loaders treat that folder as a precomputed input.

Step 7. `modules/treeBake_assignResources`
Inputs:
- `data/treeInputs/trunks/processed/*.csv`
- `data/treeInputs/trunks-elms/processed/*.csv`
- `data/treeOutputs/tree_resources.json`
Description of processing:
- Assigns canopy, branch, and ground resources across the processed trunk libraries using the legacy JSON resource dictionary and writes the legacy template dictionary.
Outputs:
- `data/treeOutputs/tree_templates.pkl`

Step 8. `modules/treeBake_treeAging.py`
Inputs:
- `data/treeOutputs/tree_templates.pkl`
- `data/treeOutputs/tree_resources.json`
Description of processing:
- Filters the legacy template dictionary to selected large improved trees, derives senescing, snag, fallen, and propped variants, recalculates resources, and writes the aged-tree dictionary.
Outputs:
- `data/treeOutputs/fallen_trees_dict.pkl`

Step 9. `modules/treeBake_recreateLogs.py`
Inputs:
- `data/treeOutputs/fallen_trees_dict.pkl`
- `data/treeOutputs/tree_resources.json`
- `data/treeOutputs/tree_templates.pkl`
Description of processing:
- Recenters fallen-log exemplars, reinserts fallen-log resources into the living templates, and writes both the adjusted template dictionary and the log library.
Outputs:
- `data/treeOutputs/adjusted_tree_templates.pkl`
- `data/treeOutputs/logLibrary.pkl`

Step 10. `final/tree_processing/f_temp_adjustTreeDict`
Inputs:
- `data/treeOutputs/adjusted_tree_templates.pkl`
- `data/treeOutputs/fallen_trees_dict.pkl`
Description of processing:
- Remaps the legacy adjusted and aged template dictionaries from the old tuple keys into the four-field revised schema and writes the revised eucalypt dictionary.
Outputs:
- `data/revised/revised_tree_dict.pkl`

Step 11. `final/tree_processing/euc_manager.py`
Inputs:
- `data/revised/revised_tree_dict.pkl`
- `data/revised/trees/resource_dicDF.csv`
Description of processing:
- Drops leaf litter, renames coordinate columns, reapplies the newer CSV resource logic, and writes the mixed-branch eucalypt dictionary.
Outputs:
- `data/revised/trees/updated_tree_dict.pkl`
Chain note:
- `final/tree_processing/combined_tree_manager.py` still comments that a missing `euc_convertPickle.py` writes this file, but the active writer in the repo is `final/tree_processing/euc_manager.py`.

### Revised elm template branch

Step 12. `final/tree_processing/adTree_InitialPlyToDF.py`
Inputs:
- `data/revised/lidar scans/elm/adtree/*_skeleton.ply`
Description of processing:
- Converts raw elm skeleton PLYs into branch-segment tables and writes the initial elm QSM CSVs.
Outputs:
- `data/revised/lidar scans/elm/adtree/QSMs/*_treeDF.csv`
Chain note:
- No in-repo script directly writes `data/revised/lidar scans/elm/adtree/*_skeleton.ply`; the chain enters the repo at those imported skeleton scans.

Step 13. `final/tree_processing/adTree_ClusterInitialDF.py`
Inputs:
- `data/revised/lidar scans/elm/adtree/QSMs/*_treeDF.csv`
Description of processing:
- Filters the elm QSM CSVs, reclusters branch segments into connected graph units, and writes clustered QSM tables plus the first cluster graphs.
Outputs:
- `data/revised/lidar scans/elm/adtree/processedQSMs/*_clusteredQSM.csv`
- `data/revised/lidar scans/elm/adtree/processedQSMs/*_clusterGraph.graphml`

Step 14. `final/tree_processing/adTree_voxelise.py`
Inputs:
- `data/revised/lidar scans/elm/adtree/processedQSMs/*_clusteredQSM.csv`
- `data/revised/lidar scans/elm/adtree/*_branches.obj`
Description of processing:
- Voxelizes the clustered elm geometry against the branch OBJ meshes, transfers nearby QSM attributes onto voxel centroids, and writes the elm voxel tables.
Outputs:
- `data/revised/lidar scans/elm/adtree/elmVoxelDFs/*_voxelDF.csv`
Chain note:
- No in-repo script directly writes `data/revised/lidar scans/elm/adtree/*_branches.obj`; the OBJ branch enters the repo as an external geometry export.

Step 15. `final/tree_processing/adTree_AssignLargerClusters.py`
Inputs:
- `data/revised/lidar scans/elm/adtree/processedQSMs/*_clusterGraph.graphml`
- `data/revised/lidar scans/elm/adtree/elmVoxelDFs/*_voxelDF.csv`
Description of processing:
- Runs community clustering over the first elm branch graphs and voxel tables and writes the processed graph files used for template generation.
Outputs:
- `data/revised/lidar scans/elm/adtree/processedGraph/*_processedGraph.graphml`
- `data/revised/lidar scans/elm/adtree/processedGraph/*_communityGraph.graphml`

Step 16. `final/tree_processing/adTree_AssignResources.py`
Inputs:
- `data/revised/trees/resource_dicDF.csv`
- `data/revised/lidar scans/elm/adtree/elmVoxelDFs/*_voxelDF.csv`
- `data/revised/lidar scans/elm/adtree/processedGraph/*_processedGraph.graphml`
- `data/treeOutputs/logLibrary.pkl`
Description of processing:
- Assigns resource distributions across the elm voxel templates using the newer CSV resource table and the processed graphs, then writes the elm template dictionary.
Outputs:
- `data/revised/trees/elm_tree_dict.pkl`

### Combined tree mesh library

Step 17. `final/tree_processing/combined_tree_manager.py`
Inputs:
- `data/revised/trees/updated_tree_dict.pkl`
- `data/revised/trees/elm_tree_dict.pkl`
- `data/revised/trees/resource_dicDF.csv`
- `data/revised/lidar scans/elm/adtree/processedGraph/*_processedGraph.graphml`
Description of processing:
- Merges the eucalypt and elm template families, regenerates and edits snag variants, and writes the combined template DataFrames used for both voxelisation and mesh regeneration.
Outputs:
- `data/revised/trees/combined_templateDF.pkl`
- `data/revised/trees/edited_combined_templateDF.pkl`
- `data/revised/trees/just_edits_templateDF.pkl`
- `data/revised/trees/regenerated_snags.pkl`
- `data/revised/trees/combined_voxelSize_{voxel_size}_templateDF.pkl`
Chain note:
- `combined_tree_manager.py` writes `combined_voxelSize_{voxel_size}_templateDF.pkl`, but `final/a_resource_distributor_dataframes.py` reads `{voxel_size}_combined_voxel_templateDF.pkl`; the filename chain breaks unless `combined_voxelise_dfs.py` is run separately.

Step 18. `final/tree_processing/combined_voxelise_dfs.py`
Inputs:
- `data/revised/trees/edited_combined_templateDF.pkl`
Description of processing:
- Voxelizes the edited combined template DataFrame at a chosen resolution, collapses point resources into per-voxel stats, and writes the baseline-facing voxel template tables.
Outputs:
- `data/revised/trees/{voxel_size}_combined_voxel_templateDF.pkl`
- `data/revised/trees/{voxel_size}_combined_voxel_adjustment_summary.csv`
- `data/revised/trees/{voxel_size}_combined_voxel_all_resource_stats.csv`
Chain note:
- `final/a_resource_distributor_dataframes.py` currently reads this standalone filename pattern.

Step 19. `final/tree_processing/combine_resource_treeMeshGenerator.py`
Inputs:
- `data/revised/trees/edited_combined_templateDF.pkl` or `data/revised/trees/just_edits_templateDF.pkl`
Description of processing:
- Extracts isosurfaces from the combined template points, transfers template attributes onto mesh vertices, and writes the tree-mesh VTK library.
Outputs:
- `data/revised/final/treeMeshes/*.vtk`

Step 20. `final/_blender/a_vtk_to_ply.py`
Inputs:
- `data/revised/final/treeMeshes/*.vtk`
- optionally filtered by `data/revised/trees/just_edits_templateDF.pkl`
Description of processing:
- Converts the tree VTK meshes into Blender-facing PLYs and preserves the numeric point attributes.
Outputs:
- `data/revised/final/treeMeshesPly/*.ply`

### Log mesh library

Step 1. `modules/treeBake_recreateLogs.py`
Inputs:
- `data/treeOutputs/fallen_trees_dict.pkl`
- `data/treeOutputs/tree_resources.json`
- `data/treeOutputs/tree_templates.pkl`
Description of processing:
- Recenters fallen-log exemplars from the legacy aged-tree and template dictionaries and writes the log library that the mesh exporter consumes.
Outputs:
- `data/treeOutputs/logLibrary.pkl`

Step 2. `final/tree_processing/a_log_mesh_generator.py`
Inputs:
- `data/treeOutputs/logLibrary.pkl`
Description of processing:
- Extracts filled isosurface meshes from the centered fallen-log point groups and writes the VTK log library.
Outputs:
- `data/revised/final/logMeshes/*.vtk`

Step 3. `final/_blender/a_vtk_to_ply.py`
Inputs:
- `data/revised/final/logMeshes/*.vtk`
Description of processing:
- Converts each log VTK into a PLY, adds the Blender-facing fallen-log resource flags, and writes the PLY log library.
Outputs:
- `data/revised/final/logMeshesPly/*.ply`

### Pole / artificial support

Step 1. `final/tree_processing/b_generate_utility_pole_and_artificial_tree.py`
Inputs:
- `data/revised/utlity poles/utility_pole.ply`
- `data/revised/final/treeMeshes/precolonial.False_size.snag_control.improved-tree_id.10.vtk`
Description of processing:
- Voxelizes the raw utility-pole point cloud, clips a donor snag mesh to the pole bounds, and writes the artificial support assets used in Blender.
Outputs:
- `data/revised/utlity poles/utility_pole_voxelised_0.1.ply`
- `data/revised/final/treeMeshes/artificial_precolonial.False_size.snag_control.improved-tree_id.10.vtk`
- `data/revised/final/treeMeshesPly/artificial_precolonial.False_size.snag_control.improved-tree_id.10.ply`
Chain note:
- No direct writer found for `data/revised/utlity poles/utility_pole.ply`; provenance inferred from comments, naming, and adjacent scripts.

## 4. Base World PLYs

`4-WORLDPLY-1. Build raw world voxel datasets`
Script:
- `final/f_manager.py`
Inputs:
- `data/revised/csv/site_locations.csv`
- `data/revised/experimental/DevelopmentActivityModel-trimmed-metric.glb`
- `data/revised/obj/_converted/*.vtk`
- `data/revised/obj/_converted/converted_mesh_centroids.csv`
- `data/revised/shapefiles/contours/EL_CONTOUR_1TO5M.shp`
- City of Melbourne road-segment records
Description of processing:
- Crops the terrain, building model, photomesh context, and road geometry to the site and writes the raw world voxel datasets.
Outputs:
- `data/revised/{site}-siteVoxels.vtk`
- `data/revised/{site}-roadVoxels.vtk`

`4-WORLDPLY-2. Clean the world voxel datasets`
Script:
- `final/f_further_processing.py`
Inputs:
- `data/revised/{site}-siteVoxels.vtk`
- `data/revised/{site}-roadVoxels.vtk`
- `data/revised/{site}-treeVoxels.vtk`
Description of processing:
- Filters the site voxels into the building world, adds building and roof metadata, transfers colour and spatial context to the road cloud, and recenters the scene.
Outputs:
- `data/revised/final/{site}-siteVoxels-masked.vtk`
- `data/revised/final/{site}-roadVoxels-coloured.vtk`

`4-WORLDPLY-3. Export Blender-facing base world PLYs`
Script:
- `final/_blender/b_extract_scene.py`
Inputs:
- `data/revised/{site}-siteVoxels-masked.vtk`
- `data/revised/{site}-roadVoxels-coloured.vtk`
Description of processing:
- Exports the masked site voxels directly as the building PLY and densifies coarse road voxels before exporting the high-resolution road PLY.
Outputs:
- `data/revised/final/{site}/{site}_buildings.ply`
- `data/revised/final/{site}/{site}_highResRoad.ply`

Chain note:
- `final/f_further_processing.py` writes the masked and coloured world VTKs under `data/revised/final/`, but `final/_blender/b_extract_scene.py` still reads them from `data/revised/`.

## 5. Envelope PLYs

Section note:
- The detailed branch is documented once, in `2.4. Envelope Branch`.
- Use the cross-references below instead of repeating the shared upstream steps.

Cross-reference:
- `2.1-WORLD-2` provides the cleaned world VTKs that the envelope shell uses as reference geometry.
- `2.2-SCENARIO-3` provides the yearly scenario VTK that both envelope exports read.
- `2.4-ENVELOPE-1` exports the bio-envelope shell.
- `2.4-GROUND-1` exports the companion rewilded ground shell.
