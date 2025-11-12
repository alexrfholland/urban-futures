## Habitat volume appendix

This appendix explains inputs, attributes, and early transformations in the habitat volume. We prioritise legibility over exhaustive replication. Each step names inputs, types, operations, and outputs. We include key attributes that differentiate voxels and support later analysis, including under‑canopy voxels and utility poles.

### Inputs as attributes

Voxel attributes act as an extensible object model. Attributes carry identifiers, indicators, classes, and flags. Numbers track measures and thresholds. Strings classify features. Binary values mark membership. The model accepts new attributes without schema changes. Analysts can group voxels by any subset to form features at multiple scales.

### Preflight steps

Step | Input | Type | Operations | Output
--- | --- | --- | --- | ---
Site frame | Site name | Bounds | `f_SiteCoordinates.get_site_coordinates` to get centre and dims; update with mesh bounds | Centre, eastingsDim, northingsDim
Meshes | Converted VTK list | Geometry | `f_photoMesh.get_meshes_in_bounds` to collect meshes within bounds | Photo meshes in site frame
Terrain | Contour shapefile | Elevation grid | `f_GeospatialModule.handle_contours` to interpolate z on a 1 m grid and clip to bounds | Terrain mesh with Elevation
Buildings | GLB scene | Mesh + IDs | `f_GetSiteMeshBuildings.process_buildings` to select, align to terrain, combine, label `building_ID` | Building mesh with cell `building_ID`
Segmentation | LAS tiles, building mesh, meshes | Points + attributes | `f_segmentation_manager.segmentFunction` to combine meshes, add LAS attributes, transfer building IDs, normals, and distances | Site voxels with `LAS_*`, `building_*`
Roads | API dataset | Polygons | `f_GeospatialModule.handle_road_segments` to trim to bounds, attach attributes | Road GeoDataFrame for voxel assignment
Canopies | API dataset | Polygons | `f_GeospatialModule.handle_tree_canopies` to trim to bounds | Under‑canopy mask for voxel assignment
Urban forest | API dataset | Points | `f_GeospatialModule.handle_urban_forest` to trim to bounds, keep species and dimensions | Urban forest table for cluster matching
Poles | Shapefiles | Points | `f_GeospatialModule.handle_poles` to fetch pylons and streetlights in bounds | Utility pole tables for supports
Green roofs | Shapefiles | Polygons | `f_GeospatialModule.handle_green_roofs` to map ratings to integers | Roof suitability tables
Parking, laneways, open space | Shapefiles | Polygons | `f_GeospatialModule.handle_parking`, `handle_other_road_info` | Supplementary deployable targets

Key attributes used at this stage include

- Identifiers: `building_ID`, `urbanForest_id`, `roadInfo_id`, `roofID`
- Geometry and position: `voxel_I`, `voxel_J`, `voxel_K`, `centroid_x`, `centroid_y`, `centroid_z`
- Building relations: `building_normalX`, `building_normalY`, `building_normalZ`, `building_distance`
- Road classes: `roadInfo_surface_type`, `roadInfo_classification` where available
- Canopy flags: `underCanopy` binary mask derived from canopies
- Roof suitability: `site_greenRoof_ratingInt`, `site_brownRoof_ratingInt`
- Pole classes: `pole_type` for pylons and streetlights

Under‑canopy voxels arise by rasterising canopy polygons to the site grid and assigning a boolean mask to grid points that fall under canopies. These voxels seed later aggregation beneath tree footprints for resistance and connectivity.

### Scenario setup to resistance

- Initialise dataset and metadata with `a_scenario_initialiseDS.initialize_dataset`
- Load node tables via `a_scenario_initialiseDS.load_node_dataframes`
- Align and clean with `a_scenario_initialiseDS.PreprocessData` and `further_xarray_processing`
- Produce per‑year states with `a_scenario_runscenario.run_scenario` or load cached
- Export per‑year VTK via `a_scenario_generateVTKs.generate_vtk`
- Assign resistance 0–100 per voxel. Aggregate under‑canopy footprints and ground‑building patches for threshold comparisons in pathways

### Roof weights and transferable logs

Assumptions

- Voxel size from dataset attribute defines area per voxel
- Green roof dead load = 300 kg per m²; brown roof dead load = 150 kg per m²
- Green roof log allowance = 50 kg per m²; brown roof log allowance = 100 kg per m²
- Ignore roofs with fewer than 10 voxels to avoid trivial allocations

Method

- Classify roof voxels as none, green roof, brown roof from `site_greenRoof_ratingInt` and `site_brownRoof_ratingInt`
- Group roof voxels into `envelope_roofID` by `site_building_ID` and `envelope_roofType`
- Compute per‑group `envelope_Roof load` and `envelope_Log load` as voxel_count × voxel_area × load factors
- Allocate logs by priority large then medium then small while capacity remains. Record `envelope_logNo`, `envelope_logMass`, `envelope_logSize`, and optional `logModel`
- Write group attributes to all voxels in a group. Write log attributes to selected voxels

Outputs

- Roof group table with IDs, counts, and loads
- Log placement table with IDs, sizes, and masses
- Updated dataset with `envelope_*` fields for analysis and visualisation

### Patch aggregation for connectivity

Model type

- Local expansion with energy‑based growth and resistance‑dependent branching
- First‑shell and second‑shell neighbourhoods precomputed on voxel indices

Parameters

- Energy per start node from size and type, e.g. tree large 2500, medium 2000, small 1000; log large 750, medium 500, small 250; pole medium 125
- Resistance factor 50, resistance threshold 50, high resistance cutoff 80
- Termination chance 0.2 by default and 0.8 in high resistance; fast split chance 0.65
- Max turns set high to allow convergence across reachable voxels

Procedure

- Seed growth from voxels with valid `analysis_nodeID`
- At each step, deduct energy by local resistance scaled by factor. In low resistance, allow fast branching. In high resistance, increase termination
- Expand to first‑shell neighbours with optional second‑shell neighbours by chance
- Track origin node per voxel in `sim_Nodes` and inclusion order in `sim_Turns`
- Cull small connected components below a size threshold to remove noise
- Summarise voxel counts and convert to areas with voxel size. Attach `sim_NodesVoxels` and `sim_NodesArea` to `treeDF`, `poleDF`, `logDF`

Outputs

- Updated dataset with `sim_startingNodes`, `sim_Nodes`, `sim_Turns`
- Area summaries per node for connectivity and support siting




