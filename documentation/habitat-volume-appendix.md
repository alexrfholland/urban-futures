## Habitat volume appendix

This appendix explains how the voxel grid stores attributes, how inputs differentiate the grid, and how the pipeline builds site representations for subsequent analysis. The focus stays on clarity over full reproducibility. Examples refer to scripts in `final/` with consistent terminology across files.

### Inputs

Voxel attributes act as properties in an object model. Attributes use different data types. Integers label identifiers or counts. Floats capture indicators or measures. Binary values act as flags. Strings classify categories. The table lists representative attributes rather than an exhaustive dictionary.

| Attribute | Type | Purpose | Example values |
| --- | --- | --- | --- |
| `voxel_I`, `voxel_J`, `voxel_K` | int | Index each voxel in the grid | 0, 1, 2 |
| `centroid_x`, `centroid_y`, `centroid_z` | float | Record voxel centre in metres | 261234.5 |
| `site_building_ID` | int | Link voxels to building segments | 42 |
| `site_road_ID` | int | Link voxels to road segments | 7 |
| `urban_tree_ID` | int | Link voxels to urban forest records | 3159 |
| `analysis_nodeID` | int | Identify starting nodes for patch growth | 1001 |
| `analysis_nodeSize` | str | Classify node size for growth energy | small, medium, large |
| `analysis_nodeType` | str | Classify node type for growth energy | tree, log, pole, unassigned |
| `analysis_combined_resistance` | float | Score local effort needed to change | 0 to 100 |
| `sim_startingNodes` | int | Flag starting nodes for patch growth | 0, 1 |
| `sim_Nodes` | int | Track patch membership by origin nodeID | 1001 |
| `sim_Turns` | int | Track order of voxel inclusion into a patch | 0, 1, 2 |
| `envelope_roofType` | str | Classify roofs that can host supports | none, green roof, brown roof |
| `envelope_roofID` | int | Group roof voxels into structural sets | 3 |
| `envelope_Roof load` | float | Estimate dead load capacity in kg | 12500.0 |
| `envelope_Log load` | float | Allocate biomass budget for logs in kg | 2400.0 |
| `envelope_logNo` | int | Index logs placed on roofs | 18 |
| `envelope_logMass` | float | Record biomass of a placed log in kg | 85.0 |
| `envelope_logSize` | str | Classify placed log size | small, medium, large |

Attributes differentiate the grid as inputs arrive. New attributes can extend the model without schema changes. Downstream steps can query any subset to form features at multiple scales.

### Preflight analysis and groupings

The preflight pipeline prepares site representations in `final/f_manager.py`. Steps run in sequence with consistent bounds to maintain alignment across datasets.

- Obtain site coordinates
  - Read site centre from `f_SiteCoordinates`
  - Print the chosen centre with extents for traceability

- Collect photo meshes
  - Load meshes within bounds via `get_meshes_in_bounds`
  - Update centre and extents with the returned mesh bounds

- Generate terrain mesh
  - Build terrain from contours via `gm.handle_contours`
  - Store a triangulated surface with point elevation for later sampling

- Parse urban forest
  - Read tree records via `gm.handle_urban_forest`
  - Convert to a structured table with locations via `f_resource_urbanForestParser`

- Prepare road network voxels
  - Read road segments via `gm.handle_road_segments`
  - Voxelise roads with `rGeoVectorGetter.getRoadVoxels` at set resolution

- Process buildings
  - Filter building meshes to site bounds via `f_GetSiteMeshBuildings.process_buildings`
  - Produce initial voxels for building envelopes with a terrain reference

- Segment site voxels
  - Refine site voxels with `f_segmentation_manager.segmentFunction`
  - Save a segmented set that aligns with terrain, buildings, and roads

This stage sets a coherent spatial frame. The grid now aligns terrain, buildings, and roads. Urban forest tables remain external at this point. A later stage can distribute resources into voxels where appropriate.

### Scenario setup up to resistance parameterisation

The scenario manager in `final/a_scenario_manager.py` drives site processing. The steps below cover setup up to the resistance field. Later sections summarise roof loads and patch aggregation due to their close links to early setup.

- Initialise site dataset
  - Create an `xarray` dataset for the site via `a_scenario_initialiseDS.initialize_dataset`
  - Attach attributes such as bounds and voxel size for consistent use

- Load node tables
  - Read initial `treeDF`, `poleDF`, `logDF` via `a_scenario_initialiseDS.load_node_dataframes`
  - Maintain links to the grid through shared identifiers and coordinates

- Preprocess data
  - Run cleaning and alignment via `a_scenario_initialiseDS.PreprocessData`
  - Compute site-level helpers via `a_scenario_initialiseDS.further_xarray_processing`

- Branch to scenario paths
  - Either run `a_scenario_runscenario.run_scenario` for a given year or load cached tables
  - Produce per-year tables for trees, logs, and poles with consistent schemas

- Generate VTKs
  - Convert per-year states to VTK via `a_scenario_generateVTKs.generate_vtk`
  - Keep one file per site, scenario, and year for downstream counting and review

- Assign resistance values
  - Score each voxel with effort needed to change on a 0 to 100 scale
  - Aggregate values beneath canopies and across patches for later threshold tests

### Roof weights and transferable logs

`final/a_logDistributor.py` estimates roof capacity and allocates potential log placements. The method keeps a simple structure so that readers can follow the logic.

- Identify roof candidates
  - Derive `envelope_roofType` from green roof and brown roof ratings
  - Form roof groups with `envelope_roofID` from building identifiers and roof type

- Estimate load budgets
  - Compute voxel area from the dataset attribute for voxel size
  - Derive `envelope_Roof load` for dead loads and `envelope_Log load` for log biomass

- Allocate logs
  - Prioritise larger logs while capacity remains within each roof group
  - Record placements with `envelope_logNo`, `envelope_logMass`, and `envelope_logSize`

- Update the dataset
  - Write roof group values to all voxels in each roof group
  - Write log placements to individual voxels selected from each roof group

This step prepares a transparent list of supports on roofs. Downstream analyses can filter by type, capacity, or size without duplicate logic.

### Patch aggregation for connectivity

`final/a_rewilding.py` groups voxels into patches that expand from starting nodes under simple local rules. The grouping provides a view of potential connectivity that respects local resistance.

- Build grid neighbour maps
  - Precompute first shell and second shell neighbours for each voxel index
  - Keep fast lookups in arrays to support repeated growth steps

- Simulate patch growth
  - Seed from `analysis_nodeID` values present in the grid
  - Track origin node and turn of inclusion with `sim_Nodes` and `sim_Turns`

- Cull small clusters
  - Label connected components in a binary mask of grown voxels
  - Remove small clusters below a chosen size threshold

- Summarise areas
  - Map node membership to counts and areas using voxel size
  - Attach per-node areas to `treeDF`, `poleDF`, and `logDF` for later use

The result yields contiguous patches that suggest corridors and hubs. Analysts can compare patch structure with resistance to locate places that accept change with modest effort.






