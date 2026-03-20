# VTK Point Data Attribute Reference

This is a living reference for every `point_data` attribute that appears in the scenario VTK files.
It follows the processing order defined by `a_scenario_manager.py`:

1. **Initialise** (`a_scenario_initialiseDS.py`) — loads the xarray dataset and node dataframes
2. **Run scenario** (`a_scenario_runscenario.py`) — ages trees, determines actions, creates rewilding masks
3. **Generate VTK** (`a_scenario_generateVTKs.py`) — integrates resources, creates bio-envelopes, saves VTK
4. **Urban elements** (`a_scenario_urban_elements_count.py`) — transfers site features, creates search layers
5. **Capabilities** (`a_info_gather_capabilities.py`) — creates indicator layers from queries

---

## 1. Habitat Feature Attributes (Trees, Logs, Poles)

These describe the volumetric structure of urban trees and other habitat features.
Created in `a_voxeliser.py` → `integrate_resources_into_xarray()`, called from `a_scenario_generateVTKs.py`.

### 1.1 `resource_*` — Ecological resource quantities

Floating-point counts per voxel, summed from all overlapping habitat feature voxelisations within that voxel.

| Attribute | Description |
|---|---|
| `resource_hollow` | Tree hollows (nesting cavities) |
| `resource_epiphyte` | Epiphytic plants (ferns, orchids, mosses) |
| `resource_dead branch` | Dead branches still attached to tree |
| `resource_perch branch` | Branches suitable for perching |
| `resource_peeling bark` | Bark in a peeling/shedding state |
| `resource_fallen log` | Fallen log material on or near ground |
| `resource_other` | Other woody material (misc. structure) |
| `resource_leaf litter` | Leaf litter accumulation |

**Source**: `aa_tree_helper_functions.resource_names()` defines the canonical list.
**Pipeline**: Raw point-cloud templates are voxelised per habitat feature, then aggregated (summed) per voxel.

### 1.2 `stat_*` — Binary resource presence per voxel

Boolean (0/1) versions of `resource_*`, indicating *presence* rather than quantity.

| Attribute | Derived from |
|---|---|
| `stat_hollow` | `resource_hollow` |
| `stat_epiphyte` | `resource_epiphyte` |
| `stat_dead branch` | `resource_dead branch` |
| `stat_perch branch` | `resource_perch branch` |
| `stat_peeling bark` | `resource_peeling bark` |
| `stat_fallen log` | `resource_fallen log` |
| `stat_other` | `resource_other` |

**Source**: Created alongside `resource_*` in `a_resource_distributor_dataframes.py` and `combined_voxelise_dfs.py`.
Each `stat_` column is generated from its corresponding `resource_` column (typically `stat_col = resource_col.replace('resource_', 'stat_')`).
Used in template adjustment functions to target specific voxel counts/percentages for each resource.

**Key distinction**: `resource_*` is the **quantity** (summed across overlapping features). `stat_*` is **binary presence** after template adjustment — these are the columns queried by capability indicators.

### 1.3 `forest_*` — Habitat feature metadata per voxel

When the combined resource dataframe is integrated into the xarray, all non-resource columns are prefixed with `forest_`.
This happens in `a_voxeliser.py` → `rename_non_resource_columns()`.

| Attribute | Type | Values | Description |
|---|---|---|---|
| `forest_size` | string | `small`, `medium`, `large`, `senescing`, `snag`, `fallen`, `artificial` | Size/life-stage of the habitat feature occupying the voxel |
| `forest_control` | string | `street-tree`, `park-tree`, `reserve-tree`, `improved-tree`, `unassigned` | Management regime. Steps up only: street → park → reserve (via `unmanagedCount`). `improved-tree` is assigned to senescing AGE-IN-PLACE trees |
| `forest_tree_id` | int | Unique tree ID or -1 | Identifier linking back to the tree/log/pole dataframe |
| `forest_precolonial` | bool | `True` / `False` | Whether the tree is a precolonial species (native replacement or original) |
| `forest_structureID` | int | Index | Unique structure identifier in the combined node dataframe |

**Source columns** (before renaming): `size`, `control`, `tree_id`, `precolonial`, `structureID` from the node metadata columns in `a_resource_distributor_dataframes.py`.

### 1.4 `updatedResource_*` — Derived resource aggregates

Created in `a_scenario_generateVTKs.py` → `finalDSprocessing()`.

| Attribute | Type | Description |
|---|---|---|
| `updatedResource_elevatedDeadBranches` | float | For senescing trees: `resource_dead branch + resource_other`. Otherwise same as `resource_dead branch` |
| `updatedResource_groundDeadBranches` | float | For fallen trees: sum of `resource_dead branch`, `resource_other`, `resource_peeling bark`, `resource_fallen log`, `resource_perch branch`. Otherwise same as `resource_fallen log` |

---

## 2. Volume Attributes (Spatial & Site)

These describe the 3D voxel grid itself and the fixed properties of each voxel's location.

### 2.1 `centroid_*` — Voxel spatial coordinates

| Attribute | Type | Description |
|---|---|---|
| `centroid_x` | float | X-coordinate of voxel centre (eastings) |
| `centroid_y` | float | Y-coordinate of voxel centre (northings) |
| `centroid_z` | float | Z-coordinate of voxel centre (elevation) |

**Source**: Created during initial voxelisation of the site.

### 2.2 `voxel_*` — Voxel grid indices

| Attribute | Type | Description |
|---|---|---|
| `voxel_I` | int | Grid index along X axis |
| `voxel_J` | int | Grid index along Y axis |
| `voxel_K` | int | Grid index along Z axis |

### 2.3 `site_*` — Fixed site properties

| Attribute | Type | Values | Description |
|---|---|---|---|
| `site_building_element` | string | `facade`, `roof`, or empty/other | Building element classification from the 3D site model |

**Source**: Loaded from the site xarray during `a_scenario_initialiseDS.py`.

### 2.4 `envelope_*` — Building envelope classification

| Attribute | Type | Values | Description |
|---|---|---|---|
| `envelope_roofType` | string | `green roof`, `brown roof`, or empty/other | Roof classification from the log distributor (`a_logDistributor.py`), based on green/brown roof ratings |

### 2.5 `analysis_*` — Pre-computed site analysis

These are computed during the precompute pipeline and loaded into the xarray.

| Attribute | Type | Description |
|---|---|---|
| `analysis_combined_resistance` | float | Combined resistance score (with canopy) |
| `analysis_combined_resistanceNOCANOPY` | float | Combined resistance score (without canopy) |
| `analysis_potentialNOCANOPY` | float | Potential without canopy |
| `analysis_busyRoadway` | bool | True if carriageway on a Council Major corridor |
| `analysis_Roadway` | bool | True if carriageway of any type |
| `analysis_nodeID` | int | ID of the rewilding node this voxel belongs to |
| `analysis_nodeType` | string | Type of rewilding node |
| `analysis_forestLane` | float | Forest lane conversion value (> 0 = existing conversion) |
| `analysis_Canopies` | bool | True if under existing canopy (`site_canopy_isCanopy` or `road_canopy_isCanopy`) |

### 2.6 `node_*` — Canopy node properties

| Attribute | Type | Description |
|---|---|---|
| `node_CanopyID` | int | ID of the canopy node this voxel belongs to |
| `node_CanopyResistance` | float | Resistance value of the canopy node |

### 2.7 `isTerrainUnderBuilding` — Terrain flag

| Attribute | Type | Description |
|---|---|---|
| `isTerrainUnderBuilding` | bool | Whether this voxel is ground beneath a building footprint |

---

## 3. Scenario / Design Action Attributes

These change per scenario (positive/trending) and per year.

### 3.1 `sim_*` — Cellular automata simulation state

From the rewilding CA precompute (`a_rewilding.py`, run via `a_manager.py`), plus derived values from `a_scenario_initialiseDS.py`.

| Attribute | Type | Description |
|---|---|---|
| `sim_Nodes` | int | Rewilding node assignment for each voxel (-1 if unassigned) |
| `sim_Turns` | int | Number of CA turns to reach this voxel (-1 if unreachable). Lower = easier to convert |
| `sim_averageResistance` | float | Mean `analysis_combined_resistanceNOCANOPY` for the voxel's rewilding node. Created in `further_xarray_processing()` |
| `sim_nodeType` | string | Node type mapped from `analysis_nodeType` via `analysis_nodeID`. Created in `further_xarray_processing()` |
| `sim_NodesArea` | float | Area (m²) of the rewilding node |

### 3.2 `scenario_*` — Scenario outputs

Created across `a_scenario_runscenario.py` and `a_scenario_generateVTKs.py`.

| Attribute | Type | Values | Description |
|---|---|---|---|
| `scenario_rewildingEnabled` | int | `years_passed` or `-1` | Voxels eligible for any rewilding (satisfies `sim_Turns ≤ threshold` AND not facade/roof) |
| `scenario_rewildingPlantings` | int | `years_passed` or `-1` | Subset of `rewildingEnabled` that is also ≥ 5m from any existing tree |
| `scenario_rewilded` | string | `none`, `rewilded`, `exoskeleton`, `footprint-depaved`, `node-rewilded` | Rewilding status per voxel based on tree AGE-IN-PLACE actions and resistance thresholds |
| `scenario_bioEnvelope` | string | All `scenario_rewilded` values plus: `otherGround`, `livingFacade`, `greenRoof`, `brownRoof` | Extends `scenario_rewilded` with building-envelope treatments where `sim_Turns ≤ turnThreshold` AND `sim_averageResistance ≤ resistanceThreshold` |
| `scenario_outputs` | string | Tree sizes + rewilding statuses | Combined categorical: tree voxels show `forest_size`, rewilding voxels show `scenario_bioEnvelope`, remainder is `none` |

### 3.3 Masks

| Attribute | Type | Description |
|---|---|---|
| `bioMask` | bool | `sim_Turns ≤ turnThreshold` AND `sim_averageResistance ≤ resistanceThreshold` AND `sim_Turns ≥ 0` |
| `maskforTrees` | bool | Voxel has any `resource_*` > 0 (except `resource_leaf litter`) |
| `maskForRewilding` | bool | `scenario_bioEnvelope != 'none'` AND NOT `maskforTrees` |
| `envelopeIsBrownRoof` | int | 1 if `envelope_roofType == 'brown roof'`, else -1 |

---

## 4. Search Layers (Urban Elements & Bioavailability)

Created in `a_scenario_urban_elements_count.py`, added to the `_urban_features.vtk` output.

### 4.1 `FEATURES-*` — Site features transferred to scenario VTK

These are spatial lookups from the full site xarray into the scenario VTK using KDTree nearest-neighbour (within 1m).

| Attribute | Transferred from |
|---|---|
| `FEATURES-site_building_element` | `site_building_element` |
| `FEATURES-site_canopy_isCanopy` | `site_canopy_isCanopy` |
| `FEATURES-road_terrainInfo_roadCorridors_str_type` | Road corridor classification |
| `FEATURES-road_roadInfo_type` | Road type (Carriageway, Footpath, etc.) |
| `FEATURES-road_terrainInfo_forest` | Forest lane value |
| `FEATURES-road_terrainInfo_isOpenSpace` | 1 if open space |
| `FEATURES-road_terrainInfo_isParkingMedian3mBuffer` | 1 if within 3m of parking median |
| `FEATURES-road_terrainInfo_isLittleStreet` | 1 if classified as little street |
| `FEATURES-road_terrainInfo_isParking` | 1 if parking area |
| `FEATURES-road_canopy_isCanopy` | 1 if under road canopy |
| `FEATURES-poles_pole_type` | Pole type classification |
| `FEATURES-envelope_roofType` | Roof type from site |
| `FEATURES-analysis_busyRoadway` | Busy roadway flag |
| `FEATURES-analysis_Roadway` | Roadway flag |
| `FEATURES-analysis_forestLane` | Forest lane conversion value |
| `FEATURES-analysis_Canopies` | Under canopy flag |

### 4.2 `search_bioavailable` — Habitat type classification

| Value | Criteria |
|---|---|
| `none` | Not bioavailable |
| `open space` | `FEATURES-road_terrainInfo_isOpenSpace == 1` |
| `low-vegetation` | Any of: `scenario_rewildingPlantings > 0`, `scenario_rewilded != 'none'`, `scenario_bioEnvelope != 'none'`, `forest_size == 'fallen'`, `resource_fallen log > 0` |
| `arboreal` | Bioavailable but not open space or low-vegetation (trees, elevated canopy) |

### 4.3 `search_design_action` — Design intervention type

| Value | Criteria |
|---|---|
| `none` | No design action |
| `rewilded` | `scenario_rewildingPlantings > 0` |
| (from `scenario_rewilded`) | `exoskeleton`, `footprint-depaved`, `node-rewilded` |
| (from `scenario_bioEnvelope`) | `otherGround`, `livingFacade`, `greenRoof`, `brownRoof` |
| `improved-tree` | `forest_control == 'improved-tree'` |

### 4.4 `search_urban_elements` — Urban feature inventory

| Value | Criteria |
|---|---|
| `none` | Unclassified |
| `open space` | `FEATURES-road_terrainInfo_isOpenSpace == 1` |
| `green roof` | `FEATURES-envelope_roofType == 'green roof'` |
| `brown roof` | `FEATURES-envelope_roofType == 'brown roof'` |
| `facade` | `FEATURES-site_building_element == 'facade'` |
| `roadway` | `FEATURES-analysis_Roadway == True` |
| `busy roadway` | `FEATURES-analysis_busyRoadway == True` |
| `existing conversion` | `FEATURES-analysis_forestLane > 0` |
| `other street potential` | `FEATURES-road_terrainInfo_isParkingMedian3mBuffer == 1` |
| `parking` | `FEATURES-road_terrainInfo_isParking == 1` |

---

## 5. Persona Capability Indicators

Created in `a_info_gather_capabilities.py`. Boolean layers added to the `_with_indicators.vtk` output.

### 5.1 `indicator_*` — Capability presence per voxel

Each indicator is named `indicator_{Persona}_{capability}_{indicator}` (dots replaced with underscores).

| Attribute | Persona | Capability | Query |
|---|---|---|---|
| `indicator_Bird_self_peeling` | Bird | self (sustain) | `stat_peeling bark > 0` |
| `indicator_Bird_others_perch` | Bird | others (connect) | `stat_perch branch > 0` |
| `indicator_Bird_generations_hollow` | Bird | generations (persist) | `stat_hollow > 0` |
| `indicator_Lizard_self_grass` | Lizard | self | `search_bioavailable == low-vegetation` |
| `indicator_Lizard_self_dead` | Lizard | self | `stat_dead branch > 0` |
| `indicator_Lizard_self_epiphyte` | Lizard | self | `stat_epiphyte > 0` |
| `indicator_Lizard_others_notpaved` | Lizard | others | `ground_not_paved` (ground mask AND NOT paved mask) |
| `indicator_Lizard_generations_nurse-log` | Lizard | generations | `stat_fallen log > 0` |
| `indicator_Lizard_generations_fallen-tree` | Lizard | generations | `forest_size == fallen` |
| `indicator_Tree_self_senescent` | Tree | self | `forest_size == senescing` |
| `indicator_Tree_others_notpaved` | Tree | others | `ground_not_paved` + within 50m of canopy-feature + ground_only |
| `indicator_Tree_generations_grassland` | Tree | generations | `search_bioavailable == low-vegetation` + within 20m of canopy-feature + ground_only |

### 5.2 Derived masks used by indicators

| Mask | Definition |
|---|---|
| `ground_not_paved` | `get_ground_mask()` AND NOT `get_paved_mask()`. Ground = `search_bioavailable` in (`low-vegetation`, `open space`). Paved = `search_urban_elements` in (`roadway`, `busy roadway`, `parking`) |
| `canopy-feature` (distance reference) | `forest_size` is not nan/none/empty, OR `stat_fallen log > 0` |
| `ground_only` filter | Excludes `search_urban_elements` in (`facade`, `green roof`, `brown roof`) |

---

## Summary: VTK File Types and Their Attributes

| VTK file suffix | Script | Contains |
|---|---|---|
| `_scenarioYR{year}.vtk` | `a_scenario_generateVTKs.py` | `resource_*`, `stat_*`, `forest_*`, `scenario_*`, `sim_*`, `centroid_*`, `node_*`, `site_*`, `envelope_*`, masks, `updatedResource_*` |
| `_scenarioYR{year}_urban_features.vtk` | `a_scenario_urban_elements_count.py` | All of the above + `FEATURES-*`, `search_bioavailable`, `search_design_action`, `search_urban_elements` |
| `_urban_features_with_indicators.vtk` | `a_info_gather_capabilities.py` | All of the above + `indicator_*` |
| `_baseline_combined_{voxel_size}.vtk` | `a_scenario_get_baselines.py` | `resource_*`, `stat_*`, `forest_*`, `centroid_*` |
| `_baseline_combined_{voxel_size}_urban_features.vtk` | `a_scenario_urban_elements_count.py` | Baseline + `search_bioavailable`, `search_design_action`, `search_urban_elements`, `forest_control` |

---

## Appendix: `resource_*` vs `stat_*` Naming

Both columns originate from the same resource distribution pipeline:

- **`resource_*`** columns carry **summed quantities** — when multiple habitat feature point-cloud voxels overlap a single grid voxel, their resource values are summed.
- **`stat_*`** columns carry **binary presence** (0 or 1) — created from `resource_*` via `stat_col = col.replace('resource_', 'stat_')` in `a_resource_distributor_dataframes.py` and `combined_voxelise_dfs.py`. After template-level adjustment (targeting real-world resource counts/percentages), `stat_*` is the adjusted binary indicator.

The capability indicator queries (`a_info_gather_capabilities.py`) query **`stat_*`**, not `resource_*`, because capabilities care about *presence* of a resource in a voxel, not its quantity.
