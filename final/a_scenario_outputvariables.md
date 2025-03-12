# Scenario Output Variables Documentation

This document provides a comprehensive reference for all variables created and modified during scenario simulations. It serves as a guide for understanding the data structure and interpretation of simulation results.

## Variables from a_scenario_runscenario.py

### 1. scenario_rewildingEnabled

**Purpose**: Identifies all voxels eligible for any rewilding  
**Values**:
- `params['years_passed']` (integer): Where voxels are eligible
- `-1`: Where voxels are not eligible

**Creation Criteria**:
- `depaved_mask`: Voxels where `sim_Turns <= depaved_threshold` (areas feasible for modification)
- `terrain_mask`: Voxels that are not 'facade' or 'roof' elements
- `combined_mask = depaved_mask & terrain_mask`

**Created in**: `assign_rewilded_status()`  
**Usage**: General indicator of rewildable areas

### 2. scenario_rewildingPlantings

**Purpose**: Identifies voxels specifically eligible for new tree planting  
**Values**:
- `params['years_passed']` (integer): Where planting is allowed
- `-1`: Where planting is not allowed

**Creation Criteria**:
- Includes all criteria from `scenario_rewildingEnabled`
- Plus `proximity_mask`: Voxels that are at least 5 meters away from any existing tree
- `final_mask = depaved_mask & terrain_mask & proximity_mask`

**Created in**: `assign_rewilded_status()`  
**Usage**: Directly used by `handle_plant_trees()` to select positions for new tree plantings

## Variables from a_scenario_generateVTKs.py

### 3. scenario_rewilded

**Purpose**: Tracks rewilding status of each voxel  
**Values**: String categories
- `'none'`: Not rewilded
- `'rewilded'`: Generic rewilding (points that are `scenario_rewildingEnabled` but don't have a specific rewilding type)
- `'exoskeleton'`: Exoskeleton support structure
- `'footprint-depaved'`: Depaved areas
- `'node-rewilded'`: Node-based rewilding

**Created in**: `update_rewilded_voxel_catagories()`  
**Usage**: Primary indicator of rewilding type for visualization and analysis

### 4. bioMask

**Purpose**: Boolean mask identifying bio-envelope eligible voxels  
**Values**:
- `True`: Eligible for bio-envelope
- `False`: Not eligible

**Creation Criteria**:
- `bioMask = (ds['sim_Turns'] <= turnThreshold) & (ds['sim_averageResistance'] <= resistanceThreshold) & (ds['sim_Turns'] >= 0)`

**Created in**: `update_bioEnvelope_voxel_catagories()`  
**Usage**: Used to determine which voxels can receive bio-envelope treatments

### 5. scenario_bioEnvelope

**Purpose**: Categorizes bio-envelope type for each voxel  
**Values**: String categories
- `'none'`: Not rewilded
- `'rewilded'`: Generic rewilding (points that are `scenario_rewildingEnabled` but don't have a specific rewilding type)
- `'exoskeleton'`: Exoskeleton support structure
- `'footprint-depaved'`: Depaved areas
- `'node-rewilded'`: Node-based rewilding
- `'otherGround'`: Ground-based bio-envelope
- `'livingFacade'`: Building facade bio-envelope
- `'greenRoof'`: Green roof bio-envelope
- `'brownRoof'`: Brown roof bio-envelope

**Created in**: `update_bioEnvelope_voxel_catagories()`  
**Usage**: Determines specific bio-envelope treatments for visualization and analysis

### 6. updatedResource_elevatedDeadBranches

**Purpose**: Tracks combined dead branch resources for senescing trees  
**Values**: Numerical values (floating point)  
**Creation Criteria**:
- For senescing trees: `resource_dead branch + resource_other`
- For other trees: Same as `resource_dead branch`

**Created in**: `finalDSprocessing()`  
**Usage**: Used for ecological resource assessment and visualization

### 7. updatedResource_groundDeadBranches

**Purpose**: Tracks combined resources for fallen trees  
**Values**: Numerical values (floating point)  
**Creation Criteria**:
- For fallen trees: Sum of `resource_dead branch`, `resource_other`, `resource_peeling bark`, `resource_fallen log`, and `resource_perch branch`
- For other trees: Same as `resource_fallen log`

**Created in**: `finalDSprocessing()`  
**Usage**: Used for ecological resource assessment and visualization

### 8. maskforTrees

**Purpose**: Boolean mask identifying voxels with tree resources  
**Values**:
- `True`: Voxel contains tree resources
- `False`: No tree resources

**Creation Criteria**:
- Voxels where any resource variable (except `resource_leaf litter`) has a value > 0

**Created in**: `finalDSprocessing()`  
**Usage**: Used for filtering tree-related voxels for visualization and analysis

### 9. maskForRewilding

**Purpose**: Boolean mask identifying voxels with rewilding status  
**Values**:
- `True`: Voxel has been rewilded
- `False`: Not rewilded

**Creation Criteria**:
- Voxels where `scenario_rewilded != 'none'`

**Created in**: `finalDSprocessing()`  
**Usage**: Used for filtering rewilded voxels for visualization and analysis

### 10. scenario_outputs

**Purpose**: Combined categorical variable for visualization  
**Values**: String categories including:
- Tree sizes: `'small'`, `'medium'`, `'large'`, `'senescing'`, `'snag'`, `'fallen'`
- Rewilding statuses: `'none'`, `'rewilded'`, `'exoskeleton'`, `'footprint-depaved'`, `'node-rewilded'`

**Creation Criteria**:
- For rewilded voxels: Uses the value from `scenario_rewilded`
- For tree voxels: Uses the value from `forest_size`
- Default: `'none'`

**Created in**: `finalDSprocessing()`  
**Usage**: Primary variable for comprehensive visualization of scenario outcomes

## Resource Variables (Created in a_voxeliser.py)

### 11. resource_* Variables

**Purpose**: Track specific ecological resources in each voxel  
**Examples**:
- `resource_dead branch`
- `resource_fallen log`
- `resource_peeling bark`
- `resource_perch branch`
- `resource_other`
- `resource_leaf litter`

**Values**: Numerical values (floating point) indicating resource quantity  
**Created in**: `integrate_resources_into_xarray()`  
**Usage**: Fundamental variables for ecological assessment

### 12. forest_* Variables

**Purpose**: Track forest characteristics in each voxel  
**Examples**:
- `forest_size`: Tree size category (`'small'`, `'medium'`, `'large'`, `'senescing'`, `'snag'`, `'fallen'`)
- `forest_tree_id`: Unique identifier for each tree
- `forest_control`: Management category (`'street-tree'`, `'park-tree'`, `'reserve-tree'`, `'improved-tree'`)

**Created in**: `integrate_resources_into_xarray()`  
**Usage**: Used for tree-specific visualization and analysis

## Simulation Control Variables

### 13. sim_* Variables

**Purpose**: Control simulation parameters and track simulation state  
**Examples**:
- `sim_Turns`: Number of turns in simulation
- `sim_Nodes`: Node identifiers
- `sim_averageResistance`: Average resistance values
- `sim_NodesArea`: Area of nodes

**Created in**: Various initialization functions  
**Usage**: Used as inputs to determine rewilding eligibility and other simulation outcomes

## Site-Specific Variables

### 14. site_* and envelope_* Variables

**Purpose**: Track site-specific characteristics  
**Examples**:
- `site_building_element`: Building element type (`'facade'`, `'roof'`, etc.)
- `envelope_roofType`: Roof type (`'green roof'`, `'brown roof'`, etc.)

**Created in**: Site initialization functions  
**Usage**: Used to determine eligibility for specific treatments like bio-envelopes

## Spatial Reference Variables

### 15. centroid_* Variables

**Purpose**: Track spatial location of each voxel  
**Examples**:
- `centroid_x`: X-coordinate of voxel centroid
- `centroid_y`: Y-coordinate of voxel centroid
- `centroid_z`: Z-coordinate of voxel centroid

**Created in**: Initialization functions  
**Usage**: Used for spatial operations like proximity calculations and visualization

## Node Reference Variables

### 16. node_* Variables

**Purpose**: Track node-specific information  
**Examples**:
- `node_CanopyID`: Canopy node identifier

**Created in**: Initialization functions  
**Usage**: Used for matching voxels to tree nodes for rewilding operations

## Variable Dependencies and Processing Flow

### Initialization Phase
1. Load site-specific data (`site_*`, `envelope_*`, `centroid_*`, `node_*`, `sim_*`)
2. Initialize tree, log, and pole dataframes

### Scenario Simulation Phase
1. Age trees and determine tree actions (`AGE-IN-PLACE`, `REPLACE`)
2. Process senescing trees and assign rewilding status
3. Create `scenario_rewildingEnabled` and `scenario_rewildingPlantings` variables
4. Handle tree planting and control reduction

### VTK Generation Phase
1. Update `scenario_rewilded` based on tree dataframe
2. Ensure rewilding variables exist by calling `assign_rewilded_status`
3. Create `scenario_bioEnvelope` and `bioMask` if logs/poles exist
4. Integrate resources into xarray (`resource_*`, `forest_*`)
5. Create derived variables (`updatedResource_*`, `maskforTrees`, `maskForRewilding`, `scenario_outputs`)
6. Convert to polydata and save VTK

## Notes on Variable Persistence

- Variables created in `a_scenario_runscenario.py` need to be recreated when skipping scenario regeneration
- The direct import of `assign_rewilded_status` in `a_scenario_generateVTKs.py` ensures consistent variable creation
- All variables are preserved in the final VTK files for visualization and analysis

## Usage in Visualization

The primary variables used for visualization are:
- `scenario_outputs`: Combined categorical variable for comprehensive visualization
- `forest_size`: For tree-specific visualization
- `scenario_rewilded`: For rewilding-specific visualization
- `scenario_bioEnvelope`: For bio-envelope visualization
- `resource_*` variables: For ecological resource visualization

## Usage in Analysis

The primary variables used for quantitative analysis are:
- `resource_*` variables: For ecological resource assessment
- `maskforTrees` and `maskForRewilding`: For area calculations
- `scenario_rewildingPlantings`: For tree planting potential assessment
- `scenario_bioEnvelope`: For bio-envelope coverage assessment 