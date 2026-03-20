"""
Scenario Generation Process - Detailed Variable Catalog
====================================================

STEP 1: Initial Voxelization (a_voxeliser)
-----------------------------------------
Purpose: Creates initial 3D grid (xarray) from site geometry

Input Files:
- site_vtk: Building geometry 
- ground_vtk: Ground surface geometry

Site Classification Variables:
- site_building_element: 
    Values: ['facade', 'roof', 'ground']
    Purpose: Classifies each voxel's building context
- site_building_id:
    Values: Integer
    Purpose: Unique identifier for each building
- envelope_roofType:
    Values: ['green roof', 'brown roof', 'none']
    Purpose: Specifies roof treatment type

Spatial Variables Created:
- centroid_x, centroid_y, centroid_z:
    Type: Float
    Purpose: Voxel center coordinates
- voxel_size:
    Type: Float
    Purpose: Size of cubic voxel
- bounds_min_x, bounds_max_x, etc:
    Type: Float
    Purpose: Spatial extent of analysis

STEP 2: Tree and Pole Integration (a_urban_forest_parser)
------------------------------------------------------
Purpose: Adds existing trees and poles to xarray

Input Data:
- Tree inventory attributes:
    - species
    - height
    - canopy_width
    - DBH (Diameter at Breast Height)
    - health_score
    - useful_life_expectancy
    - x, y coordinates
- Pole locations and specifications

Forest Variables Created:
- forest_tree_id:
    Type: Integer
    Purpose: Unique tree identifier
    Condition: NaN if no tree present
- forest_size:
    Values: ['small', 'medium', 'large', 'senescing', 'snag', 'fallen']
    Conditions:
        - small: DBH < 30
        - medium: 30 ≤ DBH < 80
        - large: DBH ≥ 80
- forest_control:
    Values: ['street-tree', 'park-tree', 'reserve-tree']
    Purpose: Management regime
- forest_precolonial:
    Type: Boolean
    Purpose: Identifies pre-existing vs newly planted trees
- forest_useful_life_expectancy:
    Type: Integer
    Default assignments:
        - small trees: 80 years
        - medium trees: 50 years
        - large trees: 10 years
- forest_diameter_breast_height:
    Type: Float
    Purpose: Current DBH measurement
- forest_species:
    Type: String
    Purpose: Tree species identification
- forest_health_score:
    Type: Integer (0-100)
    Purpose: Tree health assessment

Resource Variables Created:
- resource_dead_branch:
    Type: Float
    Purpose: Dead branch habitat value
- resource_fallen_log:
    Type: Float
    Purpose: Fallen log habitat value
- resource_hollow:
    Type: Float
    Purpose: Tree hollow habitat value
- resource_leaf_litter:
    Type: Float
    Purpose: Leaf litter habitat value
- resource_peeling_bark:
    Type: Float
    Purpose: Peeling bark habitat value
- resource_perch_branch:
    Type: Float
    Purpose: Perching branch habitat value

Pole Variables Created:
- poles_pole_number:
    Type: Integer
    Purpose: Unique pole identifier
- poles_pole_type:
    Values: ['electricity', 'light', 'other']
    Purpose: Pole classification
- poles_height:
    Type: Float
    Purpose: Pole height measurement


STEP 3: Log Distribution (a_logDistributor)
-----------------------------------------
Purpose: Places logs on suitable roof surfaces

Input Data:
- logLibraryDF attributes:
    - log_type: Classification of log
    - weight_per_meter: Load calculations
    - diameter: Log size
    - length: Log length
    - habitat_value: Resource scoring

Roof Analysis Variables:
- roof_structural_capacity:
    Type: Float
    Values: Load bearing capacity in kg/m²
    Purpose: Determines suitable log placements
- roof_slope:
    Type: Float
    Purpose: Affects log placement feasibility
- roof_area:
    Type: Float
    Purpose: Available space for log placement

Log Variables Created:
- logs_log_id:
    Type: Integer
    Purpose: Unique log identifier
- logs_log_type:
    Values: ['hollow', 'solid', 'decomposed']
    Purpose: Log classification
- logs_orientation:
    Type: Float array [x,y,z]
    Purpose: Log rotation angles
- logs_length:
    Type: Float
    Purpose: Log length in meters
- logs_diameter:
    Type: Float
    Purpose: Log diameter in meters
- logs_weight:
    Type: Float
    Purpose: Total log weight
- logs_habitat_value:
    Type: Float
    Purpose: Habitat resource score

STEP 4: Resistance Grid (a_create_resistance_grid)
-----------------------------------------------
Purpose: Calculates resistance values for possible design actions

Analysis Layer Variables Created:
- analysis_forestLane:
    Type: Float
    Range: -1 to 1
    Source: road_terrainInfo_forest
    Purpose: Normalized forest corridor metric
    
- analysis_greenRoof:
    Type: Float
    Range: -1 to 1
    Source: site_greenRoof_ratingInt
    Condition: Values where rating > 1
    
- analysis_brownRoof:
    Type: Float
    Range: -1 to 1
    Source: site_brownRoof_ratingInt
    Condition: Values where rating > 1
    
- analysis_busyRoadway:
    Type: Binary (0,1)
    Conditions: True when:
        road_roadInfo_type == 'Carriageway' AND 
        road_terrainInfo_roadCorridors_str_type == 'Council Major'
        
- analysis_Roadway:
    Type: Binary (0,1)
    Condition: True when road_roadInfo_type == 'Carriageway'
    
- analysis_Canopies:
    Type: Binary (0,1)
    Condition: True when:
        site_canopy_isCanopy == 1.0 OR
        road_canopy_isCanopy == 1.0

Potential Values Created:
- analysis_potential:
    Type: Float
    Range: -1 to 100
    Values assigned:
    - Open Space: 80 (road_terrainInfo_isOpenSpace == 1.0)
    - Parking Median: 50 (road_terrainInfo_isParkingMedian3mBuffer == 1.0)
    - Little Streets: 60 (road_terrainInfo_isLittleStreet == 1.0)
    - Private Roads: 50 (road_terrainInfo_roadCorridors_str_type == 'Private')
    - Parking Areas: 100 (road_terrainInfo_isParking == 1.0)
    - Forest Lanes: 80-100 (scaled based on analysis_forestLane)
    - Green Roofs: 30-75 (scaled based on analysis_greenRoof)
    - Brown Roofs: 20-60 (scaled based on analysis_brownRoof)
    
- analysis_potentialNOCANOPY:
    Type: Float
    Range: -1 to 100
    Purpose: Stores potential before canopy adjustment

Resistance Values Created:
- analysis_resistance:
    Type: Float
    Range: -1 to 100
    Values assigned:
    - Busy Roadways: 100 (analysis_busyRoadway == 1)
    - Regular Roadways: 80 (analysis_Roadway == 1)
    Default: -1

Combined Resistance Surface:
- analysis_combined_resistance:
    Type: Float
    Range: 0 to 100
    Calculation:
    - Resistance areas: Remapped to 80-100
    - Potential areas: Remapped to 0-50
    - Neutral areas: 60
    Purpose: Final movement cost surface

Node Analysis Variables:
- analysis_nodeType:
    Values: ['tree', 'log', 'pole', 'unassigned']
    
- analysis_nodeSize:
    Values: ['small', 'medium', 'large', 'unassigned']
    
- analysis_originalID:
    Type: Integer
    Purpose: Original ID from source dataset
    
- analysis_nodeID:
    Type: Integer
    Purpose: New unified ID system across all node types

Simulation Variables Created:
- sim_averageResistance:
    Type: Float
    Range: 0-100
    Purpose: Average resistance in node area
    
- sim_Turns:
    Type: Integer
    Purpose: Path complexity counter
    
- sim_TurnsThreshold:
    Type: Integer
    Values by year:
        0: 0
        10: 300
        30: 1250
        60: 5000
        180: 5000
        
- sim_resistanceThreshold:
    Type: Float
    Values by year:
        0: 0
        10: 50
        30: 50
        60: 68
        180: 96

Temporary Processing Variables:
- forest_lane_mask: Boolean mask for forest lane analysis
- green_roof_mask: Boolean mask for green roof analysis
- brown_roof_mask: Boolean mask for brown roof analysis
- resistance_mask: Boolean mask for resistance areas
- potential_mask: Boolean mask for potential areas
- tree_mask: Boolean mask for tree locations
- log_mask: Boolean mask for log locations
- pole_mask: Boolean mask for pole locations

STEP 5: Rewilding Nodes (a_rewilding)
-----------------------------------
Purpose: Identifies and marks areas for rewilding

Canopy Analysis Variables:
- tree_CanopyID:
    Type: Integer
    Purpose: ID of nearest tree
    Condition: Within 10m for trees, 5m for logs/poles
- tree_CanopySize:
    Values: Inherited from forest_size
    Purpose: Size of nearest tree
- tree_CanopyControl:
    Values: Inherited from forest_control
    Purpose: Management regime of nearest tree
- tree_CanopyType:
    Values: ['logs', 'trees', 'pole']
    Purpose: Type of nearest structure

Node Variables Created:
- node_CanopyResistance:
    Type: Float
    Purpose: Average resistance for canopy footprint
- node_Area:
    Type: Float
    Purpose: Area of rewilded node
- node_ID:
    Type: Integer
    Purpose: Unique identifier for rewilding node

Scenario Variables Created:
- scenario_rewilded:
    Values: ['none', 'exoskeleton', 'footprint-depaved', 'node-rewilded']
    Conditions:
        - 'exoskeleton': High resistance, structural support needed
        - 'footprint-depaved': Ground level rewilding
        - 'node-rewilded': General rewilding area
- scenario_bioEnvelope:
    Values: All scenario_rewilded values plus:
        - 'otherGround'
        - 'livingFacade'
        - 'greenRoof'
        - 'brownRoof'
    Conditions:
        - bioMask = (sim_Turns ≤ turnThreshold) & 
                   (sim_averageResistance ≤ resistanceThreshold) & 
                   (sim_Turns ≥ 0)
- scenario_rewildingPlantings:
    Values: 
        - -1: Not eligible
        - years_passed: Planting enabled
    Purpose: Tracks planting opportunities

Mask Variables:
- maskforTrees:
    Type: Boolean
    True: Whens any resource_* > 0 (except leaf litter)
- maskForRewilding:
    Type: Boolean
    True: When scenario_rewilded or scenario_bioEnvelope != 'none'

Temporary Processing Variables:
- validpointsMask:
    Type: Boolean
    Purpose: Identifies valid processing points
- bioMask:
    Type: Boolean
    Purpose: Identifies areas meeting bio criteria
- siteMask:
    Type: Boolean
    Purpose: Separates site from tree areas

"""