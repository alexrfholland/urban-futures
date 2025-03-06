# Deployable Structures Flow

Shared VTK file: `f'data/{site}/offensivePrep-{site}.vtk'`  
Properties used: `'offensiveScore'`

## A. Upgrade utility poles

1. In `deployable_manager.py`:
   a. `createDeployables(site, plotter)`
      - Reads: VTK file property `'offensiveScore'`
   b. `getDeployableStructures(site)`
   c. `deployable_poles.runUtilityPole(site)`

2. In `deployable_poles.py`:
   a. `loadFiles(site)`
      - Reads:
        - `'data/deployables/poles.shp'`
        - `'data/deployables/raster_canopy_distance.tif'`
   b. `extract_and_plot_poles(poles_gdf, bounds)`
      - Generates: `poles_within_bounds_no_data`
   c. `normalise_canopy_raster(treeRasterData, poleProbWeight)`
      - Generates: `weighted_canopy_raster`
   d. `select_poles_probabilistically(poleLocs, weighted_canopy_raster, transform)`
      - Generates: `selected_poles_gdf` with `'condition'` and `'structure'` attributes

### Input Data:

- Pole shapefile: `'data/deployables/poles.shp'`
- Canopy proximity raster: `'data/deployables/raster_canopy_distance.tif'`
- VTK file property: `'offensiveScore'`

### Intermediate Data:

- `poles_gdf`: GeoDataFrame of pole locations
- `treeRasterData`: Numpy array of canopy proximity data
- `weighted_canopy_raster`: Normalized and weighted canopy proximity data

### Output Data:

- `selected_poles_gdf`: GeoDataFrame with selected poles, including `'condition'` and `'structure'` attributes
- Saved to: `"data/testPoles.shp"`

## B. Deploy artificial trees

1. In `deployable_manager.py`:
   a. `createDeployables(site, plotter)`
      - Reads: VTK file property `'offensiveScore'`
   b. `getDeployableStructures(site)`
   c. `deployable_lightweight.getLightweights(site)`

2. In `deployable_lightweight.py`:
   a. `getLightweights(site)`
      - Reads:
        - `'data/deployables/raster_poles_distance.tif'`
        - `'data/deployables/raster_canopy_distance.tif'`
        - `'data/deployables/raster_parking-and-median-buffer.tif'`
        - `'data/deployables/raster_deployables_private.tif'`
   b. `distribute_structures_with_combined_gradients()` for public areas
      - Generates: `deployables_coordinates`, `combinedGradient`
   c. `distribute_structures_with_combined_gradients()` for private areas
      - Generates: `private_deployables_coordinates`, `privateCombinedGradient`
   d. Combine public and private deployables
      - Generates: `combined_deployables` with `'condition'` and `'structure'` attributes

### Input Data:

- Pole proximity raster: `'data/deployables/raster_poles_distance.tif'`
- Canopy proximity raster: `'data/deployables/raster_canopy_distance.tif'`
- Public deployable areas raster: `'data/deployables/raster_parking-and-median-buffer.tif'`
- Private deployable areas raster: `'data/deployables/raster_deployables_private.tif'`
- VTK file property: `'offensiveScore'`

### Intermediate Data:

- `combinedGradient`: Weighted combination of pole and canopy proximity for public areas
- `privateCombinedGradient`: Weighted combination of pole and canopy proximity for private areas
- `deployables_coordinates`: GeoDataFrame of public deployable locations
- `private_deployables_coordinates`: GeoDataFrame of private deployable locations

### Output Data:

- `combined_deployables`: GeoDataFrame with all deployable locations, including `'condition'` and `'structure'` attributes
- Saved to: `"data/teststructures.shp"`

## C. Final Output

In `paintStructuresOnSite.py`:
- Combines all structures into a multiblock dataset
- Saves final output to: `f'data/{site}/structures-{site}-{state}.vtm'`

# Defensive Structures Flow

## C. Increase green buffers

0. Relevant steps in generating `'data/{site}/updated-{site}.vtk'`:
    a. In `update_site_polydata.py`
        - Call `calculate_disturbances.py`
    b. In `@calculate_disturbances.py`
        - Initial Disturbance Potential Assignment (in `assign_conversion_potential` function):
           - Disturbance potential 4: Existing public green space
             `{'isopen_space': [True]}`
           - Disturbance potential 3: Already proposed conversions
             `{'laneways-verticalsu': [(50, float('inf'))],
              'laneways-parklane': [(50, float('inf'))],
              'laneways-forest': [(50, float('inf'))],
              'laneways-farmlane': [(50, float('inf'))]}`
           - Disturbance potential 2: Public spaces that could be converted (little streets)
             `{'road_info-str_type': ['Council Minor'],
              'little_streets-islittle_streets': [True]}`
           - Disturbance potential 1: Private spaces that could be converted
             `{'private_buffer-isprivate_buffer': [False],
              'road_info-str_type': ['Private']}`
           - Disturbance potential 0: Main roads
             `{'road_types-type': ['Carriageway']}`
           - Disturbance potential -1: Buildings
             `{'blocktype': ['buildings']}`
        - Proximity-based Disturbance Potential (in `assign_disturbance_potential_proximity` function):
           1. Classified points (from step a) keep their original disturbance potential
           2. For unclassified points:
              - Find the nearest classified point using KDTree
              - Assign the disturbance potential of the nearest classified point
           3. Create `'disturbance-potential-proximity'` attribute with these values
        - Distance to Road Calculation:
           - For classified points: distance = 0
           - For unclassified points: calculate distance to nearest classified point
           - Store in `'distance to road'` attribute
        - Bucketed Distance:
           Create `'bucketed distance'` attribute:
           - 0: Points that are classified (distance = 0)
           - 1: Points within 0-10 units of a classified point
           - 2: Points within 10-20 units of a classified point
           - 3: Points 20 or more units away from a classified point
        - Distance Groups:
           Create `'distance groups'` attribute:
           `disturbance-potential-proximity * 4 + bucketed distance`
These attributes are then used in subsequent analyses, such as calculating defensive scores and classifying urban systems.

1. Read and prepare data:
   a. In `defensive_structure_manager.py`, main function:
      - Reads: VTK file `'data/{site}/updated-{site}.vtk'`
      - Reads: MultiBlock file `'data/{site}/combined-{site}-now.vtm'`
      - Calls: `defensiveStructureManager(poly_data)`

2. Calculate defensive scores:
   a. In `assign_potential_defensive_score` function:
      - Uses: `'bucketed distance'` and `'disturbance-potential-proximity'` point data from VTK
      - Calculates: `'disturbanceScore'` based on distance and disturbance potential mappings
   b. Calls: `kdemapper.getTreeWeights` to get `'tree-weights_log'`
      - Input data: 
        - Point data from poly_data including `'_Tree_size'`, `'isPrecolonial'`, `'_Control'`
      - Processing:
        - Calculates a weighted 2D KDE (Kernel Density Estimation) for classified trees
        - Uses `classifyTrees` function to assign weights:
            - 1: Small trees
              - `_Tree_size`: `["small"]`
            - 2: Medium, non-precolonial street trees
              - `_Tree_size`: `["medium"]`
              - `isPrecolonial`: `[False]`
              - `_Control`: `["street-tree"]`
            - 3: Medium, non-precolonial park trees
              - `_Tree_size`: `["medium"]`
              - `isPrecolonial`: `[False]`
              - `_Control`: `["park-tree"]`
            - 4: Medium, precolonial street trees
              - `_Tree_size`: `["medium"]`
              - `isPrecolonial`: `[True]`
              - `_Control`: `["street-tree"]`
            - 5: Medium, precolonial park trees
              - `_Tree_size`: `["medium"]`
              - `isPrecolonial`: `[True]`
              - `_Control`: `["park-tree"]`
            - 15: Medium, precolonial reserve trees
              - `_Tree_size`: `["medium"]`
              - `isPrecolonial`: `[True]`
              - `_Control`: `["reserve-tree"]`
            - 7: Large, non-precolonial street trees
              - `_Tree_size`: `["large"]`
              - `isPrecolonial`: `[False]`
              - `_Control`: `["street-tree"]`
            - 8: Large, non-precolonial park trees
              - `_Tree_size`: `["large"]`
              - `isPrecolonial`: `[False]`
              - `_Control`: `["park-tree"]`
            - 20: Large, precolonial street trees
              - `_Tree_size`: `["large"]`
              - `isPrecolonial`: `[True]`
              - `_Control`: `["street-tree"]`
            - 25: Large, precolonial park trees
              - `_Tree_size`: `["large"]`
              - `isPrecolonial`: `[True]`
              - `_Control`: `["park-tree"]`
            - 30: Large, precolonial reserve trees
              - `_Tree_size`: `["large"]`
              - `isPrecolonial`: `[True]`
              - `_Control`: `["reserve-tree"]`
      - Intermediate data:
        - Tree classification weights
        - 2D grid points for KDE evaluation
      - Output data:
        - Adds to poly_data:
          - `'{name}-intensity'`
          - `'{name}-weights_minmax'`
          - `'{name}-weights_quartile'`
          - `'{name}-weights_zscore'`
          - `'{name}-weights_log'`
          - `'{name}-weights_robust'`
   c. Calculates: `'defensiveScore'` as geometric mean of `'disturbanceScore'` and `'tree-weights_log'`

3. Classify urban systems:
   a. In `classify_urban_systems` function:
      - Defines criteria for various urban system types
      - Uses: `searcher.classify_points_poly` to apply criteria and classify points
      - Generates: `'urban system'` point data
   b. Urban system types and their search conditions:
      1. "Adaptable Vehical Infrastructure":
         - OR:
           - `parkingmedian-isparkingmedian`: True
           - AND:
             - `disturbance-potential`: [4, 2, 3]
             - NOT:
               - `little_streets-islittle_streets`: True AND `road_types-type`: "Footway"
      2. "Private empty space":
         - `disturbance-potential`: [1]
      3. "Existing Canopies":
         - `_Tree_size`: ["large", "medium"]
         - NOT `road_types-type`: "Carriageway"
      4. "Existing Canopies Under Roads":
         - `_Tree_size`: ["large", "medium"]
         - `road_types-type`: "Carriageway"
      5. "Street pylons":
         - OR:
           - `isstreetlight`: True
           - `ispylons`: True
      6. "Load bearing roof":
         - `buildings-dip`: [0.0, 0.1]
         - `extensive_green_roof-RATING`: ["Excellent", "Good", "Moderate"]
         - `elevation`: [-20, 80]
      7. "Lightweight roof":
         - `buildings-dip`: [0.0, 0.1]
         - `intensive_green_roof-RATING`: ["Excellent", "Good", "Moderate"]
         - `elevation`: [-20, 80]
      8. "Ground floor facade":
         - `buildings-dip`: [0.8, 1.7]
         - `solar`: [0.2, 1.0]
         - `elevation`: [0, 10]
      9. "Upper floor facade":
         - `buildings-dip`: [0.8, 1.7]
         - `solar`: [0.2, 1.0]
         - `elevation`: [10, 80]
   c. Output:
      - Adds `'urban system'` attribute to poly_data with classified values

4. Assign fortified structures:
   a. In `assignFortifyStructures` function:
      - Defines design specifications for different structure types
      - Assigns structures based on `'urban system'` classification
      - Uses `'defensiveScore'` to probabilistically cull assigned structures
      - Generates: `'fortifiedStructures'` point data

5. Calculate deployable gradient:
   a. In `kdemapper.py`, `getDeployableGradient` function:
      - Calculates KDE for defensive structures
      - Calculates KDE for canopy
      - Combines scores to create `'offensiveScore'`

6. Save updated data:
   a. In main function of `defensive_structure_manager.py`:
      - Saves: Updated poly_data to `'data/{site}/defensive-{site}.vtk'`
   b. In main function of `kdemapper.py`:
      - Saves: Updated poly_data to `'data/{site}/offensivePrep-{site}.vtk'`

### Input Data:

- VTK file: `'data/{site}/updated-{site}.vtk'`
- MultiBlock file: `'data/{site}/combined-{site}-now.vtm'`
- Point data: `'bucketed distance'`, `'disturbance-potential-proximity'`
- Tree data for KDE weights

### Intermediate Data:

- `'disturbanceScore'`: Combination of distance and disturbance potential scores
- `'tree-weights_log'`: KDE weights for trees
- `'defensiveScore'`: Geometric mean of disturbanceScore and tree-weights_log
- `'urban system'`: Classification of urban systems based on various criteria
- `'fortifiedStructures'`: Assigned defensive structure types
- KDE values for defensive structures and canopy

### Output Data:

- Updated VTK file: `'data/{site}/defensive-{site}.vtk'`
- Updated VTK file: `'data/{site}/offensivePrep-{site}.vtk'`
- Point data:
  * `'fortifiedStructures'`: Assigned defensive structure types
  * `'offensiveScore'`: Combined score for offensive strategies

This process analyzes the urban environment, calculates scores based on disturbance potential and existing green features, classifies urban systems, assigns appropriate green buffer structures, and prepares data for both defensive and offensive ecological strategies.
