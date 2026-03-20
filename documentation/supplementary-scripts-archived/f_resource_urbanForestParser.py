import pandas as pd
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import pickle

# Set a fixed seed for reproducibility
SEED = 1
rng = np.random.default_rng(SEED)

def determine_precolonial_status(geoDF):
    """
    Determine if trees are precolonial based on their Scientific Name and Genus.
    """
    print("Loading pre-colonial species list...")
    species_list = pd.read_csv('data/csvs/pre-colonial-plant-list.csv')
    print(f"Loaded {len(species_list)} species from the pre-colonial list.")
    
    # Check if the scientific name is in the pre-colonial species list
    name_mask = geoDF['urbanForest_scientific_name'].isin(species_list['Species'])
    geoDF['isPrecolonial'] = name_mask
    
    # Additionally, mark trees as precolonial if they belong to the Eucalyptus genus
    geoDF.loc[geoDF['urbanForest_genus'] == 'Eucalyptus', 'isPrecolonial'] = True

    print(f"Precolonial status assigned. Number of precolonial trees: {geoDF['isPrecolonial'].sum()}")
    
    return geoDF

def assign_control_and_size(geoDF):
    """
    Assign control type and tree size based on 'Located in' and 'Diameter Breast Height'.
    """
    print("Assigning control types based on 'Located in' field...") 
    
    # Mapping urbanForest_located_in to control types
    control_map = {
        'Street': 'street-tree',
        'Park': 'park-tree',
        'Reserve': 'reserve-tree'
    }

    # Use map to assign Control types, fill with default value if no match, and convert to category
    geoDF['Control'] = geoDF['urbanForest_located_in'].map(control_map).fillna('other-tree').astype('category')

    print(f"Control types assigned. Distribution:\n{geoDF['Control'].value_counts()}")


    print("Assigning tree sizes based on 'Diameter Breast Height'...")
    
    # Initialize 'Tree_size' with the default value 'medium'
    geoDF['Tree_size'] = 'medium'
    
    # Assign tree sizes based on 'Diameter Breast Height'
    geoDF['Tree_size'] = pd.cut(
        geoDF['urbanForest_diameter_breast_height'],
        bins=[-np.inf, 50, 80, np.inf],
        labels=['small', 'medium', 'large'],
        include_lowest=True
    )
    
    # Final check for NaNs in 'Tree_size', replace with 'medium' as default
    geoDF['Tree_size'].fillna('medium', inplace=True)
    
    print(f"Tree sizes assigned. Distribution:\n{geoDF['Tree_size'].value_counts()}")
    
    # Print unique combinations of 'Control' and 'Tree_size'
    print("\nUnique combinations of Control and Tree_size:")
    unique_combinations = geoDF[['Control', 'Tree_size']].drop_duplicates()
    print(unique_combinations)

    return geoDF
def calculate_z_coordinates(processedDF, terrain_grid):
    """
    Calculate the z-coordinate for each point in processedDF using a KDTree based on an UnstructuredGrid.
    
    Parameters:
    processedDF (pd.DataFrame): The DataFrame with 'x' and 'y' coordinates.
    terrain_grid (pv.UnstructuredGrid): The terrain grid to query for z-coordinates.
    
    Returns:
    processedDF (pd.DataFrame): The DataFrame with an added 'z' column.
    """
    print("Calculating z-coordinates using the terrain grid...")
    
    # Extract the x, y, z points from the UnstructuredGrid
    grid_points = terrain_grid.points  # This returns an array of shape (N, 3) for N points
    
    print(f"Terrain grid has {grid_points.shape[0]} points.")
    
    # Create a KDTree from the terrain grid's x and y coordinates
    tree = cKDTree(grid_points[:, :2])  # Use only the x and y coordinates
    
    # Query the closest points in the terrain grid for each x, y in processedDF
    distances, indices = tree.query(processedDF[['x', 'y']].values)
    
    # Assign the corresponding z values from the grid_points array to the processedDF
    processedDF['z'] = grid_points[indices, 2]
    
    print(f"Z-coordinates assigned for {len(processedDF)} points.")
    
    return processedDF

def get_tree_id(is_precolonial, tree_size):
    """
    Get a random tree_id based on the precolonial status and tree size.
    """
    tree_id = np.select(
        [
            (is_precolonial == False) & (tree_size == 'small'),
            (is_precolonial == False) & (tree_size == 'medium'),
            (is_precolonial == False) & (tree_size == 'large'),
            (is_precolonial == True) & (tree_size == 'small'),
            (is_precolonial == True) & (tree_size == 'medium'),
            (is_precolonial == True) & (tree_size == 'large')
        ],
        [
            rng.integers(4, 7, size=len(is_precolonial)),  # False, small
            rng.integers(1, 4, size=len(is_precolonial)),  # False, medium
            rng.integers(7, 15, size=len(is_precolonial)),  # False, large
            rng.integers(1, 5, size=len(is_precolonial)),  # True, small
            rng.integers(5, 11, size=len(is_precolonial)),  # True, medium
            rng.integers(11, 17, size=len(is_precolonial))  # True, large
        ],
        default=None
    )
    return tree_id


def create_urban_forest_df(urbanforestDF, terrain_grid):
    """
    Create the processed dataframe with x, y, z, size, control, and other columns.
    """
    print("Creating processed urban forest DataFrame...")
    
    # Initialize processed DataFrame
    processedDF = pd.DataFrame()

    # Assign x, y from easting and northing
    processedDF['x'] = urbanforestDF['urbanForest_easting']
    processedDF['y'] = urbanforestDF['urbanForest_northing']
    
    print("Assigned x and y coordinates.")

    # Calculate z-coordinates
    processedDF = calculate_z_coordinates(processedDF, terrain_grid)

    # Assign tree size and control
    urbanforestDF = assign_control_and_size(urbanforestDF)
    processedDF['size'] = urbanforestDF['Tree_size']
    processedDF['control'] = urbanforestDF['Control']

    print("Tree size and control columns added.")

    # Determine precolonial status
    urbanforestDF = determine_precolonial_status(urbanforestDF)
    processedDF['precolonial'] = urbanforestDF['isPrecolonial']

    print("Precolonial status added.")

    # Copy remaining fields directly
    processedDF['life_expectency'] = urbanforestDF['urbanForest_useful_life_expectency']
    processedDF['diameter_breast_height'] = urbanforestDF['urbanForest_diameter_breast_height']
    processedDF['age_description'] = urbanforestDF['urbanForest_age_description']
    processedDF['useful_life_expectency'] = urbanforestDF['urbanForest_useful_life_expectency_value']
    processedDF['eastings'] = urbanforestDF['urbanForest_easting']
    processedDF['northings'] = urbanforestDF['urbanForest_northing']

    print("Assigning tree ID...")
    processedDF['tree_id'] = get_tree_id(processedDF['precolonial'], processedDF['size']).astype(int)

    print("Assigning tree number")
    #create a column that is the row index
    processedDF = processedDF.reset_index(drop=True)
    processedDF['tree_number'] = processedDF.index.astype(int)

    print('Data types of each column:')
    print(processedDF.dtypes)

    print("Processed DataFrame creation complete.")


    return processedDF

