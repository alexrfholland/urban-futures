import pyvista as pv
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
from typing import List, Tuple, Dict
import cmocean
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


try: 
    from modules import glyphs as glyphMapper, helper_functions, getBaselines, cameraSetUp
except Exception as e:
    print(f"An error for import script in trees.py: {e}. Continuing..")


"""
Functions:
load_urban_forest_data():
Takes in the site PolyData, a CSV path, site name, and box offset.
Constructs a k-d tree from the site PolyData points.
Filters and reads the tree data from the CSV, applying bounding box constraints.
Adds a '_Tree_size' column to categorize trees by size.
Returns a PolyData object (trees) containing the filtered tree points and additional attributes from the CSV.
assign_control_label():
Assigns a control label ('street-tree' or 'park-tree') to each tree in the given trees PolyData object.
Uses a k-d tree to find the nearest points in the site PolyData for each tree.
Modifies the trees PolyData object in-place to add a '_Control' attribute.
getTrees():
Calls load_urban_forest_data() and assign_control_label() to prepare and return tree PolyData with control labels.
main():
Prepares the initial site PolyData (point cloud of the urban context) and tree positions PolyData (x, y , z locations of trees and tree level attribtues such as their unique blockNo, size, control level, species).
Reads the 'branchPredictions - adjusted.csv' file to get scanned branch data.
Loops through each unique tree to extend branch attributes and tree attributes.
Transfers these attributes to a new branch_polydata PolyData object.
Reads the LeRoux dataset and updates the branch_polydata based on expected percentages of peeling bark and dead branches.
Debugging outputs include printing the tree IDs and unique tree numbers, as well as percentages before and after updating.
Visualization is done through pyvista's Plotter.
Core Logic:
The script starts by preparing site-specific and tree-specific PolyData objects (site_polydata and tree_polydata).
For each unique tree, the script generates a set of branches and transfers both branch-specific and tree-specific attributes to a new PolyData object (branch_polydata).
The script then updates the branch_polydata based on expected percentages from the LeRoux dataset.
Finally, the script visualizes the updated branch_polydata using pyvista.
"""

def load_urban_forest_data(site_polydata, csv_path, site, box_offset=125):

    
    # Create a mask for points where point_data['BlockType'] is not 'powerpole'
    mask = site_polydata.point_data['blocktype'] != 'powerline'

    # Filter the points and point_data using the mask
    filtered_points = site_polydata.points[mask]
    filtered_point_data = {key: np.array(val)[mask] for key, val in site_polydata.point_data.items()}

    # Create a new PolyData object with the filtered points and point_data
    filtered_site_polydata = pv.PolyData(filtered_points)
    for key, val in filtered_point_data.items():
        filtered_site_polydata.point_data[key] = val

    # Now use filtered_site_polydata in place of site_polydata for the rest of your function
    site_polydata = filtered_site_polydata
    
    tree = cKDTree(site_polydata.points[:, :2])
    df = pd.read_csv(csv_path)
    site_coord = pd.read_csv('data/site projections.csv')
    precolonial = pd.read_csv('data/csvs/pre-colonial-plant-list.csv')

    # Split the site string by hyphen and use the second part if available
    site_to_search = site.split('-')[1] if '-' in site else site

    easting_offset = site_coord.loc[site_coord['Name'].str.contains(site_to_search, case=False), 'Easting'].values[0]
    northing_offset = site_coord.loc[site_coord['Name'].str.contains(site_to_search, case=False), 'Northing'].values[0]

    #easting_offset = site_coord.loc[site_coord['Name'].str.contains(site, case=False), 'Easting'].values[0]
    #northing_offset = site_coord.loc[site_coord['Name'].str.contains(site, case=False), 'Northing'].values[0]

    #easting_offset = site_coord.loc[site_coord['Name'] == site, 'Easting'].values[0]
    #northing_offset = site_coord.loc[site_coord['Name'] == site, 'Northing'].values[0]
    bbox = [easting_offset - box_offset, northing_offset - box_offset, easting_offset + box_offset, northing_offset + box_offset]

    mask = ((df['Easting'] >= bbox[0]) & (df['Easting'] <= bbox[2]) & (df['Northing'] >= bbox[1]) & (df['Northing'] <= bbox[3]))
    df = df[mask]
    df.reset_index(drop=True, inplace=True)

    # Add a new column 'Tree Size' based on 'Diameter Breast Height'
    df['_Tree_size'] = df['Diameter Breast Height'].apply(lambda x: 'small' if x < 50 else ('medium' if 50 <= x < 80 else ('large' if x >= 80 else 'small')))
    #df['_Tree_size'] = 'large'

    points = df[['Easting', 'Northing']].values
    points[:, 0] -= easting_offset
    points[:, 1] -= northing_offset
    _, indices = tree.query(points)
    z_coords = site_polydata.points[indices, 2]
    points = np.column_stack([points, z_coords])


    ## REMOVE ALL TREE POINTS OUTSIDE SITE POLYDATA BOUNDS
    # Calculate the min and max x, y bounds for site_polydata
    min_x, min_y = np.min(site_polydata.points[:, :2], axis=0)
    max_x, max_y = np.max(site_polydata.points[:, :2], axis=0)

    # Create a mask for points that are within the x, y bounds
    mask = (
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
        (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    )

    # Apply the mask to points and df of tree attributes
    filtered_points = points[mask]
    df = df[mask]
    print(f'fitered points are {filtered_points}')
    points = filtered_points



    df['isPrecolonial'] = df['Scientific Name'].isin(precolonial['Species'])
    df.loc[df['Genus'] == 'Eucalyptus', 'isPrecolonial'] = True

    ##synthesise 

    adjustPrecolonial = True
    
    if adjustPrecolonial and (site == 'park' or site == 'street'):
        adjustedPreColonialValue = .2

        # Adjust a proportion of 'isPrecolonial' == False to True
        false_indices = df[df['isPrecolonial'] == False].index.tolist()
        num_to_adjust = int(adjustedPreColonialValue * len(false_indices))

        print(f'converting {adjustedPreColonialValue} of trees or {num_to_adjust} trees to precolonial')
        
        # Randomly select the indices to adjust
        selected_indices = np.random.choice(false_indices, num_to_adjust, replace=False)
        df.loc[selected_indices, 'isPrecolonial'] = True

    # Count how many entries have 'isPrecolonial' == True
    precolonial_count = df['isPrecolonial'].sum()
    print(f"Number of precolonial trees: {precolonial_count}")

    # List of 'Common Name' that are precolonial
    precolonial_common_names = df[df['isPrecolonial']]['Common Name'].tolist()
    print("List of precolonial common names:")
    print(precolonial_common_names)

    # List of 'Common Name' that are not precolonial
    non_precolonial_common_names = df[~df['isPrecolonial']]['Common Name'].tolist()
    print("\nList of non-precolonial common names:")
    print(non_precolonial_common_names)
        


    print(f"trees are {df['Common Name']}")

    """# Create single PolyData point cloud
    trees = pv.PolyData(points)
    for col in df.columns:
        sanitized_data = []
        for val in df[col]:
            sanitized_data.append(val.encode('ascii', 'ignore').decode() if isinstance(val, str) else val)
        trees.point_data[col] = sanitized_data
    
    trees.point_data['BlockNo'] = np.arange(len(points))
    print(f"trees are {trees.point_data['BlockNo']}")


    return trees"""

    # Assume points is a 2D array or list with shape (n_points, 3)
    df['X'] = points[:, 0]
    df['Y'] = points[:, 1]
    df['Z'] = points[:, 2]


    return df

def adjustState(df: pd.DataFrame, site: str, state: str) -> pd.DataFrame:
    if state == 'trending':
        mask = df['Useful Life Expectency Value'] < 20  # Create a boolean mask for trees with life expectancy < 20 years
        
        # Logging the number of trees to be updated
        num_trees_to_update = mask.sum()
        print(f"Number of trees to be updated: {num_trees_to_update}")
        
        df.loc[mask, 'Diameter Breast Height'] = 10  # Update DBH
        df.loc[mask, 'Genus'] = 'Eucalyptus'  # Update genus
        df.loc[mask, 'isPreColonial'] = 'yes'  # Update isPreColonial
        df.loc[mask, '_Tree_size'] = 'small'
        
        # Creating a new DataFrame with only the masked rows
        masked_df = df[mask]
        
        # Saving the masked DataFrame to a CSV file
        masked_df.to_csv(f'data/{site}/changedTrees-{site}-{state}.csv', index=False)
        

        
        # Logging the updates
        print(f"Updated DBH, Genus, and isPreColonial for {num_trees_to_update} trees.")
        
    else:
        print("State is not 'trending', no updates made to DataFrame.")

    
    return df


def assign_control_label(df: pd.DataFrame) -> pd.DataFrame:
    # Using the .loc method to update the '_Control' column based on conditions
    df.loc[df['Located in'] == 'Street', '_Control'] = 'street-tree'
    df.loc[df['Located in'] == 'Park', '_Control'] = 'park-tree'
    
    # Logging the updates
    street_tree_count = (df['_Control'] == 'street-tree').sum()
    park_tree_count = (df['_Control'] == 'park-tree').sum()
    print(f"Updated _Control column: {street_tree_count} street-trees, {park_tree_count} park-trees.")
    
    return df

def assign_control_labelPhysical(trees_df, site_polydata, min_distance, ha_threshold=.1):
    print('trees df is')
    print(trees_df)
    
    ha_values = site_polydata.point_data.get('open_space-HA')
    print(f'distinct ha_values are {np.unique(ha_values)}')
    
    relevant_indices = np.where(ha_values > ha_threshold)[0]
    relevant_points = site_polydata.points[relevant_indices, :2]
    
    # Build k-d tree for the relevant points
    tree = cKDTree(relevant_points)
    
    control = []  # use a list to store the control labels
    
    for index, row in trees_df.iterrows():
        point = row[['X', 'Y']].values
        dist, _ = tree.query(point)
        control_label = 'street-tree' if dist > min_distance else 'park-tree'
        control.append(control_label)  # append the control label to the list
    
    trees_df['_Control'] = control  # assign the list to a new column in the DataFrame
    return trees_df


def assign_control_labelPOLY(trees, site_polydata, min_distance, ha_threshold=.1):
    ha_values = site_polydata.point_data.get('open_space-HA')
    print(f'distinct ha_values are {np.unique(ha_values)}')
    relevant_indices = np.where(ha_values > ha_threshold)[0]
    relevant_points = site_polydata.points[relevant_indices, :2]
    
    # Build k-d tree for the relevant points
    tree = cKDTree(relevant_points)
    
    control = np.empty(trees.n_points, dtype=object)
    
    for i in range(trees.n_points):
        point = trees.points[i, :2]
        dist, _ = tree.query(point)
        control[i] = 'street-tree' if dist > min_distance else 'park-tree'
    
    trees.point_data['_Control'] = control
    return trees

def getTrees(site_df, site, state = 'present'):
    csv_path = 'data/csvs/trees-with-species-and-dimensions-urban-forest.csv'
    
    # Assuming load_urban_forest_data function has been updated to work with DataFrames
    #trees_df = load_urban_forest_data(site_df, csv_path, site)
    trees_df = load_urban_forest_data(site_df, csv_path, site, box_offset=3000)

    trees_df = adjustState(trees_df, site, state)

    trees_with_control_df = assign_control_label(trees_df)
    #trees_with_control_df = assign_control_labelPhysical(trees_df, site_df, 0.1)
    
    print(f"species are {trees_with_control_df['Common Name'].unique()}")
    print(f"sizes are {trees_with_control_df['_Tree_size'].unique()}")
    print(f"control levels are {trees_with_control_df['_Control'].unique()}")
    
    return trees_with_control_df

def getTreesPOLY(site_polydata, site):
    csv_path = 'data/csvs/trees-with-species-and-dimensions-urban-forest.csv'

    trees = load_urban_forest_data(site_polydata, csv_path, site)
    trees_with_control = assign_control_label(trees, site_polydata, 1)


    print(f"species are {trees.point_data['Common Name']}")
    print(f"sizes are {trees.point_data['_Tree_size']}")
    print(f"control levels are {trees.point_data['_Control']}")

    return trees_with_control


def make_canopy_resources(branch_points: np.ndarray,
                          hollow_no: float,
                          epiphyte_no: float,
                          point_area: float = 0.25) -> Dict[str, np.ndarray]:
    """
    Generate canopy resources (hollows and epiphytes) based on branch points.

    Parameters:
    - branch_points: Array of branch points
    - hollow_percentage: Expected percentage of hollows
    - epiphyte_percentage: Expected percentage of epiphytes
    - point_area: Area represented by a single point

    Returns:
    - resources: Dictionary containing arrays of generated points for hollows and epiphytes
    """
    total_points = branch_points.shape[0]

    def generate_resource_points(num_points: float, shift_z: float) -> np.ndarray:
        selected_indices = np.random.choice(total_points, round(num_points), replace=False)
        resource_points = branch_points[selected_indices].copy()
        resource_points[:, 2] += shift_z
        return np.asarray(resource_points)
        

    hollow_points = generate_resource_points(hollow_no, 0.2)
    epiphyte_points = generate_resource_points(epiphyte_no, 0.2)

    # Logging the data
    logging.info(f"Number of Hollow Points: {hollow_points.shape[0]}")
    logging.info(f"Number of Epiphyte Points: {epiphyte_points.shape[0]}")
    logging.info("---------")

    canopyDic = {"hollow": hollow_points, "epiphyte": epiphyte_points}

    logging.info(f'canopy rsource dic is {canopyDic}')

    return canopyDic

def generate_ground_cover_points(tree_location: np.ndarray, 
                                 percentage: float, 
                                 point_area: float, 
                                 tree: cKDTree,
                                 site_points: np.ndarray,
                                 ground_cover_type: str,  # either 'logs' or 'leaf litter'
                                 radius: float = 10.0) -> np.ndarray:
    """
    Generate points representing ground cover in the vicinity of a tree and log relevant information.

    Parameters:
    - tree_location: (x, y, z) location of the tree
    - percentage: Expected percentage coverage of the ground cover
    - point_area: Area represented by a single point
    - tree: k-d tree constructed from site points for quick nearest point lookup
    - site_points: Array of site points to match z-coordinates
    - ground_cover_type: Type of ground cover ('logs' or 'leaf litter')
    - radius: Radius around the tree to consider for generating points

    Returns:
    - points: Array of generated points with x, y, and z coordinates
    """
    
    # Calculate the number of points to generate
    num_points = int((percentage / 100.0) * (radius**2 * np.pi) / point_area)
    
    # Generate points within a circle around the tree location
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.random.uniform(0, radius, num_points)
    x = r * np.cos(theta) + tree_location[0]
    y = r * np.sin(theta) + tree_location[1]

    # For each (x, y) point, find the nearest point in the site and get its z-coordinate
    xy_points = np.vstack((x, y)).T
    _, indices = tree.query(xy_points)
    z = site_points[indices, 2] + 1

    points = np.vstack((x, y, z)).T

    # Logging the data
    expected_area = (percentage / 100.0) * (radius**2 * np.pi)
    actual_area = num_points * point_area

    logging.info(f"Ground Cover Type: {ground_cover_type}")
    logging.info(f"Expected Area Percentage: {percentage}%")
    logging.info(f"Number of Points: {num_points}")
    logging.info(f"Total Area Covered by Points (assuming points are 0.25m^2 each): {actual_area}m^2")
    logging.info(f"Actual Area Percentage: {actual_area / (radius**2 * np.pi) * 100}%")
    logging.info("---------")

    return points




def generate_resources_dict(field_data: Dict[str, any], leroux_df: pd.DataFrame) -> Dict[str, float]:
    
    tree_size = field_data['_Tree_size'][0]
    control = field_data['_Control'][0]

    # Renaming the column
    leroux_df = leroux_df.rename(columns={control: 'quantity'})

    mask = (leroux_df['name'].isin(['peeling bark', 'dead branch', 'fallen log', 'leaf litter', 'hollow', 'epiphyte'])) & (leroux_df['Tree Size'] == tree_size)
    
    grouped = leroux_df[mask].groupby('name')['quantity']
    
    resourceFactor = 1
    if not field_data['isPrecolonial'][0]:
        colonialFactor = .05
        resourceFactor = colonialFactor
        print(f'non precolonial tree, reducing resources by {colonialFactor}')

    resources_dict = {name: np.random.uniform(min_val, max_val) * resourceFactor for name, (min_val, max_val) in grouped.agg(['min', 'max']).iterrows()}
    logging.info(f'resources for tree {tree_size} and {control} are:')
    logging.info(resources_dict)

    return resources_dict



def update_block(points: np.ndarray, 
                 site: np.ndarray,
                 point_data: Dict[str, np.ndarray],
                 field_data: Dict[str, np.ndarray], 
                 leroux_df: pd.DataFrame,
                 tree: cKDTree,
                 ground_cover_data: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    
    
    tree_size = field_data['_Tree_size'][0]
    control = field_data['_Control'][0]
    tree_location = field_data['location'][0]
    logging.info(f'tree location is {tree_location}')

    #Get the dictionary of reseources for this tree
    resources_dict = generate_resources_dict(field_data, leroux_df)
    
    # Initialize arrays for the updated attributes
    updated_peeling_bark = np.zeros_like(point_data['peelingBark'], dtype=bool)
    updated_branch_type = point_data['Branch.type'].copy()
    perchable = np.zeros_like(point_data['peelingBark'], dtype=bool)  # Initialize the perchable attribute


    indices = np.arange(points.shape[0])  # indices for all points in the block
    total_branches = len(indices)

    current_peeling_bark_percentage = np.sum(point_data['peelingBark']) / total_branches * 100
    current_dead_branch_percentage = np.sum(point_data['Branch.type'] == 'dead') / total_branches * 100

    #dead_branch_percentage_expected = 100
    num_peeling_bark = int(resources_dict['peeling bark'] * total_branches / 100)
    num_dead_branches = int(resources_dict['dead branch'] * total_branches / 100)


    # Update the perchable attribute
    perchable_condition = (points[:, 2] > 10) & (point_data['Branch.angle'] < 21)
    perchable[perchable_condition] = True


    logging.info(f'num dead branches there should be: {num_dead_branches}')


    live_branch_indices = indices[point_data['Branch.type'] == 'live']
    dead_branch_indices = indices[point_data['Branch.type'] == 'dead']

    if current_dead_branch_percentage < resources_dict['dead branch']:
        num_dead_branches_to_update = num_dead_branches - np.sum(point_data['Branch.type'] == 'dead')
        dead_branch_indices_to_update = np.random.choice(live_branch_indices, num_dead_branches_to_update, replace=False)
        updated_branch_type[dead_branch_indices_to_update] = 'dead'
    elif current_dead_branch_percentage > resources_dict['dead branch']:
        num_live_branches_to_update = np.sum(point_data['Branch.type'] == 'dead') - num_dead_branches
        live_branch_indices_to_update = np.random.choice(dead_branch_indices, num_live_branches_to_update, replace=False)
        updated_branch_type[live_branch_indices_to_update] = 'live'

    logging.info(f"tree is {field_data['Genus'][0]}, {field_data['Common Name'][0]}")
    non_peeling_bark_indices = indices[point_data['peelingBark'] == False]
    num_peeling_bark_to_update = num_peeling_bark - np.sum(point_data['peelingBark'])
    peeling_bark_indices_to_update = np.random.choice(non_peeling_bark_indices, num_peeling_bark_to_update, replace=False)
    updated_peeling_bark[peeling_bark_indices_to_update] = True        

    num_peeling_bark_updated = np.sum(updated_peeling_bark)
    num_dead_branches_updated = np.sum(updated_branch_type == 'dead')

    new_peeling_bark_percentage = num_peeling_bark_updated / total_branches * 100
    new_dead_branch_percentage = num_dead_branches_updated / total_branches * 100

    ##ground cover

    # Assuming each point represents 0.25m^2
    point_area = 0.25


    fallen_log_points = generate_ground_cover_points(tree_location, 
                                                    resources_dict['fallen log'], 
                                                    point_area, 
                                                    tree, 
                                                    site, 
                                                    ground_cover_type='logs')

    leaf_litter_points = generate_ground_cover_points(tree_location, 
                                                    resources_dict['leaf litter'], 
                                                    point_area, 
                                                    tree, 
                                                    site, 
                                                    ground_cover_type='leaf litter')
    
    ground_cover_data["fallen_log"].extend(fallen_log_points)
    ground_cover_data["leaf_litter"].extend(leaf_litter_points)

    canopy_resources = make_canopy_resources(points, resources_dict['hollow'], resources_dict['epiphyte'])


    logging.info(f"""
            Debugging for Tree Size: {tree_size}, Control: {control}, Species: {field_data['Common Name']}
            Total branches for this tree: {total_branches}
            Expected percentages - Peeling Bark: {resources_dict['peeling bark']}, Dead Branch: {resources_dict['dead branch']}
            Current percentages - Peeling Bark: {current_peeling_bark_percentage}, Dead Branch: {current_dead_branch_percentage}
            Number of branches changed - Peeling Bark: {num_peeling_bark_updated}, Dead Branch: {num_dead_branches_updated}
            New percentages directly from branch_polydata.point_data - Peeling Bark: {new_peeling_bark_percentage}, Dead Branch: {new_dead_branch_percentage}
            """)
    
    total_branchesVoxels = len(point_data['Branch.type'])  # Assuming Branch.type exists for every branch
    #lateral >15m
        



    
    VoxelResourceCount = {
        'hollow': len(canopy_resources['hollow']),
        'peeling bark': np.sum(updated_peeling_bark),
        'dead branch': np.sum(updated_branch_type == 'dead'),
        'epiphyte': len(canopy_resources['epiphyte']),
        'leaf litter': len(leaf_litter_points),
        'fallen log': len(fallen_log_points),
        'total branches': np.sum(total_branchesVoxels),
        'perchable branches' : np.sum(perchable)
        #'sunny perch' : sunnyPerchablebranchesVoxels,
        #'shaded perch' : shadedPerchablebranchesVoxels
    }

    # Print or return the dictionary as needed:
    logging.info('voxel counts are:')
    logging.info(VoxelResourceCount)  # or return VoxelResourceCount along with other return values


    ##make the resource attribute 
        # Define the conditions and the corresponding categories
    conditions = [
        updated_branch_type == 'dead',
        perchable,
        updated_peeling_bark
    ]
    choices = ['dead branch', 'perchable branch', 'peeling bark']
    
    # Create the 'resources' attribute using np.select
    resources = np.select(conditions, choices, default='other')
    
    
    # Add it to your return dictionary along with other point attributes
    updated_point_data = {'peelingBark': updated_peeling_bark, 'Branch.type': updated_branch_type, 'resource': resources}

    
    return points, updated_point_data, canopy_resources, VoxelResourceCount


def process_site_data(site_polydata, tree_positions_df, site: str, box_offset: int) -> Dict[str, pv.PolyData]:

    
    # Create single PolyData point cloud
    points = tree_positions_df[['X', 'Y', 'Z']].values

    # Create single PolyData point cloud
    tree_positions = pv.PolyData(points)

    # Transfer the rest of the columns from the DataFrame to the PolyData point_data
    for col in tree_positions_df.columns:
        if col not in ['X', 'Y', 'Z']:  # Skip the 'X', 'Y', 'Z' columns since they're already used for the points
            sanitized_data = []
            for val in tree_positions_df[col]:
                sanitized_data.append(val.encode('ascii', 'ignore').decode() if isinstance(val, str) else val)
            tree_positions.point_data[col] = sanitized_data


    branch_data = pd.read_csv('data/csvs/branchPredictions - adjusted.csv')
    branch_data = branch_data.rename(columns={'Tree.ID': 'scanID'})

    tree_sizes = tree_positions.point_data['_Tree_size']
    scanIDs = np.zeros(tree_positions.n_points, dtype=int)
    scanIDs[tree_sizes == 'small'] = np.random.choice(np.arange(1, 5), size=(tree_sizes == 'small').sum())
    scanIDs[tree_sizes == 'medium'] = np.random.choice(np.arange(5, 10), size=(tree_sizes == 'medium').sum())
    scanIDs[tree_sizes == 'large'] = np.random.choice(np.arange(10, 17), size=(tree_sizes == 'large').sum())

    multiblock = pv.MultiBlock()
    
    grouped_branch_data = branch_data.groupby('scanID')


    for idx, (scanID, base_coord) in enumerate(zip(scanIDs, tree_positions.points)):
        
        logging.info(f'base coord is {base_coord}')

        #branches = branch_data[branch_data['scanID'] == scanID]
        branches = grouped_branch_data.get_group(scanID)
        
        
        translated_coords = branches[['x', 'y', 'z']].values + base_coord 

        tree_branch_polydata = pv.PolyData(translated_coords)

        
        # Adding branch-level attributes to each tree's branch PolyData as point_data
        for column in branches.columns:
            if column not in ['x', 'y', 'z']:
                tree_branch_polydata.point_data[column] = branches[column].values
            
        # Adding tree-level attributes to each tree's branch PolyData as field_data
        #NOTE - attributes wrapped in an array because pyvista errors when field_data are strings
        for attribute, value in tree_positions.point_data.items():
            tree_branch_polydata.field_data[attribute] = np.array([value[idx]])
            #tree_branch_polydata.field_data.set_array(value[idx], attribute)

        # Assign the tree xyz point (base_coord) and tree id (idx)
        tree_branch_polydata.field_data['location'] = np.array([base_coord])
        tree_branch_polydata.field_data['tree_id'] = np.array([idx])

        tree_branch_polydata.field_data.pop

        # Add the individual tree's branch PolyData to the MultiBlock
        multiblock.append(tree_branch_polydata)
        
        if idx % 100 == 0:
            print(f"processed tree {idx} that is  {tree_branch_polydata.field_data['_Control']},  {tree_branch_polydata.field_data['_Tree_size']}")





    leroux_df = pd.read_csv('data/csvs/lerouxdata-update.csv')

    siteKDTree = cKDTree(site_polydata.points[:, :2])


    #TODO: find max height of tree, and remove any calcluated leaf litter and fallen logs that are higher than this 
    ground_cover_data = {
        "fallen_log": [],
        "leaf_litter": []
        }
    
    canopy_resources = []



    for idx in range(multiblock.n_blocks):
        points = multiblock[idx].points
        point_data = multiblock[idx].point_data
        
        updated_points, updated_point_data, tree_canopy_resources, resourceVoxelCount = update_block(points, site_polydata.points, point_data,  multiblock[idx].field_data, leroux_df, siteKDTree, ground_cover_data)
        
        # Update the multiblock structure with the returned arrays
        multiblock[idx].points = updated_points
        for key, value in updated_point_data.items():
            multiblock[idx].point_data[key] = value
            
        # Update the field data with resource counts
        for resource_name, resource_count in resourceVoxelCount.items():
            multiblock[idx].field_data[resource_name] = resource_count

        canopy_resources.append(tree_canopy_resources)
        
        if idx % 100 == 0:
            print(f'Processing tree number {idx}...')
            print(f'resources are {resourceVoxelCount}')
        
        # Print field data for the current block
        for field_name, field_value in multiblock[idx].field_data.items():
            logging.info(f'{field_name}: {field_value}')
        
    
    #CANOPY RESOURCES
    #print(f'combined canopy resources are {canopy_resources}')

    all_points = []
    all_resource_names = []

    # Iterate over the list of dictionaries
    for resource_dict in canopy_resources:
        for resource_name, point_positions in resource_dict.items():
            if point_positions.shape[0] > 0: # If there are points
                print(f'shape of {resource_name} is {point_positions.shape[0]}')
                all_points.extend(point_positions)
                all_resource_names.extend([resource_name] * point_positions.shape[0])

    # Convert the combined points and resource names to PolyData
    canopy_polydata = pv.PolyData(np.array(all_points))
    canopy_polydata.point_data['canopy resource name'] = all_resource_names

    #GROUND COVER
    # Combine all points and their corresponding resource types from ground_cover_data
    non_empty_values = [value for value in ground_cover_data.values() if len(value) > 0]

    if non_empty_values:
        all_points = np.vstack(non_empty_values)
        resource_types = np.concatenate([[key]*len(value) for key, value in ground_cover_data.items() if len(value) > 0])
    else:
        all_points = np.array([])
        resource_types = np.array([])

    # Create PolyData from the combined points
    ground_cover_polydata = pv.PolyData(all_points)
    ground_cover_polydata.point_data["resource"] = resource_types


    # Initialize an empty dictionary to hold the results
    result_dict = {}

    # Add to result_dict only if PolyData contains points
    if site_polydata.n_points > 0:
        result_dict['site'] = site_polydata
    else:
        print(f"Warning: site_polydata for {site} is empty.")

    if tree_positions.n_points > 0:
        result_dict['tree_positions'] = tree_positions
    else:
        print("Warning: tree_positions is empty.")

    if multiblock.n_blocks > 0:
        result_dict['branches'] = multiblock
    else:
        print("Warning: multiblock is empty.")

    if canopy_polydata.n_points > 0:
        result_dict['canopy resources'] = canopy_polydata
    else:
        print("Warning: canopy_polydata is empty.")

    if ground_cover_polydata.n_points > 0:
        result_dict['ground resources'] = ground_cover_polydata
    else:
        print("Warning: ground_cover_polydata is empty.")

    return result_dict

# Function to add to plotter, shifts only in x-direction
def add_to_plotterOLD(plotter, polydata_dict, shift=None, showSite = False):
    def apply_shift(poly_data):
        if shift is not None:
            poly_data.points += shift  # Shift in x, y, and z directions

    # BRANCHES
    if 'branches' in polydata_dict:
        cube = pv.Cube()
        multiblock = polydata_dict['branches']
        for block in multiblock:
            if block is not None:
                apply_shift(block)  # Apply shift to block
                glyphs = block.glyph(geom=cube, scale=False, orient=False, factor=0.5)
                plotter.add_mesh(glyphs, scalars='Branch.angle', cmap='viridis', show_scalar_bar=False)
    
    # TREE OUTLINES
    if 'branches' in polydata_dict:
        import glyphs as glyphMapper  # Assuming this is a separate module you have
        #glyphMapper.visualise_block_outlines(plotter, polydata_dict['branches'], 'isPrecolonial', 'rainbow')
        glyphMapper.visualise_tree_outlines(plotter, polydata_dict['branches'])

    # CANOPY RESOURCES
    if 'canopy resources' in polydata_dict:
        canopy_polydata = polydata_dict['canopy resources']
        apply_shift(canopy_polydata)  # Apply shift to canopy_polydata
        canopyGlyphs = canopy_polydata.glyph(geom=cube, scale=False, orient=False, factor=5)
        plotter.add_mesh(canopyGlyphs, scalars='canopy resource name', cmap=['magenta', 'cyan'], style='wireframe', line_width=5, show_scalar_bar=False)
        #plotter.add_mesh(canopy_polydata, scalars='canopy resource name', cmap='haline', render_points_as_spheres=True, point_size=20)  # adjust point_size as desired

    # GROUND RESOURCES
    if 'ground resources' in polydata_dict:
        ground_cover_polydata = polydata_dict['ground resources']
        apply_shift(ground_cover_polydata)  # Apply shift to ground_cover_polydata
        cubeGround = pv.Cube().triangulate().subdivide(2).clean().extract_surface()  # Smoother cube
        glyphs = ground_cover_polydata.glyph(geom=cubeGround, scale=False, orient=False, factor=0.25)
        plotter.add_mesh(glyphs, scalars='resource', cmap=['orange', 'purple'], show_scalar_bar=False)

    # SITE
    if 'site' in polydata_dict and showSite == True:
        site_polydata = polydata_dict['site']
        apply_shift(site_polydata)  # Apply shift to site_polydata
        import glyphs as glyphMapper  # Assuming this is a separate module you have
        glyphMapper.add_mesh_rgba(plotter, site_polydata.points, 1.0, site_polydata.point_data["RGB"], rotation=70)




def main(sites, state):

    import packtoMultiblock
    import glyphs as glyphMapper, helper_functions, getBaselines, cameraSetUp
   
    plotter = pv.Plotter()
    box_offset = 125
    gridDist = 400
    for idx, site in enumerate(sites):
        shift =  np.array([400 * idx, 0, 0])
        site_polydata = pv.read(f'data/{site}/flattened-{site}.vtk')
        tree_positions = getTrees(site_polydata, site, state)
        print(f'tree positions are: {tree_positions}')
        data_dict = process_site_data(site_polydata, tree_positions, site, box_offset)
        print(f'data dict at end is {data_dict}')
        
        multiblock = packtoMultiblock.create_or_extend_multiblock(data_dict, new_block_name='trees')
        treeMultiblock = multiblock.get('trees')

        #treeMultiblock.save(f'data/{site}/combined-{site}-{state}.vtm')


        glyphMapper.add_trees_to_plotter(plotter, treeMultiblock, shift, showSite = False)

        #add_to_plotterOLD(plotter, data_dict, shift, showSite = False)

        #baselines
        #baseline_tree_positions, baseline_site_polydata = getBaselines.GetConditions(site)
        #shift[1] += 200
        #baseline_data_dict = process_site_data(baseline_site_polydata, baseline_tree_positions, site, box_offset)
        #add_to_plotter(plotter, baseline_data_dict, shift, showSite = True)
        


    # Additional settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()

    cameraSetUp.setup_camera(plotter, 400, 600)
    plotter.remove_legend()



    plotter.show()



if __name__ == "__main__":
    #sites = ['city', 'street', 'park']  # List of sites
    sites = ['city']
    main(sites, 'present')


