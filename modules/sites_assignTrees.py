import pandas as pd
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import pickle

import pandas as pd
from scipy.spatial import cKDTree
import pyvista as pv
import numpy as np

def load_and_cull_urban_forest(site_polydata, site):
    def get_fill_value(dtype):
        if pd.api.types.is_numeric_dtype(dtype):
            return 0  # Return 0 for numeric data types
        elif pd.api.types.is_string_dtype(dtype):
            return ''  # Return empty string for string data types
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.NaT  # Return NaT (Not a Timestamp) for datetime data types
        elif pd.api.types.is_boolean_dtype(dtype):
            return False  # Return False for boolean data types
        else:
            return None  # Return None for other data types
        
    def transform_and_cull_points(df, easting_offset, northing_offset):
        print(f"Transforming and culling points for {df.shape[0]} rows.")
        # Recenter the coordinates
        points = df[['Easting', 'Northing']].values
        points[:, 0] -= easting_offset
        points[:, 1] -= northing_offset
        
        # Cull points by the x y bounds before performing the KDTree query
        mask = (
            (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
            (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
        )

        
        filtered_points = points[mask]
        df = df[mask]
        
        print(f"After culling by x y bounds, {df.shape[0]} rows remain.")
        
        # Perform the KDTree query to find the z positions
        _, indices = tree.query(filtered_points)
        z_coords = filtered_site_polydata.points[indices, 2]
        filtered_points = np.column_stack([filtered_points, z_coords])
        
        return df, filtered_points

    urban_forest_data = 'data/csvs/trees-with-species-and-dimensions-urban-forest.csv'
    extra_trees_data = 'data/csvs/extra-trees.csv'

    # Load site projections and get offsets
    site_coord = pd.read_csv('data/site projections.csv')
    site_to_search = site.split('-')[1] if '-' in site else site
    easting_offset = site_coord.loc[site_coord['Name'].str.contains(site_to_search, case=False), 'Easting'].values[0]
    northing_offset = site_coord.loc[site_coord['Name'].str.contains(site_to_search, case=False), 'Northing'].values[0]

    print('finding kd trees')

    
    # Filtering the pyvista polydata
    mask = site_polydata.point_data['blocktype'] != 'powerline'
    filtered_points = site_polydata.points[mask]
    filtered_point_data = {key: np.array(val)[mask] for key, val in site_polydata.point_data.items()}
    filtered_site_polydata = pv.PolyData(filtered_points)
    for key, val in filtered_point_data.items():
        filtered_site_polydata.point_data[key] = val

    # Create a KDTree for spatial queries, only once, on the filtered site polydata
    tree = cKDTree(filtered_site_polydata.points[:, :2])
    
    # Get bounds for culling
    min_x, max_x, min_y, max_y = filtered_site_polydata.bounds[0:4]
    
    """min_x, min_y = np.min(filtered_site_polydata.points[:, :2], axis=0)
    max_x, max_y = np.max(filtered_site_polydata.points[:, :2], axis=0)"""

    # Load CSV and other data files
    df_urban = pd.read_csv(urban_forest_data)
    print(f'tree df immediately after loading \n {df_urban}')

    

    # Specify the columns to be imported
    columns_to_import = ['X', 'Y', 'evolved', 'size', 'isDevelop', 'area']

    # Load the data
    extra_trees_data = 'data/csvs/extra-trees.csv'
    df_extra = pd.read_csv(extra_trees_data, usecols=columns_to_import, dtype={'isDevelop': 'bool'})

    # Rename and transform columns for df_extra
    df_extra.rename(columns={'X': 'Easting', 'Y': 'Northing', 'evolved': 'isPrecolonial', 'size': '_Tree_size', 'area': 'Located in'}, inplace=True)
    df_extra['isPrecolonial'] = df_extra['isPrecolonial'].astype(str).str.lower() == 'true'
    
    # Create Diameter Breast Height column based on _Tree_size
    size_to_dbh = {'small': 10, 'medium': 50, 'large': 80}
    size_to_life = {'small': 100, 'medium': 40, 'large': 10}
    
    df_extra['Diameter Breast Height'] = df_extra['_Tree_size'].map(size_to_dbh)
    df_extra['Useful Life Expectency Value'] = df_extra['_Tree_size'].map(size_to_life)

    df_extra = df_extra.drop(columns=['_Tree_size'])


    print(f'tree df before cull \n {df_urban}')


    df_urban, points_urban = transform_and_cull_points(df_urban, easting_offset, northing_offset)
    df_extra, points_extra = transform_and_cull_points(df_extra, easting_offset, northing_offset)


    print(f'tree df after cull \n {df_urban}')



    # Assign a blockID to each tree coordinate based on the index
    df_urban['blockID'] = df_urban.index
    df_extra['blockID'] = df_extra.index + len(df_urban)  # Continue the index from where df_urban left off
    df_urban['isPrecolonial'] = False

    # Assign X, Y, Z coordinates
    df_urban['X'] = points_urban[:, 0]
    df_urban['Y'] = points_urban[:, 1]
    df_urban['Z'] = points_urban[:, 2]

    df_extra['X'] = points_extra[:, 0]
    df_extra['Y'] = points_extra[:, 1]
    df_extra['Z'] = points_extra[:, 2]

  
    # Mark the source of the data
    df_urban['extraTree'] = False
    df_urban['isDevelop'] = False
    df_extra['extraTree'] = True


    df_extra['Located in'] = df_extra['Located in'].str.capitalize()

    # Function to determine the appropriate fill value based on the data type
    def get_fill_value(dtype):
        if pd.api.types.is_numeric_dtype(dtype):
            return -1  # Return -1 for numeric data types
        elif pd.api.types.is_string_dtype(dtype):
            return 'default'  # Return '-1' for string data types
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.NaT  # Return NaT (Not a Timestamp) for datetime data types
        elif pd.api.types.is_boolean_dtype(dtype):
            return False  # Return False for boolean data types
        else:
            return 'default'

    # Step 1: Identify missing columns
    missing_columns_extra = set(df_urban.columns) - set(df_extra.columns)

    # Step 2: Add missing columns with appropriate default values
    for col in missing_columns_extra:
        default_value = get_fill_value(df_urban[col].dtype)
        df_extra[col] = default_value

    # Step 3: Merge the DataFrames
    df_combined = pd.concat([df_urban, df_extra], ignore_index=True)

    print(f"After merging, the combined DataFrame has {df_combined.shape[0]} rows.")

    # Reset the index of DataFrame df_combined to start at 0
    df_combined.reset_index(drop=True, inplace=True)

    # Convert each 'object' type column to string type
    df_combined = df_combined.convert_dtypes()

    # Get the list of columns that have data type 'object'
    object_cols = df_combined.select_dtypes(include='object').columns

    # Convert each 'object' type column to string type
    df_combined[object_cols] = df_combined[object_cols].astype(str)

    # Print the data types of the final produced columns of branch_data
    for col, dtype in df_combined.dtypes.items():
        print(f" - Column '{col}' has data type: {dtype}")


    print(f'tree df \n {df_urban}')
    print(f'extra df \n {df_extra}')
    print(f'combined df \n {df_combined}')

    return df_combined


def determine_precolonial_status(df, site, adjustPrecolonial=False, adjustedPreColonialValue=0.2):
    """
    Determine if trees are precolonial based on their Scientific Name and Genus.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing tree information.
    site (str): The site identifier.
    adjustPrecolonial (bool): Whether to adjust the precolonial status synthetically.
    adjustedPreColonialValue (float): The proportion of trees to adjust to precolonial.

    Returns:
    df (pd.DataFrame): The DataFrame with an updated 'isPrecolonial' column.
    """
    precolonial_species_list = pd.read_csv('data/csvs/pre-colonial-plant-list.csv')

    mask = df['extraTree'] == False

    # Create a new mask for the condition on Scientific Name
    name_mask = df.loc[mask, 'Scientific Name'].isin(precolonial_species_list['Species'])

    # Use this mask to set the values in 'isPrecolonial' for the rows specified by the original mask
    df.loc[mask, 'isPrecolonial'] = name_mask


    # Determine precolonial status based on Scientific Name and Genus, only for rows where extraTree == False
    #df.loc[mask, 'isPrecolonial'] = df.loc[mask, 'Scientific Name'].isin(precolonial_species_list['Species'])

    
    #other condition
    df.loc[mask & (df['Genus'] == 'Eucalyptus'), 'isPrecolonial'] = True


    # Optionally adjust precolonial status synthetically for certain sites
    if adjustPrecolonial:
        false_indices = df[df['isPrecolonial'] == False].index.tolist()
        num_to_adjust = int(adjustedPreColonialValue * len(false_indices))
        selected_indices = np.random.choice(false_indices, num_to_adjust, replace=False)
        df.loc[selected_indices, 'isPrecolonial'] = True

    return df


def adjustState(df: pd.DataFrame, site: str, state: str) -> pd.DataFrame:
    if state == 'trending':
        
        # Create two separate masks
        mask_life_expectancy = df['Useful Life Expectency Value'] < 20
        mask_is_develop = df['isDevelop'] == True
        
        # Logging the number of true values in each mask
        print(f"Adjusting DBH, size, precolnial state of trees with life expectancy under 20: {mask_life_expectancy.sum()}")
        print(f"Removing this many trees where isDevelop is True: {mask_is_develop.sum()}")

        df = df[~mask_is_develop]
        df.reset_index(drop=True, inplace=True)
        
        
        df.loc[mask_life_expectancy, 'Diameter Breast Height'] = 10  # Update DBH
        df.loc[mask_life_expectancy, 'Genus'] = 'Eucalyptus'  # Update genus
        df.loc[mask_life_expectancy, 'isPreColonial'] = 'yes'  # Update isPreColonial
        df.loc[mask_life_expectancy, '_Tree_size'] = 'small'
        
        # Creating a new DataFrame with only the masked rows and saving the masked DataFrame to a CSV file
        mask = mask_life_expectancy | mask_is_develop
        masked_df = df[mask]
        masked_df.to_csv(f'outputs/changedTrees-{site}-{state}.csv', index=False)
        
        
        # Logging the updates
        print(f"Updated DBH, Genus, and isPreColonial or removed the trees for {mask.sum()} trees.")
        
    elif state == 'preferable':
        print('state is preferable, improving trees...')
        df['isPrecolonial'] = True 
        df['_Control'] = 'reserve-tree'
        df['_Tree_size'] = 'large'
    
    else:
        print("State is not 'trending', no updates made to DataFrame.")

    
    return df

def assign_control_and_size(df: pd.DataFrame) -> pd.DataFrame:
    # Assigning control labels based on the 'Located in' column
    df.loc[df['Located in'] == 'Street', '_Control'] = 'street-tree'
    df.loc[df['Located in'] == 'Park', '_Control'] = 'park-tree'
    df.loc[df['Located in'] == 'Reserve', '_Control'] = 'reserve-tree'

    #for the default trees
    df.loc[df['Located in'] == 'default', '_Control'] = 'street-tree'
    
    # Logging the control label updates
    street_tree_count = (df['_Control'] == 'street-tree').sum()
    park_tree_count = (df['_Control'] == 'park-tree').sum()
    print(f"Updated _Control column: {street_tree_count} street-trees, {park_tree_count} park-trees.")
    
    # Assigning tree sizes based on the 'Diameter Breast Height' column
    df['_Tree_size'] = df['Diameter Breast Height'].apply(
        lambda x: 'medium' if pd.isna(x) else ('small' if x < 50 else ('medium' if 50 <= x < 80 else 'large'))
    )


    
    
    # Logging the size assignment
    small_tree_count = (df['_Tree_size'] == 'small').sum()
    medium_tree_count = (df['_Tree_size'] == 'medium').sum()
    large_tree_count = (df['_Tree_size'] == 'large').sum()
    print(f"Updated _Tree_size column: {small_tree_count} small trees, {medium_tree_count} medium trees, {large_tree_count} large trees.")
    return df

def assign_tree_model_id(df):
    # Create a dictionary mapping tree sizes to ranges of model IDs
    model_id_ranges = {
        'small': np.arange(1, 5),
        'medium': np.arange(5, 10),
        'large': np.arange(10, 17)
    }
    
    # Use np.random.choice to assign a random model ID from the appropriate range to each tree
    df['TreeModelID'] = df.apply(
        lambda row: np.random.choice(model_id_ranges[row['_Tree_size']]), axis=1
    )
    
    return df

def check_datatype(data_dict):
    for key, df in data_dict.items():
        print(f"Checking data types for key: {key}")
        for column in df.columns:
            print(f" - Column '{column}' has data type: {df[column].dtype}")


def create_canopy_array(site, state, processed_tree_coords, isDeployable = False):
    print("Loading prebaked branches dictionary...")
    with open('data/prebaked-branches.pkl', 'rb') as f:
        categorized_tree_samples = pickle.load(f)

    print("Successfully loaded prebaked branches dictionary.")

    with open('data/prebaked-tree-resources.pkl', 'rb') as f:
        tree_level_resource_dict = pickle.load(f)

    print("Successfully loaded prebaked tree resources dictionary.")

    original_dtypes = processed_tree_coords.dtypes.to_dict()

    # Initialize empty lists to store canopy and ground resource data
    canopy_resources_list = []
    #ground_resources_list = []

    ground_resources_data = {
        'resource': [],
        'blockID': [],
        'TreeX': [],
        'TreeY': [],
        'TreeZ': [],
        '_Tree_size': [],
        'isPrecolonial': [],
        '_Control': [],
        'TreeModelID': [],
        'X': [],
        'Y': [],
        'Z': []
    }

    def lookup_attributes(treeInfo):
        #GET BRANCHES
        key = (treeInfo['_Tree_size'], treeInfo['isPrecolonial'], treeInfo['_Control'], treeInfo['TreeModelID'])
        #print(f'Key: {key}')  # Debugging line
        #print(f'lookups are {key}')
        if key not in categorized_tree_samples:
            print(f"Missing key: {key}")

        branch_data = categorized_tree_samples.get(key, np.nan).copy(deep=True)


        base_coord = treeInfo[['X', 'Y', 'Z']].values
        translated_coords = branch_data[['x', 'y', 'z']].values + base_coord
        branch_data['BranchX'], branch_data['BranchY'], branch_data['BranchZ'] = translated_coords.T
        
        branch_data[['ScanX', 'ScanY', 'ScanZ']] = branch_data[['x', 'y', 'z']]
        branch_data['elevation'] = branch_data['y']
        branch_data['blockID'] = treeInfo['blockID']
        branch_data['extraTree'] = treeInfo['extraTree']
        branch_data['TreeX'], branch_data['TreeY'], branch_data['TreeZ'] = base_coord
        #print(f'BlockID: {treeInfo["blockID"]}')  # Debugging line


        # Adding these lines to populate the new columns
        branch_data['_Tree_size'] = key[0]  # Tree size
        branch_data['isPrecolonial'] = key[1]  # Precolonial status
        branch_data['_Control'] = key[2]  # Control type
        branch_data['TreeModelID'] = key[3]  # Tree I
        


        #GET CANOPY AND GROUND RESOURCES
        otherResourceKey = (treeInfo['_Tree_size'], treeInfo['isPrecolonial'], treeInfo['_Control'])

        if otherResourceKey not in tree_level_resource_dict:
            print(f"Missing key: {otherResourceKey}")
            return None  # Return early if key is missing
        
        resources_dict = tree_level_resource_dict[otherResourceKey]

        # Handle canopy resources
        for canopy_resource_type in ['epiphyte', 'hollow']:
            quantity = resources_dict[canopy_resource_type][1]
            whole_part = int(quantity)
            fractional_part = quantity % 1

            #print(f'{whole_part} whole and {fractional_part} fractional resources of {canopy_resource_type}')

            for _ in range(whole_part):
                canopy_resource = {
                    'resource': canopy_resource_type,
                    'blockID': treeInfo['blockID'],
                    'X': base_coord[0],
                    'Y': base_coord[1],
                    'Z': base_coord[2] + 40,
                    'TreeX': base_coord[0],
                    'TreeY': base_coord[1],
                    'TreeZ': base_coord[2],
                    '_Tree_size': key[0],
                    'isPrecolonial': key[1],
                    '_Control': key[2],
                    'TreeModelID': key[3]
                }
                canopy_resources_list.append(canopy_resource)

            # Handling the fractional part
            random_number = np.random.random()
            if random_number < fractional_part:
                canopy_resource = {
                    'resource': canopy_resource_type,
                    'blockID': treeInfo['blockID'],
                    'X': base_coord[0],
                    'Y': base_coord[1],
                    'Z': base_coord[2] + 40,
                    'TreeX': base_coord[0],
                    'TreeY': base_coord[1],
                    'TreeZ': base_coord[2],
                    '_Tree_size': key[0],
                    'isPrecolonial': key[1],
                    '_Control': key[2],
                    'TreeModelID': key[3]
                }
                canopy_resources_list.append(canopy_resource)
                #print(f'Added an additional {canopy_resource_type} to canopy_resources_list')
        
        # Handle ground resources
        for ground_resource_type in ['fallen log', 'leaf litter']:
            points = resources_dict[ground_resource_type]

            """for point in points:
                ground_resource = {
                    'resource': ground_resource_type,
                    'blockID': treeInfo['blockID'],
                    'TreeX': base_coord[0],
                    'TreeY': base_coord[1],
                    'TreeZ': base_coord[2],
                    '_Tree_size': key[0],
                    'isPrecolonial': key[1],
                    '_Control': key[2],
                    'TreeModelID': key[3],
                    'X': point[0] + base_coord[0],
                    'Y': point[1] + base_coord[1],
                    'Z': point[2] + base_coord[2]
                }
                ground_resources_list.append(ground_resource)"""

            n_points = len(points)
            if n_points > 0:
                if n_points > 0:
                    ground_resources_data['resource'].extend([ground_resource_type] * n_points)
                    ground_resources_data['blockID'].extend([treeInfo['blockID']] * n_points)
                    ground_resources_data['TreeX'].extend([base_coord[0]] * n_points)
                    ground_resources_data['TreeY'].extend([base_coord[1]] * n_points)
                    ground_resources_data['TreeZ'].extend([base_coord[2]] * n_points)
                    ground_resources_data['_Tree_size'].extend([key[0]] * n_points)
                    ground_resources_data['isPrecolonial'].extend([key[1]] * n_points)
                    ground_resources_data['_Control'].extend([key[2]] * n_points)
                    ground_resources_data['TreeModelID'].extend([key[3]] * n_points)
                    ground_resources_data['X'].extend(points[:, 0] + base_coord[0])
                    ground_resources_data['Y'].extend(points[:, 1] + base_coord[1])
                    #ground_resources_data['Z'].extend(points[:, 2] + base_coord[2])
                    ground_resources_data['Z'].extend([base_coord[2]] * n_points)


        return branch_data

    print("Starting vectorized lookup and assignment...")
    canopy_data_series = processed_tree_coords.apply(lookup_attributes, axis=1)
    print("Vectorized lookup and assignment completed.")




    print("Converting list of DataFrames to a single DataFrame...")

    
    # This line will concatenate all the individual DataFrames in canopy_data_list into a single DataFrame
    branch_data_df = pd.concat(canopy_data_series.values, ignore_index=True)

    # Once all trees have been processed, convert the lists to DataFrames
    canopy_resources_df = pd.DataFrame(canopy_resources_list)

    ground_resources_df = pd.DataFrame(ground_resources_data)

    print(f'ground resources are:\n{ground_resources_df}')
    print(f'canopy resources are:\n{canopy_resources_df}')


    #asign correct types
    original_dtypes.update({
            'BranchX': 'float64',
            'BranchY': 'float64',
            'BranchZ': 'float64',
            'ScanX': 'float64',
            'ScanY': 'float64',
            'ScanZ': 'float64'
        })

    print(f'original_dtypes are {original_dtypes}')
    print(branch_data_df)


    for col, dtype in original_dtypes.items():
        if col in branch_data_df.columns:
            branch_data_df[col] = branch_data_df[col].astype(dtype)
        else:
            print(f"Column {col} not found in canopy_data_df.")



    # List of columns you want first
    cols_to_move = ['BranchX', 'BranchY', 'BranchZ']

    # Other columns
    remaining_cols = [col for col in branch_data_df.columns if col not in cols_to_move]

    # Reorder columns
    branch_data_df = pd.concat([branch_data_df[cols_to_move], branch_data_df[remaining_cols]], axis=1)


        
    """print("The final branch DataFrame:")
    print(branch_data_df)

    print("The final ground DataFrame:")
    print(ground_resources_df)

    print("The final canopy resource DataFrame:")
    print(canopy_resources_df)"""

    print(f'created branch df of length {len(branch_data_df)}')
    print(f'created ground df of length {len(ground_resources_df)}')
    print(f'created canopy resource df of length {len(canopy_resources_df)}')



    return branch_data_df, ground_resources_df, canopy_resources_df

def paint_resources(polydata, resources_df, mask=None, use_xyz=False, jitter=0):
    print("Starting paint_resources function...")

    # Prepare a mask to filter points from the polydata
    if mask is not None:
        print(f"Applying mask: {mask}")
        print(f'Number of points before mask: {polydata.n_points}')



        boolean_mask = np.full(polydata.n_points, False, dtype=bool)  # Start with all False
        for field, values in mask.items():
            field_data = polydata.point_data[field]
            print(f'Unique values for {field}: {np.unique(field_data)}')  # Prints unique values of the mask key
            field_mask = np.isin(field_data, values)
            boolean_mask = np.logical_or(boolean_mask, field_mask)  # Combine with OR
        
        polydata = polydata.extract_points(boolean_mask)
        print(f"Number of points after mask applied: {polydata.n_points}")
    
    # Prepare a k-d tree for nearest neighbor search
    print("Preparing k-d tree for nearest neighbor search...")
    kd_tree_dimension = 3 if use_xyz else 2
    kd_tree = cKDTree(polydata.points[:, :kd_tree_dimension])
    
    # Prepare the resources DataFrame for updating
    print("Preparing resources DataFrame for updating...")
    updated_resources_df = resources_df.copy()
    
    # If jitter is specified, generate random offsets for x and y coordinates
    if jitter != 0:
        xy_jitter = np.random.normal(scale=jitter, size=(len(resources_df), 2))
        updated_resources_df[['X', 'Y']] += xy_jitter
    
    # Prepare the array of query points
    print("Preparing array of query points...")
    query_points = updated_resources_df[['X', 'Y']].values  # 2D query points
    if use_xyz:
        query_points = np.hstack((query_points, updated_resources_df[['Z']].values))  # 3D query points
    
    # Find the nearest points in the polydata
    print("Finding nearest points in the polydata...")
    _, indices = kd_tree.query(query_points)
    
    # Update the z coordinates in the resources DataFrame
    print("Updating z coordinates in the resources DataFrame...")
    updated_resources_df['Z'] = polydata.points[indices][:, 2]
    
    # If use_xyz is True, also update the x and y coordinates
    if use_xyz:
        print("Updating x and y coordinates in the resources DataFrame...")
        updated_resources_df[['X', 'Y']] = polydata.points[indices][:, :2]
    
    print("Finished paint_resources function.")
    return updated_resources_df

   



def process_urban_forest(site_df, site, state='present'):
    # Step 1: Load and Cull aging done, moved on to fallen Data
    print("Step 1: Loading and Culling aging done, moved on to fallen Data...")
    tree_coords_df = load_and_cull_urban_forest(site_df, site)
    print(f"Number of tree coordinates obtained: {len(tree_coords_df)}\n")

    # Step 2: Determine Precolonial Status
    print("Step 2: Determining Precolonial Status...")
    tree_coords_df = determine_precolonial_status(tree_coords_df, site)
    precolonial_count = tree_coords_df['isPrecolonial'].sum()
    print(f"Number of precolonial trees: {precolonial_count}\n")

    # Step 3: Assign Control Label
    print("Step 3: Assigning Control Label...")
    tree_coords_df = assign_control_and_size(tree_coords_df)
    # Assuming assign_control_label function logs its own output

    # Step 4: Adjust State
    print("Step 4: Adjusting State...")
    tree_coords_df = adjustState(tree_coords_df, site, state)
    # Assuming adjustState function logs its own output

    print(f'unique values for control in tree df are {tree_coords_df["_Control"].value_counts()}')
    print(f'unique values for size in tree df are {tree_coords_df["_Tree_size"].value_counts()}')
    print(f'unique values for precolonial in tree df are {tree_coords_df["isPrecolonial"].value_counts()}')

    # Step 5: Assign Tree Model ID
    print("Step 5: Assigning Tree Model ID...")
    tree_coords_df = assign_tree_model_id(tree_coords_df)
    print(f"Tree Model IDs assigned.\n")

    # Step 6: Display Summary
    print(f"Step 6: Summary of {len(tree_coords_df)} trees")
    unique_species = tree_coords_df['Common Name'].unique()
    unique_sizes = tree_coords_df['_Tree_size'].unique()
    unique_controls = tree_coords_df['_Control'].unique()
    unique_precolonials = tree_coords_df['isPrecolonial'].unique()  # Assuming 'isPrecolonial' is the column name for precolonial statuses

    for index, item in enumerate(unique_species):
        if not isinstance(item, str):
            print(f"Item at index {index} is not a string: {item} (type: {type(item)})")

    
    print(f"Unique species: {len(unique_species)}, Names: {', '.join(unique_species)}")
    print(f"Unique sizes: {len(unique_sizes)}, Names: {', '.join(unique_sizes)}")
    print(f"Unique controls: {unique_controls}")
    print(f"Unique control levels: {len(unique_controls)}, Names: {', '.join(unique_controls)}")
    print(f"Unique precolonial statuses: {len(unique_precolonials)}, Values: {', '.join(map(str, unique_precolonials))}")
    print(f"Unique blockIDS: {len(tree_coords_df['blockID'])}")
    return tree_coords_df

def create_canopy_dict2(sitePolyData, branchdf, groundresourcesdf, canopyresourcesdf):
    points = branchdf[['BranchX', 'BranchY', 'BranchZ']].values
    #print(f'Type of points: {type(points)}')  # This will print the type of 'points'


    #BRANCHES
    # Create a single PolyData point cloud
    branch_polydata = pv.PolyData(points)

    # Transfer the rest of the columns from the DataFrame to the PolyData point_data
    for col in branchdf.columns:
        if col not in ['BranchX', 'BranchY', 'BranchZ']:  # Skip the 'X', 'Y', 'Z' columns since they're already used for the points
            sanitized_data = []
            for val in branchdf[col]:
                sanitized_data.append(val.encode('ascii', 'ignore').decode() if isinstance(val, str) else val)
            branch_polydata.point_data[col] = sanitized_data

    
    # CANOPY RESOURCES
    canopyresourcesdf = paint_resources(branch_polydata, canopyresourcesdf, use_xyz=True, jitter=10)
    points = canopyresourcesdf[['X', 'Y', 'Z']].values
    
    print('after jitter')
    print(canopyresourcesdf[['X', 'Y']].head())  # Print values after

    print('canopy resources dataframe looks like')

    # Assuming there's a column called 'resource_type' in canopyresourcesdf which indicates the type of each resource
    hollows_mask = canopyresourcesdf['resource'] == 'hollow'
    hollows_df = canopyresourcesdf.loc[hollows_mask]
    print(f'hollow df looks like')
    print(hollows_df)

    epiphytes_mask = canopyresourcesdf['resource'] == 'epiphyte'
    epiphytes_df = canopyresourcesdf.loc[epiphytes_mask]
    print(f'epiphytes df looks like')
    print(epiphytes_df)



    print(canopyresourcesdf)

    
    canopy_resource_polydata = pv.PolyData(points)

    # Transfer the rest of the columns from the DataFrame to the PolyData point_data
    for col in canopyresourcesdf.columns:
        if col not in ['X', 'Y', 'Z']:  # Skip the 'X', 'Y', 'Z' columns since they're already used for the points
            sanitized_data = []
            for val in canopyresourcesdf[col]:
                sanitized_data.append(val.encode('ascii', 'ignore').decode() if isinstance(val, str) else val)
            canopy_resource_polydata.point_data[col] = sanitized_data


    #GROUND RESOURCES
    groundMask = {'blocktype' : ['road_types', 'topo']}

    groundresourcesdf = paint_resources(sitePolyData, groundresourcesdf, mask=groundMask)

    groundresourcesdf = paint_resources(sitePolyData, groundresourcesdf, mask=groundMask)
    
    # Create a single PolyData point cloud
    points = groundresourcesdf[['X', 'Y', 'Z']].values
    ground_resource_polydata = pv.PolyData(points)

    # Transfer the rest of the columns from the DataFrame to the PolyData point_data
    for col in groundresourcesdf.columns:
        if col not in ['X', 'Y', 'Z']:  # Skip the 'X', 'Y', 'Z' columns since they're already used for the points
            sanitized_data = []
            for val in groundresourcesdf[col]:
                sanitized_data.append(val.encode('ascii', 'ignore').decode() if isinstance(val, str) else val)
            ground_resource_polydata.point_data[col] = sanitized_data

    tree_dict = {
        'branches' : branch_polydata,
        'canopy resources' : canopy_resource_polydata,
        'ground resources' : ground_resource_polydata}

    return tree_dict

def create_canopy_dict(sitePolyData, branchdf, groundresourcesdf, canopyresourcesdf):
    points = branchdf[['BranchX', 'BranchY', 'BranchZ']].values
    #print(f'Type of points: {type(points)}')  # This will print the type of 'points'


    #BRANCHES
    # Create a single PolyData point cloud
    branch_polydata = pv.PolyData(points)

    # Transfer the rest of the columns from the DataFrame to the PolyData point_data
    for col in branchdf.columns:
        if col not in ['BranchX', 'BranchY', 'BranchZ']:  # Skip the 'X', 'Y', 'Z' columns since they're already used for the points
            sanitized_data = []
            for val in branchdf[col]:
                sanitized_data.append(val.encode('ascii', 'ignore').decode() if isinstance(val, str) else val)
            branch_polydata.point_data[col] = sanitized_data

    
    # CANOPY RESOURCES
    if len(canopyresourcesdf) > 0:
        canopyresourcesdf = paint_resources(branch_polydata, canopyresourcesdf, use_xyz=True, jitter=10)
        points = canopyresourcesdf[['X', 'Y', 'Z']].values
        canopy_resource_polydata = pv.PolyData(points)

        # Transfer the rest of the columns from the DataFrame to the PolyData point_data
        for col in canopyresourcesdf.columns:
            if col not in ['X', 'Y', 'Z']:  # Skip the 'X', 'Y', 'Z' columns since they're already used for the points
                sanitized_data = []
                for val in canopyresourcesdf[col]:
                    sanitized_data.append(val.encode('ascii', 'ignore').decode() if isinstance(val, str) else val)
                canopy_resource_polydata.point_data[col] = sanitized_data
    else:
        canopy_resource_polydata = pv.PolyData()

    # GROUND RESOURCES
    if len(groundresourcesdf) > 0:
        groundMask = {'blocktype': ['road_types', 'topo']}
        groundresourcesdf = paint_resources(sitePolyData, groundresourcesdf, mask=groundMask)
        points = groundresourcesdf[['X', 'Y', 'Z']].values
        ground_resource_polydata = pv.PolyData(points)

        # Transfer the rest of the columns from the DataFrame to the PolyData point_data
        for col in groundresourcesdf.columns:
            if col not in ['X', 'Y', 'Z']:  # Skip the 'X', 'Y', 'Z' columns since they're already used for the points
                sanitized_data = []
                for val in groundresourcesdf[col]:
                    sanitized_data.append(val.encode('ascii', 'ignore').decode() if isinstance(val, str) else val)
                ground_resource_polydata.point_data[col] = sanitized_data
    else:
        ground_resource_polydata = pv.PolyData()


    tree_dict = {
        'branches' : branch_polydata,
        'canopy resources' : canopy_resource_polydata,
        'ground resources' : ground_resource_polydata}

    return tree_dict


def main(site, state):
    # Load the site polydata from a file
    site_polydata = pv.read(f'data/{site}/flattened-{site}.vtk')
    print(f'Site polydata for {site} loaded successfully.')

    # Call process_urban_forest function with the loaded site_polydata, site, and state arguments
    processed_tree_coords = process_urban_forest(site_polydata, site, state)
    print(f'Processing completed for site: {site}, state: {state}.')

    canopydf = create_canopy_array(site, state, processed_tree_coords)
    print(f'created canopy df of length {len(canopydf)}')


    unique_block_ids_count = canopydf['blockID'].nunique()
    print(f'The number of unique BlockIDs is: {unique_block_ids_count}')

    canopyDict = create_canopy_dict(canopydf)
    print(f'created canopy dict')

    canopy_polydata = canopyDict['branches']


    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Adding the canopy data to the plotter
    plotter.add_mesh(canopy_polydata, color="green", point_size=5, render_points_as_spheres=True)

    # Adding the site data as a point cloud to the plotter
    plotter.add_mesh(site_polydata, color="blue", point_size=5, render_points_as_spheres=True, label="Site")


    # Show the plotter
    plotter.show()

    # Optionally, you could save the processed_tree_coords DataFrame to a file or perform further analysis
    # For example:
    # processed_tree_coords.to_csv(f'processed_tree_coords_{site}_{state}.csv', index=False)

if __name__ == "__main__":
    #sites = ['city', 'street', 'park']  # List of sites
    main('trimmed-parade', 'present')
    #main('parade', 'present')
