# Assume tree_templates is already loaded in RAM as a dictionary
# processedDF is the dataframe containing tree instances (e.g., x, y, z values to translate)

import dask
from dask import delayed
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import pickle
import pyvista as pv
import pandas as pd
import random
import numpy as np
from pathlib import Path


# Set a random seed for reproducibility
random.seed(42)


# Assume tree_templates is already loaded in RAM as a dictionary
# processedDF is the dataframe containing tree instances (e.g., x, y, z values to translate)

def create_pyvista_object(combined_df):
    """
    Converts the processed DataFrame into a PyVista PolyData object.
    """
    print('##########################')
    print(f'Checking if correct combined_df variable: {combined_df.head()}')
    
    # Ensure combined_df is a pandas DataFrame and not a tuple
    print(f"Type of combined_df: {type(combined_df)}")
    
    if not isinstance(combined_df, pd.DataFrame):
        raise ValueError("combined_df is not a pandas DataFrame")

    # Extract the X, Y, Z columns as points
    points = combined_df[['x', 'y', 'z']].to_numpy()

    # Create the PolyData object with points
    polydata = pv.PolyData(points) 

    # Add other columns as point data
    for col in combined_df.columns:
        if col not in ['x', 'y', 'z']:
            polydata.point_data[col] = combined_df[col].to_numpy()

    return polydata

def convert_key_to_tuple(row):
    """
    Converts a string representation of a tuple into an actual tuple.
    This assumes the new key format (isPreColonial, size, control, treeID).
    """
    if isinstance(row['tree_template_key'], str):
        # Assume the string is formatted as "(isPreColonial, size, control, treeID)"
        return eval(row['tree_template_key'])  # Convert the string back to a tuple
    return row['tree_template_key']


###LOG ASSIGNMENT FUNCTIONS###
def preprocess_logLibrary():
    #Step 0: get log library
    logLibrary = pd.read_pickle('data/treeOutputs/logLibrary.pkl')

    print(f'first bit of log library is {logLibrary.head()}')
    # Group by logNo and count rows
    log_counts = logLibrary.groupby('logNo').size()
    
    # Filter to keep only logs with >= 200 points
    valid_logs = log_counts[log_counts >= 200].index
    
    # Filter logLibrary to only include valid logs
    logLibrary = logLibrary[logLibrary['logNo'].isin(valid_logs)]
    
    print(f'After filtering, {len(valid_logs)} logs remain')

    return logLibrary

def preprocess_logLocationsDF(logLocationsDF, logLibraryDF, seed = 42):
    #initialise column in logLocationsDF called logModel
    #create a logInfo csv that is the grouped logLibraryDF by logNo, with the first row of each group as the row
    #assign each row in logLocationsDF a logNo from logInfo, 
    #make sure logLocationsDF[logSize] matches logInfo[logSize]
    #otherwise, choose at random using the seed




def initialise_and_translate_tree(tree_template, row):
    """
    Translates the tree template by the given x, y, z offsets and initializes additional columns.
    """
    tree_template_copy = tree_template.copy()

    # Translate the X, Y, Z coordinates by offsets from the row
    tree_template_copy['x'] += row['x']
    tree_template_copy['y'] += row['y']
    tree_template_copy['z'] += row['z']

    # Initialize additional columns: precolonial, size, control, tree_id, useful_life_expectency
    tree_template_copy['precolonial'] = row['precolonial']
    tree_template_copy['size'] = row['size']
    tree_template_copy['control'] = row['control']
    tree_template_copy['tree_id'] = row['tree_id']
    tree_template_copy['diameter_breast_height'] = row['diameter_breast_height']
    tree_template_copy['tree_number'] = row['tree_number']
    tree_template_copy['NodeID'] = row['NodeID']
    tree_template_copy['structureID'] = row['structureID']
    tree_template_copy['useful_life_expectancy'] = row['useful_life_expectancy']

    if 'isNewTree' in row:
        tree_template_copy['isNewTree'] = row['isNewTree']


    return tree_template_copy

def update_old_template(tree_templateDF):
    # Find the unique values for tree_template['resource']
    unique_resources = tree_templateDF['resource'].unique()
    print(f'Found {len(unique_resources)} unique resources: {unique_resources}')

    # For each resource in unique resources
    for resource in unique_resources:
        # Create column name
        col_name = f'treeResource_{resource}'
        print(f'Creating column {col_name}')
        
        # Initialize column as False
        tree_templateDF[col_name] = False
        
        # Get mask where resource matches
        mask = tree_templateDF['resource'] == resource
        
        # Update column to True where mask matches
        tree_templateDF.loc[mask, col_name] = True
        print(f'Updated {mask.sum()} rows for {resource}')
    
    print('Finished updating template with resource columns')
    return tree_templateDF

def convert_dict_to_dataframe(tree_templates, updated_elm_tree_templates):
    """
    Converts the dictionary of (tuple: template) to a DataFrame for easier querying.
    """
    rows = []
    for key, template in tree_templates.items():
        precolonial, size, control, tree_id = key
        
        if not precolonial and size not in ['snag', 'fallen','propped']:
            print(f"\nSearching for template with: precolonial={precolonial}, size={size}, control={control}, tree_id={tree_id}")
            
            # Start with full dataset and progressively filter
            current_matches = updated_elm_tree_templates.copy()
            
            # Check precolonial
            if len(current_matches[current_matches['precolonial'] == precolonial]) > 0:
                print(f"✓ Precolonial={precolonial} matches")
                current_matches = current_matches[current_matches['precolonial'] == precolonial]
            else:
                print(f"✗ Precolonial={precolonial} not found. Available values in dataset: {current_matches['precolonial'].unique()}")
                raise ValueError("No matching precolonial value")
            
            # Check size
            if len(current_matches[current_matches['size'] == size]) > 0:
                print(f"✓ Size={size} matches")
                current_matches = current_matches[current_matches['size'] == size]
            else:
                print(f"✗ Size={size} not found. Available sizes for current matches: {current_matches['size'].unique()}")
                raise ValueError("No matching size")
            
            # Handle improved-tree conversion
            if control == 'improved-tree':
                control = 'reserve-tree'
                print("Converting 'improved-tree' to 'reserve-tree'")

            
            # Check control
            if len(current_matches[current_matches['control'] == control]) > 0:
                print(f"✓ Control={control} matches")
                current_matches = current_matches[current_matches['control'] == control]
            else:
                print(f"✗ Control={control} not found. Available controls for current matches: {current_matches['control'].unique()}")
                raise ValueError("No matching control")
            
            # Check tree_id
            if len(current_matches[current_matches['tree_id'] == tree_id]) > 0:
                print(f"✓ TreeID={tree_id} matches")
                current_matches = current_matches[current_matches['tree_id'] == tree_id]
            else:
                print(f"✗ TreeID={tree_id} not found. Available TreeIDs for current matches: {current_matches['tree_id'].unique()}")
                # Instead of raising error, use the first available tree_id
                replacement_tree_id = current_matches['tree_id'].iloc[0]
                print(f"Using replacement TreeID={replacement_tree_id}")
                current_matches = current_matches[current_matches['tree_id'] == replacement_tree_id]
            
            template = current_matches.iloc[0]['template']
            print("Successfully found matching template")
        
        rows.append({'precolonial': precolonial, 'size': size, 'control': control, 'tree_id': tree_id, 'template': template})
    
    df = pd.DataFrame(rows)
    return df

def query_tree_template(df, precolonial, size, control, tree_id, rng):
    """
    Attempts to query the tree template using fallbacks if necessary.
    Logs when fallback steps are used. Uses an external random number generator (rng)
    to ensure the same pattern across runs but different samples per call.
    """
    # 1. Try to find an exact match
    result = df.loc[(df['precolonial'] == precolonial) & 
                    (df['size'] == size) & 
                    (df['control'] == control) & 
                    (df['tree_id'] == tree_id)]
    
    if not result.empty:
        print(f"Exact match found: {(precolonial, size, control, tree_id)}")
        return result.iloc[0]['template']

    # 2. Try to find a match with the first three columns, pick any tree_id if no match found
    result = df.loc[(df['precolonial'] == precolonial) & 
                    (df['size'] == size) & 
                    (df['control'] == control)]
    
    if not result.empty:
        print(f"Falling back to control and picking random tree_id for: {(precolonial, size, control)}")
        chosen_template = result.sample(1, random_state=rng.integers(1e9)).iloc[0]['template']
        print(f"Found template with key: {result.sample(1, random_state=rng.integers(1e9)).iloc[0][['precolonial', 'size', 'control', 'tree_id']]}")
        return chosen_template

    # 3. Try to match by just `precolonial` and `size`, and randomly pick control and tree_id
    result = df.loc[(df['precolonial'] == precolonial) & 
                    (df['size'] == size)]
    
    if not result.empty:
        print(f"Falling back to size and picking random control and tree_id for: {(precolonial, size)}")
        chosen_template = result.sample(1, random_state=rng.integers(1e9)).iloc[0]['template']
        print(f"Found template with key: {result.sample(1, random_state=rng.integers(1e9)).iloc[0][['precolonial', 'size', 'control', 'tree_id']]}")
        return chosen_template

    # 4. If no match found at all
    print(f"######################ERROR######################")
    print(f"No match found for: {(precolonial, size, control, tree_id)}")
    return None

def process_single_tree(row, tree_templates_df, tree_templates_dict, rng):
    """
    Processes a single tree using fallbacks.
    """
    precolonial = bool(row['precolonial'])
    size = str(row['size']).strip()
    control = str(row['control']).strip()
    tree_id = int(row['tree_id'])

    # Query the DataFrame using the fallback logic
    template = query_tree_template(tree_templates_df, precolonial, size, control, tree_id, rng)

    if template is None:
        print(f"No match found for: {(precolonial, size, control, tree_id)}")
        return None

    # You can now use the retrieved template and apply it in your logic
    return initialise_and_translate_tree(template, row)

def process_all_trees(locationDF, voxel_size=0.5):
    """
    Process all trees with the fallback logic in parallel using Dask.
    """
    print(f'Loading tree templates of voxel size {voxel_size}')
    
    #file paths
    templateDir = Path('data/revised/trees') 
    
    euc_voxelised_name = f'{voxel_size}_euc_voxel_tree_dict.pkl'
    elm_voxelised_name = f'{voxel_size}_elm_voxel_templateDF.pkl'

    euc_original_name = 'updated_tree_dict.pkl'
    elm_original_name = 'elm_tree_dict.pkl'

    if voxel_size == 0:
        print(f'loading original tree templates')
        euc_input_path = templateDir / euc_original_name
        elm_input_patth = templateDir / elm_original_name
        euc_tree_templates = pickle.load(open(euc_input_path, 'rb'))
        elm_tree_templates = pd.read_pickle(elm_input_patth)
        print(f'loaded euc tree templates from {euc_input_path}')
        print(f'loaded elm tree templates from {elm_input_patth}')
    else:
        euc_input_path = templateDir / euc_voxelised_name
        elm_input_patth = templateDir / elm_voxelised_name
        euc_tree_templates = pickle.load(open(euc_input_path, 'rb'))
        elm_tree_templates = pd.read_pickle(elm_input_patth)
        print(f'loaded voxel size {voxel_size} euc tree templates from {euc_input_path}')
        print(f'loaded voxel size {voxel_size} elm tree templates from {elm_input_patth}')
    
    # Debug the loaded DataFrame
    print("DataFrame columns:", elm_tree_templates.columns)
    print("First row:", elm_tree_templates.iloc[0])
    
    # Get the first template
    first_template = elm_tree_templates.iloc[0]['template']
    print(f'template df is {first_template.head()}')

    # Convert the dictionary to a DataFrame for querying
    tree_templates_df = convert_dict_to_dataframe(euc_tree_templates, elm_tree_templates)

    # Enforce Python native types and strip potential hidden characters
    locationDF['precolonial'] = locationDF['precolonial'].astype(bool)  # Convert to Python bool
    locationDF['size'] = locationDF['size'].astype(str).str.strip()  # Ensure string and strip
    locationDF['control'] = locationDF['control'].astype(str).str.strip()  # Ensure string and strip
    locationDF['tree_id'] = locationDF['tree_id'].astype(int)  # Ensure integer type

    # Convert processedDF to a Dask DataFrame
    dask_df = dd.from_pandas(locationDF, npartitions=10)

    # Initialize a random generator
    rng = np.random.default_rng(seed=42)

    # Use TQDM to show progress during map_partitions
    delayed_results = []
    for _, row in tqdm(dask_df.iterrows(), total=len(locationDF)):
        delayed_results.append(delayed(process_single_tree)(row, tree_templates_df, euc_tree_templates, rng))

    # Use Dask's ProgressBar
    with dask.diagnostics.ProgressBar():
        # Compute the results in parallel and show progress
        results = dask.compute(*delayed_results)

    print(f'Resources distributed')
    print('Combining results into a single DataFrame...')

    # Combine the results into a single DataFrame, excluding None results
    resourceDF = pd.concat([res for res in results if res is not None])
    return locationDF, resourceDF

import os

def convertToPoly(resourceDF):
    
    # Extracting coordinates from the DataFrame
    points = resourceDF[['x', 'y', 'z']].values
    print("Extracted coordinates from DataFrame for PolyData creation.")

    #define columns to use as point_data attributes
    #all columns starting with resource_
    resource_columns = [col for col in resourceDF.columns if col.startswith('resource_')]

    #extend with 'precolonial', 'size', 'control', 'tree_id', 'nodeID', 'structureID'
    attribute_columns = ['precolonial', 'size', 'control', 'tree_id']

     


    # Creating a PolyData object from the extracted points
    poly = pv.PolyData(points)
    print("Created PolyData object from extracted points.")
    
    # Iterating through each column in the DataFrame to add as point data attributes
    """for col in resourcDF.columns:
        if col not in ['x', 'y', 'z']:  # Skipping coordinate columns
            poly.point_data[col] = resourcDF[col].values
            print(f"Added column '{col}' as point data attribute to PolyData.")"""

    
    for resource in resource_columns:
        print(f'adding resource column {resource}')
        #print values, counts and types of values per column
        print(f'counts: {resourceDF[resource].value_counts()}')
        print(f'types: {resourceDF[resource].dtype}')

        poly.point_data[resource] = resourceDF[resource].values
        print(f"Added column '{resource}' as point data attribute to PolyData.")

    for col in attribute_columns:
        poly.point_data[col] = resourceDF[col].values
        print(f"Added column '{col}' as point data attribute to PolyData.")
    
    print(f'PolyData object created with {len(poly.points)} points and {len(attribute_columns)} point data attributes')
    return poly

    #poly.plot(scalars=treeResource_peeling bark', render_points_as_spheres=True)

def rotate_resource_structures(locationDF, resourceDF):
    def rotate_points(points, angle_deg, pivot):
        angle_rad = np.deg2rad(angle_deg)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]])
        shifted_points = points - pivot
        rotated_shifted = shifted_points @ rotation_matrix.T
        rotated = rotated_shifted + pivot
        return rotated

    def apply_rotation(df, angle, pivot):
        points = df[['x', 'y']].values
        rotated = rotate_points(points, angle, pivot)
        df = df.copy()
        df['x'] = rotated[:, 0]
        df['y'] = rotated[:, 1]
        return df

    def rotate_group(group):
        # Get rotation angle and pivot point from respective dictionaries
        angle = rotation_angles.get(group.name, 0.0)
        pivot = tree_positions.get(group.name, np.array([0.0, 0.0]))
        if angle == 0.0:
            return group
        return apply_rotation(group, angle, pivot)

    # Create two separate dictionaries
    temp_dict = locationDF.set_index('structureID')[['x', 'y', 'rotateZ']].to_dict('index')
    
    # Dictionary for pivot points (as numpy arrays)
    tree_positions = {k: np.array([v['x'], v['y']]) for k, v in temp_dict.items()}
    
    # Dictionary for rotation angles
    rotation_angles = {k: v['rotateZ'] for k, v in temp_dict.items()}

    # Group by 'structureID' and apply rotation
    updated_resourceDF = resourceDF.groupby('structureID').apply(rotate_group).reset_index(drop=True)
    return updated_resourceDF

# Main execution
if __name__ == "__main__":
    site = 'trimmed-parade'
    scenarioVoxelSize = 1
    outputVoxelSize = 0
    year = 60
    # Assuming processedDF is already loaded with tree instance data
    filepath = f'data/revised/final/{site}/{site}_{scenarioVoxelSize}_treeDF_{year}.csv'
    treeDF = pd.read_csv(filepath)
    locationDF, resourceDF = process_all_trees(treeDF, voxel_size=outputVoxelSize)
    print("\nColumns in treeDF:")
    for col in treeDF.columns:
        print(col)
        
    print("\nColumns in locationDF:") 
    for col in locationDF.columns:
        print(col)
    
    resourceDF = rotate_resource_structures(locationDF, resourceDF)

    print(f'combined resourceDF is {len(resourceDF)} voxels long')

    poly = convertToPoly(resourceDF)
    polyfilePath = f'data/revised/final/{site}/{site}_{outputVoxelSize}_treeDF_{year}_poly.vtk'
    print(f'saving polydata to {polyfilePath}')
    poly.save(polyfilePath)
    print(f'exported poly to {polyfilePath}')



    





    print(f'processing complete')


