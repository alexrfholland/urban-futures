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


def convert_dict_to_dataframe(tree_templates):
    """
    Converts the dictionary of (tuple: template) to a DataFrame for easier querying.
    """
    rows = []
    for key, template in tree_templates.items():
        precolonial, size, control, tree_id = key
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


def query_tree_template2(df, precolonial, size, control, tree_id):
    """
    Attempts to query the tree template using fallbacks if necessary.
    Logs when fallback steps are used.
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
        chosen_template = result.sample(1, random_state=42).iloc[0]['template']
        print(f"Found template with key: {result.sample(1, random_state=42).iloc[0][['precolonial', 'size', 'control', 'tree_id']]}")
        return chosen_template

    # 3. Try to match by just `precolonial` and `size`, and randomly pick control and tree_id
    result = df.loc[(df['precolonial'] == precolonial) & 
                    (df['size'] == size)]
    
    if not result.empty:
        print(f"Falling back to size and picking random control and tree_id for: {(precolonial, size)}")
        chosen_template = result.sample(1, random_state=42).iloc[0]['template']
        print(f"Found template with key: {result.sample(1, random_state=42).iloc[0][['precolonial', 'size', 'control', 'tree_id']]}")
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

def process_all_trees(locationDF):
    """
    Process all trees with the fallback logic in parallel using Dask.
    """
    voxel_size = 0.5
    print(f'Loading tree templates of voxel size {voxel_size}')
    tree_templates = pickle.load(open(f'data/revised/trees/{voxel_size}_voxel_tree_dict.pkl', 'rb'))

    # Convert the dictionary to a DataFrame for querying
    tree_templates_df = convert_dict_to_dataframe(tree_templates)

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
        delayed_results.append(delayed(process_single_tree)(row, tree_templates_df, tree_templates, rng))

    # Use Dask's ProgressBar
    with dask.diagnostics.ProgressBar():
        # Compute the results in parallel and show progress
        results = dask.compute(*delayed_results)

    print(f'Resources distributed')
    print('Combining results into a single DataFrame...')

    # Combine the results into a single DataFrame, excluding None results
    resourceDF = pd.concat([res for res in results if res is not None])
    return locationDF, resourceDF


def process_all_trees2(locationDF):
    """
    Process all trees with the fallback logic in parallel using Dask.
    """
    voxel_size = 0.1
    print(f'Loading tree templates of voxel size {voxel_size}')
    tree_templates = pickle.load(open(f'data/revised/trees/{voxel_size}_voxel_tree_dict.pkl', 'rb'))

    # Convert the dictionary to a DataFrame for querying
    tree_templates_df = convert_dict_to_dataframe(tree_templates)

    # Enforce Python native types and strip potential hidden characters
    locationDF['precolonial'] = locationDF['precolonial'].astype(bool)  # Convert to Python bool
    locationDF['size'] = locationDF['size'].astype(str).str.strip()  # Ensure string and strip
    locationDF['control'] = locationDF['control'].astype(str).str.strip()  # Ensure string and strip
    locationDF['tree_id'] = locationDF['tree_id'].astype(int)  # Ensure integer type

    # Convert processedDF to a Dask DataFrame
    dask_df = dd.from_pandas(locationDF, npartitions=10)

    # Use TQDM to show progress during map_partitions
    delayed_results = []
    for _, row in tqdm(dask_df.iterrows(), total=len(locationDF)):
        delayed_results.append(delayed(process_single_tree)(row, tree_templates_df, tree_templates))

    # Use Dask's ProgressBar
    with dask.diagnostics.ProgressBar():
        # Compute the results in parallel and show progress
        results = dask.compute(*delayed_results)

    print(f'Resources distributed')
    print('Combining results into a single DataFrame...')

    # Combine the results into a single DataFrame, excluding None results
    resourceDF = pd.concat([res for res in results if res is not None])
    return locationDF, resourceDF

# Main execution
if __name__ == "__main__":
    site = 'trimmed-parade'
    # Assuming processedDF is already loaded with tree instance data
    processedDF = pd.read_csv(f'data/revised/{site}-tree-locations.csv')
    
    print(processedDF)
    combinedDF = process_all_trees(processedDF)

