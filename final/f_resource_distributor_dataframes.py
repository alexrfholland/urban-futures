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
    
    #check if useful_life_expectency or useful_life_exptectancy is in the row, and use the one that is in the row
    if 'useful_life_expectency' in row:
        tree_template_copy['useful_life_expectency'] = row['useful_life_expectency']
    elif 'useful_life_expectancy' in row:
        tree_template_copy['useful_life_expectancy'] = row['useful_life_expectancy']
    else:
        print('no useful_life_expectency or useful_life_expectancy found in the row')

    return tree_template_copy


def process_single_tree(row, tree_templates):
    """
    Processes a single tree: constructs the key from actual columns and translates it.
    """
    # Construct the tree_template_key directly from the columns
    tree_template_key = (
        bool(row['precolonial']),  # Ensure it's a Python boolean
        str(row['size']).strip(),  # Ensure it's a string and strip any whitespace
        str(row['control']).strip(),  # Ensure it's a string and strip any whitespace
        int(row['tree_id'])  # Ensure it's an integer
    )

    # Debugging: print the constructed key
    #print(f"Constructed key: {tree_template_key}")

    # Lookup the tree template using the constructed key
    try:
        tree_template = tree_templates[tree_template_key]
    except KeyError:
        print(f"KeyError: {tree_template_key} not found in tree_templates")
        return None
    
    # Translate the tree template and add additional columns from the row
    return initialise_and_translate_tree(tree_template, row)

def process_all_trees(locationDF):
    """
    Process all trees in parallel using Dask with progress bars.
    """
    voxel_size = 0.5
    print(f'Loading tree templates of voxel size {voxel_size}')
    tree_templates = pickle.load(open(f'data/revised/trees/{voxel_size}_voxel_tree_dict.pkl', 'rb'))

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
        delayed_results.append(delayed(process_single_tree)(row, tree_templates))

    # Use Dask's ProgressBar
    with dask.diagnostics.ProgressBar():
        # Compute the results in parallel and show progress
        results = dask.compute(*delayed_results)
    
    print(f'Resources distributed')
    print('Combining results into a single DataFrame...')
    
    # Combine the results into a single DataFrame, excluding None results
    combined_df = pd.concat([res for res in results if res is not None])

    """print(f'Resources distributed, results preview is {combined_df.head()}')
    
    print('Creating PyVista object of canopy resources...')
    resourcePoly = create_pyvista_object(combined_df)
    resourcePoly.plot(scalars='useful_life_expectency')"""
    return combined_df

# Main execution
if __name__ == "__main__":
    site = 'trimmed-parade'
    # Assuming processedDF is already loaded with tree instance data
    processedDF = pd.read_csv(f'data/revised/{site}-tree-locations.csv')
    
    print(processedDF)
    combinedDF = process_all_trees(processedDF)


