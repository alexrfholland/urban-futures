
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
    points = combined_df[['X', 'Y', 'Z']].to_numpy()

    # Create the PolyData object with points
    polydata = pv.PolyData(points)

    # Add other columns as point data
    for col in combined_df.columns:
        if col not in ['X', 'Y', 'Z']:
            polydata.point_data[col] = combined_df[col].to_numpy()

    return polydata

def convert_key_to_tuple(row):
    """
    Converts a string representation of a tuple into an actual tuple.
    This assumes the format ('size', is_precolonial, 'control', improvement, tree_id).
    """
    if isinstance(row['tree_template_key'], str):
        return eval(row['tree_template_key'])  # Safely convert the string back to a tuple
    return row['tree_template_key']

def translate_tree(tree_template, x_offset, y_offset, z_offset):
    """
    Translates the tree template by the given x, y, z offsets.
    """
    tree_template_copy = tree_template.copy()
    tree_template_copy['X'] += x_offset
    tree_template_copy['Y'] += y_offset
    tree_template_copy['Z'] += z_offset
    return tree_template_copy

def process_single_tree(row, tree_templates):
    """
    Processes a single tree: looks up the template, copies it, and translates it.
    """
    #tree_template_key = row['tree_template_key']  # Already converted to a tuple
    tree_template_key = (False, 'large', 'park-tree', False, 13)
    tree_template = tree_templates[tree_template_key]  # Lookup from dictionary
    
    # Translate the tree template
    return translate_tree(tree_template, row['x'], row['y'], row['z'])

def process_all_trees(processedDF):
    """
    Process all trees in parallel using Dask with progress bars.
    """
    print('Loading tree templates...')
    tree_templates = pickle.load(open('data/treeOutputs/adjusted_tree_templates.pkl', 'rb'))

    print(f'loaded {len(tree_templates)} tree templates')
    
    # Batch convert the 'tree_template_key' column to tuples if necessary
    processedDF['tree_template_keyTEST'] = processedDF.apply(convert_key_to_tuple, axis=1)

    # Convert processedDF to a Dask DataFrame
    dask_df = dd.from_pandas(processedDF, npartitions=10)

    # Use TQDM to show progress during map_partitions
    delayed_results = []
    for _, row in tqdm(dask_df.iterrows(), total=len(processedDF)):
        delayed_results.append(delayed(process_single_tree)(row, tree_templates))

    # Use Dask's ProgressBar
    with dask.diagnostics.ProgressBar():
        # Compute the results in parallel and show progress
        results = dask.compute(*delayed_results)
    
    print(f'resources distributed')
    print('Combining results into a single DataFrame...')
    
    # Combine the results into a single DataFrame, excluding None results
    combined_df = pd.concat([res for res in results if res is not None])

    print(f'resources distributed, results preview is {combined_df.head()}')
    
    print('Creating PyVista object of canopy resources...')
    resourcePoly = create_pyvista_object(combined_df)
    return resourcePoly
    #resourcePoly.plot(scalars='resource')

def create_pyvista_object(combined_df):
    """
    Converts the processed DataFrame into a PyVista PolyData object.
    """
    print('##########################')
    # Ensure combined_df is a pandas DataFrame and not a tuple
    print(f"Type of combined_df: {type(combined_df)}")
    
    if not isinstance(combined_df, pd.DataFrame):
        raise ValueError("combined_df is not a pandas DataFrame")

    # Extract the X, Y, Z columns as points
    points = combined_df[['X', 'Y', 'Z']].to_numpy()

    # Create the PolyData object with points
    polydata = pv.PolyData(points)

    # Add other columns as point data
    for col in combined_df.columns:
        if col not in ['X', 'Y', 'Z']:
            polydata.point_data[col] = combined_df[col].to_numpy()

    return polydata


"""tree_templates = pickle.load(open('data/treeOutputs/adjusted_tree_templates.pkl', 'rb'))
print(tree_templates.keys())
print(tree_templates[(False, 'large', 'park-tree', False, 13)])"""
#print(tree_templates[(False, 'large', 'street-tree', False, 8)])
