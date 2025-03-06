import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import random

import pyvista as pv

try:
    from . import getHabitatStructureVoxels  # Relative import for package execution
except ImportError:
    import getHabitatStructureVoxels  # Direct import for script execution

isChatGPT = False

def create_polydata(df):
        # Create an array for points using x, y, z columns

        print(df)
        points = df[['X', 'Y', 'Z']].values
        polydata = pv.PolyData(points)

        # Add other columns as point data, sanitizing non-numeric data
        for col in df.columns.difference(['X', 'Y', 'Z']):
            sanitized_data = []
            for val in df[col]:
                if isinstance(val, str):
                    sanitized_val = val.encode('ascii', 'ignore').decode()
                    sanitized_data.append(sanitized_val)
                else:
                    sanitized_data.append(val)
            polydata.point_data[col] = sanitized_data

        polydata.point_data['ScanZ'] = points = df['Z']
        return polydata


def load_and_process_data(urban_systems_compat_matrix_path, siteDF):
    """
    Load urban systems and street segments datasets, perform matching based on defined rules,
    and retrieve habitat structures for a given row index in the urban system dataset.

    Parameters:
    urban_systems_path (str): File path for the urban systems CSV file.
    street_segments_path (str): File path for the street segments CSV file.

    Returns:
    pd.DataFrame: The updated street segments dataframe with additional columns
                  'urban_system_index_rule_based' and 'random_habitat_structure'.
    """
    # Load the datasets
    urban_systems_df = pd.read_csv(urban_systems_compat_matrix_path)
    
    print(f'site opportunties df is \n {siteDF}')
    
    

    # Split 'urban system' column in the urban systems dataset
    urban_systems_df['split_urban_system'] = urban_systems_df['urban system'].str.split('.')

    # Creating reverse lookup dictionary from urban systems dataframe
    simplified_reverse_lookup_dict = {}
    for idx, row in urban_systems_df.iterrows():
        parts = row['split_urban_system']
        if len(parts) >= 2:  # At least one dot, meaning we only need to look up action and type
            key_urban_system_type = parts[0] + '.' + parts[1]
            simplified_reverse_lookup_dict[key_urban_system_type] = idx
        if len(parts) == 3:  # Two dots, meaning we need to look up action, type, and subtype
            key_full = parts[0] + '.' + parts[1] + '.' + parts[2]
            simplified_reverse_lookup_dict[key_full] = idx

    # Function to match based on rules
    def match_based_on_rules(row):
        key_urban_system_type = row['urban system'] + '.' + row['type']
        key_full = key_urban_system_type + '.' + row['subtype']
        return simplified_reverse_lookup_dict.get(key_full, simplified_reverse_lookup_dict.get(key_urban_system_type, None))
    
    # Applying the matching function
    siteDF['urban_system_index_rule_based'] = siteDF.apply(match_based_on_rules, axis=1)

    # Creating a mapping of indexes to habitat structures
    index_to_habitat_structures = {}
    for idx in urban_systems_df.index:
        habitat_structures = get_habitat_structures(urban_systems_df, idx)
        index_to_habitat_structures[idx] = habitat_structures

    # Random assignment function
    def assign_random_habitat_structure(row):
        index = row['urban_system_index_rule_based']
        habitat_structures = index_to_habitat_structures.get(index, [])
        return random.choice(habitat_structures) if habitat_structures else None

    # Applying the function to the street segments dataframe
    siteDF['random_habitat_structure'] = siteDF.apply(assign_random_habitat_structure, axis=1)

    print(f'potential structures are \n {siteDF}')

    print(f'urban systems are \n {siteDF["urban system"].value_counts()}')
    print(f'urban system row indexes are \n {siteDF["urban_system_index_rule_based"].value_counts()}')
    print(f'potential structures are \n {siteDF["random_habitat_structure"].value_counts()}')

    """poly = create_polydata(siteDF)

    plotter = pv.Plotter()
    plotter.add_mesh(poly, scalars = 'random_habitat_structure', cmap = 'Set1', render_points_as_spheres=True, point_size=5)
    plotter.add_scalar_bar(title='Urban System Index')
    plotter.show()"""




    return siteDF

def get_habitat_structures(urban_systems_df, row_index):
    """
    Retrieve habitat structures for a given row index in the urban system dataset.

    Parameters:
    urban_systems_df (pd.DataFrame): The dataframe containing urban systems data.
    row_index (int): The index of the row in the urban system dataset.

    Returns:
    list: A list of habitat structures (column names) where the cell value is 'x' for the given row index.
    """

    # Getting the row in the urban systems dataframe
    urban_system_row = urban_systems_df.iloc[row_index]

    print(f'getting habitat structures that are compatible with {urban_system_row}')

    # Columns that have 'x' in this row
    habitat_structures = urban_system_row[urban_system_row == 'x'].index.tolist()
    

    return habitat_structures



def distribute_structures_statistically(df, habitat_to_cubic_meters):
    """
    Allocate structures considering the buffer size requirements for each habitat structure.

    Parameters:
    df (pd.DataFrame): The dataframe containing street segments data.
    habitat_to_cubic_meters (dict): A mapping from habitat structure types to their buffer sizes.

    Returns:
    pd.DataFrame: Updated dataframe with 'allocated_structure' assigned based on buffer requirements.
    """
    for structure in df['random_habitat_structure'].unique():
        buffer_size = habitat_to_cubic_meters.get(structure, 1)  # Default to 1 if not specified
        buffer_ratio = buffer_size

        # Calculate the maximum number of structures that can be allocated considering the buffer
        total_structure_voxels = df[df['random_habitat_structure'] == structure].shape[0]
        max_allocatable_structures = total_structure_voxels // buffer_ratio

        print(f"Processing Habitat Structure: {structure}, Buffer Size: {buffer_size}")
        print(f"Total Available Voxels: {total_structure_voxels}, Max Allocatable Structures: {max_allocatable_structures}")

        if max_allocatable_structures > 0:
            allocated_indices = np.random.choice(df[df['random_habitat_structure'] == structure].index, max_allocatable_structures, replace=False)
            df.loc[allocated_indices, 'allocated_structure'] = structure
        else:
            df.loc[df['random_habitat_structure'] == structure, 'allocated_structure'] = 'too close'

    print(f'unique structures in df are {df["allocated_structure"].value_counts()}')

    """poly = create_polydata(df)

    plotter = pv.Plotter()
    plotter.add_mesh(poly, scalars = 'allocated_structure', cmap = 'Set1', render_points_as_spheres=True, point_size=5)
    plotter.add_scalar_bar(title='Urban System Index')
    plotter.show()"""


    return df

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import pandas as pd


def distribute_resources(siteDF, habitat_cubic_meters_df):
    """
    Distribute resources for each habitat structure based on specified ratios, 
    and calculate randomized XYZ positions for these resources.

    Parameters:
    df (pd.DataFrame): DataFrame containing street segments data.
    habitat_cubic_meters_df (pd.DataFrame): DataFrame containing details about habitat structures and resource ratios.

    Returns:
    pd.DataFrame: A DataFrame with distributed resources including their XYZ positions and types.
    """
    resource_types = ['Leaf clumps', 'other', 'peeling bark', 'dead branch', 'epiphyte', 
                      'leaf litter', 'perchable branch', 'hollow', 'fallen log']

    resource_distribution = pd.DataFrame()

    print(f'cubic meters per habitat structure are \n{habitat_cubic_meters_df}')

    print(f'site df before allocation of resources is {siteDF}')

    # Iterate over each resource type
    for resource in resource_types:
        # Iterate over each unique habitat structure
        for structure in siteDF['random_habitat_structure'].unique():
            print(f'allocating {structure}-{resource}')
            if structure in habitat_cubic_meters_df['Structure'].values:
                structure_row = habitat_cubic_meters_df[habitat_cubic_meters_df['Structure'] == structure].iloc[0]

                if structure_row['Enabled'] != 0:
                    
                    resource_ratio = structure_row[resource] / structure_row['Buffer space']
                    

                    resource_count = int(np.round(siteDF[siteDF['random_habitat_structure'] == structure].shape[0] * resource_ratio))

                    print(f'{resource} per cubic meter for {structure} is {resource_ratio}, allocating {resource_count}')


                    if resource_count > 0:
                        # Filter siteDF for the current structure
                        filtered_df = siteDF[siteDF['random_habitat_structure'] == structure]

                        elevation = 0

                        if resource not in ['fallen log, leaf litter']:
                            elevation = structure_row['Elevation']

                        xExtent, yExtent, zExtent = structure_row['xExtent']/2, structure_row['yExtent']/2, structure_row['zExtent']/2

                        # Generate jittered positions for the resources within the filtered DataFrame
                        x_jittered = filtered_df['X'] + filtered_df['nx'] + np.random.uniform(-xExtent, xExtent, size=filtered_df.shape[0])
                        y_jittered = filtered_df['Y'] + filtered_df['ny'] + np.random.uniform(-yExtent, yExtent, size=filtered_df.shape[0])
                        z_jittered = filtered_df['Z'] + filtered_df['nz'] + np.random.uniform(-zExtent, zExtent, size=filtered_df.shape[0]) + elevation

                        # Select random indices from the filtered DataFrame based on resource count
                        allocated_indices = np.random.choice(filtered_df.index, resource_count, replace=True)

                        # Create a temporary DataFrame with the selected indices and jittered positions
                        temp_df = pd.DataFrame({
                            'voxelX' : siteDF.loc[allocated_indices, 'X'],
                            'voxelY' : siteDF.loc[allocated_indices, 'Y'],
                            'voxelZ' : siteDF.loc[allocated_indices, 'Z'],
                            'X': x_jittered[allocated_indices], 
                            'Y': y_jittered[allocated_indices], 
                            'Z': z_jittered[allocated_indices], 
                            'structure': structure, 
                            'resource': resource
                        })

                        # Concatenate the allocated resources to the main resource distribution DataFrame
                        resource_distribution = pd.concat([resource_distribution, temp_df])
                    else:
                        print(f"structure {structure} is not enabled")

            else:
                print(f"Structure '{structure}' not found in habitat_cubic_meters_df.")


    
    print (f'resource distribution df is \n{resource_distribution}')
    """poly = create_polydata(resource_distribution)

    plotter = pv.Plotter()
    plotter.add_mesh(poly, scalars = 'structure', cmap = 'Set1', render_points_as_spheres=True, point_size=5)
    plotter.add_scalar_bar(title='Urban System Index')
    plotter.show()"""


    
    print(f'structures allocated for site are \n {siteDF["allocated_structure"].value_counts()}')

    print(f'resource df is \n {resource_distribution}')
    print(f'resources are \n {resource_distribution["resource"].value_counts()}')

    print(resource_distribution)



    return resource_distribution



def create_multiblock_from_df(df):
    print(f'creating multiblock')
    # Function to create PolyData from DataFrame
    def create_polydataB(df):
        # Create an array for points using x, y, z columns
        points = df[['x', 'y', 'z']].values
        polydata = pv.PolyData(points)

        # Add other columns as point data, sanitizing non-numeric data
        for col in df.columns.difference(['x', 'y', 'z']):
            sanitized_data = []
            for val in df[col]:
                if isinstance(val, str):
                    sanitized_val = val.encode('ascii', 'ignore').decode()
                    sanitized_data.append(sanitized_val)
                else:
                    sanitized_data.append(val)
            polydata.point_data[col] = sanitized_data

        polydata.point_data['ScanZ'] = points = df['z']
        return polydata

    # Filtering DataFrames based on resource type
    branches_df = df[df['resource'].isin(['perchable branch', 'peeling bark', 'dead branch'])]
    ground_resources_df = df[df['resource'].isin(['fallen log', 'leaf litter'])]
    canopy_resources_df = df[df['resource'].isin(['epiphyte', 'hollow'])]

    print('creating branches polydata')
    # Creating PolyData for each category
    branches_polydata = create_polydata(branches_df)

    print('creating ground polydata')
    ground_resources_polydata = create_polydata(ground_resources_df)

    print('creating canopy polydata')
    canopy_resources_polydata = create_polydata(canopy_resources_df)

    # Creating MultiBlock
    multiblock = pv.MultiBlock()
    multiblock['branches'] = branches_polydata
    multiblock['ground resources'] = ground_resources_polydata
    multiblock['canopy resources'] = canopy_resources_polydata

    return multiblock

# Example usage
# multiblock = create_multiblock_from_df(your_dataframe)

# Example usage
# multiblock = create_multiblock_from_df(your_dataframe)

# Example usage
# multiblock = create_multiblock_from_df(your_dataframe)


def visualize_results(df):
    """
    Visualize the points in the dataframe, coloring them based on their 'allocated_structure' value.

    Parameters:
    df (pd.DataFrame): The dataframe containing the street segments data with 'allocated_structure' information.
    """
    # Assign unique numbers to each unique 'allocated_structure'
    unique_structures = df['allocated_structure'].unique()
    structure_to_number = {structure: i for i, structure in enumerate(unique_structures)}

    # Map structures to numbers for coloring
    df['structure_number'] = df['allocated_structure'].map(structure_to_number)

    # Define the colormap
    cmap = plt.cm.get_cmap('Set1', len(unique_structures))

    plt.scatter(df['X'], df['Y'], c=df['structure_number'], cmap=cmap, alpha=0.5)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Visualization of Street Segments by Allocated Structure')
    plt.colorbar(label='Structure Type', ticks=range(len(unique_structures)))
    plt.clim(-0.5, len(unique_structures) - 0.5)
    plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_resources_2d(resource_df):
    """
    Visualize the distributed resources in 2D.

    Parameters:
    resource_df (pd.DataFrame): DataFrame containing the distributed resources with 'x', 'y', and 'resource'.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define unique colors for each resource type
    resource_types = resource_df['resource'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(resource_types)))
    color_dict = dict(zip(resource_types, colors))

    # Plot each resource type in 2D with a unique color
    for resource_type, color in color_dict.items():
        sub_df = resource_df[resource_df['resource'] == resource_type]
        ax.scatter(sub_df['x'], sub_df['y'], color=color, label=resource_type, alpha=0.3, s=1)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('2D Visualization of Distributed Resources')
    ax.legend(title='Resource Types')

    plt.show()


def distributeStructuresAndResources(siteDF):
    # Define file paths


    
    #urban_systems_compatibility_matrix_path = 'current tests/reformatted_urban_systems_linked_to_habitat_structures.csv'
    #habitat_cubic_meters_path = 'current tests/habitat structure cubic meters.csv'
    #habitat_cubic_meters_df = pd.read_csv(habitat_cubic_meters_path)



    urban_systems_compatibility_matrix_path = 'modules/painter/data/urban systems.csv'

    # Load and process data
    urban_systems_df = load_and_process_data(urban_systems_compatibility_matrix_path, siteDF)

    habitat_cubic_meters_df = getHabitatStructureVoxels.getRatios()

    # Load habitat cubic meters data
    habitat_to_cubic_meters = pd.Series(habitat_cubic_meters_df['Buffer space'].values, index=habitat_cubic_meters_df['Structure']).to_dict()

    # Get habitat structures for a specific urban system
    habitat_structures_example = get_habitat_structures(urban_systems_df, 15)
    print("Habitat structures example:", habitat_structures_example)

    print(f'dataframe is {len(urban_systems_df)} rows')

    site_df = urban_systems_df
    
    site_df['filled_structure'] = None  # Initialize 'filled_structure' column

    # Apply fixed buffer logic
    print('allocating...')

    site_df = distribute_structures_statistically(site_df, habitat_to_cubic_meters)
    
    print(site_df)

    print('distributing resources....')

    resourcedf = distribute_resources(site_df, habitat_cubic_meters_df)

    print(resourcedf)

    print('plotting...')

    print('create resource polydata block')
    multiblock = create_multiblock_from_df(resourcedf)

    print(f'multiblock of resources created, is {multiblock}')

    return multiblock


    # Visualize the results
    #visualize_results(sampled_street_segments_df)

    #visualize_resources_2d(resourcedf)

def main():
    street_segments_path = 'current tests/street-segments.csv'
    street_segments_df = pd.read_csv(street_segments_path)
    distributeStructuresAndResources(street_segments_df)

if __name__ == "__main__":
    main()

