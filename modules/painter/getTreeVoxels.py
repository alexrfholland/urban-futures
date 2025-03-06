import pyvista as pv
import numpy as np
import pickle


#names are 'peeling bark', 'dead branch', 'fallen log', 'leaf litter', 'hollow', 'epiphyte'

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Load the pickled data
branches_data = load_pickle('data/prebaked-branches.pkl')
ground_cover_data = load_pickle('data/prebaked-tree-resources.pkl')

def plot_tree(key):
    # Fetch data from dictionaries using the provided key
    branches_df = branches_data.get(key)
    # For ground_cover_data, adjust the key to match its structure
    ground_cover_key = key[:-1]
    ground_cover_dict = ground_cover_data.get(ground_cover_key)

    if branches_df is None or ground_cover_dict is None:
        print(f"No data found for key: {key}")
        return

    # Create a PyVista plotter object
    plotter = pv.Plotter()

    # Preparing data for PyVista
    # Assuming branches_df has columns 'x', 'y', 'z', 'resource', 'Branch.type'
    branches_polydata = pv.PolyData(branches_df[['x', 'y', 'z']].values)
    branches_polydata['resource'] = branches_df['resource'].values

    cube = pv.Cube()



    glyphs = branches_polydata.glyph(geom=cube, scale=False, orient=False, factor=1)
    plotter.add_mesh(glyphs, scalars='resource', show_scalar_bar=False)  # Assuming 'z' is a relevant scalar

    ground_points = []
    ground_resource_type = []

    # Adding ground cover points as glyphs
    for ground_cover_type in ['fallen log', 'leaf litter']:
        points = ground_cover_dict[ground_cover_type]
        ground_points.extend(points)
        ground_resource_type.extend([ground_cover_type] * len(points))  # Extend the ground_resource_type list with the ground_cover_type, repeated for the number of points

            
    ground_polydata = pv.PolyData(ground_points)
    ground_polydata['resource'] = ground_resource_type

    groundGlyphs = ground_polydata.glyph(geom=cube, scale=False, orient=False, factor=0.25)
    plotter.add_mesh(groundGlyphs, scalars='resource', show_scalar_bar=False)  # Assuming 'z' is a relevant scalar

    # Display the plot
    plotter.show()


import numpy as np
from scipy.sparse import coo_matrix

def sparse_tree(key, cell_size):
    # Load data as in plot_tree
    branches_df = branches_data.get(key)
    ground_cover_key = key[:-1]
    ground_cover_dict = ground_cover_data.get(ground_cover_key)

    if branches_df is None or ground_cover_dict is None:
        print(f"No data found for key: {key}")
        return None

    # Log the shape and sample data of branch points
    print("Branch Points Shape:", branches_df[['x', 'y', 'z']].shape)
    print("Branch Points Sample:\n", branches_df[['x', 'y', 'z']].head(10))

    # Extract and log the resources from branches
    branches_resources = branches_df['resource'].values
    print("Branch Resources Shape:", branches_resources.shape)
    print("Branch Resources Sample:", branches_resources[:10])

    # Initialize a list to store all XYZ coordinates
    all_points = branches_df[['x', 'y', 'z']].values.tolist()

    # Iterate over each ground cover type, log their data
    for ground_cover_type in ['fallen log', 'leaf litter']:
        ground_cover_points = ground_cover_dict[ground_cover_type]
        print(f"{ground_cover_type} Points Shape:", len(ground_cover_points))
        print(f"{ground_cover_type} Points Sample:", ground_cover_points[:10])
        all_points.extend(ground_cover_points)

    # Vertically stack the points to create a single array
    all_points = np.vstack(all_points)
    print("All Points Combined Shape:", all_points.shape)

    # Initialize a list to store all resources
    all_resources = list(branches_resources)

    # Iterate over each ground cover type and log their resources
    for ground_cover_type in ['fallen log', 'leaf litter']:
        num_points = len(ground_cover_dict[ground_cover_type])
        ground_cover_resources = np.array([ground_cover_type] * num_points)
        print(f"{ground_cover_type} Resources Shape:", ground_cover_resources.shape)
        print(f"{ground_cover_type} Resources Sample:", ground_cover_resources[:10])
        all_resources.extend(ground_cover_resources)

    # Concatenate to create a single array of resources
    all_resources = np.concatenate(all_resources)
    print("All Resources Combined Shape:", all_resources.shape)

    # Determine bounds and digitize points into voxel grid
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    bins = [np.arange(min_coords[i], max_coords[i] + cell_size, cell_size) for i in range(3)]
    digitized = np.vstack([np.digitize(all_points[:, i], bins[i]) for i in range(3)]).T
    print("Digitized Points Shape:", digitized.shape)

    # Map resources to indices
    unique_resources, resource_indices = np.unique(all_resources, return_inverse=True)
    print("Unique Resources:", unique_resources)
    print("Resource Indices Shape:", resource_indices.shape)

    # Define the shape of the higher-dimensional space
    shape = (len(bins[0]), len(bins[1]), len(bins[2]), len(unique_resources))
    print("Shape for the Sparse Matrix:", shape)

    # Create linear indices for the sparse matrix
    linear_indices = np.ravel_multi_index(
        (*digitized.T, resource_indices),
        shape
    )

    # Count occurrences using bincount
    counts = np.bincount(linear_indices, minlength=np.prod(shape))
    print("Counts Shape:", counts.shape)

    # Create higher-dimensional sparse matrix
    sparse_matrix = coo_matrix(np.reshape(counts, shape))

    return sparse_matrix

import pandas as pd



def getDf(key):
    # Load data as in plot_tree
    branches_df = branches_data.get(key)
    ground_cover_key = key[:-1]
    ground_cover_dict = ground_cover_data.get(ground_cover_key)

    if branches_df is None or ground_cover_dict is None:
        print(f"No data found for key: {key}")
        return None

    # Log the shape and sample data of branch points
    print("Branch Points Shape:", branches_df[['x', 'y', 'z']].shape)
    print("Branch Points Sample:\n", branches_df[['x', 'y', 'z']].head(10))

    # Extract and log the resources from branches
    branches_resources = branches_df['resource'].values
    print("Branch Resources Shape:", branches_resources.shape)
    print("Branch Resources Sample:", branches_resources[:10])

    # Initialize a list to store all XYZ coordinates
    testbranchDF = branches_df[['x', 'y', 'z','resource']]

    print(f'extracted branch points are \n{testbranchDF}')

    log_points = ground_cover_dict['fallen log']
    leaf_points = ground_cover_dict['leaf litter']

    log_resources = np.array(['fallen log'] *  len(ground_cover_dict['fallen log']))
    leaf_resources = np.array(['leaf litter'] *  len(ground_cover_dict['leaf litter']))
    
    ground_points = np.concatenate([log_points, leaf_points])
    ground_resources = np.concatenate([log_resources,leaf_resources])

    test_ground_df = pd.DataFrame(ground_points, columns=['x', 'y', 'z'])
    test_ground_df['resource'] = ground_resources

    #testbranchDF.to_csv('testBranches.csv', index=False)
    #test_ground_df.to_csv('testGround.csv', index=False)


    return testbranchDF, test_ground_df


import numpy as np

def create_sparse_voxel_grid(combined_df, cell_size):
    # Determine the bounds for voxel grid
    min_coords = combined_df[['x', 'y', 'z']].min().values
    max_coords = combined_df[['x', 'y', 'z']].max().values

    # Create bins for digitizing
    x_bins = np.arange(min_coords[0], max_coords[0] + cell_size, cell_size)
    y_bins = np.arange(min_coords[1], max_coords[1] + cell_size, cell_size)
    z_bins = np.arange(min_coords[2], max_coords[2] + cell_size, cell_size)

    # Digitize the points into voxel grid coordinates
    combined_df['voxel_x'] = np.digitize(combined_df['x'], bins=x_bins) - 1
    combined_df['voxel_y'] = np.digitize(combined_df['y'], bins=y_bins) - 1
    combined_df['voxel_z'] = np.digitize(combined_df['z'], bins=z_bins) - 1

    # Group by voxel and resource to get counts
    voxel_resource_counts = combined_df.groupby(['voxel_x', 'voxel_y', 'voxel_z', 'resource']).size().reset_index(name='count')

    # Pivot the data to have one column per resource type with counts as values
    sparse_voxel_grid = voxel_resource_counts.pivot_table(index=['voxel_x', 'voxel_y', 'voxel_z'], 
                                                          columns='resource', 
                                                          values='count').fillna(0).astype(int).reset_index()

    return sparse_voxel_grid


# Define cell size
cell_size = 1

branches_df, ground_df = getDf(('large', True, 'reserve-tree', 16))

def prepData(branches_df, ground_df):
    """
    Prepare the data by incorporating hollows and epiphytes into branches_df and then combining it with ground_df.
    
    Parameters:
    - branches_df: DataFrame containing branches data with 'x', 'y', 'z', 'resource' columns.
    - ground_df: DataFrame containing ground data with the same columns as branches_df.
    
    Returns:
    - combined_df: DataFrame with combined data, including new resources 'hollows' and 'epiphytes'.
    """
    np.random.seed(0)  # Seed for reproducibility

    # Create a mask for rows where resource is 'other'
    mask_other = branches_df['resource'] == 'other'

    # Randomly select 6 indices for 'epiphytes' and 4 for 'hollows' from the 'other' mask
    epiphytes_indices = np.random.choice(branches_df[mask_other].index, size=6, replace=False)
    hollows_indices = np.random.choice(branches_df[mask_other].index, size=4, replace=False)

    # Assign 'epiphytes' to the randomly chosen indices
    branches_df.loc[epiphytes_indices, 'resource'] = 'epiphyte'

    # Assign 'hollows' to the randomly chosen indices
    branches_df.loc[hollows_indices, 'resource'] = 'hollow'

    # Combine branches_df and ground_df
    combined_df = pd.concat([branches_df, ground_df], ignore_index=True)

    # Find all unique resources and their counts in the combined DataFrame
    resource_counts = combined_df['resource'].value_counts()

    # Print the unique resources and their counts
    print(resource_counts)

    return combined_df

# Apply the prepData function to the branches and ground dataframes
combined_df = prepData(branches_df, ground_df)

print(f'combined df is \n{combined_df}')

# Show the head of the combined dataframe to verify the changes
combined_df.head()

print(f'creating grid')
sparse_voxel_grid = create_sparse_voxel_grid(combined_df, 1)

print(sparse_voxel_grid.head())








import matplotlib.pyplot as plt

# Define the plotting function
def plot_voxel_distribution(sparse_voxel_grid):
    plt.figure(figsize=(12, 10))

    # Iterate over each resource type column except for 'voxel_x', 'voxel_y', 'voxel_z'
    for resource in sparse_voxel_grid.columns[3:]:
        # Get the coordinates and counts for the current resource
        x_coords = sparse_voxel_grid['voxel_x']
        z_coords = sparse_voxel_grid['voxel_z']
        counts = sparse_voxel_grid[resource]

        # Only plot the points where the count is non-zero
        mask = counts > 0
        plt.scatter(x_coords[mask], z_coords[mask], s=counts[mask] * 10, label=resource, alpha=0.6)

    plt.xlabel('Voxel X Coordinate')
    plt.ylabel('Voxel Z Coordinate')
    plt.title('Resource Distribution in the Voxel Grid (XZ Plane)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plotRaresOnly(sparse_voxel_grid):

    resource_colors = {
        'hollows': 'red',
        'epiphytes': 'green'
    }


    # Filter the DataFrame for 'hollows' and 'epiphytes'
    hollows_df = combined_df[combined_df['resource'] == 'hollows']
    epiphytes_df = combined_df[combined_df['resource'] == 'epiphytes']

    plt.figure(figsize=(12, 10))

    # Plot 'hollows' with size proportional to their count (which is 1 for each occurrence)
    plt.scatter(hollows_df['x'], hollows_df['z'], s=100, color=resource_colors['hollows'], label='Hollows')

    # Plot 'epiphytes' with size proportional to their count (which is 1 for each occurrence)
    plt.scatter(epiphytes_df['x'], epiphytes_df['z'], s=100, color=resource_colors['epiphytes'], label='Epiphytes')

    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.title('Distribution of Hollows and Epiphytes')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_violins(sparse_voxel_grid):
    
    """
    Each 'violin' represents the distribution of a particular resource type, 
    showing the density at different count levels. The width of each violin corresponds to the frequency of voxels at each count, 
    providing a clear picture of how the counts of each resource are spread out
    Thicker parts of the violin show where counts are most common.
    Thinner parts indicate fewer voxels with those counts.
    The white dot represents the median count.
    The thick bar in the center of each violin indicates the interquartile range.
    """
    
    # Using violin plots to visualize the distribution of resource counts in the voxels
    import seaborn as sns

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Prepare data for violin plot
    # We need to transform the DataFrame to have 'resource' and 'count' columns
    melted_data = sparse_voxel_grid.melt(id_vars=['voxel_x', 'voxel_y', 'voxel_z'], 
                                         var_name='resource', value_name='count')

    # Create the violin plot
    plt.figure(figsize=(14, 8))
    violin_plot = sns.violinplot(x='resource', y='count', data=melted_data[melted_data['count'] > 0], 
                                 scale='width', palette='Set3')

    # Set the title and labels of the plot
    violin_plot.set_title('Violin Plot of Resource Counts in Voxels')
    violin_plot.set_xlabel('Resource Type')
    violin_plot.set_ylabel('Count')

    # Show the plot
    plt.xticks(rotation=45)
    plt.show()


def save_resource_stats_to_csv(resource_stats_df, filename):
    """
    Save the resource statistics DataFrame to a CSV file.
    
    Parameters:
    - resource_stats_df: DataFrame containing the statistics of resource counts.
    - filename: The name of the file to save the data to.
    """
    # Save the DataFrame to a CSV file
    resource_stats_df.to_csv(filename, index=True)
    print(f"Resource statistics saved as {filename}")

# Define the filename for the stats CSV
stats_filename = '/mnt/data/resource_stats.csv'

plot_violins(sparse_voxel_grid)

# We need to adjust our function to work off the sparse_voxel_grid to get the resource stats.


def calculate_and_save_resource_stats(sparse_voxel_grid, filename):
    """
    Calculate the resource statistics from the sparse voxel grid and save it to a CSV file.
    
    Parameters:
    - sparse_voxel_grid: DataFrame representing the sparse voxel grid with resource counts.
    - filename: The name of the file to save the stats to.
    """
    # Calculate descriptive statistics for each resource
    resource_stats = sparse_voxel_grid.describe().T[['min', '25%', '50%', '75%', 'max']]
    resource_stats.columns = ['min', 'lower_quartile', 'median', 'upper_quartile', 'max']


    resource_stats = resource_stats.drop(['voxel_x', 'voxel_y', 'voxel_z'])



    print(resource_stats)

    print(resource_stats.index)


    # Save the stats DataFrame to a CSV file
    resource_stats.to_csv(filename, index=True)
    print(f"Resource statistics saved as {filename}")

# Define the filename for the stats CSV
#stats_filename = 'outputs/voxel_resource_stats.csv'

stats_filename = 'painter/data/tree_voxels.csv'


# Calculate statistics and save to CSV
calculate_and_save_resource_stats(sparse_voxel_grid, stats_filename)



# Usage:
# Replace 'some_key' with the actual key you wish to use
#plot_tree(('large', True, 'park-tree', 16))


#sparse_tree(('large', True, 'reserve-tree', 16), 1)
#print('done')