import pyvista as pv
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import stamp_cloud
import stamp_csv, stamp_shapefile, trees, create_additions, calculate_disturbances


import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree
import numpy as np

def get_canopies(poly_data, tree_df):
    tree_df.reset_index(drop=True, inplace=True)  # Reset the index of tree_df
    print(f'Index of tree_df after resetting:\n{tree_df.index}\n')  # Print the index of tree_df

    tree_df.rename(columns={'blockID': 'treeBlockID'}, inplace=True)
    print(f'Columns of tree_df after renaming:\n{tree_df.columns}\n')  # Print the columns of tree_df

    print(f'getting canopy projections...')
    n_points = poly_data.n_points  # Get the number of points in poly_data
    print(f'Number of points in poly_data: {n_points}\n')  # Print the number of points in poly_data

    
    print(tree_df)

    
    
    ####CREATE ATTRIBUTE ARRAYS
    attributeFields = ['treeBlockID', 'Useful Life Expectency Value', '_Tree_size', '_Control', 'isPrecolonial', 'isDevelop']
    default_value = [-1, -1, 'none', 'none', False, False]

    # Iterate through each attribute and replace NA or NaN values with the corresponding default value
    for i, attr in enumerate(attributeFields):
        # Count the number of NA/Nan values before replacing
        na_count_before = tree_df[attr].isna().sum()
        print(f'Number of NA/Nan values in {attr} before replacing: {na_count_before}')

        # Replace NA/Nan values with the corresponding default value
        tree_df[attr].fillna(default_value[i], inplace=True)

        # Optionally, count the number of NA/Nan values after replacing to ensure they have been replaced
        na_count_after = tree_df[attr].isna().sum()
        print(f'Number of NA/Nan values in {attr} after replacing: {na_count_after}')


    n_points = poly_data.n_points  # Get the number of points in poly_data
    attributeArrays = {}

    # Initialize new arrays with appropriate types and sizes
    # Determine the data type and default value
    for attr in attributeFields:
        dtype = tree_df[attr].dtype

        if pd.api.types.is_integer_dtype(dtype):
            np_dtype = np.int64  # Convert to NumPy integer type
        elif pd.api.types.is_float_dtype(dtype):
            np_dtype = np.float64  # Convert to NumPy float type
        elif pd.api.types.is_bool_dtype(dtype):
            np_dtype = np.bool_  # Convert to NumPy boolean type
        else:
            np_dtype = object  # Default to object type for other data types
        
        # Use np_dtype instead of dtype for the np.issubdtype check
        if np.issubdtype(np_dtype, np.number):
            default_value = '-1'  # Set default value to NaN for numerical data
        elif np.issubdtype(np_dtype, np.bool_):
            default_value = False  # Set default value to False for boolean data
        else:
            default_value = 'none'  # Set default value to an empty string for other data types

        # Initialize the array
        array = np.full(n_points, default_value, dtype=np_dtype)

        # Create or update the point_data attribute
        attributeArrays[attr] = array


   



    ###FIND THE NEAREST INDEX OF EACH CANOPY SITE VOXEL OF THE TREE DF

    # Create an array to hold the index of the nearest tree for each canopy point
    nearest_tree_index_array = np.full(n_points, -1, dtype=np.int64)  # Initialize with -1 to indicate no nearest tree found
    print(f'Initial nearest_tree_index_array:\n{nearest_tree_index_array}\n')  # Print the initial nearest_tree_index_array

    # Construct KDTree from tree DataFrame
    tree_coords = tree_df[['X', 'Y']].values
    kd_tree = cKDTree(tree_coords)

    print(poly_data.point_data)

    # Get the indices of canopy points
    canopy_indices = np.where(poly_data.point_data['canopy-iscanopy'] == 1)[0]
    print(f'Indices of canopy points:\n{canopy_indices}\n')  # Print the indices of canopy points

    # Get the coordinates of canopy points
    canopy_coords = poly_data.points[canopy_indices, :2]  # Considering only X and Y dimensions

    # Query the KDTree to find the index and distance of the nearest tree for each canopy point
    distances, nearest_tree_indices = kd_tree.query(canopy_coords)
    print(f'Distances and indices of nearest trees:\n{distances}\n{nearest_tree_indices}\n')  # Print distances and indices of nearest trees

    # Filter out canopy points that are >5 away from the nearest tree
    valid_indices_mask = distances <= 15
    valid_canopy_indices = canopy_indices[valid_indices_mask]
    valid_nearest_tree_indices = nearest_tree_indices[valid_indices_mask]
    print(f'Valid canopy indices and valid nearest tree indices:\n{valid_canopy_indices}\n{valid_nearest_tree_indices}\n')  # Print valid indices

    # Set isCanopy to 0 for invalid canopy points
    invalid_canopy_indices = canopy_indices[~valid_indices_mask]
    poly_data.point_data['canopy-iscanopy'][invalid_canopy_indices] = 0

    # Update the nearest_tree_index_array with the indices of the nearest trees
    nearest_tree_index_array[valid_canopy_indices] = valid_nearest_tree_indices
    print(f'Updated nearest_tree_index_array:\n{nearest_tree_index_array}\n')  # Print the updated nearest_tree_index_array

    # Optionally, you can add this array to the point_data of poly_data if you need to keep this information
    poly_data.point_data['nearest_tree_index'] = nearest_tree_index_array

    
    ##TRANSFER THE ATRRIBUTE

    print(f'valid inices {np.unique(nearest_tree_index_array[valid_canopy_indices])}')


    for attr in attributeFields:
        print(f'transferring attribute {attr}')
        attributeValues = tree_df[attr].values[nearest_tree_index_array[valid_canopy_indices]]
        print(f'Unique values in tree_df[{attr}]: {np.unique(tree_df[attr])}')
        print(f'Unique values in attributeValues before transfer: {np.unique(attributeValues)}')
        # Obtain the values of the current attribute for the nearest trees

        # Handle the replacement of Pandas NA values similar to your example
        if pd.api.types.is_integer_dtype(attributeValues.dtype):
            # Replace Pandas NA values with -1
            attributeValues = pd.Series(attributeValues).fillna(-1).values
        
        # Transfer the attribute values to 'poly_data'
        attributeArrays[attr][valid_canopy_indices] = attributeValues
        poly_data.point_data[attr] = attributeArrays[attr]


        print(f'Unique values in poly_data.point_data[{attr}] after transfer: {np.unique(attributeArrays[attr])}')
    
    print('canopy projection attributes transferred successfuly')


    return poly_data


def update_flattened_site(poly_data, site):

    #EXTRA
    print(f'getting csvs for {site}...')
    poly_data = stamp_csv.stampCsvs(poly_data, site)
    print(f'stamped csvs for {site}')


    #PYLONS AND STREETLIGHTS
    print(f'getting pylons for {site}')
    poly_data = create_additions.create_pylon_tower(poly_data)
    print(f'got pylons for {site}')
    
    #TREE FOOTPRINTS
    tree_positions = trees.process_urban_forest(poly_data, site, 'current')
    poly_data = get_canopies(poly_data, tree_positions)
 
    #ELEV
    print(f'getting topos for {site}')
    topo_path = f'data/{site}/topography.vtm'
    topo = pv.read(topo_path)

    poly_data = stamp_cloud.getElev(topo, poly_data)
    print(f'found elevation for {site}')

    poly_data = stamp_cloud.calculate_distance_to_habitable_zone(poly_data)

    #ADDITIONAL SHAPEFILES
    
    shapefiles_delete_attributes = [
        'data/deployables/shapefiles/parkingmedian/parking_median_buffer.shp',
        'data/deployables/shapefiles/private/deployables_private_empty_space.shp',
        'data/deployables/shapefiles/little_streets/little_streets.shp'

    ]

    #stamp_shapefile.read_and_plot(site, poly_data, shapefiles_delete_attributes, 1000,deleteAttributes=True)
    stamp_shapefile.read_and_plot(site, poly_data, shapefiles_delete_attributes, 1000,deleteAttributes=False)

    shapefiles_preserve_attributes = [
        'data/deployables/shapefiles/road_info/road_segments.shp',
        'data/deployables/shapefiles/laneways/laneways-greening.shp'
    ]

    stamp_shapefile.read_and_plot(site, poly_data, shapefiles_preserve_attributes, 1000)


    return poly_data

    

if __name__ == "__main__":
    #sites = ['trimmed-parade', 'street', 'city']
    #sites = ['city', 'trimmed-parade']
    #sites = ['city']
    sites = ['trimmed-parade']
    for site in sites:

        
        
        vtk_path = f'data/{site}/flattened-{site}.vtk'
        poly_data = pv.read(vtk_path)

        poly_data = update_flattened_site(poly_data, site)

        print(f'calculating disturbances....')

        poly_data = calculate_disturbances.calculate_disturbance_groups(poly_data, site)
        


        poly_data.save(f'data/{site}/updated-{site}-2.vtk')
        print(f'saved {site}')

         # Visualization logic
        """plotter = pv.Plotter()


        cube = pv.Cube()  # Create a cube geometry for glyphing
        glyphs = poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
        #plotter.add_mesh(glyphs, scalars = 'solar', cmap = 'jet')
        #plotter.add_mesh(glyphs, scalars = 'extensive_green_roof-RATING', cmap = 'tab20')
        plotter.add_mesh(glyphs, scalars = 'ispylons', cmap = 'tab20')


        # Settings for better visualization
        plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
        light2 = pv.Light(light_type='cameralight', intensity=.5)
        light2.specular = 0.5  # Reduced specular reflection
        plotter.add_light(light2)
        plotter.enable_eye_dome_lighting()
        plotter.show()"""
        
        


