import pyvista as pv
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from scipy.stats import mode
from scipy.spatial import cKDTree



import rReadDXF as objStamper
#import rStampShapefile as geoStamper
import rGeoStamp as geoStamper


##INITIAL FUNCTIONS
def vtk_to_xarrayOLD(filename):
    """Convert VTK file to xarray Dataset."""
    mesh = pv.read(filename)

    # Print all point_data attribute names
    print("Point data attributes:")
    for attr_name in mesh.point_data.keys():
        print(attr_name)
    print()  # Add a blank line after printing attribute names

    points = mesh.points
    data_dict = {'x': ('point', points[:, 0]),
                 'y': ('point', points[:, 1]),
                 'z': ('point', points[:, 2])}
    
    for key, value in mesh.point_data.items():
        if isinstance(value, pv.pyvista_ndarray):
            if value.ndim == 1:
                data_dict[key] = ('point', value)
            elif value.ndim == 2:
                for i in range(value.shape[1]):
                    data_dict[f'{key}_{i}'] = ('point', value[:, i])
            else:
                print(f"Skipping {key} as it has unexpected dimensions: {value.ndim}")
        else:
            data_dict[key] = ('point', value)
    
    dataset = xr.Dataset(data_dict)
    
    # Set x, y, z as coordinates
    dataset = dataset.set_coords(['x', 'y', 'z'])

    return dataset

def vtk_to_xarray(polydata):
    """Convert VTK file to xarray Dataset."""

    # Print all point_data attribute names
    print("Point data attributes:")
    for attr_name in polydata.point_data.keys():
        print(attr_name)
    print()  # Add a blank line after printing attribute names

    points = polydata.points
    data_dict = {'x': ('point_index', points[:, 0]),
                 'y': ('point_index', points[:, 1]),
                 'z': ('point_index', points[:, 2])}
    
    for key, value in polydata.point_data.items():
        if isinstance(value, pv.pyvista_ndarray):
            if value.ndim == 1:
                data_dict[key] = ('point_index', value)
            elif value.ndim == 2:
                for i in range(value.shape[1]):
                    data_dict[f'{key}_{i}'] = ('point_index', value[:, i])
            else:
                print(f"Skipping {key} as it has unexpected dimensions: {value.ndim}")
        else:
            data_dict[key] = ('point_index', value)
    
    dataset = xr.Dataset(data_dict)
    
    # Set 'point_index' as the coordinate
    dataset = dataset.assign_coords(point_index=('point_index', np.arange(len(points))))

    return dataset

import xarray as xr
import numpy as np




def safe_eval(expr, dataset):
    try:
        return eval(expr, {"dataset": dataset, "np": np})
    except KeyError as e:
        print(f"Warning: Missing variable in condition, treating as False: {e}")
        return xr.full_like(dataset['x'], False, dtype=bool)

def search(dataset, conditions, condition_name='condition', isPlot=False):
    """print("Variables in dataset:")
    for var in dataset.variables:
        print(var)
    print("\n")"""

    print(f'Searching for {condition_name}')

    # Check if the condition_name exists and reset its contents
    if condition_name in dataset:
        print(f"'{condition_name}' exists in the dataset. Clearing its contents.")
        del dataset[condition_name]

    # Initialize the classification column with a default value based on the conditions
    first_key = next(iter(conditions.keys()))

    if first_key.lower() in ['true', 'false']:  # Boolean type
        default_value = False
        dtype = 'bool'
    elif first_key.isdigit() or first_key.replace('.', '', 1).isdigit():  # Numeric type
        default_value = np.nan
        dtype = 'float64'
    else:  # Assume string type
        default_value = 'unclassified'
        dtype = 'object'

    classification = xr.full_like(dataset['x'], default_value, dtype=dtype)
    
    for category, condition in conditions.items():
        try:
            print(f"Evaluating condition for '{category}': {condition}")

            # Break down the condition to ensure each part can be safely evaluated
            condition_parts = condition.split("|")
            mask = xr.full_like(dataset['x'], False, dtype=bool)

            for part in condition_parts:
                part = part.strip()
                part_mask = safe_eval(part, dataset)
                mask = mask | part_mask  # Combine with OR logic

            print(f"Mask for {category}: {mask.sum().item()} True values out of {mask.size} total")

            if dtype == 'object':
                classification = xr.where(mask, str(category), classification)
            elif dtype == 'bool':
                classification = xr.where(mask, bool(category == 'True'), classification)
            elif dtype == 'float64':
                classification = xr.where(mask, float(category), classification)

            print(f"Unique values in classification after {category}: {np.unique(classification)}")
        except Exception as e:
            print(f"Warning: An error occurred while processing condition for '{category}'. Skipping. Error: {e}")
    
    dataset[condition_name] = classification
    print(f"Final unique values in {condition_name}: {np.unique(dataset[condition_name])}")

    if isPlot:
        plotXarray(dataset, condition_name)

    return dataset

def plotXarray(dataset, scalar):

    # Extract x, y, z coordinates
    x = dataset['x'].values
    y = dataset['y'].values
    z = dataset['z'].values
    
    # Create points array
    points = np.column_stack((x, y, z))
    
    # Create polydata
    polydata = pv.PolyData(points)

    polydata.point_data[scalar] = dataset[scalar].values
    
    # Add all other variables as point data
    for var in dataset.data_vars:
        if var not in ['x', 'y', 'z']:
            polydata.point_data[var] = dataset[var].values
    
    # Plot the polydata as a point cloud with the default scalar field
    print("Plotting polydata as point cloud...")
    plotter = pv.Plotter()
    plotter.add_mesh(polydata, render_points_as_spheres=True, point_size=5, scalars = scalar, cmap = 'Set1')
    
    # Set up the camera for a top-down view
    plotter.view_xy()
    
    # Display the plot
    plotter.show()


def search2(dataset, conditions, condition_name='condition'):
    print("Variables in dataset:")
    for var in dataset.variables:
        print(var)
    print("\n")

    # Check if the condition_name exists and reset its contents
    if condition_name in dataset:
        print(f"'{condition_name}' exists in the dataset. Clearing its contents.")
        del dataset[condition_name]

    # Detect the type of the first key in conditions to initialize the classification array correctly
    first_key = next(iter(conditions.keys()))

    # Initialize the classification column based on the inferred type
    if first_key.lower() in ['true', 'false']:  # Boolean type
        default_value = False
        dtype = 'bool'
    elif first_key.isdigit() or first_key.replace('.', '', 1).isdigit():  # Numeric type
        default_value = np.nan
        dtype = 'float64'
    else:  # Assume string type
        default_value = 'unclassified'
        dtype = 'object'

    # Initialize the classification column with the appropriate default value
    classification = xr.full_like(dataset['x'], default_value, dtype=dtype)
    
    for category, condition in conditions.items():
        try:
            print(f"Evaluating condition for '{category}': {condition}")
            mask = eval(condition)
            print(f"Mask for {category}: {mask.sum().item()} True values out of {mask.size} total")
            
            # Ensure consistent type when updating the classification
            if dtype == 'object':
                classification = xr.where(mask, str(category), classification)
            elif dtype == 'bool':
                classification = xr.where(mask, bool(category == 'True'), classification)
            elif dtype == 'float64':
                classification = xr.where(mask, float(category), classification)
            
            print(f"Unique values in classification after {category}: {np.unique(classification)}")
        except NameError as e:
            print(f"Warning: Condition for '{category}' could not be evaluated. Skipping. Error: {e}")
        except KeyError as e:
            print(f"Warning: Attribute not found for condition '{category}'. Skipping. Error: {e}")
        except Exception as e:
            print(f"Warning: An error occurred while processing condition for '{category}'. Skipping. Error: {e}")
    
    # Assign the classification to the dataset, overwriting the old variable
    dataset[condition_name] = classification

    print(f"Final unique values in {condition_name}: {np.unique(dataset[condition_name])}")

    plotXarray(dataset, condition_name)
    return dataset

def plot_site(dataset, hue_column):
    # Convert the xarray Dataset to a pandas DataFrame
    df = dataset.to_dataframe().reset_index()

    # Create a scatter plot with 'x' and 'y' as coordinates and color by the specified hue column
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue=hue_column, palette='viridis', edgecolor='none')

    # Enhance plot aesthetics
    plt.title('Urban Systems Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(title=hue_column)
    plt.grid(True)

    # Show the plot
    plt.show()

def save_xarray_to_netcdf(dataset, filepath, downsample=False):
    """
    Save an xarray Dataset to a NetCDF file, with an option to save a downsampled version.
    Logs specific columns causing data type compatibility issues.

    Args:
    dataset (xarray.Dataset): The xarray Dataset to save.
    filepath (str): The path where the NetCDF file will be saved.
    downsample (bool): If True, also save a downsampled version with every 10th point.

    Returns:
    None
    """
    try:
        print("Attempting to save dataset with individual variable checks:")
        for var_name, data_array in dataset.data_vars.items():
            try:
                # Check conversion and simulate save to catch data type errors
                test_data_array = data_array.astype(np.int16) if data_array.dtype == np.uint8 else data_array
                dataset[var_name] = test_data_array  # Reassign to trigger any potential issues
            except Exception as e:
                print(f"Error processing {var_name}: {e}")

        # After checking each variable, attempt to save the entire dataset
        print(f"Saving xarray Dataset to NetCDF file using NetCDF4: {filepath}")
        dataset.to_netcdf(filepath, engine='h5netcdf')
        print(f"Dataset successfully saved to {filepath} using NetCDF4")

        if downsample:
            downsampled_filepath = filepath.replace('.nc', '_downsampled.nc')
            print(f"Saving downsampled xarray Dataset to NetCDF file: {downsampled_filepath}")
            downsampled_dataset = dataset.isel(point_index=slice(None, None, 10))
            downsampled_dataset.to_netcdf(downsampled_filepath, engine='h5netcdf')
            print(f"Downsampled dataset successfully saved to {downsampled_filepath}")

    except ImportError as imp_err:
        print(f"Failed to save dataset because the required library is missing: {imp_err}")
    except Exception as e:
        print(f"Failed to save dataset due to: {e}")


def plot_resources_and_spatial_coords(dsGrid, combined_resources):
    """
    Plot both the original combined_resources and the spatial coordinates from dsGrid using PyVista.
    
    Parameters:
    - combined_resources: PyVista PolyData object containing the original point cloud data
    - dsGrid: xarray Dataset containing the voxelized data with spatialX, spatialY, and spatialZ coordinates
    """
    print("Preparing to plot combined_resources and spatial coordinates from dsGrid...")

    # Extract spatial coordinates from dsGrid
    spatial_coords = np.column_stack((
        dsGrid.spatialX.values.ravel(),
        dsGrid.spatialY.values.ravel(),
        dsGrid.spatialZ.values.ravel()
    ))

    # Create a mask for populated voxels
    is_populated = dsGrid['isResource'].values.ravel()

    # Filter out unpopulated voxels
    spatial_coords_populated = spatial_coords[is_populated]

    # Create PolyData for spatial coordinates
    spatial_polydata = pv.PolyData(spatial_coords_populated)

    # Create a plotter
    plotter = pv.Plotter()

    # Plot the original combined_resources
    plotter.add_mesh(combined_resources, color='red', point_size=5, render_points_as_spheres=True, label="Original Data")

    # Plot the spatial coordinates
    plotter.add_mesh(spatial_polydata, color='blue', point_size=5, render_points_as_spheres=True, label="Voxel Centers")

    # Set up the legend
    plotter.add_legend()

    # Set up camera and display
    plotter.camera_position = 'xy'
    plotter.show_axes()
    plotter.show_grid()

    print("Showing the plot...")
    plotter.show()

    print("Finished plotting combined_resources and spatial coordinates.")

####VOXELISE CANOPY VTK

def encode_categories_numpy(categories):
    unique_categories, encoded_categories = np.unique(categories, return_inverse=True)
    category_mapping = {i: cat for i, cat in enumerate(unique_categories)}
    return encoded_categories, category_mapping

def extract_voxels_as_xarray_grid(site, voxel_size=1):
    print(f"Starting extractResourceGrid for site: {site} using voxel size: {voxel_size}")

    # Read the MultiBlock dataset
    siteStateMULTIBLOCK = pv.read(f'data/{site}/combined-{site}-now.vtm')
    print(f"Loaded MultiBlock dataset for {site}")

    treePOLYMULTIBLOCK = siteStateMULTIBLOCK.get('trees')
    if treePOLYMULTIBLOCK is None:
        raise ValueError(f"'trees' block not found in the MultiBlock dataset for site: {site}")
    print("Retrieved 'trees' from MultiBlock dataset")

    # Combine branches and ground resources
    combined_resources = treePOLYMULTIBLOCK['branches'] + treePOLYMULTIBLOCK['ground resources']
    print("Combined branches and ground resources")
    
    # Extract coordinates and attributes from combined_resources polydata
    points = combined_resources.points
    resources = combined_resources['resource']
    attributes = {name: combined_resources[name] for name in combined_resources.point_data.keys() if name != 'resource'}

    # Encode resources using numpy
    encoded_resource, resource_categories = encode_categories_numpy(resources)

    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    i_min, j_min, k_min = voxel_indices.min(axis=0)
    i_max, j_max, k_max = voxel_indices.max(axis=0)

    # Create the integer voxel coordinates
    i_coords = np.arange(i_min, i_max + 1)
    j_coords = np.arange(j_min, j_max + 1)
    k_coords = np.arange(k_min, k_max + 1)

    shape = (i_coords.size, j_coords.size, k_coords.size)

    # Initialize the xarray Dataset
    data_vars = {f'resource_{cat}': (('i', 'j', 'k'), np.zeros(shape, dtype=int)) for cat in resource_categories.values()}
    for attr_name in attributes.keys():
        data_vars[attr_name] = (('i', 'j', 'k'), np.full(shape, -1, dtype=attributes[attr_name].dtype))  # Initialize with -1

    dsGrid = xr.Dataset(data_vars, coords={'i': i_coords, 'j': j_coords, 'k': k_coords})

    # Efficiently count and populate resource occurrences per voxel
    i_idx = voxel_indices[:, 0] - i_min
    j_idx = voxel_indices[:, 1] - j_min
    k_idx = voxel_indices[:, 2] - k_min

    # Vectorized operation to populate the resource counts
    idx = np.ravel_multi_index((i_idx, j_idx, k_idx), dsGrid[f'resource_{resource_categories[0]}'].shape)
    for cat in np.unique(encoded_resource):
        resource_mask = (encoded_resource == cat)
        np.add.at(dsGrid[f'resource_{resource_categories[cat]}'].values.ravel(), idx[resource_mask], 1)

    # Get the first occurrence of each voxel index
    unique_indices, unique_positions = np.unique(idx, return_index=True)

    # Assign the first occurrence values for each attribute
    for attr_name in attributes.keys():
        flattened_attr = attributes[attr_name]
        dsGrid[attr_name].values.flat[unique_indices] = flattened_attr[unique_positions]

    # Calculate the spatial coordinates as floats, correctly adjusted by voxel size and bounds
    spatialX = i_coords * voxel_size + voxel_size / 2
    spatialY = j_coords * voxel_size + voxel_size / 2
    spatialZ = k_coords * voxel_size + voxel_size / 2

    # Update the coordinates of the dataset
    dsGrid = dsGrid.assign_coords(
        spatialX=('i', spatialX),
        spatialY=('j', spatialY),
        spatialZ=('k', spatialZ)
    )

    # Assign the isResource flag based on whether any resource is present in the voxel
    isResource = np.any([dsGrid[f'resource_{cat}'] > 0 for cat in resource_categories.values()], axis=0)
    dsGrid['isResource'] = (('i', 'j', 'k'), isResource)

    # Print final details for dsGrid and dsList
    print("Final dsGrid columns (variables):")
    print(list(dsGrid.data_vars.keys()))
    print(f"dsGrid shape (voxel grid shape): {dsGrid.sizes}")
    print(f"Total number of voxels in dsGrid: {dsGrid.sizes['i'] * dsGrid.sizes['j'] * dsGrid.sizes['k']}")

    print("\nComparing bounds of original points and voxelized grid:")
    print("Original polydata points bounds:")
    print(f"X: {points[:,0].min()} to {points[:,0].max()}")
    print(f"Y: {points[:,1].min()} to {points[:,1].max()}")
    print(f"Z: {points[:,2].min()} to {points[:,2].max()}")

    print("\nVoxelized grid spatial coordinates bounds:")
    print(f"X: {dsGrid.spatialX.min().item()} to {dsGrid.spatialX.max().item()}")
    print(f"Y: {dsGrid.spatialY.min().item()} to {dsGrid.spatialY.max().item()}")
    print(f"Z: {dsGrid.spatialZ.min().item()} to {dsGrid.spatialZ.max().item()}")

    return dsGrid, combined_resources


def flatten_to_xarray_list(dsGrid):
    # Identify populated voxels
    is_populated = dsGrid['isResource'].values

    # Get the number of populated voxels
    num_populated = np.sum(is_populated)
    print(f"Number of populated voxels: {num_populated}")

    # Create a meshgrid of the spatial coordinates
    xx, yy, zz = np.meshgrid(dsGrid['spatialX'], dsGrid['spatialY'], dsGrid['spatialZ'], indexing='ij')

    # Flatten the meshgrid and select only the populated voxels
    x_coords = xx[is_populated]
    y_coords = yy[is_populated]
    z_coords = zz[is_populated]

    # Initialize a dictionary to store the data variables for each populated voxel
    data_vars = {}

    # Iterate through the variables in dsGrid
    for var_name, da in dsGrid.data_vars.items():
        if var_name not in ['spatialX', 'spatialY', 'spatialZ']:
            # For each variable, flatten and select only the populated voxels
            flattened_data = da.values[is_populated]
            # Assign the flattened data to the 'point_index' dimension
            data_vars[var_name] = (['point_index'], flattened_data)

    # Add x, y, z coordinates as data variables
    data_vars['x'] = (['point_index'], x_coords)
    data_vars['y'] = (['point_index'], y_coords)
    data_vars['z'] = (['point_index'], z_coords)

    # Create a new xarray Dataset for the flattened data with 'point_index' as the dimension
    ds_flattened = xr.Dataset(
        data_vars=data_vars,
        coords={'point_index': np.arange(num_populated)}
    )

    return ds_flattened



def plot_dsList_as_polydata(dsList, site_poly):
    """
    Convert dsList (xarray Dataset) to PyVista PolyData and plot it.
    
    Parameters:
    - dsList: xarray Dataset with 'x', 'y', 'z' as coordinates.
    """
    print("Converting dsList to PyVista PolyData...")
    
    # Extract coordinates (x, y, z) from the dataset
    coords = np.column_stack((dsList.x.values, dsList.y.values, dsList.z.values))
    print(f"Coordinate shape: {coords.shape}")
    
    # Create PolyData object with coordinates
    polydata = pv.PolyData(coords)
    
    # Assign other attributes to PolyData
    for var_name in dsList.data_vars:
        if var_name not in ['x', 'y', 'z']:
            polydata.point_data[var_name] = dsList[var_name].values
    
    print("Converted dsList to PyVista PolyData.")
    
    # Create plotter
    plotter = pv.Plotter()
    # Add point cloud to plotter
    plotter.add_mesh(polydata, render_points_as_spheres=True, point_size=50, scalars = 'resource_dead branch')

    # Add site_poly to plotter
    if site_poly is not None:
        # Ensure site_poly has 'rgb' data
        if 'rgb' in site_poly.point_data:
            rgb = site_poly.point_data['rgb']
            # Normalize RGB values if they're not already in [0, 1] range
            if rgb.max() > 1:
                rgb = rgb / 255.0
            # Add site_poly to plotter with RGB colors
            plotter.add_mesh(site_poly, rgb=True, scalars=rgb, opacity=0.7)
        else:
            # If 'rgb' is not available, plot with a default color
            plotter.add_mesh(site_poly, color='lightgray', opacity=0.7)
        print("Added site_poly to the plot.")
    else:
        print("site_poly is None, skipping its visualization.")
    # Set up camera and display
    plotter.camera_position = 'xy'
    print("Showing the plot...")
    plotter.show()
    print("Finished plotting dsList.")


def voxelise_urban_forest(site):
    voxel_grid, orig_poly = extract_voxels_as_xarray_grid(site, voxel_size=1)
    voxel_list = flatten_to_xarray_list(voxel_grid)
    
    output_path = f'data/revised/{site}-forestVoxels.nc'
    print(f"Saving voxels xarray to {output_path}")
    save_xarray_to_netcdf(voxel_list, output_path, downsample=True)
    print(f"Saved voxels xarray to {output_path}")

def search_urban_elements(site_data):
    # Search Syntax Guide:
    # - Use 'dataset' to refer to the xarray Dataset
    # - Enclose the entire condition string in double quotes ("")
    # - Use single quotes ('') for string values within conditions
    # - Logical operators (&, |) go at the end of a condition, before the closing "
    # - Use '~' for NOT, placed inside the parentheses of the condition
    # - Group complex conditions with parentheses ()
    # - Use .isin() for multiple value checks
    # - Comparison operators: ==, !=, <, >, <=, >=
    #
    # Example breakdown:
    # "Condition Name": (
    #     "(dataset['column1'] == 'value1') & "  # Condition 1 with & at the end
    #     "((dataset['column2'] > 10) | "        # Opening ( for complex condition
    #     "(dataset['column3'].isin(['a', 'b', 'c']))) & "  # Closing )) and & for next condition
    #     "~(dataset['column4'] < 5)"            # NOT condition with ~ inside ()
    # )
    #
    # This example searches for:
    # 1. column1 equals 'value1' AND
    # 2. (column2 is greater than 10 OR column3 is one of 'a', 'b', or 'c') AND
    # 3. column4 is NOT less than 5
    #
    # The search function evaluates these conditions and classifies data accordingly,
    # handling boolean, numeric, and string classifications based on the condition format.
    
    plot = False
    condition_name = 'urban systems'
    
    urban_element_search = {
        "Footways": (
            "(dataset['road_types-type'] == 'Footway')"
        ),
        "Roads": (
            "(dataset['road_types-type'] == 'Carriageway')"
        ),
        "Adaptable Vehicle Infrastructure": (
            "(dataset['parkingmedian-isparkingmedian'] == True) | "
            "(dataset['disturbance-potential'].isin([4, 2, 3])) | "
            "(~(dataset['little_streets-islittle_streets'] == True))"
        ),
        "Street Pylons": (
            "dataset['isstreetlight'] | dataset['ispylons']"
        ),
        "Load Bearing Roofs": (
            "(dataset['buildings-dip'] >= 0.0) & "
            "(dataset['buildings-dip'] <= 0.1) & "
            "dataset['extensive_green_roof-RATING'].isin(['Excellent', 'Good', 'Moderate']) & "
            "(dataset['elevation'] >= -20) & "
            "(dataset['elevation'] <= 80)"
        ),
        "Lightweight Roofs": (
            "(dataset['buildings-dip'] >= 0.0) & "
            "(dataset['buildings-dip'] <= 0.1) & "
            "dataset['intensive_green_roof-RATING'].isin(['Excellent', 'Good', 'Moderate']) & "
            "(dataset['elevation'] >= -20) & "
            "(dataset['elevation'] <= 80)"
        ),
        "Ground Floor Facades": (
            "(dataset['buildings-dip'] >= 0.8) & "
            "(dataset['buildings-dip'] <= 1.7) & "
            "(dataset['solar'] >= 0.2) & "
            "(dataset['solar'] <= 1.0) & "
            "(dataset['elevation'] >= 0) & "
            "(dataset['elevation'] <= 10)"
        ),
        "Upper Floor Facades": (
            "(dataset['buildings-dip'] >= 0.8) & "
            "(dataset['buildings-dip'] <= 1.7) & "
            "(dataset['solar'] >= 0.2) & "
            "(dataset['solar'] <= 1.0) & "
            "(dataset['elevation'] > 10) & "
            "(dataset['elevation'] <= 80)"
        )
    }

    urban_element_search = {
        "Footways": (
            "(dataset['road_types-type'] == 'Footway')"
        ),
        "Roads": (
            "(dataset['road_types-type'] == 'Carriageway')"
        ),
        "Adaptable Vehicle Infrastructure": (
            "(dataset['parkingmedian-isparkingmedian'] == True) | "
            "(dataset['disturbance-potential'].isin([4, 2, 3])) | "
            "(~(dataset['little_streets-islittle_streets'] == True))"
        ),
        "Street Pylons": (
            "dataset['isstreetlight'] | dataset['ispylons']"
        ),
        "Load Bearing Roofs": (
            "(dataset['buildings-dip'] >= 0.0) & "
            "(dataset['buildings-dip'] <= 0.1) & "
            "dataset['extensive_green_roof-RATING'].isin(['Excellent', 'Good', 'Moderate']) & "
            "(dataset['elevation'] >= -20) & "
            "(dataset['elevation'] <= 80)"
        ),
        "Lightweight Roofs": (
            "(dataset['buildings-dip'] >= 0.0) & "
            "(dataset['buildings-dip'] <= 0.1) & "
            "dataset['intensive_green_roof-RATING'].isin(['Excellent', 'Good', 'Moderate']) & "
            "(dataset['elevation'] >= -20) & "
            "(dataset['elevation'] <= 80)"
        ),
        "Ground Floor Facades": (
            "(dataset['buildings-dip'] >= 0.8) & "
            "(dataset['buildings-dip'] <= 1.7) & "
            "(dataset['solar'] >= 0.2) & "
            "(dataset['solar'] <= 1.0) & "
            "(dataset['elevation'] >= 0) & "
            "(dataset['elevation'] <= 10)"
        ),
        "Upper Floor Facades": (
            "(dataset['buildings-dip'] >= 0.8) & "
            "(dataset['buildings-dip'] <= 1.7) & "
            "(dataset['solar'] >= 0.2) & "
            "(dataset['solar'] <= 1.0) & "
            "(dataset['elevation'] > 10) & "
            "(dataset['elevation'] <= 80)"
        )
    }

    site_data = search(site_data, urban_element_search, condition_name)


    condition_name = 'potential design conditions'
    potential_design_conditions_search = {
        "Rewilded Ground": (
            "(dataset['urban systems'] == 'Adaptable Vehicle Infrastructure')"
        ),
        "Existing Canopies Under Footways": (
            "(dataset['urban systems'] == 'Footways') & "
            "dataset['_Tree_size'].isin(['large', 'medium'])"
        ),
        "Existing Canopies Under Roads": (
            "(dataset['urban systems'] == 'Roads') & "
            "dataset['_Tree_size'].isin(['large', 'medium'])"
        ),
        "Existing Canopies Under Adaptable Vehicle Infrastructure": (
            "(dataset['urban systems'] == 'Adaptable Vehicle Infrastructure') & "
            "dataset['_Tree_size'].isin(['large', 'medium'])"
        ),
        "Potential Deployable Trees": (
            "dataset['isstreetlight'] | dataset['ispylons']"
        ),
        "Green Roofs": (
            "(dataset['urban systems'] == 'Load Bearing Roofs')"
        ),
        "Arid Log Roofs": (
            "(dataset['urban systems'] == 'Lightweight Roofs')"
        ),
        "Habitat-ready facades": (
            "(dataset['urban systems'] == 'Ground Floor Facades')"
        ),
        "Arid Facades": (
            "(dataset['urban systems'] == 'Upper Floor Facades')"
        )
    }

    site_data = search(site_data, potential_design_conditions_search, condition_name)
    return site_data

def refine_search(site_data):
    print("Starting refine_search function...")

    plotXarray(site_data, 'urban systems')

    condition_name = 'Potential pathway routing'
    new_pathways_search = {
        "True": (
        "((dataset['urban systems'] == 'Adaptable Vehicle Infrastructure') | "
        "(dataset['open_space-OS_STATUS'] == 'Existing') | "
        "(dataset['private-isprivate'] == True)) & "
        "~(dataset['canopy-iscanopy'] == 1)"
        )
        }
    
    new_pathways_search = {
        "True": (
        "((dataset['urban systems'] == 'Adaptable Vehicle Infrastructure') | "
        "(dataset['open_space-OS_STATUS'] == 'Existing') | "
        "(dataset['private-isprivate'] == True)) & "
        "~(dataset['canopy-iscanopy'] == 1)"
        )
        }
    
    site_data = search(site_data, new_pathways_search, condition_name)

    plotXarray(site_data, condition_name)



    new_pathways_search2 = {
        "True": (
            "((dataset['urban systems'] == 'Adaptable Vehicle Infrastructure') | "
            "(dataset['open_space-OS_STATUS'] == 'Existing') | "
            "(dataset['private-isprivate'] == True)) & "
            "(dataset['canopy-iscanopy'] != True)"
        )
    }
    print(f"Applying search condition: {new_pathways_search}")
    site_data = search(site_data, new_pathways_search, condition_name)

    #plotXarray(site_data, condition_name)

    return


    print("Initializing 'pathway routing' variable...")
    site_data['pathway routing'] = ('point_index', np.full(site_data.dims['point_index'], False))

    print("Getting coordinates of all points...")
    all_coords = np.column_stack((site_data['x'].values, site_data['y'].values, site_data['z'].values))
    print(f"Total number of points: {len(all_coords)}")

    print("Creating masks for canopy points and potential pathway points...")
    canopy_mask = site_data['urban systems'] == 'Existing Canopies Under Footways'
    potential_pathway_mask = site_data['Potential pathway routing'] == True
    
    print(f"Number of canopy points: {np.sum(canopy_mask)}")
    print(f"Number of potential pathway points: {np.sum(potential_pathway_mask)}")

    print("Creating KDTree for canopy points...")
    canopy_tree = cKDTree(all_coords[canopy_mask])

    print("Finding potential pathway points within 5m of canopy points...")
    distances, _ = canopy_tree.query(all_coords[potential_pathway_mask], k=1, distance_upper_bound=5)

    print("Getting indexes of potential pathway points within 5m of canopy points...")
    within_range_indexes = np.where(potential_pathway_mask)[0][distances < 5]
    print(f"Number of potential pathway points within 5m of canopy points: {len(within_range_indexes)}")

    print("Setting 'pathway routing' = True for these points...")
    site_data['pathway routing'].values[within_range_indexes] = True

    print("Refine_search function completed.")

    plotXarray(site_data, 'pathway routing')
    return site_data


        


def stampShapeFiles(site, site_poly):
    
    shapefiles_delete_attributes = [
        'data/revised/shapefiles/Aparkingmedian/parking_median_buffer.shp',
        'data/revised/shapefiles/Aprivate/deployables_private_empty_space.shp',
    ]
    #stamp_shapefile.read_and_plot(site, poly_data, shapefiles_delete_attributes, 1000,deleteAttributes=True)
    geoStamper.read_and_plot(site, site_poly, shapefiles_delete_attributes, 1000, deleteAttributes=True)

    return site_poly


def saveXarraytoPolydata(dataset, site):
    """
    Export the xarray dataset as a polydata to a VTK file.
    
    Args:
    dataset (xarray.Dataset): The xarray Dataset to export.
    site (str): The name of the site.
    default_scalar_field (str, optional): The name of the default scalar field to set.
    
    Returns:
    None
    """
    print(f"Exporting polydata for site: {site}")
    
    # Extract x, y, z coordinates
    x = dataset['x'].values
    y = dataset['y'].values
    z = dataset['z'].values
    
    # Create points array
    points = np.column_stack((x, y, z))
    
    # Create polydata
    polydata = pv.PolyData(points)
    
    # Add all other variables as point data
    for var in dataset.data_vars:
        if var not in ['x', 'y', 'z']:
            polydata.point_data[var] = dataset[var].values
    
    # Check if 'parkingmedian-isparkingmedian' is in point_data
    if 'parkingmedian-isparkingmedian' in polydata.point_data:
        print("'parkingmedian-isparkingmedian' is present in point_data")
        
        # Get unique values and their counts
        unique_values, counts = np.unique(polydata.point_data['parkingmedian-isparkingmedian'], return_counts=True)
        
        print("Unique values and their counts in 'parkingmedian-isparkingmedian':")
        for value, count in zip(unique_values, counts):
            print(f"{value}: {count}")
    else:
        print("'parkingmedian-isparkingmedian' is not present in point_data")


    # Plot the polydata as a point cloud with the default scalar field
    """print("Plotting polydata as point cloud...")
    plotter = pv.Plotter()
    plotter.add_mesh(polydata, render_points_as_spheres=True, point_size=5)
    
    # Set a default scalar bar if a default scalar field is provided
    if default_scalar_field and default_scalar_field in polydata.point_data:
        plotter.add_scalar_bar(title=default_scalar_field)
    
    # Set up the camera for a top-down view
    plotter.view_xy()
    
    # Display the plot
    plotter.show()"""

    print('saving polydata...')


    # Create the output directory if it doesn't exist
    output_dir = 'data/revised/processed'
    # Save the polydata to a VTK file
    output_file = f'{output_dir}/{site}-preview3.vtk'
    polydata.save(output_file)
    
    print(f"Polydata exported to: {output_file}")

def fix_trimmed_parade(site, site_data):
    site_data = objStamper.get_obj_properties(site, site_data, 'parking')
    

    field_name = 'parkingmedian-isparkingmedian'
    
    search_conditions = {
        "True": (
            "(dataset['parking-isparking'] == True) | "
            "(dataset['road_types-type'] == 'Median')"
        )
    }

    site_data = search(site_data, search_conditions, field_name)

    site_data = objStamper.get_obj_properties(site, site_data, 'private')

    return site_data

def create_data_info(site_data, site):
    import json
    import numpy as np
    import xarray as xr

    def safe_str(x):
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return 'NaN'
        elif x is None:
            return 'None'
        else:
            return str(x)

    def is_continuous(data):
        # Check if the data is likely continuous
        unique_values = np.unique(data)
        return len(unique_values) > 20 and np.issubdtype(data.dtype, np.number)

    data_info = {}
    for var in site_data.data_vars:
        if site_data[var].ndim == 1:
            if is_continuous(site_data[var].values):
                # For continuous variables, just include the variable name
                data_info[var] = "continuous"
            else:
                # For discrete variables, include the value counts
                values, counts = np.unique(site_data[var].values, return_counts=True)
                data_info[var] = {safe_str(k): int(v) for k, v in zip(values, counts)}

    output_path = f'info/info_{site}.json'
    with open(output_path, 'w') as f:
        json.dump(data_info, f, indent=2)

    print(f"Data info saved to {output_path}")


def export_processed_data(classified_dataset, site, default_scalar_field=None):
    output_filename = f'data/revised/{site}-processed.nc'
    print(f"Processed data for {site} saved to {output_filename}")
    save_xarray_to_netcdf(classified_dataset, output_filename, downsample=True)
    print('saving  polydata...')
    saveXarraytoPolydata(classified_dataset, site)
    print('saving data info...')
    create_data_info(classified_dataset, site)




def process_site(site):
    #voxelise the urban forest and save it to f'data/revised/{site}-forestVoxels.nc'
    #voxelise_urban_forest(site)

    #load polydata
    site_poly = pv.read(f'data/{site}/updated-{site}-2.vtk')

    #add extra GIS
    #site_poly = stampShapeFiles(site, site_poly)

    #convert site into xarray
    site_data = vtk_to_xarray(site_poly)

    if site == 'trimmed-parade':
        site_data = fix_trimmed_parade(site, site_data)
        
    #search for urban elements
    site_data = search_urban_elements(site_data)

    #site_data = refine_search(site_data)
        
    #save site xarray to f'data/revised/{site}-processed.nc'
    export_processed_data(site_data, site)



def main():
    sites = ['street', 'city', 'trimmed-parade']
    sites = ['trimmed-parade']
    
    import os
    if not os.path.exists('data/revised'):
        os.makedirs('data/revised')

    for site in sites:
        process_site(site)

if __name__ == "__main__":
    main()

