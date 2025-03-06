import os
import xarray as xr
import pyvista as pv
import numpy as np
import pandas as pd

def update_xarray_with_subset(xarray_dataset, subset_xarray):
    print(f"Updating xarray dataset with subset data...")
    for var in subset_xarray.data_vars:
        if var in xarray_dataset:
            print(f"Updating existing variable: {var}")
            xarray_dataset[var] = subset_xarray[var]
        else:
            print(f"Adding new variable: {var}")
            xarray_dataset = xarray_dataset.assign({var: subset_xarray[var]})
    print("Xarray dataset update complete.")
    return xarray_dataset


def create_subset_dataset(ds, variables, attributes):
    """
    Create a subset of the xarray Dataset by dropping all variables not in the specified list
    and keeping only the specified attributes.
    
    Parameters:
    ds (xarray.Dataset): The original Dataset
    variables (list): List of variable names to keep
    attributes (list): List of attribute names to keep
    
    Returns:
    xarray.Dataset: A new Dataset with only the specified variables and attributes
    """
    # Create a copy of the original dataset
    subset_ds = ds.copy(deep=True)
    
    # List variables found and not found in ds
    variables_found = [var for var in variables if var in ds.variables]
    variables_not_found = [var for var in variables if var not in ds.variables]
    
    print("Variables found in dataset:", variables_found)
    print("Variables not found in dataset:", variables_not_found)
    
    # Get the list of variables to drop
    vars_to_drop = [var for var in subset_ds.variables if var not in variables and var not in subset_ds.dims]
    
    # Drop the variables not in the list
    subset_ds = subset_ds.drop_vars(vars_to_drop)
    
    # Keep only the specified attributes
    subset_ds.attrs = {k: v for k, v in subset_ds.attrs.items() if k in attributes}
    
    return subset_ds


def GetResourseColors():
    # Define the colors dictionary
    colours = {
        "other": "#B4B4B4",
        "perch branch": "#FC8E62",
        "dead branch": "#8AA1CC",
        "peeling bark": "#FFDA2B",
        "epiphyte": "#A7D953",
        "fallen log": "#E6C595",
        "hollow": "#F782C0"
    }
    return colours


def convert_xarray_into_polydata(xarray_dataset):
    # Create PyVista PolyData
    print("Creating PyVista PolyData object...")

    # Extract points from 'centroid_x', 'centroid_y', 'centroid_z'
    try:
        points = np.vstack((
            xarray_dataset['centroid_x'].values,
            xarray_dataset['centroid_y'].values,
            xarray_dataset['centroid_z'].values
        )).T  # Shape: (num_voxels, 3)
    except KeyError as e:
        print(f"Missing required coordinate data in xarray dataset: {e}")
        return

    # Create the PolyData object
    polydata = pv.PolyData(points)

    # Add all other variables as point data
    # Exclude 'centroid_x', 'centroid_y', 'centroid_z' as they are used as coordinates
    print("Adding point data to PolyData...")
    for var in xarray_dataset.data_vars:
        if var not in ['centroid_x', 'centroid_y', 'centroid_z']:
            data = xarray_dataset[var].values
            if data.dtype.kind in {'S', 'U'}:
                # Convert bytes or unicode to string
                data = data.astype(str)
            polydata.point_data[var] = data

    return polydata


def initialize_xarray_variables_generic_auto(xarray_dataset, df, column_names, prefix):
    """
    Automatically initializes variables in an xarray dataset using inferred data types
    from a pandas DataFrame for specified columns.
    
    Parameters:
        xarray_dataset (xr.Dataset): The xarray dataset to update.
        df (pd.DataFrame): The DataFrame containing the data.
        column_names (list): List of column names to initialize in the xarray.
        prefix (str): The prefix to add to the variable names (e.g., 'trees_' or 'poles_').
    
    Returns:
        xr.Dataset: Updated xarray dataset with initialized variables.
    """
    for col in column_names:
        variable_name = f"{prefix}{col}"
        dtype = infer_dtype(df[col])
        
        # Initialize based on the inferred dtype
        if dtype == float:
            xarray_dataset[variable_name] = (('voxel',), np.full(xarray_dataset.sizes['voxel'], np.nan))
        elif dtype == int:
            xarray_dataset[variable_name] = (('voxel',), np.full(xarray_dataset.sizes['voxel'], -1))
        elif dtype == bool:
            xarray_dataset[variable_name] = (('voxel',), np.full(xarray_dataset.sizes['voxel'], False))
        elif dtype == str:
            xarray_dataset[variable_name] = (('voxel',), np.full(xarray_dataset.sizes['voxel'], 'unassigned'))

    return xarray_dataset

# Redefine the function to infer dtype from pandas Series
def infer_dtype(series):
    """
    Infers the dtype for initializing values based on a pandas Series.
    
    Parameters:
        series (pd.Series): The pandas Series to infer the dtype from.
        
    Returns:
        type: The inferred dtype (float, int, bool, or str).
    """
    if pd.api.types.is_float_dtype(series):
        return float
    elif pd.api.types.is_integer_dtype(series):
        return int
    elif pd.api.types.is_bool_dtype(series):
        return bool
    else:
        return str
    


if __name__ == "__main__":
    # Ask for voxel size and site
    voxel_size = input("Enter the voxel size: ")
    site = input("Enter the site name: ")

    # Construct the input folder path
    input_folder = f'data/revised/final/{site}'

    # Find matching .nc files
    matching_files = []
    for f in os.listdir(input_folder):
        if f.endswith('.nc'):
            parts = f.split('_', 2)
            if len(parts) >= 2:
                file_site, file_voxel_size = parts[:2]
                if file_site == site and file_voxel_size == voxel_size:
                    matching_files.append(f)

    # Print matching files with numbers
    if not matching_files:
        print("No matching files found.")
    else:
        print("Matching files:")
        for i, file in enumerate(matching_files, 1):
            print(f"{i}. {file}")

        # Query which file to load
        file_number = int(input("Enter the number of the file to load: "))
        if 1 <= file_number <= len(matching_files):
            selected_file = matching_files[file_number - 1]
            xarray_path = os.path.join(input_folder, selected_file)

            # Load the xarray dataset
            xarray_dataset = xr.open_dataset(xarray_path)

            # Print all xarray variables with numbers
            print("\nXarray variables:")
            variables = list(xarray_dataset.variables)
            for i, var in enumerate(variables, 1):
                print(f"{i}. {var}")

            # Modified variable selection prompt
            while True:
                var_input = input("\nEnter the number of the variable to analyze (or press enter to save the vtk): ")
                
                if var_input == "":
                    # Convert xarray to PyVista PolyData
                    polydata = convert_xarray_into_polydata(xarray_dataset)
                    
                    # Create VTK filename by replacing .nc extension with .vtk
                    vtk_path = xarray_path.replace('.nc', '.vtk')
                    
                    # Save the VTK file
                    print(f"Saving VTK file to: {vtk_path}")
                    polydata.save(vtk_path)
                    print("VTK file saved successfully.")
                    break
                
                try:
                    var_number = int(var_input)
                    if 1 <= var_number <= len(variables):
                        selected_var = variables[var_number - 1]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(variables)}.")
                except ValueError:
                    if var_input != "":  # Only show error if input wasn't empty
                        print("Please enter a valid number or press enter to save the vtk.")

            # Continue with existing code for analysis and plotting if a variable was selected
            if var_input != "":
                # Print unique elements and their counts for the selected variable
                unique_elements, counts = np.unique(xarray_dataset[selected_var].values, return_counts=True)
                print(f"\nUnique elements and counts for {selected_var}:")
                for elem, count in zip(unique_elements, counts):
                    print(f"{elem}: {count}")

                # Ask if user wants to plot in PyVista
                plot_choice = input("\nDo you want to plot this variable in PyVista? (y/n): ").lower()
                if plot_choice == 'y':
                    # Convert xarray to PyVista PolyData
                    polydata = convert_xarray_into_polydata(xarray_dataset)

                    # Create PyVista plotter
                    plotter = pv.Plotter()
                    #enable eyedome lighting
                    plotter.enable_eye_dome_lighting()
                    plotter.add_mesh(polydata, scalars=selected_var, cmap='viridis', show_edges=False)
                    plotter.show()
                else:
                    print("Plotting cancelled.")
