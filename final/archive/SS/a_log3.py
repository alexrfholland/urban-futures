import pandas as pd
import pickle
import os

import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ast
import pickle
import a_helper_functions



resource_names = [
        'perch branch', 'peeling bark', 'dead branch', 'other',
        'fallen log', 'leaf litter', 'epiphyte', 'hollow', 'leaf cluster'
    ]


brownRoof = ['fallen log', 
             'leaf litter',
             'dead branch']
greenRoof = ['epiphyte', 
             'leaf cluster', 
             'perch branch',
             'other']
greenFacade = ['leaf cluster', 
               'other', 
               'epiphyte']


#green roof: dead load 300kg
    #100kg for growing medium
    #100kg for live plants ['epiphyte, leaf cluster', 'perch branch]
    #50 kg for dead logs ['fallen log']

#brown roof: dead load 150kg
    #100 kg for dead logs ['fallen log']
    #50kg for growing medium  ['leaf litter']


def calculate_roof_voxels(xarray_dataset):
    """
    Processes the xarray dataset to calculate roof types and compute roof and log loads.
    
    Parameters:
        xarray_dataset (xr.Dataset): Input xarray dataset.
    
    Returns:
        xarray.Dataset: Updated xarray dataset with 'site_roofType' and new envelope attributes.
        pd.DataFrame: DataFrame containing grouped roof information with loads.
    """
    # Initialize 'site_roofType' with 'none'
    roof_types = ['none', 'green roof', 'brown roof']
    max_length = max(len(s) for s in roof_types)
    print("Assigning initial roof types...")
    new_site_roofType = np.full(xarray_dataset.sizes['voxel'], 'none', dtype=f'U{max_length}')
    xarray_dataset = xarray_dataset.assign(site_roofType=('voxel', new_site_roofType))
    
    # Initialize new envelope attributes
    max_length = max(len(s) for s in ['green roof', 'brown roof', 'none'])
    xarray_dataset['envelope_roofType'] = xr.DataArray(np.full(xarray_dataset.sizes['voxel'], 'none', dtype=f'U{max_length}'), dims='voxel')
    xarray_dataset['envelope_assigned_log'] = xr.DataArray(np.full(xarray_dataset.sizes['voxel'], -1, dtype=int), dims='voxel')
    xarray_dataset['envelope_logSize'] = xr.DataArray(np.full(xarray_dataset.sizes['voxel'], np.nan, dtype=object), dims='voxel')
    xarray_dataset['envelope_logID'] = xr.DataArray(np.full(xarray_dataset.sizes['voxel'], -1, dtype=int), dims='voxel')
    
    # Filter voxels based on roof ratings
    print("Filtering voxels based on roof ratings...")
    green_roof_mask = xarray_dataset.site_greenRoof_ratingInt >= 4
    brown_roof_mask = (xarray_dataset.site_brownRoof_ratingInt > 1) & (xarray_dataset.site_greenRoof_ratingInt < 4)
    xarray_dataset['envelope_roofType'] = xr.where(green_roof_mask, 'green roof', xarray_dataset.envelope_roofType)
    xarray_dataset['envelope_roofType'] = xr.where(brown_roof_mask, 'brown roof', xarray_dataset.envelope_roofType)
    xarray_dataset['site_roofType'] = xr.where(green_roof_mask, 'green roof', xarray_dataset.site_roofType)
    xarray_dataset['site_roofType'] = xr.where(brown_roof_mask, 'brown roof', xarray_dataset.site_roofType)
    
    # Verify roof types
    unique_roof_types = pd.Series(xarray_dataset['site_roofType'].values).unique()
    print("Unique roof types after assignment:", unique_roof_types)
    
    # Flatten the xarray into a DataFrame and filter out 'none' roof types
    print("Flattening xarray dataset...")
    flattened_df = xarray_dataset.to_dataframe().reset_index()
    flattened_df['Voxel coordinate'] = flattened_df['voxel']
    roof_info_df = flattened_df[flattened_df['site_roofType'] != 'none'].copy()
    
    # Create roofID and group by building and roof type
    print("Grouping roof information...")
    print(roof_info_df)
    roof_info_df['roofID'] = roof_info_df.groupby(['site_building_ID', 'site_roofType']).ngroup()
    grouped_roof_info = roof_info_df.groupby('roofID').agg(
        site_building_ID=('site_building_ID', 'first'),
        site_roofType=('site_roofType', 'first'),
        voxel_coordinates=('Voxel coordinate', list),
        centroid_x=('centroid_x', list),
        centroid_y=('centroid_y', list),
        centroid_z=('centroid_z', list),
        number_of_voxels=('site_building_ID', 'count')
    ).reset_index()
    
    # Calculate roof loads
    print("Calculating roof and log loads...")
    green_roof_dead_load = 300  # kg/m^2
    brown_roof_dead_load = 150  # kg/m^2
    voxel_area = xarray_dataset.attrs['voxel_size'] * xarray_dataset.attrs['voxel_size'] # m^2
    grouped_roof_info['Roof load'] = grouped_roof_info['number_of_voxels'] * voxel_area * grouped_roof_info['site_roofType'].map({
        'green roof': green_roof_dead_load,
        'brown roof': brown_roof_dead_load
    })
    
    # Calculate log loads
    green_roof_log_load = 50  # kg for green roof logs
    brown_roof_log_load = 100  # kg for brown roof logs
    grouped_roof_info['Log load'] = grouped_roof_info['number_of_voxels'] * voxel_area * grouped_roof_info['site_roofType'].map({
        'green roof': green_roof_log_load,
        'brown roof': brown_roof_log_load
    })
    
    print("Roof voxel calculation completed.")
    return xarray_dataset, grouped_roof_info, roof_info_df

def assign_logs(grouped_roof_info, log_library_df, xarray_dataset):
    """
    Assigns logs to roof voxels based on load requirements and tracks the total roof load and log load.
    Also assigns envelope attributes to the xarray dataset.
    
    Parameters:
        grouped_roof_info (pd.DataFrame): DataFrame containing grouped roof information with loads.
        log_library_df (pd.DataFrame): DataFrame containing log library statistics.
        xarray_dataset (xarray.Dataset): xarray dataset containing voxel data.
    
    Returns:
        pd.DataFrame: DataFrame with log assignments, including bounding boxes and load information.
        xarray.Dataset: Updated xarray dataset with envelope attributes.
    """
    print("Assigning logs to roof voxels...")
    
    # Ensure LogNo in log_library_df is treated as a string for consistency
    log_library_df['LogNo'] = log_library_df['LogNo'].astype(str)
    
    # Initialize columns for logs and assigned translations
    grouped_roof_info['Included logs'] = [[] for _ in range(len(grouped_roof_info))]
    grouped_roof_info['Assigned voxel indexes'] = [[] for _ in range(len(grouped_roof_info))]
    grouped_roof_info['Log load assigned'] = 0.0  # Track how much load the logs account for
    grouped_roof_info['Remaining roof load'] = grouped_roof_info['Roof load']  # Track remaining load after assigning logs
    
    # Create a mapping of LogNo to LogSize
    log_size_map = dict(zip(log_library_df['LogNo'], log_library_df['LogSize']))
    

    # Initialize 'envelope_roofID' with -1 to indicate no roofID initially
    xarray_dataset['envelope_roofID'] = xr.DataArray(np.full(xarray_dataset.sizes['voxel'], -1, dtype=int), dims='voxel')

    
    # Iterate over each row in grouped_roof_info
    for idx, row in grouped_roof_info.iterrows():
        print(f"\n--- Processing RoofID {row['roofID']} ---")
        total_roof_load = row['Roof load']
        remaining_log_load = row['Log load']
        roof_id = row['roofID']
        print(f"Total roof load: {total_roof_load}, Remaining log load: {remaining_log_load}")
        
        # Copy log_library_df to allow removing used logs
        available_logs = log_library_df.copy()
        print(f"Available logs: {len(available_logs)} logs in total")

        # Assign roofID to the voxels corresponding to this roof
        roof_id = row['roofID']
        voxel_indexes = row['voxel_coordinates']
        xarray_dataset['envelope_roofID'].loc[dict(voxel=voxel_indexes)] = roof_id
        
        log_load_assigned = 0  # Track the log load assigned to this roof

        
        
        # Assign logs to the roof based on remaining load
        while remaining_log_load > 0 and not available_logs.empty:
            selected_log = available_logs.sample(1)
            log_biomass = selected_log['Biomass_kg'].values[0]
            log_id = selected_log['LogNo'].values[0]
            
            # Append selected log
            grouped_roof_info.at[idx, 'Included logs'].append(log_id)
            remaining_log_load -= log_biomass
            log_load_assigned += log_biomass
            print(f"Selected log: {log_id}, Biomass: {log_biomass}kg")
            print(f"Remaining log load after assignment: {remaining_log_load}")
            
            # Remove the selected log from the available logs
            available_logs = available_logs[available_logs['LogNo'] != log_id]
        
        # Assign voxel indexes
        voxel_indexes = row['voxel_coordinates']
        print(f"Voxel coordinates available for assignment: {len(voxel_indexes)}")
        
        if grouped_roof_info.at[idx, 'Included logs']:
            # Randomly assign voxel indexes based on the number of assigned logs
            assigned_voxel_indexes = np.random.choice(voxel_indexes, len(grouped_roof_info.at[idx, 'Included logs']), replace=True)
            grouped_roof_info.at[idx, 'Assigned voxel indexes'] = list(assigned_voxel_indexes)
            print(f"Assigned voxel indexes: {assigned_voxel_indexes}")
        else:
            grouped_roof_info.at[idx, 'Assigned voxel indexes'] = []

        # Update envelope attributes for assigned logs
        for log_id, voxel_index in zip(row['Included logs'], row['Assigned voxel indexes']):
            xarray_dataset['envelope_assigned_log'].loc[dict(voxel=voxel_index)] = int(log_id)
            xarray_dataset['envelope_logID'].loc[dict(voxel=voxel_index)] = int(log_id)
            xarray_dataset['envelope_logSize'].loc[dict(voxel=voxel_index)] = log_size_map.get(log_id, np.nan)

        # Update the total assigned log load and remaining roof load
        grouped_roof_info.at[idx, 'Log load assigned'] = log_load_assigned
        grouped_roof_info.at[idx, 'Remaining roof load'] = total_roof_load - log_load_assigned
        print(f"Total log load assigned: {log_load_assigned}, Remaining roof load: {total_roof_load - log_load_assigned}")

    # Flatten arrays for logs and voxel indexes
    assigned_voxel_indexes_flat = grouped_roof_info['Assigned voxel indexes'].explode().dropna().astype(int).values
    included_logs_flat = grouped_roof_info['Included logs'].explode().dropna().astype(str).values

    print(f"\nFlattened log assignments: {included_logs_flat}")
    print(f"Flattened voxel assignments: {assigned_voxel_indexes_flat}")
    
    # Get the centroids at those voxel indexes
    centroid_x = xarray_dataset['centroid_x'].sel(voxel=assigned_voxel_indexes_flat).values
    centroid_y = xarray_dataset['centroid_y'].sel(voxel=assigned_voxel_indexes_flat).values
    centroid_z = xarray_dataset['centroid_z'].sel(voxel=assigned_voxel_indexes_flat).values
    
    print(f"\nRetrieved centroids (x, y, z):")
    print(f"Centroid X: {centroid_x}")
    print(f"Centroid Y: {centroid_y}")
    print(f"Centroid Z: {centroid_z}")
    
    # Create a numpy array of centroids
    centroids_flat = np.column_stack([centroid_x, centroid_y, centroid_z])
    
    # Get the list of log bounding boxes from log_library_df by looking up the Included logs
    log_bounding_boxes = []
    
    for log_id in included_logs_flat:
        log_row = log_library_df[log_library_df['LogNo'] == str(log_id)]  # Ensure matching datatype
        if not log_row.empty:
            cornerA = np.array(ast.literal_eval(log_row['CornerA'].values[0]))
            cornerB = np.array(ast.literal_eval(log_row['CornerB'].values[0]))
            log_bounding_boxes.append([cornerA, cornerB])
        else:
            # Let it break if the log isn't found
            raise ValueError(f"Log ID {log_id} not found in the log library!")
    
    log_bounding_boxes = np.array(log_bounding_boxes)
    
    # Translate the bounding boxes to align with the centroids
    translated_bounding_boxes = []
    
    for i, (cornerA, cornerB) in enumerate(log_bounding_boxes):
        current_center = (cornerA + cornerB) / 2
        target_centroid = centroids_flat[i]
        translation_vector = target_centroid - current_center
        translated_cornerA = cornerA + translation_vector
        translated_cornerB = cornerB + translation_vector
        translated_bounding_boxes.append([translated_cornerA, translated_cornerB])
        
        # Log the translation for debugging
        print(f"\nTranslating Bounding Box for Log {included_logs_flat[i]}:")
        print(f"Original center: {current_center}")
        print(f"Target centroid: {target_centroid}")
        print(f"Translation vector: {translation_vector}")
        print(f"Translated Corner A: {translated_cornerA}, Translated Corner B: {translated_cornerB}")
    
    translated_bounding_boxes = np.array(translated_bounding_boxes)

    # Create log_info_df
    log_info_df = pd.DataFrame({
        'logID': included_logs_flat,
        'voxelID': assigned_voxel_indexes_flat,
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'centroid_z': centroid_z,
        'boundAx': translated_bounding_boxes[:, 0, 0],
        'boundAy': translated_bounding_boxes[:, 0, 1],
        'boundAz': translated_bounding_boxes[:, 0, 2],
        'boundBx': translated_bounding_boxes[:, 1, 0],
        'boundBy': translated_bounding_boxes[:, 1, 1],
        'boundBz': translated_bounding_boxes[:, 1, 2]
    })

    # Print summary information for each roof
    print("\n--- Roof Summary ---")
    for idx, row in grouped_roof_info.iterrows():
        roof_id = row['roofID']
        building_id = row['site_building_ID']
        roof_type = row['site_roofType']
        num_voxels = row['number_of_voxels']
        num_logs = len(row['Included logs'])
        total_roof_load = row['Roof load']
        assigned_log_load = row['Log load assigned']
        remaining_roof_load = row['Remaining roof load']

        print(f"RoofID: {roof_id}, BuildingID: {building_id}, RoofType: {roof_type}, "
              f"Voxels: {num_voxels}, Logs: {num_logs}, "
              f"Total Roof Load: {total_roof_load}kg, "
              f"Assigned Log Load: {assigned_log_load}kg, "
              f"Remaining Load: {remaining_roof_load}kg")
    
    print("Assigning logs to xarray dataset...")
    assigned_logs_df = log_info_df[['voxelID', 'logID']].copy()
    assigned_logs_agg = assigned_logs_df.groupby('voxelID')['logID'].apply(list).reset_index(name='assigned_logs')
    assigned_logs_dict = pd.Series(assigned_logs_agg.assigned_logs.values, index=assigned_logs_agg.voxelID).to_dict()
    assigned_logs_list = [assigned_logs_dict.get(v, []) for v in xarray_dataset['voxel'].values]
    assigned_logs_str = [','.join(map(str, logs)) for logs in assigned_logs_list]
    max_logs_length = max(len(s) for s in assigned_logs_str) if assigned_logs_str else 1
    xarray_dataset = xarray_dataset.assign(assigned_logs=('voxel', np.array(assigned_logs_str, dtype=f'U{max_logs_length}')))
    
    print("\nLog assignment completed.")
    return log_info_df, xarray_dataset


def save_results(site, grouped_roof_info, log_info_df, xarray_dataset):
    """
    Saves the processed data as CSV files and pickles the xarray dataset.
    
    Parameters:
        site (str): Site identifier.
        grouped_roof_info (pd.DataFrame): DataFrame containing grouped roof information.
        log_info_df (pd.DataFrame): DataFrame containing log assignments.
        xarray_dataset (xarray.Dataset): Updated xarray dataset with 'assigned_logs'.
    """
    print("Saving results...")
    # Define output folder
    output_folder = f"data/revised/envelopes"
    os.makedirs(output_folder, exist_ok=True)
    
    # Save grouped_roof_info as {site}_roofInfo.csv
    roof_info_path = f"{output_folder}/{site}_roofInfo.csv"
    grouped_roof_info.to_csv(roof_info_path, index=False)
    print(f"Saved roof info to {roof_info_path}")
    
    # Save log_info_df as {site}_logInfo.csv
    log_info_path = f"{output_folder}/{site}_logInfo.csv"
    log_info_df.to_csv(log_info_path, index=False)
    print(f"Saved log info to {log_info_path}")
    
    # Save xarray dataset using Pickle
    pickle_file_path = f"{output_folder}/{site}xarray_voxels.pkl"
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(xarray_dataset, f)
    print(f"Pickle saved at {pickle_file_path}")

def plot_roofs(xarray_dataset, grouped_roof_info_df, roof_info_df, log_info_df, site):
    """
    Plots the translated bounding boxes along with voxel centroids and site bounds.
    
    Parameters:
        xarray_dataset (xarray.Dataset): The xarray dataset containing spatial data.
        roof_info_df (pd.DataFrame): DataFrame containing roof information.
        translated_bounding_boxes (np.ndarray): Array of translated bounding boxes.
        centroids_flat (np.ndarray): Array of centroid coordinates for assigned logs.
        site (str): Site identifier.
    """
    print("Plotting roofs...")
    # Extract the site's overall bounds from the xarray dataset attributes
    
    # Reconstruct translated_bounding_boxes and centroids_flat for plotting
    translated_bounding_boxes = log_info_df[['boundAx', 'boundAy', 'boundAz', 'boundBx', 'boundBy', 'boundBz']].values.reshape(-1, 2, 3)
    centroids_flat = log_info_df[['centroid_x', 'centroid_y', 'centroid_z']].values
    
    
    bounds = xarray_dataset.attrs.get('bounds', None)

    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract all voxel centroids from the original roof_info_df (x and y only)
    all_centroids_x = roof_info_df['centroid_x'].values
    all_centroids_y = roof_info_df['centroid_y'].values

    print(f'all centroids y is {all_centroids_y}')
    
    # Plot all voxel centroids in grey
    ax.scatter(all_centroids_x, all_centroids_y, color='grey', label='All Voxel Centroids', s=10, alpha=0.3)
    
    # Define colormap and number of boxes
    cmap = plt.cm.get_cmap('viridis')
    num_boxes = len(translated_bounding_boxes)
    
    # Plot the bounding boxes with 50% opacity
    for i, (cornerA, cornerB) in enumerate(translated_bounding_boxes):
        width = cornerB[0] - cornerA[0]
        height = cornerB[1] - cornerA[1]
        rect = Rectangle((cornerA[0], cornerA[1]), width, height,
                         edgecolor='black', facecolor=cmap(i / num_boxes), alpha=0.5)
        ax.add_patch(rect)
    
    # Overlay the voxel centroids from assigned logs in red
    ax.scatter(centroids_flat[:, 0], centroids_flat[:, 1], color='red',
               label='Assigned Voxel Centroids', s=20)
    
    # Overlay the site's overall bounds
    site_boundary = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(site_boundary)
    
    # Set plot limits and labels
    ax.set_xlim(x_min - 10, x_max + 10)
    ax.set_ylim(y_min - 10, y_max + 10)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Translated Bounding Boxes with All Voxel Centroids and Site Overall Bounds ({site})')
    
    # Add legend
    ax.legend()
    
    # Save the plot
    plot_path = f"data/revised/facades/{site}_roofs_plot.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved at {plot_path}")



def process_site(site, xarray_dataset):
    print(f"Processing site: {site}")

    # Define required variables and attributes
    required_variables = ['centroid_x', 'centroid_y', 'centroid_z', 'site_greenRoof_ratingInt', 'site_brownRoof_ratingInt', 'site_building_ID']
    required_attributes = ['bounds', 'voxel_size']

    # Create subset
    subset_xarray = a_helper_functions.create_subset_dataset(xarray_dataset, required_variables, required_attributes)
    #save subset_xarray to netcdf

    # Load the log library CSV
    log_library_path = 'data/revised/trees/logLibraryStats.csv'
    log_library_df = pd.read_csv(log_library_path)
    
    # Process the subset
    subset_xarray, grouped_roof_info, roof_info_df = calculate_roof_voxels(subset_xarray)
    log_info_df, subset_xarray = assign_logs(grouped_roof_info, log_library_df, subset_xarray)
   
    # Update the full dataset with the processed results
    """for var in subset_xarray.variables:
        if var in xarray_dataset:
            xarray_dataset[var].values = subset_xarray[var].values
        else:
            xarray_dataset[var] = subset_xarray[var]"""
    
    for var in subset_xarray.data_vars:
        if var in xarray_dataset:
            xarray_dataset[var] = subset_xarray[var]
        else:
            xarray_dataset = xarray_dataset.assign({var: subset_xarray[var]})

    
    xarray_dataset.attrs.update(subset_xarray.attrs)

    # Save results and plot
    save_results(site, grouped_roof_info, log_info_df, xarray_dataset)
    plot_roofs(xarray_dataset, grouped_roof_info, roof_info_df, log_info_df, site)

    print(f"Processing for site '{site}' completed successfully.")
    return xarray_dataset, grouped_roof_info, log_info_df


def process_site2(site, xarray_dataset):
    """
    Manager function to process the data for a given site.
    
    Parameters:
        site (str): Site identifier (e.g., 'uni').
        xarray_dataset (xr.Dataset): Input xarray dataset.
    
    Returns:
        xarray.Dataset: Updated xarray dataset with envelope attributes.
        pd.DataFrame: DataFrame containing grouped roof information.
        pd.DataFrame: DataFrame with log assignments.
    """
    print(f"Processing site: {site}")

    # Load the log library CSV
    log_library_path = 'data/revised/trees/logLibraryStats.csv'
    log_library_df = pd.read_csv(log_library_path)
    
    # Step 1 & 2: Calculate roof voxels and roof loads
    xarray_dataset, grouped_roof_info, roof_info_df = calculate_roof_voxels(xarray_dataset)
    
    # Step 3: Assign logs based on roof loads
    log_info_df, xarray_dataset = assign_logs(grouped_roof_info, log_library_df, xarray_dataset)
   
    # Step 4: Save results
    save_results(site, grouped_roof_info, log_info_df, xarray_dataset)
    
    # Step 5: Plot roofs
    plot_roofs(xarray_dataset, grouped_roof_info, roof_info_df, log_info_df, site)

    print(f"Processing for site '{site}' completed successfully.")
    return xarray_dataset, grouped_roof_info, log_info_df

if __name__ == "__main__":
    # Example usage
    site = 'uni'  # Define the site identifier
    
    # Define file paths based on the site
    print("Loading xarray dataset...")
    xarray_input_path = f"data/revised/xarray_voxels_{site}_2.5.nc"
    xarray_dataset = xr.open_dataset(xarray_input_path)
    
    process_site(site, xarray_dataset)