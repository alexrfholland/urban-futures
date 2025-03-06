import f_GeospatialModule
import pandas as pd
import pickle
import os

#resourceMapper
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




#GO THROUGH LARGE TREES
#FOR EACH DF, GROUP INTO 1M VOXELS

#444.65 kg/m3 to 578.19 kg/m3





#CREATE A NEW DF that has a row per log
##GO THROUGH THE LOG LIBRARY, getting the LogNo, logSize, and number of voxels (ie. length of each log df)
#GET BOUNDING BOX FOR EACH LOG (ie. cornerA, cornerB, etc). save x y z positions of each corner
#NUMBER OF VOXELS (ie. npoints)
#GET BIOVOLUME (voxels are 0.25m3 cubes)
#GET BIOMASS (assume density of 500kg/m3)
#SAVE DATAFRAME

# Import necessary libraries
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ast  # to safely evaluate string representations of tuples

# Load the log library CSV and xarray dataset from the provided paths
log_library_df = pd.read_csv('data/revised/trees/logLibraryStats.csv')
xarray_dataset = xr.open_dataset('data/revised/xarray_voxels_uni_2.5.nc')


# Step 2: Initialize 'site_roofType' in the xarray dataset
xarray_dataset = xarray_dataset.assign(site_roofType=(['voxel'], ['none'] * xarray_dataset.sizes['voxel']))

# Step 3: Filter voxels based on roof types
green_roof_mask = xarray_dataset.site_greenRoof_ratingInt > 3
brown_roof_mask = (xarray_dataset.site_brownRoof_ratingInt > 3) & (xarray_dataset.site_greenRoof_ratingInt < 3)

# Count the number of voxels for each roof type
green_roof_count = green_roof_mask.sum().item()
brown_roof_count = brown_roof_mask.sum().item()

# Print the counts
print(f"Number of green roof voxels: {green_roof_count}")
print(f"Number of brown roof voxels: {brown_roof_count}")

xarray_dataset['site_roofType'] = xr.where(green_roof_mask, 'green roof', xarray_dataset.site_roofType)
xarray_dataset['site_roofType'] = xr.where(brown_roof_mask, 'brown roof', xarray_dataset.site_roofType)

# Print unique values and counts in site_roofType
unique_roof_types = xarray_dataset.site_roofType.values
unique_roof_counts = pd.Series(unique_roof_types).value_counts()

print("Unique values and counts in site_roofType:")
print(unique_roof_counts)
print("\n")

# Step 4: Flatten the xarray into a DataFrame
flattened_df = xarray_dataset.to_dataframe().reset_index()
flattened_df['Voxel coordinate'] = flattened_df['voxel']

print(flattened_df)

# Step 5: Filter out 'none' roof types
roof_info_df = flattened_df[flattened_df['site_roofType'] != 'none'].copy()



# Step 6: Create roofID and group by building and roof type
roof_info_df['roofID'] = roof_info_df.groupby(['site_building_ID', 'site_roofType']).ngroup()

grouped_roof_info = roof_info_df.groupby('roofID').agg(
    site_building_ID=('site_building_ID', 'first'),
    site_roofType=('site_roofType', 'first'),
    Voxel_coordinates=('Voxel coordinate', list),
    Centroid_x=('centroid_x', list),
    Centroid_y=('centroid_y', list),
    Centroid_z=('centroid_z', list),
    Number_of_voxels=('site_building_ID', 'count')
).reset_index()

# Step 6.5: Print informative statistics about unique roof types and groups
print("Informative Statistics:")
unique_roof_groups = grouped_roof_info['site_roofType'].value_counts()
print(f"Unique green roof groups: {unique_roof_groups['green roof']}")
print(f"Unique brown roof groups: {unique_roof_groups['brown roof']}")

# Step 7: Define roof dead load constants and calculate loads
green_roof_dead_load = 300  # kg/m^2
brown_roof_dead_load = 150  # kg/m^2
voxel_area = 2.5 * 2.5  # area of a voxel in square meters

# Calculate the roof load for each group
grouped_roof_info['Roof load'] = grouped_roof_info['Number_of_voxels'] * voxel_area * grouped_roof_info['site_roofType'].map({
    'green roof': green_roof_dead_load,
    'brown roof': brown_roof_dead_load
})

# Step 8: Define log loads and calculate them
green_roof_log_load = 50  # kg for green roof logs
brown_roof_log_load = 100  # kg for brown roof logs

grouped_roof_info['Log load'] = grouped_roof_info['Number_of_voxels'] * voxel_area * grouped_roof_info['site_roofType'].map({
    'green roof': green_roof_log_load,
    'brown roof': brown_roof_log_load
})

# Step 9: Initialize columns for logs and assigned translations
grouped_roof_info['Included logs'] = [[] for _ in range(len(grouped_roof_info))]
grouped_roof_info['Assigned voxel indexes'] = [[] for _ in range(len(grouped_roof_info))]

# Step 9.5: Assign logs based on voxel load and assign voxel indexes
for idx, row in grouped_roof_info.iterrows():
    remaining_log_load = row['Log load']
    available_logs = log_library_df.copy()

    # Randomly assign logs until the log load is filled
    while remaining_log_load > 0 and not available_logs.empty:
        selected_log = available_logs.sample(1)
        log_biomass = selected_log['Biomass_kg'].values[0]
        grouped_roof_info.at[idx, 'Included logs'].append(selected_log['LogNo'].values[0])
        remaining_log_load -= log_biomass
        available_logs = available_logs[available_logs['LogNo'] != selected_log['LogNo'].values[0]]

    # Randomly assign voxel indexes for the included logs
    voxel_indexes = row['Voxel_coordinates']
    assigned_voxel_indexes = np.random.choice(voxel_indexes, len(row['Included logs']), replace=True)
    grouped_roof_info.at[idx, 'Assigned voxel indexes'] = list(assigned_voxel_indexes)

# Step 10: Flatten numpy arrays for Assigned voxel indexes and Included logs
assigned_voxel_indexes_flat = np.concatenate(grouped_roof_info['Assigned voxel indexes'].values)
included_logs_flat = np.concatenate(grouped_roof_info['Included logs'].values)

# Step 10.5: Get the centroids at those voxel indexes
centroid_x = xarray_dataset['centroid_x'].sel(voxel=assigned_voxel_indexes_flat).values
centroid_y = xarray_dataset['centroid_y'].sel(voxel=assigned_voxel_indexes_flat).values
centroid_z = xarray_dataset['centroid_z'].sel(voxel=assigned_voxel_indexes_flat).values

# Create a numpy array of centroids
centroids_flat = np.column_stack([centroid_x, centroid_y, centroid_z])

# Step 11: Get the list of log bounding boxes from log_library_df by looking up the Included logs
log_bounding_boxes = []

for log_id in included_logs_flat:
    log_row = log_library_df[log_library_df['LogNo'] == log_id]
    cornerA = np.array(ast.literal_eval(log_row['CornerA'].values[0]))
    cornerB = np.array(ast.literal_eval(log_row['CornerB'].values[0]))
    log_bounding_boxes.append([cornerA, cornerB])

log_bounding_boxes = np.array(log_bounding_boxes)

# Step 11.5: Translate the bounding boxes such that their centers align with the centroids
translated_bounding_boxes = []

for i, (cornerA, cornerB) in enumerate(log_bounding_boxes):
    current_center = (cornerA + cornerB) / 2
    target_centroid = centroids_flat[i]
    translation_vector = target_centroid - current_center
    translated_cornerA = cornerA + translation_vector
    translated_cornerB = cornerB + translation_vector
    translated_bounding_boxes.append([translated_cornerA, translated_cornerB])

translated_bounding_boxes = np.array(translated_bounding_boxes)

# Step 12: Plot the translated bounding boxes along with the voxel centroids and the site's overall bounds
# Extract the site's overall bounds from the xarray dataset attributes
x_min, x_max, y_min, y_max, z_min, z_max = xarray_dataset.attrs['bounds']

# Step 12.7: Plot the voxel centroids in grey (back-most layer) and set 50% opacity for the bounding boxes
fig, ax = plt.subplots(figsize=(10, 10))

# Extract all voxel centroids from the original roof_info_df (x and y only)
all_centroids_x = np.concatenate(roof_info_df['centroid_x'].values)
all_centroids_y = np.concatenate(roof_info_df['centroid_y'].values)

# Plot all voxel centroids in grey (back-most layer)
ax.scatter(all_centroids_x, all_centroids_y, color='grey', label='All Voxel Centroids', s=10, alpha=0.3)

# Plot the bounding boxes (x and y only) with 50% opacity
for i, (cornerA, cornerB) in enumerate(translated_bounding_boxes):
    width = cornerB[0] - cornerA[0]
    height = cornerB[1] - cornerA[1]
    rect = Rectangle((cornerA[0], cornerA[1]), width, height, edgecolor='black', facecolor=cmap(i / num_boxes), alpha=0.5)
    ax.add_patch(rect)

# Overlay the voxel centroids from assigned logs (highlighted in red)
ax.scatter(centroids_flat[:, 0], centroids_flat[:, 1], color='red', label='Assigned Voxel Centroids', s=20)

# Overlay the site's overall bounds
site_boundary = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='black', facecolor='none', linewidth=2)
ax.add_patch(site_boundary)

# Set plot limits and labels
ax.set_xlim(x_min - 10, x_max + 10)
ax.set_ylim(y_min - 10, y_max + 10)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Translated Bounding Boxes with All Voxel Centroids and Site Overall Bounds')

# Add legend
ax.legend()

plt.show()
