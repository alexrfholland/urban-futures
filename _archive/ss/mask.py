import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np

# Purpose: This script enriches a given point cloud dataset with additional attributes from a shapefile.
# Specifically, it assigns each point in the cloud an ID corresponding to a shape it falls within in the shapefile.
# The point cloud and shapefile are assumed to belong to the same site but may have different coordinate systems.
# To align them, the script translates the coordinates of the shapefile to match those of the point cloud.

# Steps:
# 1. Read the site coordinates from a CSV file and obtain the easting and northing offsets for a particular site ('city' in this case).
# 2. Load a flattened multi-block VTK file (VTM) that contains the point cloud as well as atttibutes for the site (ie. the point cloud polydata is multiblock[0])
# 3. Read a shapefile and translate its coordinates based on the easting and northing offsets to align it with the point cloud.
# 4. Create a unique identifier ('shape_id') for each shape in the shapefile.
# 5. Initialize an array ('point_ids') to store the shape IDs corresponding to each point in the cloud.
# 6. Use an R-tree spatial index to speed up the point-in-polygon queries.
# 7. Loop through the point cloud and populate 'point_ids' based on which shape each point falls within.
# 8. Attach the 'point_ids' array to the point cloud as a new attribute.
# 9. Using the point_ids, transfer all attributes from the corroposnding geometries in the shapefile to point_data attributes in the polydata 
# 10. Save the enriched point cloud as a new VTK file.

"""import time

# Read the site coordinate from CSV
site_coord = pd.read_csv('data/site projections.csv')
print("Site coordinates loaded:", site_coord)

easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

# Read the VTM file
print("Reading VTM file...")
start_time = time.time()
site = 'city'
multiblock = pv.read(f'data/{site}/flattened-{site}.vtm')
combined_polydata = multiblock[0]
print(f'Polydata loaded in {time.time() - start_time:.2f} seconds')

# Read and translate the shapefile
print("Reading and translating shapefile...")
start_time = time.time()
shapefile = gpd.read_file('data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp')
shapefile.geometry = shapefile.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
shapefile['shape_id'] = shapefile.index
print(f'Shapefile loaded and translated in {time.time() - start_time:.2f} seconds')

# Initialize an ID array for the point cloud
point_ids = np.full(combined_polydata.n_points, -1, dtype=int)

# Create an R-tree spatial index for the shapefile
print("Creating R-tree spatial index...")
start_time = time.time()
spatial_index = shapefile.sindex
print(f'Spatial index created in {time.time() - start_time:.2f} seconds')

# Populate the ID array for point-in-polygon queries
print("Running point-in-polygon queries...")
start_time = time.time()
for i, point in enumerate(combined_polydata.points):
    if i % 1000 == 0:
        print(f"Processing point {i}")
    possible_matches_index = list(spatial_index.intersection((point[0], point[1], point[0], point[1])))
    possible_matches = shapefile.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(gpd.points_from_xy([point[0]], [point[1]])[0])]
    
    if not precise_matches.empty:
        point_ids[i] = precise_matches.iloc[0]['shape_id']
print(f'Queries completed in {time.time() - start_time:.2f} seconds')

# Assign the ID array to the combined PolyData
combined_polydata['shape_id'] = point_ids

# Save the modified point cloud
print("Saving modified point cloud...")
combined_polydata.save('modified_point_cloud.vtk')

print("Point cloud modification complete.")

"""
"""import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to process a single point
def process_point(i, point, spatial_index, shapefile):
    if i % 1000 == 0:
        print(f"Processing point {i}")
    possible_matches_index = list(spatial_index.intersection((point[0], point[1], point[0], point[1])))
    possible_matches = shapefile.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(gpd.points_from_xy([point[0]], [point[1]])[0])]
    if not precise_matches.empty:
        return i, precise_matches.iloc[0]['shape_id']
    else:
        return i, -1

# Read the site coordinate from CSV
site_coord = pd.read_csv('data/site projections.csv')
print("Loaded site coordinate CSV")

easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

# Read the VTM file
site = 'city'
multiblock = pv.read(f'data/{site}/flattened-{site}.vtm')
combined_polydata = multiblock[0]

print(f'Loaded polydata for {site}')

# Read and translate the shapefile
shapefile = gpd.read_file('data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp')
shapefile.geometry = shapefile.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
shapefile['shape_id'] = shapefile.index

print(f'Loaded and translated shapefile')

# Initialize an ID array for the point cloud
point_ids = np.full(combined_polydata.n_points, -1, dtype=int)

# Create an R-tree spatial index for the shapefile
spatial_index = shapefile.sindex

# Populate the ID array for point-in-polygon queries
print("Running point-in-polygon queries...")
start_time = time.time()

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_point, i, point, spatial_index, shapefile): point for i, point in enumerate(combined_polydata.points)}
    for future in as_completed(futures):
        i, shape_id = future.result()
        point_ids[i] = shape_id

print(f'Queries completed in {time.time() - start_time:.2f} seconds')

# Assign the ID array to the combined PolyData
combined_polydata['shape_id'] = point_ids

# Save the modified point cloud
combined_polydata.save('modified_point_cloud.vtk')

print("Point cloud modification complete.")


""""""

import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import time

# Read the site coordinate from CSV
site_coord = pd.read_csv('data/site projections.csv')
print("Loaded site coordinate CSV")

easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

# Read the VTM file
site = 'city'
multiblock = pv.read(f'data/{site}/flattened-{site}.vtm')
combined_polydata = multiblock[0]

print(f'Loaded polydata for {site}')

# Read and translate the shapefile
shapefile = gpd.read_file('data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp')
shapefile.geometry = shapefile.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
shapefile['shape_id'] = shapefile.index

print(f'Loaded and translated shapefile')

# Initialize an ID array for the point cloud
point_ids = np.full(combined_polydata.n_points, -1, dtype=int)

# Create an R-tree spatial index for the shapefile
spatial_index = shapefile.sindex

# Populate the ID array for point-in-polygon queries
print("Running point-in-polygon queries...")
start_time = time.time()

for i, point in enumerate(combined_polydata.points):
    if i % 1000 == 0:
        print(f"Processing point {i}")
    possible_matches_index = list(spatial_index.intersection((point[0], point[1], point[0], point[1])))
    possible_matches = shapefile.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(gpd.points_from_xy([point[0]], [point[1]])[0])]
    
    if not precise_matches.empty:
        point_ids[i] = precise_matches.iloc[0]['shape_id']

print(f'Queries completed in {time.time() - start_time:.2f} seconds')

# Assign the ID array to the combined PolyData
combined_polydata['shape_id'] = point_ids

# Save the modified point cloud
combined_polydata.save('modified_point_cloud.vtk')

print("Point cloud modification complete.")"""

"""import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import time

# Read the site coordinate from CSV
site_coord = pd.read_csv('data/site projections.csv')
print("Loaded site coordinate CSV")

easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

# Read the VTM file
site = 'city'
multiblock = pv.read(f'data/{site}/flattened-{site}.vtm')
combined_polydata = multiblock[0]

print(f'Loaded polydata for {site}')

# Read and translate the shapefile
shapefile = gpd.read_file('data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp')
shapefile.geometry = shapefile.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
shapefile['shape_id'] = shapefile.index

print(f'Loaded and translated shapefile')

# Initialize an ID array for the point cloud
point_ids = np.full(combined_polydata.n_points, -1, dtype=int)

# Create an R-tree spatial index for the shapefile
spatial_index = shapefile.sindex

# Define batch size and prepare for batching
batch_size = 1000
n_batches = int(np.ceil(combined_polydata.n_points / batch_size))

# Populate the ID array for point-in-polygon queries
print("Running point-in-polygon queries...")
start_time = time.time()

for b in range(n_batches):
    start_idx = b * batch_size
    end_idx = (b + 1) * batch_size
    print(f"Processing batch {b + 1} of {n_batches}")

    for i in range(start_idx, min(end_idx, combined_polydata.n_points)):
        point = combined_polydata.points[i]
        possible_matches_index = list(spatial_index.intersection((point[0], point[1], point[0], point[1])))
        possible_matches = shapefile.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(gpd.points_from_xy([point[0]], [point[1]])[0])]

        if not precise_matches.empty:
            point_ids[i] = precise_matches.iloc[0]['shape_id']

print(f'Queries completed in {time.time() - start_time:.2f} seconds')

# Assign the ID array to the combined PolyData
combined_polydata['shape_id'] = point_ids

# Save the modified point cloud
combined_polydata.save('modified_point_cloud.vtk')

print("Point cloud modification complete.")
"""

#batched

"""

import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import time

# Function to handle batch processing of points
def process_batch(points, shapefile, spatial_index):
    min_x = min(points[:, 0])
    min_y = min(points[:, 1])
    max_x = max(points[:, 0])
    max_y = max(points[:, 1])
    
    possible_matches_index = list(spatial_index.intersection((min_x, min_y, max_x, max_y)))
    possible_matches = shapefile.iloc[possible_matches_index]
    
    batch_ids = np.full(points.shape[0], -1, dtype=int)
    
    for i, point in enumerate(points):
        precise_matches = possible_matches[possible_matches.intersects(gpd.points_from_xy([point[0]], [point[1]])[0])]
        
        if not precise_matches.empty:
            batch_ids[i] = precise_matches.iloc[0]['shape_id']
    
    return batch_ids

# Read the site coordinate from CSV
site_coord = pd.read_csv('data/site projections.csv')
print("Loaded site coordinate CSV")

easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

# Read the VTM file
site = 'city'
multiblock = pv.read(f'data/{site}/flattened-{site}.vtm')
combined_polydata = multiblock[0]

print(f'Loaded polydata for {site}')

# Read and translate the shapefile
shapefile = gpd.read_file('data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp')
shapefile.geometry = shapefile.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
shapefile['shape_id'] = shapefile.index

print(f'Loaded and translated shapefile')

# Initialize an ID array for the point cloud
point_ids = np.full(combined_polydata.n_points, -1, dtype=int)

# Create an R-tree spatial index for the shapefile
spatial_index = shapefile.sindex

# Perform point-in-polygon queries in batches
print("Running point-in-polygon queries...")
start_time = time.time()

batch_size = 1000
n_batches = int(np.ceil(combined_polydata.n_points / batch_size))

for i in range(n_batches):
    if i % 1 == 0:
        print(f"Processing batch {i+1} of {n_batches}")
    
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, combined_polydata.n_points)
    batch_points = combined_polydata.points[start_idx:end_idx]
    
    batch_ids = process_batch(batch_points, shapefile, spatial_index)
    point_ids[start_idx:end_idx] = batch_ids

print(f'Queries completed in {time.time() - start_time:.2f} seconds')

# Assign the ID array to the combined PolyData
combined_polydata['shape_id'] = point_ids

# Save the modified point cloud
combined_polydata.save('modified_point_cloud.vtk')

print("Point cloud modification complete.")
"""

#batched and multithreaded

import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

# Function to process a batch of points
def process_batch(batch_points, batch_idx):
    local_point_ids = np.full(len(batch_points), -1, dtype=int)
    print(f"Processing batch {batch_idx}")
    
    for i, point in enumerate(batch_points):
        possible_matches_index = list(spatial_index.intersection((point[0], point[1], point[0], point[1])))
        possible_matches = shapefile.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(gpd.points_from_xy([point[0]], [point[1]])[0])]
        
        if not precise_matches.empty:
            local_point_ids[i] = precise_matches.iloc[0]['shape_id']
    
    print(f"Completed batch {batch_idx}")
    return local_point_ids

# Read the site coordinate from CSV
site_coord = pd.read_csv('data/site projections.csv')
print("Loaded site coordinate CSV")

easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

# Read the VTM file
site = 'city'
multiblock = pv.read(f'data/{site}/flattened-{site}.vtm')
combined_polydata = multiblock[0]

print(f"Loaded polydata for {site}")

# Read and translate the shapefile
shapefile = gpd.read_file('data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp')
shapefile.geometry = shapefile.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
shapefile['shape_id'] = shapefile.index

print(f"Loaded and translated shapefile")

# Initialize an ID array for the point cloud
point_ids = np.full(combined_polydata.n_points, -1, dtype=int)

# Create an R-tree spatial index for the shapefile
spatial_index = shapefile.sindex

# Divide points into batches
batch_size = 1000
total_points = combined_polydata.n_points
num_batches = int(np.ceil(total_points / batch_size))

# Create a ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    # Create a list of futures
    futures = [executor.submit(process_batch, combined_polydata.points[i * batch_size : (i + 1) * batch_size], i) for i in range(num_batches)]
    
    # Populate the ID array for point-in-polygon queries
    print("Running point-in-polygon queries in parallel...")
    start_time = time.time()
    
    for i, future in enumerate(futures):
        point_ids[i * batch_size : (i + 1) * batch_size] = future.result()

print(f"Queries completed in {time.time() - start_time:.2f} seconds")

# Assign the ID array to the combined PolyData
combined_polydata['shape_id'] = point_ids

# Save the modified point cloud
combined_polydata.save('modified_point_cloud.vtk')

print("Point cloud modification complete.")
