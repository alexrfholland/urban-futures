""""""import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import time
from shapely.geometry import box

# Function to convert a GeoDataFrame to PyVista-compatible PolyData
def geodataframe_to_pyvista(gdf):
    print("Converting GeoDataFrame to PyVista PolyData...")
    polydata_list = []
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            exterior = np.array(geom.exterior.coords)
            exterior_3d = np.c_[exterior, np.zeros(len(exterior))]
            polydata = pv.PolyData(exterior_3d)
            polydata_list.append(polydata)
        elif geom.geom_type == 'MultiPolygon':
            for polygon in geom:
                exterior = np.array(polygon.exterior.coords)
                exterior_3d = np.c_[exterior, np.zeros(len(exterior))]
                polydata = pv.PolyData(exterior_3d)
                polydata_list.append(polydata)
    return polydata_list

print("Reading site coordinate from CSV...")
site_coord = pd.read_csv('data/site projections.csv')
easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

print("Creating a 250m bounding box centered around the site coordinate...")
bbox = box(easting_offset - 250, northing_offset - 250, easting_offset + 250, northing_offset + 250)

print("Reading and cropping the shapefile...")
shapefile = gpd.read_file('data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp')
shapefile_cropped = shapefile[shapefile.geometry.intersects(bbox)]

print("Translating the cropped shapefile...")
shapefile_cropped.geometry = shapefile_cropped.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
shapefile_cropped['shape_id'] = shapefile_cropped.index

print("Reading the VTM file...")
site = 'city'
multi_block = pv.read(f'data/{site}/flattened-{site}.vtm')

# Convert cropped GeoDataFrame to PyVista PolyData
shapefile_polydata_list = geodataframe_to_pyvista(shapefile_cropped)

print("Initializing Plotter...")
plotter = pv.Plotter()

print("Adding shapefile geometries to the plotter...")
for polydata in shapefile_polydata_list:
    plotter.add_mesh(polydata)

print(multi_block[0].point_data)

print("Adding the point cloud to the plotter...")
plotter.add_mesh(multi_block[0], color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)


print("Setting the plotter to view from the top...")
plotter.view_xy()

print("Showing plotter...")
plotter.show()

print("Visualization complete.")
"""


import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import time
from shapely.geometry import box

# Function to handle batch processing of points
def process_batch(points, shapefile, spatial_index):
    print("Processing batch...")
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

# Read site coordinate from CSV
print("Reading site coordinate from CSV...")
site_coord = pd.read_csv('data/site projections.csv')
easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

# Create a 250m bounding box centered around the site coordinate
print("Creating a 250m bounding box...")
bbox = box(easting_offset - 250, northing_offset - 250, easting_offset + 250, northing_offset + 250)

# Read and crop the shapefile
print("Reading and cropping the shapefile...")
shapefile = gpd.read_file('data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp')
shapefile_cropped = shapefile[shapefile.geometry.intersects(bbox)]

# Translate the cropped shapefile and assign shape_id
print("Translating the shapefile...")
shapefile_cropped.geometry = shapefile_cropped.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
shapefile_cropped['shape_id'] = shapefile_cropped.index

# Read the VTM file
print("Reading the VTM file...")
site = 'city'
multiblock = pv.read(f'data/{site}/flattened-{site}.vtm')
combined_polydata = multiblock[0]

# Initialize an ID array for point cloud
print("Initializing ID array...")
point_ids = np.full(combined_polydata.n_points, -1, dtype=int)

# Create an R-tree spatial index for the shapefile
print("Creating spatial index...")
spatial_index = shapefile_cropped.sindex

# Perform point-in-polygon queries in batches
print("Running point-in-polygon queries...")
start_time = time.time()

batch_size = 1000
n_batches = int(np.ceil(combined_polydata.n_points / batch_size))

for i in range(n_batches):
    print(f"Processing batch {i+1} of {n_batches}")
    
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, combined_polydata.n_points)
    batch_points = combined_polydata.points[start_idx:end_idx]
    
    batch_ids = process_batch(batch_points, shapefile_cropped, spatial_index)
    point_ids[start_idx:end_idx] = batch_ids

print(f'Queries completed in {time.time() - start_time:.2f} seconds')

# Assign the ID array to the combined PolyData
combined_polydata['shape_id'] = point_ids

# Save the modified point cloud
combined_polydata.save('modified_point_cloud.vtk')

print("Point cloud modification complete.")
"""