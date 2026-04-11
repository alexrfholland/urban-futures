import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import rasterio
from rasterio.features import geometry_mask

# Convert shapefile to a raster mask
def shapefile_to_raster(shapefile, pixel_size):
    xmin, ymin, xmax, ymax = shapefile.total_bounds
    shape_height = int((ymax - ymin) / pixel_size)
    shape_width = int((xmax - xmin) / pixel_size)
    
    transform = rasterio.transform.from_origin(xmin, ymax, pixel_size, pixel_size)
    mask = geometry_mask(shapefile.geometry, transform=transform, invert=True, out_shape=(shape_height, shape_width))
    
    return mask, transform

# Read site coordinate from CSV
site_coord = pd.read_csv('data/site projections.csv')
print("Loaded site coordinate CSV")

easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

# Read VTM file
site = 'city'
multiblock = pv.read(f'data/{site}/flattened-{site}.vtm')
combined_polydata = multiblock[0]
print(f"Loaded polydata for {site}")

# Read and translate the shapefile
shapefile = gpd.read_file('data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp')
shapefile.geometry = shapefile.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
shapefile['shape_id'] = shapefile.index
print(f"Loaded and translated shapefile")

# Convert shapefile to raster mask
pixel_size = 1  # Set pixel size based on your requirements
mask, transform = shapefile_to_raster(shapefile, pixel_size)
print(f"Converted shapefile to raster mask")

# Initialize an ID array for point cloud
point_ids = np.full(combined_polydata.n_points, -1, dtype=int)

# Convert point cloud coordinates to raster indices
xmin, ymin, _, _ = shapefile.total_bounds
raster_x_indices = np.floor((combined_polydata.points[:, 0] - xmin) / pixel_size).astype(int)
raster_y_indices = np.floor((combined_polydata.points[:, 1] - ymin) / pixel_size).astype(int)

# Populate point_ids using raster indices
point_ids = mask[raster_y_indices, raster_x_indices]

# Assign the ID array to combined PolyData
combined_polydata['shape_id'] = point_ids

# Save the modified point cloud
combined_polydata.save('modified_point_cloud.vtk')

print("Point cloud modification complete.")


