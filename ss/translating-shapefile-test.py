import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box

# Read the VTM file
site = 'city'
multi_block = pv.read(f'data/{site}/flattened-{site}.vtm')

# Function to convert a GeoDataFrame to PyVista-compatible PolyData
def geodataframe_to_pyvista(gdf):
    polydata_list = []

    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            exterior = np.array(geom.exterior.coords)
            exterior_3d = np.c_[exterior, np.zeros(len(exterior))]  # Append z-coordinates
            polydata = pv.PolyData(exterior_3d)
            polydata_list.append(polydata)
        elif geom.geom_type == 'MultiPolygon':
            for polygon in geom:
                exterior = np.array(polygon.exterior.coords)
                exterior_3d = np.c_[exterior, np.zeros(len(exterior))]  # Append z-coordinates
                polydata = pv.PolyData(exterior_3d)
                polydata_list.append(polydata)
    return polydata_list

# Read site coordinate from CSV
site_coord = pd.read_csv('data/site projections.csv')
easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

# Create a 250m bounding box centered around the site coordinate
bbox = box(easting_offset - 250, northing_offset - 250, easting_offset + 250, northing_offset + 250)

# Read and crop the shapefile
shapefile = gpd.read_file('data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp')
shapefile_cropped = shapefile[shapefile.geometry.intersects(bbox)]

# Translate the cropped shapefile
shapefile_cropped.geometry = shapefile_cropped.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)

# Convert cropped GeoDataFrame to PyVista PolyData
shapefile_polydata_list = geodataframe_to_pyvista(shapefile_cropped)

# Initialize Plotter
plotter = pv.Plotter()

# Add shapefile geometries to plotter
for polydata in shapefile_polydata_list:
    plotter.add_mesh(polydata)

# Add the point cloud to the plotter
plotter.add_mesh(multi_block[0], color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)

# Set the plotter to view from the top
plotter.view_xy()

# Show plotter
plotter.show()
