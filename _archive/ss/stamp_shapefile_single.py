import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from scipy.spatial import cKDTree

def transfer_attributes(siteData, raster_cloud, shapefile_cropped):
    print("Unique IDs in raster_cloud:", np.unique(raster_cloud.point_data['id']))
    print("Indices in shapefile_cropped:", shapefile_cropped.index)

    print("Properties of shapefile:", shapefile_cropped.columns)

    # Build k-d tree for raster_cloud using only x and y coordinates
    tree = cKDTree(raster_cloud.points[:, :2])
    
    # Query the k-d tree using only x and y coordinates from siteData
    print("Performing k-d tree query...")
    _, indices = tree.query(siteData.points[:, :2])
    print("k-d tree query complete.")

    # Extract IDs from raster_cloud point data to map to original shapefile rows
    print("Extracting nearest IDs...")
    nearest_ids = raster_cloud.point_data['id'][indices]
    print("Extraction complete.")

    # Exclude the geometry column
    columns_to_transfer = [col for col in shapefile_cropped.columns if col != 'geometry']

    # Create an empty DataFrame to hold the attribute data
    attribute_df = pd.DataFrame(index=np.arange(siteData.n_points), columns=columns_to_transfer)

    print("Transferring attributes...")
    # Directly assign values to DataFrame using Pandas' `.loc` based on the nearest_ids
    attribute_df.loc[:, columns_to_transfer] = shapefile_cropped.loc[nearest_ids, columns_to_transfer].values
    print("Attribute transfer complete.")

    # Convert DataFrame to a dictionary and update siteData point_data
    attribute_data = attribute_df.to_dict(orient='list')
    for attr, values in attribute_data.items():
        siteData.point_data[attr] = np.array(values)

        # Initialize Plotter
    """plotter = pv.Plotter()

    # Add the point clouds to the plotter
    plotter.add_mesh(siteData, scalars='OS_TYPE', cmap='rainbow', point_size=5.0, render_points_as_spheres=True)

    #plotter.add_mesh(siteData, color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)
    #plotter.add_mesh(raster_cloud, scalars='id', cmap='rainbow', point_size=5.0, render_points_as_spheres=True)

    # Set the plotter to view from the top
    plotter.view_xy()

    # Show plotter
    plotter.show()"""


def transfer_attributes2(siteData, raster_cloud, shapefile_cropped):
    print("Unique IDs in raster_cloud:", np.unique(raster_cloud.point_data['id']))
    print("Indices in shapefile_cropped:", shapefile_cropped.index)


    # Build k-d tree for raster_cloud using only x and y coordinates
    tree = cKDTree(raster_cloud.points[:, :2])
    
    # Query the k-d tree using only x and y coordinates from siteData
    _, indices = tree.query(siteData.points[:, :2])

    # Extract IDs from raster_cloud point data to map to original shapefile rows
    nearest_ids = raster_cloud.point_data['id'][indices]

    # Initialize empty dictionaries for each attribute column
    attribute_data = {}
    
    # Exclude the geometry column
    columns_to_transfer = [col for col in shapefile_cropped.columns if col != 'geometry']
    
    # Populate attribute data by looking up in the original shapefile
    for attr in columns_to_transfer:
        dtype = shapefile_cropped[attr].dtype
        attribute_data[attr] = np.empty(siteData.n_points, dtype=dtype)
        for i, nearest_id in enumerate(nearest_ids):
            attribute_data[attr][i] = shapefile_cropped.loc[nearest_id, attr]

    # Add attribute data to siteData point_data
    for attr, values in attribute_data.items():
        siteData.point_data[attr] = values

    return siteData

def raster_with_ids(shapefile_cropped, transform, out_shape):
    shapes = [(geom, value) for value, geom in enumerate(shapefile_cropped.geometry)]
    raster_array = rasterize(shapes, transform=transform, out_shape=out_shape)
    return raster_array

def raster_to_point_cloud(mask, box_offset):
    rows, cols = np.where(mask >= 0)
    ids = mask[rows, cols]
    x = cols - box_offset
    y = -rows + box_offset
    z = np.zeros_like(rows)
    coordinates = np.column_stack((x, y, z))
    cloud = pv.PolyData(coordinates)
    cloud.point_data['id'] = ids
    return cloud

def geodataframe_to_pyvista(gdf):
    polydata_list = []
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            exterior = np.array(geom.exterior.coords)
            exterior_3d = np.c_[exterior, np.zeros(len(exterior))]
            polydata = pv.PolyData(exterior_3d)
            polydata_list.append(polydata)
    return polydata_list

def read_and_plot(site_path, shapefile_path, bounding_box_size):
    PIXEL_RESOLUTION = 1
    BOX_OFFSET = bounding_box_size // 2

    multi_block = pv.read(site_path)
    siteData = multi_block[0]

    # Read site coordinate from CSV
    easting_offset = site_coord.loc[site_coord['Name'] == 'city', 'Easting'].values[0]
    northing_offset = site_coord.loc[site_coord['Name'] == 'city', 'Northing'].values[0]

    # Create bounding box
    bbox = box(easting_offset - BOX_OFFSET, northing_offset - BOX_OFFSET, easting_offset + BOX_OFFSET, northing_offset + BOX_OFFSET)

    # Read and crop the shapefile
    shapefile = gpd.read_file(shapefile_path)
    shapefile_cropped = shapefile[shapefile.geometry.intersects(bbox)]

    # Translate the cropped shapefile
    shapefile_cropped.geometry = shapefile_cropped.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
    shapefile_cropped.reset_index(drop=True, inplace=True)


    # Convert cropped GeoDataFrame to PyVista PolyData
    shapefile_polydata_list = geodataframe_to_pyvista(shapefile_cropped)

    # Rasterize the shapefile
    transform = rasterio.transform.from_origin(-BOX_OFFSET, BOX_OFFSET, PIXEL_RESOLUTION, PIXEL_RESOLUTION)
    out_shape = (bounding_box_size, bounding_box_size)
    id_raster = raster_with_ids(shapefile_cropped, transform, out_shape)

    # Convert raster mask to point cloud
    raster_cloud = raster_to_point_cloud(id_raster, BOX_OFFSET)

    # Transfer attributes from nearest points in raster_cloud to siteData
    transfer_attributes(siteData, raster_cloud, shapefile_cropped)

    #Initialize Plotter
    plotter = pv.Plotter()

    # Add shapefile geometries to plotter
    for polydata in shapefile_polydata_list:
        plotter.add_mesh(polydata)

    print(f'properties of processed site data are: {siteData.point_data}')

    # Add the point clouds to the plotter
    plotter.add_mesh(siteData, scalars='RATING', cmap='rainbow', point_size=5.0, render_points_as_spheres=True)

    #plotter.add_mesh(siteData, color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)
    plotter.add_mesh(raster_cloud, scalars='id', cmap='rainbow', point_size=5.0, render_points_as_spheres=True)

    # Set the plotter to view from the top
    plotter.view_xy()

    # Show plotter
    plotter.show()

if __name__ == "__main__":
    BOUNDING_BOX_SIZE = 500
    SITE = 'city'
    site_coord = pd.read_csv('data/site projections.csv')
    SITEPATH = f'data/{SITE}/flattened-{SITE}.vtm'
    SHAPEFILE_PATH = 'data/shapefiles/green-roof/mga55_gda94_green_roof_intensive.shp'
    #SHAPEFILE_PATH = 'data/shapefiles/laneyways/laneways.shp'
    #SHAPEFILE_PATH = 'data/shapefiles/open_space/open-space.shp'


    read_and_plot(SITEPATH, SHAPEFILE_PATH, BOUNDING_BOX_SIZE)
