import pyvista as pv
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os

VISUALISE = False


def transfer_attributes(siteData, raster_cloud, shapefile_cropped, identifier=''):
    print("Properties of shapefile:", shapefile_cropped.columns)

    # Exclude the geometry column and initialize attribute_df
    columns_to_transfer = [col for col in shapefile_cropped.columns if col != 'geometry']
    attribute_df = pd.DataFrame(index=np.arange(siteData.n_points), columns=columns_to_transfer)

    # Assign default values based on data type
    for col in columns_to_transfer:
        if pd.api.types.is_bool_dtype(shapefile_cropped[col]):
            attribute_df[col] = False
        elif pd.api.types.is_string_dtype(shapefile_cropped[col]):
            attribute_df[col] = "-1"
        elif pd.api.types.is_numeric_dtype(shapefile_cropped[col]):
            attribute_df[col] = -1
        else:
            attribute_df[col] = np.nan  # Handle other data types as NaN

    # Check if raster_cloud is None
    if raster_cloud is None:
        print("raster_cloud is None, assigning default values to all attributes.")
    else:
        print("Unique IDs in raster_cloud:", np.unique(raster_cloud.point_data['id']))
        print("Indices in shapefile_cropped:", shapefile_cropped.index)

        # Build k-d tree for raster_cloud using only x and y coordinates
        tree = cKDTree(raster_cloud.points[:, :2])

        # Query the k-d tree using only x and y coordinates from siteData
        print("Performing k-d tree query...")
        _, indices = tree.query(siteData.points[:, :2])  
        print("k-d tree query complete.")

        # Get nearest IDs from the raster_cloud
        nearest_ids = raster_cloud.point_data['id'][indices]

        # Find indices where the nearest ID is not -1
        valid_indices = np.where(nearest_ids != -1)[0]

        # Directly assign values to DataFrame based on the nearest_ids
        attribute_df.loc[valid_indices, columns_to_transfer] = shapefile_cropped.loc[nearest_ids[valid_indices], columns_to_transfer].values

        # Create 'is{identifier}' column and initialize with False
        attribute_df[f'is{identifier}'] = False

        # Set the value of 'is{identifier}' column to True for valid_indices
        attribute_df.loc[valid_indices, f'is{identifier}'] = True

    # Convert DataFrame to a dictionary and update siteData point_data
    attribute_data = attribute_df.to_dict(orient='list')
    for attr, values in attribute_data.items():
        siteData.point_data[f"{identifier}-{attr}"] = np.array(values)  # add attribute with identifier appended to attribute name

    print(f"Attribute transfer complete.")


def transfer_attributesOLD(siteData, raster_cloud, shapefile_cropped, identifier=''):
    print("Unique IDs in raster_cloud:", np.unique(raster_cloud.point_data['id']))
    print("Indices in shapefile_cropped:", shapefile_cropped.index)
    print("Properties of shapefile:", shapefile_cropped.columns)

    # Build k-d tree for raster_cloud using only x and y coordinates
    tree = cKDTree(raster_cloud.points[:, :2])

    # Query the k-d tree using only x and y coordinates from siteData
    print("Performing k-d tree query...")
    _, indices = tree.query(siteData.points[:, :2])  
    print("k-d tree query complete.")

    # Exclude the geometry column
    columns_to_transfer = [col for col in shapefile_cropped.columns if col != 'geometry']

    # Initialize the attributes with a default value of -1
    attribute_df = pd.DataFrame(index=np.arange(siteData.n_points), columns=columns_to_transfer)
    attribute_df.fillna(-1, inplace=True)

    # Get nearest IDs from the raster_cloud
    nearest_ids = raster_cloud.point_data['id'][indices]

    # Find indices where the nearest ID is not -1
    valid_indices = np.where(nearest_ids != -1)[0]

    # Directly assign values to DataFrame using Pandas' `.loc` based on the nearest_ids
    attribute_df.loc[valid_indices, columns_to_transfer] = shapefile_cropped.loc[nearest_ids[valid_indices], columns_to_transfer].values

    
    # Create 'is{identifier}' column and initialize with 0
    attribute_df[f'is{identifier}'] = 0

    # Set the value of 'is{identifier}' column to 1 for valid_indices
    attribute_df.loc[valid_indices, f'is{identifier}'] = 1

    # Convert DataFrame to a dictionary and update siteData point_data
    attribute_data = attribute_df.to_dict(orient='list')
    for attr, values in attribute_data.items():
        if attr == f'is{identifier}':
            siteData.point_data[attr] = np.array(values)  # add attributes
        else:
            siteData.point_data[f"{identifier}-{attr}"] = np.array(values)  # add attribute with identifier appended to attribute name

    print(f"Attribute transfer for {shapefile_cropped} complete.")



def raster_with_ids(shapefile_cropped, transform, out_shape):
    shapes = [(geom, value) for value, geom in enumerate(shapefile_cropped.geometry)]

    try:
        raster_array = rasterize(shapes, transform=transform, out_shape=out_shape, fill=-1)
    except ValueError as e:
        print(f"no shapefile shapes within bounds: {e}")
        # Handle the exception, e.g., by setting raster_array to None or an array of fill values
        raster_array = None  # Or any other fallback action

    return raster_array

def raster_to_point_cloud(mask, box_offset, pixel_resolution):
    rows, cols = np.indices(mask.shape)
    ids = mask[rows, cols]
    x = (cols.flatten() - box_offset) * pixel_resolution
    y = (-rows.flatten() + box_offset) * pixel_resolution
    z = np.zeros_like(x)
    coordinates = np.column_stack((x, y, z))
    cloud = pv.PolyData(coordinates)
    cloud.point_data['id'] = ids.flatten()

    print(f"unique attributes for id are {np.unique(cloud.point_data['id'])}")
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

def read_and_plot(site, siteData, shapefile_paths, bounding_box_size=None, deleteAttributes=False, buffer=0):
    # Set the pixel resolution to 0.25
    PIXEL_RESOLUTION = 1
    # Calculate the offset for the bounding box based on its size

    if bounding_box_size is None:
        bounds = siteData.bounds
        bounding_box_size = max(bounds[1] - bounds[0], bounds[5] - bounds[4])  # Use the larger of X or Z dimensions

    BOX_OFFSET = bounding_box_size // 2

    # Read site coordinates from a CSV file
    site_coord = pd.read_csv('data/site projections.csv')
    print(site)
    print(site_coord)

    # Extract easting and northing offsets for the site
    easting_offset = site_coord.loc[site_coord['Name'].str.contains(site, case=False), 'Easting'].values[0]
    northing_offset = site_coord.loc[site_coord['Name'].str.contains(site, case=False), 'Northing'].values[0]

    # Create a bounding box around the site
    bbox = box(easting_offset - BOX_OFFSET, northing_offset - BOX_OFFSET, easting_offset + BOX_OFFSET, northing_offset + BOX_OFFSET)

    # Iterate over all shapefiles in the paths list
    for shapefile_path in shapefile_paths:
        
        # Read the shapefile
        shapefile = gpd.read_file(shapefile_path)
        if(deleteAttributes):
            shapefile = shapefile[['geometry']]

        if(shapefile_path == 'data/deployables/shapefiles/parkingmedian/parking_median_buffer.shp'):
            buffer_distance = -2  # Example buffer distance
            shapefile['geometry'] = shapefile['geometry'].buffer(buffer_distance)

        if(buffer != 0):
            buffer_distance = buffer  # Example buffer distance
            shapefile['geometry'] = shapefile['geometry'].buffer(buffer_distance)

        


        # Crop the shapefile to the area within the bounding box
        shapefile_cropped = shapefile[shapefile.geometry.intersects(bbox)]

        # Translate the cropped shapefile geometry to align with the site coordinates
        shapefile_cropped.geometry = shapefile_cropped.geometry.translate(xoff=-easting_offset, yoff=-northing_offset)
        shapefile_cropped.reset_index(drop=True, inplace=True)

        # Define the transformation for rasterizing the shapefile
        # The origin is the top-left corner of the raster
        transform = rasterio.transform.from_origin(-BOX_OFFSET, BOX_OFFSET, PIXEL_RESOLUTION, PIXEL_RESOLUTION)
        # Define the output shape of the raster based on the new resolution
        out_shape = (int(bounding_box_size / PIXEL_RESOLUTION), int(bounding_box_size / PIXEL_RESOLUTION))
        # Rasterize the shapefile
        id_raster = raster_with_ids(shapefile_cropped, transform, out_shape)

        if VISUALISE:
            if id_raster is not None:
                # Plot the raster for visualization
                plt.imshow(id_raster, cmap='rainbow', origin='upper')
                plt.colorbar(label='ID')
                plt.title('ID Raster')
                plt.show()
            else:
                print("No valid geometry objects found for rasterization.")

        
        if id_raster is None:
            #raster not within bounds of site
            raster_cloud = None
            shapefile_cropped = shapefile
        else:
        # Convert the raster to a point cloud
            raster_cloud = raster_to_point_cloud(id_raster, BOX_OFFSET, PIXEL_RESOLUTION)

        # Transfer attributes from the raster cloud to the site data
        attribute_header = os.path.basename(os.path.dirname(shapefile_path))

        if(buffer != 0):
            attribute_header = f'{attribute_header}_buffer'


        transfer_attributes(siteData, raster_cloud, shapefile_cropped, attribute_header)

        """attribute_header = os.path.basename(os.path.dirname(shapefile_path))

        if id_raster is None:
            print(f'no {attribute_header} in bounds of {site}')
        else:
            raster_cloud = raster_to_point_cloud(id_raster, BOX_OFFSET, PIXEL_RESOLUTION)
            transfer_attributesOLD(siteData, raster_cloud, shapefile_cropped, attribute_header)"""

    print(f'all shapefiles transferred')
    print(f'polydata now has attributes {siteData.point_data}')

    if VISUALISE:

        plotter = pv.Plotter()
        plotter.add_mesh(siteData, scalars = 'private_buffer-isprivate_buffer', cmap = 'rainbow', point_size=5.0, render_points_as_spheres=True)
        plotter.show()

    return siteData