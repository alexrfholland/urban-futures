import rasterio
import numpy as np
import random
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import rasterio.plot as rioplot
import matplotlib.pyplot as plt



def distribute_structures_with_combined_gradients(deployable_area_raster, pole_proximity_raster, canopy_proximity_raster, min_distance_meters, min_pole_distance_meters, pole_weight, canopy_weight, flip=False):
    """
    Distribute structures within deployable land areas, considering both pole and canopy proximity.
    Combines the gradients from both rasters using weights.
    """
    with rasterio.open(deployable_area_raster) as deployable_src, \
         rasterio.open(pole_proximity_raster) as pole_src, \
         rasterio.open(canopy_proximity_raster) as canopy_src:

        deployable_area = deployable_src.read(1)
        pole_distance = pole_src.read(1)
        canopy_distance = canopy_src.read(1)
        transform = deployable_src.transform

        if flip:
            deployable_area = 1 - deployable_area

            # Plotting

        # Set the border pixels to 0
        deployable_area[0, :] = 0  # Top border
        deployable_area[-1, :] = 0  # Bottom border
        deployable_area[:, 0] = 0  # Left border
        deployable_area[:, -1] = 0  # Right border


        # Normalize both rasters to a common scale (0 to 1)
        normalized_pole_distance = pole_distance / pole_distance.max()
        normalized_canopy_distance = canopy_distance / canopy_distance.max()

        # Combine the rasters using weights
        combined_gradient = (pole_weight * normalized_pole_distance + canopy_weight * normalized_canopy_distance) / (pole_weight + canopy_weight)

        x_res, y_res = deployable_src.res
        spacing_x = int(np.ceil(min_distance_meters / x_res))
        spacing_y = int(np.ceil(min_distance_meters / y_res))
        min_pole_dist_pixels = int(np.ceil(min_pole_distance_meters / x_res))

        valid_points = []
        for y in range(0, deployable_area.shape[0], spacing_y):
            for x in range(0, deployable_area.shape[1], spacing_x):
                if deployable_area[y, x] > 0 and pole_distance[y, x] >= min_pole_dist_pixels:
                    prob = combined_gradient[y, x]
                    if random.random() < prob:
                        valid_points.append((x, y))
        


        valid_coords = [rasterio.transform.xy(transform, y, x, offset='center') for x, y in valid_points]

        print(f'structures deployed are: {len(valid_coords)}')

        # Create a GeoDataFrame with geometry column
        gdf = gpd.GeoDataFrame(columns=['geometry'])
        gdf['geometry'] = [Point(easting, northing) for easting, northing in valid_coords]
        print()

        # Set the CRS (coordinate reference system) for the GeoDataFrame
        # Replace 'EPSG:28355' with the appropriate EPSG code for your data
        gdf.crs = 'EPSG:28355'

        """fig, ax = plt.subplots(figsize=(10, 6))
        rioplot.show(deployable_area, ax=ax, transform=transform)
        gdf.plot(ax=ax, marker='o', markersize=10, legend=True)
        plt.show()"""

        return gdf, combined_gradient, transform


def getLightweights(site):
    # Paths to the raster files

    isChatGpt = False

    if isChatGpt:

        pole_proximity_raster = '/mnt/data/geotiff-city-pole_proximity.tif'
        canopy_proximity_raster = '/mnt/data/geotiff-city-canopy_proximity.tif'
        deployable_area_raster = '/mnt/data/geotiff-city-public_deployable.tif'
        private_deployable_area_raster = '/mnt/data/geotiff-city-private_deployable.tif'

    else:

        import utilities.cropToPolydata as cropToPoly

        pole_proximity_rasterFull = 'data/deployables/raster_poles_distance.tif'
        canopy_proximity_rasterFull = 'data/deployables/raster_canopy_distance.tif'
        deployable_area_rasterFull = 'data/deployables/raster_parking-and-median-buffer.tif'
        private_deployable_area_rasterFull = 'data/deployables/raster_deployables_private.tif'

        pole_proximity_raster = cropToPoly.cropAndSaveGeoTiff(site, pole_proximity_rasterFull, 'pole_proximity')
        canopy_proximity_raster = cropToPoly.cropAndSaveGeoTiff(site, canopy_proximity_rasterFull, 'canopy_proximity')
        deployable_area_raster = cropToPoly.cropAndSaveGeoTiff(site, deployable_area_rasterFull, 'public_deployable')
        private_deployable_area_raster = cropToPoly.cropAndSaveGeoTiff(site, private_deployable_area_rasterFull, 'private_deployable')






    # Parameters
    min_distance_meters = 20  # Minimum distance between structures
    min_pole_distance_meters = 20  # Minimum distance from poles
    pole_weight = 5  # Weight for pole proximity
    canopy_weight = 5  # Weight for canopy proximity

    # Execute the function with the provided parameters
    deployables_coordinates, combinedGradient, transform = distribute_structures_with_combined_gradients(deployable_area_raster, pole_proximity_raster, canopy_proximity_raster, min_distance_meters, min_pole_distance_meters, pole_weight, canopy_weight)
    print(deployables_coordinates[:10])  # Display the first 10 coordinates for brevity

    private_deployables_coordinates, privateCombinedGradient,transform2 = distribute_structures_with_combined_gradients(private_deployable_area_raster, pole_proximity_raster, canopy_proximity_raster, min_distance_meters, min_pole_distance_meters, pole_weight, canopy_weight,flip=True)

    deployables_coordinates['condition'] = 'public'
    private_deployables_coordinates['condition'] = 'private'

    print(f'private deployables are {private_deployables_coordinates}')

    # Combine the two GeoDataFrames
    combined_deployables = gpd.GeoDataFrame(pd.concat([deployables_coordinates, private_deployables_coordinates], ignore_index=True))
    combined_deployables['structure'] = 'lightweight'


    print(f'combined deployables are {combined_deployables}')

    return combined_deployables, combinedGradient, privateCombinedGradient, transform


    #return combined_deployables, pole_proximity_raster, canopy_proximity_raster, pole_weight, canopy_weight, deployable_area_raster, private_deployable_area_raster



import matplotlib.pyplot as plt
import rasterio.plot as rioplot

def plotPoles(combined_gradient, transform, selected_poles_gdf):
    # Plot the combined gradient raster
    fig, ax = plt.subplots(figsize=(10, 6))
    rioplot.show(combined_gradient, ax=ax, transform=transform)

    # Plot poles using a colormap based on the 'structureType' values
    # Matplotlib will automatically handle the categorical data
    selected_poles_gdf.plot(ax=ax, marker='o', column='condition', cmap='Set1', markersize=10, legend=True)

    ax.set_title('Utility Poles Selection Based on Combined Gradient')
    plt.show()


def getPlot(structures_gdf, combined_gradient, transform):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display the combined gradient raster and capture the mappable object
    rioplot.show(combined_gradient, ax=ax, transform=transform)

    # Plotting the structures using a colormap based on the 'condition' values
    # Matplotlib will automatically handle the categorical data
    structures_gdf.plot(ax=ax, marker='o', column='condition', cmap='Set1', markersize=10, legend=True)

    ax.set_title('Combined Gradient and Structure Locations')
    ax.set_xlabel('Eastings')
    ax.set_ylabel('Northings')
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    site = 'city'
    #combined_deployables, pole_proximity_raster, canopy_proximity_raster, pole_weight, canopy_weight, deployableRaster, privateDeployableRaster = getLightweights(site)
    combined_deployables, deployableRaster, privateDeployableRaster, transform = getLightweights(site)
    combined_deployables.to_file("data/teststructures.shp", driver='ESRI Shapefile')
    getPlot(combined_deployables, deployableRaster, transform)
    print(f'point dataframe is {combined_deployables}')


