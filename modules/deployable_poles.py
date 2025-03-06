"""
To create a weighting function that accounts for canopy gaps, we'll first need to normalize the gradient canopy map, where the values represent the distance to the nearest tree canopy. The normalization will ensure that the values range between 0 and 1. Then, we can apply a weight factor, such as 50 in your example, to indicate that poles within 50 meters of a canopy gap will have a 100% probability of being selected as deployable locations.

Here's a step-by-step approach to achieve this:

Normalize the Gradient Canopy Map: Rescale the distance values in the GeoTIFF so that they range from 0 to 1. A value of 1 would indicate the maximum distance within the raster, and 0 would be the minimum (closest to the canopy).
Apply Weight Factor: Multiply the normalized values by the weight factor (e.g., 50). This transforms the normalized distance values, so that a value of 1 now represents a distance of 50 meters.
Select Poles Probabilistically: For each pole, retrieve the corresponding weighted value from the raster. This value represents the probability of a pole being selected. Poles closer to canopy gaps (higher weighted values) will have a higher chance of being selected.
"""



import geopandas as gpd
import rasterio
import rasterio.plot as rioplot
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np
import utilities.cropToPolydata as cropToPoly
import pandas as pd

import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

from rasterio.transform import rowcol




import rasterio
import numpy as np
import geopandas as gpd
from rasterio.transform import rowcol


def extract_and_plot_poles(poles_gdf, bounds):
    """
    Extracts all poles within the bounds of a new GeoTIFF and plots the results with the GeoTIFF.

    Parameters:
    poles_gdf (GeoDataFrame): GeoDataFrame containing pole locations.
    bounds (tuple): A tuple of (min_x, min_y, max_x, max_y) defining the bounds.

    Returns:
    GeoDataFrame: A GeoDataFrame containing poles within the bounds.
    """

    # Print the input bounds
    print(f"Input bounds: {bounds}")

    # Create a bounding box from the bounds tuple
    min_x, min_y, max_x, max_y = bounds
    area_of_interest = box(min_x, min_y, max_x, max_y)

    # Extract poles within the bounds
    poles_within_bounds = poles_gdf[poles_gdf.geometry.intersects(area_of_interest)]

    # Print the bounds of the cropped poles
    if not poles_within_bounds.empty:
        print(f"Bounds of cropped poles: {poles_within_bounds.total_bounds}")
    else:
        print("No poles found within the specified bounds.")

    # Plot the results
    """fig, ax = plt.subplots(figsize=(10, 6))
    poles_within_bounds.plot(ax=ax, marker='o', color='red', markersize=5)
    ax.set_title('Utility Poles within the New GeoTIFF Bounds')
    plt.show()
    """

    poles_within_bounds_no_data = poles_within_bounds[['geometry']].copy()


    return poles_within_bounds_no_data

def normalise_canopy_raster(raster_data, weight_factor):
    """
    Normalizes a raster file and applies a weight factor.

    Parameters:
    raster_path (str): Path to the raster file.
    weight_factor (float): The weight factor to apply after normalization.

    Returns:
    numpy.ndarray: A normalized and weighted raster array.
    """

    # Normalizing the raster data
    min_val = raster_data.min()
    max_val = raster_data.max()
    normalized_raster = (raster_data - min_val) / (max_val - min_val)

    # Applying the weight factor
    weighted_raster = normalized_raster * weight_factor

    return weighted_raster

def select_poles_probabilistically(poles_gdf, weighted_raster, transform):
    """
    Selects poles probabilistically based on the weighted raster values and marks selected poles with their condition.

    Parameters:
    poles_gdf (GeoDataFrame): GeoDataFrame containing pole locations.
    weighted_raster (numpy.ndarray): The weighted raster array.
    transform (Affine): The affine transformation of the raster.

    Returns:
    GeoDataFrame: A GeoDataFrame containing all poles with additional attributes 'condition' and 'structure'.
    """
    # Initialize 'structure' and 'condition' columns
    poles_gdf['structure'] = 'pole'
    poles_gdf['condition'] = 'empty'  # Default condition

    for idx, pole in poles_gdf.iterrows():
        # Get row, col in the raster for each pole
        row, col = rowcol(transform, pole.geometry.x, pole.geometry.y)

        try:
            weight = weighted_raster[row, col]
            print('weight of pole is', weight)
        # Rest of the logic...
        except IndexError:
            print('Index out of bounds for row:', row, 'col:', col)

        # Check if the pole is within the raster bounds
        if 0 <= row < weighted_raster.shape[0] and 0 <= col < weighted_raster.shape[1]:
            weight = weighted_raster[row, col]

            # Use the weight as the probability for selecting the pole
            # If the random number is less than or equal to the weight, the condition is set to 'enriched'. 
            # This means that the higher the weight, the higher the probability that the random number falls below it, 
            # and thus the pole is more likely to be 'enriched'.
            # If the random number is greater than the weight, the condition remains 'empty'.
            if np.random.random() <= weight:
                poles_gdf.at[idx, 'condition'] = 'enriched'

    return poles_gdf

def loadFiles(site):

    canopy_proximity_path = 'data/deployables/raster_canopy_distance.tif'
    pole_shapefile_path = 'data/deployables/poles.shp'

    canopyPath = cropToPoly.cropAndSaveGeoTiff(site, canopy_proximity_path, 'canopy-proximity')
    
    # Load the shapefile containing pole locations
    poles_gdf = gpd.read_file(pole_shapefile_path)
    
    return poles_gdf, canopyPath


def plotPoles(combined_gradient, transform, selected_poles_gdf):
    # Plot the combined gradient raster
    fig, ax = plt.subplots(figsize=(10, 6))
    rioplot.show(combined_gradient, ax=ax, transform=transform)

    # Plot poles using a colormap based on the 'structureType' values
    # Matplotlib will automatically handle the categorical data
    selected_poles_gdf.plot(ax=ax, marker='o', column='condition', cmap='Set1', markersize=10, legend=True)

    ax.set_title('Utility Poles Selection Based on Combined Gradient')
    plt.show()


def runUtilityPole(site, poleProbWeight=50):
    poles_gdf, canopyProxyFilePath = loadFiles(site)

    # Open the raster file
    with rasterio.open(canopyProxyFilePath) as src:
        # Read the first band (or another band if necessary)
        treeRasterData = src.read(1)
        # Get the bounds of the raster
        bounds = src.bounds
        # Get the affine transform for the raster
        transform = src.transform

    poleLocs = extract_and_plot_poles(poles_gdf, bounds)
    weighted_canopy_raster = normalise_canopy_raster(treeRasterData, poleProbWeight)

    selected_poles_gdf = select_poles_probabilistically(poleLocs, weighted_canopy_raster, transform)

    return selected_poles_gdf, weighted_canopy_raster, transform


if __name__ == "__main__":
    site = 'city'
    selected_poles_gdf, weighted_canopy_raster, transform = runUtilityPole(site)
    selected_poles_gdf.to_file("data/testPoles.shp", driver='ESRI Shapefile')
    plotPoles(weighted_canopy_raster, transform, selected_poles_gdf)


    ##easting_offset, northing_offset = cropToPoly.getEastingAndNorthing(site)

