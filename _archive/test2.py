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
import modules.utilities.cropToPolydata as cropToPoly
import pandas as pd


def extract_and_plot_poles(poles_gdf, bounds_gdf):
    """
    Extracts all poles within the bounds of a new GeoTIFF and plots the results with the GeoTIFF.

    Parameters:
    poles_gdf (GeoDataFrame): GeoDataFrame containing pole locations.
    bounds_gdf (GeoDataFrame): GeoDataFrame defining the area of interest.
    transform (Affine): The affine transform from the GeoTIFF.

    Returns:
    GeoDataFrame: A GeoDataFrame containing poles within the bounds.
    """

    # Use the first (and possibly only) geometry in the bounds GeoDataFrame as the area of interest
    area_of_interest = bounds_gdf.geometry.iloc[0]

    # Extract poles within the bounds
    poles_within_bounds = poles_gdf[poles_gdf.geometry.within(area_of_interest)]

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    poles_within_bounds.plot(ax=ax, marker='o', color='red', markersize=5)
    ax.set_title('Utility Poles within the New GeoTIFF Bounds')
    plt.show()

    return poles_within_bounds


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


def select_poles_probabilistically(raster_path, poles_gdf, weighted_raster):
    """
    Selects poles probabilistically based on the weighted raster values, considering only poles within the GeoTIFF bounds.

    Parameters:
    raster_path (str): Path to the raster file.
    poles_gdf (GeoDataFrame): GeoDataFrame containing pole locations within the bounds of the raster.
    weighted_raster (numpy.ndarray): The weighted raster array.

    Returns:
    GeoDataFrame: A GeoDataFrame containing selected poles.
    """
    with rasterio.open(raster_path) as src:
        transform = src.transform

    selected_poles = []
    for _, pole in poles_gdf.iterrows():
        # Get row, col in the raster for each pole
        row, col = rowcol(transform, pole.geometry.x, pole.geometry.y)

        # Check if the pole is within the raster bounds
        if 0 <= row < weighted_raster.shape[0] and 0 <= col < weighted_raster.shape[1]:
            weight = weighted_raster[row, col]

            # Use the weight as the probability for selecting the pole
            if np.random.random() <= weight / 50:  # Dividing by 50 to get the probability
                selected_poles.append(pole)

    return gpd.GeoDataFrame(selected_poles, columns=poles_gdf.columns)

from rasterio.transform import rowcol

def select_poles_probabilistically(poles_gdf, weighted_raster, transform):
    """
    Selects poles probabilistically based on the weighted raster values.

    Parameters:
    poles_gdf (GeoDataFrame): GeoDataFrame containing pole locations.
    weighted_raster (numpy.ndarray): The weighted raster array.
    transform (Affine): The affine transformation of the raster.

    Returns:
    GeoDataFrame: A GeoDataFrame containing selected poles.
    """
    selected_poles = []
    for _, pole in poles_gdf.iterrows():
        # Get row, col in the raster for each pole
        row, col = rowcol(transform, pole.geometry.x, pole.geometry.y)

        # Check if the pole is within the raster bounds
        if 0 <= row < weighted_raster.shape[0] and 0 <= col < weighted_raster.shape[1]:
            weight = weighted_raster[row, col]

            # Use the weight as the probability for selecting the pole
            if np.random.random() <= weight / 50:  # Dividing by 50 to get the probability
                selected_poles.append(pole)

    return gpd.GeoDataFrame(selected_poles, columns=poles_gdf.columns)

def loadFiles(site):
    # Paths to the raster files
    #canopy_proximity_path = 'modules/deployables/data/canopy-proximity.tif'
    #pole_shapefile_path = 'modules/deployables/data/poles/poles.shp'

    canopy_proximity_path = 'data/deployables/raster_canopy_distance.tif'
    pole_shapefile_path = 'data/deployables/poles.shp'
    
    treeRasterData, treeTransform, treeBounds = cropToPoly.cropGeoTiff(site, canopy_proximity_path)

    # Load the shapefile containing pole locations
    poles_gdf = gpd.read_file(pole_shapefile_path)

    """ Load the GeoTIFF file to get its bounds and image data
    with rasterio.open(canopy_proximity_path) as src:
        treeBounds = src.bounds
        treeRasterData = src.read(1)
        treeTransform = src.transform
    """
    return poles_gdf, treeRasterData, treeBounds, treeTransform

def GetPoles(poles_gdf, easting_offset, northing_offset, weight_factor, weighted_raster, transform):
    """
    Extracts pole locations, transforms their coordinates, and checks for deployment probability.

    Parameters:
    poles_gdf (GeoDataFrame): GeoDataFrame containing pole locations.
    easting_offset (float): The easting offset to apply.
    northing_offset (float): The northing offset to apply.
    weight_factor (float): The weight factor for deployment probability.
    weighted_raster (numpy.ndarray): The weighted raster array.
    transform (Affine): The affine transformation of the raster.

    Returns:
    DataFrame, GeoDataFrame: DataFrames with transformed and original pole coordinates.
    """
    pole_data = []
    pole_geo_data = []

    for _, pole in poles_gdf.iterrows():
        # Transform coordinates to polydata coords
        x_poly, y_poly = pole.geometry.x - easting_offset, pole.geometry.y - northing_offset

        # Determine deployment probability
        row, col = rowcol(transform, pole.geometry.x, pole.geometry.y)
        is_deployed = False
        if 0 <= row < weighted_raster.shape[0] and 0 <= col < weighted_raster.shape[1]:
            weight = weighted_raster[row, col]
            is_deployed = np.random.random() <= weight / weight_factor

        pole_data.append({'x': x_poly, 'y': y_poly, 'isDeployedPole': is_deployed})
        pole_geo_data.append({'geometry': pole.geometry, 'isDeployedPole': is_deployed})

    # Create DataFrame and GeoDataFrame
    df = pd.DataFrame(pole_data)
    gdf = gpd.GeoDataFrame(pole_geo_data, geometry='geometry', crs=poles_gdf.crs)

    return df, gdf

    
site = 'city'
poles_gdf, treeRasterData, treeBounds, treeTransform = loadFiles(site)    
poleLocs = extract_and_plot_poles(poles_gdf, treeBounds)
weighted_canopy_raster = normalise_canopy_raster(treeRasterData, 500)
selected_poles_gdf = select_poles_probabilistically(poleLocs, weighted_canopy_raster, treeTransform)

easting_offset, northing_offset = cropToPoly.getEastingAndNorthing(site)

# Plot the selected poles
fig, ax = plt.subplots(figsize=(10, 6))
rioplot.show(weighted_canopy_raster, ax=ax, transform=treeTransform)
poleLocs.plot(ax=ax, marker='o', color='white', markersize=5)
selected_poles_gdf.plot(ax=ax, marker='o', color='green', markersize=10)
ax.set_title('Selected Utility Poles Based on Canopy Proximity')
plt.show()