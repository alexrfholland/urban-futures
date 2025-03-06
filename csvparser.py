# Script to create the combined gradient with correct weightings for pole and canopy proximity

import rasterio
import numpy as np

def create_combined_gradient(pole_proximity_raster_path, canopy_proximity_raster_path, pole_weight, canopy_weight):
    """
    Create a combined gradient raster based on pole and canopy proximity rasters with given weightings.

    Args:
    - pole_proximity_raster_path (str): Path to the pole proximity raster file.
    - canopy_proximity_raster_path (str): Path to the canopy proximity raster file.
    - pole_weight (float): Weighting factor for the pole proximity.
    - canopy_weight (float): Weighting factor for the canopy proximity.

    Returns:
    - np.array: Combined gradient array.
    """
    with rasterio.open(pole_proximity_raster_path) as pole_src, \
         rasterio.open(canopy_proximity_raster_path) as canopy_src:

        pole_distance = pole_src.read(1)
        canopy_distance = canopy_src.read(1)

        # Normalize both rasters to a common scale (0 to 1)
        normalized_pole_distance = pole_distance / pole_distance.max()
        normalized_canopy_distance = canopy_distance / canopy_distance.max()

        # Combine the rasters using weights
        combined_gradient = (pole_weight * normalized_pole_distance + canopy_weight * normalized_canopy_distance) / (pole_weight + canopy_weight)

        return combined_gradient

# Define file paths and weightings
pole_proximity_raster_path = '/mnt/data/pole-proximity.tif'
canopy_proximity_raster_path = '/mnt/data/canopy-proximity.tif'
pole_weight = 50
canopy_weight = 30

# Create the combined gradient
combined_gradient = create_combined_gradient(pole_proximity_raster_path, canopy_proximity_raster_path, pole_weight, canopy_weight)

# The combined_gradient variable now holds the weighted combination of pole and canopy proximity
combined_gradient.shape  # Outputting the shape as a basic check# Script to create the combined gradient with correct weightings for pole and canopy proximity

import rasterio
import numpy as np

def create_combined_gradient(pole_proximity_raster_path, canopy_proximity_raster_path, pole_weight, canopy_weight):
    """
    Create a combined gradient raster based on pole and canopy proximity rasters with given weightings.

    Args:
    - pole_proximity_raster_path (str): Path to the pole proximity raster file.
    - canopy_proximity_raster_path (str): Path to the canopy proximity raster file.
    - pole_weight (float): Weighting factor for the pole proximity.
    - canopy_weight (float): Weighting factor for the canopy proximity.

    Returns:
    - np.array: Combined gradient array.
    """
    with rasterio.open(pole_proximity_raster_path) as pole_src, \
         rasterio.open(canopy_proximity_raster_path) as canopy_src:

        pole_distance = pole_src.read(1)
        canopy_distance = canopy_src.read(1)

        # Normalize both rasters to a common scale (0 to 1)
        normalized_pole_distance = pole_distance / pole_distance.max()
        normalized_canopy_distance = canopy_distance / canopy_distance.max()

        # Combine the rasters using weights
        combined_gradient = (pole_weight * normalized_pole_distance + canopy_weight * normalized_canopy_distance) / (pole_weight + canopy_weight)

        return combined_gradient

# Define file paths and weightings
#pole_proximity_raster_path = '/mnt/data/pole-proximity.tif'
#canopy_proximity_raster_path = '/mnt/data/canopy-proximity.tif'

pole_proximity_raster_path = 'modules/deployables/data/pole-proximity.tif'
canopy_proximity_raster_path = 'modules/deployables/data/canopy-proximity.tif'

pole_weight = 50
canopy_weight = 30

# Create the combined gradient
combined_gradient = create_combined_gradient(pole_proximity_raster_path, canopy_proximity_raster_path, pole_weight, canopy_weight)

# The combined_gradient variable now holds the weighted combination of pole and canopy proximity
print(combined_gradient.shape)  # Outputting the shape as a basic check

# Function to plot the combined gradient

def plot_combined_gradient(combined_gradient):
    """
    Plot the combined gradient array.

    Args:
    - combined_gradient (np.array): The combined gradient array to plot.

    Returns:
    - None: Displays the plot.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(combined_gradient, cmap='plasma')
    plt.colorbar(label='Combined Gradient Value')
    plt.title('Combined Gradient of Pole and Canopy Proximity')
    plt.show()

# Call the function to plot the combined gradient
plot_combined_gradient(combined_gradient)
