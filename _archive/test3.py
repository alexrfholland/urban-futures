import rasterio
import numpy as np
import random

# Paths to the raster files
deployable_area_raster = 'modules/deployables/data/deployable_area.tif'
pole_proximity_raster = 'modules/deployables/data/pole-proximity.tif'
canopy_proximity_raster = 'modules/deployables/data/canopy-proximity.tif'
private_deployable_area_raster = 'modules/deployables/data/deployable private.tif'


def distribute_structures_with_combined_gradients(deployable_area_raster, pole_proximity_raster, canopy_proximity_raster, min_distance_meters, min_pole_distance_meters, pole_weight, canopy_weight):
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

        


        # Normalize both rasters to a common scale (0 to 1)
        normalized_pole_distance = pole_distance / pole_distance.max()
        normalized_canopy_distance = canopy_distance / canopy_distance.max()

        # Flip the raster values (1's to 0's and vice versa)
        
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
                    # Adjust probability based on the combined gradient
                    prob = combined_gradient[y, x]
                    if random.random() < prob:
                        valid_points.append((x, y))

        valid_coords = [rasterio.transform.xy(transform, y, x, offset='center') for x, y in valid_points]

        return valid_coords

# Parameters
min_distance_meters = 10  # Minimum distance between structures
min_pole_distance_meters = 5  # Minimum distance from poles
pole_weight = 50  # Weight for pole proximity
canopy_weight = 30  # Weight for canopy proximity

# Execute the function with the provided parameters
deployables_coordinates = distribute_structures_with_combined_gradients(deployable_area_raster, pole_proximity_raster, canopy_proximity_raster, min_distance_meters, min_pole_distance_meters, pole_weight, canopy_weight)
print(deployables_coordinates[:10])  # Display the first 10 coordinates for brevity

private_deployables_coordinates = distribute_structures_with_combined_gradients(private_deployable_area_raster, pole_proximity_raster, canopy_proximity_raster, min_distance_meters, min_pole_distance_meters, pole_weight, canopy_weight)


import matplotlib.pyplot as plt

def plot_combined_gradient_and_structures(combined_gradient, public_structures, private_structures):
    """
    Plot the combined gradient of pole and canopy proximity and the positions of deployable structures.
    """
    with rasterio.open(deployable_area_raster) as src:
        # Read the first band of the raster
        deployable_area = src.read(1)
        # Get the spatial extent of the raster
        extent = src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top

    # Convert structure coordinates to array indices
    valid_x, valid_y = zip(*[(x, y) for x, y in public_structures])
    valid_x_private, valid_y_private = zip(*[(x, y) for x, y in private_structures])

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.imshow(combined_gradient, cmap='viridis', extent=extent)
    plt.colorbar(label='Combined Gradient')
    plt.scatter(valid_x, valid_y, color='red', s=10)  # Plot the structure locations
    plt.scatter(valid_x_private, valid_y_private, color='red', s=10)  # Plot the structure locations
    plt.title('Combined Gradient and Structure Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

# Recompute the combined gradient
with rasterio.open(pole_proximity_raster) as pole_src, \
     rasterio.open(canopy_proximity_raster) as canopy_src:

    pole_distance = pole_src.read(1)
    canopy_distance = canopy_src.read(1)

    # Normalize both rasters to a common scale (0 to 1)
    normalized_pole_distance = pole_distance / pole_distance.max()
    normalized_canopy_distance = canopy_distance / canopy_distance.max()

    # Combine the rasters using weights
    combined_gradient = (pole_weight * normalized_pole_distance + canopy_weight * normalized_canopy_distance) / (pole_weight + canopy_weight)

# Plot the combined gradient and structure positions
plot_combined_gradient_and_structures(combined_gradient, deployables_coordinates, private_deployables_coordinates)