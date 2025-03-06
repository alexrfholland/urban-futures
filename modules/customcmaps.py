import pyvista as pv
import numpy as np
import palettable as pal
import matplotlib.colors as mcolors
import matplotlib as plt
import matplotlib.cm as cm

# Function to create custom colormap
def create_custom_colormap(palettes):
    colors = []  # List to hold the colors from specified palettes
    for palette in palettes:
        # Access the palette from palettable
        cmap = eval(f'pal.colorbrewer.sequential.{palette}_4')  # Accessing the _4 version of each palette
        # Append the colors to the colors list
        normalized_colors = (np.array(cmap.colors) / 255.0).tolist()
        colors.extend(normalized_colors[::-1])
    
    # Create a new colormap from the combined colors
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, len(colors))
    return custom_cmap

def main():
    # Specify the palettes to combine
    palettes = ['Reds', 'Blues']
    # Create the custom colormap
    custom_cmap = create_custom_colormap(palettes)

    # Generate 1000 random points
    np.random.seed(0)  # For reproducibility
    points = np.random.rand(100, 3)

    # Create a PyVista point cloud object
    cloud = pv.PolyData(points)

    # Create a Plotter object
    plotter = pv.Plotter()

    # Set the render_points_as_spheres option
    pv.global_theme.render_points_as_spheres = True

    # Add the point cloud to the plotter, coloring by z-coordinate and using the custom colormap
    plotter.add_mesh(cloud, scalars=points[:, 2], cmap=custom_cmap, render_points_as_spheres=True)

    # Reset the camera to show the full object
    plotter.reset_camera()

    # Show the plot
    plotter.show()
