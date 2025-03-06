import pyvista as pv
import numpy as np
import random


def generate_random_color():
    return list(np.random.choice(range(256), size=3)/255.0)

def visualize_point_cloud_with_roof_type(vtk_file_path: str) -> None:
    # Load the VTK file into a MultiBlock object
    multi_block = pv.read(vtk_file_path)
    
    # Initialize a Plotter object
    plotter = pv.Plotter()
    
    # Initialize a dictionary to store unique roof types and their corresponding colors
    roof_type_to_color = {}
    
    # Iterate through each block (point cloud) in the MultiBlock object
    for idx, block in enumerate(multi_block):
        print(f"Processing block {idx + 1}...")
        
        # Access the points and their "roof_type" attributes
        points = block.points
        roof_types = block.point_data['roof_type']
        
        # Get unique roof types in this block
        unique_roof_types = np.unique(roof_types)
        
        # For each unique roof type, if it's not in our dictionary, assign a random color to it
        for roof_type in unique_roof_types:
            if roof_type not in roof_type_to_color:
                roof_type_to_color[roof_type] = generate_random_color()
        
        # Create an array to store colors for each point
        point_colors = np.zeros((len(points), 3))
        
        # Assign colors to points based on their roof_type
        for i, roof_type in enumerate(roof_types):
            point_colors[i] = roof_type_to_color[roof_type]
        
        # Create a new PolyData object with the colors
        colored_point_cloud = pv.PolyData(points)
        colored_point_cloud['colors'] = point_colors
        
        # Add the colored point cloud to the plotter
        plotter.add_mesh(colored_point_cloud, color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)
    
    # Show the visualization
    plotter.show()

# Call the function
visualize_point_cloud_with_roof_type("test.vtm")

