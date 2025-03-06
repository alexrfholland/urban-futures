import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def query_and_visualize(polyData):

    print(f'block choses in {polyData.point_data}')

    
    # Initialize the plotter
    plotter = pv.Plotter()
    # Compile a list of all attributes in the PolyData
    all_attributes = polyData.point_data.keys()

    # First Question: Which field to color map?
    print(f"Which field would you like to use for the color map? Options: {list(all_attributes)}")
    selected_field = input("Enter field name: ")

    # Get the list of unique attributes in the selected field
    unique_attributes = np.unique(polyData.point_data[selected_field])
    print(f'unique attributes are: {unique_attributes}')

    # Generate a color map based on unique attributes
    cmap = plt.cm.get_cmap("rainbow")
    attribute_colors = {attr: (np.array(cmap(i / (len(unique_attributes) - 1))[:3]) * 255).astype(int) for i, attr in enumerate(unique_attributes)}

    # Map the colors to the selected field in the PolyData
    polyData.point_data['RGB'] = [attribute_colors[attr] for attr in polyData.point_data[selected_field]]

    # Add the PolyData to the plotter
    plotter.add_mesh(polyData, scalars='RGB', rgb=True, point_size=5.0, render_points_as_spheres=True)

    # Show the plot
    plotter.show()

print(f"Which site?")
site = input("Enter field name: ")
vtkBlocks = pv.read(f'data/{site}/flattened-{site}.vtk')

query_and_visualize(vtkBlocks)
