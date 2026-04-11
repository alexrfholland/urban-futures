import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

site = 'streetbuildi'

def query_and_visualize(vtkBlocks):
    # Initialize the plotter
    plotter = pv.Plotter()

    print(vtkBlocks.point_data)

    # Compile a unique list of all attributes across all blocks
    all_attributes = set()
    for block in vtkBlocks:
        all_attributes.update(block.point_data.keys())

    # First Question: Which block type to display?
    all_block_types = [block.point_data.get('blocktype') for block in vtkBlocks if 'blocktype' in block.point_data]
    unique_block_types = np.unique(np.concatenate(all_block_types))
    print(f"Which block type would you like to display? Options: {unique_block_types}")
    selected_block_type = input("Enter block type: ")

    # Filter blocks based on the selected block type
    filtered_blocks = [block for block in vtkBlocks if 'blocktype' in block.point_data and np.any(block.point_data['blocktype'] == selected_block_type)]

    # Second Question: Which field to color map?
    print(f"Which field would you like to use for the color map? Options: {list(all_attributes)}")
    selected_field = input("Enter field name: ")

    # Get the list of unique attributes in the selected field
    unique_attributes = np.unique(np.concatenate([block.point_data[selected_field] for block in filtered_blocks]))
    print(f'unique attributes are: {unique_attributes}')

    # Generate a color map based on unique attributes
    cmap = plt.cm.get_cmap("rainbow")
    attribute_colors = {attr: (np.array(cmap(i / (len(unique_attributes) - 1))[:3])*255).astype(int) for i, attr in enumerate(unique_attributes)}

    #plotter.add_mesh(block, scalars = 'material', cmap = 'rainbow', rgb=False, point_size=5.0, render_points_as_spheres=True)

        
    # Iterate through all blocks for visualization
    for block in vtkBlocks:
        if 'blocktype' in block.point_data and np.any(block.point_data['blocktype'] == selected_block_type):
            first_attribute = block.point_data[selected_field][0]  # Assuming all points in a block have the same attributes
            if first_attribute in attribute_colors:
                block.point_data['RGB'] = [attribute_colors[first_attribute]] * block.n_points

        # Add the block to the plotter
        #plotter.add_mesh(block, color=None, rgb=True)
        plotter.add_mesh(block, color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)


    # Show the plot
    plotter.show()

vtkBlocks = pv.read(f'data/{site}/{site}.vtm')

query_and_visualize(vtkBlocks)
