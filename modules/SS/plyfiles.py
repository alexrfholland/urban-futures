import pyvista as pv

# Load the PLY and OBJ files
ply_data = pv.read('data/tree-cropped_skeleton.ply')
obj_data = pv.read('data/tree-cropped_branches.obj')

print(f'obj data is {obj_data}')
# Get the number of points and cells from the PLY and OBJ files
n_points_ply = ply_data.n_points
n_cells_ply = ply_data.n_cells
n_points_obj = obj_data.n_points
n_cells_obj = obj_data.n_cells

# Print the counts
print(f'PLY file: {n_points_ply} points, {n_cells_ply} cells')
print(f'OBJ file: {n_points_obj} points, {n_cells_obj} cells')

# Optionally, compare the counts
# This will just print whether the counts are equal or not
# Replace with your own logic as needed
print(f'Number of points equal: {n_points_ply == n_points_obj}')
print(f'Number of cells equal: {n_cells_ply == n_cells_obj}')

# Create a plotter to visualize the data
plotter = pv.Plotter()

# Add the PLY and OBJ data to the plotter
plotter.add_mesh(ply_data, color='blue', label='PLY Data')
plotter.add_mesh(obj_data, color='red', label='OBJ Data')

# Show the legend and the plot
plotter.add_legend()
plotter.show()
