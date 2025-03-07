

"""site = "trimmed-parade"  # Replace with the actual site name

import xarray as xr
import numpy as np
import pyvista as pv
import json
import os
from scipy.spatial import KDTree

# User-defined parameters
voxel_size = 5  # Replace with actual voxel size

# Construct dataset path
dataset_path = f'data/revised/xarray_voxels_{site}_{voxel_size}.nc'

# Load the dataset
if not os.path.isfile(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")
ds = xr.open_dataset(dataset_path)
print("Dataset loaded successfully.")

# Function to convert voxel indices to centroids
def ijk_to_centroid(i, j, k, ds):
    bounds = ds.attrs['bounds']
    voxel_size = ds.attrs['voxel_size']
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x_centroid = xmin + (i + 0.5) * voxel_size
    y_centroid = ymin + (j + 0.5) * voxel_size
    z_centroid = zmin + (k + 0.5) * voxel_size
    return x_centroid, y_centroid, z_centroid

# Convert dataset to PyVista PolyData
def create_pyvista_polydata(ds):
    I = ds['voxel_I'].values
    J = ds['voxel_J'].values
    K = ds['voxel_K'].values

    x_coords, y_coords, z_coords = [], [], []
    for i, j, k in zip(I, J, K):
        x, y, z = ijk_to_centroid(i, j, k, ds)
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        
    # Combine coordinates into a single array
    points = np.column_stack((x_coords, y_coords, z_coords))

    # Create a PyVista PolyData object
    polydata = pv.PolyData(points)

    # Add voxel indices as point data
    polydata.point_data["voxel_I"] = I
    polydata.point_data["voxel_J"] = J
    polydata.point_data["voxel_K"] = K

    return polydata, points

# Create PolyData object from dataset
polydata, all_vertices = create_pyvista_polydata(ds)

# Create a KDTree with all vertices for quick nearest neighbor search
vertex_tree = KDTree(all_vertices)

# Initialize variables for storing picked nodes
source_nodes = []
target_nodes = []
current_mode = 'source'
num_sources = 0
num_targets = 0

# Function to get user input for number of nodes
def get_number_of_nodes(node_type):
    while True:
        try:
            num = int(input(f"Enter the number of {node_type} nodes: "))
            if num <= 0:
                print("Please enter a positive integer.")
                continue
            return num
        except ValueError:
            print("Invalid input. Please enter an integer.")

# Callback function for point picking
def callback(point, picker):
    global current_mode
    if picker.GetActor() is None:
        print("No mesh picked")
        return

    print(f"Raw picked point: {point}")

    # Get the picked position on the mesh surface
    picked_position = picker.GetPickPosition()
    print(f"Picked position on mesh: {picked_position}")

    # Find the nearest vertex
    distance, index = vertex_tree.query(picked_position)
    nearest_vertex = all_vertices[index]
    
    print(f"Nearest vertex: {nearest_vertex}")

    # Extract voxel indices corresponding to the nearest vertex
    voxel_i = polydata.point_data["voxel_I"][index]
    voxel_j = polydata.point_data["voxel_J"][index]
    voxel_k = polydata.point_data["voxel_K"][index]

    # Add the selected node to the corresponding list
    if current_mode == 'source':
        source_nodes.append({
            "centroid": list(nearest_vertex),
            "voxel_indices": [int(voxel_i), int(voxel_j), int(voxel_k)]
        })
        print(f"Source node added: Centroid={nearest_vertex}, Voxel Indices=({voxel_i}, {voxel_j}, {voxel_k})")
        if len(source_nodes) >= num_sources:
            current_mode = 'target'
            plotter.add_text(f"Source nodes selection complete. Now, pick {num_targets} target nodes.", position='upper_left', font_size=12)
    elif current_mode == 'target':
        target_nodes.append({
            "centroid": list(nearest_vertex),
            "voxel_indices": [int(voxel_i), int(voxel_j), int(voxel_k)]
        })
        print(f"Target node added: Centroid={nearest_vertex}, Voxel Indices=({voxel_i}, {voxel_j}, {voxel_k})")
        if len(target_nodes) >= num_targets:
            print("Target nodes selection complete.")

    # Add a text label at the picked point
    plotter.add_point_labels([picked_position], [f"Node {len(source_nodes) if current_mode == 'source' else len(target_nodes)}"], point_size=20, font_size=10)
    # Render to update the visualization
    plotter.render()

# Get number of source and target nodes
num_sources = get_number_of_nodes("source")
num_targets = get_number_of_nodes("target")

# Setup PyVista plotter
plotter = pv.Plotter()
plotter.add_mesh(polydata, color='white', point_size=5, render_points_as_spheres=True, opacity=0.5)
plotter.add_axes()
plotter.add_text(f"Click on the mesh to select {num_sources} source nodes", position='upper_left', font_size=12)

# Enable point picking
plotter.enable_point_picking(
    callback=callback,
    show_message=True,
    tolerance=0.01,
    use_picker=True,
    pickable_window=False,
    show_point=True,
    point_size=20,
    picker='point',
    font_size=20
)

# Non-blocking show, allowing the loop to continue
plotter.show(interactive_update=True)

# Wait until all nodes are selected
while len(source_nodes) < num_sources or len(target_nodes) < num_targets:
    # Keep updating the plotter until all nodes are selected
    plotter.update()

# Close the plotter window after all nodes are selected
plotter.close()

# Create a dictionary to store all selected nodes
movement_nodes = {
    "site": site,
    "state": "initialized",
    "source_nodes": source_nodes,
    "target_nodes": target_nodes
}

# Save the nodes to a JSON file
output_file = f"data/revised/{site}-{voxel_size}-movementnodes.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(movement_nodes, f, indent=4)

print(f"Source and target nodes saved to '{output_file}'.")
"""

import xarray as xr
import numpy as np
import pyvista as pv
import json
import os
from scipy.spatial import KDTree

def process_agent(site, agent, voxel_size=5):
    # Construct dataset path
    dataset_path = f'data/revised/xarray_voxels_{site}_{voxel_size}.nc'

    # Load the dataset
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")
    ds = xr.open_dataset(dataset_path)
    print(f"Dataset for site {site} and agent {agent} loaded successfully.")

    # Convert voxel indices to centroids
    def ijk_to_centroid(i, j, k, ds):
        bounds = ds.attrs['bounds']
        voxel_size = ds.attrs['voxel_size']
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        x_centroid = xmin + (i + 0.5) * voxel_size
        y_centroid = ymin + (j + 0.5) * voxel_size
        z_centroid = zmin + (k + 0.5) * voxel_size
        return x_centroid, y_centroid, z_centroid

    # Convert dataset to PyVista PolyData
    def create_pyvista_polydata(ds):
        I = ds['voxel_I'].values
        J = ds['voxel_J'].values
        K = ds['voxel_K'].values

        x_coords, y_coords, z_coords = [], [], []
        for i, j, k in zip(I, J, K):
            x, y, z = ijk_to_centroid(i, j, k, ds)
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
        
        points = np.column_stack((x_coords, y_coords, z_coords))

        polydata = pv.PolyData(points)
        polydata.point_data["voxel_I"] = I
        polydata.point_data["voxel_J"] = J
        polydata.point_data["voxel_K"] = K

        return polydata, points

    polydata, all_vertices = create_pyvista_polydata(ds)

    vertex_tree = KDTree(all_vertices)

    # Moved the get_number_of_nodes function above where it's called
    def get_number_of_nodes(node_type):
        while True:
            try:
                num = int(input(f"Enter the number of {node_type} nodes: "))
                if num <= 0:
                    print("Please enter a positive integer.")
                    continue
                return num
            except ValueError:
                print("Invalid input. Please enter an integer.")

    source_nodes = []
    target_nodes = []
    current_mode = 'source'
    num_sources = get_number_of_nodes("source")
    num_targets = get_number_of_nodes("target")

    def callback(point, picker):
        nonlocal current_mode
        if picker.GetActor() is None:
            print("No mesh picked")
            return

        picked_position = picker.GetPickPosition()
        distance, index = vertex_tree.query(picked_position)
        nearest_vertex = all_vertices[index]

        voxel_i = polydata.point_data["voxel_I"][index]
        voxel_j = polydata.point_data["voxel_J"][index]
        voxel_k = polydata.point_data["voxel_K"][index]

        if current_mode == 'source':
            source_nodes.append({
                "centroid": list(nearest_vertex),
                "voxel_indices": [int(voxel_i), int(voxel_j), int(voxel_k)]
            })
            print(f"Source node added: Centroid={nearest_vertex}, Voxel Indices=({voxel_i}, {voxel_j}, {voxel_k})")
            if len(source_nodes) >= num_sources:
                current_mode = 'target'
                plotter.add_text(f"Source nodes selection complete. Now, pick {num_targets} target nodes.", position='upper_left', font_size=12)
        elif current_mode == 'target':
            target_nodes.append({
                "centroid": list(nearest_vertex),
                "voxel_indices": [int(voxel_i), int(voxel_j), int(voxel_k)]
            })
            print(f"Target node added: Centroid={nearest_vertex}, Voxel Indices=({voxel_i}, {voxel_j}, {voxel_k})")
            if len(target_nodes) >= num_targets:
                print("Target nodes selection complete.")

        # Add a text label at the picked point
        plotter.add_point_labels([picked_position], [f"Node {len(source_nodes) if current_mode == 'source' else len(target_nodes)}"], point_size=20, font_size=10)
        plotter.render()  # Update the visualization


    plotter = pv.Plotter()
    plotter.add_mesh(polydata, color='white', point_size=5, render_points_as_spheres=True, opacity=0.5)
    plotter.add_axes()
    plotter.enable_point_picking(
        callback=callback,
        show_message=True,
        tolerance=0.01,
        use_picker=True,
        pickable_window=False,
        show_point=True,
        point_size=20,
        picker='point',
        font_size=20
    )


    plotter.show(interactive_update=True)

    while len(source_nodes) < num_sources or len(target_nodes) < num_targets:
        plotter.update()

    plotter.close()

    movement_nodes = {
        "site": site,
        "agent": agent,
        "state": "initialized",
        "source_nodes": source_nodes,
        "target_nodes": target_nodes
    }

    output_file = f"data/revised/{site}-{agent}-{voxel_size}-movementnodes.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(movement_nodes, f, indent=4)

    print(f"Source and target nodes for agent {agent} saved to '{output_file}'.")

    return movement_nodes

def getNodes(site, agents, voxel_size=5):
    movement_nodes = {}
    for agent in agents:
        movement_nodes[agent] = process_agent(site, agent, voxel_size)
    return movement_nodes

if __name__ == '__main__':
    getNodes('city', ['bird'], voxel_size=5)
