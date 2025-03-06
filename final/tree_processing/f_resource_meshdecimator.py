import pyvista as pv
from scipy.spatial import cKDTree

def decimate_mesh_levels(mesh, decimation_levels=[0.9], plot=False, tree=None):
    """
    Decimates a mesh at different levels and returns a dictionary of decimated meshes.

    Parameters:
    mesh (pyvista.PolyData): The input mesh.
    decimation_levels (list of float): Decimation levels (default is [0.9]).
    plot (bool): Whether to plot the original and decimated meshes (default is False).
    tree (scipy.spatial.cKDTree, optional): Precomputed cKDTree for point matching. If not provided, it will be built.

    Returns:
    dict: A dictionary where keys are the decimation levels and values are the corresponding decimated meshes.
    """
    # Initialize dictionary to store decimated meshes
    decimated_meshes = {}

    # Triangulate the mesh only once
    tri_mesh = mesh.triangulate()
    print(f"Original mesh has {tri_mesh.n_cells} cells and {tri_mesh.n_points} points.")

    # Store the original mesh in the dictionary with key 'original'
    decimated_meshes['original'] = tri_mesh.copy()

    # Extract original points for KDTree (point_data)
    original_points = tri_mesh.points

    # Build cKDTree if it is not provided
    if tree is None:
        tree = cKDTree(original_points)
        print("KDTree built using original mesh points.")
    else:
        print("Using precomputed KDTree.")

    # Extract original resource data from point_data
    if 'resource' not in tri_mesh.point_data:
        raise KeyError("The input mesh does not contain 'resource' in point_data.")
    
    original_resource_data = tri_mesh.point_data['resource']
    print(f"Original mesh has 'resource' data with {len(original_resource_data)} entries.")

    # Decimate the mesh for each level
    for level in decimation_levels:
        print(f"\nDecimating mesh to {level * 100}% of original complexity...")
        # Apply decimation
        decimated_mesh = tri_mesh.decimate_pro(level, preserve_topology=True)
        print(f"Decimated mesh has {decimated_mesh.n_cells} cells and {decimated_mesh.n_points} points.")

        # Extract decimated points
        decimated_points = decimated_mesh.points

        # Find nearest neighbors in the original mesh for each decimated point
        distances, indices = tree.query(decimated_points)
        print(f"Mapped {len(decimated_points)} decimated points to original points.")

        # Map resource data from original points to decimated points
        mapped_resource_data = original_resource_data[indices]
        decimated_mesh.point_data['resource'] = mapped_resource_data
        print(f"'resource' data mapped to decimated mesh point_data.")

        # Transfer other attributes
        for attr in tri_mesh.point_data.keys():
            if attr != 'resource':
                original_attr_data = tri_mesh.point_data[attr]
                mapped_attr_data = original_attr_data[indices]
                decimated_mesh.point_data[attr] = mapped_attr_data
                print(f"'{attr}' data mapped to decimated mesh point_data.")

        # Store the decimated mesh in the dictionary with the decimation level as the key
        decimated_meshes[level] = decimated_mesh.copy()
        print(f"Decimated mesh at {level * 100}% reduction stored.")

    # Plot if requested
    if plot:
        print("\nGenerating plot...")
        num_plots = len(decimation_levels) + 1
        plotter = pv.Plotter(shape=(1, num_plots), border=True)
        
        # Plot the original mesh
        plotter.subplot(0, 0)
        plotter.add_mesh(decimated_meshes['original'], scalars='resource', cmap='viridis', show_edges=False)
        plotter.add_text("Original Mesh", font_size=10)
        
        # Plot each decimated mesh
        for i, level in enumerate(decimation_levels):
            plotter.subplot(0, i + 1)
            mesh_to_plot = decimated_meshes[level]
            plotter.add_mesh(mesh_to_plot, scalars='resource', cmap='viridis', show_edges=False)
            plotter.add_text(f"Decimated: {level * 100}%", font_size=10)
        
        # Link views for synchronized zooming and panning
        plotter.link_views()
        
        # Display the plot
        plotter.show()
        print("Plot displayed.")

    return decimated_meshes

if __name__ == "__main__":
    # Path to the input VTK file
    input_vtk_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/treeMeshes/(True, 'large', 'reserve-tree', True, 11).vtk"
    
    # Read the mesh
    print(f"Reading mesh from {input_vtk_path}...")
    mesh = pv.read(input_vtk_path)
    print("Mesh successfully read.")

    # Perform decimation
    decimated_meshes = decimate_mesh_levels(mesh, decimation_levels=[0.5, 0.9], plot=True)
    print("Decimation process completed.")
