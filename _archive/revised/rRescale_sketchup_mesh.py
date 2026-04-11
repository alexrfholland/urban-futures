import trimesh
import numpy as np
from scipy.spatial import KDTree
import pyvista as pv
import collada  # For loading and exporting COLLADA files

"""def plot_entire_mesh(file_path):
    print("Loading COLLADA file...")
    scene = trimesh.load(file_path, force='mesh')

    print("Initializing PyVista plotter...")
    plotter = pv.Plotter()

    print("Processing mesh data...")
    if isinstance(scene, trimesh.Trimesh):
        print("Processing single mesh...")
        vertices = scene.vertices
        faces = scene.faces
        face_array = np.hstack([np.full((faces.shape[0], 1), 3), faces])
        pyvista_mesh = pv.PolyData(vertices, face_array.flatten())
        plotter.add_mesh(pyvista_mesh, color='lightblue')
    else:
        print("Processing multiple geometries...")
        for i, geometry in enumerate(scene.geometry.values()):
            print(f"Processing geometry {i+1}...")
            vertices = geometry.vertices
            faces = geometry.faces
            face_array = np.hstack([np.full((faces.shape[0], 1), 3), faces])
            pyvista_mesh = pv.PolyData(vertices, face_array.flatten())
            plotter.add_mesh(pyvista_mesh, color='lightblue')

    print("Displaying the plot...")
    plotter.show()

if __name__ == "__main__":
    file_path = "data/revised/experimental/DevelopmentActivityModel-trimmed.dae"
    print(f"Processing file: {file_path}")
    plot_entire_mesh(file_path)

"""
"""
import trimesh
import pyvista as pv
import numpy as np

def inspect_scene(file_path, max_geometries=1000):
    # Load the COLLADA file
    print("Loading COLLADA file...")
    scene = trimesh.load(file_path)
    
    if isinstance(scene, trimesh.Trimesh):
        print("Processing single mesh...")
        vertices = scene.vertices
        faces = scene.faces
        face_array = np.hstack([np.full((faces.shape[0], 1), 3), faces])
        pyvista_mesh = pv.PolyData(vertices, face_array.flatten())
        plotter = pv.Plotter()
        plotter.add_mesh(pyvista_mesh, color='lightblue')
        plotter.show()
    
    else:
        # If it's a scene, print out the key properties
        print("Processing scene...")
        print(f"Number of geometries: {len(scene.geometry)}")
        print(f"Number of nodes: {len(scene.graph.nodes)}")

        # Optionally visualize the geometries
        plotter = pv.Plotter()
        
        # Plot only the first max_geometries geometries
        for i, geometry in enumerate(scene.geometry.values()):
            if i >= max_geometries:
                break  # Stop once we reach the limit
            vertices = geometry.vertices
            faces = geometry.faces
            face_array = np.hstack([np.full((faces.shape[0], 1), 3), faces])
            pyvista_mesh = pv.PolyData(vertices, face_array.flatten())
            plotter.add_mesh(pyvista_mesh, color='lightblue')

        print(f"Plotted {min(max_geometries, len(scene.geometry))} geometries.")
        plotter.show()

if __name__ == "__main__":
    file_path = "data/revised/experimental/DevelopmentActivityModel-trimmed.dae"
    inspect_scene(file_path)
"""
"""
import trimesh
import numpy as np
from scipy.spatial import KDTree
import pyvista as pv

def inspect_scene_and_search(file_path, search_radius):
    # Load the COLLADA file
    print(f"Loading COLLADA file from {file_path}...")
    scene = trimesh.load(file_path)
    
    if isinstance(scene, trimesh.Trimesh):
        print("Processing single mesh...")
        vertices = scene.vertices
        faces = scene.faces
        face_array = np.hstack([np.full((faces.shape[0], 1), 3), faces])
        pyvista_mesh = pv.PolyData(vertices, face_array.flatten())
        plotter = pv.Plotter()
        plotter.add_mesh(pyvista_mesh, color='lightblue')
        plotter.show()
    
    else:
        # If it's a scene, print out the key properties
        print("Processing scene...")
        print(f"Number of geometries: {len(scene.geometry)}")
        print(f"Number of nodes: {len(scene.graph.nodes)}")

        # Extract the centroids for all geometries
        valid_geometries = [geom for geom in scene.geometry.values() if hasattr(geom, 'centroid')]
        centroids = np.array([geom.centroid for geom in valid_geometries])

        print(f"Centroids extracted for {len(valid_geometries)} valid geometries.")
        
        # Print the first 10 centroids
        print("First 10 centroids:")
        print(centroids[:10])
        
        # Compute and print the bounding box of centroids
        min_bound = centroids.min(axis=0)
        max_bound = centroids.max(axis=0)
        print("Bounding box of centroids (min, max):")
        print(f"Min: {min_bound}")
        print(f"Max: {max_bound}")

        # Calculate and print the dimensions of the bounding box
        dimensions = max_bound - min_bound
        print(f"Bounding box dimensions (X, Y, Z):")
        print(f"X: {dimensions[0]}")
        print(f"Y: {dimensions[1]}")
        print(f"Z: {dimensions[2]}")

        # Compute and print the centroid of all centroids
        centroid_of_centroids = centroids.mean(axis=0)
        print("Centroid of all centroids:")
        print(centroid_of_centroids)
        
        # Use the centroid of centroids as the search point
        search_point = centroid_of_centroids[:2]  # Only x, y for 2D search

        # Build a KD-Tree for fast spatial search
        kdtree = KDTree(centroids[:, :2])  # Use x, y for 2D search
        
        print(f"Searching for geometries within {search_radius} units of centroid {search_point}...")
        indices = kdtree.query_ball_point(search_point, search_radius)
        
        if indices:
            print(f"Found {len(indices)} geometries within {search_radius} units of centroid {search_point}.")
        else:
            print(f"No geometries found within {search_radius} units of centroid {search_point}.")
        
        # Optionally visualize the geometries
        plotter = pv.Plotter()
        
        # Plot only the geometries within the search radius
        for i in indices:
            geometry = valid_geometries[i]
            vertices = geometry.vertices
            faces = geometry.faces
            face_array = np.hstack([np.full((faces.shape[0], 1), 3), faces])
            pyvista_mesh = pv.PolyData(vertices, face_array.flatten())
            plotter.add_mesh(pyvista_mesh, color='lightblue')

        print(f"Plotted {len(indices)} geometries.")
        plotter.show()

# Example usage
file_path = "data/revised/experimental/DevelopmentActivityModel-trimmed.dae"
inspect_scene_and_search(file_path, 10000)"""


import trimesh
import numpy as np

def analyze_and_scale_scene(file_path, scale_factor):
    """
    Load, analyze, and scale a 3D scene from a file by directly modifying the vertices.

    Args:
    file_path (str): Path to the input 3D file.
    scale_factor (float): Factor to scale the scene by.

    Returns:
    None
    """
    print(f"Loading trimesh from {file_path}...")

    # Load the scene
    scene = trimesh.load(file_path)
    
    # Get the bounding box and extents in original units before scaling
    bounding_box_original = scene.bounds
    extents_original = scene.extents
    dimensions_original = extents_original

    # Print bounding box, extents, and dimensions before scaling
    print("Before scaling (in original units):")
    print(f"Bounding box: {bounding_box_original}")
    print(f"Extents: {extents_original}")
    print(f"Dimensions (x, y, z): {dimensions_original[0]}, {dimensions_original[1]}, {dimensions_original[2]}")

    # Initialize centroid and bounding box checks
    valid_geometries = []
    
    if isinstance(scene, trimesh.Trimesh):
        print("Applying scale to a single mesh.")
        # Scale vertices for single mesh
        scene.vertices *= scale_factor
        valid_geometries.append(scene)

    else:
        print("Iterating over geometries...")
        # Iterate over geometries in the scene and scale their vertices
        for i, geom in enumerate(scene.geometry.values()):
            if isinstance(geom, trimesh.Trimesh):
                geom.vertices *= scale_factor
                valid_geometries.append(geom)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} geometries.")

    # After scaling, extract the centroids
    centroids = np.array([geom.centroid for geom in valid_geometries])

    print(f"Centroids extracted for {len(valid_geometries)} valid geometries.")
    
    # Print the first 10 centroids
    print("First 10 centroids (after scaling):")
    print(centroids[:10])
    
    # Compute and print the bounding box of centroids
    min_bound = centroids.min(axis=0)
    max_bound = centroids.max(axis=0)
    print("Bounding box of centroids (min, max) after scaling:")
    print(f"Min: {min_bound}")
    print(f"Max: {max_bound}")

    # Calculate and print the dimensions of the bounding box
    dimensions = max_bound - min_bound
    print(f"Bounding box dimensions (X, Y, Z) after scaling:")
    print(f"X: {dimensions[0]}")
    print(f"Y: {dimensions[1]}")
    print(f"Z: {dimensions[2]}")

    # Compute and print the centroid of all centroids
    centroid_of_centroids = centroids.mean(axis=0)
    print("Centroid of all centroids (after scaling):")
    print(centroid_of_centroids)

    # Check if the scaling is correct
    expected_dimensions = dimensions_original * scale_factor
    print("\nChecking scaling accuracy:")
    print(f"Expected dimensions (x, y, z): {expected_dimensions[0]:.6f}, {expected_dimensions[1]:.6f}, {expected_dimensions[2]:.6f}")
    print(f"Actual bounding box dimensions (x, y, z): {dimensions[0]:.6f}, {dimensions[1]:.6f}, {dimensions[2]:.6f}")

    # Calculate and print the difference
    difference = np.abs(expected_dimensions - dimensions)
    print(f"Difference (x, y, z): {difference[0]:.6f}, {difference[1]:.6f}, {difference[2]:.6f}")

    # Save the scaled scene as a GLB file
    output_file_path = file_path.replace('.dae', '-metric.glb')
    trimesh.exchange.export.export_mesh(scene, output_file_path, file_type='glb')
    print(f"\nScaled mesh exported successfully in GLB format to: {output_file_path}")

    # Load the exported GLB file and confirm dimensions
    exported_scene = trimesh.load(output_file_path)
    
    # Get the dimensions of the exported scene
    exported_dimensions = exported_scene.extents
    
    print("\nComparing exported GLB dimensions with expected dimensions:")
    print(f"Expected dimensions (x, y, z): {expected_dimensions[0]:.6f}, {expected_dimensions[1]:.6f}, {expected_dimensions[2]:.6f}")
    print(f"Exported dimensions (x, y, z): {exported_dimensions[0]:.6f}, {exported_dimensions[1]:.6f}, {exported_dimensions[2]:.6f}")
    
    # Calculate and print the difference
    export_difference = np.abs(expected_dimensions - exported_dimensions)
    print(f"Difference (x, y, z): {export_difference[0]:.6f}, {export_difference[1]:.6f}, {export_difference[2]:.6f}")

# Example usage
file_path = "data/revised/experimental/DevelopmentActivityModel-trimmed.dae"
analyze_and_scale_scene(file_path, 0.0254)  # Scale factor from inches to meters

