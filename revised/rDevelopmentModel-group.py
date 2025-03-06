import trimesh
import numpy as np
import pyvista as pv
from random import randint

def load_glb(glb_file_path):
    """
    Load and analyze the GLB file to extract the scene.
    """
    print(f"Loading GLB model from: {glb_file_path}")
    scene = trimesh.load(glb_file_path)
    
    if isinstance(scene, trimesh.Scene):
        print(f"GLB file contains a scene with {len(scene.geometry)} geometries/groups.")
    else:
        print(f"GLB file contains a single mesh with {len(scene.vertices)} vertices and {len(scene.faces)} faces.")
    
    return scene


def batch_merge(meshes):
    """
    Batch merge multiple meshes to minimize the number of copy operations.
    """
    # Collect all vertices and faces
    all_vertices = []
    all_faces = []
    current_offset = 0

    for mesh in meshes:
        all_vertices.append(mesh.vertices)
        # Add the current offset to the faces so they refer to the correct vertex indices
        all_faces.append(mesh.faces + current_offset)
        current_offset += len(mesh.vertices)
    
    # Concatenate the vertices and faces to form a single large mesh
    merged_vertices = np.vstack(all_vertices)
    merged_faces = np.vstack(all_faces)
    
    return trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces)


def merge_connected_components(scene):
    """
    Merge connected components of the scene geometry in batches for efficiency.
    """
    merged_buildings = []

    for name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            # Use trimesh's split method to identify connected components
            components = geometry.split(only_watertight=False)
            print(f"Merging {len(components)} components for {name}.")
            
            # Merge the components in batches
            merged_mesh = batch_merge(components)
            merged_buildings.append(merged_mesh)

    return merged_buildings


def random_color():
    """
    Generate a random color for each building.
    """
    return [randint(0, 255) / 255.0 for _ in range(3)]


def plot_merged_buildings(buildings):
    """
    Plot the merged buildings in PyVista, each with a different color.
    """
    plotter = pv.Plotter()

    for building in buildings:
        # Convert Trimesh geometry to PyVista mesh
        pv_mesh = pv.PolyData(building.vertices, np.hstack([np.full((len(building.faces), 1), 3), building.faces]))

        # Generate a random color for the building
        color = random_color()
        
        # Add the building mesh to the plotter with a random color
        plotter.add_mesh(pv_mesh, color=color, opacity=1.0)

    # Display the plot
    plotter.show()


def process_city_mesh(glb_file_path):
    """
    Full process:
    1. Load GLB file.
    2. Merge connected components for each building.
    3. Plot the merged buildings in PyVista with different colors.
    """
    # Step 1: Load the GLB file and extract the scene
    scene = load_glb(glb_file_path)
    
    # Step 2: Merge connected components into individual buildings
    merged_buildings = merge_connected_components(scene)
    
    # Step 3: Plot the merged buildings in PyVista
    plot_merged_buildings(merged_buildings)


# Example usage
glb_file_path = 'data/revised/experimental/DevelopmentActivityModel.glb'
process_city_mesh(glb_file_path)
