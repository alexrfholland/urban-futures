import pyvista as pv
from pyproj import Transformer
import xml.etree.ElementTree as ET
import re
import trimesh
import numpy as np

def load_and_analyze_glb(glb_file_path):
    print(f"Loading and analyzing GLB model: {glb_file_path}")
    scene = trimesh.load(glb_file_path)
    
    if isinstance(scene, trimesh.Scene):
        print(f"The GLB file contains a scene with {len(scene.geometry)} geometries/groups:")
        for name, geometry in scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                bounding_box = geometry.bounds
                dimensions = bounding_box[1] - bounding_box[0]
                print(f"  {name}: Mesh with {len(geometry.vertices)} vertices and {len(geometry.faces)} faces")
                print(f"    Bounding Box: {bounding_box}")
                print(f"    Dimensions: {dimensions}")
            elif isinstance(geometry, trimesh.path.Path3D):
                print(f"  {name}: Path with {len(geometry.entities)} entities")
            else:
                print(f"  {name}: Other geometry type ({type(geometry)})")
    else:
        bounding_box = scene.bounds
        dimensions = bounding_box[1] - bounding_box[0]
        print("The GLB file contains a single mesh:")
        print(f"  Vertices: {len(scene.vertices)}")
        print(f"  Faces: {len(scene.faces)}")
        print(f"  Bounding Box: {bounding_box}")
        print(f"  Dimensions: {dimensions}")
    
    return scene

# Function to compute and print overall scene bounding box, dimensions, and midpoint
def print_scene_details(scene):
    mesh_bbox = scene.bounds
    mesh_dimensions = np.array([mesh_bbox[1][0] - mesh_bbox[0][0], mesh_bbox[1][2] - mesh_bbox[0][2]])  # X and Z
    mesh_midpoint = (mesh_bbox[0] + mesh_bbox[1]) / 2

    print("Overall Mesh Bounding Box:", mesh_bbox)
    print("Overall Mesh Dimensions:", mesh_dimensions)
    print("Overall Mesh Midpoint:", mesh_midpoint)

# Define the GDA Zone 55 origin in mesh space (0, 0, 0 in mesh corresponds to this in GDA)
gda_origin = np.array([323500.00411082, 5817500.00049888])

# Function to translate GDA Zone 55 coordinates to mesh space
def gda_to_mesh(gda_coords, gda_origin):
    mesh_x = gda_coords[0] - gda_origin[0]
    mesh_z = gda_origin[1] - gda_coords[1]  # GDA northing decreases as mesh z increases (north-south inversion)
    
    print(f"Converting GDA coordinates {gda_coords} to mesh coordinates...")
    print(f"GDA origin: {gda_origin}")
    print(f"Calculated mesh coordinates: [{mesh_x}, {mesh_z}]")
    
    # Return as a 3D vector (adding Y as 0 for flat ground)
    return np.array([mesh_x, 0, mesh_z])

# Function to find all vertices and corresponding faces within 1000m radius of a given mesh space point
def find_meshes_within_radius(scene, target_point, radius):
    meshes_in_range = []
    for name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            # Calculate distances of vertices from the target point in XZ plane
            distances = np.linalg.norm(geometry.vertices[:, [0, 2]] - target_point[[0, 2]], axis=1)
            within_radius_mask = distances <= radius
            if np.any(within_radius_mask):
                # Keep only the vertices within the radius
                vertices_in_range = geometry.vertices[within_radius_mask]
                
                # Find corresponding faces that use the vertices within the radius
                vertex_indices = np.nonzero(within_radius_mask)[0]
                faces_in_range = []
                
                for face in geometry.faces:
                    if all(v in vertex_indices for v in face):
                        faces_in_range.append(face)
                
                if len(faces_in_range) > 0:
                    # Calculate center point and bounding box for the filtered vertices
                    filtered_bbox = np.array([np.min(vertices_in_range, axis=0), np.max(vertices_in_range, axis=0)])
                    center_point = np.mean(filtered_bbox, axis=0)

                    print(f"  Mesh {name} center point: {center_point}")
                    print(f"  Mesh {name} bounding box: {filtered_bbox}")
                    
                    meshes_in_range.append((name, vertices_in_range, np.array(faces_in_range), filtered_bbox, center_point))
    
    return meshes_in_range

# Function to plot the filtered meshes in red and all others in grey with opacity
def plot_filtered_meshes(scene, meshes_within_1000m):
    plotter = pv.Plotter()

    # Add all meshes in grey with 0.3 opacity
    for name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            print(f"Plotting full scene mesh: {name} in grey.")
            try:
                # Create a PyVista mesh for all scene geometry
                pv_mesh = pv.PolyData(geometry.vertices, np.hstack([np.full((len(geometry.faces), 1), 3), geometry.faces]))
                plotter.add_mesh(pv_mesh, color='grey', opacity=0.3)
            except Exception as e:
                print(f"Error plotting full scene mesh {name}: {str(e)}")
    
    # Plot the filtered meshes (within 1000m radius) in red
    for name, vertices, faces, filtered_bbox, center_point in meshes_within_1000m:
        print(f"Plotting filtered mesh: {name} with {len(vertices)} vertices and {len(faces)} faces in red.")
        
        try:
            # Create a PyVista mesh using only the filtered vertices and faces
            pv_mesh = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]))
            
            # Add the filtered mesh to the PyVista plotter
            plotter.add_mesh(pv_mesh, color='red', opacity=1.0)
        except Exception as e:
            print(f"Error plotting filtered mesh {name}: {str(e)}")

    plotter.show()

# Load and analyze GLB file
glb_file_path = 'data/revised/experimental/DevelopmentActivityModel.glb'
glb_file_path = 'data/revised/experimental/DevelopmentActivityModel-trimmed.glb'
glb_file_path = 

scene = load_and_analyze_glb(glb_file_path)

# Print overall scene details
print_scene_details(scene)

# Example GDA coordinates (point to check within 1000 meters of)
gda_target = np.array([320757.79029528715, 5814438.136253171])

# Convert the target GDA coordinates to mesh space
mesh_target = gda_to_mesh(gda_target, gda_origin)

# Find all vertices and faces within 1000 meters of the target point in mesh space
meshes_within_1000m = find_meshes_within_radius(scene, mesh_target, 1000)

# Display the result
total_vertices = sum(len(vertices) for _, vertices, _, _, _ in meshes_within_1000m)
print(f"Number of vertices found within 1000m radius: {total_vertices}")

# Plot the scene: filtered meshes in red, all other meshes in grey with opacity 0.3
plot_filtered_meshes(scene, meshes_within_1000m)
