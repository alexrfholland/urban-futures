import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

obj_file_path = 'data/revised/experimental/CoM.obj'
#sketchup_file_path = 'data/revised/experimental/DevelopmentActivityModel.skp'


import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_and_process_mesh(obj_file_path, center_easting, center_northing, radius):
    print("Loading mesh...")
    mesh = trimesh.load(obj_file_path)
    
    print("\nMesh Information:")
    print(f"Type of loaded object: {type(mesh)}")
    
    if isinstance(mesh, trimesh.Scene):
        print("The OBJ file contains a scene with multiple meshes.")
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
            for g in mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ])
    
    print("Original mesh:")
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Number of faces: {len(mesh.faces)}")
    print(f"Bounding box: {mesh.bounds}")
    
    # Scale down the mesh from cm to m
    scale_factor = 0.1
    mesh.apply_scale(scale_factor)
    
    print("\nScaled mesh (cm to m):")
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Number of faces: {len(mesh.faces)}")
    print(f"Bounding box: {mesh.bounds}")
    
    center_point = np.array([center_easting, center_northing, 0])
    
    print(f"\nCenter point: {center_point}")
    print(f"Radius: {radius}")
    
    print("\nProcessing mesh...")
    distances = np.linalg.norm(mesh.vertices[:, :2] - center_point[:2], axis=1)
    
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    print(f"Minimum distance from center: {min_distance}")
    print(f"Maximum distance from center: {max_distance}")
    
    mask = distances <= radius
    vertices_within_radius = mesh.vertices[mask]
    
    if len(vertices_within_radius) == 0:
        print("\nWARNING: No vertices found within the specified radius.")
        print("Possible reasons:")
        print("1. The center point might be far from the mesh.")
        print("2. The radius might be too small.")
        print("3. The mesh and center point might use different coordinate systems.")
        return None
    
    face_mask = mask[mesh.faces].all(axis=1)
    submesh = trimesh.Trimesh(vertices=vertices_within_radius, faces=mesh.faces[face_mask])
    
    print(f"\nExtracted submesh has {len(submesh.vertices)} vertices and {len(submesh.faces)} faces")
    
    return submesh

def plot_mesh(mesh, center_easting, center_northing, radius):
    if mesh is None or len(mesh.vertices) == 0:
        print("No mesh to plot.")
        return
    
    print("\nPreparing to plot...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    print("Plotting mesh...")
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                    triangles=mesh.faces, cmap='viridis', alpha=0.7)
    
    ax.scatter([center_easting], [center_northing], [0], color='red', s=100, label='Center')
    
    circle_points = 100
    theta = np.linspace(0, 2*np.pi, circle_points)
    x = center_easting + radius * np.cos(theta)
    y = center_northing + radius * np.sin(theta)
    z = np.zeros(circle_points)
    ax.plot(x, y, z, color='red', label=f'{radius}m Radius')
    
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Elevation (m)')
    ax.legend()
    
    plt.title(f'Mesh within {radius}m radius')
    print("Displaying plot...")
    plt.show()

def main():
    center_easting, center_northing = 320266.26, 5815638.74  # meters
    radius = 500  # meters
    
    submesh = load_and_process_mesh(obj_file_path, center_easting, center_northing, radius)
    if submesh is not None:
        plot_mesh(submesh, center_easting, center_northing, radius)
    else:
        print("Failed to create submesh.")

if __name__ == "__main__":
    main()
