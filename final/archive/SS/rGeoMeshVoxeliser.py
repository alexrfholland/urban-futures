import trimesh
import numpy as np
import pyvista as pv
import os


# Load the mesh
#mesh_path = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/experimental/B4-22/B4-22.ply'
#mesh_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/experimental/B4-22/B4-22-exportedFLAT.glb"
mesh_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/obj/B4-22/B4-22.obj"
#mesh_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/obj/B4-22/output/B4-22.glb"

print(f"Loading mesh from: {mesh_path}")
scene = trimesh.load(mesh_path)
print(f"Scene loaded successfully")


# Flatten the scene into a single mesh
if isinstance(scene, trimesh.Scene):
    print("Scene contains multiple objects, flattening into a single mesh")
    mesh = scene.dump(concatenate=True)
    print(f"Flattened mesh created with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")


    output_dir = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/obj/B4-22/output'
    os.makedirs(output_dir, exist_ok=True)

    """print("Exporting flattened mesh as GLB...")
    
    # Generate the output file path
    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    output_glb_path = os.path.join(output_dir, f"{base_name}_flattened.glb")
    # Export the mesh as GLB
    mesh.export(output_glb_path, file_type='glb')
    
    print(f"Mesh exported as GLB: {output_glb_path}")"""

else:
    print("Scene contains a single mesh, no flattening needed")
    mesh = scene

print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")

# Function to calculate min, max, and average vertex distances
def calculate_vertex_distances(mesh):
    edges = mesh.edges_unique
    vertices = mesh.vertices
    distances = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return np.min(distances), np.max(distances), np.mean(distances)

# Add subdivision option
subdivide = True  # Set this to False if you don't want to subdivide
if subdivide:
    print("Calculating initial vertex distances...")
    min_dist, max_dist, avg_dist = calculate_vertex_distances(mesh)
    print(f"Initial - Min: {min_dist:.4f}, Max: {max_dist:.4f}, Avg: {avg_dist:.4f}")

    print("Subdividing mesh...")
    original_vertex_count = len(mesh.vertices)
    for i in range(2):  # Subdivide twice
        mesh = mesh.subdivide()
        print(f"Subdivision {i+1} complete. Vertex count: {len(mesh.vertices)}")

    print(f"Subdivision complete. Vertex count increased from {original_vertex_count} to {len(mesh.vertices)}")

    print("Calculating final vertex distances...")
    min_dist, max_dist, avg_dist = calculate_vertex_distances(mesh)
    print(f"Final - Min: {min_dist:.4f}, Max: {max_dist:.4f}, Avg: {avg_dist:.4f}")

print(f'Voxelizing mesh...')
# Voxelize the loaded mesh
voxel_size = .25
voxel_grid = mesh.voxelized(voxel_size)

print(f"Voxelized mesh created with shape: {voxel_grid.shape}")

# Add superSize option
"""superSize = True  # Set this to False if you don't want to supersize

if superSize:
    print("Supersizing voxels...")
    original_shape = voxel_grid.shape
    
    # Create a new array with 8 times as many voxels (2x2x2)
    new_shape = (original_shape[0]*2, original_shape[1]*2, original_shape[2]*2)
    new_voxels = np.zeros(new_shape, dtype=bool)
    
    # Fill the new array
    new_voxels[::2, ::2, ::2] = voxel_grid.matrix
    new_voxels[1::2, ::2, ::2] = voxel_grid.matrix
    new_voxels[::2, 1::2, ::2] = voxel_grid.matrix
    new_voxels[::2, ::2, 1::2] = voxel_grid.matrix
    new_voxels[1::2, 1::2, ::2] = voxel_grid.matrix
    new_voxels[1::2, ::2, 1::2] = voxel_grid.matrix
    new_voxels[::2, 1::2, 1::2] = voxel_grid.matrix
    new_voxels[1::2, 1::2, 1::2] = voxel_grid.matrix
    
    # Create a new VoxelGrid with the supersized voxels
    voxel_grid = trimesh.voxel.VoxelGrid(new_voxels, voxel_grid.transform)
    
    print(f"Supersized voxelized mesh created with shape: {voxel_grid.shape}")"""

# Get voxel centers
voxel_centers = voxel_grid.points

# Transform the color information to vertex colors if it's not already
if not hasattr(mesh.visual, 'vertex_colors'):
    print("Mesh does not have vertex colors. Converting visual to color representation...")
    original_visual_type = type(mesh.visual)
    mesh.visual = mesh.visual.to_color()
else:
    print("Mesh already has vertex colors.")
    print(f"Vertex colors shape: {mesh.visual.vertex_colors.shape}")
    print(f"Sample of vertex colors: {mesh.visual.vertex_colors[:5]}")

# Get colors for each voxel center
_, closest_vertex_indices = trimesh.proximity.ProximityQuery(mesh).vertex(voxel_centers)
voxel_colors = mesh.visual.vertex_colors[closest_vertex_indices][:, :3]  # Use only RGB channels

print(f"Voxel colors shape: {voxel_colors.shape}")
print(f"Sample of voxel colors: {voxel_colors[:5]}")

# Create PyVista PolyData object for the point cloud
point_cloud = pv.PolyData(voxel_centers)
point_cloud.point_data['colors'] = voxel_colors

# Convert PyVista PolyData to trimesh Trimesh
points_trimesh = trimesh.Trimesh(vertices=point_cloud.points, 
                                 vertex_colors=voxel_colors)

# Set up lighting
print("Setting up lighting...")
plotter = pv.Plotter(lighting="light_kit")

# Enable Eye-Dome Lighting (EDL)
print("Enabling Eye-Dome Lighting (EDL)...")
plotter.enable_eye_dome_lighting()

# Enable Screen Space Ambient Occlusion (SSAO)
print("Enabling Screen Space Ambient Occlusion (SSAO)...")
plotter.enable_ssao()

# Create a custom light
custom_light = pv.Light(position=(0, 0, 1), focal_point=(0, 0, 0), color='white', intensity=0.8)
plotter.add_light(custom_light)

print("Lighting, EDL, and SSAO setup complete.")

# Create a plotter
plotter = pv.Plotter()

# Add the point cloud to the plotter
plotter.add_mesh(point_cloud, scalars='colors', rgb=True, render_points_as_spheres=True, point_size=5)

# Uncomment the next lines if you want to show the original mesh as well
# original_pv_mesh = pv.wrap(mesh)
# original_pv_mesh.point_data['colors'] = mesh.visual.vertex_colors[:, :3]
# plotter.add_mesh(original_pv_mesh, scalars='colors', rgb=True, opacity=0.5)

# Show the plot
plotter.show()