import trimesh
import numpy as np
import pyvista as pv
import os

# Load the mesh with double precision
mesh_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/obj/B4-22/B4-22.obj"
print(f"Loading mesh from: {mesh_path}")
scene = trimesh.load(mesh_path, force='mesh', process=False, use_embree=True, dtype=np.float64)
print(f"Scene loaded successfully")

# Function to center a mesh
def center_mesh(mesh):
    center = mesh.centroid
    mesh.apply_translation(-center)
    return center

# Center the entire scene
if isinstance(scene, trimesh.Scene):
    print("Scene contains multiple objects")
    total_center = np.zeros(3)
    mesh_count = 0
    for geometry in scene.geometry.values():
        if isinstance(geometry, trimesh.Trimesh):
            total_center += center_mesh(geometry)
            mesh_count += 1
    if mesh_count > 0:
        average_center = total_center / mesh_count
        print(f"Scene centered. Average centroid: {average_center}")
    
    # Flatten the scene into a single mesh
    mesh = scene.dump(concatenate=True)
    print(f"Flattened mesh created with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
else:
    print("Scene contains a single mesh")
    mesh = scene
    center = center_mesh(mesh)
    print(f"Mesh centered. Original centroid: {center}")

print(f"Mesh properties:")
print(f"  Vertices: {len(mesh.vertices)}")
print(f"  Faces: {len(mesh.faces)}")
print(f"  Has texture: {hasattr(mesh.visual, 'material') and mesh.visual.material.image is not None}")
if hasattr(mesh.visual, 'material') and mesh.visual.material.image is not None:
    print(f"  Texture shape: {mesh.visual.material.image.shape}")

print(f'Voxelizing mesh...')
# Voxelize the loaded mesh
voxel_size = .5
voxel_grid = mesh.voxelized(voxel_size)

print(f"Voxelized mesh created with shape: {voxel_grid.shape}")

# Get voxel centers
voxel_centers = voxel_grid.points

# Transform the color information to vertex colors if it's not already
print("Checking mesh visual properties:")
print(f"  Type of mesh.visual: {type(mesh.visual)}")
print(f"  Available attributes: {dir(mesh.visual)}")

if not hasattr(mesh.visual, 'vertex_colors'):
    print("Mesh does not have vertex colors. Converting visual to color representation...")
    original_visual_type = type(mesh.visual)
    mesh.visual = mesh.visual.to_color()
    print(f"Conversion complete. Visual type changed from {original_visual_type} to {type(mesh.visual)}")
    if hasattr(mesh.visual, 'vertex_colors'):
        print(f"Vertex colors created. Shape: {mesh.visual.vertex_colors.shape}")
        print(f"Sample of vertex colors: {mesh.visual.vertex_colors[:5]}")
    else:
        print("Warning: Conversion did not create vertex colors as expected.")
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

# Add the point cloud to the plotter
plotter.add_mesh(point_cloud, scalars='colors', rgb=True, render_points_as_spheres=True, point_size=5)

# Show the plot
plotter.show()