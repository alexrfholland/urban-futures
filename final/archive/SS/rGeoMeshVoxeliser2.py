import trimesh
import numpy as np
import pyvista as pv
import os

# Load the mesh
mesh_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/experimental/B4-22/B4-22-exportedFLAT.glb"

print(f"Loading mesh from: {mesh_path}")
scene = trimesh.load(mesh_path)
print(f"Scene loaded successfully")

# Flatten the scene into a single mesh
if isinstance(scene, trimesh.Scene):
    print("Scene contains multiple objects, flattening into a single mesh")
    mesh = scene.dump(concatenate=True)
    print(f"Flattened mesh created with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
else:
    print("Scene contains a single mesh, no flattening needed")
    mesh = scene

print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")

print(f'Voxelizing mesh...')
# Voxelize the loaded mesh
voxel_size = .5
voxel_grid = mesh.voxelized(voxel_size)

print(f"Voxelized mesh created with shape: {voxel_grid.shape}")

# Get voxel centers
voxel_centers = voxel_grid.points

print("Sampling colors for voxel centers...")
if hasattr(mesh.visual, 'texture'):
    # If the mesh has a texture
    print("Mesh has a texture. Sampling colors from texture.")
    texture = mesh.visual.texture
    
    # Find the closest points on the mesh for each voxel center
    closest_points, distances, triangle_id = trimesh.proximity.closest_point(mesh, voxel_centers)
    
    # Get UV coordinates for the closest points
    uv = trimesh.visual.texture.unmerge_faces(mesh.visual.uv)[triangle_id]
    
    # Compute barycentric coordinates
    barycentric = trimesh.triangles.points_to_barycentric(
        mesh.triangles[triangle_id],
        closest_points)
    
    # Interpolate UV coordinates
    interpolated_uv = (uv * barycentric[:, np.newaxis]).sum(axis=1)
    
    # Sample colors from the texture
    voxel_colors = trimesh.visual.color.uv_to_color(interpolated_uv, texture)[:, :3]

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

    # Create a plotter
    plotter = pv.Plotter()

    # Add the point cloud to the plotter
    plotter.add_mesh(point_cloud, scalars='colors', rgb=True, render_points_as_spheres=True, point_size=5)

    # Show the plot
    plotter.show()
else:
    raise ValueError("Mesh does not have a texture. Cannot proceed with color sampling.")