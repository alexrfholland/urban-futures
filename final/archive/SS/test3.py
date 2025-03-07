import trimesh
import numpy as np
import pyvista as pv

# Load the mesh
mesh_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/experimental/B4-22/B4-22-exportedFLAT.glb"

print(f"Loading mesh from: {mesh_path}")
scene = trimesh.load(mesh_path)
print(f"Scene loaded successfully")

print(f"Scene contains {len(scene.geometry)} meshes")

# Create a PyVista plotter
plotter = pv.Plotter()

# Set up lighting
print("Setting up lighting...")
plotter.enable_eye_dome_lighting()
plotter.enable_ssao()
custom_light = pv.Light(position=(0, 0, 1), focal_point=(0, 0, 0), color='white', intensity=0.8)
plotter.add_light(custom_light)
print("Lighting setup complete.")

# Process each mesh in the scene
for name, mesh in scene.geometry.items():
    print(f"Processing mesh: {name}")
    
    # Convert trimesh to PyVista mesh
    vertices = mesh.vertices
    faces = mesh.faces
    pv_mesh = pv.PolyData(vertices, faces)
    
    # Handle colors/textures
    if hasattr(mesh.visual, 'vertex_colors'):
        pv_mesh.point_data['colors'] = mesh.visual.vertex_colors[:, :3]
        plotter.add_mesh(pv_mesh, scalars='colors', rgb=True)
    elif hasattr(mesh.visual, 'material'):
        # For textured meshes, you might need to handle textures differently
        # This is a simplified approach
        diffuse_color = mesh.visual.material.diffuse
        plotter.add_mesh(pv_mesh, color=diffuse_color)
    else:
        plotter.add_mesh(pv_mesh)

# Optional: Add a bounding box to visualize the scene extent
bounds = scene.bounds
bounding_box = pv.Box(bounds=bounds)
plotter.add_mesh(bounding_box, style='wireframe', color='black', opacity=0.5)

# Set up the camera
center = scene.centroid
max_dim = np.max(scene.extents)
plotter.camera_position = [
    (center[0], center[1], center[2] + max_dim),  # Camera position
    center,  # Focal point
    (0, 1, 0)  # View up direction
]

print("Displaying the scene...")
plotter.show()