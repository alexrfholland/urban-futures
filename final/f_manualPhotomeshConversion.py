import trimesh
import numpy as np
import pyvista as pv
import os


def load_and_prepare_mesh(mesh_path):
    print(f"Loading mesh from: {mesh_path}")
    scene_or_mesh = trimesh.load(mesh_path)
    print("Scene loaded successfully")

    total_geometries = len(scene_or_mesh.geometry)
    print(f"Total number of geometries: {total_geometries}")

    processed_geometries = {}
    # Create a PyVista MultiBlock to hold all the meshes
    multiblock = pv.MultiBlock()

    # Iterate through each geometry in the scene
    for idx, (name, mesh) in enumerate(scene_or_mesh.geometry.items()):
        """if idx >= 1000:
            print(f"Reached 1000 geometries, stopping further processing.")
            break  # Stop the loop after processing 1000 geometries"""

        if idx % 100 == 0:
            print(f"Processing geometry {idx+1}/{total_geometries}: {name}")

        # Perform subdivision and visual update on each mesh
        pv_mesh = process_mesh(mesh)

        multiblock.append(pv_mesh)

        # Store the processed mesh for reference
        processed_geometries[name] = mesh

    print (f'all meshes collected, flattening multiblock')
    # Flatten the MultiBlock into a single mesh
    combined_mesh = multiblock.combine()

    print(f'flattened multiblock!')

    print(f'converting from usntructured grid to polydata...')
    combined_mesh = combined_mesh.extract_surface()
    print(f'converted!')

    output_path = 'data/revised/obj/C4-12_combined.vtk'
    pv.save_meshio(output_path, combined_mesh)

    #combined_mesh.plot(rgb=True)

    return combined_mesh

def process_mesh(mesh):
    # Subdivide mesh and update vertex colors
    for i in range(2):  # Subdivide twice
        mesh = mesh.subdivide()

    mesh.visual = mesh.visual.to_color()

    # Wrap the trimesh object with PyVista's PolyData
    vertices = mesh.vertices
    faces = np.hstack([[3] + list(face) for face in mesh.faces])  # Convert to PyVista face format
    
    # Create a PolyData mesh
    pv_mesh = pv.PolyData(vertices, faces)

    # Convert colors to float32 and normalize to [0, 1]
    colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    pv_mesh.point_data['colors'] = colors  # Assign colors as point data

    return pv_mesh


def voxelize_mesh(mesh, voxel_size):
    
    


    
    print(f'Voxelizing mesh with voxel size {voxel_size}...')
    voxel_grid = mesh.voxelized(voxel_size)
    print(f"Voxelized mesh created with shape: {voxel_grid.shape}")

    voxel_centers = voxel_grid.points

    print("Assigning colors and normals to voxels...")
    proximity = trimesh.proximity.ProximityQuery(mesh)
    _, closest_vertex_indices = proximity.vertex(voxel_centers)
    
    # Get colors from closest vertices
    voxel_colors = mesh.visual.vertex_colors[closest_vertex_indices][:, :3]

    # Get original normals from closest vertices
    orig_normals = mesh.vertex_normals[closest_vertex_indices]

    print(f"Voxel colors shape: {voxel_colors.shape}")
    print(f"Sample of voxel colors: {voxel_colors[:5]}")
    print(f"Original normals shape: {orig_normals.shape}")
    print(f"Sample of original normals: {orig_normals[:5]}")

    # Create PyVista PolyData object for the point cloud
    point_cloud = pv.PolyData(voxel_centers)
    point_cloud.point_data['colors'] = voxel_colors
    point_cloud.point_data['orig_normal_x'] = orig_normals[:, 0]
    point_cloud.point_data['orig_normal_y'] = orig_normals[:, 1]
    point_cloud.point_data['orig_normal_z'] = orig_normals[:, 2]

    # Try to calculate new normals for the voxelized point cloud
    try:
        print("Calculating new normals for the voxelized point cloud...")
        point_cloud.compute_normals(inplace=True)

        # Rename the computed normals
        point_cloud.point_data['new_normal_x'] = point_cloud.point_data.pop('Normals')[:, 0]
        point_cloud.point_data['new_normal_y'] = point_cloud.point_data.pop('Normals')[:, 1]
        point_cloud.point_data['new_normal_z'] = point_cloud.point_data.pop('Normals')[:, 2]

        print(f"New normals shape: {point_cloud.point_data['new_normal_x'].shape}")
        print(f"Sample of new normals: {point_cloud.point_data['new_normal_x'][:5]}")
    except Exception as e:
        print(f"Error computing new normals: {str(e)}")
        print("Continuing without new normals...")

    return point_cloud

if __name__ == "__main__":
    
    
    file_name = 'C4-12'
    base_dir = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/obj"
    mesh_path = f'{base_dir}/{file_name}.glb'
    output_dir = os.path.join(base_dir, '_converted')

    combined_mesh = load_and_prepare_mesh(mesh_path)
    voxel_sizes_to_process = [1,0.5,0.25]
    for voxel_size in voxel_sizes_to_process:
                print(f"\nProcessing {file_name} with voxel size {voxel_size}")

                #convert pyvista mesh back into trimesh, preserving vertex colour information
                vertices = combined_mesh.points
                faces = combined_mesh.faces.reshape((-1, 4))[:, 1:]  # Remove the first column (face size)
                colors = combined_mesh.point_data['colors']

                print(f'converting back to trimesh')
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
                print(f'converted mesh back to trimesh')

                point_cloud = voxelize_mesh(mesh, voxel_size)
                output_file = os.path.join(output_dir, f"{file_name}-{voxel_size}.vtk")
                point_cloud.save(output_file)
                print(f"Saved voxelized mesh to {output_file}")

    # Export the combined mesh
    output_path = 'data/revised/obj/C4-12_combined.vtk'
    pv.save_meshio(output_path, combined_mesh)
    
    print('Combined mesh exported!')
