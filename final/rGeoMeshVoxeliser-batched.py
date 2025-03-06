import trimesh
import numpy as np
import pyvista as pv
import os
import gc
import psutil
import signal
import logging
from datetime import datetime

def calculate_vertex_distances(mesh):
    edges = mesh.edges_unique
    vertices = mesh.vertices
    distances = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return np.min(distances), np.max(distances), np.mean(distances)

def load_and_prepare_mesh(mesh_path):
    print(f"Loading mesh from: {mesh_path}")
    scene = trimesh.load(mesh_path,force='mesh')
    print(f"Scene loaded successfully")
 
    if isinstance(scene, trimesh.Scene):
        print("Scene contains multiple objects, flattening into a single mesh")
        mesh = scene.dump(concatenate=True)
        print(f"Flattened mesh created with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    else:
        print("Scene contains a single mesh, no flattening needed")
        mesh = scene

    print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")

    # Subdivision
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

    if not hasattr(mesh.visual, 'vertex_colors'):
        print("Mesh does not have vertex colors. Converting visual to color representation...")
        mesh.visual = mesh.visual.to_color()
    else:
        print("Mesh already has vertex colors.")
        print(f"Vertex colors shape: {mesh.visual.vertex_colors.shape}")
        print(f"Sample of vertex colors: {mesh.visual.vertex_colors[:5]}")

    return mesh

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

def process_obj_files(base_dir, voxel_sizes, handle_errors=True):
    obj_files = []
    for root, dirs, files in os.walk(base_dir):
        if '_converted' in dirs:
            dirs.remove('_converted')
        for file in files:
            if file.endswith('.obj'):
                obj_files.append(os.path.join(root, file))

    output_dir = os.path.join(base_dir, '_converted')
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_dir, 'processing_log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    for obj_file in obj_files:
        file_name = os.path.splitext(os.path.basename(obj_file))[0]
        print(f"\nChecking file: {file_name}")

        # Check for incomplete processing
        if check_incomplete_processing(log_file, file_name):
            print(f"Skipping {file_name} due to incomplete previous processing.")
            continue

        # Check which voxel sizes need processing
        voxel_sizes_to_process = []
        for voxel_size in voxel_sizes:
            output_file = os.path.join(output_dir, f"{file_name}-{voxel_size}.vtk")
            if not os.path.exists(output_file):
                voxel_sizes_to_process.append(voxel_size)

        if not voxel_sizes_to_process:
            print(f"All voxel sizes for {file_name} have already been processed. Skipping.")
            continue

        print(f"Processing file: {file_name} for voxel sizes: {voxel_sizes_to_process}")
        logging.info(f"START - Processing {file_name}")

        try:
            mesh = load_and_prepare_mesh(obj_file)
            
            for voxel_size in voxel_sizes_to_process:
                print(f"\nProcessing {file_name} with voxel size {voxel_size}")
                point_cloud = voxelize_mesh(mesh, voxel_size)
                output_file = os.path.join(output_dir, f"{file_name}-{voxel_size}.vtk")
                point_cloud.save(output_file)
                print(f"Saved voxelized mesh to {output_file}")
                
                # Clear point_cloud from memory
                del point_cloud
                gc.collect()

            # Clear mesh from memory
            del mesh
            gc.collect()

            # Print memory usage
            process = psutil.Process()
            print(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

            logging.info(f"END - Successfully processed {file_name}")

        except Exception as e:
            error_message = f"Error processing {file_name}: {str(e)}"
            print(error_message)
            logging.error(error_message)
            if not handle_errors:
                raise

    print("Batch processing complete.")

def check_incomplete_processing(log_file, file_name):
    if not os.path.exists(log_file):
        return False

    with open(log_file, 'r') as f:
        lines = f.readlines()

    start_found = False
    for line in reversed(lines):
        if f"END - Successfully processed {file_name}" in line:
            return False
        if f"START - Processing {file_name}" in line:
            start_found = True
            break

    return start_found

def main():
    base_dir = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/obj"
    voxel_sizes = [1, 0.5, 0.25]
    handle_errors = True  # Set this to False if you want the script to stop on errors
    print("Starting batch processing of OBJ files...")
    
    # Monitor memory usage
    process = psutil.Process()
    print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    try:
        process_obj_files(base_dir, voxel_sizes, handle_errors)
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        print("Batch processing stopped due to error.")
    else:
        print("Batch processing complete.")
    
    print(f"Final memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()