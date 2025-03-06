

import pyvista as pv
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd





def transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize(
    vtm_name: str, distance_threshold: float, site) -> None:
    
    vtm_file_path = f'data/{site}/{vtm_name}.vtm'
    raw_ply_file_path = f'data/{site}/{site}-points.ply'
    print(raw_ply_file_path)


    # Initialize a new MultiBlock object
    processed_multi_block = pv.MultiBlock()

    # Read the VTM and raw PLY files
    raw_multiblock = pv.read(vtm_file_path)
    raw_point_cloud = pv.read(raw_ply_file_path)
    
     # Check if the attributes are present in the VTM file
    for idx, block in enumerate(raw_multiblock):
        attributes = [key for key in block.point_data.keys()]
        print(f"Attributes for original block {idx + 1} in VTM: {attributes}")

    # Extract RGB and normal information from the raw point cloud
    raw_rgb = raw_point_cloud.point_data['RGB'] / 255.0  # Normalizing to 0-1
    raw_normals = raw_point_cloud.point_data['Normals']
    # Initialize a KDTree for the raw point cloud
    kd_tree = cKDTree(raw_point_cloud.points)
    
 
    
    # Iterate through each block in the multi-block dataset
    for idx, block in enumerate(raw_multiblock):
        print(f"Processing block {idx + 1}...")
        
        # Find the closest points and distances in the raw point cloud
        distances, closest_points = kd_tree.query(block.points)
        
        # Filter out points exceeding the distance threshold
        valid_indices = np.where(distances <= distance_threshold)[0]
        if valid_indices.size == 0:
            continue
        
        # Transfer RGB and normal attributes to valid points
        valid_rgb = raw_rgb[closest_points][valid_indices]
        valid_normals = raw_normals[closest_points][valid_indices]
        
        # Extract valid points for visualization
        valid_points = block.points[valid_indices]
        
        # Create a new PolyData object for the valid points
        valid_block = pv.PolyData(valid_points)
        valid_block.point_data['RGB'] = valid_rgb
        valid_block.point_data['Normals'] = valid_normals

        
        # Copy over existing attributes from the original block
        for attribute in block.point_data.keys():
            if attribute not in ['RGB', 'Normals']:
                valid_block.point_data[attribute] = block.point_data[attribute][valid_indices]
        
        # Add the modified block to the new MultiBlock
        processed_multi_block.append(valid_block)

    # Check attributes in the processed multi-block dataset
    for idx, block in enumerate(processed_multi_block):
        attributes = block.point_data.keys()
        print(f"Attributes for processed block {idx + 1}: {attributes}")
        for attribute in attributes:
            print(f"First point value for {attribute}: {block.point_data[attribute][0]}")
    
    return processed_multi_block, raw_multiblock
    

        

#if main
if __name__ == "__main__":

    site = "street"
    # Initialize the plotter
    plotter = pv.Plotter()
    # Call the function with distance threshold of 1.0 units
    #transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("data/buildings.vtm", "data/city-points.ply", 10)
    vtkmesh = transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("buildings", 10, site)
    
    #vtkmesh = transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("data/topography.vtm", "data/city-points.ply", 10)
    for idx, block in enumerate(vtkmesh):
        plotter.add_mesh(block, color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)
    plotter.show()
    
    #plotter.add_mesh(vtkmesh, color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)
    #plotter.show()

