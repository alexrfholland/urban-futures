import os
import csv
import trimesh
import numpy as np
import pandas as pd
import pyvista as pv

manualFiles = {'city' : ['C4-12', 'C4-18'],
               'trimmed-parade': ['A4-32','A4-26'],
               'uni' : ['B4-22', 'B4-21']
               }

def update_metadata_for_processed_meshes():
    """
    Updates metadata by extracting easting, northing, x_dimension, y_dimension, z_dimension, and file_name
    from all 0.25 voxel size .vtk files in the '_converted' folder, and saves the information into a CSV.
    
    Returns:
        None
    """
    # Path to the _converted folder
    converted_folder = 'data/revised/obj/_converted'
    
    # Initialize an empty list to store metadata
    metadata = []

    # Get a list of all files in the _converted folder
    print(f"Scanning folder: {converted_folder}")
    all_files = os.listdir(converted_folder)
    
    # Filter out .vtk files with voxel size 0.25
    vtk_files = [f for f in all_files if f.endswith('.vtk') and '-0.25.vtk' in f]

    print(f"Found {len(vtk_files)} .vtk files with voxel size 0.25.")

    # Process each .vtk file
    for vtk_file in vtk_files:
        # Extract the file_name without voxel size (i.e., the 'xx-xx' part)
        file_name = vtk_file.rsplit('-', 1)[0]
        
        # Full path to the vtk file
        vtk_path = os.path.join(converted_folder, vtk_file)
        
        # Load the vtk file using PyVista
        mesh = pv.read(vtk_path)
        
        # Compute the centroid of the mesh (easting, northing -> x, y coordinates)
        centroid = mesh.center
        
        # Extract x_dimension, y_dimension, z_dimension from the bounding box
        bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        x_dimension = bounds[1] - bounds[0]  # xmax - xmin
        y_dimension = bounds[3] - bounds[2]  # ymax - ymin
        z_dimension = bounds[5] - bounds[4]  # zmax - zmin

        # Append the extracted metadata to the list
        metadata.append({
            'file_name': file_name,
            'easting': centroid[0],  # x coordinate
            'northing': centroid[1],  # y coordinate
            'x_dimension': x_dimension,
            'y_dimension': y_dimension,
            'z_dimension': z_dimension
        })
    
    # Convert the metadata list to a DataFrame
    df = pd.DataFrame(metadata)
    
    # Path to save the metadata CSV
    output_csv_path = 'data/revised/obj/_converted/converted_mesh_centroids.csv'
    
    # Save the DataFrame to CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f"Metadata successfully saved to {output_csv_path}")
#AI PLEASE DO NOT DELETE THIS FUNCTION
def process_obj_files(folder):
    print(f"Starting to process OBJ files in folder: {folder}")
    results = []
    total_files = 0
    processed_files = 0
    skipped_files = 0
    error_files = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.obj') and 'centre' not in file.lower():
                total_files += 1
                obj_path = os.path.join(root, file)
                print(f"Processing file: {obj_path}")
                try:
                    scene = trimesh.load(obj_path)
                    if isinstance(scene, trimesh.Scene):
                        print(f"  File contains multiple geometries (Scene)")
                        all_vertices = []
                        for mesh in scene.geometry.values():
                            all_vertices.extend(mesh.vertices)
                        vertices = np.array(all_vertices)
                    elif isinstance(scene, trimesh.Trimesh):
                        print(f"  File contains a single mesh")
                        vertices = scene.vertices
                    else:
                        print(f"  Skipping {obj_path}: Unknown object type")
                        skipped_files += 1
                        continue
                    
                    centroid = np.mean(vertices, axis=0)
                    dimensions = np.ptp(vertices, axis=0)  # Calculate range (max - min) for each dimension
                    
                    results.append({
                        'file_name': file,
                        'easting': centroid[0],
                        'northing': centroid[1],
                        'x_dimension': dimensions[0],
                        'y_dimension': dimensions[1],
                        'z_dimension': dimensions[2]
                    })
                    print(f"  Centroid calculated: Easting={centroid[0]:.2f}, Northing={centroid[1]:.2f}")
                    print(f"  Dimensions: X={dimensions[0]:.2f}, Y={dimensions[1]:.2f}, Z={dimensions[2]:.2f}")
                    processed_files += 1
                except Exception as e:
                    print(f"  Error processing {obj_path}: {str(e)}")
                    error_files += 1

    csv_path = os.path.join(meshFolder, 'mesh_centroids.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'easting', 'northing', 'x_dimension', 'y_dimension', 'z_dimension']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results: 
            writer.writerow(row)
    
    print(f"CSV file saved at: {csv_path}")
    print(f"Processing complete. Total files: {total_files}, Processed: {processed_files}, Skipped: {skipped_files}, Errors: {error_files}")

import os
import pandas as pd
import pyvista as pv


def get_meshes_in_bounds(easting, northing, eastingsDim, northingsDim, voxel_size=None, update_metadata=False, manualSite=None):
    """
    Finds meshes within specified bounds or uses predefined meshes from the manualSite dictionary if provided.
    
    Args:
        easting (float): Center easting coordinate.
        northing (float): Center northing coordinate.
        eastingsDim (float): Width of the bounding box.
        northingsDim (float): Height of the bounding box.
        voxel_size (str): Voxel size to use for the meshes ('0.25', '0.5', '1'). Defaults to None.
        update_metadata (bool): Whether to update the metadata before fetching meshes. Defaults to False.
        manualSite (str): Key to look up predefined files in manualFiles. If provided, skips bounding box search.
    
    Returns:
        list: List of PyVista mesh objects within the bounding box or from the manualSite.
    """
    if update_metadata:
        print("Updating metadata...")
        update_metadata_for_processed_meshes()

    # Path to converted folder and the CSV metadata file
    converted_folder = 'data/revised/obj/_converted'
    csv_path = os.path.join(converted_folder, 'converted_mesh_centroids.csv')

    if manualSite:
        print(f"Using manual site: {manualSite}")
        mesh_names = manualFiles.get(manualSite, [])
        if not mesh_names:
            print(f"No manual files found for site: {manualSite}")
            return []

        # Fetch the manual mesh files
        meshes = []
        for mesh_name in mesh_names:
            size_to_use = voxel_size if voxel_size else '0.25'  # Default to 0.25 if voxel size is not provided
            vtk_path = os.path.join(converted_folder, f"{mesh_name}-{size_to_use}.vtk")
            if os.path.exists(vtk_path):
                print(f"Found mesh: {vtk_path}")
                mesh = pv.read(vtk_path)
                meshes.append(mesh)
            else:
                print(f"File not found: {vtk_path}")
        return meshes

    print(f"Finding meshes within bounds. Center: ({easting}, {northing}), Dimensions: {eastingsDim}x{northingsDim}")
    
    # Create the bounding box of the region of interest
    min_easting = easting - eastingsDim / 2
    max_easting = easting + eastingsDim / 2
    min_northing = northing - northingsDim / 2
    max_northing = northing + northingsDim / 2
    print(f"Bounding box: ({min_easting}, {min_northing}) to ({max_easting}, {max_northing})")

    # Read the CSV file containing mesh centroids
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")

    # Create a function to check if two bounding boxes overlap
    def bounding_boxes_overlap(mesh_min_e, mesh_max_e, mesh_min_n, mesh_max_n, 
                               region_min_e, region_max_e, region_min_n, region_max_n):
        # Check for overlap in easting and northing directions
        return not (mesh_max_e < region_min_e or mesh_min_e > region_max_e or
                    mesh_max_n < region_min_n or mesh_min_n > region_max_n)

    # Filter meshes by checking their bounding boxes
    meshes = []
    for _, row in df.iterrows():
        mesh_name = row['file_name']
        mesh_easting = row['easting']
        mesh_northing = row['northing']
        
        # Load the mesh
        available_sizes = ['0.25', '0.5', '1']
        size_to_use = voxel_size if voxel_size else '0.25'  # Default to 0.25 if voxel size is not provided
        vtk_path = os.path.join(converted_folder, f"{mesh_name}-{size_to_use}.vtk")
        
        if os.path.exists(vtk_path):
            print(f"Found mesh: {vtk_path}")
            mesh = pv.read(vtk_path)
            
            # Get the bounding box of the mesh
            bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            mesh_min_easting, mesh_max_easting = bounds[0], bounds[1]
            mesh_min_northing, mesh_max_northing = bounds[2], bounds[3]

            # Check if the mesh bounding box overlaps with the region bounding box
            if bounding_boxes_overlap(mesh_min_easting, mesh_max_easting, mesh_min_northing, mesh_max_northing,
                                      min_easting, max_easting, min_northing, max_northing):
                meshes.append(mesh)
        else:
            print(f"File not found: {vtk_path}")
    
    print(f"Total meshes found within bounds: {len(meshes)}")
    return meshes




def get_meshes_in_bounds2(easting, northing, eastingsDim, northingsDim, voxel_size=None):
    print(f"Finding meshes within bounds. Center: ({easting}, {northing}), Dimensions: {eastingsDim}x{northingsDim}")
    
    # Create bounding box
    min_easting = easting - eastingsDim / 2
    max_easting = easting + eastingsDim / 2
    min_northing = northing - northingsDim / 2
    max_northing = northing + northingsDim / 2
    print(f"Bounding box: ({min_easting}, {min_northing}) to ({max_easting}, {max_northing})")

    # Read CSV file
    csv_path = 'data/revised/obj/_converted/converted_mesh_centroids.csv'
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")

    # Filter meshes within bounds
    within_bounds = df[
        (df['easting'] >= min_easting) & (df['easting'] <= max_easting) &
        (df['northing'] >= min_northing) & (df['northing'] <= max_northing)
    ]
    print(f"Meshes within bounds: {len(within_bounds)}")

    # Process each mesh within bounds
    meshes = []
    for _, row in within_bounds.iterrows():
        mesh_name = row['file_name']
        
        # Always use the smallest available voxel size
        available_sizes = ['0.25', '0.5', '1']
        size_to_use = min(available_sizes, key=float)
        
        vtk_path = os.path.join('data', 'revised', 'obj', '_converted', f"{mesh_name}-{size_to_use}.vtk")
        
        if os.path.exists(vtk_path):
            print(f"Found mesh: {vtk_path}")
            # Load mesh with PyVista
            mesh = pv.read(vtk_path)
            meshes.append(mesh)
        else:
            print(f"File not found: {vtk_path}")

    print(f"Total meshes found: {len(meshes)}")
    return meshes





def process_converted_vtk_files(folder):
    print(f"Starting to process converted VTK files in folder: {folder}")
    results = {}
    total_files = 0
    processed_files = 0
    skipped_files = 0
    error_files = 0

    # First pass: identify all mesh names and their available voxel sizes
    mesh_voxel_sizes = {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.vtk'):
                parts = file.split('-')
                mesh_name = '-'.join(parts[:-1])
                voxel_size = parts[-1].split('.')[0]
                if mesh_name not in mesh_voxel_sizes:
                    mesh_voxel_sizes[mesh_name] = set()
                mesh_voxel_sizes[mesh_name].add(voxel_size)

    # Second pass: process files
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.vtk'):
                total_files += 1
                vtk_path = os.path.join(root, file)
                parts = file.split('-')
                mesh_name = '-'.join(parts[:-1])
                voxel_size = parts[-1].split('.')[0]

                # Skip if we've already processed a better voxel size for this mesh
                if mesh_name in results:
                    skipped_files += 1
                    continue

                # Prioritize 0.25 voxel size
                if voxel_size != '0.25' and '0.25' in mesh_voxel_sizes[mesh_name]:
                    skipped_files += 1
                    continue

                print(f"Processing file: {vtk_path}")
                try:
                    # Load the VTK file
                    mesh = pv.read(vtk_path)
                    
                    # Calculate centroid and dimensions
                    bounds = np.array(mesh.bounds)
                    centroid = (bounds[1::2] + bounds[::2]) / 2
                    dimensions = bounds[1::2] - bounds[::2]
                    
                    results[mesh_name] = {
                        'file_name': mesh_name,
                        'easting': centroid[0],
                        'northing': centroid[1],
                        'x_dimension': dimensions[0],
                        'y_dimension': dimensions[1],
                        'z_dimension': dimensions[2],
                        'voxel_size': voxel_size
                    }
                    print(f"  Centroid calculated: Easting={centroid[0]:.2f}, Northing={centroid[1]:.2f}")
                    print(f"  Dimensions: X={dimensions[0]:.2f}, Y={dimensions[1]:.2f}, Z={dimensions[2]:.2f}")
                    processed_files += 1
                except Exception as e:
                    print(f"  Error processing {vtk_path}: {str(e)}")
                    error_files += 1

    csv_path = os.path.join(folder, 'converted_mesh_centroids.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'easting', 'northing', 'x_dimension', 'y_dimension', 'z_dimension', 'voxel_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results.values():
            writer.writerow(row)
    
    print(f"CSV file saved at: {csv_path}")
    print(f"Processing complete. Total files: {total_files}, Processed: {processed_files}, Skipped: {skipped_files}, Errors: {error_files}")

# Example usage:
"""if __name__ == "__main__":
    converted_folder = 'data/revised/obj/_converted'
    process_converted_vtk_files(converted_folder)"""

import pandas as pd
import pyvista as pv
import numpy as np

def debug_plot_mesh_bounds(plotter):
    """
    Loads the CSV file and creates bounding boxes centered on easting and northing, 
    using x_dimension and y_dimension. Labels the centroid with file_name on the plotter.
    
    Args:
        plotter (pv.Plotter): PyVista plotter where the bounding boxes and labels will be added.
        
    Returns:
        None
    """
    csv_path = 'data/revised/obj/_converted/converted_mesh_centroids.csv'
    
    # Load the CSV file
    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")
    
    # Iterate through each row to create bounding boxes
    for _, row in df.iterrows():
        file_name = row['file_name']
        easting = row['easting']
        northing = row['northing']
        x_dim = row['x_dimension']
        y_dim = row['y_dimension']
        
        # Define the bounding box coordinates (2D rectangle)
        min_easting = easting - x_dim / 2
        max_easting = easting + x_dim / 2
        min_northing = northing - y_dim / 2
        max_northing = northing + y_dim / 2
        
        # Create the points of the bounding box
        corners = np.array([
            [min_easting, min_northing, 0],  # Bottom-left
            [max_easting, min_northing, 0],  # Bottom-right
            [max_easting, max_northing, 0],  # Top-right
            [min_easting, max_northing, 0],  # Top-left
            [min_easting, min_northing, 0]   # Closing the loop
        ])
        
        # Create a polyline for the bounding box
        line = pv.PolyData(corners)
        lines = np.array([5, 0, 1, 2, 3, 4])  # 5 points, closed loop
        line.lines = lines
        
        # Add the bounding box to the plotter
        plotter.add_mesh(line, color="blue", line_width=2)
        
        # Label the centroid with file_name
        plotter.add_point_labels([easting, northing, 0], [file_name], point_size=10, text_color="black")
    
    print(f"Bounding boxes and labels added to plotter.")



# Example usage:
if __name__ == "__main__":
    import f_SiteCoordinates
    
    meshFolder = 'data/revised/obj'
    print("Starting main execution")
    site_name = 'city'
    easting, northing = f_SiteCoordinates.get_site_coordinates(site_name)
    eastingsDim, northingsDim = 500,500
    
    meshes = get_meshes_in_bounds(easting, northing, eastingsDim, northingsDim,update_metadata=True,manualSite=site_name)
    
    # Create a plotter and add meshes
    plotter = pv.Plotter()
    for mesh in meshes:
        plotter.add_mesh(mesh,rgb=True)

    debug_plot_mesh_bounds(plotter)



    #bounding box
    # Define the bounding box corners (centered on easting/northing)
    min_easting = easting - eastingsDim / 2
    max_easting = easting + eastingsDim / 2
    min_northing = northing - northingsDim / 2
    max_northing = northing + northingsDim / 2
    
    # Bounding box coordinates (z = 0 for all)
    corners = np.array([
        [min_easting, min_northing, 0],  # Bottom-left
        [max_easting, min_northing, 0],  # Bottom-right
        [max_easting, max_northing, 0],  # Top-right
        [min_easting, max_northing, 0],  # Top-left
        [min_easting, min_northing, 0]   # Closing the loop to bottom-left
    ])
    

    # Create the polyline
    line = pv.PolyData(corners)
    lines = np.array([5, 0, 1, 2, 3, 4])  # 5 points, closed loop
    line.lines = lines

    # Add the bounding box polyline
    plotter.add_mesh(line, color="red", line_width=3)
    
    # Set up the camera
    camera_height = max(eastingsDim, northingsDim)
    plotter.camera_position = [
        (easting, northing, camera_height),  # Camera position
        (easting, northing, 0),  # Focal point
        (0, 1, 0)  # View up direction
    ]
    
    print("Displaying the plot...")
    plotter.show()
    
    print("Main execution completed")
