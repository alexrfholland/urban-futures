import os
import pyvista as pv
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


def filter_vtk_by_bounds2(site: str, length: float, width: float, center: [float, float]):
    # Read the VTK file
    vtk_path = f'data/{site}/flattened-{site}.vtk'
    poly_data = pv.read(vtk_path)
    
    # Read site projections to find the translation
    df = pd.read_csv('data/site projections.csv')
    site_data = df[df['Name'] == site]
    translation = [float(site_data['Easting']), float(site_data['Northing'])]
    
    # Calculate bounds
    min_x = center[0] - length / 2 - translation[0]
    max_x = center[0] + length / 2 - translation[0]
    min_y = center[1] - width / 2 - translation[1]
    max_y = center[1] + width / 2 - translation[1]
    
    # Filter the points
    mask = ((poly_data.points[:, 0] >= min_x) & (poly_data.points[:, 0] <= max_x) & 
            (poly_data.points[:, 1] >= min_y) & (poly_data.points[:, 1] <= max_y))
    filtered_poly_data = poly_data.extract_points(mask)
    
    # Create a folder if it doesn't exist
    folder_path = f'data/trimmed-{site}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # Save the filtered PolyData
    save_path = os.path.join(folder_path, f'flattend-trimmed-{site}.vtk')
    filtered_poly_data.save(save_path)
    
    # Plot the filtered PolyData
    plotter = pv.Plotter()
    plotter.add_mesh(filtered_poly_data)
    plotter.show()
    
    return filtered_poly_data


import pyvista as pv
import pandas as pd
import os
def transfer_attributes(source_polydata, target_polydata):
    # Ensure RGB and Normal data exist in the source PolyData
    assert 'RGB' in source_polydata.point_data, "Source PolyData must have RGB data"
    
    # Build a k-D tree for the source PolyData points
    source_tree = cKDTree(source_polydata.points)
    
    # Query the k-D tree to find indices of the closest points in source_polydata for each point in target_polydata
    _, indices = source_tree.query(target_polydata.points)
    
    # Transfer RGB and Normal data from source_polydata to target_polydata based on the closest point indices
    target_polydata.point_data['RGB'] = source_polydata.point_data['RGB'][indices]
    
    return target_polydata

def get_translation(site):
    # Read site projections to find the translation
    df = pd.read_csv('data/site projections.csv')
    site_data = df[df['Name'] == site]
    translation = [float(site_data['Easting']), float(site_data['Northing'])]
    return translation

def trim_vtk(poly_data, length, width, center, translation):
    # Adjust the center coordinates based on translation
    translated_center = [center[0] - translation[0], center[1] - translation[1]]
    # Calculate bounds
    min_x = translated_center[0] - length / 2
    max_x = translated_center[0] + length / 2
    min_y = translated_center[1] - width / 2
    max_y = translated_center[1] + width / 2
    
    # Filter the points
    mask = ((poly_data.points[:, 0] >= min_x) & (poly_data.points[:, 0] <= max_x) & 
            (poly_data.points[:, 1] >= min_y) & (poly_data.points[:, 1] <= max_y))
    filtered_poly_data = poly_data.extract_points(mask)
    
    return filtered_poly_data

def filter_vtk_by_bounds(site: str, length: float, width: float, center: [float, float]):
    # Read the VTK file
    vtk_path = f'data/{site}/flattened-{site}.vtk'
    poly_data = pv.read(vtk_path)

    topo_path = f'data/{site}/topography.vtm'
    topoMulti = pv.read(topo_path)
    topo = topoMulti[0]

    
    # Get the translation for the site
    translation = get_translation(site)
    
    # Call the trimming function
    filtered_poly_data = trim_vtk(poly_data, length, width, center, translation)
    filtered_topo = trim_vtk(topo, length, width, center, translation)
    filtered_topo = transfer_attributes(filtered_poly_data, filtered_topo)
        
    # Create a folder if it doesn't exist
    folder_path = f'data/trimmed-{site}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # Save the filtered PolyData
    save_path = os.path.join(folder_path, f'flattened-trimmed-{site}.vtk')
    filtered_poly_data.save(save_path)

    save_path_topo = os.path.join(folder_path, 'coloured-topography.vtk')
    filtered_topo.save(save_path_topo)
    
    # Plot the filtered PolyData
    plotter = pv.Plotter()
    plotter.add_mesh(filtered_poly_data)
    plotter.add_mesh(filtered_topo)
    plotter.show()
    
    return filtered_poly_data



# Test the function
site = "parade"
length = 300
width = 300
center = [320266.26,5815638.74]
filtered_poly_data = filter_vtk_by_bounds(site, length, width, center)
