import os, pyvista as pv
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from noise import pnoise2  # for Perlin noise

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
import a_vtk_to_ply

def generate_normals(voxelPolydata: pv.PolyData, site):
    print(f"Loading site and road voxels for {site}...")
    siteVoxels = pv.read(f'data/revised/{site}-siteVoxels-masked.vtk')
    roadVoxels = pv.read(f'data/revised/{site}-roadVoxels-coloured.vtk')

    print("Extracting points and normals...")
    # Extract points
    siteVoxelsPoints = siteVoxels.points
    roadVoxelsPoints = roadVoxels.points
    combinedPoints = np.concatenate((siteVoxelsPoints, roadVoxelsPoints), axis=0)

    # Extract and combine normals 
    siteNormals = np.column_stack((siteVoxels.point_data['orig_normal_x'],
                                  siteVoxels.point_data['orig_normal_y'],
                                  siteVoxels.point_data['orig_normal_z']))
    roadNormals = np.full((roadVoxelsPoints.shape[0], 3), [0, 0, 1])
    combinedNormals = np.concatenate((siteNormals, roadNormals), axis=0)

    print(f"Building KD-tree for {len(combinedPoints)} points...")
    tree = cKDTree(combinedPoints)

    print("Finding nearest neighbors and averaging normals...")
    # Query all points at once
    distances, indices = tree.query(voxelPolydata.points, k=100, distance_upper_bound=1.0)
    
    # Create mask for valid distances
    valid_mask = distances < np.inf
    
    # Initialize normals array
    normals = np.zeros((voxelPolydata.n_points, 3))
    
    # Calculate average normals for points with valid neighbors
    valid_points = valid_mask.any(axis=1)
    n_defaults = np.sum(~valid_points)
    print(f"Assigning default normal to {n_defaults} points with no valid neighbors")
    
    # Handle points with valid neighbors
    for_averaging = np.zeros_like(normals)
    for_averaging[valid_points] = np.array([
        np.mean(combinedNormals[indices[i][valid_mask[i]]], axis=0)
        for i in np.where(valid_points)[0]
    ])
    
    # Normalize the averaged normals
    norms = np.linalg.norm(for_averaging[valid_points], axis=1)
    for_averaging[valid_points] /= norms[:, np.newaxis]
    
    # Set default normal [0,0,1] for points with no valid neighbors
    for_averaging[~valid_points] = [0, 0, 1]
    
    normals[:] = for_averaging

    print("Adding normals to point data...")
    voxelPolydata.point_data['normals'] = normals
    voxelPolydata.point_data['normal_magnitude'] = np.linalg.norm(normals, axis=1)

    print("\nCreating oriented glyphs...")
    # Get the normal vectors directly from your data
    normal_vectors = voxelPolydata.point_data['normals']
    
    # Set as vector field for visualization
    voxelPolydata.point_data['Normals'] = normal_vectors
    
    # Create glyph with color based on normal magnitude
    """glyph = voxelPolydata.glyph(
        orient='Normals',
        scale=False  
        )
    
    # Create plotter
    plotter = pv.Plotter()

    
    # Add the normal glyphs
    plotter.add_mesh(
        glyph,
    )
    
    # Show the plot
    plotter.show()
    """

    return voxelPolydata





# Function to extract isosurface from PolyData. AI get rid of the resource_name parameter, and use spacing of 0.25 for all cases.
def extract_isosurface_from_polydata(polydata: pv.PolyData, isovalue: float = .5) -> pv.PolyData:
    print(f'polydata has {polydata.n_points} points')
    spacing = (1, 1, 1)  # Fixed spacing as requested
    
    if polydata is not None and polydata.n_points > 0:
        points = polydata.points
        x, y, z = points.T
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

        dims = (
            int((x_max - x_min) / spacing[0]) + 1,
            int((y_max - y_min) / spacing[1]) + 1,
            int((z_max - z_min) / spacing[2]) + 1
        )

        grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=(x_min, y_min, z_min))
        scalars = np.zeros(grid.n_points)

        for px, py, pz in points:
            ix = int((px - x_min) / spacing[0])
            iy = int((py - y_min) / spacing[1])
            iz = int((pz - z_min) / spacing[2])
            grid_idx = ix + iy * dims[0] + iz * dims[0] * dims[1]
            scalars[grid_idx] = 2

        grid.point_data['values'] = scalars
        isosurface = grid.contour(isosurfaces=[isovalue], scalars='values', method='flying_edges', compute_normals=True)
        return isosurface.extract_surface()
    else:
        return None

def transfer_point_data(original_poly: pv.PolyData, target_poly: pv.PolyData):
    if original_poly is None or original_poly.n_points == 0:
        print("WARNING: No valid original PolyData to transfer from.")
        return target_poly

    if target_poly is None or target_poly.n_points == 0:
        print("WARNING: No valid target PolyData to transfer onto.")
        return target_poly

    # Build a cKDTree on the original points
    kd_tree = cKDTree(original_poly.points)

    # Find the nearest neighbor indices for each target point
    distances, indices = kd_tree.query(target_poly.points)

    # Transfer all point data attributes
    for key in original_poly.point_data.keys():
        print(f"Transferring attribute '{key}'...")
        target_poly.point_data[key] = original_poly.point_data[key][indices]
    
    return target_poly

def subdivide_and_add_detail(mesh: pv.PolyData, detail_scale: float = 5.0, detail_amplitude: float = 0.1) -> pv.PolyData:
    """
    Subdivides the mesh and applies additional Perlin noise for finer detail
    
    Args:
        mesh: Input mesh to subdivide and detail
        detail_scale: Scale of the fine detail noise (higher = more frequent variations)
        detail_amplitude: Maximum height of the fine detail variations in meters
    """
    # Subdivide the mesh
    subdivided = mesh.subdivide(1, subfilter='linear')
    
    # Get points for noise application
    points = subdivided.points
    
    # Apply fine detail noise
    for i, (x, y, z) in enumerate(points):
        # Use a higher frequency noise for detail
        detail_noise = pnoise2(x * detail_scale, y * detail_scale)
        # Add small height variation
        points[i, 2] += detail_noise * detail_amplitude
    
    subdivided.points = points
    return subdivided

def generate_rewilded_ground(site: str, voxel_size: int, year: int, 
                           noise_scale: float = 1.0, 
                           max_height_variation: float = 0.5,
                           detail_scale: float = 10.0,
                           detail_amplitude: float = 0.1) -> pv.PolyData:
    """
    Generate rewilded ground surface with Perlin noise-based terrain variations
    
    Args:
        site: Name of the site
        voxel_size: Voxel size for processing
        year: Scenario year
        noise_scale: Scale of primary Perlin noise (lower = smoother variations)
        max_height_variation: Maximum height variation in meters
        detail_scale: Scale of fine detail noise
        detail_amplitude: Height of fine detail variations
    
    Returns:
        pv.PolyData: Processed ground surface with terrain variations
    """
    filePATH = f'data/revised/final/{site}'
    vtkPath = f'{filePATH}/{site}_{voxel_size}_scenarioYR{year}.vtk'
    print(f'loading polydata from {vtkPath}')

    poly = pv.read(vtkPath)

    generate_normals(poly, site)

    """# Convert to pandas Series for value_counts()
    rewilding_enabled = pd.Series(poly['scenario_rewildingEnabled'])
    rewilded = pd.Series(poly['scenario_rewilded'])

    print("Unique values and counts for scenario_rewildingEnabled:")
    print(rewilding_enabled.value_counts())
    print("\nUnique values and counts for scenario_rewilded:")
    print(rewilded.value_counts())

    # Create masks and extract ground points
    under_canopy_mask = (poly['scenario_rewilded'] != 'None') & (poly['scenario_rewilded'] != 'none')
    rewilded_mask = poly['scenario_rewildingEnabled'] > 0
    combined_mask = under_canopy_mask | rewilded_mask
    ground_poly = poly.extract_points(combined_mask)

    # Store original heights
    ground_poly.point_data['orgZ'] = ground_poly.points[:, 2].copy()

    # Apply primary Perlin noise
    points = ground_poly.points
    new_z = np.zeros(len(points))
    for i, (x, y, _) in enumerate(points):
        noise_val = pnoise2(x * noise_scale, y * noise_scale)
        height_offset = (noise_val + 1) * 0.5 * max_height_variation
        new_z[i] = points[i, 2] + height_offset

    ground_poly.point_data['newZ'] = new_z
    modified_points = points.copy()
    modified_points[:, 2] = new_z
    ground_poly.points = modified_points

    # Generate surface
    iso_surface = extract_isosurface_from_polydata(ground_poly)
    iso_surface = transfer_point_data(ground_poly, iso_surface)
    
    # Add fine detail
    iso_surface = subdivide_and_add_detail(
        iso_surface,
        detail_scale=detail_scale,
        detail_amplitude=detail_amplitude
    )"""

    return

    return iso_surface

def main():
    # Configuration
    site = 'trimmed-parade'
    site = 'city'
    voxel_size = 1
    years = [30]  # or [10, 30, 60, 180]
    
    # Create output directory
    filePATH = f'data/revised/final/{site}'
    os.makedirs(os.path.dirname(f'{filePATH}/ply'), exist_ok=True)

    # Process each year
    for year in years:
        iso_surface = generate_rewilded_ground(
            site=site,
            voxel_size=voxel_size,
            year=year,
            noise_scale=1.0,
            max_height_variation=0.5,
            detail_scale=10.0,
            detail_amplitude=0.1
        )

        iso_surface.plot(scalars='scenario_rewilded')

        # Export with vertex attributes
        attributes = ['scenario_rewilded', 'sim_Turns']
        outputFilePath = f'{filePATH}/ply/{site}_{voxel_size}_ground_scenarioYR{year}.ply'
        a_vtk_to_ply.export_polydata_to_ply(iso_surface, outputFilePath, attributesToTransfer=attributes)

if __name__ == "__main__":
    main()

