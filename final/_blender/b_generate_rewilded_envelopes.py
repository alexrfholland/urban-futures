import os, pyvista as pv
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from noise import pnoise2  # for Perlin noise

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path
import a_vtk_to_ply


def transfer_point_data(original_poly: pv.PolyData, target_poly: pv.PolyData, attributes=['sim_Turns']):
    """Transfer point data attributes from original to target polydata using nearest neighbor"""
    if original_poly is None or target_poly is None:
        return target_poly
        
    print(f"Transferring attributes from original ({original_poly.n_points} points) to target ({target_poly.n_points} points)")
    
    # Build KD-tree on original points
    tree = cKDTree(original_poly.points)
    
    # Find nearest neighbors for all target points
    _, indices = tree.query(target_poly.points)
    
    # Transfer specified attributes
    for attr in attributes:
        if attr in original_poly.point_data:
            print(f"Transferring {attr}")
            target_poly.point_data[attr] = original_poly.point_data[attr][indices]
        else:
            print(f"Warning: {attr} not found in original polydata")
    
    return target_poly
    

def load_reference_data(site):
    """Load site and road data and create KDTree"""
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
    
    return tree, combinedPoints, combinedNormals

def generate_normals(voxelPolydata: pv.PolyData, tree: cKDTree, combinedPoints: np.ndarray, combinedNormals: np.ndarray):
    """Generate normals using pre-computed KDTree and reference data"""
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

    return voxelPolydata





# Function to extract isosurface from PolyData. AI get rid of the resource_name parameter, and use spacing of 0.25 for all cases.
def extract_isosurface_from_polydata(polydata: pv.PolyData, spacing: tuple[float, float, float], bioenv: str, surface_cat: str) -> pv.PolyData:
    """Extract isosurface from point cloud data using simple approach"""
    category = f"{bioenv}-{surface_cat}"
    print(f'{category} polydata has {polydata.n_points} points')
    
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
            if grid_idx < len(scalars):
                scalars[grid_idx] = 2

        grid.point_data['values'] = scalars
        isosurface = grid.contour(isosurfaces=[0.5], scalars='values', method='flying_edges', compute_normals=True)
        surface = isosurface.extract_surface()
        
        # Add bioenv and surface category to point data
        surface.point_data['scenario_bioEnvelope'] = np.full(surface.n_points, bioenv)
        surface.point_data['surfaceCat'] = np.full(surface.n_points, surface_cat)
        
        return surface
    else:
        return None


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


def categorize_surface_by_normal(poly):
    """Categorize surface points by their normal direction"""
    normals = poly.point_data['normals']
    
    # Define angle threshold for vertical surfaces (in radians)
    threshold = np.cos(np.radians(45))  # 45-degree threshold
    
    # Create masks for vertical directions
    x_pos_facing = (normals[:, 0] > threshold)
    x_neg_facing = (normals[:, 0] < -threshold)
    y_pos_facing = (normals[:, 1] > threshold)
    y_neg_facing = (normals[:, 1] < -threshold)
    
    # Store categories in point data - anything not vertical is 'up'
    poly.point_data['surface_category'] = np.select(
        [x_pos_facing, x_neg_facing, y_pos_facing, y_neg_facing],
        ['x_pos', 'x_neg', 'y_pos', 'y_neg'],
        default='up'
    )
    
    # Print counts for each category and bioenvelope type
    print(f"\nFor bioenvelope type: {poly.point_data['scenario_bioEnvelope'][0]}")
    unique, counts = np.unique(poly.point_data['surface_category'], return_counts=True)
    for cat, count in zip(unique, counts):
        print(f"{cat}: {count} points")
    
    return poly

def apply_perlin_noise_along_normal(poly):
    """Apply Perlin noise displacement along normal direction"""
    import noise
    
    points = poly.points
    normals = poly.point_data['normals']
    
    # Generate Perlin noise for each point
    noise_values = np.zeros(len(points))
    for i in range(len(points)):
        # Use x,y coordinates for noise input
        noise_values[i] = noise.pnoise2(
            points[i,0],
            points[i,1],
            octaves=3,
            persistence=0.5,
            lacunarity=2.0,
            repeatx=1000,
            repeaty=1000,
            base=0
        )
    
    # Scale noise to desired amplitude (0.25m)
    noise_values = noise_values * 0.25
    
    # Displace points along their normals
    new_points = points + (normals * noise_values[:, np.newaxis])
    
    # Update polydata points
    poly.points = new_points
    
    return poly


def process_bioenvelope_surfaces(poly):
    """Process each bioenvelope category"""
    # Get mask for non-'none' bioenvelope points
    valid_mask = poly.point_data['scenario_bioEnvelope'] != 'none'
    valid_surface = poly.extract_points(valid_mask)
    
    # Get unique bioenvelope categories
    unique_categories = np.unique(valid_surface.point_data['scenario_bioEnvelope'])
    print("Found bioenvelope categories:", unique_categories)
    
    processed_surfaces = []
    
    for category in unique_categories:
        # Extract surface for this category
        category_mask = valid_surface.point_data['scenario_bioEnvelope'] == category
        category_surface = valid_surface.extract_points(category_mask)
        
        if len(category_surface.points) > 0:
            print(f"Processing {category} with {len(category_surface.points)} points")
            
            # Categorize by normal direction
            category_surface = categorize_surface_by_normal(category_surface)
            
            # Apply Perlin noise along normal direction
            category_surface = apply_perlin_noise_along_normal(
                category_surface,
                amplitude=0.25,
                frequency=1.0
            )
            
            processed_surfaces.append(category_surface)
    
    # Combine all processed surfaces
    if processed_surfaces:
        combined = processed_surfaces[0].merge(processed_surfaces[1:])
        return combined
    return None


def generate_rewilded_envelopes(voxelPolydata: pv.PolyData, site: str, tree: cKDTree, 
                              combinedPoints: np.ndarray, combinedNormals: np.ndarray):
    """Generate rewilded envelope surfaces with appropriate spacing"""
    
    # 1. Get valid bioenvelope subset
    valid_mask = voxelPolydata.point_data['scenario_bioEnvelope'] != 'none'
    valid_polydata = voxelPolydata.extract_points(valid_mask)
    print(f"Found {len(valid_polydata.points)} valid bioenvelope points")
    
    # 2. Generate normals using pre-computed data
    valid_polydata = generate_normals(valid_polydata, tree, combinedPoints, combinedNormals)
    
    # 3. Categorize surfaces by normal direction
    valid_polydata = categorize_surface_by_normal(valid_polydata)
    
    # 4. Generate isosurfaces for each combination
    processed_surfaces = []
    unique_bioenvelopes = np.unique(valid_polydata.point_data['scenario_bioEnvelope'])
    
    for bioenv in unique_bioenvelopes:
        print(f"\nProcessing {bioenv}")
        
        # Extract points for this bioenvelope type
        bioenv_mask = valid_polydata.point_data['scenario_bioEnvelope'] == bioenv
        points_subset = valid_polydata.extract_points(bioenv_mask)
        
        if bioenv == 'livingFacade':
            # For livingFacade, use normal-based spacing
            unique_surfaces = np.unique(points_subset.point_data['surface_category'])
            for surf_cat in unique_surfaces:
                cat_mask = points_subset.point_data['surface_category'] == surf_cat
                cat_points = points_subset.extract_points(cat_mask)
                
                if cat_points.n_points > 0:
                    # Set spacing based on surface category
                    spacing = [1.0, 1.0, 1.0]
                    if surf_cat == 'x_pos' or surf_cat == 'x_neg':
                        #spacing[0] = 0.25
                        spacing[0] = 1
                    elif surf_cat == 'y_pos' or surf_cat == 'y_neg':
                        #spacing[1] = 0.25
                        spacing[1] = 1
                        spacing[1] = 1
                        
                    
                    surface = extract_isosurface_from_polydata(
                        cat_points,
                        tuple(spacing),
                        bioenv,
                        surf_cat
                    )
                    
                    if surface is not None:
                        # Transfer point data from original
                        #surface = transfer_point_data(valid_polydata, surface)
                        processed_surfaces.append(surface)
        
        else:
            # For all other types, use upward facing spacing
            if points_subset.n_points > 0:
                surface = extract_isosurface_from_polydata(
                    points_subset,
                    #(1.0, 1.0, 0.25),  # upward facing spacing
                    (1.0, 1.0, 1.0),
                    bioenv,
                    'up'
                )
                
                if surface is not None:
                    # Transfer point data from original
                    #surface = transfer_point_data(valid_polydata, surface)
                    processed_surfaces.append(surface)
    
    # Combine and save results
    if processed_surfaces:
        combined = processed_surfaces[0].merge(processed_surfaces[1:])
        combined = transfer_point_data(valid_polydata, combined)
        #combined.save(f'data/revised/final/{site}-rewilded_envelopes.vtk')
        return combined
    
    return None

def scenario_bioenvelope_map_to_int_simple(iso_surface):
    """Map scenario_rewilded strings to integer values for easier processing"""
    # First mapping (full)
    category_map = {
        'exoskeleton': 1,
        'brownRoof': 2, 
        'otherGround': 3,
        'node-rewilded': 4,
        'footprint-depaved': 5,
        'livingFacade': 6,
        'greenRoof': 7
    }

    # Second mapping (simplified)
    simplified_map = {
        'brownRoof': 2,
        'livingFacade': 3,
        'greenRoof': 4
    }

    # Convert to pandas series and map values
    scenario_bioenvelope = pd.Series(iso_surface.point_data['scenario_bioEnvelope'])
    scenario_bioenvelope_int = scenario_bioenvelope.map(category_map).to_numpy()
    
    # Create simplified mapping (default to 1 for unmapped categories)
    scenario_bioenvelope_simple = scenario_bioenvelope.map(simplified_map).fillna(1).to_numpy()
    
    # Add both arrays to point data
    iso_surface.point_data['scenario_bioEnvelope_int'] = scenario_bioenvelope_int
    iso_surface.point_data['scenario_bioEnvelope_simple_int'] = scenario_bioenvelope_simple
    
    # Print value counts for both versions
    print("\nOriginal scenario_rewilded value counts:")
    print(scenario_bioenvelope.value_counts())
        
    print("\nInteger scenario_rewilded value counts:")
    print(pd.Series(scenario_bioenvelope_int).value_counts().sort_index())
    
    return iso_surface

def main():
    # Configuration
    site = 'uni'
    scenarios = ['positive', 'trending']
    voxel_size = 1
    years = [10, 30, 60, 180]

    # Create ply directory if it doesn't exist
    filePATH = f'data/revised/final/{site}'
    ply_dir = f'{filePATH}/ply'
    os.makedirs(ply_dir, exist_ok=True)
    
    # Load reference data once
    tree, combinedPoints, combinedNormals = load_reference_data(site)
    
    # Process each year
    for scenario in scenarios:
        for year in years:

            # Load voxel polydata
            
            #vtkPath = f'{filePATH}/{site}_{voxel_size}_scenarioYR{year}.vtk'
            vtkPath = f'{filePATH}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk'
            print(f'loading polydata from {vtkPath}')
            voxelPolydata = pv.read(vtkPath)
            
            # Generate envelopes using pre-computed reference data
            iso_surface = generate_rewilded_envelopes(
                voxelPolydata, 
                site, 
                tree, 
                combinedPoints, 
                combinedNormals
            )

            iso_surface = scenario_bioenvelope_map_to_int_simple(iso_surface)
            
            if iso_surface is not None:
                print(f"Successfully generated rewilded envelopes for year {year} and {scenario}")
                attributes = ['scenario_bioEnvelope_int', 'scenario_bioEnvelope_simple_int', 'sim_Turns']
                
                outputFilePath = f'{ply_dir}/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.ply'
                a_vtk_to_ply.export_polydata_to_ply(iso_surface, outputFilePath, attributesToTransfer=attributes)
                outputFilePath = f'{filePATH}/vtk/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.vtk'
                iso_surface.save(outputFilePath)



if __name__ == "__main__":
    main()


