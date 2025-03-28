import pandas as pd
import xarray as xr
import numpy as np
import pyvista as pv
import a_helper_functions, a_resource_distributor_dataframes
import os


#f"data/revised/final/{site}-roadVoxels-coloured.vtk"

def getGround(xarray_dataset, terrain_polydata):
    """Find ground voxels using optimized KDTree search"""
    # Convert relevant xarray variables to pandas dataframe
    df = pd.DataFrame({
        'voxel_I': xarray_dataset['voxel_I'].values,
        'voxel_J': xarray_dataset['voxel_J'].values,
        'voxel_K': xarray_dataset['voxel_K'].values,
        'centroid_x': xarray_dataset['centroid_x'].values,
        'centroid_y': xarray_dataset['centroid_y'].values,
        'centroid_z': xarray_dataset['centroid_z'].values,
    })

    # Initialize isGround as False
    df['isGround'] = False
    
    # Get terrain points and build KDTree
    terrain_points = terrain_polydata.points
    from scipy.spatial import cKDTree
    
    
    # Build tree with reduced point set
    tree = cKDTree(terrain_points, leafsize=16)  # optimized leafsize
    
    # Get voxel centroids
    voxel_points = df[['centroid_x', 'centroid_y', 'centroid_z']].values
    
    # Optimization 2: Use approximate nearest neighbor search
    distances, _ = tree.query(
        voxel_points,
        k=1,
        eps=0.1,  # allow 10% error for faster search
        distance_upper_bound=1.0  # early stopping for distances > 1.0
    )
    
    # Mark voxels within 1m of terrain as ground
    df.loc[distances <= 1.0, 'isGround'] = True
    
    num_ground = df['isGround'].sum()
    print(f'Found {num_ground} ground voxels (within 1m of terrain)')
    print(f'Using {len(terrain_points)} terrain points (reduced from {len(terrain_polydata.points)})')
    
    """# Quick visualization of results
    ground_df = df[df['isGround']]
    cloud = pv.PolyData(np.column_stack((
        ground_df['centroid_x'].values,
        ground_df['centroid_y'].values, 
        ground_df['centroid_z'].values
    )))
    cloud.plot(point_size=5)"""
    
    return df

def getPositions(baseline_tree_df, terrain_df, seed=42):
    """Assign positions to trees based on ground voxels"""
    # Set random seed
    np.random.seed(seed)
    
    # Shuffle the baseline dataframe
    baseline_tree_df = baseline_tree_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Get ground voxels
    ground_df = terrain_df[terrain_df['isGround']].reset_index()  # index becomes voxelID
    
    # Check we have enough ground voxels
    n_trees = len(baseline_tree_df)
    n_ground = len(ground_df)
    if n_trees > n_ground:
        raise ValueError(f"More trees ({n_trees}) than ground voxels ({n_ground})")
    
    # Randomly select ground voxels
    selected_rows = ground_df.sample(n=n_trees, random_state=seed)
    
    # Update tree positions
    baseline_tree_df['voxelID'] = selected_rows['index']
    baseline_tree_df['x'] = selected_rows['centroid_x'].values
    baseline_tree_df['y'] = selected_rows['centroid_y'].values
    baseline_tree_df['z'] = selected_rows['centroid_z'].values
    baseline_tree_df['voxel_I'] = selected_rows['voxel_I'].values
    baseline_tree_df['voxel_J'] = selected_rows['voxel_J'].values
    baseline_tree_df['voxel_K'] = selected_rows['voxel_K'].values
    
    return baseline_tree_df



def calculate_useful_life_expectancy(df):
    """Calculate useful life expectancy for trees based on their DBH and state"""
    print("\nCalculating useful life expectancy...")
    
    # Calculate mean growth factor
    growth_factors = [0.37, 0.51]
    mean_growth_factor = np.mean(growth_factors)
    print(f"Using mean growth factor: {mean_growth_factor:.2f} cm/year")
    
    # Calculate base age for all trees using DBH
    df['age'] = df['diameter_breast_height'] / mean_growth_factor
    print("\nAge ranges for normal trees:")
    age_stats = df[df['size'].isin(['small', 'medium', 'large'])]['age'].describe()
    print(f"Min age: {age_stats['min']:.1f} years")
    print(f"Max age: {age_stats['max']:.1f} years")
    print(f"Mean age: {age_stats['mean']:.1f} years")
    
    # Calculate senescing addition (difference between max normal age and 600)
    max_dbh_age = 80 / mean_growth_factor
    senescing_addition = 600 - max_dbh_age
    print(f"\nSenescing addition: {senescing_addition:.1f} years")
    
    # Add random years to senescing trees
    senescing_mask = df['size'] == 'senescing'
    num_senescing = senescing_mask.sum()
    if num_senescing > 0:
        np.random.seed(42)
        senescing_ages = df.loc[senescing_mask, 'age'] + np.random.uniform(0, senescing_addition, num_senescing)
        df.loc[senescing_mask, 'age'] = senescing_ages
        print(f"\nAdjusted ages for {num_senescing} senescing trees")
        print(f"Senescing age range: {senescing_ages.min():.1f} to {senescing_ages.max():.1f} years")
    
    # Set ages for snag trees
    snag_mask = df['size'] == 'snag'
    num_snags = snag_mask.sum()
    if num_snags > 0:
        np.random.seed(43)
        df.loc[snag_mask, 'age'] = np.random.uniform(450, 500, num_snags)
        print(f"\nSet ages for {num_snags} snag trees (450-500 years)")
    
    # Set ages for fallen trees
    fallen_mask = df['size'] == 'fallen'
    num_fallen = fallen_mask.sum()
    if num_fallen > 0:
        np.random.seed(44)
        df.loc[fallen_mask, 'age'] = np.random.uniform(500, 600, num_fallen)
        print(f"Set ages for {num_fallen} fallen trees (500-600 years)")
    
    # Calculate useful life expectancy
    df['useful_life_expectancy'] = 600 - df['age']
    
    # Print final statistics
    print("\nFinal useful life expectancy statistics:")
    for size in df['size'].unique():
        ule_stats = df[df['size'] == size]['useful_life_expectancy'].describe()
        print(f"\n{size.capitalize()} trees:")
        print(f"Count: {ule_stats['count']:.0f}")
        print(f"Mean: {ule_stats['mean']:.1f} years")
        print(f"Min: {ule_stats['min']:.1f} years")
        print(f"Max: {ule_stats['max']:.1f} years")
    
    return df

##terrain_surface = cloud.reconstruct_surface(nbr_sz=20)

def get_terrain_poly(terrain_df):
    """Create and visualize terrain points with spatially coherent Perlin noise"""
    # Filter ground points
    ground_points = terrain_df[terrain_df['isGround']]
    
    # Create terrain point cloud
    terrain_points = np.column_stack((
        ground_points['centroid_x'].values,
        ground_points['centroid_y'].values,
        ground_points['centroid_z'].values
    ))
    terrain_cloud = pv.PolyData(terrain_points)
    
    # Get bounds for Perlin noise
    bounds = terrain_cloud.bounds
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    # Scale the coordinates to get larger noise features
    scale = 0.02  # Decrease this value for larger features
    frequency = (2, 2, 1)  # Adjust frequency for each dimension
    
    # Generate Perlin noise for terrain points
    noise = pv.perlin_noise(scale, frequency, (0, 0, 0))
    noise_values = np.zeros(len(terrain_points))
    
    # Normalize coordinates to noise space
    x_norm = (terrain_points[:, 0] - x_min) / (x_max - x_min)
    y_norm = (terrain_points[:, 1] - y_min) / (y_max - y_min)
    
    # Generate spatially coherent noise
    for i, (x, y) in enumerate(zip(x_norm, y_norm)):
        noise_values[i] = noise.EvaluateFunction(x * 10, y * 10, 0)  # Multiply by 10 to spread out the noise
    
    # Normalize noise to 0-1 range
    noise_values = (noise_values - noise_values.min()) / (noise_values.max() - noise_values.min())
    
    # Add noise values to terrain cloud
    terrain_cloud.point_data['noise'] = noise_values

    return terrain_cloud

def combine_polydata(resourcePoly, terrainPoly):
    """Combine resource and terrain polydata with appropriate search variables"""
    print("Combining resource and terrain polydata...")
    
    # Initialize search_bioavailable for resourcePoly as 'arboreal'
    search_bioavailable_resource = np.full(resourcePoly.n_points, 'arboreal', dtype='<U20')
    resourcePoly.point_data['search_bioavailable'] = search_bioavailable_resource
    print(f"  Set {resourcePoly.n_points:,} resource points to 'arboreal'")
    
    # Initialize search_bioavailable for terrainPoly as 'low-vegetation'
    search_bioavailable_terrain = np.full(terrainPoly.n_points, 'low-vegetation', dtype='<U20')
    terrainPoly.point_data['search_bioavailable'] = search_bioavailable_terrain
    print(f"  Set {terrainPoly.n_points:,} terrain points to 'low-vegetation'")

    # Print original resourcePoly point_data variables
    print("\nOriginal resourcePoly point_data variables:")
    for key in resourcePoly.point_data.keys():
        print(f"  - {key}")

    # Append 'forest_' to resourcePoly point_data attribute names
    for attr in ['useful_life_expectancy', 'precolonial', 'size', 'control']:
        if attr in resourcePoly.point_data:
            # Create a new attribute with 'forest_' prefix and copy the data
            resourcePoly.point_data[f'forest_{attr}'] = resourcePoly.point_data[attr].copy()
            # Remove the original attribute
            del resourcePoly.point_data[attr]
            print(f"  Renamed '{attr}' to 'forest_{attr}'")

    # Create a copy of resourcePoly
    combinedPoly = resourcePoly.copy()
    
    # Initialize any missing variables in terrainPoly based on resourcePoly
    terrainPoly = a_helper_functions.initialize_polydata_variables_generic_auto(terrainPoly, resourcePoly)
    
    # Initialize any missing variables in resourcePoly based on terrainPoly
    resourcePoly = a_helper_functions.initialize_polydata_variables_generic_auto(resourcePoly, terrainPoly)
    
    # Print resourcePoly point_data variables after initialization
    print("\nresourcePoly point_data variables after initialization:")
    for key in resourcePoly.point_data.keys():
        print(f"  - {key}")
    
    # Append terrainPoly to combinedPoly
    combinedPoly = combinedPoly.append_polydata(terrainPoly)
    print(f"  Combined polydata has {combinedPoly.n_points:,} total points")
    
    # Print combinedPoly point_data variables
    print("\ncombinedPoly point_data variables:")
    for key in combinedPoly.point_data.keys():
        print(f"  - {key}")
    
    combinedPoly.plot(scalars='resource_fallen log', render_points_as_spheres=True)
    
    return combinedPoly

def generate_baseline(site, voxel_size=1, output_folder='data/revised/final/baselines'):
    """
    Generate baseline trees and resources for a site.
    
    Parameters:
    site (str): Site name (e.g., 'trimmed-parade', 'city', 'uni')
    voxel_size (int): Voxel size for the simulation
    output_folder (str): Folder to save output files
    
    Returns:
    tuple: Paths to the generated files (trees_csv, resource_vtk, terrain_vtk, combined_vtk)
    """
    # Load and process baseline densities
    baselineDensities = pd.read_csv('data/csvs/tree-baseline-density.csv')
    baselineDensities = baselineDensities.rename(columns={'Size': 'diameter_breast_height'})
    print('\nBaseline Densities:')
    print(baselineDensities.to_string(index=False))

    # Load site data
    print(f'\nProcessing site: {site} with voxel size: {voxel_size}m')

    input_folder = f'data/revised/final/{site}'
    filepath = f'{input_folder}/{site}_{voxel_size}_subsetForScenarios.nc'
    xarray_dataset = xr.open_dataset(filepath)

    terrain_polydata_path= f"data/revised/final/{site}-roadVoxels-coloured.vtk"
    terrain_polydata = pv.read(terrain_polydata_path)

    print(f'Loaded xarray dataset from: {filepath}')

    # Calculate area
    bounds = xarray_dataset.attrs['bounds']
    Xmin, Xmax, Ymin, Ymax = bounds[:4]
    area = ((Xmax - Xmin) * (Ymax - Ymin)) / 10000
    print(f'\nSite bounds: Xmin={Xmin:.1f}, Xmax={Xmax:.1f}, Ymin={Ymin:.1f}, Ymax={Ymax:.1f}')
    print(f'Site area: {area:.2f} hectares')

    # Generate baseline trees
    print('\nGenerating baseline trees...')
    baseline_tree_df = pd.DataFrame()

    total_trees = 0
    for index, row in baselineDensities.iterrows():
        diameter_breast_height = row['diameter_breast_height']
        flat_density = row['Flat']
        num_points = int((area * flat_density) / 0.1)
        total_trees += num_points
        print(f'DBH {diameter_breast_height}cm: {num_points} trees ({flat_density:.1f} trees/ha)')
        
        # Create new rows for this DBH class
        new_rows = pd.DataFrame({
            'x': [-1] * num_points,
            'y': [-1] * num_points,
            'z': [-1] * num_points,
            'voxelID': [-1] * num_points,
            'voxel_I': [-1] * num_points,
            'voxel_J': [-1] * num_points,
            'voxel_K': [-1] * num_points,
            'precolonial': True,
            'control': 'reserve-tree',
            'diameter_breast_height': diameter_breast_height,
            'tree_id': [-1] * num_points,
            'useful_life_expectancy': -1
        })
        
        baseline_tree_df = pd.concat([baseline_tree_df, new_rows], ignore_index=True)

    print(f'\nTotal baseline trees generated: {len(baseline_tree_df)}')

    # Size categorization
    print('\nCategorizing trees by size...')
    baseline_tree_df['size'] = 'small'  # default value
    baseline_tree_df.loc[baseline_tree_df['diameter_breast_height'] >= 50, 'size'] = 'medium'
    baseline_tree_df.loc[baseline_tree_df['diameter_breast_height'] >= 80, 'size'] = 'large'

    print('\nInitial size distribution:')
    size_counts = baseline_tree_df['size'].value_counts()
    for size, count in size_counts.items():
        print(f'{size}: {count} trees ({count/len(baseline_tree_df)*100:.1f}%)')

    # Handle senescing trees and their states
    print('\nProcessing senescing trees...')
    # First mark half of large trees as senescings
    large_trees = baseline_tree_df['size'] == 'large'
    num_large = large_trees.sum()
    print(f'Found {num_large} large trees, converting half to senescing')

    # Create array of indices for large trees
    large_tree_indices = baseline_tree_df[large_trees].index.values
    # Randomly select half of these indices
    np.random.seed(42)  # for reproducibility
    senescing_indices = np.random.choice(large_tree_indices, size=num_large//2, replace=False)

    # Mark selected trees as senescing
    baseline_tree_df.loc[senescing_indices, 'size'] = 'senescing'

    # Get all senescing trees to create variants
    senescing_trees = baseline_tree_df[baseline_tree_df['size'] == 'senescing'].copy()
    num_senescing = len(senescing_trees)
    print(f'Creating {num_senescing} snags and {num_senescing} fallen trees')

    # Create snags (standing dead trees)
    snags = senescing_trees.copy()
    snags['size'] = 'snag'

    # Create fallen trees
    fallen = senescing_trees.copy()
    fallen['size'] = 'fallen'

    # Add snags and fallen trees to original dataframe
    baseline_tree_df = pd.concat([
        baseline_tree_df,  # Keep all original trees including senescing
        snags,            # Add snag variants
        fallen            # Add fallen variants
    ], ignore_index=True)

    #add 'tree_number', 'Node_ID', 'structure_id', which are all just the index
    baseline_tree_df['tree_number'] = baseline_tree_df.index
    baseline_tree_df['NodeID'] = baseline_tree_df.index
    baseline_tree_df['structureID'] = baseline_tree_df.index

    #initialsie ['rotateZ'] as a random rotation between 0 and 360 use a seed
    np.random.seed(42)
    baseline_tree_df['rotateZ'] = np.random.uniform(0, 360, len(baseline_tree_df))

    # Print final distribution
    print('\nFinal size distribution:')
    size_counts = baseline_tree_df['size'].value_counts()
    for size, count in size_counts.items():
        print(f'{size}: {count} trees ({count/len(baseline_tree_df)*100:.1f}%)')

    print('\nControl type distribution:')
    control_counts = baseline_tree_df['control'].value_counts()
    for control, count in control_counts.items():
        print(f'{control}: {count} trees ({count/len(baseline_tree_df)*100:.1f}%)')

    # Ground detection and position assignment
    print('\nDetecting ground voxels...')
    terrain_df = getGround(xarray_dataset, terrain_polydata)

    print('\nAssigning tree positions...')
    baseline_tree_df = getPositions(baseline_tree_df, terrain_df)

    print('\nCalculating useful life expectancy...')
    baseline_tree_df = calculate_useful_life_expectancy(baseline_tree_df)

    #resource assignments
    print('\nAssigning resources...')
    baseline_tree_df, resourceDF = a_resource_distributor_dataframes.process_all_trees(baseline_tree_df, voxel_size=voxel_size)
    print(f'resourceDF: {resourceDF}')


    resourceDF = a_resource_distributor_dataframes.rotate_resource_structures(baseline_tree_df, resourceDF)

    print('\nCreating resource polydata...')
    resourcePoly = a_resource_distributor_dataframes.convertToPoly(resourceDF)

    print('\nCreating resource polydata...')
    resourcePoly = a_resource_distributor_dataframes.convertToPoly(resourceDF)

    # Add the missing attributes to resourcePoly with 'forest_' prefix
    for column in ['tree_number', 'NodeID', 'structureID','diameter_breast_height']:
        if column in resourceDF.columns:
            forest_column = f'forest_{column}'
            resourcePoly.point_data[forest_column] = resourceDF[column].values
            print(f"Added missing column '{column}' as '{forest_column}' to resourcePoly point_data")

    print('\nCreating terrain polydata...')
    terrain_polydata = get_terrain_poly(terrain_df)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define output paths
    resource_vtk_path = f'{output_folder}/{site}_baseline_resources_{voxel_size}.vtk'
    trees_csv_path = f'{output_folder}/{site}_baseline_trees.csv'
    terrain_vtk_path = f'{output_folder}/{site}_baseline_terrain_{voxel_size}.vtk'
    combined_vtk_path = f'{output_folder}/{site}_baseline_combined_{voxel_size}.vtk'

    # Create combined polydata
    print('Creating combined polydata...')
    combinedPoly = combine_polydata(resourcePoly, terrain_polydata)

    # Save outputs
    print(f'Saving resource polydata to {resource_vtk_path}')
    resourcePoly.save(resource_vtk_path)
    
    print(f'Saving baseline trees to {trees_csv_path}')
    baseline_tree_df.to_csv(trees_csv_path, index=False)
    
    print(f'Saving terrain polydata to {terrain_vtk_path}')
    terrain_polydata.save(terrain_vtk_path)
    
    print(f'Saving combined polydata to {combined_vtk_path}')
    combinedPoly.save(combined_vtk_path)
    
    return trees_csv_path, resource_vtk_path, terrain_vtk_path, combined_vtk_path

def main():
    """
    Main function to run the baseline generation from command line.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate baseline trees and resources for a site.')
    parser.add_argument('site', type=str, help='Site name (e.g., trimmed-parade, city, uni)')
    parser.add_argument('--voxel-size', type=int, default=1, help='Voxel size for the simulation (default: 1)')
    parser.add_argument('--output-folder', type=str, default='data/revised/final/baselines', 
                        help='Folder to save output files (default: data/revised/final/baselines)')
    
    args = parser.parse_args()
    
    # Generate baseline
    generate_baseline(args.site, args.voxel_size, args.output_folder)

if __name__ == "__main__":
    main()