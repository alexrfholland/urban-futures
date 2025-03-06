import pyvista as pv
import os
import b_generate_rewilded_ground
import pandas as pd
import a_vtk_to_ply
import numpy as np
from scipy.spatial import cKDTree

def calculate_average_point_spacing(points):
    """Calculate average distance to nearest neighbor for each point"""
    tree = cKDTree(points)
    # Find distance to nearest neighbor (k=2 because the nearest point to itself is itself)
    distances, _ = tree.query(points, k=2)
    # Take the second column (distances to actual nearest neighbor, not self)
    nearest_neighbor_distances = distances[:, 1]
    return np.mean(nearest_neighbor_distances), np.std(nearest_neighbor_distances)

def interpolate_road_voxels(road_polydata):
    """
    Interpolate road voxels that have scale > 0.25 into finer resolution points.
    Returns a new PolyData with interpolated points.
    """
    print("\nStarting road voxel interpolation...")
    print(f"Original road points: {road_polydata.n_points}")
    
    # Calculate initial point spacing
    mean_dist, std_dist = calculate_average_point_spacing(road_polydata.points)
    print(f"Initial average point spacing: {mean_dist:.3f} ± {std_dist:.3f}")
    
    # Split into high res and low res polydata
    high_res_mask = road_polydata.point_data['scale'] <= 0.5
    low_res_mask = road_polydata.point_data['scale'] > 0.5
    
    print(f"Number of high res points: {np.sum(high_res_mask)}")
    print(f"Number of low res points: {np.sum(low_res_mask)}")
    
    if np.sum(high_res_mask) == 0 and np.sum(low_res_mask) == 0:
        print("WARNING: No points matched either mask!")
        return road_polydata
    
    # Create high res polydata
    high_res_road = pv.PolyData(road_polydata.points[high_res_mask])
    high_res_colors = road_polydata.point_data['colors'][high_res_mask]
    high_res_road.point_data['colors'] = high_res_colors
    high_res_road.point_data['scale'] = road_polydata.point_data['scale'][high_res_mask]
    
    if high_res_road.n_points > 0:
        mean_dist, std_dist = calculate_average_point_spacing(high_res_road.points)
        print(f"High res point spacing: {mean_dist:.3f} ± {std_dist:.3f}")
    
    # Create low res polydata
    low_res_road = pv.PolyData(road_polydata.points[low_res_mask])
    low_res_colors = road_polydata.point_data['colors'][low_res_mask]
    low_res_road.point_data['colors'] = low_res_colors
    low_res_road.point_data['scale'] = road_polydata.point_data['scale'][low_res_mask]
    
    if low_res_road.n_points > 0:
        mean_dist, std_dist = calculate_average_point_spacing(low_res_road.points)
        print(f"Low res point spacing: {mean_dist:.3f} ± {std_dist:.3f}")
    
    print(f"Split into:")
    print(f"High res points (scale <= 0.5): {high_res_road.n_points}")
    print(f"Low res points (scale > 0.5): {low_res_road.n_points}")
    
    if low_res_road.n_points > 0:
        print("\nInterpolating low res points...")
        # Create a 4x4 grid of points with 0.25m spacing
        x = np.array([-0.375, -0.125, 0.125, 0.375])
        y = np.array([-0.375, -0.125, 0.125, 0.375])
        xx, yy = np.meshgrid(x, y)
        offsets = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(16)))
        
        print(f"Using {len(offsets)} offset points to create grid:")
        for i, offset in enumerate(offsets):
            print(f"Offset {i}: ({offset[0]:.3f}, {offset[1]:.3f})")
        
        # Create all interpolated points at once using broadcasting
        interpolated_points = low_res_road.points[:, np.newaxis, :] + offsets
        interpolated_points = interpolated_points.reshape(-1, 3)
        print(f"Created {len(interpolated_points)} interpolated points")

        # Create interpolated data
        interpolated_road = pv.PolyData(interpolated_points)
        
        # Transfer and interpolate colors and scale
        interpolated_colors = np.repeat(low_res_colors, len(offsets), axis=0)
        interpolated_road.point_data['colors'] = interpolated_colors
        interpolated_road.point_data['scale'] = np.full(len(interpolated_points), 0.25)

        mean_dist, std_dist = calculate_average_point_spacing(interpolated_points)
        print(f"Interpolated point spacing: {mean_dist:.3f} ± {std_dist:.3f}")

        # Merge high res and interpolated points
        if high_res_road.n_points > 0:
            # Create a new PolyData with combined points
            combined_points = np.vstack([high_res_road.points, interpolated_road.points])
            combined_colors = np.vstack([high_res_colors, interpolated_colors])
            combined_scales = np.concatenate([
                high_res_road.point_data['scale'],
                interpolated_road.point_data['scale']
            ])
            
            result = pv.PolyData(combined_points)
            result.point_data['colors'] = combined_colors
            result.point_data['scale'] = combined_scales
        else:
            result = interpolated_road
            
        print("\nMerged results:")
        print(f"High res points: {high_res_road.n_points}")
        print(f"Interpolated points: {interpolated_road.n_points}")
        print(f"Total points after merge: {result.n_points}")
        
        mean_dist, std_dist = calculate_average_point_spacing(result.points)
        print(f"Final point spacing: {mean_dist:.3f} ± {std_dist:.3f}")
    else:
        result = high_res_road
        print("\nNo low res points to interpolate, returning high res only")

    print(f"\nFinal road points: {result.n_points}")
    print("Final scale distribution:", np.unique(result.point_data['scale'], return_counts=True))

    #result.plot(scalars='colors', cmap='viridis')
    
    return result

def extract_ground(site, voxel_size, years, filePATH):
    # Process each year
        for year in years:
            print(f"\nProcessing year {year}...")
            
            ground = b_generate_rewilded_ground.generate_rewilded_ground(
                site=site,
                voxel_size=voxel_size,
                year=year,
                noise_scale=1.0,
                max_height_variation=1,
                detail_scale=10.0,
                detail_amplitude=0.1
            )
            print("Generated rewilded ground")

            output_base = f'{filePATH}/scenes/{site}_{year}_rewilded.ply'
            print("Saving ground...")
            a_vtk_to_ply.export_polydata_to_ply(
                ground, 
                output_base, 
                attributesToTransfer=['scenario_rewilded', 'sim_Turns']
            )
    
def main():
    # Configuration
    #site = 'trimmed-parade'
    site = 'uni'
    voxel_size = 1
    years = [10, 30, 60, 180]  # or [10, 30, 60, 180]

    filePATH = f'data/revised/final/{site}'

    siteVoxels = pv.read(f"data/revised/{site}-siteVoxels-masked.vtk")
    siteVoxels.save(f'{filePATH}/{site}_buildings.ply', texture='colors')

    roadVoxels = pv.read(f"data/revised/{site}-roadVoxels-coloured.vtk")
    
    print(f"Starting scene extraction for site: {site}")
    
    # Create output directory
    os.makedirs(os.path.dirname(f'{filePATH}/scenes'), exist_ok=True)

    #print the bounds and centre of the siteVoxels
    print(f"SiteVoxels bounds: {siteVoxels.bounds}")
    print(f"SiteVoxels center: {siteVoxels.center}")

    highResRoad = interpolate_road_voxels(roadVoxels)
    output_base = f'{filePATH}/{site}_highResRoad.ply'
    highResRoad.save(output_base, texture='colors')

    print(f"High res road points: {highResRoad.n_points}")
    print(f"Saved high res road ply saved to to {filePATH}/{site}_highResRoad.ply")



    #extract_ground(site, voxel_size, years, filePATH)


if __name__ == "__main__":
    main()





"""    
    for scene in scenes.keys():
        print(f"\nProcessing scene {scene}...")
        
        #check if subfolder scene name exists
        if not os.path.exists(f'{filePATH}/scenes/{scene}'):
            os.makedirs(f'{filePATH}/scenes/{scene}')
            print(f"Created directory: {filePATH}/scenes/{scene}")

        # Get scene parameters
        scene_distance = scenes[scene][0]
        print(f"Extracting subset at distance: {scene_distance}")

        # Create boolean masks for points within bounds
        site_mask = (
            (siteVoxels.points[:, 0] >= scene_distance - distance) & 
            (siteVoxels.points[:, 0] <= scene_distance + distance) & 
            (siteVoxels.points[:, 1] >= -distance) & 
            (siteVoxels.points[:, 1] <= distance)
        )
        road_mask = (
            (roadVoxels.points[:, 0] >= scene_distance - distance) & 
            (roadVoxels.points[:, 0] <= scene_distance + distance) & 
            (roadVoxels.points[:, 1] >= -distance) & 
            (roadVoxels.points[:, 1] <= distance)
        )

        # Extract subsets using masks
        siteVoxelsSubset = pv.PolyData(siteVoxels.points[site_mask])
        roadVoxelsSubset = pv.PolyData(roadVoxels.points[road_mask])

        # Transfer essential point data to subsets
        siteVoxelsSubset.point_data['colors'] = siteVoxels.point_data['colors'][site_mask]
        roadVoxelsSubset.point_data['colors'] = roadVoxels.point_data['colors'][road_mask]
        roadVoxelsSubset.point_data['scale'] = roadVoxels.point_data['scale'][road_mask]

        # Interpolate road voxels
        roadVoxelsSubset = interpolate_road_voxels(roadVoxelsSubset)

        print(f"Extracted voxel subsets - Site: {siteVoxelsSubset.n_points} points, Road: {roadVoxelsSubset.n_points} points")
        

            #load tree csv
            treePath = f'{filePATH}/{site}_1_treeDF_{year}.csv'
            treesDF = pd.read_csv(treePath)
            print(f"Loaded {len(treesDF)} trees from CSV")

            # Filter trees within the scene bounds
            treesDF_subset = treesDF[
                (treesDF['x'] >= scene_distance - distance) &
                (treesDF['x'] <= scene_distance + distance) &
                (treesDF['y'] >= -distance) &
                (treesDF['y'] <= distance)
            ]
            print(f"Filtered to {len(treesDF_subset)} trees within scene bounds")

            # Create boolean mask for ground cells within bounds
            ground_mask = (
                (ground.cell_centers().points[:, 0] >= scene_distance - distance) & 
                (ground.cell_centers().points[:, 0] <= scene_distance + distance) & 
                (ground.cell_centers().points[:, 1] >= -distance) & 
                (ground.cell_centers().points[:, 1] <= distance)
            )

            # Extract ground subset using mask
            groundSubset = ground.extract_cells(ground_mask)

            # Transfer any cell data attributes from ground
            for key in ground.cell_data.keys():
                print(f"Transferring cell data attribute: {key}")
                groundSubset.cell_data[key] = ground.cell_data[key][ground_mask]

            #groundSubset.plot(scalars='sim_Turns')

            # Save outputs
            output_base = f'{filePATH}/scenes/{scene}/{site}'
            
            print("Saving scene components...")
            siteVoxelsSubset.save(f'{output_base}_siteVoxelsSubset.ply', texture='colors')
            roadVoxelsSubset.save(f'{output_base}_roadVoxelsSubset.ply', texture='colors')

          

            print("Saving ground...")
            a_vtk_to_ply.export_polydata_to_ply(
                groundSubset, 
                f'{output_base}_ground.ply', 
                attributesToTransfer=['scenario_rewilded', 'sim_Turns']
            )
            treesDF_subset.to_csv(f'{output_base}_trees_yr{year}.csv', index=False)
            
            print(f"Successfully saved all components for scene {scene}, year {year}")"""



