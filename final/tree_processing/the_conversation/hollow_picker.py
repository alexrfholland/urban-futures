import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
import os

def main():
    # File paths
    tree_vtk = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/stanislav/Tree 13 - Trunk.vtk'
    hollow_vtp = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/stanislav/tree 13- hollow locations.vtp'
    
    # Load the tree trunk
    print(f"Loading tree trunk from: {tree_vtk}")
    tree_trunk = pv.read(tree_vtk)
    print(f"Loaded {tree_trunk.n_points:,} points")
    
    # Initialize hollow array with 'N' (no hollow)
    hollow_array = np.full(tree_trunk.n_points, 'N', dtype='<U1')
    
    # Load hollow locations
    print(f"\nLoading hollow locations from: {hollow_vtp}")
    hollow_locations = pv.read(hollow_vtp)
    print(f"Loaded {hollow_locations.n_points} hollow locations")
    
    # Create KDTree for efficient spatial queries
    print("\nBuilding KDTree for efficient spatial queries...")
    tree_points = np.array(tree_trunk.points)
    kdtree = cKDTree(tree_points)
    
    # Define radius mapping for each hollow size
    radius_mapping = {
        'S': {'radius': 0.25},
        'M': {'radius': 0.35},
        'L': {'radius': 0.7}
    }
    
    # Get hollow center points and names
    hollow_centers = np.array(hollow_locations.points)
    hollow_names = hollow_locations.point_data['Name']
    
    # Group hollow centers by size for efficient processing
    size_groups = {}
    for size in ['S', 'M', 'L']:
        mask = hollow_names == size
        size_groups[size] = hollow_centers[mask]
        print(f"Found {np.sum(mask)} hollows of size '{size}'")
    
    # Process each size group using masks
    print("\nProcessing hollows...")
    for size, config in radius_mapping.items():
        if size not in size_groups or len(size_groups[size]) == 0:
            continue
            
        print(f"\nProcessing size '{size}' (radius={config['radius']})...")
        centers = size_groups[size]
        
        # Create mask for all points affected by this size
        size_mask = np.zeros(tree_trunk.n_points, dtype=bool)
        
        # Find all points within radius for all centers
        for i, center in enumerate(centers):
            # Query points within radius
            indices = kdtree.query_ball_point(center, config['radius'])
            
            # Update mask
            if indices:
                size_mask[indices] = True
                print(f"  Hollow {i+1}/{len(centers)}: Found {len(indices)} points")
        
        # Apply mask to update hollow array
        hollow_array[size_mask] = size
        print(f"  Total points marked as '{size}': {np.sum(size_mask):,}")
    
    # Assign the hollow array to point_data
    tree_trunk.point_data['hollows'] = hollow_array
    
    # Summary statistics
    unique, counts = np.unique(hollow_array, return_counts=True)
    print("\nHollow assignment summary:")
    for label, count in zip(unique, counts):
        label_name = {
            'N': 'No hollow',
            'S': 'Small hollow',
            'M': 'Medium hollow',
            'L': 'Large hollow'
        }.get(label, label)
        print(f"  {label_name} ('{label}'): {count:,} points")
    
    # Save the output
    output_dir = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/stanislav'
    output_file = os.path.join(output_dir, 'Tree 13 - HollowsOnTrees.vtk')
    
    print(f"\nSaving output to: {output_file}")
    tree_trunk.save(output_file)
    print("Done!")

if __name__ == "__main__":
    main() 