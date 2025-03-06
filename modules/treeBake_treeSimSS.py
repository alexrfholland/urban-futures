import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Union
import pickle
import pyvista as pv
import cameraSetUpRevised, glyphs
import os


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_resources_dict(field_data: Dict[str, any], leroux_df: pd.DataFrame) -> Dict[str, float]:
    tree_size = field_data['_Tree_size'][0]
    control = field_data['_Control'][0]

    print(f'tree size is {tree_size} and control is {control}')
    print(leroux_df)

    leroux_df = leroux_df.rename(columns={control: 'quantity'})

    mask = (leroux_df['name'].isin(['peeling bark', 'dead branch', 'fallen log', 'leaf litter', 'hollow', 'epiphyte'])) & (leroux_df['Tree Size'] == tree_size)
    
    grouped = leroux_df[mask].groupby('name')['quantity']
    
    resourceFactor = 1
    if not field_data['isPrecolonial'][0]:
        #colonialFactor = .05
        colonialFactor = 1
        resourceFactor = colonialFactor
        print(f'non precolonial tree, reducing resources by {colonialFactor}')

    resources_dict = {}
    for name, (min_val, max_val) in grouped.agg(['min', 'max']).iterrows():
        print(f'max val for {name} resource is {max_val}')
        if name in ['dead branch']:
            #resources_dict[name] = np.random.uniform(min_val, max_val)
            resources_dict[name] = (max_val + min_val)/2
        else:
             #resources_dict[name] = np.random.uniform(min_val, max_val) * resourceFactor
            resources_dict[name] = (max_val + min_val)/2 * resourceFactor
    
    logging.info(f'resources for tree {tree_size} and {control} are:')
    logging.info(resources_dict)

    print(f'resource dict is {resources_dict}')

    return resources_dict

def update_tree_sample_attributes(tree_sample, resources_dict, tree_id, tree_size, control, precolonial):
    tree_sample = tree_sample.copy(deep=True)
    
    total_branches = len(tree_sample)

    target_num_peeling_bark = int(resources_dict['peeling bark'] * total_branches / 100)
    target_num_dead_branches = int(resources_dict['dead branch'] * total_branches / 100)

    live_branch_indices = tree_sample.index[tree_sample['Branch.type'] == 'live']
    dead_branch_indices = tree_sample.index[tree_sample['Branch.type'] == 'dead']

    if (tree_sample['Branch.type'] == 'dead').sum() < target_num_dead_branches:
        num_dead_branches_to_update = target_num_dead_branches - (tree_sample['Branch.type'] == 'dead').sum()
        dead_branch_indices_to_update = np.random.choice(live_branch_indices, num_dead_branches_to_update, replace=False)
        tree_sample.loc[dead_branch_indices_to_update, 'Branch.type'] = 'dead'
    elif (tree_sample['Branch.type'] == 'dead').sum() > target_num_dead_branches:
        num_live_branches_to_update = (tree_sample['Branch.type'] == 'dead').sum() - target_num_dead_branches
        live_branch_indices_to_update = np.random.choice(dead_branch_indices, num_live_branches_to_update, replace=False)
        tree_sample.loc[live_branch_indices_to_update, 'Branch.type'] = 'live'

    tree_sample['peelingBark'] = False
    non_peeling_bark_indices = tree_sample.index[tree_sample['peelingBark'] == False]
    num_peeling_bark_to_update = target_num_peeling_bark - tree_sample['peelingBark'].sum()
    peeling_bark_indices_to_update = np.random.choice(non_peeling_bark_indices, num_peeling_bark_to_update, replace=False)
    tree_sample.loc[peeling_bark_indices_to_update, 'peelingBark'] = True
    
    conditions = [
        tree_sample['Branch.type'] == 'dead',
        (tree_sample['Branch.angle'] <= 20) & (tree_sample['z'] > 10),
        tree_sample['peelingBark']
    ]
    choices = ['dead branch', 'perchable branch', 'peeling bark']
    tree_sample['resource'] = np.select(conditions, choices, default='other')

    return tree_sample[['Tree.ID', 'Tree.size', 'Branch.angle', 'Branch.length', 'DBH', 'transform', 'x', 'y', 'z', 'resource']]

def voxelize_branches(df, voxel_size):
    # Create a copy of the DataFrame
    voxelized_df = df.copy()
    
    # Voxelize the x, y, and z columns
    for col in ['x', 'y', 'z']:
        voxelized_df[col] = (voxelized_df[col] // voxel_size) * voxel_size
    
    # Remove duplicate rows
    voxelized_df = voxelized_df.drop_duplicates(subset=['x', 'y', 'z'])
    
    return voxelized_df

def addCanopy(tree_id, tree_template_df, voxel_size, resourceName):
    # Construct the file path
    file_path = f'data/treeInputs/leaves/{tree_id}.csv'
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Warning: Canopy file for tree_id {tree_id} not found.")
        return tree_template_df
    
    # Read the CSV file
    canopy_df = pd.read_csv(file_path)
    
    # Keep only x, y, and z columns
    canopy_df = canopy_df[['x', 'y', 'z']]
    
    # Voxelize the canopy
    voxelized_canopy = voxelize_branches(canopy_df, voxel_size)
    
    # Create new rows for the canopy points
    new_rows = []
    for _, row in voxelized_canopy.iterrows():
        new_row = tree_template_df.iloc[0].copy()
        new_row['x'], new_row['y'], new_row['z'] = row['x'], row['y'], row['z']
        new_row['resource'] = resourceName
        new_rows.append(new_row)
    
    # Create a DataFrame from the new rows and concatenate with the original DataFrame
    new_canopy_df = pd.DataFrame(new_rows)
    print(f'Number of leaf cluster rows: {len(new_canopy_df)}')
    updated_tree_template_df = pd.concat([tree_template_df, new_canopy_df], ignore_index=True)
    
    return updated_tree_template_df

def generate_tree_templates(branch_data, leroux_df):
    controls = ['street-tree', 'park-tree', 'reserve-tree']
    precolonial_states = [True, False]

    tree_templates = {}
    tree_level_resources = {}

    #go through each scanned tree...
    for tree_id in branch_data['Tree.ID'].unique():
        tree_branches = branch_data[branch_data['Tree.ID'] == tree_id]
        tree_size = tree_branches['Tree.size'].iloc[0]


        # Voxelize the tree branches
        voxel_size = 0.2
        tree_branches = voxelize_branches(tree_branches, voxel_size)


        #add a precolonial and colonial version...
        for precolonial in precolonial_states:
            
            #add control versions... 
            for control in controls:
                field_data = {
                    '_Tree_size': [tree_size],
                    '_Control': [control],
                    'isPrecolonial': [precolonial]
                }

                #generate a dictionary of resources
                resources_dict = generate_resources_dict(field_data, leroux_df)
                
                #update branch statistics based on resource dic
                updated_tree_sample = update_tree_sample_attributes(tree_branches, resources_dict, tree_id, tree_size, control, precolonial)

                #add ground resources
                #assume points cover 0.25 square meters
                updated_tree_sample = get_ground_resources(resources_dict, updated_tree_sample, 0.25)

                #add canopy resources
                updated_tree_sample = get_canopy_resources(resources_dict, updated_tree_sample)

                # Add canopy
                updated_tree_sample = addCanopy(tree_id, updated_tree_sample, 1, 'leaf cluster')
                
                state_key = (tree_size, precolonial, control, tree_id)

                #save the templates to dictionary
                tree_templates[state_key] = updated_tree_sample

                #save the tree-level resources to the dictionary
                tree_level_resources[state_key] = resources_dict

                 # Print resource counts for this tree
                resource_counts = updated_tree_sample['resource'].value_counts()
                print(f"\nResource counts for tree {tree_id} ({tree_size}, {'precolonial' if precolonial else 'colonial'}, {control}):")
                print(resource_counts.to_string())

            
    return tree_templates, tree_level_resources


def get_ground_resources(resources_dict, tree_template_df, point_area,):
    
    def generate_ground_cover_points(ground_resource_val: float, point_area: float, ground_cover_type: str, radius: float = 10.0) -> np.ndarray:
        num_points = -1
        
        # If leaf litter, value is a percentage
        if ground_cover_type == 'leaf litter':
            num_points = int((ground_resource_val / 100.0) * (radius**2 * np.pi) / point_area)
        
        # If log, value is a count of logs
        elif ground_cover_type == 'fallen log':
            num_points = int(ground_resource_val)
        
        # Generate points in a grid
        grid_size = int(np.ceil(2 * radius / point_area))
        x = np.linspace(-radius, radius, grid_size)
        y = np.linspace(-radius, radius, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Flatten and combine x and y coordinates
        points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())))
        
        # Filter points within the circle
        mask = np.sum(points[:, :2]**2, axis=1) <= radius**2
        points = points[mask]
        
        # Randomly select the required number of points
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        else:
            # If we don't have enough points, duplicate some randomly
            extra_points = num_points - len(points)
            extra_indices = np.random.choice(len(points), extra_points, replace=True)
            points = np.vstack((points, points[extra_indices]))
        
        # Snap points to voxel grid
        points[:, :2] = np.round(points[:, :2] / point_area) * point_area
        
        logging.info(f"Ground Cover Type: {ground_cover_type}")
        logging.info(f"Original Resource Val: {ground_resource_val} Number of Points: {num_points}, Total Area Covered: {num_points * point_area} m²")
        
        return points

    
    def generate_ground_cover_pointsOLD(ground_resource_val: float, point_area: float, ground_cover_type: str, radius: float = 10.0) -> np.ndarray:

        num_points = -1
        
        # If leaf litter, value is a percentage
        if ground_cover_type == 'leaf litter':
            num_points = int((ground_resource_val / 100.0) * (radius**2 * np.pi) / point_area)
        
        # If log, value is a count of logs
        elif ground_cover_type == 'fallen log':
            num_points = int(ground_resource_val)
        
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        r = np.random.uniform(0, radius, num_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(num_points)

        points = np.vstack((x, y, z)).T

        logging.info(f"Ground Cover Type: {ground_cover_type}")
        logging.info(f"Original Resource Val: {ground_resource_val} Number of Points: {num_points}, Total Area Covered: {num_points * point_area} m²")

        return points
    
    new_rows = []
    
    for ground_resource in ['fallen log', 'leaf litter']:
        
        #print(f'resource dic is {resources_dict}')
        ground_cover_points = generate_ground_cover_points(resources_dict[ground_resource], point_area, ground_resource)
        
        for voxel in ground_cover_points:
            new_row = tree_template_df.iloc[0].copy()  # Copy the first row to get the structure
            new_row['x'], new_row['y'], new_row['z'] = voxel
            new_row['resource'] = ground_resource
            new_rows.append(new_row)
    
    # Convert the list of new rows to a DataFrame and concatenate with the original tree_template_df
    new_rows_df = pd.DataFrame(new_rows, columns=tree_template_df.columns)
    print(new_rows_df)
    tree_template_df = pd.concat([tree_template_df, new_rows_df], ignore_index=True)
    
    return tree_template_df


def get_canopy_resources(resources_dict, tree_template_df):
    import random

    for resource in ['hollow', 'epiphyte']:
        # Find all rows where resource column = 'other' and z > 10
        eligible_rows = tree_template_df[(tree_template_df['resource'] == 'other') & (tree_template_df['z'] > 10)]
        
        # Calculate the number of rows to select
        n = round(resources_dict[resource])
        
        # Randomly select n rows
        if n > 0:
            selected_indices = random.sample(eligible_rows.index.tolist(), min(n, len(eligible_rows)))
            
            # Update the resource column for selected rows
            tree_template_df.loc[selected_indices, 'resource'] = resource

    return tree_template_df


def create_multiblock_from_df(df):
    print(f'creating multiblock')

    def create_polydata(df):
        points = df[['x', 'y', 'z']].values
        polydata = pv.PolyData(points)

        for col in df.columns.difference(['x', 'y', 'z']):
            sanitized_data = []
            for val in df[col]:
                if isinstance(val, str):
                    sanitized_val = val.encode('ascii', 'ignore').decode()
                    sanitized_data.append(sanitized_val)
                else:
                    sanitized_data.append(val)
            polydata.point_data[col] = sanitized_data

        polydata.point_data['ScanZ'] = df['z']
        return polydata

    branches_df = df[df['resource'].isin(['perchable branch', 'peeling bark', 'dead branch'])]
    ground_resources_df = df[df['resource'].isin(['fallen log', 'leaf litter'])]
    print(f'ground resource df is {ground_resources_df}')
    canopy_resources_df = df[df['resource'].isin(['epiphyte', 'hollow'])]
    leaf_cluster_df = df[df['resource'].isin(['leaf cluster'])]

    print('creating branches polydata')
    branches_polydata = create_polydata(branches_df)

    print('creating ground polydata')
    ground_resources_polydata = create_polydata(ground_resources_df)

    print('creating canopy polydata')
    canopy_resources_polydata = create_polydata(canopy_resources_df)

    print('creating leaf clusters')
    leaf_clusters_polydata = create_polydata(leaf_cluster_df)
    print(f'leaf cluster has {leaf_clusters_polydata.n_points} points')

    multiblock = pv.MultiBlock()
    multiblock['branches'] = branches_polydata
    multiblock['ground resources'] = ground_resources_polydata
    multiblock['canopy resources'] = canopy_resources_polydata
    multiblock['green biovolume'] = leaf_clusters_polydata

    return multiblock


def plot_example_trees(tree_samples_with_voxels, tree_size, precolonial, tree_id):
    controls = ['reserve-tree', 'park-tree', 'street-tree']
    plotter = pv.Plotter(shape=(1, 3))

    for i, control in enumerate(controls):
        key = (tree_size, precolonial, control, tree_id)
        if key not in tree_samples_with_voxels:
            print(f"No tree found with key: {key}")
            continue

        df = tree_samples_with_voxels[key]
        
        # Create multiblock
        multiblock = create_multiblock_from_df(df)

        # Set the current subplot
        plotter.subplot(0, i)
        
        # Add trees to the current subplot
        glyphs.add_trees_to_plotter(plotter, multiblock, (0, 0, 0))
        
        # Optionally, add a title to the subplot
        plotter.add_title(f"{control}")

    # Set up the camera for all subplots
    cameraSetUpRevised.setup_camera(plotter, 50, 600)

    # Link views and show the plot
    plotter.link_views()
    plotter.show()

def generate_statistics(tree_samples_with_voxels):
    stats = []
    for key, df in tree_samples_with_voxels.items():
        resource_counts = df['resource'].value_counts().to_dict()
        stats.append({
            'key': key,
            'total_voxels': len(df),
            **resource_counts
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.set_index('key', inplace=True)
    
    print(stats_df)
    return stats_df

# Main execution
branch_data = pd.read_csv('data/csvs/branchPredictions - adjusted.csv')
leroux_df = pd.read_csv('data/csvs/lerouxdata-update.csv')

#for scanned trees, create versions with different control and precolonial versions, as well as the resource dictionary for other resources
tree_voxel_templates, tree_level_resources_dics = generate_tree_templates(branch_data, leroux_df)


# Generate statistics
stats_df = generate_statistics(tree_voxel_templates)

# Save the final result
with open('data/treeSim.pkl', 'wb') as f:
    pickle.dump(tree_voxel_templates, f)

    # Export statistics to CSV
stats_df.to_csv('outputs/tree_statistics.csv')
print('Statistics exported to outputs/tree_statistics.csv')


print('Tree simulation completed and saved to data/treeSim.pkl')



# Plot example trees
plot_example_trees(tree_voxel_templates, 'large', False, 13)

#plot_example_trees(tree_voxel_templates, 'large', False, 11)

#plot_example_trees(tree_samples_with_voxels, 'small', False, 1)
