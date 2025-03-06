import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Union
import pickle


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')






def generate_resources_dict(field_data: Dict[str, any], leroux_df: pd.DataFrame, return_minmax: bool = False) -> Dict[str, Union[float, Tuple[float, float]]]:
    
    tree_size = field_data['_Tree_size'][0]  # Changed back to '_Tree_size'
    control = field_data['_Control'][0]  # Changed back to '_Control'

    print(f'tree size is {tree_size} and control is {control}')
    print(leroux_df)

    # Renaming the column
    leroux_df = leroux_df.rename(columns={control: 'quantity'})

    mask = (leroux_df['name'].isin(['peeling bark', 'dead branch', 'fallen log', 'leaf litter', 'hollow', 'epiphyte'])) & (leroux_df['Tree Size'] == tree_size)
    
    grouped = leroux_df[mask].groupby('name')['quantity']
    
    resourceFactor = 1
    if not field_data['isPrecolonial'][0]:
        colonialFactor = .05
        resourceFactor = colonialFactor
        print(f'non precolonial tree, reducing resources by {colonialFactor}')

    #resources_dict = {name: np.random.uniform(min_val, max_val) * resourceFactor for name, (min_val, max_val) in grouped.agg(['min', 'max']).iterrows()}
    
    if return_minmax:
        # Return min, max, and a random value between min and max
        resources_dict = {
            name: (min_val, max_val, np.random.uniform(min_val, max_val) * resourceFactor)
            for name, (min_val, max_val) in grouped.agg(['min', 'max']).iterrows()
        }
    else:
        # Return a random value between min and max
        resources_dict = {
            name: np.random.uniform(min_val, max_val) * resourceFactor
            for name, (min_val, max_val) in grouped.agg(['min', 'max']).iterrows()
        }

    
    
    logging.info(f'resources for tree {tree_size} and {control} are:')
    logging.info(resources_dict)

    print(f'resource dict is {resources_dict}')

    return resources_dict


def update_tree_sample_attributes(tree_sample, resources_dict, tree_id, tree_size, control, precolonial):

    # Create a deep copy of the tree_sample DataFrame
    tree_sample = tree_sample.copy(deep=True)
    
    # Get total number of branches in this tree sample
    total_branches = len(tree_sample)

    # Determine current percentages of peeling bark and dead branches
    current_peeling_bark_percentage = tree_sample['peelingBark'].sum() / total_branches * 100
    current_dead_branch_percentage = (tree_sample['Branch.type'] == 'dead').sum() / total_branches * 100
    
    logging.info(f'Current percentages - Peeling Bark: {current_peeling_bark_percentage}, Dead Branch: {current_dead_branch_percentage}')

    # Determine the target number of peeling bark and dead branches based on resources_dict
    target_num_peeling_bark = int(resources_dict['peeling bark'] * total_branches / 100)
    target_num_dead_branches = int(resources_dict['dead branch'] * total_branches / 100)

    logging.info(f'Target numbers - Peeling Bark: {target_num_peeling_bark}, Dead Branch: {target_num_dead_branches}')

    # Update Branch.type attribute
    live_branch_indices = tree_sample.index[tree_sample['Branch.type'] == 'live']
    dead_branch_indices = tree_sample.index[tree_sample['Branch.type'] == 'dead']

    if current_dead_branch_percentage < resources_dict['dead branch']:
        num_dead_branches_to_update = target_num_dead_branches - (tree_sample['Branch.type'] == 'dead').sum()
        dead_branch_indices_to_update = np.random.choice(live_branch_indices, num_dead_branches_to_update, replace=False)
        tree_sample.loc[dead_branch_indices_to_update, 'Branch.type'] = 'dead'
    elif current_dead_branch_percentage > resources_dict['dead branch']:
        num_live_branches_to_update = (tree_sample['Branch.type'] == 'dead').sum() - target_num_dead_branches
        live_branch_indices_to_update = np.random.choice(dead_branch_indices, num_live_branches_to_update, replace=False)
        tree_sample.loc[live_branch_indices_to_update, 'Branch.type'] = 'live'

    # Update peelingBark attribute
    tree_sample['peelingBark'] = False
    non_peeling_bark_indices = tree_sample.index[tree_sample['peelingBark'] == False]
    num_peeling_bark_to_update = target_num_peeling_bark - tree_sample['peelingBark'].sum()
    peeling_bark_indices_to_update = np.random.choice(non_peeling_bark_indices, num_peeling_bark_to_update, replace=False)
    tree_sample.loc[peeling_bark_indices_to_update, 'peelingBark'] = True
    
    # Create/update 'resource' attribute
    conditions = [
        tree_sample['Branch.type'] == 'dead',
        (tree_sample['Branch.angle'] <= 20) & (tree_sample['z'] > 10),
        tree_sample['peelingBark']
    ]
    choices = ['dead branch', 'perchable branch', 'peeling bark']
    tree_sample['resource'] = np.select(conditions, choices, default='other')
    
    # Logging updated percentages
    new_peeling_bark_percentage = tree_sample['peelingBark'].sum() / total_branches * 100
    new_dead_branch_percentage = (tree_sample['Branch.type'] == 'dead').sum() / total_branches * 100
    logging.info(f'Updated percentages - Peeling Bark: {new_peeling_bark_percentage}, Dead Branch: {new_dead_branch_percentage}')

    # Final check and logging
    print("")
    actual_num_peeling_bark = (tree_sample['resource'] == 'peeling bark').sum()
    actual_peeling_bark_percentage = (actual_num_peeling_bark / total_branches) * 100  # assuming total_num_resources is defined

    actual_num_dead_branches = (tree_sample['resource'] == 'dead branch').sum()
    actual_dead_branch_percentage = (actual_num_dead_branches / total_branches) * 100  # assuming total_num_resources is defined

    actual_num_perchable_branches = (tree_sample['resource'] == 'perchable branch').sum()
    actual_perchable_branch_percentage = (actual_num_perchable_branches / total_branches) * 100  # assuming total_num_resources is defined

    logging.info(f'Final check for Tree ID {tree_id}, Size: {tree_size}, Control: {control}, Precolonial: {precolonial}')
    logging.info(f'Peeling Bark - Actual: {actual_num_peeling_bark} ({actual_peeling_bark_percentage:.2f}%), Expected: {target_num_peeling_bark} ({resources_dict["peeling bark"]:.2f}%)')  # Update expected_peeling_bark_percentage accordingly
    logging.info(f'Dead Branch - Actual: {actual_num_dead_branches} ({actual_dead_branch_percentage:.2f}%), Expected: {target_num_dead_branches} ({resources_dict["dead branch"]:.2f}%)')  # Update expected_dead_branch_percentage accordingly
    logging.info(f'Perchable Branch - Actual: {actual_num_perchable_branches} ({actual_perchable_branch_percentage:.2f}%)')




    return tree_sample
def prepare_categorized_tree_samples(branch_data, leroux_df):
    # Define the possible states
    controls = ['street-tree', 'park-tree', 'reserve-tree']
    precolonial_states = [True, False]

    # Prepare dictionaries to hold the updated tree samples and resources data
    categorized_tree_samples = {}
    random_resources_data = {}
    tree_level_resource_dict = {}

    # Loop through each unique tree
    for idx, tree_id in enumerate(branch_data['Tree.ID'].unique()):
        # Separate the branches for this tree
        tree_branches = branch_data[branch_data['Tree.ID'] == tree_id]
        # Get the tree size for this tree
        tree_size = tree_branches['Tree.size'].iloc[0]

        # Iterate through each combination of precolonial and control states
        for precolonial in precolonial_states:
            for control in controls:
                tree_level_state_key = (tree_size, precolonial, control)

                # Set up the field_data with the current state combination
                field_data = {
                    '_Tree_size': [tree_size],  # Keeping the key as '_Tree_size'
                    '_Control': [control],  # Keeping the key as '_Control'
                    'isPrecolonial': [precolonial]
                }

                if tree_level_state_key not in tree_level_resource_dict:
                    # Call generate_resources_dict to get the resources dictionary for this state combination
                    minmax_resources_dict = generate_resources_dict(field_data, leroux_df, return_minmax=True)
                    # Store the minmax_resources_dict in the tree_level_resource_dict dictionary using the state_key
                    tree_level_resource_dict[tree_level_state_key] = minmax_resources_dict
                
                # Generate random resources dictionary for this state combination
                random_resources_dict = generate_resources_dict(field_data, leroux_df)

                # Store the resources dictionaries using the state_key
                state_key = (tree_size, precolonial, control, tree_id)
                random_resources_data[state_key] = random_resources_dict

                # Update the attributes for this tree sample
                updated_tree_sample = update_tree_sample_attributes(tree_branches, random_resources_dict, tree_id, tree_size, control, precolonial)

                # Store the updated tree sample using the state_key
                categorized_tree_samples[state_key] = updated_tree_sample

    # Return all three dictionaries
    return categorized_tree_samples, tree_level_resource_dict


# Assume branch_data and leroux_df are already loaded

branch_data = pd.read_csv('data/csvs/branchPredictions - adjusted.csv')
leroux_df = pd.read_csv('data/csvs/lerouxdata-update.csv')

outputs = prepare_categorized_tree_samples(branch_data, leroux_df)
# Call the function to prepare the categorized tree samples
categorized_tree_samples = outputs[0]

def check_datatype(data_dict):
    for key, df in data_dict.items():
        print(f"Checking data types for key: {key}")
        for column in df.columns:
            print(f" - Column '{column}' has data type: {df[column].dtype}")

#check_datatype(categorized_tree_samples)

def generate_tree_provider_stats_csv(categorized_tree_samples: Dict[Tuple, pd.DataFrame]) -> None:
    # Prepare a list to hold the data rows for the output CSV
    output_data = []

    # Iterate through the keys and dataframes in the categorized_tree_samples dictionary
    for (tree_size, precolonial, control, tree_id), tree_sample in categorized_tree_samples.items():

        
        # Get total number of branches in this tree sample
        total_branches = len(tree_sample)
        
        # Get counts of each resource type
        resource_counts = tree_sample['resource'].value_counts().to_dict()
        
        # Prepare a dictionary to hold the data for this row
        row_data = {
            'isPreColonial': precolonial,
            'size': tree_size,
            'control': control,
            'total_branches': total_branches,
            'dead_branch_count': resource_counts.get('dead branch', 0),
            'peeling_bark_count': resource_counts.get('peeling bark', 0),
            'perchable_branch_count': resource_counts.get('perchable branch', 0),
            'other_count': resource_counts.get('other', 0)
        }

        print(f'Tree ID: {tree_size, precolonial, control, tree_id}, Row Data: {row_data}')  # Debugging line right before appending row_data to output_data


        # Append this row data to the output data list
        output_data.append(row_data)

    # Convert the output data list to a DataFrame
    output_df = pd.DataFrame(output_data)

    # Output the DataFrame to a CSV file
    output_csv_path = 'outputs/treeProviderStats.csv'
    output_df.to_csv(output_csv_path, index=False)
    print(f'Data has been written to {output_csv_path}')
    
    

# Call the function to generate the CSV file
generate_tree_provider_stats_csv(categorized_tree_samples)



# Saving dictionary to a pickle file
with open('data/prebaked-branches.pkl', 'wb') as f:
    pickle.dump(categorized_tree_samples, f)



##resource dict json
import json


def save_resources_json(resources_data: dict, file_path: str) -> None:
    # Convert tuple keys to string keys with an underscore divider
    resources_data_str_keys = {'_'.join(map(str, key)): value for key, value in resources_data.items()}
    
    # Convert the resources_data dictionary to a JSON string
    resources_json = json.dumps(resources_data_str_keys, indent=4)
    
    # Write the JSON string to the specified file_path
    with open(file_path, 'w') as f:
        f.write(resources_json)
    print(f'Data has been written to {file_path}')


def generate_ground_cover_points(percentage: float, point_area: float, ground_cover_type: str, radius: float = 10.0) -> np.ndarray:
    """
    Generate points representing ground cover within a specified radius around the origin point (0,0,0).
    
    Parameters:
    - percentage: Expected percentage coverage of the ground cover
    - point_area: Area represented by a single point
    - ground_cover_type: Type of ground cover ('logs' or 'leaf litter')
    - radius: Radius around the tree to consider for generating points

    Returns:
    - points: Array of generated points with x, y, and z coordinates
    """

    # Calculate the number of points to generate
    num_points = int((percentage / 100.0) * (radius**2 * np.pi) / point_area)

    # Generate points within a circle around the origin (0,0,0)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.random.uniform(0, radius, num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(num_points)  # Setting z-coordinate to 0 for all points

    points = np.vstack((x, y, z)).T

    # Logging the data
    expected_area = (percentage / 100.0) * (radius**2 * np.pi)
    actual_area = num_points * point_area
    logging.info(f"Ground Cover Type: {ground_cover_type}")
    logging.info(f"Expected Area Percentage: {percentage}%")
    logging.info(f"Number of Points: {num_points}")
    logging.info(f"Total Area Covered by Points (assuming points are {point_area}m^2 each): {actual_area}m^2")
    logging.info(f"Actual Area Percentage: {actual_area / (radius**2 * np.pi) * 100}%")
    logging.info("---------")

    return points


def extend_resource_dict_with_ground_cover(tree_level_resource_dict, point_area):
    """
    Extend the tree level resource dictionary with ground cover points for fallen logs and leaf litter.

    Parameters:
    - tree_level_resource_dict: Dictionary containing resource data for each tree level state
    - point_area: Area represented by a single point

    Returns:
    - tree_level_resource_dict: Extended resource dictionary with ground cover points
    """
    for state_key, resources_dict in tree_level_resource_dict.items():
        for ground_cover_type in ['fallen log', 'leaf litter']:
            # Generate ground cover points
            ground_cover_points = generate_ground_cover_points(resources_dict[ground_cover_type][2], 
                                                               point_area, 
                                                               ground_cover_type)

            # Replace the percentage value with the generated points in the resources dictionary
            resources_dict[ground_cover_type] = ground_cover_points

            # Update the tree level resource dictionary with the extended resources dictionary
            tree_level_resource_dict[state_key] = resources_dict

    return tree_level_resource_dict

# Usage:

# Call the function to generate the resources data
tree_level_resource_dict = outputs[1]

# Get the ground cover details
# Assuming each point represents 0.25m^2
point_area = 0.25
extended_tree_level_resource_dict = extend_resource_dict_with_ground_cover(tree_level_resource_dict, point_area)


# Print the resources_data dictionary for debugging
print(tree_level_resource_dict)

def save_resources_pickle(resources_data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(resources_data, file)

# Usage:
save_resources_pickle(tree_level_resource_dict, 'data/prebaked-tree-resources.pkl')


# Call the function to save the resources data to a JSON file
#save_resources_json(tree_level_resource_dict, 'data/csvs/tree-level-resources.json')

print('done')