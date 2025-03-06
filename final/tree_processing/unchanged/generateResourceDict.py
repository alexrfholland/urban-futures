import pandas as pd
import json
import os


# ------------------------------------------------------------------------------------------------------------
# Structure and Keys of `tree_resources_dict`:
# ------------------------------------------------------------------------------------------------------------
# The `tree_resources_dict` is a dictionary where:
# - The **keys** are tuples containing four elements:
#   1. **is_precolonial**: A boolean value (`True` or `False`) indicating whether the tree is precolonial.
#   2. **size**: A string representing the size of the tree (`'small'`, `'medium'`, or `'large'`).
#   3. **control**: A string representing the control category (`'street-tree'`, `'park-tree'`, or `'reserve-tree'`).
#   4. **improvement**: A boolean value (`True` or `False`) indicating whether the improvement logic has been applied.
# 
# - The **values** are dictionaries where:
#   - The keys are resource names (`'peeling bark'`, `'dead branch'`, `'fallen log'`, `'leaf litter'`, `'hollow'`, `'epiphyte'`).
#   - The values are the computed resource counts for that specific tree configuration.
# ------------------------------------------------------------------------------------------------------------

def apply_senescing_logic(tree_resources_dict):
    print("\nApplying senescing logic to tree resources...")
    
    # Define the keys for the trees we want to modify
    improved_elm_key = (False, 'large', 'reserve-tree', True)
    reserve_euc_key = (True, 'large', 'reserve-tree', False)
    
    print(f"Processing improved elm key: {improved_elm_key}")
    print(f"Processing reserve eucalypt key: {reserve_euc_key}")
    
    # Create new senescing keys
    senescing_improved_elm_key = (False, 'senescing', 'reserve-tree', True) #saving as 'reserve-tree' for now
    senescing_reserve_euc_key = (True, 'senescing', 'reserve-tree', False) 
    
    for source_key, target_key in [(improved_elm_key, senescing_improved_elm_key), 
                                  (reserve_euc_key, senescing_reserve_euc_key)]:
        print(f"\nProcessing source key: {source_key}")
        if source_key in tree_resources_dict:
            resources = tree_resources_dict[source_key].copy()
            print("Original resource values:")
            for resource, value in resources.items():
                print(f"  {resource}: {value}")
            
            # Apply senescing effects
            print("\nApplying senescing effects...")
            
            # Dead branch: +30% (max 100%)
            old_dead = resources['dead branch']
            resources['dead branch'] = min(100, resources['dead branch'] + 30)
            print(f"  Dead branch: {old_dead} -> {resources['dead branch']}")
            
            # Peeling bark: +20% (max 100%)
            old_bark = resources['peeling bark']
            resources['peeling bark'] = min(100, resources['peeling bark'] + 20)
            print(f"  Peeling bark: {old_bark} -> {resources['peeling bark']}")
            
            # Fallen log: +2
            old_log = resources['fallen log']
            resources['fallen log'] = resources['fallen log'] + 2
            print(f"  Fallen log: {old_log} -> {resources['fallen log']}")
            
            # Hollow: +1
            old_hollow = resources['hollow']
            resources['hollow'] = resources['hollow'] + 1
            print(f"  Hollow: {old_hollow} -> {resources['hollow']}")
            
            # Epiphyte: +1
            old_epiphyte = resources['epiphyte']
            resources['epiphyte'] = resources['epiphyte'] + 1
            print(f"  Epiphyte: {old_epiphyte} -> {resources['epiphyte']}")
            
            # Add the new senescing entry to the dictionary
            tree_resources_dict[target_key] = resources
            print(f"Added senescing resources for target key: {target_key}")
        else:
            print(f"Warning: Source key {source_key} not found in tree_resources_dict")

    return tree_resources_dict


def generate_resources_dict(leroux_df, tree_conditions):
    # Dictionary to hold the resources for each tree configuration
    tree_resources = {}
    
    # Iterate over all conditions
    for (is_precolonial, size, control, improvement) in tree_conditions:
        # Apply the resource factor based on precolonial status
        resource_factor = 1 if is_precolonial else 0.05

        # Filter the data for the specific size
        mask = (leroux_df['Tree Size'] == size)
        filtered_df = leroux_df[mask]

        # Initialize resource counts
        resources_dict = {res: 0 for res in ['peeling bark', 'dead branch', 'fallen log', 'leaf litter', 'hollow', 'epiphyte']}

        # Calculate resource values from the filtered DataFrame
        for name in resources_dict.keys():
            if name in filtered_df['name'].values:
                subset = filtered_df[filtered_df['name'] == name]
                min_val = subset[control].min()
                max_val = subset[control].max()

                if name in ['dead branch', 'fallen log'] or control == 'reserve-tree':
                    resource_factor = 1 #have full values even for elms
                # Calculate the resource value as average of min and max, apply resource factor
                resources_dict[name] = ((min_val + max_val) / 2) * resource_factor

        # Store the result in the dictionary with the configuration tuple as the key
        tree_resources[(is_precolonial, size, control, improvement)] = resources_dict

    return tree_resources

def apply_improvement_logic(tree_resources_dict, is_precolonial_list, size_list, control_list):
    # Apply the improvement logic and update the dictionary
    for (is_precolonial, size, control) in [(is_precolonial, size, control) 
                                            for is_precolonial in is_precolonial_list
                                            for size in size_list
                                            for control in control_list]:
        
        if is_precolonial:
            # For precolonial trees with improvements
            if size == 'large':
                # Improvement condition:
                # Resources are the same values as returned by changing 'control' to 'reserve-tree'
                improved_key = (is_precolonial, size, 'reserve-tree', False)
                tree_resources_dict[(is_precolonial, size, control, True)] = tree_resources_dict.get(improved_key, {})
            
            elif size == 'medium':
                # Improvement condition:
                # Resources are the larger number between:
                #   - Same values as returned by changing 'control' to 'reserve-tree'
                #   - Same values as returned by changing 'control' to 'reserve-tree' and 'size' to 'large', multiplied by 0.5
                reserve_key = (is_precolonial, size, 'reserve-tree', False)
                large_reserve_key = (is_precolonial, 'large', 'reserve-tree', False)

                reserve_resources = tree_resources_dict.get(reserve_key, {})
                large_reserve_resources = {k: v * 0.5 for k, v in tree_resources_dict.get(large_reserve_key, {}).items()}

                improved_resources = {k: max(reserve_resources.get(k, 0), large_reserve_resources.get(k, 0))
                                      for k in reserve_resources.keys()}
                tree_resources_dict[(is_precolonial, size, control, True)] = improved_resources
        
        else:
            # For non-precolonial trees with improvements
            # Improvement condition:
            # Resources are the same values as returned by changing 'control' to 'reserve-tree' and 'size' to 'large', multiplied by 0.5
            improved_key = (True, 'large', 'reserve-tree', False)
            improved_resources = {k: v * 0.5 for k, v in tree_resources_dict.get(improved_key, {}).items()}
            tree_resources_dict[(is_precolonial, size, control, True)] = improved_resources

def format_and_print_results(specific_resources):
    for key, resources in specific_resources.items():
        print(f"Resources for {key}:")
        if resources == 'Key not found':
            print(resources)
        else:
            for resource, value in resources.items():
                print(f"  {resource}: {value}")
        print()  # Add a blank line between different keys

def save_resources_to_json(tree_resources_dict, output_dir="data/treeOutputs", filename="tree_resources.json"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tuple keys to strings for JSON serialization
    tree_resources_dict_serializable = {str(key): value for key, value in tree_resources_dict.items()}
    
    # Construct the full path for the output file
    output_path = os.path.join(output_dir, filename)
    
    # Save the dictionary to a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(tree_resources_dict_serializable, json_file, indent=4)


if __name__ == "__main__":
    # Load the data
    leroux_df = pd.read_csv('data/csvs/lerouxdata-update.csv')

    # Define the parameters
    is_precolonial_list = [True, False]  # True: resourceFactor = 1, False: resourceFactor = 0.05
    size_list = ['small', 'medium', 'large']  # Tree sizes
    control_list = ['street-tree', 'park-tree', 'reserve-tree']  # Control categories
    improvement_list = [False]  # No improvement at first

    # Generate the base dictionary without improvements
    base_tree_conditions = [(is_precolonial, size, control, False)
                            for is_precolonial in is_precolonial_list
                            for size in size_list
                            for control in control_list]
    
    tree_resources_dict = generate_resources_dict(leroux_df, base_tree_conditions)

    # Apply the improvement logic
    apply_improvement_logic(tree_resources_dict, is_precolonial_list, size_list, control_list)

    #apply senescing logic
    apply_senescing_logic(tree_resources_dict)
    
    # Output the specific dictionaries for the given keys
    specific_keys = [
        (True, 'large', 'reserve-tree', False),
        (True, 'large', 'street-tree', False),
        (False, 'large', 'street-tree', False),
        (False, 'large', 'street-tree', True),
    ]
    
    specific_keys = [
        (True, 'large', 'reserve-tree', False),
        (True, 'medium', 'street-tree', False),
        (True, 'medium', 'street-tree', True)
        ]

    # Display results for the specific keys
    specific_resources_updated = {key: tree_resources_dict.get(key, 'Key not found') for key in specific_keys}
    
    # Format and print the results
    format_and_print_results(specific_resources_updated)

    # Save the entire resources dictionary to a JSON file
    save_resources_to_json(tree_resources_dict)

    # Print a message indicating that the file has been saved
    print(f"Resources have been saved to {os.path.join('data/treeOutputs', 'tree_resources.json')}")
