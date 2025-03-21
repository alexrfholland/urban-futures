import pandas as pd
import json
import os
import numpy as np
from pathlib import Path


    # 1. **precolonial**: A boolean value (True/False) indicating if the tree is precolonial
    # 2. **size**: String - 'small', 'medium', 'large', 'senescing', 'snag', 'fallen'
    # 3. **control**: String - 'reserve-tree', 'park-tree', 'street-tree', or 'improved-tree'

def generate_resources_dict(leroux_df, tree_conditions):
    # Create lists to store the data
    rows = []
    
    # Iterate over all conditions
    for (is_precolonial, size, control) in tree_conditions:
        # Apply the resource factor based on precolonial status
        base_resource_factor = 1 if is_precolonial else 0.05

        # Filter the data for the specific size
        mask = (leroux_df['Tree Size'] == size)
        filtered_df = leroux_df[mask]

        # Initialize resource values
        resources = {res: 0 for res in ['peeling bark', 'dead branch', 'fallen log', 'leaf litter', 'hollow', 'epiphyte']}

        # Calculate resource values from the filtered DataFrame
        for name in resources.keys():
            if name in filtered_df['name'].values:
                subset = filtered_df[filtered_df['name'] == name]
                min_val = subset[control].min()
                max_val = subset[control].max()

                # Reset resource factor for each resource
                current_resource_factor = base_resource_factor
                
                # Override for specific cases
                if name in ['dead branch', 'fallen log'] and is_precolonial == False:
                    current_resource_factor = 0.5  # have increased values for these resources even for elms
                    
                # Calculate the resource value
                resources[name] = ((min_val + max_val) / 2) * current_resource_factor

        # Add row to our data
        row = {
            'precolonial': is_precolonial,
            'size': size,
            'control': control,
            **resources  # unpack the resources dictionary
        }
        rows.append(row)

    # Create DataFrame from rows
    return pd.DataFrame(rows)


def add_improvement_rows(df):
    """
    For size == 'large' and precolonial == True:
        # Resources are the same values as returned by changing 'control' to 'reserve-tree'

    For (size == 'large' and precolonial == False) OR (size == 'medium' and precolonial == True):
            # - Dead branch: 20%
            # - Peeling bark: 5%
            # - Fallen log: 2
            # - Hollow: 2
            # - Epiphyte: 2 
            # - Leaf litter: copied from base tree

    For size == 'medium' and precolonial == False:
            # - Dead branch: 20%
            # - Peeling bark: 5%
            # - Fallen log: 1
            # - Hollow: 1
            # - Epiphyte: 1
            # - Leaf litter: copied from base tree
    """
    # Get resource columns
    resource_cols = [col for col in df.columns 
                    if col not in ['precolonial', 'size', 'control']]
    
    # Get reference values from reserve trees
    reserve_trees = df[df['control'] == 'reserve-tree'].copy()
    
    improved_rows = []
    
    # Get large precolonial reserve tree values (used as reference)
    large_reserve = reserve_trees[
        (reserve_trees['precolonial'] == True) & 
        (reserve_trees['size'] == 'large')
    ][resource_cols].iloc[0]
    
    # Case 1: Precolonial large trees - same as reserve tree
    improved_rows.append({
        'precolonial': True,
        'size': 'large',
        'control': 'improved-tree',
        **large_reserve
    })
    
    # Case 2: Precolonial medium trees - special modifications
    precolonial_medium_values = large_reserve.copy()
    # Set absolute values as per docstring
    precolonial_medium_values['dead branch'] = 20
    precolonial_medium_values['peeling bark'] = 5
    precolonial_medium_values['fallen log'] = 2
    precolonial_medium_values['hollow'] = 2
    precolonial_medium_values['epiphyte'] = 2
    # Keep leaf litter the same
    
    improved_rows.append({
        'precolonial': True,
        'size': 'medium',
        'control': 'improved-tree',
        **precolonial_medium_values
    })
    
    # Case 3: Non-precolonial large trees - special modifications
    nonprecolonial_large_values = large_reserve.copy()
    # Set absolute values as per docstring
    nonprecolonial_large_values['dead branch'] = 20
    nonprecolonial_large_values['peeling bark'] = 5
    nonprecolonial_large_values['fallen log'] = 2
    nonprecolonial_large_values['hollow'] = 2
    nonprecolonial_large_values['epiphyte'] = 2
    # Keep leaf litter the same
    
    improved_rows.append({
        'precolonial': False,
        'size': 'large',
        'control': 'improved-tree',
        **nonprecolonial_large_values
    })
    
    # Case 4: Non-precolonial medium trees - different values
    nonprecolonial_medium_values = pd.Series({col: 0 for col in resource_cols})
    # Set absolute values as per docstring
    nonprecolonial_medium_values['dead branch'] = 20
    nonprecolonial_medium_values['peeling bark'] = 5
    nonprecolonial_medium_values['fallen log'] = 1
    nonprecolonial_medium_values['hollow'] = 1
    nonprecolonial_medium_values['epiphyte'] = 1
    # Leaf litter remains 0
    
    improved_rows.append({
        'precolonial': False,
        'size': 'medium',
        'control': 'improved-tree',
        **nonprecolonial_medium_values
    })
    
    # Add the new rows to the original DataFrame
    return pd.concat([
        df,
        pd.DataFrame(improved_rows)
    ], ignore_index=True)

def add_senescing_rows(df):
    """
    For senescing trees:
        Creates three versions:
        1. Precolonial senescing improved-tree (based on precolonial large improved-tree)
        2. Non-precolonial senescing improved-tree (based on non-precolonial large improved-tree)
        3. Precolonial senescing reserve-tree (based on precolonial large reserve-tree)
        
        Modifications for all versions:
        # - Dead branch: +15% (max 100%)
        # - Peeling bark: +10% (max 100%)
        # - Fallen log: +2
        # - Hollow: +1
        # - Epiphyte: +1
        # - Leaf litter: copied from base tree
    """
    senescing_rows = []
    
    # First handle improved-tree based senescing versions
    improved_large = df[
        (df['control'] == 'improved-tree') & 
        (df['size'] == 'large')
    ].copy()
    
    # Create senescing version for both precolonial and non-precolonial improved trees
    for is_precolonial in [True, False]:
        base_values = improved_large[
            improved_large['precolonial'] == is_precolonial
        ].iloc[0].copy()
        
        # Apply senescing effects
        senescing_row = {
            'precolonial': is_precolonial,
            'size': 'senescing',
            'control': 'improved-tree',
            'dead branch': min(100, base_values['dead branch'] + 15),
            'peeling bark': min(100, base_values['peeling bark'] + 10),
            'fallen log': base_values['fallen log'] + 2,
            'hollow': base_values['hollow'] + 1,
            'epiphyte': base_values['epiphyte'] + 1,
            'leaf litter': base_values['leaf litter']  # copy leaf litter
        }
        
        senescing_rows.append(senescing_row)
    
    # Now handle reserve-tree based senescing version
    reserve_large = df[
        (df['control'] == 'reserve-tree') & 
        (df['size'] == 'large') &
        (df['precolonial'] == True)
    ].iloc[0].copy()
    
    senescing_row = {
        'precolonial': True,
        'size': 'senescing',
        'control': 'reserve-tree',
        'dead branch': min(100, reserve_large['dead branch'] + 15),
        'peeling bark': min(100, reserve_large['peeling bark'] + 10),
        'fallen log': reserve_large['fallen log'] + 2,
        'hollow': reserve_large['hollow'] + 1,
        'epiphyte': reserve_large['epiphyte'] + 1,
        'leaf litter': reserve_large['leaf litter']  # copy leaf litter
    }
    
    senescing_rows.append(senescing_row)
    
    # Add the new rows to the original DataFrame
    return pd.concat([
        df,
        pd.DataFrame(senescing_rows)
    ], ignore_index=True)

def add_snag_and_fallen_rows(df):
    """
    For snag trees:
        Creates a snag version of each senescing tree, with:
        # - Dead branch: 100%
        # - All other stats same as senescing
        # - Size changed to 'snag'
    
    For fallen trees:
        Creates a fallen version of each senescing tree, with:
        # - Dead branch: 100%
        # - All other stats: 0
        # - Size changed to 'fallen'
    """
    new_rows = []
    
    # Get all senescing trees
    senescing_trees = df[df['size'] == 'senescing'].copy()
    
    # Get resource columns
    resource_cols = [col for col in df.columns 
                    if col not in ['precolonial', 'size', 'control']]
    
    # Create snag and fallen versions for each senescing tree
    for _, base_tree in senescing_trees.iterrows():
        # Create snag version
        snag_row = base_tree.copy()
        snag_row['size'] = 'snag'
        snag_row['dead branch'] = 100
        new_rows.append(snag_row)
        
        # Create fallen version
        fallen_row = base_tree.copy()
        fallen_row['size'] = 'fallen'
        fallen_row['dead branch'] = 100
        # Set all other resources to 0
        for col in resource_cols:
            if col != 'dead branch':
                fallen_row[col] = 0
        new_rows.append(fallen_row)
    
    # Add the new rows to the original DataFrame
    return pd.concat([
        df,
        pd.DataFrame(new_rows)
    ], ignore_index=True)

if __name__ == "__main__":
    # Load the data
    leroux_df = pd.read_csv('data/csvs/lerouxdata-update.csv')

    # Define the parameters
    is_precolonial_list = [True, False]  # True: resourceFactor = 1, False: resourceFactor = 0.05
    size_list = ['small', 'medium', 'large']  # Tree sizes
    control_list = ['street-tree', 'park-tree', 'reserve-tree']  # Control categories

    # Generate the base dictionary without improvements
    base_tree_conditions = [(is_precolonial, size, control)
                            for is_precolonial in is_precolonial_list
                            for size in size_list
                            for control in control_list]
    
    
    # Generate the resources DataFrame
    tree_resources_df = generate_resources_dict(leroux_df, base_tree_conditions)

    tree_resources_df = add_improvement_rows(tree_resources_df)

    tree_resources_df = add_senescing_rows(tree_resources_df)

    tree_resources_df = add_snag_and_fallen_rows(tree_resources_df)

    output_dir = Path('data/revised/trees')

    tree_resources_df.to_csv(output_dir / 'resource_dicDF.csv', index=False)

    print(f"Resource dictionary has been saved to {output_dir / 'resource_dicDF.csv'}")

    print(tree_resources_df)

    