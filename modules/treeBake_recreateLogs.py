import pandas as pd
import os
import pickle
import json

def CreateLogLibrary(tree_templates):

    outputFolder = 'data/treeOutputs'

    keys = [
        (False, 10, 'senescing'), 
        (True, 11, 'senescing'),
        (False, 10, 'snag'), 
        (True, 11, 'snag'),
        (False, 10, 'fallen'), 
        (True, 11, 'fallen'),
        (False, 10, 'propped'), 
        (True, 11, 'propped'),
    ]

    # Initialize an empty DataFrame for the log library
    logLibrary = pd.DataFrame()
    logCount = 0

    print(tree_templates.keys())



    for key in keys:
        if key in tree_templates:
            print(f"Key {key} found, adding to log library")
            treeDf = tree_templates[key].copy()  # Use a copy to avoid modifying the original data
            treeDf['logNo'] = -1  # Initialize the 'logNo' column
            treeDf['logSize'] = -1  # Initialize the 'logSize' column

            print(treeDf.columns)

            # Group by 'log_id' column
            grouped = treeDf.groupby('log_id')

            # Process each group
            for log_id, group in grouped:
                logCount += 1

                print(f'log count is {logCount}')

                # Calculate the center point of X and Y columns
                center_x = group['X'].mean()
                center_y = group['Y'].mean()

                # Transform X and Y columns to move them to the origin (0,0)
                group['X'] = group['X'] - center_x
                group['Y'] = group['Y'] - center_y

                # Assign a unique global log ID
                group['logNo'] = logCount

                # Count the number of rows in the group
                num_points = group.shape[0]

                # Assign the 'logSize' based on the number of points
                if num_points < 1000:
                    group['logSize'] = 'small'
                elif 1000 <= num_points <= 7000:
                    group['logSize'] = 'medium'
                else:
                    group['logSize'] = 'large'

                # Add the transformed group to the log library DataFrame
                logLibrary = pd.concat([logLibrary, group], ignore_index=True)
        else:
            print(f"ERROR! Key {key} not found!")


    # Group by 'log_id' and count the occurrences
    log_id_counts = logLibrary.groupby('logNo').size().reset_index(name='count')

    # Print the results
    print("Logs and counts in log library:")
    print(log_id_counts)

    # Ensure the output folder exists
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # Pickle the log library to the output folder
    logLibraryPath = os.path.join(outputFolder, 'logLibrary.pkl')
    with open(logLibraryPath, 'wb') as file:
        pickle.dump(logLibrary, file)

    print(f"Log library saved to {logLibraryPath}")

    return logLibrary

import numpy as np
import pandas as pd
import math

def get_ground_resources(resources_dict, tree_template_df, logLibrary, key):
    tree_template_df = tree_template_df.copy()
    # Step 0: Remove all rows where 'resource' equals 'fallen log'
    tree_template_df = tree_template_df[tree_template_df['resource'] != 'fallen log']

    # Step 1: Calculate the number of logs to generate
    print(resources_dict)
    noLogs = round(resources_dict['fallen log'])
    if noLogs > 0:
        logs = []
        new_rows = []

        for _ in range(noLogs):
            # Step 2: Create start points (x, y) coords randomly in a circle of radius 10
            radius = 10
            angle = np.random.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(np.random.uniform(0, 1))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            start_point = np.array([x, y, 0])

            # Step 3: Create a direction vector, a random 2D vector in the x, y plane
            direction_angle = np.random.uniform(0, 2 * np.pi)
            direction_vector = np.array([np.cos(direction_angle), np.sin(direction_angle), 0.1])

            # Normalize the direction vector
            direction_vector /= np.linalg.norm(direction_vector)

            # Step 4: Determine the appropriate log size based on the 'Control' column
            control_value = tree_template_df['Control'].iloc[0]

            print(key)
            if key[2] == 'reserve-tree' or key[3] == True: #either reserve tree or improved tree
                logSize = 'medium'
            else:
                logSize = 'small'

            # Filter the log library by the selected log size
            included_logs = logLibrary[logLibrary['logSize'] == logSize]
            selected_logNo = np.random.choice(included_logs['logNo'].unique())
            selected_log = included_logs[included_logs['logNo'] == selected_logNo]

            print(f'Selected logNo: {selected_logNo} with size: {logSize}')

            # Extract X, Y, Z coordinates
            log_points = selected_log[['X', 'Y', 'Z']].values

            print(f'log {selected_logNo} chosen with {selected_log.shape[0]} points')

            # Step 5: Apply transformation to the log points (translation and rotation)
            rotation_matrix = np.eye(3)
            rotation_matrix[:2, :2] = [[np.cos(direction_angle), -np.sin(direction_angle)],
                                    [np.sin(direction_angle),  np.cos(direction_angle)]]
            
            transformed_points = (log_points @ rotation_matrix.T) + start_point

            logs.append(transformed_points)

            # Step 6: Add the transformed points to new rows
            log_resources = np.full(transformed_points.shape[0], 'fallen log')
            new_rows.append(np.column_stack((transformed_points, log_resources)))

        if logs:
            all_points = np.vstack(logs)
            print(f"Fallen Log: Number of Logs: {noLogs}, Points Generated: {len(all_points)}")
        else:
            all_points = np.array([])

        if new_rows:
            new_rows = np.vstack(new_rows)
            new_df = pd.DataFrame(new_rows, columns=['X', 'Y', 'Z', 'resource'])

            # Ensure X, Y, Z are numeric
            new_df[['X', 'Y', 'Z']] = new_df[['X', 'Y', 'Z']].apply(pd.to_numeric)

            # Copy the remaining columns from the template DataFrame
            for col in tree_template_df.columns:
                if col not in new_df.columns:
                    new_df[col] = tree_template_df[col].iloc[0]

            print(new_df)

        
            # Concatenate the new rows with the original DataFrame
            tree_template_df = pd.concat([tree_template_df, new_df], ignore_index=True)

    else:
        print("No logs to generate.")

    print(f'final tree_template being returned is {tree_template_df}')
    return tree_template_df

def ProcessLogs(tree_templates, logLibrary, all_tree_resources_dict):
    updatedTreeTemplates = {}

    print(all_tree_resources_dict.keys())

    keys = [(True, 'large', 'street-tree', False, 15)]

    for key in tree_templates.keys():
    #for key in keys: #test on just one
        # Construct the initial resourceKey
        resourceKey = (key[0], key[1], key[2], key[3])

        print(f"Processing resourceKey: {resourceKey}")

        # Get the corresponding DataFrame from tree_templates
        tree_template_df = tree_templates[key]

        # Get the resources_dict from all_tree_resources_dict using resourceKey
        resource_dict = all_tree_resources_dict.get(str(resourceKey), {})

        # Check if the resource_dict is empty
        if not resource_dict:
            print(f"Improvement resource dictionary is empty for key: {resourceKey}")

            # Construct a new key with True for the third element
            alternate_resourceKey = (key[0], key[1], key[2], False)
            print(f"Trying with non improved resourceKey: {alternate_resourceKey}")

            # Attempt to get the resource_dict again using the alternate key
            resource_dict = all_tree_resources_dict.get(str(alternate_resourceKey), {})

            if not resource_dict:
                print(f"Alternate resource dictionary is also empty for key: {alternate_resourceKey}")

        # Call get_ground_resources to get the updated DataFrame
        updatedTreeDF = get_ground_resources(resource_dict, tree_template_df, logLibrary, key)

        # Add the updated DataFrame to updatedTreeTemplates with the original key
        updatedTreeTemplates[key] = updatedTreeDF

        if(key == (True, 'large', 'park-tree', False, 11)):
            print("test (True, 'large', 'park-tree', False, 11)")
            print(updatedTreeDF)

    # Pickle the updated tree templates and save to 'data/treeOutputs/adjusted_tree_templates.pkl'
    outputPath = 'data/treeOutputs/adjusted_tree_templates.pkl'
    with open(outputPath, 'wb') as file:
        pickle.dump(updatedTreeTemplates, file)

    print(f"Updated tree templates saved to {outputPath}")

    return updatedTreeTemplates


def main():
    
    with open('data/treeOutputs/fallen_trees_dict.pkl', 'rb') as file:
        fallenTrees = pickle.load(file)

    with open('data/treeOutputs/tree_resources.json', 'r') as file:
        all_tree_resources_dict = json.load(file)

    with open('data/treeOutputs/tree_templates.pkl', 'rb') as file:
        tree_templates = pickle.load(file)

    # Call the CreateLogLibrary function
    logLibrary = CreateLogLibrary(fallenTrees)
    print("Log library created successfully.")

    print("Distributing logs")

    ProcessLogs(tree_templates, logLibrary, all_tree_resources_dict)

    

if __name__ == "__main__":
    main()