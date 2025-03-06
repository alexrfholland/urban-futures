import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def getRatios(showPlot=False):
    print('calculating ratios for voxels')
    # Load the files
    treeVoxelPath = 'modules/painter/data/tree_voxels.csv'
    #habitat_structures_path = 'modules/painter/data/habitat-structure-multipliers.csv'
    habitat_structures_path = 'modules/painter/data/updated-habitat-structure-multipliers.csv'



    treeVoxelStats = pd.read_csv(treeVoxelPath)
    habitatStructureInfo = pd.read_csv(habitat_structures_path)

    # Iterate through the 'resource' column in voxel_resource_stats and update habitat_structures
    for index, row in treeVoxelStats.iterrows():
        resource_name = row['resource']
        max_value = row['max']

        # Check if the resource exists as a column in habitat_structures
        if resource_name in habitatStructureInfo.columns:
            # Multiply all values in this column by the max value
            habitatStructureInfo[resource_name] *= max_value

    # Create a new row for 'large remnant tree' and set 'Action' to 'None'
    new_row = pd.DataFrame([{'Structure': 'Baseline.large remnant tree', 'Extent' : 20, 'Geometry approximation' : "Volume, large tree","Buffer space" : 1, "Enabled" : 0}])

    # Populate the new row with 'max' values from voxel_resource_stats
    for index, row in treeVoxelStats.iterrows():
        resource_name = row['resource']
        max_value = row['max']

        if resource_name in habitatStructureInfo.columns:
            new_row[resource_name] = max_value

    # Concatenate the new row to the DataFrame
    habitatStructureInfo = pd.concat([habitatStructureInfo, new_row], ignore_index=True)

    # Save the final updated habitat_structures DataFrame to a new CSV file
    final_updated_habitat_structures_path = 'modules/painter/data/habitat structure cubic meters.csv'
    habitatStructureInfo.to_csv(final_updated_habitat_structures_path, index=False)

    print(f'ratios for habitat structures are \n {habitatStructureInfo}')

    if(showPlot):
        plotStructures(habitatStructureInfo)

    return habitatStructureInfo

def plotStructures(habitatStructureInfo):
    # Plotting
    # Filter out non-finite values
    habitat_structures_long = pd.melt(habitatStructureInfo, 
                                    id_vars=['Structure', 'Extent'], 
                                    value_vars=['other', 'peeling bark', 'dead branch', 
                                                'epiphyte', 'leaf litter', 'perchable branch', 
                                                'hollow', 'fallen log'],
                                    var_name='Resource', value_name='Value').dropna()

    # Defining the jitter amount for both axes
    jitter_amount_x = .5
    jitter_amount_y = .5

    # Defining a larger offset for text annotations
    text_offset_x = 0.2
    text_offset_y = 0.2

    # Creating a color palette for the habitat structures
    palette = sns.color_palette("deep", len(habitatStructureInfo['Structure'].unique()))
    structure_colors = {structure: color for structure, color in zip(habitatStructureInfo['Structure'].unique(), palette)}

    plt.figure(figsize=(12, 8))



    # Plotting each resource type with right-justified annotations
    for resource in ['other', 'peeling bark', 'dead branch', 'epiphyte', 'leaf litter', 'perchable branch', 'hollow', 'fallen log']:
        # Define the marker as the first character of the resource name
        marker = 'per' if resource == 'perchable branch' else resource[0]
        subset = habitat_structures_long[habitat_structures_long['Resource'] == resource]

        # Adding jitter to the x and y coordinates
        jittered_x = subset['Extent'] + np.random.normal(0, jitter_amount_x, len(subset))
        jittered_y = subset['Value'] + np.random.normal(0, jitter_amount_y, len(subset))

        # Scatter plot with jittered dots and right-justified annotations
        for i in range(len(subset)):
            plt.scatter(jittered_x.iloc[i], jittered_y.iloc[i], 
                        color=structure_colors.get(subset['Structure'].iloc[i]), 
                        alpha=0.7, edgecolor='black')
            # Detailed annotation: first character of resource name followed by the structure name
            annotation_text = f'{marker}-{subset["Structure"].iloc[i]}'
            plt.text(jittered_x.iloc[i] + text_offset_x, jittered_y.iloc[i] + text_offset_y, 
                    annotation_text, ha='left', va='center', 
                    color=structure_colors.get(subset['Structure'].iloc[i]))

    plt.title('Scatter Plot of Habitat Structures by Resource and Cubic Meters (with Right-Justified Annotations)')
    plt.xlabel('Cubic Meters')
    plt.ylabel('Resource Value')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    getRatios(showPlot=True)

