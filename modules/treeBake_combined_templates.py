import os
import pandas as pd
import numpy as np

# Paths to the input directories
input_elm_dfs_path = 'data/treeInputs/trunks-elms/initial templates'
input_euc_dfs_path = 'data/treeInputs/trunks/initial_templates'
canopy_dfs_path = 'data/treeInputs/leaves/'

def load_dfs(input_folder):
    """
    Load all CSV files from a folder into a dictionary with Tree.ID as the key.
    """
    dfs = {}
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path, delimiter=';')
            tree_id = df['Tree.ID'].iloc[0]  # Assuming Tree.ID is in the first row
            dfs[tree_id] = df
    return dfs

def load_canopy_dfs(canopy_folder):
    """
    Load canopy CSV files from a folder into a dictionary with Tree.ID as the key.
    """
    canopy_dfs = {}
    for file in os.listdir(canopy_folder):
        if file.endswith(".csv"):
            tree_id = int(file.replace('.csv', ''))
            file_path = os.path.join(canopy_folder, file)
            canopy_dfs[tree_id] = pd.read_csv(file_path)
    return canopy_dfs

# Load the dataframes for ELM, EUC, and Canopy datasets
elm_dfs = load_dfs(input_elm_dfs_path)
euc_dfs = load_dfs(input_euc_dfs_path)
canopy_dfs = load_canopy_dfs(canopy_dfs_path)

# Print column names for the first ELM and EUC dataframes
first_elm_key = next(iter(elm_dfs))
first_euc_key = next(iter(euc_dfs))

print(f"Columns in the first ELM dataframe (Tree.ID = {first_elm_key}):", elm_dfs[first_elm_key].columns)
print(f"Columns in the first EUC dataframe (Tree.ID = {first_euc_key}):", euc_dfs[first_euc_key].columns)

def transform_euc_df(df, canopy_df):
    """
    Apply transformation to the EUC dataframe using values from its corresponding canopy dataframe.
    """
    
    print(canopy_df)
    # Get transformation values from the first row of the canopy dataframe
    transform_x = canopy_df['transform_x'].iloc[0]
    transform_y = canopy_df['transform_y'].iloc[0]
    transform_z = canopy_df['transform_z'].iloc[0]

    # Add new columns for original coordinates and transformations
    df['origX'] = df['X']
    df['origY'] = df['Y']
    df['origZ'] = df['Z']

    df['transformX'] = transform_x
    df['transformY'] = transform_y
    df['transformZ'] = transform_z

    # Apply the transformation
    df['X'] = df['origX'] + transform_x
    df['Y'] = df['origY'] + transform_y
    df['Z'] = df['origZ'] + transform_z

    return df

####

def transform_elm_df(df):
    """
    Apply transformation to center the ELM tree's coordinates
    by moving the minimum Z point to (0,0,0) and centering X and Y.
    """
    # Calculate the centerpoint of X and Y
    centre_x = df['X'].mean()
    centre_y = df['Y'].mean()

    # Find the coordinates of the point with the minimum Z value
    lowest_z_point = df.loc[df['Z'].idxmin()]

    # Calculate the transformation to move this point to (0,0,0)
    transformX = -lowest_z_point['X']
    transformY = -lowest_z_point['Y']
    transformZ = -lowest_z_point['Z']

    df['transformX'] = transformX
    df['transformY'] = transformY
    df['transformZ'] = transformZ

    # Rename original coordinates
    df.rename(columns={'X': 'origX', 'Y': 'origY', 'Z': 'origZ'}, inplace=True)

    # Apply the transformation to the DataFrame
    df['X'] = df['origX'] + transformX - centre_x
    df['Y'] = df['origY'] + transformY - centre_y
    df['Z'] = df['origZ'] + transformZ

    return df



def transform_elm_df2(df):
    """
    Apply transformation to center the ELM tree's coordinates.
    """
    # Calculate the centrepoint of X, Y, and Z
    centre_x = df['X'].mean()
    centre_y = df['Y'].mean()

    # Find the minimum Z value
    min_z = df['Z'].min()

    # Calculate the transformation required to center X, Y, and set min Z to 0
    transform_x = -centre_x
    transform_y = -centre_y
    transform_z = -min_z

    # Add new columns for original coordinates and transformations
    df['origX'] = df['X']
    df['origY'] = df['Y']
    df['origZ'] = df['Z']

    df['transformX'] = transform_x
    df['transformY'] = transform_y
    df['transformZ'] = transform_z

    # Apply the transformation to center the tree
    df['X'] = df['X'] + transform_x
    df['Y'] = df['Y'] + transform_y
    df['Z'] = df['Z'] + transform_z

    return df


"""Cylinders:
- Cylinders are the smallest units of tree structure, representing individual segments extracted from point cloud data.
- 'cylinder': The ID of the cylinder, derived from 'branch_ID'.
- 'cylinderOrder': Represents the order of cylinders, with the lowest being the cylinder at the bottom of the main stem and the highest being the end cylinders of the tip branches.
- 'inverseCylinderOrder': The reverse order of 'cylinderOrder', with the lowest being the end cylinders of the tip branches and the highest being at the bottom of the main stem.
- 'cylinderOrderInBranch': The order of cylinders within each branch, where the lowest are at the beginning of branches and the highest are at the end.
- 'inverseCylinderOrderInBranch': The reverse order of 'cylinderOrderInBranch', with the lowest at the end of branches and the highest at the beginning.

Branches:
- Branches are formed by groups of connected cylinders, representing larger segments of the tree.
- 'branch': The ID of the branch, derived from 'segment_ID'.
- 'branchOrder': Indicates the order of branches, with the lowest being the main stem branch and the highest being the end tip branches.

Clusters:
- Clusters are higher-level aggregations of branches, typically representing significant substructures such as major offshoots from the main stem.
- 'cluster': The ID of the cluster, calculated by dividing the range of segment IDs into groups.
- 'cylinderOrderInCluster': The order of cylinders within each cluster, with the lowest connecting the cluster to the main stem and the highest at the end tip branches.
- 'inverseCylinderOrderInCluster': The reverse order of 'cylinderOrderInCluster', with the highest connecting the cluster to the main stem and the lowest at the end tip branches.
- 'branchOrderInCluster': The order of branches within each cluster, with the lowest connecting the cluster to the main stem and the highest at the end tip branches.
- 'inverseBranchOrderInCluster': The reverse order of 'branchOrderInCluster', with the highest connecting the cluster to the main stem and the lowest at the end tip branches.
- 'clusterOrder': The overall order of clusters, with the lowest representing the main stem.

Other Topological Properties:
- These properties capture additional hierarchical relationships within the tree structure beyond individual cylinders, branches, and clusters.
- 'cylinderOrderInBranch': The order of cylinders within each branch, from the base to the tip.
- 'inverseCylinderOrderInBranch': The reverse order of 'cylinderOrderInBranch'.
- 'branchOrderInCluster': The order of branches within a cluster, from the main stem to the tip branches.
- 'inverseBranchOrderInCluster': The reverse order of 'branchOrderInCluster'.
- 'clusterOrder': The overall hierarchical position of clusters, with the main stem being the lowest.
"""



def process_euc_df(voxel_properties_df):
    #####CYLINDERS#####
    #Individual lines extracted from the point cloud with structural properties

    #'cylinder'. ID of cylinder
    voxel_properties_df['cylinder'] = voxel_properties_df['branch_ID']

    #cylinderOrder.  Lowest is the cylinder at the bottom of the main stem, highest are the end cylinders of the end tip branches
    #inverseCylinderOrder. Lowest are the end cylinders of the end tip branches, highest is cylinder at the bottom of the main stem, highest a    
    #NOTE. In the eucs qsm data, 'branch order cum' is the inverse_branch_order. Ie, highest is the bottom of the main stem, lowest are the end cylinders of the end tip branches 
    voxel_properties_df['inverseCylinderOrder'] = voxel_properties_df['branch_order_cum']
    voxel_properties_df['cylinderOrder'] = voxel_properties_df['inverseCylinderOrder'].max() + 1 - voxel_properties_df['inverseCylinderOrder']

    
    #####BRANCHES#####
    #connecting cylinders form a group called a branch
    
    #'branch'. ID of the branch
    voxel_properties_df['branch'] = voxel_properties_df['segment_ID'] 
    
    #branchOrder'. The lowest is the main stem branch, the highest is the end tip branches
    voxel_properties_df['branchOrder'] = voxel_properties_df.groupby('branchOrder') #ie the lowest is the main stem branch, the highest is the end tip branches

    #cylinderOrderInBranch. The lowest are cylinders at the beginning of branches, the highest are cylinders at the end of the branches
    voxel_properties_df['cylinderOrderInBranch'] = voxel_properties_df.groupby('branch')['cylinderOrder'].rank(method='first').astype(int)    
    #cylinderOrderInBranch. The lowest are cylinders at the end of branches, the highest are cylinders at the beginning of the branches
    voxel_properties_df['inverseCylinderOrderInBranch'] = voxel_properties_df.groupby('branch')['cylinderOrderInBranch'].transform(lambda x: x.max() + 1 - x).astype(int)

    ####CLUSTERS####
    #Clusters are larger aggregations of branches with a common larger stem, ie. an offshoot of the main stem

    cluster_number = 20
    #'cluster' is the ID of the cluster
    voxel_properties_df = create_euc_clusters(voxel_properties_df, cluster_number)


    ###OTHER TOPOLOGICAL GROUPINGS####

    #'cylinderOrderInCluster'. Lowest are the cylinders connecting the cluster to the main sten. Highest are the cylinders at the end tip branches
    #'inverseCylinderOrderInCluster'. Highest are the cylinders connecting the cluster to the main sten. Lowest are the cylinders at the end tip branches  
    #'branchOrderInCluster'. Lowest are the branches connecting the cluster to the main sten. Highest are the branches at the end tip branches
    #'inverseBranchOrderInCluster'. Highest are the branches connecting the cluster to the main sten. Lowest are the branches at the end tip branches
    #'clusterOrder'. Lowest is the main stem.
    voxel_properties_df = calculate_topological_groupings(voxel_properties_df)

    return voxel_properties_df

# Define the create_clusters function. 
def create_euc_clusters(voxel_properties_df, cluster_number):
    
    
    # Determine the range of segment IDs and create groups based on the cluster number
    min_segment_id = voxel_properties_df['segment_ID'].min()
    max_segment_id = voxel_properties_df['segment_ID'].max()
    segment_id_range = max_segment_id - min_segment_id + 1
    
    # Create bins for the segment IDs based on the cluster number
    bins = np.linspace(min_segment_id, max_segment_id + 1, cluster_number + 1)
    voxel_properties_df['cluster'] = np.digitize(voxel_properties_df['segment_ID'], bins) - 1

    # Assign any rows with branchOrder = 0 to the first cluster
    voxel_properties_df.loc[voxel_properties_df['branchOrder'] == 0, 'cluster'] = 0

    return voxel_properties_df

def calculate_topological_groupings(voxel_properties_df):
    #####CYLINDERS#####
    
    #'cylinderOrderInCluster'. Lowest are the cylinders connecting the cluster to the main sten. Highest are the cylinders at the end tip branches
    voxel_properties_df['cylinderOrderInCluster'] = voxel_properties_df.groupby('cluster')['cylinderOrder'].rank(method='first').astype(int)

    #'inverseCylinderOrderInCluster'. Highest are the cylinders connecting the cluster to the main sten. Lowest are the cylinders at the end tip branches    
    voxel_properties_df['inverseCylinderOrderInCluster'] = voxel_properties_df.groupby('cluster')['cylinderOrderInCluster'].transform(lambda x: x.max() + 1 - x).astype(int)

    
    #####BRANCHES#####
    
    #branchOrderInCluster. Lowest are the branches connecting the cluster to the main sten. Highest are the branches at the end tip branches
    voxel_properties_df['branchOrderInCluster'] 

    #branchOrderInCluster. Highest are the branches connecting the cluster to the main sten. Lowest are the branches at the end tip branches
    voxel_properties_df['inverseBranchOrderInCluster'] 

    ####CLUSTERS####.  Larger aggregations of branches.

    # Calculate clusterOrder. Lowest is the main stem.
    voxel_properties_df['clusterOrder'] = voxel_properties_df.groupby('cluster')['branchOrder'].transform('min')

    return voxel_properties_df


def format_euc_df(voxel_properties_df):
        properties = [
            'cylinder',
            'cylinderOrder',
            'inverseCylinderOrder',
            'cylinderOrderInBranch',
            'inverseCylinderOrderInBranch',
            'branch',
            'branchOrder',
            'cluster',
            'cylinderOrderInCluster',
            'inverseCylinderOrderInCluster',
            'branchOrderInCluster',
            'inverseBranchOrderInCluster',
            'clusterOrder'
        ]

        #besides from X,Y,Z, resource, Tree.ID, isPrecolonial, radius, length, move all other columns to the end and append them in format f'_{column name}'.
        #return the df




#process elms
#for elms

#cylinder = 'cylinder_ID'
#cylinderOrder = ?
#cylinderOrderInBranch = 'PositionInBranch'

#branch = 'branch'
#branchOrder = 'BranchOrder'

#cluster = 'segment'
#clusterOrder = NONE YET

#cylinderOrderInCluster
#inverseCylinderOrderInCluster





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def convert_voxels_to_resource(voxel_properties_df, resource_dict, patchiness_level, exclude=None):
    # Initialize the 'resource' column with 'other'
    voxel_properties_df['resource'] = 'other'
    
    # Calculate the total number of voxels
    total_voxels = len(voxel_properties_df)
    
    # Determine the range of segment IDs and create groups based on the patchiness level
    min_segment_id = voxel_properties_df['segment_ID'].min()
    max_segment_id = voxel_properties_df['segment_ID'].max()
    segment_id_range = max_segment_id - min_segment_id + 1
    
    # Create bins for the segment IDs based on the patchiness level
    bins = np.linspace(min_segment_id, max_segment_id + 1, patchiness_level + 1)
    voxel_properties_df['segment_group'] = np.digitize(voxel_properties_df['segment_ID'], bins) - 1
    
    for resource, percentage in resource_dict.items():
        quantity_to_convert = int(np.ceil((percentage / 100) * total_voxels))
        converted_voxels = 0
        
        # Group by the newly created segment_group
        group_clusters = voxel_properties_df.groupby('segment_group')
        
        # Shuffle the order of the groups randomly
        group_ids = np.random.permutation(group_clusters.size().index.tolist())
        
        for group_id in group_ids:
            group = group_clusters.get_group(group_id)
            group_size = len(group)
            
            # Check exclusion criteria if provided
            if exclude:
                excluded = False
                for col, conditions in exclude.items():
                    if col in group.columns and group[col].isin(conditions).any():
                        excluded = True
                        print(f"Skipping group {group_id} due to exclusion condition on column '{col}' with conditions {conditions}.")
                        break
                if excluded:
                    continue
            
            # Proceed with conversion if no exclusions apply
            if group['resource'].eq('other').all():
                if converted_voxels + group_size <= quantity_to_convert:
                    voxel_properties_df.loc[group.index, 'resource'] = resource
                    converted_voxels += group_size
                    print(f"Converted entire group with segment_group {group_id}: {group_size} voxels")
                else:
                    remaining_voxels = quantity_to_convert - converted_voxels
                    sorted_group = group.sort_values(by='inverse_branch_order', ascending=True)
                    voxel_properties_df.loc[sorted_group.index[:remaining_voxels], 'resource'] = resource
                    converted_voxels += remaining_voxels
                    print(f"Partially converted group with segment_group {group_id}: {remaining_voxels} voxels")
                    break

        final_percentage = (converted_voxels / total_voxels) * 100
        print(f"Total converted voxels for {resource}: {converted_voxels} ({final_percentage:.2f}%)")

    return voxel_properties_df



def format_cylinder_dataframes(cylinder_dataframes):
    """Add cylinder order columns to each cylinder dataframe in the list."""
    for cylinder_filename, cylinder_df in cylinder_dataframes:
        # Initialize the new columns
        cylinder_df['cylinder_order_in_segment'] = 0
        cylinder_df['inverse_cylinder_order_in_segment'] = 0

        # Group by the segment to order cylinders within each segment
        for segment_id, group in cylinder_df.groupby('segment'):
            # Sort by branch and then by PositionInBranch (order within the branch)
            group = group.sort_values(by=['branch', 'PositionInBranch'])

            # Assign the cylinder_order_in_segment
            group['cylinder_order_in_segment'] = range(len(group))

            # Assign the inverse_cylinder_order_in_segment
            group['inverse_cylinder_order_in_segment'] = group['cylinder_order_in_segment'].max() - group['cylinder_order_in_segment']

            # Update the original dataframe with these new values
            cylinder_df.loc[group.index, 'cylinder_order_in_segment'] = group['cylinder_order_in_segment']
            cylinder_df.loc[group.index, 'inverse_cylinder_order_in_segment'] = group['inverse_cylinder_order_in_segment']
    return cylinder_dataframes


# Transform all ELM dataframes
for tree_id, df in elm_dfs.items():
    print(f"Transforming ELM dataframe with Tree.ID = {tree_id}")
    elm_dfs[tree_id] = transform_elm_df(df)

# Transform all EUC dataframes using corresponding canopy data
for tree_id, df in euc_dfs.items():
    if tree_id in canopy_dfs:
        print(f"Transforming EUC dataframe with Tree.ID = {tree_id} using its canopy data")
        euc_dfs[tree_id] = transform_euc_df(df, canopy_dfs[tree_id])
    else:
        print(f"No canopy data found for Tree.ID = {tree_id}. Skipping transformation.")

# Example: Save the transformed dataframes back to files (optional)
output_folder_elm = 'data/treeInputs/trunks-elms/transformed_templates/'
output_folder_euc = 'data/treeInputs/trunks/transformed_templates/'

if not os.path.exists(output_folder_elm):
    os.makedirs(output_folder_elm)
if not os.path.exists(output_folder_euc):
    os.makedirs(output_folder_euc)

for tree_id, df in elm_dfs.items():
    output_path = os.path.join(output_folder_elm, f'{tree_id}.csv')
    df.to_csv(output_path, sep=';', index=False)
    print(f"Saved transformed ELM dataframe for Tree.ID = {tree_id} to {output_path}")

for tree_id, df in euc_dfs.items():
    output_path = os.path.join(output_folder_euc, f'{tree_id}.csv')
    df.to_csv(output_path, sep=';', index=False)
    print(f"Saved transformed EUC dataframe for Tree.ID = {tree_id} to {output_path}")
