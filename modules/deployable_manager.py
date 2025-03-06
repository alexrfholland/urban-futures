import deployable_poles, deployable_lightweight
import geopandas as gpd
import pandas as pd
import utilities.cropToPolydata as cropToPoly
import stamp_shapefile
import pyvista as pv
import numpy as np
import trees, flattenmultiblock, getColorsAndAttributes
from scipy.spatial import cKDTree
import random

def getDeployableStructures(site):
    lightweights = deployable_lightweight.getLightweights(site)[0]
    poles = deployable_poles.runUtilityPole(site)[0]

    print(f'poles are \n{poles}')
    print(f'lightweights are \n{lightweights}')

    combined_deployables = gpd.GeoDataFrame(pd.concat([lightweights, poles], ignore_index=True))

    easting_offset, northing_offset = cropToPoly.getEastingAndNorthing(site)
    combined_deployables['modelX'] = combined_deployables.geometry.x - easting_offset
    combined_deployables['modelY'] = combined_deployables.geometry.y - northing_offset

    combined_deployables['modelX'] = pd.to_numeric(combined_deployables['modelX'], errors='coerce')
    combined_deployables['modelY'] = pd.to_numeric(combined_deployables['modelY'], errors='coerce')

    nan_count_modelX = combined_deployables['modelX'].isna().sum()
    nan_count_modelY = combined_deployables['modelY'].isna().sum()

    print(f'nan counts in columns modelX: {nan_count_modelX}, modelY: {nan_count_modelY}')
    print(combined_deployables[['modelX', 'modelY']].dtypes)

    print(f'{site} combined deployables are {combined_deployables}')

    filepath = f'outputs/deployables/deployables-{site}.shp'

    combined_deployables.to_file(filepath, driver='ESRI Shapefile')

    print(f'combined deployables are \n{combined_deployables}')

    print(f'{site}: combined deployables saved to {filepath}')

    return combined_deployables, filepath

def alignToModelOLD(site, combined_deployables, sitePoly):

    # If you want to include only specific columns
    alignedDF = combined_deployables[['modelX', 'modelY', 'condition', 'structure']].copy()
    print(f'aligned df is \n{alignedDF}')
    alignedDF = alignedDF.rename(columns={'modelX': 'X', 'modelY': 'Y'})

    """if 'trimmed' not in site:
        topo = getColorsAndAttributes.transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("topography", 100, site)
        topo = flattenmultiblock.flattenBlock(topo)
    else: 
        topo = pv.read(f'data/{site}/coloured-topography.vtk')

    topo_points = topo.points"""

    topo_points = sitePoly.points
    tree = cKDTree(topo_points[:, :2])  # considering only x, y coordinates

    # Function to find nearest Z
    def find_nearest_z(row):
        x, y = row['modelX'], row['modelY']
        distance, index = tree.query([x, y])
        return topo_points[index, 2]

    # Apply the function to each row
    alignedDF['Z'] = combined_deployables.apply(find_nearest_z, axis=1)


    print(f'aligned deployable dataframe is \n{alignedDF}')

    return alignedDF





def mark_close_points_for_removal(df, distance_threshold):
    # Filter to consider only points where 'isRetained' is True
    points_to_consider = df[df['isRetained'] == True]

    tree = cKDTree(points_to_consider[['X', 'Y']])
    points = points_to_consider[['X', 'Y']].to_numpy()
    indices_to_consider = points_to_consider.index

    # Counter for structures marked as too close
    too_close_count = 0

    # Query all points at once
    all_indices = tree.query_ball_point(points, distance_threshold)

    # Process the indices to determine which points to mark
    for idx, indices in enumerate(all_indices):
        actual_idx = indices_to_consider[idx]
        indices = set(indices_to_consider[indices]) - {actual_idx}
        if indices:  # if there are close points
            min_index = min(indices)
            if df.at[min_index, 'isRetained']:
                df.at[min_index, 'isRetained'] = False  # mark the one with the smallest index as not retained
                too_close_count += 1

    # Log the number of structures marked as too close
    print(f"Number of structures identified as too close: {too_close_count}")

    return df





def alignToModel(site, combined_deployables, sitePoly, aggressive=.5):
    # Copy the DataFrame including necessary columns
    alignedDF = combined_deployables[['modelX', 'modelY', 'condition', 'structure']].copy()
    alignedDF = alignedDF.rename(columns={'modelX': 'X', 'modelY': 'Y'})

    # Prepare for spatial queries and calculations
    topo_points = sitePoly.points
    tree = cKDTree(topo_points[:, :2])  # considering only x, y coordinates
    deployable_scores = sitePoly.point_data['offensiveScore']

    # Function to find nearest Z, calculate retention probability, and decide retention
    def process_row(row):
        x, y = row['X'], row['Y']
        distance, index = tree.query([x, y])
        row['Z'] = topo_points[index, 2]
        row['deployableScore'] = deployable_scores[index]

        # Calculate retain probability based on offensiveScore and aggressive parameter
        retain_probability = deployable_scores[index] * (1 - aggressive) + aggressive
        row['isRetained'] = random.random() < retain_probability
        return row

    # Apply the function to each row
    alignedDF = alignedDF.apply(process_row, axis=1)

    # Apply the mark_close_points_for_removal function
    alignedDF = mark_close_points_for_removal(alignedDF, 50)

    #alignedDF = mark_close_points_for_removal(alignedDF, 50)

    #alignedDF = mark_close_points_for_removal(alignedDF, 50)



    # Filter out rows where 'isRetained' is False
    filtered_df = alignedDF[alignedDF['isRetained'] == True]

    # Reset the index of the filtered DataFrame
    filtered_df.reset_index(drop=True, inplace=True)

    # Count the number of isRetained True and False
    is_retained_counts = alignedDF['isRetained'].value_counts()
    print(f'Count of isRetained True and False:\n{is_retained_counts}')

    #scatterDfWithSitePoly(alignedDF, sitePoly)

    return filtered_df

    return
    return filtered_df






    print(f'aligned deployable dataframe is \n{alignedDF}')

    # Filter out rows where 'isRemoved' is False
    filtered_df = alignedDF[alignedDF['isRemoved'] == False]

    # Reset the index of the filtered DataFrame
    filtered_df.reset_index(drop=True, inplace=True)




    # Assuming filtered_df is your filtered DataFrame
    # Plotting modelX and modelY, colored by isRemoved status'

def scatterDf(alignedDF, sitePoly):

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming alignedDF is your DataFrame
    # Plotting modelX and modelY, colored by isRemoved status as a categorical variable

    plt.figure(figsize=(10, 6))

    # Scatter plot using seaborn for automatic color coding based on 'isRemoved' category
    sns.scatterplot(data=alignedDF, x='X', y='Y', hue='isRetained', palette=['red', 'green'])

    # Adding labels and title
    plt.xlabel('modelX')
    plt.ylabel('modelY')
    plt.title('Scatter Plot of modelX and modelY by Removal Status')

    # Show the plot
    plt.show()


def scatterDfWithSitePoly(alignedDF, sitePoly):
    # Create a PolyData object from alignedDF
    print(f'aligned deployable dataframe is \n{alignedDF}')

    points = alignedDF[['X', 'Y', 'Z']].values
    point_cloud = pv.PolyData(points)
    point_cloud['isRetained'] = alignedDF['isRetained'] 

    # Create a plotter
    plotter = pv.Plotter()

    # Add the point cloud of alignedDF, large points colored by isRetained
    plotter.add_mesh(point_cloud, point_size=100, render_points_as_spheres=True, scalars='isRetained', cmap=['red', 'green'])

    # Add the sitePoly, colored by offensiveScore
    if 'offensiveScore' in sitePoly.point_data:
        plotter.add_mesh(sitePoly, scalars='offensiveScore', cmap='viridis', point_size=10, render_points_as_spheres=True)

    # Customize the camera and display settings
    plotter.show_grid(color='black')
    plotter.camera_position = 'xy'
    plotter.show()





import numpy as np
import pyvista as pv

# Dummy data
"""centers = np.random.random((10, 3)) * 10
types =  np.random.choice(['enriched', 'public', 'private'], 10) """

def make_deployable_glyphs(centers, types, plotter):

    enrichedBounds = pv.Cube(x_length=5, y_length=5, z_length=20, center=(0,0,10))
    publicBounds = pv.Cube(x_length=5, y_length=5, z_length=20, center=(0,0,10))
    privateBounds = pv.Cube(x_length=5, y_length=5, z_length=20, center=(0,0,10))

    deployTypes = ['enriched', 'public' , 'private']
    cols = {'enriched' : 'red', "public" : 'green', 'private' : 'yellow'}
    geoms = {'enriched' : enrichedBounds, "public" : publicBounds, "private" : privateBounds}

    for deployType in deployTypes:
        mask = types == deployType
        bounds = pv.PolyData(centers[mask])
        glyphs = bounds.glyph(geom=geoms[deployType])
        plotter.add_mesh(glyphs, color = cols[deployType])

"""plotter = pv.Plotter()  
make_deployable_glyphs(centers, types, plotter)
plotter.show()"""

def createDeployables(site, plotter=None):
    #path = f'data/{site}/flattened-{site}.vtk'
    path = f'data/{site}/offesnivePrep-{site}.vtk'
    sitePoly = pv.read(path)
    
    deployableShapefile = getDeployableStructures(site)[0]
    deployableDF = alignToModel(site, deployableShapefile, sitePoly)
    filledMask = deployableDF['condition'] != 'empty'
    filtered_df = deployableDF[filledMask]

    if plotter is None:
        plotter = pv.Plotter()
        plotter.add_mesh(sitePoly, scalars = 'deployables-condition', cmap = 'rainbow', point_size=5.0, render_points_as_spheres=True)

    #make_deployable_glyphs(deployablecenters, deployabletypes, plotter)
    canopyMultiblock = getDeployableCanopy(filtered_df, site, sitePoly)
    return canopyMultiblock

import numpy as np

def adjust_resource_distribution_per_resource2(df, deployableType, span, deployableResources):
    print(f'df is {df}')
    original_length = len(df)
    print(f'Initial DataFrame length: {original_length}')
    print(f'Conditions in DataFrame before adjusting {deployableType} are: {df["condition"].value_counts()}')

    # Step 1: Extract rows matching the specific condition 'deployableType'
    condition_subset = df[df['condition'] == deployableType]
    print(f"Step 1 - Extracted {len(condition_subset)} rows matching condition '{deployableType}'.")

    if('ScanX' in df.columns):
        # Step 2: From this extraction, further extract rows within a certain span
        span_subset = condition_subset[(condition_subset['ScanX'] <= span) & (condition_subset['ScanY'] <= span)]
        print(f"Step 2 - Further extracted {len(span_subset)} rows within the span.")

    else:
        span_subset = condition_subset
        print(f"df has no ScanX, not cropping deployable structures")

    # Reporting resource counts before and after adjustment
    print(f'Without adjustment, resources for {deployableType} are: {span_subset["resource"].value_counts()}')

    # Step 3: Loop over all unique resources in span_subset and adjust resources accordingly
    unique_resources = span_subset['resource'].unique()
    indices_to_replace = []

    for resource in unique_resources:
        if resource in deployableResources.columns:
            unique_subtypes = span_subset['deployableSubType'].unique()
            for subtype in unique_subtypes:
                print(f'adjusting: {resource}-{subtype}')
                subtype_resource_subset = span_subset[(span_subset['deployableSubType'] == subtype) & (span_subset['resource'] == resource)]
                reductionRate = 1 - deployableResources.at[subtype, resource]
                print(f'### reduction rate for {resource}-{subtype} is {reductionRate}')
                indices = subtype_resource_subset.index
                replace_count = int(reductionRate * len(indices))
                print(f'replacing {replace_count} indices for {resource}-{subtype}')
                selected_indices = np.random.choice(indices, size=replace_count, replace=False)
                indices_to_replace.extend(selected_indices)

                if resource in ['dead branch', 'perchable branch', 'peeling bark']:
                    print(f'branch resource detected...')
                    span_subset.loc[selected_indices, 'resource'] = 'other'
                elif resource not in ['dead branch', 'perchable branch', 'peeling bark']:
                    print(f'other resource detected')
                    # Remove the selected indices from span_subset
                    span_subset = span_subset.drop(selected_indices)


    print(f'After adjustment, resources for {deployableType} are: {span_subset["resource"].value_counts()}')

    # Step 4: Extract rows not matching 'deployableType'
    otherConditions = df[df['condition'] != deployableType]
    print(f'Conditions in otherConditions are: {otherConditions["condition"].value_counts()}')

    print(f'otherCondition df is \n{otherConditions}')

    # Create a final DataFrame based on whether otherConditions is empty
    final_df = pd.concat([otherConditions, span_subset], ignore_index=True)
    final_df.reset_index(drop=True, inplace=True)

    # Reporting on the number of rows culled
    final_length = len(final_df)
    rows_culled = original_length - final_length
    print(f'Final DataFrame length: {final_length}')
    print(f'Number of rows culled: {rows_culled}')

    return final_df



def adjust_resource_distribution_per_resource(df, deployableResources, deployableTypeDf):
    print(f'df is {df}')
    original_length = len(df)
    print(f'Initial DataFrame length: {original_length}')

    # Iterate over each deployableType
    for deployableType in df['condition'].unique():
        print(f'Conditions in DataFrame before adjusting {deployableType} are: {df["condition"].value_counts()}')

        # Step 1: Extract rows matching the specific condition 'deployableType'
        condition_subset = df[df['condition'] == deployableType]
        print(f"Step 1 - Extracted {len(condition_subset)} rows matching condition '{deployableType}'.")

        # Extract 'span' for this deployableType
        span = deployableTypeDf.at[deployableType, 'span'] if 'span' in deployableTypeDf.columns else None

        if 'ScanX' in df.columns and span is not None:
            # Step 2: From this extraction, further extract rows within a certain span
            span_subset = condition_subset[(condition_subset['ScanX'] <= span) & (condition_subset['ScanY'] <= span)]
            print(f"Step 2 - Further extracted {len(span_subset)} rows within the span.")
        else:
            span_subset = condition_subset
            print(f"df has no ScanX, or span is undefined, not cropping deployable structures")

        # Reporting resource counts before and after adjustment
        print(f'Without adjustment, resources for {deployableType} are: {span_subset["resource"].value_counts()}')

        # Step 3: Loop over all unique resources in span_subset and adjust resources accordingly
        unique_resources = span_subset['resource'].unique()
        indices_to_replace = []

        for resource in unique_resources:
            if resource in deployableResources.columns:
                unique_subtypes = span_subset['deployableSubType'].unique()
                for subtype in unique_subtypes:
                    print(f'adjusting: {resource}-{subtype}')

                    # Calculate composite reduction rate
                    default_reduction = 1 - deployableResources.at[subtype, resource]
                    custom_reduction = deployableTypeDf.at[deployableType, resource] if resource in deployableTypeDf.columns else -1
                    reductionRate = custom_reduction if custom_reduction != -1 else default_reduction

                    print(f'### reduction rate for {resource}-{subtype} is {reductionRate}')
                    subtype_resource_subset = span_subset[(span_subset['deployableSubType'] == subtype) & (span_subset['resource'] == resource)]
                    indices = subtype_resource_subset.index
                    replace_count = int(reductionRate * len(indices))
                    print(f'replacing {replace_count} indices for {resource}-{subtype}')
                    selected_indices = np.random.choice(indices, size=replace_count, replace=False)
                    indices_to_replace.extend(selected_indices)

                    if resource in ['dead branch', 'perchable branch', 'peeling bark']:
                        print(f'branch resource detected...')
                        span_subset.loc[selected_indices, 'resource'] = 'other'
                    elif resource not in ['dead branch', 'perchable branch', 'peeling bark']:
                        print(f'other resource detected')
                        # Remove the selected indices from span_subset
                        span_subset = span_subset.drop(selected_indices)

        print(f'After adjustment, resources for {deployableType} are: {span_subset["resource"].value_counts()}')

        # Step 4: Extract rows not matching 'deployableType'
        otherConditions = df[df['condition'] != deployableType]
        print(f'Conditions in otherConditions are: {otherConditions["condition"].value_counts()}')

        print(f'otherCondition df is \n{otherConditions}')

        # Create a final DataFrame based on whether otherConditions is empty
        final_df = pd.concat([otherConditions, span_subset], ignore_index=True)
        final_df.reset_index(drop=True, inplace=True)

        # Reporting on the number of rows culled
        final_length = len(final_df)
        rows_culled = original_length - final_length
        print(f'Final DataFrame length: {final_length}')
        print(f'Number of rows culled: {rows_culled}')

    return final_df

def getDeployableCanopy(points_df, site, polydata):

    points_df = points_df.copy()
    filePath = 'modules/painter/data/deployable-resources.csv'
    deployableResources = pd.read_csv(filePath)
    deployableResources.set_index('deployableSubType', inplace=True)

    conditionsPath = 'modules/painter/data/deployable-conditions.csv'
    deployableConditions = pd.read_csv(conditionsPath)
    deployableConditions.set_index('deployableType', inplace=True)

    print(f'points_df dataframe is \n {points_df}')

    points_df['_Tree_size'] = 'large'
    points_df['isPrecolonial'] = True
    points_df['_Control'] = 'reserve-tree'
    points_df['Genus'] = 'Eucalyptus'
    points_df['Scientific Name'] = 'Eucalyptus'
    points_df['Common Name'] = 'Eucalyptus'
    points_df['blockID'] = points_df.index
    points_df['blocktype'] = 'tree'
    points_df['extraTree'] = True


    # Assuming 'deployableSubType' is the index of deployableResources
    print(f'subtypes are:\n{deployableResources.index.value_counts()}')


    # Use apply to create a list of tuples (subtype, count based on percentage)
    # Revised code to create subtype tuples
    subtype_tuples = deployableResources.apply(lambda row: [(row.name, int(row['breakdown'] * len(points_df)))], axis=1).sum()


    # Flatten the list of tuples and repeat subtypes according to their counts
    subtype_list = [subtype for subtype, count in subtype_tuples for _ in range(count)]

    # Adjust the list size to match points_df in case of rounding differences
    if len(subtype_list) > len(points_df):
        subtype_list = subtype_list[:len(points_df)]  # Truncate if longer
    elif len(subtype_list) < len(points_df):
        # Add random subtypes if the list is shorter
        additional_subtypes = np.random.choice([subtype for subtype, _ in subtype_tuples], size=(len(points_df) - len(subtype_list)))
        subtype_list.extend(additional_subtypes)

    # Shuffle the list to randomly distribute subtypes
    np.random.shuffle(subtype_list)

    # Assign the subtype list to the points_df
    points_df['deployableSubType'] = subtype_list
    print(f'subtypes in deployable centers are:\n{points_df["deployableSubType"].value_counts()}')


        


    #baselines
    #baseline_tree_positions, baseline_site_polydata = trees.getBaselines.GetConditions(site)
    baseline_tree_positions = trees.assign_tree_model_id(points_df)
    baselineBranchdf, basegrounddf, basecanopydf = trees.create_canopy_array(site, 'test', baseline_tree_positions)

    baselineBranchdf['condition'] = baselineBranchdf['blockID'].map(points_df['condition'])
    baselineBranchdf['structure'] = baselineBranchdf['blockID'].map(points_df['structure'])
    baselineBranchdf['deployableSubType'] = baselineBranchdf['blockID'].map(points_df['deployableSubType'])

    basegrounddf['condition'] = basegrounddf['blockID'].map(points_df['condition'])
    basegrounddf['structure'] = basegrounddf['blockID'].map(points_df['structure'])
    basegrounddf['deployableSubType'] = basegrounddf['blockID'].map(points_df['deployableSubType'])

    basecanopydf['condition'] = basecanopydf['blockID'].map(points_df['condition'])
    basecanopydf['structure'] = basecanopydf['blockID'].map(points_df['structure'])
    basecanopydf['deployableSubType'] = basecanopydf['blockID'].map(points_df['deployableSubType'])

    # Copy the original DataFrame
    adjusted_branch_df = baselineBranchdf.copy()
    adjusted_canopy_df = basecanopydf.copy()
    adjusted_ground_df = basegrounddf.copy()

    adjusted_ground_df.rename(columns={'TreeX': 'ScanX', 'TreeY': 'ScanY', 'TreeZ': 'ScanZ'}, inplace=True)

 
    # Apply the function for the 'enriched' condition
    original_length = len(adjusted_branch_df)

    adjusted_branch_df = adjust_resource_distribution_per_resource(adjusted_branch_df, deployableResources, deployableConditions)
    adjusted_canopy_df = adjust_resource_distribution_per_resource(adjusted_canopy_df, deployableResources, deployableConditions)
    adjusted_ground_df = adjust_resource_distribution_per_resource(adjusted_ground_df, deployableResources, deployableConditions)



    """"print('#################### \n####################\n#################### BRANCH DEPLOYABLE ADJUSTMENT #################### \n#################### \n####################')
    adjusted_branch_df = adjust_resource_distribution_per_resource(adjusted_branch_df, 'enriched',2, deployableResources)
    adjusted_branch_df = adjust_resource_distribution_per_resource(adjusted_branch_df, 'private', 2, deployableResources)
    adjusted_branch_df = adjust_resource_distribution_per_resource(adjusted_branch_df, 'public', 2, deployableResources)

    print(f'canopy is \n{adjusted_canopy_df}')
    print(f'ground is \n{adjusted_ground_df}')


    print('#################### \n####################\n#################### CANOPY RESOURCES DEPLOYABLE ADJUSTMENT #################### \n#################### \n####################')
    adjusted_canopy_df = adjust_resource_distribution_per_resource(adjusted_canopy_df, 'enriched',2, deployableResources)
    adjusted_canopy_df = adjust_resource_distribution_per_resource(adjusted_canopy_df, 'private', 2, deployableResources)
    adjusted_canopy_df = adjust_resource_distribution_per_resource(adjusted_canopy_df, 'public', 2, deployableResources)

    print('#################### \n####################\n#################### GROUND DEPLOYABLE ADJUSTMENT #################### \n#################### \n####################')
    adjusted_ground_df = adjust_resource_distribution_per_resource(adjusted_ground_df, 'enriched',2, deployableResources)
    adjusted_ground_df = adjust_resource_distribution_per_resource(adjusted_ground_df, 'private', 2, deployableResources)
    adjusted_ground_df = adjust_resource_distribution_per_resource(adjusted_ground_df, 'public', 2, deployableResources)"""



    final_length = len(adjusted_branch_df)
    rows_culled = original_length - final_length
    print(f'#############################\nOriginal DataFrame length: {original_length}')
    print(f'Final DataFrame length: {final_length}')
    print(f'Number of rows culled: {rows_culled}')


    baseline_tree_dict = trees.create_canopy_dict(polydata, adjusted_branch_df, adjusted_ground_df, adjusted_canopy_df)

    # Creating MultiBlock
    multiblock = pv.MultiBlock()
    multiblock['branches'] = baseline_tree_dict['branches']
    multiblock['ground resources'] = baseline_tree_dict['ground resources']
    multiblock['canopy resources'] = baseline_tree_dict['canopy resources']

    return multiblock


