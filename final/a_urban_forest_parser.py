import pandas as pd
import xarray as xr
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import f_SiteCoordinates, f_GeospatialModule, f_resource_urbanForestParser, a_helper_functions

def getPoles(site):
    print(f"Getting poles for site: {site}")
    terrainVTK = pv.read(f'data/revised/{site}-roadVoxels.vtk')
    siteVTK = pv.read(f'data/revised/{site}-siteVoxels.vtk')

    centroidEastings, centroidNorthings, eastingsDim, northingsDim = f_SiteCoordinates.get_center_and_dims([siteVTK])
    transformEasting, transformNorthing = f_SiteCoordinates.get_site_coordinates(site)

    gdfPylons, gdfPoles = f_GeospatialModule.handle_poles(centroidEastings, centroidNorthings, eastingsDim, northingsDim)

    print(f'Site center - easting: {centroidEastings}, northing: {centroidNorthings}, eastingsDim: {eastingsDim}, northingsDim: {northingsDim}')
    
    # Convert gdfPylons to DataFrame
    pylons_df = pd.DataFrame({
        'eastings': gdfPylons.geometry.x,
        'northings': gdfPylons.geometry.y,
        'poletype': 'pylon'
    })
    
    # Convert gdfPoles to DataFrame
    poles_df = pd.DataFrame({
        'eastings': gdfPoles.geometry.x,
        'northings': gdfPoles.geometry.y,
        'poletype': 'pole'
    })
    
    # Combine the two DataFrames
    utilityPoleLocations = pd.concat([pylons_df, poles_df], ignore_index=True)
    utilityPoleLocations['isArtificial'] = True

    # Assign x and y columns to utilityPoleLocations
    utilityPoleLocations['x'] = utilityPoleLocations['eastings']
    utilityPoleLocations['y'] = utilityPoleLocations['northings']

    # Get z coordinate 
    utilityPoleLocations = f_resource_urbanForestParser.calculate_z_coordinates(utilityPoleLocations, terrainVTK)

    # Transform to 0,0,0 coordinates
    utilityPoleLocations['x'] = utilityPoleLocations['x'] - transformEasting
    utilityPoleLocations['y'] = utilityPoleLocations['y'] - transformNorthing    

    utilityPoleLocations['pole_number'] = utilityPoleLocations.index.astype(int)

    print(f"Total utility poles found: {len(utilityPoleLocations)}")
    return utilityPoleLocations

def fill_useful_life_expectancy(df):
    # Define the mapping based on size
    life_expectancy_map = {
        'small': 80.0,
        'medium': 50.0,
        'large': 20.0
    }
    # Fill the missing values in 'useful_life_expectency' column
    df['OLDuseful_life_expectancy'] = df['useful_life_expectancy'].fillna(df['size'].map(life_expectancy_map))
    df['useful_life_expectancy'] = df['useful_life_expectancy'].fillna(df['size'].map(life_expectancy_map))
    return df


# Redefine the main function using the updated generic initializer
def initialize_and_assign_variables(xarray_dataset, treeDF, poleDF):
    # Step 1: Extract centroids from xarray dataset
    centroids = np.column_stack((
        xarray_dataset.centroid_x.values,
        xarray_dataset.centroid_y.values,
        xarray_dataset.centroid_z.values
    ))

    # Step 2: Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(centroids)
    
    # Find nearest voxel for each tree and pole
    tree_positions = treeDF[['x', 'y', 'z']].values
    pole_positions = poleDF[['x', 'y', 'z']].values

    _, tree_voxel_indices = tree.query(tree_positions)
    _, pole_voxel_indices = tree.query(pole_positions)
    
    print(f"Assigned voxel indices to {len(tree_voxel_indices)} trees")
    print(f"Assigned voxel indices to {len(pole_voxel_indices)} poles")
    
    print(f'treeDF cols are {treeDF.columns}')
    print(f'poleDF cols are {poleDF.columns}')

    
    
    # Step 3: Define the columns and initialize variables in the xarray with the convention
    tree_columns = ['size', 'control', 'precolonial', 'useful_life_expectancy', 'tree_id', 'tree_number', 'diameter_breast_height']

    

    pole_columns = ['pole_number']
    #print pole_number of poleDF
    print(f'poleDF pole_number is {poleDF.pole_number}')
    xarray_dataset = a_helper_functions.initialize_xarray_variables_generic_auto(xarray_dataset, treeDF, tree_columns, 'trees_')
    xarray_dataset = a_helper_functions.initialize_xarray_variables_generic_auto(xarray_dataset, poleDF, pole_columns, 'poles_')

    # Step 4: Assign pole and tree data to the nearest voxel, dropping duplicates
    treeDF['voxel_index'] = tree_voxel_indices
    poleDF['voxel_index'] = pole_voxel_indices
    
    # Drop duplicates for each voxel index, keeping the first occurrence
    treeDF = treeDF.drop_duplicates(subset=['voxel_index'])
    poleDF = poleDF.drop_duplicates(subset=['voxel_index'])

    # Assign tree data to xarray
    for col in tree_columns:
        xarray_dataset[f'trees_{col}'].values[treeDF['voxel_index']] = treeDF[col].values

    # Assign pole data to xarray
    for col in pole_columns:
        xarray_dataset[f'poles_{col}'].values[poleDF['voxel_index']] = poleDF[col].values

    # Check unique values in pole_number column of poleDF
    unique_pole_numbers_df = poleDF['pole_number'].nunique()
    print(f"Number of unique pole numbers in poleDF: {unique_pole_numbers_df}")

    # Check unique values in poles_pole_number variable of xarray
    unique_pole_numbers_xarray = np.unique(xarray_dataset['poles_pole_number'].values[xarray_dataset['poles_pole_number'].values != 0]).size
    print(f"Number of unique pole numbers in xarray: {unique_pole_numbers_xarray}")

    # Compare and report error if different
    if unique_pole_numbers_df != unique_pole_numbers_xarray:
        print("ERROR: The number of unique pole numbers in poleDF and xarray do not match.")
        print(f"poleDF unique count: {unique_pole_numbers_df}")
        print(f"xarray unique count: {unique_pole_numbers_xarray}")
    else:
        print("The number of unique pole numbers in poleDF and xarray match.")
    
    return xarray_dataset


def get_resource_dataframe(site, xarray_dataset, filePath):
    print(f"Processing resource dataframe for site: {site}")

    # Load the resource DataFrame for trees
    treeDF = pd.read_csv(f'data/revised/{site}-tree-locations.csv')
    print(f"Loaded {len(treeDF)} trees from CSV")

    treeDF = treeDF.rename(columns={'useful_life_expectency': 'useful_life_expectancy'})

    print('filling in life expectency')
    treeDF = fill_useful_life_expectancy(treeDF)

    #Adjust tree x y z columns to be 0,0,0
    transformEasting, transformNorthing = f_SiteCoordinates.get_site_coordinates(site)
    treeDF['x'] = treeDF['x'] - transformEasting
    treeDF['y'] = treeDF['y'] - transformNorthing   

    # Get poles data
    poleDF = getPoles(site)

    xarray_dataset = initialize_and_assign_variables(xarray_dataset, treeDF, poleDF)

    #print unique values in xarray_dataset['poles_pole_number']

    return xarray_dataset, treeDF, poleDF


if __name__ == '__main__':
    site = 'trimmed-parade'
    filePATH = f'data/revised/final/{site}'
    voxel_size = 1
    ds = xr.open_dataset(f'{filePATH}/{site}_{voxel_size}_voxelArray_initial.nc')


    ds, treeDf, poleDf = get_resource_dataframe(site, ds, filePATH)

    #save treeDF to csv
    treeDf.to_csv(f'{filePATH}/{site}_{voxel_size}_treeDF.csv', index=False)
    poleDf.to_csv(f'{filePATH}/{site}_{voxel_size}_poleDF.csv', index=False)

    print(f"Unique values in xarray_dataset['poles_pole_number']: {np.unique(ds['poles_pole_number'].values)}")
