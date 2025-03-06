import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree
import numpy as np

import os
import glob

import trees


def loadAndCull(dfs_dict, site_polydata, site, attributeNamesToTransfer):
    filtered_points = site_polydata.points
    kd_tree = cKDTree(filtered_points[:, :2])
    
    # Load site projections and get offsets
    site_coord = pd.read_csv('data/site projections.csv')
    site_to_search = site.split('-')[1] if '-' in site else site
    easting_offset = site_coord.loc[site_coord['Name'].str.contains(site_to_search, case=False), 'Easting'].values[0]
    northing_offset = site_coord.loc[site_coord['Name'].str.contains(site_to_search, case=False), 'Northing'].values[0]

    
    # Get bounds for culling
    #min_x, max_x, min_y, max_y = site_polydata.bounds[0], site_polydata.bounds[1], site_polydata.bounds[2], site_polydata.bounds[3]

    min_x, max_x, min_y, max_y = site_polydata.bounds[0:4]  # Extracting the first four values from bounds


    for csv_name, df in dfs_dict.items():
        print(f'Processing {csv_name}')

        # Rename columns 'X' and 'Y' to 'Easting' and 'Northing' if 'Easting' and 'Northing' are not present
        if 'Easting' not in df.columns and 'X' in df.columns:
            df.rename(columns={'X': 'Easting'}, inplace=True)
        if 'Northing' not in df.columns and 'Y' in df.columns:
            df.rename(columns={'Y': 'Northing'}, inplace=True)
        
        # Transform and cull points in the dataframe
        print(f"Transforming and culling points for {df.shape[0]} rows.")
        points = df[['Easting', 'Northing']].values
        points[:, 0] -= easting_offset
        points[:, 1] -= northing_offset

        # Cull points by the x y bounds
        mask = (
            (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
            (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
        )
        df = df[mask]
        points = points[mask]

        df.reset_index(drop=True, inplace=True)
        df['blockNo'] = df.index

        # Query KDTree to find the index of the nearest point in site_polydata for each point in the dataframe
        _, nearest_indices = kd_tree.query(points)


        is_csv_name_array = np.full(site_polydata.points.shape[0], False)
        is_csv_name_array[nearest_indices] = True
        true_count = np.count_nonzero(is_csv_name_array)
        isName = f'is{csv_name}'
        site_polydata.point_data[isName] = is_csv_name_array
        print(f'The number of True values in {isName} attribute is {true_count}.')
        site_polydata.point_data[isName] = is_csv_name_array
        print(f'Transferred attribute {isName}')

        
        # Transfer specified attributes from dataframe to site_polydata
        attributes_to_transfer = attributeNamesToTransfer.get(csv_name, [])
        print(f'attributes to transfer from {csv_name} are {attributes_to_transfer}')
        for attribute in attributes_to_transfer:
            if attribute in df.columns:  # Ensure the attribute exists in the DataFrame
                attribute_name_in_polydata = f'{csv_name}_{attribute}'
                default_value = -1 if pd.api.types.is_numeric_dtype(df[attribute]) else None
                temp_array = np.full(site_polydata.points.shape[0], default_value)
                temp_array[nearest_indices] = df[attribute].values
                site_polydata.point_data[attribute_name_in_polydata] = temp_array
                print(f'Transferred attribute {attribute_name_in_polydata}')


    return site_polydata  # Return the modified site_polydata


def get_csvs(folder_name):
    # Create a pattern to match all CSV files in the specified folder
    pattern = os.path.join(folder_name, "*.csv")

    # Use glob to find all matching files
    csv_files = glob.glob(pattern)

    # Initialize an empty dictionary
    dfs_dict = {}

    # Loop through each file, print the file name, and add to the dictionary
    for file in csv_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        print(f'Processing file: {file_name}')
        dfs_dict[file_name] = pd.read_csv(file)

    return dfs_dict


def stampCsvs(site_polydata, site):

    print(f'stamping Csvs for site {site}')

    attributeNamesToTransfer = {
        'pylons' : ['blockNo'],
        'street-furniture' : ['blockNo, asset_class'],
        'streetlight' : ['blockNo']
    }

    csvPath = 'data/input_csvs'

  
    dfs_dict = get_csvs(csvPath)
    




    
    # Call loadAndCull
    site_polydata = loadAndCull(dfs_dict, site_polydata, site, attributeNamesToTransfer)
    print(f'Modified site_polydata with {len(site_polydata.point_data.keys())} attributes.')

    return site_polydata

