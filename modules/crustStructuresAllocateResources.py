import glyphs, cameraSetUpRevised, packtoMultiblock
import painter.paint as paint

import numpy as np
import time

import os
import sys
import pyvista as pv
import deployable_manager
import defensive_structure_manager

import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree
import numpy as np


def pyvista_to_pandas(polydata):
    """
    Convert PyVista PolyData to a Pandas DataFrame.
    """
    # Extracting the X, Y, Z coordinates
    xyz = polydata.points
    data = {'X': xyz[:, 0], 'Y': xyz[:, 1], 'Z': xyz[:, 2]}

    # Adding other point data attributes
    for attr in polydata.point_data.keys():
        array = polydata.point_data[attr]
        if len(array.shape) == 1:
            # Single-dimensional attribute
            data[attr] = array
        else:
            # Multi-dimensional attribute
            data[attr] = list(map(tuple, array))

    # Convert to DataFrame
    try:
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"Error while creating DataFrame: {e}")
        return None
    



    print(df)

    return df


def create_multiblock_from_df(df):
    print(f'creating multiblock')

    def create_polydata(df):
        points = df[['X', 'Y', 'Z']].values
        polydata = pv.PolyData(points)

        for col in df.columns.difference(['X', 'Y', 'Z']):
            sanitized_data = []
            for val in df[col]:
                if isinstance(val, str):
                    sanitized_val = val.encode('ascii', 'ignore').decode()
                    sanitized_data.append(sanitized_val)
                else:
                    sanitized_data.append(val)
            polydata.point_data[col] = sanitized_data

        polydata.point_data['ScanZ'] = df['Z']
        return polydata

    branches_df = df[df['resource'].isin(['perchable branch', 'peeling bark', 'dead branch'])]
    ground_resources_df = df[df['resource'].isin(['fallen log', 'leaf litter'])]
    canopy_resources_df = df[df['resource'].isin(['epiphyte', 'hollow'])]

    print('creating branches polydata')
    branches_polydata = create_polydata(branches_df)

    print('creating ground polydata')
    ground_resources_polydata = create_polydata(ground_resources_df)

    print('creating canopy polydata')
    canopy_resources_polydata = create_polydata(canopy_resources_df)

    multiblock = pv.MultiBlock()
    multiblock['branches'] = branches_polydata
    multiblock['ground resources'] = ground_resources_polydata
    multiblock['canopy resources'] = canopy_resources_polydata

    return multiblock

def distribute_resources(polydataDF, habitat_cubic_meters_df):
    """
    Distribute resources for each habitat structure based on specified ratios, 
    and calculate randomized XYZ positions for these resources.

    Parameters:
    polydataDF (pd.DataFrame): DataFrame containing details about polydata.
    habitat_cubic_meters_df (pd.DataFrame): DataFrame containing details about habitat structures and resource ratios.

    Returns:
    pd.DataFrame: A DataFrame with distributed resources including their XYZ positions and types.
    """
    
    resource_types = ['other', 'peeling bark', 'dead branch', 'epiphyte', 
                      'leaf litter', 'perchable branch', 'hollow', 'fallen log']

    resource_distribution = pd.DataFrame()

    for resource in resource_types:
        resource_counts = habitat_cubic_meters_df.set_index('Structure')[resource]
        print(f"Resource counts for {resource}: \n{resource_counts}")

        for structure, count in resource_counts.items():
            print(f'calculating {structure} with count {count}')
            
            structure_df = polydataDF[polydataDF['fortifiedStructures'] == structure]


            if not structure_df.empty:
                num_resources = int(count * len(structure_df))
                print(f"Structure: {structure}, Resource: {resource}, Ratio: {count}, Structure Count: {len(structure_df)}, Resources Created: {num_resources}")

                if num_resources > 0:
                    allocated_indices = np.random.choice(structure_df.index, num_resources, replace=True)
                    allocated_df = structure_df.loc[allocated_indices].copy()
                    allocated_df['resource'] = resource

                    resource_distribution = pd.concat([resource_distribution, allocated_df])


    resource_distribution['Xcenter'] = resource_distribution['X']
    resource_distribution['Ycenter'] = resource_distribution['Y']
    resource_distribution['Zcenter'] = resource_distribution['Z']

    resource_distribution['X'] = resource_distribution['Xcenter'] + np.random.uniform(-.5, .5, size=resource_distribution.shape[0])
    resource_distribution['Y'] = resource_distribution['Ycenter'] + np.random.uniform(-.5, .5, size=resource_distribution.shape[0])
    resource_distribution['Z'] = resource_distribution['Zcenter'] + np.random.uniform(-.5, .5, size=resource_distribution.shape[0])


    """import matplotlib.pyplot as plt

    # Plotting the X and Y coordinates before and after adding jitter
    plt.figure(figsize=(12, 6))

    # Original Coordinates
    plt.subplot(1, 2, 1)
    plt.scatter(resource_distribution['X'], resource_distribution['Y'], alpha=0.6)
    plt.title('Original Coordinates')
    plt.xlabel('X')
    plt.ylabel('Y')

    
    plt.show()"""


    return resource_distribution


def distribute_resource_manager(polydata):
    siteDF = pyvista_to_pandas(polydata)
    resourceRatios =    pd.read_csv('modules/painter/data/habitat structure cubic meters.csv')
    resourceDF = distribute_resources(siteDF, resourceRatios)
    canopyMultiblock = create_multiblock_from_df(resourceDF)
    return canopyMultiblock


def main(sites, states):

    gridDist = 300

    plotter = pv.Plotter()
    cameraSetUpRevised.setup_camera(plotter, gridDist, 600)

    # Usage:
    for siteNo, site in enumerate(sites):

        for stateNo, state in enumerate(states):
            
            multiblock = pv.read(f'data/{site}/structures-{site}-{state}.vtm')
            print(f'{site} - {state} loaded')

            z_translation = multiblock.get('rest of points').bounds[4]
            translation_amount = np.array([gridDist * stateNo, gridDist * siteNo, z_translation])

            structures = multiblock.get('structure voxels')
            structuresMultiblock = distribute_resource_manager(structures)

            multiblock['branches'] = multiblock['branches'] + structuresMultiblock['branches'] 
            multiblock['ground resources'] = multiblock['ground resources'] + structuresMultiblock['ground resources']
            multiblock['canopy resources'] = multiblock['canopy resources'] + structuresMultiblock['canopy resources']

            multiblock.save(f'data/{site}/all_resources-{site}-{state}.vtm')





            
            glyphs.add_trees_to_plotter(plotter, multiblock, translation_amount)
            glyphs.plotRestofSitePoints(plotter, multiblock.get('rest of points'), translation_amount)

            
            print(f'added to plotter: {site} - {state}')

    
    plotter.show()


if __name__ == "__main__":
    #sites = ['city', 'street', 'trimmed-parade']  # List of sites
    #sites = ['city', 'trimmed-parade']
    #sites = ['parade']
    #state = ['baseline', 'now', 'trending']
    #sites = ['street', 'city','trimmed-parade']
    sites = ['street']
    #sites = ['city']
    #sites = ['city']

    #sites = ['street']
    #sites = ['parade']
    
    state = ['now']
    #state = ['baseline']
    main(sites, state)

