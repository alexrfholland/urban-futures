import pyvista as pv
import getColorsAndAttributes, flattenmultiblock
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
def GetConditions(site):
    if 'trimmed' not in site:
        topo = getColorsAndAttributes.transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("topography", 100, site)
        topo = flattenmultiblock.flattenBlock(topo)
    else: 
        topo = pv.read(f'data/{site}/coloured-topography.vtk')

    baselineDensities = pd.read_csv('data/csvs/tree-baseline-density.csv')
    print('Baseline Densities:\n', baselineDensities)
    
    bounds = topo.bounds
    print(f'Bounds: {bounds}')
    Xmin, Xmax, Ymin, Ymax = bounds[:4]
    area = ((Xmax - Xmin) * (Ymax - Ymin)) / 10000
    print(f'Calculated Area: {area} hectares')
    
    
    topo.point_data['blocktype'] = ['topo'] * topo.n_points
    print(f'baseline blocktype is {topo.point_data["blocktype"]}')
    
    topo_points = topo.points
    tree = cKDTree(topo_points[:, :2])  # considering only x, y coordinates
    
    data = []  # Initialize an empty list to collect the data
    
    for index, row in baselineDensities.iterrows():
        size = row['Size']
        flat_density = row['Flat']
        num_points = int((area * flat_density) / 0.1)
        
        random_x = np.random.uniform(Xmin, Xmax, num_points)
        random_y = np.random.uniform(Ymin, Ymax, num_points)
        random_z = np.array([find_nearest_z(x, y, tree, topo_points) for x, y in zip(random_x, random_y)])
        
        # Append tuples containing the X, Y, Z coordinates and the size to the data list
        data.extend(zip(random_x, random_y, random_z, [size] * num_points))
    
    # Convert the data list to a DataFrame
    columns = ['X', 'Y', 'Z', 'Diameter Breast Height']
    points_df = pd.DataFrame(data, columns=columns)
    points_df['_Tree_size'] = points_df['Diameter Breast Height'].apply(lambda x: 'small' if x < 50 else ('medium' if 50 <= x < 80 else ('large' if x >= 80 else 'small')))
    points_df['isPrecolonial'] = True
    points_df['_Control'] = 'reserve-tree'
    points_df['Genus'] = 'Eucalyptus'
    points_df['Scientific Name'] = 'Eucalyptus'
    points_df['Common Name'] = 'Eucalyptus'
    points_df['blockID'] = points_df.index
    points_df['blocktype'] = 'tree'

    print(f'baselines are {points_df}')  # Print the DataFrame to check the result

    return points_df, topo

    
def find_nearest_z(x, y, tree, topo_points):
    # Find the index of the nearest point in topo_points
    distance, index = tree.query([x, y])
    # Return the z coordinate of the nearest point
    return topo_points[index, 2]

#GetConditions('city')
