import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import stamp_shapefile, searcher, boundary
import kdemapper
import cameraSetUpRevised

def filter_points(poly_data, search: dict[str, dict[str, dict[str, list]]]) -> tuple:
    """
    Filters the points of a given PolyData object based on specified attribute values, supporting 'AND' and 'OR' conditions.

    Parameters:
    poly_data (pv.PolyData): The PolyData object whose points are to be filtered.
    search (dict[str, dict[str, dict[str, list]]]): A dictionary containing the criteria for filtering points, with 'AND'/'OR' conditions.

    Returns:
    tuple: A tuple containing two elements:
        - The name of the search (`search_name`).
        - A NumPy array of indices of the points in the PolyData object that meet the specified criteria.
    """
    # Assuming there's only one key-value pair in the search dictionary
    search_name, conditions = list(search.items())[0]
    valid_indices = set(range(len(poly_data.points)))

    print(f'Initial number of indices: {len(valid_indices)}')
    print(f'Search is {search}')

    for condition_type, query in conditions.items():
        if condition_type.upper() == 'AND':
            for attr, values in query.items():
                print(f'Checking attribute field (AND): {attr}')
                if attr not in poly_data.point_data:
                    print(f'Attribute {attr} not found in AND condition. Skipping.')
                    continue
                current_valid_indices = set(process_search_values(poly_data, attr, values))
                valid_indices = valid_indices.intersection(current_valid_indices) if valid_indices else current_valid_indices

        elif condition_type.upper() == 'OR':
            temp_indices = set()
            for attr, values in query.items():
                print(f'Checking attribute field (OR): {attr}')
                if attr in poly_data.point_data:
                    indices = process_search_values(poly_data, attr, values)
                    temp_indices.update(indices)
                    print(f'Found {len(indices)} indices for attribute {attr} in OR condition.')
            valid_indices = valid_indices.intersection(temp_indices) if valid_indices else temp_indices

    final_indices = np.array(list(valid_indices))
    print(f'Final number of filtered indices: {len(final_indices)}')
    return search_name, final_indices


def process_search_values(poly_data, attr, values):
    """ Helper function to process search values and return indices """
    indices = set()
    for value in values:
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
            # Range condition
            range_condition = (poly_data.point_data[attr] >= value[0]) & (poly_data.point_data[attr] <= value[1])
            indices.update(np.where(range_condition)[0])
        elif isinstance(value, (int, float)):
            # Numeric condition
            numeric_condition = poly_data.point_data[attr] == value
            indices.update(np.where(numeric_condition)[0])
        else:
            # String condition
            string_condition = poly_data.point_data[attr] == value
            indices.update(np.where(string_condition)[0])
    return indices


def classify_points(poly_data, search):
    """
    Classifies points in a PolyData object based on specified criteria.

    Parameters:
    poly_data (dict): A dictionary representing the PolyData object. It should have keys
                      corresponding to point attributes used in the search criteria.
    search (dict): A dictionary defining classification categories and their criteria.
                   The structure of this dictionary is as follows:
                   {
                       'CategoryName': {
                           'AND/OR': {
                               'attribute_name1': [value1, value2, ...],
                               'attribute_name2': [value1, value2, ...],
                               ...
                           }
                       },
                       ...
                   }
                   'AND' implies all conditions under it must be met.
                   'OR' implies any condition under it is sufficient.

    Returns:
    dict: A modified version of the input poly_data with an added key 'classifications',
          which contains the classification results as an array of strings.

    Example:
    search_example = {
        'Category1': {
            'AND': {
                'attribute1': [value1, value2],
                'attribute2': [value3, value4]
            }
        },
        'Category2': {
            'OR': {
                'attribute3': [value5, value6],
                'attribute4': [value7, value8]
            }
        }
    }
    poly_data_example = {
        'points': ...,
        'attribute1': ...,
        'attribute2': ...,
        'attribute3': ...,
        'attribute4': ...
    }
    classified_data = classify_points(poly_data_example, search_example)
    """
    classifications = np.full(poly_data['points'].shape[0], 'unclassified', dtype=object)

    for category, criteria in search.items():
        print(f'Classifying for category: {category}...')
        _, indices = filter_points(poly_data, {category: criteria})

        for idx in indices:
            if classifications[idx] == 'unclassified':
                classifications[idx] = category
        print(f'{len(indices)} points classified as {category}.')

    poly_data['classifications'] = classifications
    return poly_data

def classify_urban_systems(poly_data):
    # Initialize the 'roof_segments' attribute with a default value
    # Define the roof segment types and their criteria

    urbanSystemSearch = (
    {
        "Adaptable Vehical Infrastructure": 
        {
            "OR":
            [
                {"parkingmedian-isparkingmedian": [True]},
                {
                    "AND":
                    [
                        {"disturbance-potential": [4, 2, 3]},
                        {"NOT":
                            {
                                "AND": 
                                [
                                    {"little_streets-islittle_streets": [True]},
                                    {"road_types-type": ["Footway"]}
                                ]
                            }
                        }
                    ]
                }
            ]
        },
        "Private empty space": 
        {
            "AND": 
            [
                {"disturbance-potential": [1]}
            ]
        },
        "Existing Canopies": 
        {
            "AND": 
            [
                {"_Tree_size": ["large", "medium"]},
                {"NOT": {"road_types-type": ["Carriageway"]}}
            ]
        },
        "Existing Canopies Under Roads": 
        {
            "AND": 
            [
                {"_Tree_size": ["large", "medium"]},
                {"road_types-type": ["Carriageway"]}
            ]
        },
        "Street pylons": 
        {
            "OR": 
            [
                {"isstreetlight": [True]},
                {"ispylons": [True]}
            ]
        },
        "Load bearing roof": 
        { 
            "AND": 
            [
                {"buildings-dip": [[0.0, 0.1]]},
                {"extensive_green_roof-RATING": ["Excellent", "Good", "Moderate"]},
                {"elevation": [[-20, 80]]}
            ]
        },
        "Lightweight roof": 
        {
            "AND": 
            [
                {"buildings-dip": [[0.0, 0.1]]},
                {"intensive_green_roof-RATING": ["Excellent", "Good", "Moderate"]},
                {"elevation": [[-20, 80]]}
            ]
        },
        "Ground floor facade": 
        {
            "AND": 
            [
                {"buildings-dip": [[0.8, 1.7]]},
                {"solar": [[0.2, 1.0]]},
                {"elevation": [[0, 10]]}
            ]
        },
        "Upper floor facade": 
        {
            "AND": 
            [
                {"buildings-dip": [[0.8, 1.7]]},
                {"solar": [[0.2, 1.0]]},
                {"elevation": [[10, 80]]}
            ]
        }
    }
    )

    structureArray = searcher.classify_points_poly(poly_data, urbanSystemSearch)
    print(structureArray)

    poly_data['urban system'] = structureArray

    print(f'Unique values in roof segments {np.unique(poly_data["urban system"])}')

    return poly_data



def assign_potential_defensive_score(poly_data):
    # Mappings with index as the key and tuples (value, description)
    distance_mapping_dict = {0: (1, "is it"), 1: (0.8, "within 10m"), 2: (0.8, "within 20m"), 3: (0.7, ">20m")}
    disturbance_mapping_dict = {0: (0, "main road"), 1: (0.6, "private potential"), 2: (0.8, "public potential"), 3: (1, "already recognised"), 4: (1, "empty space")}

    # Convert dictionaries to arrays
    distance_mapping = np.array([value for value, description in distance_mapping_dict.values()])
    disturbance_mapping = np.array([value for value, description in disturbance_mapping_dict.values()])

    # Retrieve the necessary data
    bucketed_distance_indices = poly_data.point_data['bucketed distance']
    disturbance_potential_indices = poly_data.point_data['disturbance-potential-proximity']

    # Map the values using vectorized operations
    distance_scores = distance_mapping[bucketed_distance_indices]
    disturbance_scores = disturbance_mapping[disturbance_potential_indices]

    # Calculate disturbanceScore as a combination of both mappings
    disturbanceScore = distance_scores * disturbance_scores


    # Apply exceptions
    exception_value = distance_mapping[3] * disturbance_mapping[2]  # >20m * public potential
    # When disturbance-potential-proximity = 0 AND bucketed distance = 3
    exception_mask = (disturbance_potential_indices == 0) & (bucketed_distance_indices == 3)
    disturbanceScore[exception_mask] = exception_value

    # When disturbance-potential-proximity = 4 AND bucketed distance = 0
    exception_mask = (disturbance_potential_indices == 4) & (bucketed_distance_indices == 0)
    disturbanceScore[exception_mask] = exception_value

    # Update poly_data with the new attribute
    poly_data.point_data['disturbanceScore'] = disturbanceScore

    #get natural features score
    poly_data = kdemapper.getTreeWeights(poly_data,'tree')

    #get combinedscore
    geometric_mean = np.sqrt(poly_data.point_data['disturbanceScore'] * poly_data.point_data['tree-weights_log'] )

    poly_data.point_data['defensiveScore'] = geometric_mean

    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)


    plotter = pv.Plotter()

    # Settings for better visualization
    """plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.add_mesh(glyphs, scalars='disturbanceScore', cmap='viridis')
    #plotter.add_mesh(glyphs, scalars='fortifiedStructures', cmap='Set1')
    plotter.show()

    plotter = pv.Plotter()

    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.add_mesh(glyphs, scalars='tree-weights_log', cmap='viridis')
    #plotter.add_mesh(glyphs, scalars='fortifiedStructures', cmap='Set1')
    plotter.show()

    plotter = pv.Plotter()

    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0)))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.add_mesh(glyphs, scalars='defensiveScore', cmap='viridis')
    #plotter.add_mesh(glyphs, scalars='fortifiedStructures', cmap='Set1')
    plotter.show()"""





    return poly_data


import numpy as np
import pyvista as pv





def map_roof_segments(segment):
    mapping = {"Load bearing": 1, "Lightweight": 2, 
               "Ground floor facade": 3, "Upper floor facade": 4}
    return mapping.get(segment, -1)

import random
from scipy.spatial import cKDTree




from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool


def kde_evaluate_chunk(args):
    kde, centers_chunk = args
    return kde.evaluate(centers_chunk.T)

def compute_bandwidth_factor(data, weights, method='scott', factor=.1):
    # Instantiate a temporary KDE to compute the bandwidth
    kde_temp = gaussian_kde(data, weights=weights, bw_method=method)
    # Apply the scaling factor directly to the bandwidth factor of the temporary KDE
    bandwidth = kde_temp.factor * factor
    return bandwidth

def create_evaluation_grid(bounds, cell_size):
    # Unpack the bounds
    min_x, max_x, min_y, max_y, min_z, max_z = bounds

    # Create arrays of points along each axis
    x_grid = np.arange(min_x, max_x, cell_size)
    y_grid = np.arange(min_y, max_y, cell_size)
    z_grid = np.arange(min_z, max_z, cell_size)

    # Create a meshgrid from the axes arrays
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # Combine the grid points into a single array of 3D points
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    print(f'evaluating grid total points: {len(grid_points)}')


    return grid_points


def weighted_3d_kde(centers_1d, weights, cell_size, bandwidth_factor=0.5, useGrid=False):
    print("Reshaping the centers array...")
    centers = centers_1d.reshape(kdvalues-1, 3) if centers_1d.ndim == 1 else centers_1d

    print("Computing the bandwidth factor...")
    bandwidth = compute_bandwidth_factor(centers.T, weights, method='scott', factor=bandwidth_factor)

    print("Creating the gaussian_kde object...")
    kde = gaussian_kde(centers.T, weights=weights, bw_method=bandwidth)

    if useGrid:
        print(f"Creating a regular grid with voxel size {cell_size}...")
        # Calculate the 3D bounds of the points
        min_bounds = np.min(centers, axis=0)
        max_bounds = np.max(centers, axis=0)
        bounds = (min_bounds[0], max_bounds[0], min_bounds[1], max_bounds[1], min_bounds[2], max_bounds[2])
        grid_points = create_evaluation_grid(bounds, cell_size)
        evaluation_points = grid_points
    else:
        evaluation_points = centers

    print("Setting up multiprocessing for KDE evaluation...")
    num_processes = cpu_count()
    centers_chunks = np.array_split(evaluation_points, num_processes)

    with Pool(processes=num_processes) as pool:
        args = [(kde, chunk) for chunk in centers_chunks]
        results = pool.map(kde_evaluate_chunk, args)

    print("Combining the results...")
    kdvalues = np.concatenate(results)

    print(f'kd values: {len(kdvalues)}, grid points: {len(evaluation_points)}')

    print("Processing complete.")
    return kde, evaluation_points, 



def createAborealHabitatWeightings(polydata, siteMultiBlock):

    points, weights, allResources, bounds, cmap = boundary.process_agent_resources(siteMultiBlock, agent_name='survey')
    print(f'new bounds are {bounds}')

    print(f'points are length {len(points)}')
    print(f'weights are length {len(weights)}')
    print(f'bounds are {bounds}')

    grid = boundary.construct_kde_weighting(points, weights, bounds, 2.5, usesGrid=True)

    grid_tree = cKDTree(grid.points)

    # Find the nearest grid point for each point in polydata
    _, nearest_indices = grid_tree.query(polydata.points)

    for name in ['intensity', 'weights_minmax', 'weights_quartile', 'weights_zscore', 'weights_log', 'weights_robust']:
        polydata[f'KDE-{name}'] = grid.point_data[name][nearest_indices]

    return polydata
        

def assignFortifyStructures(polydata):
    # Initialize 'unassigned' array
    unassigned = np.full(len(polydata.point_data['urban system']), 'unassigned', dtype=object)

    # Design specifications
    designs = {
        'habitat island': {'urban system': ['Existing Canopies', 'Adaptable Vehical Infrastructure', 'Private empty space'], 'voxels': 100},
        'exoskeleton': {'urban system': ['Existing Canopies Under Roads'], 'voxels': 100},
        'living roof': {'urban system': ['Load bearing roof', 'Lightweight roof'], 'voxels': 80},
        'brown roof': {'urban system': ['Lightweight roof'], 'voxels': 80},
        'habitat brick': {'urban system': ['Ground floor facade'], 'voxels': 50},
        'living facade': {'urban system': ['Upper floor facade', 'Ground floor facade'], 'voxels': 80}
    }

    # Assign designs randomly for overlapping indices
    for design, specs in designs.items():
        for system in specs['urban system']:
            indices = np.where(polydata.point_data['urban system'] == system)[0]
            if len(indices) > 1:  # If more than one index, choose randomly
                indices = random.sample(list(indices), int(len(indices) * (specs['voxels'] / 100.0)))
            unassigned[indices] = design

    print("Step 1 - Initial Design Assignment")
    print(np.unique(unassigned, return_counts=True))

    # Apply further culling based on potential score
    potential_scores = polydata.point_data['defensiveScore']
    for i, score in enumerate(potential_scores):
        if random.random() > score:
            unassigned[i] = 'unassigned'

    print("Step 2 - Post Potential Score Culling")
    print(np.unique(unassigned, return_counts=True))

    # Update polydata with the new attribute
    polydata.point_data['fortifiedStructures'] = unassigned




    return polydata

def defensiveStructureManager(poly_data):
        poly_data = assign_potential_defensive_score(poly_data)
        #poly_data = createAborealHabitatWeightings(poly_data, siteMultiBlock)
        poly_data = classify_urban_systems(poly_data)
        poly_data = assignFortifyStructures(poly_data)

        return poly_data
        



def main(sites, states):
    import cameraSetUp
    import trees, getBaselines
    import glyphs
    import packtoMultiblock
    
    plotter = pv.Plotter()
    

    for siteNo, site in enumerate(sites):

        if (site != 'parade'):
            gridDist = 700
        else:
            gridDist = 800

        for stateNo, state in enumerate(states):
            print(f'processing {site} - {state}...')

            vtk_path = f'data/{site}/updated-{site}.vtk'
            siteMultiBlock = pv.read(f'data/{site}/combined-{site}-now.vtm')
            poly_data = pv.read(vtk_path)



            poly_data = defensiveStructureManager(poly_data)


            poly_data.save(f'data/{site}/defensive-{site}.vtk')
            

            """mask = poly_data.point_data['fortifiedStructures'] != 'unassigned'

            # Create zero arrays for x, y, z coordinates
            xshift = np.zeros(poly_data.n_points)
            yshift = np.zeros(poly_data.n_points)
            zshift = np.zeros(poly_data.n_points)

            # Update arrays with new values where mask is True
            xshift[mask] = poly_data.point_data['nx'][mask]
            yshift[mask] = poly_data.point_data['ny'][mask]
            zshift[mask] = poly_data.point_data['nz'][mask]

            # Combine the shift arrays into a single 2D array
            shifts = np.column_stack((xshift, yshift, zshift))

            # Add the shifts to the original points
            poly_data.points += shifts"""

            """from utilities.getCmaps import create_colormaps
            
            colormaps = create_colormaps()

            thresholdMask = poly_data.point_data['urban system'] != 'unclassified'


            #split
            thresholadarray = np.zeros(poly_data.n_points)
            thresholadarray[thresholdMask] = 1
            poly_data.point_data['threshold'] = thresholadarray
            poly_data.set_active_scalars('threshold')
            structurePoly = poly_data.threshold(1)
            sitePoly = poly_data.threshold(1, invert=True)



            # Visualization code
            plotter = pv.Plotter()
            cube = pv.Cube()  # Create a cube geometry for glyphing
            

            glyphsStructures = structurePoly.glyph(geom=cube, scale=False, orient=False, factor=1.5)
            glyphsSite = sitePoly.glyph(geom=cube, scale=False, orient=False, factor=1.5)

            # Settings for better visualization
            plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
            cameraSetUpRevised.setup_camera(plotter, 50, 600)
            light2 = pv.Light(light_type='cameralight', intensity=.5)
            light2.specular = 0.5  # Reduced specular reflection
            plotter.add_light(light2)
            plotter.enable_eye_dome_lighting()

            plotter.add_mesh(glyphsStructures, scalars='urban system', cmap='Set1')
            plotter.add_mesh(glyphsSite, color = 'white')
            #plotter.add_mesh(glyphs, scalars='structuresClassified', cmap='tab20c', clim = [0,19])
            #plotter.add_mesh(glyphs, scalars='structuresClassified', cmap='tab20b', clim = [0,19])


            #plotter.add_mesh(glyphsStructures, scalars='urban system', cmap=colormaps['discrete-1-7-section-muted'])


            
            #plotter.add_mesh(glyphs, scalars='KDE-weights_log', cmap='viridis', clim = [0,.1])
            #plotter.add_mesh(glyphs, scalars='fortifiedStructures', cmap='Set1')
            plotter.show()"""

            


if __name__ == "__main__":
    #sites = ['city', 'street', 'trimmed-parade']  # List of sites
    #sites = ['city']
    #sites = ['trimmed-parade']
    #sites = ['trimmed-parade']
    #sites = ['parade']
    #sites = ['trimmed-parade']
    sites = ['trimmed-parade', 'city', 'street']
    #sites = ['street']
    sites = ['street']
    #sites = ['street']
    sites = ['city']
    sites = ['parade']
    #sites = ['city']
    #sites = ['street']
    states = ['potential']
    #states = ['baseline']
    main(sites, states)




