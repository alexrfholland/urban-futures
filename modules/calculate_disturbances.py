import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import stamp_shapefile

#can improve noise modelling https://github.com/Universite-Gustave-Eiffel/NoiseModelling/releases/tag/v4.0.5

def filter_points(poly_data, search: dict[str, dict[str, list]]) -> tuple:
    """
    Filters the points of a given PolyData object based on specified attribute values.

    Parameters:
    poly_data (pv.PolyData): The PolyData object whose points are to be filtered.
    search (dict[str, dict[str, list]]): A dictionary containing the criteria for filtering points.
        The structure of the dictionary is as follows:
        {
            'search_name' : {
                'attribute_name1': [value1, value2, ...],
                'attribute_name2': [value1, value2, ...],
                ...
            }
        }
        - 'search_name' (str): A string representing the name of the search.
        - 'attribute_name' (str): A string representing the name of the attribute field in the PolyData object.
        - value a list of (varied): The values to search for within the specified attribute field. These can be individual numbers or strings or ranges specified as a tuple of (min_value, max_value).

    Returns:
    tuple: A tuple containing two elements:
        - The name of the search (`search_name`).
        - A NumPy array of indices of the points in the PolyData object that meet the specified criteria.
    """
        
    # Assuming there's only one key-value pair in the search dictionary
    search_name, query = list(search.items())[0]
    
    # Start with all indices
    valid_indices = set(range(len(poly_data.points)))
    print(f'Initial number of indices: {len(valid_indices)}')
    print(f'Search is {search}')

    # Iterate over each key-value pair in the search dictionary
    for attr, values in query.items():
        print(f'Checking attribute field: {attr}')

        # Check if the attribute exists in poly_data
        if attr not in poly_data.point_data:
            print(f'Attribute {attr} not found. Treating as if no indices were found for this attribute.')
            continue

        # Initialize an empty set to collect indices that satisfy the current key-value pair
        current_valid_indices = set()

        # Iterate over each value in the list of values for the current key
        for value in values:
            print(f'Searching for values matching {value}')
            if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
                # Range condition
                range_condition = (poly_data.point_data[attr] >= value[0]) & (poly_data.point_data[attr] <= value[1])
                current_valid_indices.update(np.where(range_condition)[0])
            elif isinstance(value, (int, float)):
                # Numeric condition
                numeric_condition = poly_data.point_data[attr] == value
                current_valid_indices.update(np.where(numeric_condition)[0])
            else:
                # String condition
                string_condition = poly_data.point_data[attr] == value
                current_valid_indices.update(np.where(string_condition)[0])
        
        # Update valid_indices to retain only those indices that satisfy all key-value pairs processed so far
        valid_indices.intersection_update(current_valid_indices)
        print(f'Valid number of indices after processing {attr}: {len(current_valid_indices)}')
    
    # Convert the set of valid indices to an array and return it
    final_indices = np.array(list(valid_indices))
    print(f'Final number of filtered indices: {len(final_indices)}')
    return (search_name, final_indices)


def filter_points_OR(poly_data, level_criteria):
    combined_indices = set()
    for attr, values in level_criteria.items():
        # Construct the input for filter_points correctly
        search_criteria = {'level_search': {attr: values}}
        _, indices = filter_points(poly_data, search_criteria)
        if attr in poly_data.point_data:
            combined_indices.update(indices)
    return np.array(list(combined_indices))


def assign_conversion_potential(poly_data):
    # Initialize the 'conversion-potential' attribute with -1
    conversion_potential = np.full(len(poly_data.points), -1)

    # Define the levels and their criteria in reverse order
    levels = {
        0: {'road_types-type': ['Carriageway']}, #main roads
        1: {
            #'private-isprivate': [False],
            'private_buffer-isprivate_buffer' : [False], #private-potential
            'road_info-str_type': ['Private']
        },
        2: {
            'road_info-str_type': ['Council Minor'], #public-potential
            'little_streets-islittle_streets': [True]
        },    
        3: { #already allocated
            'laneways-verticalsu': [(50, float('inf'))],
            'laneways-parklane': [(50, float('inf'))],
            'laneways-forest': [(50, float('inf'))],
            'laneways-farmlane': [(50, float('inf'))]
        },
        4: {'isopen_space': [True]},
        -1: {'blocktype' : ['buildings']},
    
    }

    # Process each level from lowest to highest
    for level, criteria in levels.items():
        print(f'Searching for level {level}...')
        indices = filter_points_OR(poly_data, criteria)

        # Update the conversion potential for indices in this level
        if len(indices) > 0:
            conversion_potential[indices] = level
            print(f'{len(indices)} indices found for level {level}.')
        else:
            print(f'No indices found for level {level}.')

    # Assign the 'conversion-potential' attribute to poly_data
    poly_data.point_data['disturbance-potential'] = conversion_potential

    print(f'Unique values in disturbance potential {np.unique(poly_data.point_data["disturbance-potential"])}')

    return poly_data


from scipy.spatial import KDTree
import numpy as np

def assign_disturbance_potential_proximity(poly_data):
    # Extract points and their disturbance potential
    points = np.array(poly_data.points)
    disturbance_potential = np.array(poly_data.point_data['disturbance-potential'])

    # Initialize 'distance to road' with NaN for all points
    distances_to_nearest = np.full(len(points), np.nan)

    # Identify classified points and set their 'distance to road' to 0
    classified_indices = np.where(disturbance_potential != -1)[0]
    distances_to_nearest[classified_indices] = 0

    # Build KDTree with classified points
    tree = KDTree(points[classified_indices])

    # For unclassified points, find the nearest classified point and get its disturbance potential
    unclassified_indices = np.where(disturbance_potential == -1)[0]
    if len(unclassified_indices) > 0:
        distances_to_nearest[unclassified_indices], nearest_classified_indices = tree.query(points[unclassified_indices])
        nearest_classified_potentials = disturbance_potential[classified_indices][nearest_classified_indices]

        # Assign nearest disturbance potential level to unclassified points
        disturbance_potential[unclassified_indices] = nearest_classified_potentials

    # Bucketize 'distance to road'
    bucketized_distance = np.zeros_like(disturbance_potential, dtype=int)
    bucketized_distance[(distances_to_nearest > 0) & (distances_to_nearest < 10)] = 1
    bucketized_distance[(distances_to_nearest >= 10) & (distances_to_nearest < 20)] = 2
    bucketized_distance[distances_to_nearest >= 20] = 3

    # Update poly_data with the new attributes
    poly_data.point_data['disturbance-potential-proximity'] = disturbance_potential
    poly_data.point_data['distance to road'] = distances_to_nearest
    poly_data.point_data['bucketed distance'] = bucketized_distance
    poly_data.point_data['distance groups'] = poly_data.point_data['disturbance-potential-proximity'] * 4 + poly_data.point_data['bucketed distance']

    # Print unique values in 'distance groups'
    print(f'Unique values in disturbance potential {np.unique(poly_data.point_data["disturbance-potential"])}')
    print(f'Unique values in bucketed distance {np.unique(poly_data.point_data["bucketed distance"])}')
    print(f'Unique values in disturbance-potential-proximity {np.unique(poly_data.point_data["disturbance-potential-proximity"])}')
    print(f'Unique values in distance groups {np.unique(poly_data.point_data["distance groups"])}')

    return poly_data

    return poly_data


def calculate_disturbance_groups(polydata, site):

    shapefiles_buffer = [
    'data/deployables/shapefiles/private/deployables_private_empty_space.shp',
    ]

    polydata = stamp_shapefile.read_and_plot(site, polydata, shapefiles_buffer, 1000,deleteAttributes=True,buffer=2)

    
    #classify ground
    ##five tiers. already (openspace) very high (convertableLaneway), high  (other minor laneway) , medium, (isPrivatespace), low (adjacent to busy road), verylow (is busy road)

    ##classify rest of the point cloud to the closest point to this

    import pandas as pd
    import numpy as np

    # Assuming 'polydata' is your PolyData object and 'road_info-str_type' is the attribute of interest
    #data_array = polydata.point_data["road_info-str_type"]
    data_array = polydata.point_data["blocktype"]

    # If it's a structured or record array (common in PolyData), convert it to a plain numpy array
    if isinstance(data_array, np.void):  # np.void type is used for numpy record arrays
        data_array = np.array(data_array.tolist())

    # Find the unique values
    unique_values = np.unique(data_array)

    # Print the unique values
    print(unique_values)

    polydata = assign_conversion_potential(polydata)

    polydata = assign_disturbance_potential_proximity(polydata)

    
    #plotter.add_mesh(glyphs, scalars = 'disturbance-potential', cmap = 'viridis')
    #plotter.add_mesh(glyphs, scalars = 'distance to road', cmap = 'viridis')

    
    

    
    #ground

    #roof

        #quite, exposed
        #intensive, extensive

    #facade

    




        
        
    
    
    #buildings-dip
    #elevation

    #nearest
    
    ##To do:
        #consider green walls - high elevation, low elevation, quiet, busy
        #consider green roofs - 22
    
    #distance to 
    

    #laneways-forest
    #laneways-farmlane
    #laneways-parklane
    #verticalsu

    #islittle_streets

    #is

    #isparkingmedian

    #isprivate (value f 0)
    return polydata



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
            poly_data = pv.read(vtk_path)



            print(poly_data.point_data)


            poly_data = calculate_disturbance_groups(poly_data, site)

            poly_data.save(f'data/{site}/TESTflattened-{site}.vtk')

            # Visualization logic

            plotter = pv.Plotter()


            cube = pv.Cube()  # Create a cube geometry for glyphing
            glyphs = poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
            #plotter.add_mesh(glyphs, scalars = 'solar', cmap = 'jet')
            #plotter.add_mesh(glyphs, scalars = 'bucketed distance', cmap = 'viridis')
            
            plotter = pv.Plotter()
            plotter.add_mesh(glyphs, scalars = 'disturbance-potential', cmap = 'viridis')
            plotter.show()

            
            plotter = pv.Plotter()
            plotter.add_mesh(glyphs, scalars = 'bucketed distance', cmap = 'viridis')
            plotter.show()

            plotter = pv.Plotter()
            plotter.add_mesh(glyphs, scalars = 'disturbance-potential-proximity', cmap = 'viridis')
            plotter.show()
        


            plotter = pv.Plotter()

                    # Settings for better visualization
            plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
            light2 = pv.Light(light_type='cameralight', intensity=.5)
            light2.specular = 0.5  # Reduced specular reflection
            plotter.add_light(light2)
            plotter.enable_eye_dome_lighting()

            plotter.add_mesh(glyphs, scalars = 'distance groups', cmap = 'tab20c', clim=[0,19])
            plotter.show()



            """if state == 'potential':
                segmented_data, rest_of_points = process_site(site,  ('remnant ground', 'exotic ground', 'buildings', 'street features'))
                rest_of_points = adjust_colour(rest_of_points, .5, .3)
                siteDict = {'habitat potential' : segmented_data, 'rest of points' : rest_of_points}
                siteMultiBlock = packtoMultiblock.create_or_extend_multiblock(siteDict)
                #siteMultiBlock = packtoMultiblock.create_or_extend_multiblock({'rest of points' : rest_of_points}, siteMultiBlock)
                
                z_translation = GetMinElev(rest_of_points)
                print(f'z translation is {z_translation}')
                translation_amount = np.array([gridDist * stateNo, gridDist * siteNo, z_translation])
                
                site_polydata = pv.read(f'data/{site}/flattened-{site}.vtk')
                
                tree_positions = trees.process_urban_forest(site_polydata, site, state)
                print(f'Processing completed for site: {site}, state: {state}.')

                branchdf, grounddf, canopydf  = trees.create_canopy_array(site, state, tree_positions)
                print(f'created canopy df of length {len(branchdf)}')

                tree_dict = trees.create_canopy_dict(site_polydata, branchdf, grounddf, canopydf)
                canopyPoly = tree_dict['branches']
                #canopyPotential, restOfCanopy = filter_and_segment(canopyPoly, ['support','extend'])
                canopyPotential, restOfCanopy = filter_and_segment(canopyPoly, ['remnant canopy provider','exotic canopy provider'])
                canopyPotentialDict = {'canopy habitat potential' : canopyPotential, 'rest of canopy' : restOfCanopy}

                siteMultiBlock = packtoMultiblock.create_or_extend_multiblock(canopyPotentialDict, siteMultiBlock)
                
                siteMultiBlock.save(f'data/{site}/combined-{site}-{state}.vtm')
                
                glyphs.plotSite(plotter, siteMultiBlock, translation_amount)"""

            print(f'added to plotter: {site} - {state}')

    


if __name__ == "__main__":
    #sites = ['city', 'street', 'trimmed-parade']  # List of sites
    #sites = ['city']
    #sites = ['trimmed-parade']
    #sites = ['trimmed-parade']
    #sites = ['parade']
    #sites = ['trimmed-parade']
    sites = ['city', 'trimmed-parade','street']
    #sites = ['city']
    #sites = ['city']
    sites = ['street']
    states = ['potential']
    
    #states = ['baseline', 'now', 'trending']
    #states = ['baseline']
    main(sites, states)




