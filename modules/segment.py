import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import stamp_shapefile

def create_additional_point_datas(polydata):
    
    #classify ground
    ##five tiers. veryhigh (openspace) high (convertableLaneway), medium (other minor laneway) , low (adjacent to busy road), verylow (is busy road)

    ##classify rest of the point cloud to the closest point to this




    
    
    
    
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
    test = 0




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

    Example:
    The following `search` dictionary will filter points where the 'temperature' attribute is either 20 or within the range of 30 to 40.
    {
        'temperature_search': {
            'temperature': [20, (30, 40)]
        }
    }
    """


    

    
    # Assuming there's only one key-value pair in the search dictionary
    search_name, query = list(search.items())[0]
        
    
    # Start with all indices
    valid_indices = set(range(len(poly_data.points)))
    print(f'Initial no of indices: {len(valid_indices)}')
    print(f'search is {search}')


    # Iterate over each key-value pair in the search dictionary
    for attr, values in query.items():
        
        # Initialize an empty set to collect indices that satisfy the current key-value pair
        current_valid_indices = set()

        print(f'attribute field is {attr}')
        
        # Iterate over each value in the list of values for the current key
        for value in values:
            print(f'searching for values matching {value}')
            if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
                # If value specifies a range, retain indices where the attribute falls within this range
                range_condition = (poly_data.point_data[attr] >= value[0]) & (poly_data.point_data[attr] <= value[1])
                current_valid_indices.update(np.where(range_condition)[0])
                print(f'Range condition for {attr} between {value[0]} and {value[1]}: {range_condition}')
            else:
                # If value is a single numeric value, retain indices where the attribute equals this value
                if isinstance(value, (int, float)):
                    numeric_condition = poly_data.point_data[attr] == value
                    current_valid_indices.update(np.where(numeric_condition)[0])
                    print(f'Numeric condition for {attr} equals {value}: {numeric_condition}')
                else:
                    # If value is a string, retain indices where the attribute equals this string
                    string_condition = poly_data.point_data[attr] == value
                    current_valid_indices.update(np.where(string_condition)[0])
                    print(f'String condition for {attr} equals {value}: {string_condition}')
        
        # Update valid_indices to retain only those indices that satisfy all key-value pairs processed so far
        valid_indices.intersection_update(current_valid_indices)
        print(f'Valid no. indices after processing {attr}: {len(current_valid_indices)}')
    
    # Convert the set of valid indices to an array and return it
    final_indices = np.array(list(valid_indices))
    print(f'Final no filtered indices: {len(final_indices)}')
    return (search_name, final_indices)

def filter_and_segment(poly_data: pv.PolyData, actions_list):
    
    def generate_color_scalar(json_data):
        type_scalar = 0  # Initialize a scalar value for types
        color_scalar_dict = {}  # Dictionary to hold the scalar values for each type and subtype
        
        for action in actions_list:

            action_dict = json_data[action]
            for type_name, type_details in action_dict.items():
                type_scalar_base = type_scalar * 4  # Each type gets a block of 4 scalar values
                subtype_scalar = 0  # Initialize a scalar value for subtypes
                for subtype_name in type_details['subtypes'].keys():
                    # Assign a scalar value to each subtype under the current type
                    color_scalar_dict[(type_name, subtype_name)] = type_scalar_base + subtype_scalar
                    print(f'colour scalar value for {type_name} and {subtype_name} is \n{type_scalar_base} + {subtype_scalar} = {type_scalar_base + subtype_scalar}')
                    subtype_scalar += 1  # Increment the scalar value for subtypes
                type_scalar += 1  # Increment the scalar value for types

        return color_scalar_dict
        
    # Load the JSON data from the specified file path
    with open('habitat-systems2.json', 'r') as file:
        json_data = json.load(file)

    # Initialize arrays to hold action, actionType and actionSubtype values
    #num_points = len(poly_data.points)
    num_points = poly_data.n_points

    action_array = np.full(num_points, 'none', dtype=object)
    actionType_array = np.full(num_points, 'none', dtype=object)
    actionSubtype_array = np.full(num_points, 'none', dtype=object)
    searchName_array = np.full(num_points, 'none', dtype=object)
    hasPotentialarray = np.full(len(poly_data.points), False, dtype=bool)





    """action_array = np.empty(num_points, dtype=str)
    actionType_array = np.empty(num_points, dtype=str)
    actionSubtype_array = np.empty(num_points, dtype=str)"""



    colScalar_array = np.full(num_points, -1, dtype=int)
    sucessfulSearches = []
    
        # Specify the columns
    columns = ['action', 'type', 'subtype', 'searchName','voxels detected', 'colourScalar', 'search']

    # Create an empty DataFrame
    logger_df = pd.DataFrame(columns=columns)

    
    
    color_scalar_dict = generate_color_scalar(json_data)

    print(f' top level keys are {json_data.keys()}')
    
    for action in actions_list:
        print(f'\n ######### \n ######### \n ########### {action} \n ######### \n ######### \n #########')
        action_dict = json_data[action]
        for type_name, type_dictionary in action_dict.items():

            print(f'\n ######### \n ########### Processing type: {type_name} \n ######### \n #########')
            
            # Initialize an empty dictionary for type_search_dict
            type_search_dict = {}
            # Iterate through the key-value pairs in type_details['search'] to construct type_search_dict
            for key, value_list in type_dictionary['search'].items():
                type_search_dict[key] = value_list  # Assign the entire list to the key
            
            print(f'\t\tType search parameters: {type_search_dict}')  # Print the type search parameters
            
            for subtype_name, subtype_dictionary in type_dictionary['subtypes'].items():
                searchNameIs = f'{action}-{type_name}-{subtype_name}'

                print(f'\n ################################# \n     SEARCHING SUBTYPE: {searchNameIs} \n')

    
                
                # Initialize an empty dictionary for subtype_search_dict
                subtype_search_dict = {}
                # Iterate through the key-value pairs in subtype_details to construct subtype_search_dict
                for key, value_list in subtype_dictionary.items():
                    subtype_search_dict[key] = value_list  # Assign the entire list to the key
                
                print(f'\t\t\tSubtype search parameters: {subtype_search_dict}')  # Print the subtype search parameters
                
                # Merge type and subtype search dictionaries
                merged_search_dict = {**type_search_dict, **subtype_search_dict}
                print(f'\t\t\tMerged search parameters: {merged_search_dict}')  # Print the merged search parameters
                
                # Get indices of points that meet the conditions

                searchName = f'{type_name} {subtype_name}'

                search = {searchName : merged_search_dict}

                print(f'searching for indicies matching {searchName}')

                search_result = filter_points(poly_data, search)
                search_name, filtered_indices = search_result  # Unpack the result

                ColScalar_value = color_scalar_dict[(type_name, subtype_name)]

                
                if len(filtered_indices) > 0:  # Only include searches with filtered indices length > 0
                    # Update the action, actionType and actionSubtype arrays based on the filtered indices
                    searchName_array[filtered_indices] = searchName
                    action_array[filtered_indices] = action
                    actionType_array[filtered_indices] = type_name
                    actionSubtype_array[filtered_indices] = subtype_name
                    sucessfulSearches.append(search)
                    hasPotentialarray[filtered_indices] = True
                    colScalar_array[filtered_indices] = ColScalar_value

                #add info
                # Create a new dictionary with the required information
                new_row = {
                    'action': action,
                    'type': type_name,
                    'subtype': subtype_name,
                    'searchName' : searchName,
                    'voxels detected': len(filtered_indices),
                    'scalar': ColScalar_value,
                    'search' : search
                }

                # Append the new row to the DataFrame
                new_row_df = pd.DataFrame([new_row])
                logger_df = pd.concat([logger_df, new_row_df], ignore_index=True)


         
    # Assign the action, actionType and actionSubtype arrays to the point_data of poly_data
    poly_data.point_data['action'] = action_array
    poly_data.point_data['actionType'] = actionType_array
    poly_data.point_data['actionSubtype'] = actionSubtype_array
    poly_data.point_data['segmentedCols'] = colScalar_array
    poly_data.point_data['searchName'] = searchName_array
    poly_data.point_data['hasPotential'] = hasPotentialarray


    print(f'colscalar unique values are: {np.unique(colScalar_array)}')

    print(f'action subtype array is" {actionSubtype_array}')

     # Use the 'hasPotential' field to create a mask for splitting the poly_data
    #potential_mask = poly_data.point_data['hasPotential']
    #rest_mask = ~potential_mask  # Negate the potential_mask to get the mask for the rest of the points

    # Create two separate PolyData objects for potentialPolydata and restOfPoints
    #potentialPolydata = poly_data.extract_points(potential_mask)
    #restOfPoints = poly_data.extract_points(rest_mask)

    ###ITS THE THRESHOLD RANGE
    threshold_range = (0, float('inf'))  # This will extract points with scalar values greater than or equal to 0

    """# Visualization logic
    plotter = pv.Plotter()


    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
    #plotter.add_mesh(glyphs, scalars = 'solar', cmap = 'jet')
    #plotter.add_mesh(glyphs, scalars = 'extensive_green_roof-RATING', cmap = 'tab20')
    plotter.add_mesh(glyphs, scalars = 'ispylons', cmap = 'tab20')


    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()"""
        

    # Use threshold method to create two separate PolyData objects for potentialPolydata and restOfPoints
    potentialPolydata = poly_data.threshold(scalars='segmentedCols', value=threshold_range)
    restOfPoints = poly_data.threshold(scalars='segmentedCols', invert=True, value=threshold_range)

    #restOfPoints: pv.PolyData = poly_data.extract_points(~hasPotentialarray)

    #potentialPolydata: pv.PolyData = poly_data.extract_points(hasPotentialarray)

    # Create a boolean mask where ispylons == True


    # Extract points using the boolean mask
    #





    print(f'Total number of points in potentialPolydata: {potentialPolydata.n_points}')
    
    print(f'successful searchers are: /n {sucessfulSearches}')

    unique_subtypes = np.unique(potentialPolydata.point_data['actionSubtype'])
    print(f'Reconised potential for: {unique_subtypes}')

    # Print the number of points for each unique value in point_data['actionSubtype']
    for subtype in unique_subtypes:
        subtype_mask = potentialPolydata.point_data['actionSubtype'] == subtype
        subtype_count = np.sum(subtype_mask)
        print(f'Number of points for actionSubtype "{subtype}": {subtype_count}')

    
    logger_df.to_csv('outputs/segementedresults.csv', index = False)
    # Return the two PolyData objects and the list of detected systems names

    return potentialPolydata, restOfPoints




def generate_color_dict(search_names):
    cmap = plt.cm.get_cmap('Set1', len(search_names))
    
    # Generate a color for each search name
    color_dict = {name: cmap(i) for i, name in enumerate(search_names)}
    
    return color_dict







def generate_color_dict(search_names):
    cmap = plt.cm.get_cmap('Set1', len(search_names))
    color_dict = {name: cmap(i) for i, name in enumerate(search_names)}
    return color_dict

def visualize_segmented_data(plotter, multi_block, rest_of_points, color_dict, translation_amount):
    import pyvista as pv
    import glyphs as glyphMapper    
    
    # Create or extend a multi_block    

    cube = pv.Cube()  # Create a cube geometry for glyphing

    for block_name in multi_block.keys():
        segmented_poly_data = multi_block[block_name]
        if segmented_poly_data.n_points > 0:
            color = color_dict[block_name]
            # Translate the segmented_poly_data points
            segmented_poly_data.points += translation_amount  # Shift in x, y, and z directions

            glyphs = segmented_poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
            plotter.add_mesh(glyphs, color=color)
        else:
            print(f"No points found for {block_name}")

    # Translate the rest_of_points
    rest_of_points.points += translation_amount
    glyphMapper.add_mesh_rgba(plotter, rest_of_points.points, 1.0, rest_of_points.point_data["RGB"], rotation=70)
    
    return multi_block  # Return the multi_block for further use

def process_site(site, search):
    #vtk_path = f'data/{site}/flattened-{site}.vtk'
    vtk_path = f'data/{site}/updated-{site}.vtk'
    poly_data = pv.read(vtk_path)

    print(f'{site} attributes are \n {poly_data.point_data}')
    
    segmented_data, rest_of_points = filter_and_segment(poly_data, search)
    


    return segmented_data, rest_of_points


def adjust_colour(polydata, saturation_percentage, value_increase):
    import cv2
    import numpy as np

    rgb = polydata.point_data["RGB"]
    # Normalize the RGB values to the range [0, 1] if necessary
    # rgb_normalized = rgb / 255.0
    rgb_normalized = rgb
    
    # Convert the RGB image to HSV
    hsv = cv2.cvtColor(np.float32(rgb_normalized.reshape(-1, 1, 3)), cv2.COLOR_RGB2HSV)
    
    # Desaturate the saturation channel
    hsv[:, 0, 1] = hsv[:, 0, 1] * (1 - saturation_percentage)
    
    # Increase the value channel to lighten the image.
    # Make sure the values do not exceed the maximum value of 1.0
    hsv[:, 0, 2] = np.clip(hsv[:, 0, 2] + value_increase, 0, 1)
    
    # Convert the HSV image back to RGB
    rgb_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Rescale the RGB values to the range [0, 255] if necessary
    # rgb_adjusted = (rgb_adjusted.reshape(-1, 3) * 255).astype(np.uint8)
    
    # Create a new field for the adjusted RGB values
    polydata.point_data["RGBdesat"] = rgb_adjusted
    
    return polydata


def GetMinElev (polydata):
    # Find minimum z-value in rest_of_points
    min_z = np.min(polydata.points[:, 2])
    z_translation = 0 - min_z

    print(f'min points are: {z_translation}')
    return z_translation

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

            if state == 'potential':
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
                
                glyphs.plotSite(plotter, siteMultiBlock, translation_amount)

            print(f'added to plotter: {site} - {state}')

        


    # Additional settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.4))
    light2 = pv.Light(light_type='cameralight', intensity=.4)
    light2.specular = 0.3  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()

    cameraSetUp.setup_camera(plotter, gridDist, 1000)






    plotter.show()



if __name__ == "__main__":
    #sites = ['city', 'street', 'trimmed-parade']  # List of sites
    #sites = ['city']
    #sites = ['trimmed-parade']
    #sites = ['trimmed-parade']
    #sites = ['parade']
    #sites = ['city', 'trimmed-parade','street']
    sites = ['street', 'trimmed-parade']
    #sites = ['city']
    #sites = ['street']
    states = ['potential']
    #states = ['baseline', 'now', 'trending']
    #states = ['baseline']
    main(sites, states)




