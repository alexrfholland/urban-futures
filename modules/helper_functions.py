import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt



def get_attributes(poly_data):
    all_fields = poly_data.point_data.keys()
    searches = {}  # Dictionary to hold all searches
    current_search_name = ''
    current_search = {}  # Dictionary to hold the current search

    while True:
        print("\nFields available: ", list(all_fields))
        field_selection = input("Enter field name to explore or 'done' to finish: ")

        if field_selection.lower() == 'done':
            if current_search:
                searches[current_search_name] = current_search  # Save the current search before exiting
            break  # Exit the loop if user is done

        if field_selection not in all_fields:
            print(f"Invalid field name: {field_selection}")
            continue  # Skip to next iteration if field name is invalid

        unique_values = np.unique(poly_data.point_data[field_selection])
        print(f"\nUnique values in field '{field_selection}': ", unique_values)

        if np.issubdtype(poly_data.point_data[field_selection].dtype, np.number):
            print(f"Field '{field_selection}' is a scalar field.")
            range_or_value = input(f"Enter a single value or a range (min,max) for field '{field_selection}': ")
            if ',' in range_or_value:
                # Parse a range of values
                min_val, max_val = map(float, range_or_value.split(','))
                value = (min_val, max_val)
            else:
                # Parse a single value
                value = float(range_or_value)
        else:
            value = input(f"Enter a string value for field '{field_selection}': ")

        # Add the value to the current search dictionary
        if field_selection in current_search:
            current_search[field_selection].append(value)
        else:
            current_search[field_selection] = [value]
        print(f"Current search: {current_search}")

        # Prompt user for next action
        next_action = input("Enter 'add' to add more to this search, 'new' to start a new search, or 'done' to finish: ").lower()
        if next_action == 'new':
            if current_search:
                search_name = input("Enter a name for the current search: ")
                searches[search_name] = current_search  # Save the current search
            current_search = {}  # Start a new search
        elif next_action == 'done':
            if current_search:
                search_name = input("Enter a name for the current search: ")
                searches[search_name] = current_search  # Save the current search before exiting
            break  # Exit the loop
    

    print(f'searches are {searches}')

    return searches  # Return the list of searches


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

def filter_and_segment(poly_data, attribute_values_dict):
    segmented_data = {}  # Initialize an empty dictionary to store the segmented data
    all_filtered_indices = []  # List to store all indices that have been filtered
    
    for search_name, attr_values in attribute_values_dict.items():
        # Get indices of points that meet the conditions
        search_result = filter_points(poly_data, {search_name: attr_values})
        search_name, filtered_indices = search_result  # Unpack the result
        
        if len(filtered_indices) > 0:  # Only include searches with filtered indices length > 0
            # Create a new PolyData object with only the filtered points
            segmented_poly_data = poly_data.extract_points(filtered_indices)

            # Store the segmented PolyData object in the result dictionary using the search name as the key
            segmented_data[search_name] = segmented_poly_data

            # Collect all filtered indices for later use
            all_filtered_indices.extend(filtered_indices)
    
    # Get the indices of the rest of the points
    all_indices = set(range(len(poly_data.points)))
    rest_indices = all_indices.difference(all_filtered_indices)
    rest_indices = np.array(list(rest_indices))
    
    # Add a new PolyData object with the rest of the points
    rest_of_points = poly_data.extract_points(rest_indices)
        
    return segmented_data, rest_of_points



def TestSearch():
    # Assuming `poly_data` is your existing PolyData object
    site = 'city'
    vtk_path = f'data/{site}/flattened-{site}.vtk'


    poly_data = pv.read(vtk_path)  # Replace 'your_file.vtk' with your actual file


    print(poly_data.point_data)
    #print(np.unique(poly_data.point_data['blocktype']))
    #print(np.unique(poly_data.point_data['type']))
    #print(np.unique(poly_data.point_data['material']))


    # Attribute-value pairs to filter by, given as a dictionary
    #attr_values = {'buildings-dip': [(1.8, 1.8101531482918205)]}
    
    
    attr_values = {'roads': {'road_types-material' : ['HMA']}}

  
    # Get indices of points that meet the conditions
    filtered_indices = filter_points(poly_data, attr_values)
    
    # Perform further operations on the filtered indices, if needed
    print(f"Filtered indices for {filtered_indices[0]} : {filtered_indices[1]}")

def generate_color_dict(search_names):
    cmap = plt.cm.get_cmap('Set1', len(search_names))
    
    # Generate a color for each search name
    color_dict = {name: cmap(i) for i, name in enumerate(search_names)}
    
    return color_dict

def TestSplit(site):
    import glyphs as glyphMapper


        # Assuming `poly_data` is your existing PolyData object
    site
    vtk_path = f'data/{site}/flattened-{site}.vtk'


    poly_data = pv.read(vtk_path)  # Replace 'your_file.vtk' with your actual file

    # Call the get_attributes function to obtain the searches interactively
    #attr_values_list = get_attributes(poly_data)

    attr_values_list = {'green walls' : {'dip (degrees)': [(70.0, 90.0)], 'solar': [(0.2, 1.0)]}, #red
                         'extensive green roof' : {'dip (degrees)': [(0.0, 20.0)], 'extensive_green_roof-RATING': ['Excellent', 'Good']}, #green
                         'intensive green roof' : {'dip (degrees)': [(0.0, 20.0)], 'intensive_green_roof-RATING': ['Excellent', 'Good']}, 
                         'pushed green roof' : {'dip (degrees)': [(0.0, 20.0)], 'extensive_green_roof-RATING': ['Moderate']}, 
                         'modular' : {'road_types-material': ['Granitic Gravel']}, 
                         'modular parking' : {'isparking': [1.0]},
                         'powerlines': {'blocktype': ['powerline']},
                         'open-space': {'open_space-HA': [(0.0, 999999.0)]}                    
                        }

    color_dict = generate_color_dict(attr_values_list.keys())
    
    
    # List of dictionaries containing attribute-value pairs for filtering
    #greenattr_values_list = [{'material': 'Granitic Gravel'}, {'material': 'Dressed Bluestone'}]
    
    # Get a dictionary of segmented PyVista PolyData objects and the rest of the points
    segmented_data, rest_of_points = filter_and_segment(poly_data, attr_values_list)
    
    # Visualization logic
    plotter = pv.Plotter()
    
    cube = pv.Cube()  # Create a cube geometry for glyphing

    # Iterate through the segmented_data dictionary, adding each PolyData object to the plotter
    for key, segmented_poly_data in segmented_data.items():
        # Check if segmented_poly_data is not empty
        if segmented_poly_data.n_points > 0:
            # Determine the color for this segmented PolyData from the color dictionary
            color = color_dict[key]  # Lookup color using search name

            # Generate cube glyphs from the points in the segmented_poly_data
            glyphs = segmented_poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)

            # Add the glyphs to the plotter
            plotter.add_mesh(glyphs, color=color)  # Use color from color dictionary
        else:
            print(f"No points found for {key}")

    glyphMapper.add_mesh_rgba(plotter, rest_of_points.points, 1.0, rest_of_points.point_data["RGB"], rotation=70)

    
    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()

    # Set the view to XY plane and show the plot
    plotter.view_xy()
    plotter.show()

"""# Example usage
if __name__ == "__main__":
    site = 'park'
    TestSplit(site)
    #TestSearch()
"""






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

def process_site(site):
    vtk_path = f'data/{site}/flattened-{site}.vtk'
    poly_data = pv.read(vtk_path)

    # Define your attribute values list based on the site or other criteria
    attr_values_list = {'green walls' : {'dip (degrees)': [(70.0, 90.0)], 'solar': [(0.2, 1.0)]}, #red - encrust
                         'extensive green roof' : {'dip (degrees)': [(0.0, 20.0)], 'extensive_green_roof-RATING': ['Excellent', 'Good']}, #blue - encrust
                         'intensive green roof' : {'dip (degrees)': [(0.0, 20.0)], 'intensive_green_roof-RATING': ['Excellent', 'Good']}, #green - encrust
                         'modular parking' : {'isparking': [1.0]}, #orange - create
                         'powerlines': {'blocktype': ['powerline']}, #yellow - create
                         #additional: 
                         #support (existing old trees)
                         #extend (existing large nonnative trees, existing middle-aged native trees)


                         #'plant'
                         #'modular' : {'road_types-material': ['Granitic Gravel']},  #purple
                         #'open-space': {'open_space-HA': [(0.0, 999999.0)]}                    
                        }


    segmented_data, rest_of_points = filter_and_segment(poly_data, attr_values_list)
    color_dict = generate_color_dict(attr_values_list.keys())



    return segmented_data, rest_of_points, color_dict




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

            if state != 'baseline':
                """segmented_data, rest_of_points, color_dict = process_site(site)
                rest_of_points = adjust_colour(rest_of_points, .5, .3)
                siteMultiBlock = packtoMultiblock.create_or_extend_multiblock(segmented_data, new_block_name = 'segmented')
                siteMultiBlock = packtoMultiblock.create_or_extend_multiblock({'rest of points' : rest_of_points}, siteMultiBlock)"""

                rest_of_points = pv.read(f'data/{site}/flattened-{site}.vtk')
                rest_of_points = adjust_colour(rest_of_points, .5, .3)
                siteMultiBlock = packtoMultiblock.create_or_extend_multiblock({'rest of points' : rest_of_points})
                
                z_translation = GetMinElev(rest_of_points)
                print(f'z translation is {z_translation}')
                translation_amount = np.array([gridDist * stateNo, gridDist * siteNo, z_translation])
                
                site_polydata = pv.read(f'data/{site}/flattened-{site}.vtk')
                tree_positions = trees.process_urban_forest(site_polydata, site, state)
                print(f'Processing completed for site: {site}, state: {state}.')

                branchdf, grounddf, canopydf  = trees.create_canopy_array(site, state, tree_positions)
                print(f'created canopy df of length {len(branchdf)}')

                tree_dict = trees.create_canopy_dict(site_polydata, branchdf, grounddf, canopydf)
                siteMultiBlock = packtoMultiblock.create_or_extend_multiblock(tree_dict, siteMultiBlock, new_block_name='trees')
                
                siteMultiBlock.save(f'data/{site}/combined-{site}-{state}.vtm')
                
                glyphs.plotSite(plotter, siteMultiBlock, translation_amount)

            else: 

                #baselines
                baseline_tree_positions, baseline_site_polydata = getBaselines.GetConditions(site)
                baseline_tree_positions = trees.assign_tree_model_id(baseline_tree_positions)
                baseline_site_polydata = adjust_colour(baseline_site_polydata, .5, .3)

                z_translation = GetMinElev(baseline_site_polydata)
                print(f'z translation is {z_translation}')
                translation_amount = np.array([gridDist * stateNo, gridDist * siteNo, z_translation])

                print(f'translation amount is {translation_amount}')
                baselineBranchdf, basegrounddf, basecanopydf = trees.create_canopy_array(site, 'past', baseline_tree_positions)
                baseline_tree_dict = trees.create_canopy_dict(baseline_site_polydata, baselineBranchdf, basegrounddf, basecanopydf)

                baselineMultiBlock = packtoMultiblock.create_or_extend_multiblock(baseline_tree_dict, new_block_name='trees')
                baselineMultiBlock = packtoMultiblock.create_or_extend_multiblock({'rest of points' : baseline_site_polydata}, baselineMultiBlock)
                baselineMultiBlock.save(f'data/{site}/combined-{site}-{state}.vtm')
                

                glyphs.plotSite(plotter, baselineMultiBlock, translation_amount)

            print(f'added to plotter: {site} - {state}')

        


    # Additional settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.4))
    light2 = pv.Light(light_type='cameralight', intensity=.4)
    light2.specular = 0.3  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()

    cameraSetUp.setup_camera(plotter, gridDist, 600)






    plotter.show()



if __name__ == "__main__":
    sites = ['city']  # List of sites
    sites = ['city', 'street', 'trimmed-parade']  # List of sites
    #sites = ['city']
    #sites = ['trimmed-parade']
    #sites = ['trimmed-parade']
    sites = ['parade']
    #sites = ['street']
    states = ['trending']
    #states = ['preferable']
    
    #states = ['baseline', 'now', 'trending']
    #states = ['baseline']
    main(sites, states)




