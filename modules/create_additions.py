import pyvista as pv
import numpy as np

def create_pylon_tower(poly_data):
    print("Initiating create_pylon_tower function...")

    is_electricalpole = poly_data.point_data['ispylons'].astype(bool)
    print(f"Count of electrical points in original data: {np.sum(is_electricalpole)}")

    is_streetlight = poly_data.point_data['isstreetlight'].astype(bool)
    print(f"Count of streetlight points in original data: {np.sum(is_streetlight)}")

    # Create the mask with logical OR
    is_pylon_mask = is_electricalpole | is_streetlight

    pylon_points = poly_data.points[is_pylon_mask]
    print(f"Count of extracted pylon points: {len(pylon_points)}")

    if len(pylon_points) == 0:
        print("No matching points found. Returning the original poly_data.")
        return poly_data

    pylon_attributes = {name: array[is_pylon_mask] for name, array in poly_data.point_data.items()}
    print(f"Count of extracted attributes: {len(pylon_attributes)}")

    new_points = []
    new_attributes = {name: [] for name in pylon_attributes.keys()}

    for i, point in enumerate(pylon_points):
        for j in range(1, 9):
            new_point = point.copy()
            new_point[2] += j
            new_points.append(new_point)
            for name, array in pylon_attributes.items():
                new_value = array[i] if name != 'isSynthesised' else 'True'
                new_attributes[name].append(new_value)

    print(f"Count of newly created points: {len(new_points)}")

    new_points = np.array(new_points)
    new_attributes = {name: np.array(array) for name, array in new_attributes.items()}

    pylon_tower_poly_data = pv.PolyData(new_points)
    for name, array in new_attributes.items():
        pylon_tower_poly_data.point_data[name] = array

    merged_poly_data = pv.PolyData()
    merged_poly_data.points = np.vstack([poly_data.points, pylon_tower_poly_data.points])

    for name in poly_data.point_data.keys():
        if name in pylon_tower_poly_data.point_data.keys():
            merged_poly_data.point_data[name] = np.concatenate([poly_data.point_data[name], pylon_tower_poly_data.point_data[name]])
        else:
            merged_poly_data.point_data[name] = np.concatenate([poly_data.point_data[name], np.empty(len(pylon_tower_poly_data.points))])

    for name in pylon_tower_poly_data.point_data.keys():
        if name not in poly_data.point_data.keys():
            merged_poly_data.point_data[name] = np.concatenate([np.empty(len(poly_data.points)), pylon_tower_poly_data.point_data[name]])

    print(f"Total point count in updated poly_data: {merged_poly_data.n_points}")

        
    """plotter = pv.Plotter()

    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = merged_poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
    #plotter.add_mesh(glyphs, scalars = 'solar', cmap = 'jet')
    plotter.add_mesh(glyphs, scalars = 'intensive_green_roof-RATING', cmap = 'jet')


    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()"""
    return merged_poly_data

def create_pylon_towerB(poly_data):
    print("Initiating create_pylon_tower function...")

    # Obtain a Boolean mask for pylon points
    is_pylon_mask = poly_data.point_data['ispylons'].astype(bool)
    print(f"Count of pylon points in original data: {np.sum(is_pylon_mask)}")

    # Retrieve the pylon points and their associated attributes
    pylon_points = poly_data.points[is_pylon_mask]
    print(f"Count of extracted pylon points: {len(pylon_points)}")

    # Return original data if no matching points exist
    if len(pylon_points) == 0:
        print("No matching points found. Returning the original poly_data.")
        return poly_data

    # Extract attributes corresponding to pylon points
    pylon_attributes = {name: array[is_pylon_mask] for name, array in poly_data.point_data.items()}
    print(f"Count of extracted attributes: {len(pylon_attributes)}")

    # Initialize containers for new points and attributes
    new_points = []
    new_attributes = {name: [] for name in pylon_attributes.keys()}

    # Create new points above each pylon point
    for i, point in enumerate(pylon_points):
        for j in range(1, 9):  # Z-coordinate increment from 1 to 8 meters
            new_point = point.copy()
            new_point[2] += j
            new_points.append(new_point)

            # Assign attributes to new points
            for name, array in pylon_attributes.items():
                new_value = array[i] if name != 'isSynthesised' else 'True'
                new_attributes[name].append(new_value)

    print(f"Count of newly created points: {len(new_points)}")

    # Transform lists to numpy arrays
    new_points = np.array(new_points)
    new_attributes = {name: np.array(array) for name, array in new_attributes.items()}

    plotter = pv.Plotter()

    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
    #plotter.add_mesh(glyphs, scalars = 'solar', cmap = 'jet')
    plotter.add_mesh(glyphs, scalars = 'intensive_green_roof-RATING', cmap = 'jet')


    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()
    
    # Create a new PolyData object for the pylon tower
    pylon_tower_poly_data = pv.PolyData(new_points)
    for name, array in new_attributes.items():
        pylon_tower_poly_data.point_data[name] = array

    # Merge the new PolyData with the existing one
    merged_poly_data = poly_data + pylon_tower_poly_data

    plotter = pv.Plotter()

    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = merged_poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
    #plotter.add_mesh(glyphs, scalars = 'solar', cmap = 'jet')
    plotter.add_mesh(glyphs, scalars = 'intensive_green_roof-RATING', cmap = 'jet')


    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()



    
    print(f"Total point count in updated poly_data: {merged_poly_data.n_points}")

    return merged_poly_data


def create_pylon_tower2(poly_data):
    print("Initiating create_pylon_tower function...")

    # Obtain a Boolean mask for pylon points
    is_pylon_mask = poly_data.point_data['ispylons'].astype(bool)
    print(f"Count of pylon points in original data: {np.sum(is_pylon_mask)}")

    # Retrieve the pylon points and their associated attributes
    pylon_points = poly_data.points[is_pylon_mask]
    print(f"Count of extracted pylon points: {len(pylon_points)}")

    # Return original data if no matching points exist
    if len(pylon_points) == 0:
        print("No matching points found. Returning the original poly_data.")
        return poly_data

    # Extract attributes corresponding to pylon points
    pylon_attributes = {name: array[is_pylon_mask] for name, array in poly_data.point_data.items()}
    print(f"Count of extracted attributes: {len(pylon_attributes)}")

    # Initialize containers for new points and attributes
    new_points = []
    new_attributes = {name: [] for name in pylon_attributes.keys()}

    # Create new points above each pylon point
    for i, point in enumerate(pylon_points):
        for j in range(1, 9):  # Z-coordinate increment from 1 to 8 meters
            new_point = point.copy()
            new_point[2] += j  
            new_points.append(new_point)

            # Assign attributes to new points
            for name, array in pylon_attributes.items():
                new_value = array[i] if name != 'isSynthesised' else 'True'
                new_attributes[name].append(new_value)

    print(f"Count of newly created points: {len(new_points)}")

    # Transform lists to numpy arrays
    new_points = np.array(new_points)
    new_attributes = {name: np.array(array) for name, array in new_attributes.items()}

    # Add the new points and their attributes to the existing PolyData
    poly_data.points = np.vstack([poly_data.points, new_points])
    for name, array in new_attributes.items():
        if name in poly_data.point_data:
            poly_data.point_data[name] = np.concatenate([poly_data.point_data[name], array])
        else:
            poly_data.point_data[name] = array

    print(f"Total point count in updated poly_data: {poly_data.n_points}")

    return poly_data

# Usage:
# Assuming 'original_poly_data' is your input PolyData object
# tower_poly_data = create_pylon_tower(original_poly_data)

# Usage:
# Assuming 'original_poly_data' is your input PolyData object
# tower_poly_data = create_pylon_tower(original_poly_data)



if __name__ == "__main__":
    


    sites = ['city', 'trimmed-parade']
    #sites = ['street']
    for site in sites:
        
        
        vtk_path = f'data/{site}/flattened-{site}.vtk'
        poly_data = pv.read(vtk_path)
        create_pylon_tower(poly_data)
        
        """plotter = pv.Plotter()


        cube = pv.Cube()  # Create a cube geometry for glyphing
        glyphs = poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
        #plotter.add_mesh(glyphs, scalars = 'solar', cmap = 'jet')
        plotter.add_mesh(glyphs, scalars = 'intensive_green_roof-RATING', cmap = 'jet')


        # Settings for better visualization
        plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
        light2 = pv.Light(light_type='cameralight', intensity=.5)
        light2.specular = 0.5  # Reduced specular reflection
        plotter.add_light(light2)
        plotter.enable_eye_dome_lighting()
        plotter.show()"""



        """poly_data.save(f'data/{site}/flattened-{site}.vtk')
        print(f'saved {site}')"""


