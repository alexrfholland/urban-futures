import pyvista as pv
import pandas as pd
from pvlib import solarposition
from datetime import datetime
import numpy as np

import pyvista as pv
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np


def calculate_distance_to_habitable_zone(poly_data):
    # Assuming the elevation data is stored in a point_data array called 'elevation'
    elevation = poly_data.point_data['elevation']

    elevLower = 15
    elevUpper = 30
    
    # Vectorized computation of distance to habitable zone
    below_habitable_zone = elevLower - elevation
    above_habitable_zone = elevation - elevUpper
    
    # Use numpy's where function to choose values based on elevation
    distance_to_habitable_zone = np.where(elevation < elevLower, below_habitable_zone, 
                                          np.where(elevation > elevUpper, above_habitable_zone, 0))
    
    # Add the distance_to_habitable_zone array to the point_data of poly_data
    poly_data.point_data['habitableZone'] = distance_to_habitable_zone

    print(f'habitable zone calculated')

    return poly_data



def getElev(source_vtm, target_polydata):
    # Ensure the source VTM contains a single point cloud
    assert len(source_vtm) == 1, "Source VTM should only contain one point cloud."
    source_point_cloud = source_vtm[0]

    # Initialize an empty numpy array for elevations
    elevations = np.empty(target_polydata.points.shape[0])

    # Build kd-tree only for x, y coordinates of the source point cloud
    source_xy = source_point_cloud.points[:, :2]
    kd_tree = cKDTree(source_xy)

    # Query kd-tree to find the closest point in the source point cloud for each point in the target point cloud
    _, closest_points_indices = kd_tree.query(target_polydata.points[:, :2])

    # Compute elevations using vectorized operations
    elevations = target_polydata.points[:, 2] - source_point_cloud.points[closest_points_indices, 2]

    # Add elevations array to the target point cloud point_data
    target_polydata.point_data['elevation'] = elevations


    """# Visualization logic
    plotter = pv.Plotter()

    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = target_polydata.glyph(geom=cube, scale=False, orient=False, factor=1)
    #plotter.add_mesh(glyphs, scalars = 'solar', cmap = 'jet')
    plotter.add_mesh(glyphs, scalars = 'elevation', cmap = 'jet')




    
    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()

    # Set the view to XY plane and show the plot
    plotter.view_xy()
    plotter.show()"""

    return target_polydata  # Optionally return the modified target point cloud


def stampCloud(site, point_cloud):
    extendedSite = f'data/{site}/{site}-extended.csv'
    
    # Load the CSV file
    extended_site_df = pd.read_csv(extendedSite, sep=',')
    print('getting extended point cloud attributes...')
    print(extended_site_df)

    # Rename the 'Illuminance (PCV)' column to 'solar'
    extended_site_df.rename(columns={'Illuminance (PCV)': 'solar'}, inplace=True)

    # Convert the CSV coordinates to a numpy array for use with cKDTree
    csv_coords = extended_site_df[['//X', 'Y', 'Z']].values
    
    # Create a k-D tree for the CSV coordinates
    tree = cKDTree(csv_coords)
    
    # Query the k-D tree to find the index of the nearest CSV point for each point in the point cloud
    distances, indices = tree.query(point_cloud.points)
    
    # Transfer all attributes except 'X', 'Y', 'Z', 'R', 'G', 'B' from the DataFrame to the point cloud
    for column in extended_site_df.columns:
        if column not in ['//X', 'Y', 'Z', 'R', 'G', 'B']:
            attribute_values = extended_site_df[column].values
            point_cloud.point_data[column.lower()] = attribute_values[indices]  # Use lower-case column name as the field name

    return point_cloud  # Return the updated point cloud

# Example usage:
# stampSolar('city')


def create_sun_sphere(site):
    # Load the site coordinates CSV
    site_coord = pd.read_csv('data/site projections.csv')

    # Get latitude and longitude from the CSV
    latitude = site_coord.loc[site_coord['Name'] == site, 'Latitude'].values[0]
    longitude = site_coord.loc[site_coord['Name'] == site, 'Longitude'].values[0]

    print(f'Latitude: {latitude}, Longitude: {longitude}')

    # Load the PLY file
    raw_ply_file_path = f'data/{site}/{site}-points.ply'
    ply_data = pv.read(raw_ply_file_path)

    # Get the minimum Z coordinate (elevation) from the PLY file
    elevation = ply_data.bounds[4]  # bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    print(f'Elevation: {elevation}')

    # Define the date and time for the winter solstice
    date_time = datetime(2022, 6, 21, 12, 0, 0)  # Winter Solstice at solar noon
    print(f'Date and Time: {date_time}')

    # Calculate the solar position
    solar_pos = solarposition.get_solarposition(date_time, latitude, longitude, elevation, method='nrel_numpy')
    solar_zenith = solar_pos['zenith'].values[0]
    solar_azimuth = solar_pos['azimuth'].values[0]

    print(f'Solar Zenith: {solar_zenith}, Solar Azimuth: {solar_azimuth}')

    # Convert solar zenith and azimuth to a direction vector
    theta = np.radians(90 - solar_zenith)  # Convert zenith to polar angle
    phi = np.radians(solar_azimuth)  # Azimuth is already in correct format
    sun_direction = np.array([
        np.sin(theta) * np.cos(phi),  # X-component
        np.sin(theta) * np.sin(phi),  # Y-component
        np.cos(theta)  # Z-component
    ])

    print(f'Sun Direction Vector: {sun_direction}')

    # Define a distance for placing the sun sphere
    sun_distance = 1000  # Arbitrary distance
    sun_position = sun_direction * sun_distance
    print(f'Sun Position: {sun_position}')

    # Create a sphere to represent the sun
    sun_sphere = pv.Sphere(radius=100, center=sun_position)  # Radius is arbitrary

    # Save the sun sphere as a PLY file
    sun_sphere_path = f'data/{site}/sun_sphere.ply'
    sun_sphere.save(sun_sphere_path)
    print(f'Sun sphere saved at {sun_sphere_path}')




    





if __name__ == "__main__":
    import glyphs as glyphMapper
    # Call the function with the site name
    #create_sun_sphere('street')
    #stampSolar('city')

      # Visualization logic
    plotter = pv.Plotter()
    
    site = 'street'
    vtk_file_path = f'data/{site}/flattened-{site}.vtk'
    point_cloud = pv.read(vtk_file_path)
    model = stampCloud(site, point_cloud)

    cube = pv.Cube()  # Create a cube geometry for glyphing
    glyphs = model.glyph(geom=cube, scale=False, orient=False, factor=1)
    #plotter.add_mesh(glyphs, scalars = 'solar', cmap = 'jet')
    plotter.add_mesh(glyphs, scalars = 'dip (degrees)', cmap = 'jet')




    
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