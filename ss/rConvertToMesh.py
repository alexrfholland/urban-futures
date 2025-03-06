import numpy as np
import pyvista as pv
import os
import matplotlib.pyplot as plt
from matplotlib import cm

def filter_points_by_property(polydata, property_name):
    # Extract the values of the property
    property_values = polydata[property_name]

    # Check if the property values are numeric or not
    if np.issubdtype(property_values.dtype, np.number):
        # For numeric types, use np.isnan to create the mask
        mask = ~np.isnan(property_values)
    else:
        # For non-numeric types, check for None or empty strings
        mask = property_values != None
        mask &= property_values != ""

    # Filter the points and corresponding property values
    filtered_points = polydata.points[mask]
    filtered_property_values = property_values[mask]
    
    return filtered_points, filtered_property_values

def create_colored_glyph_mesh(filtered_points, filtered_property_values, cube_size=1.0, cmap_name='Set1'):
    # Get unique property values and corresponding colors
    unique_values = np.unique(filtered_property_values)
    cmap = plt.colormaps[cmap_name] if hasattr(plt, "colormaps") else plt.get_cmap(cmap_name, len(unique_values))
    color_map = {value: cmap(i)[:3] for i, value in enumerate(unique_values)}

    # Create the point cloud as a PyVista PolyData object
    point_cloud = pv.PolyData(filtered_points)

    # Assign the corresponding color to each point
    color_data = np.array([color_map[value] for value in filtered_property_values]) * 255
    point_cloud.point_data['RGB'] = color_data.astype(np.uint8)

    # Create a cube to use as the glyph
    cube = pv.Cube().scale(cube_size)

    # Apply the glyph to the point cloud
    glyphs = point_cloud.glyph(scale=False, geom=cube, factor=cube_size)

    # Ensure colors are assigned correctly to the glyphs
    repeated_colors = np.repeat(color_data, cube.n_points, axis=0)
    glyphs.point_data['RGB'] = repeated_colors.astype(np.uint8)

    return glyphs

def main():
    # List of sites to process
    sites = ['street', 'city', 'trimmed-parade']

    for site in sites:
        # Define the input and output file paths
        vtk_filename = f'data/revised/processed/{site}-processed.vtk'
        ply_filename = f'data/revised/processed/{site}-colored-cubes.ply'

        # Check if the input file exists
        if not os.path.exists(vtk_filename):
            print(f"File not found: {vtk_filename}")
            continue

        # Load the PolyData from the VTK file
        print(f"Processing site: {site}")
        polydata = pv.read(vtk_filename)

        # Filter points by the 'road_types-type' property
        filtered_points, filtered_property_values = filter_points_by_property(polydata, 'road_types-type')

        # Create a colored mesh using glyphs
        combined_mesh = create_colored_glyph_mesh(filtered_points, filtered_property_values, cube_size=1.0)

        # Ensure the RGB array matches the number of points in the combined mesh
        if combined_mesh.n_points != len(combined_mesh.point_data['RGB']):
            raise ValueError("The number of points in the mesh does not match the number of RGB colors.")

        # Preview the combined mesh in PyVista without using scalars
        plotter = pv.Plotter()
        plotter.add_mesh(combined_mesh, scalars="RGB", rgb=True, show_edges=False)
        plotter.show_grid()
        plotter.show()

        # Save the combined mesh as a PLY file with the colors as a texture
        combined_mesh.save(ply_filename, binary=True, texture='RGB')
        print(f"Saved combined glyph mesh as PLY for {site}")

if __name__ == "__main__":
    main()
