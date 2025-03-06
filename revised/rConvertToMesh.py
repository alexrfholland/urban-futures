import numpy as np
import pyvista as pv
import os
import matplotlib.pyplot as plt

def filter_polydata_by_isResource(polydata, property_name):
    property_values = polydata[property_name]
    
    # Print unique values in the isResource property
    unique_values = np.unique(property_values)
    print(f"Unique values in {property_name}: {unique_values}")
    
    # Filter based on string values representing '1.0' (True) and 'nan' (NaN)
    maskSite = property_values == 'nan'
    maskTree = property_values == '1.0'
    
    # Create two PolyData objects: one for 'nan' values and one for '1.0' values
    filtered_polydata_Site = polydata.extract_points(maskSite)
    filtered_polydata_Tree = polydata.extract_points(maskTree)
    
    return filtered_polydata_Site, filtered_polydata_Tree

    return filtered_polydata_Site, filtered_polydata_not_nan
def create_colored_glyph_mesh2(polydata, property_name, cube_size=1.0, cmap_name='Set1'):
    property_values = polydata[property_name]
    unique_values = np.unique(property_values)
    cmap = plt.get_cmap(cmap_name, len(unique_values))
    color_map = {value: cmap(i)[:3] for i, value in enumerate(unique_values)}

    color_data = np.array([color_map[value] for value in property_values]) * 255
    polydata.point_data['RGB'] = color_data.astype(np.uint8)

    cube = pv.Cube().scale(cube_size)
    glyphs = polydata.glyph(scale=False, geom=cube, factor=cube_size)

    repeated_colors = np.repeat(color_data, cube.n_points, axis=0)
    glyphs.point_data['RGB'] = repeated_colors.astype(np.uint8)

    return glyphs

def create_colored_glyph_mesh(polydata, property_name, is_tree=False, cube_size=1.0, cmap_name='Set1'):
    property_values = polydata[property_name]
    unique_values = np.unique(property_values)
    cmap = plt.get_cmap(cmap_name, len(unique_values))
    color_map = {value: cmap(i)[:3] for i, value in enumerate(unique_values)}

    # Create a color array with default values from the color_map
    color_data = np.array([color_map[value] for value in property_values]) * 255

    # If is_tree is True, override all colors to green
    if is_tree:
        color_data = np.array([[0, 255, 0] for _ in property_values])  # Green color

    polydata.point_data['RGB'] = color_data.astype(np.uint8)

    cube = pv.Cube().scale(cube_size)
    glyphs = polydata.glyph(scale=False, geom=cube, factor=cube_size)

    repeated_colors = np.repeat(color_data, cube.n_points, axis=0)
    glyphs.point_data['RGB'] = repeated_colors.astype(np.uint8)

    return glyphs

def main():
    sites = ['street', 'city', 'trimmed-parade']

    for site in sites:
        vtk_filename = f'data/revised/processed/{site}-processed.vtk'
        ply_filename_trees = f'data/revised/processed/{site}_tree.ply'
        ply_filename_combined = f'data/revised/processed/{site}-colored-cubes.ply'

        if not os.path.exists(vtk_filename):
            print(f"File not found: {vtk_filename}")
            continue

        print(f"Processing site: {site}")
        polydata = pv.read(vtk_filename)

        # Filter polydata based on 'isResource' being NaN and non-NaN
        filtered_polydata_combined, filtered_polydata_trees = filter_polydata_by_isResource(polydata, 'isResource')

        # Print the number of points in each PolyData object
        print(f"Number of tree points (NaN isResource): {filtered_polydata_trees.n_points}")
        print(f"Number of combined points (non-NaN isResource): {filtered_polydata_combined.n_points}")

        # Create the glyph mesh for isResource = NaN
        if filtered_polydata_trees.n_points > 0:
            tree_mesh = create_colored_glyph_mesh(filtered_polydata_trees, 'road_types-type', is_tree=True, cube_size=.2)
            tree_mesh.save(ply_filename_trees, binary=True, texture='RGB')
            print(f"Saved tree glyph mesh as PLY for {site}")

        # Create the glyph mesh for the combined non-NaN isResource
        if filtered_polydata_combined.n_points > 0:
            combined_mesh = create_colored_glyph_mesh(filtered_polydata_combined, 'road_types-type', cube_size=1.0)
            combined_mesh.save(ply_filename_combined, binary=True, texture='RGB')
            print(f"Saved combined glyph mesh as PLY for {site}")

        # Preview the combined mesh in PyVista
        """plotter = pv.Plotter()
        if filtered_polydata_trees.n_points > 0:
            plotter.add_mesh(tree_mesh, scalars="RGB", rgb=True, show_edges=False)
        if filtered_polydata_combined.n_points > 0:
            plotter.add_mesh(combined_mesh, scalars="RGB", rgb=True, show_edges=False)
        plotter.show_grid()
        plotter.show()"""

if __name__ == "__main__":
    main()
