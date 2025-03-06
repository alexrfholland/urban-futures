import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors
import matplotlib as plt
import matplotlib.cm as cm
#import customcmaps
import colormaps as cmaps
import time

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

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = te - ts  # Now logging time in seconds
        else:
            print(f'time: {method.__name__!r}  {((te - ts)):2.2f} s')  # Now printing time in seconds with 's' for seconds
        return result
    return timed


def print_block_info(multi_block):
    for i, block in enumerate(multi_block):
        if block is not None:  # Check if the block is not None
            block_name = multi_block.get_block_name(i)  # Get the block name
            if isinstance(block, pv.core.pointset.PolyData):
                num_points = block.n_points  # Get the number of points in the block if it's a PolyData
                print(f'Block {i} Name: {block_name}, Number of Points: {num_points}')
            elif isinstance(block, pv.core.composite.MultiBlock):
                print(f'Block {i} Name: {block_name} is a MultiBlock with {len(block)} sub-blocks.')
                print_block_info(block)  # Recursively call the function for nested MultiBlocks
            else:
                print(f'Block {i} Name: {block_name} is of type: {type(block)}')
        else:
            print(f'Block {i} is None')


def add_mesh_rgba(plotter, positions, sizes, colors, base_voxel_size=1.0, rotation=0, isCol = True):
    positions_np = np.array(positions)
    
    # Check the type of sizes. If it's a single float or int, convert it into an array.
    if isinstance(sizes, (float, int)):
        sizes_np = np.full((len(positions_np),), sizes)
    else:
        sizes_np = np.array(sizes)
    
    colors_np = np.array(colors)
    
    # Check if colors are in RGBA format; if not, convert to RGBA by adding alpha channel
    if colors_np.shape[1] == 3:
        colors_np = np.hstack((colors_np, np.ones((colors_np.shape[0], 1))))

    ugrid = pv.PolyData(positions_np)
    ugrid.point_data["sizes"] = sizes_np
    ugrid.point_data["colors"] = colors_np
    
    glyph = pv.Box().scale(base_voxel_size / 2.0, base_voxel_size / 2.0, base_voxel_size / 2.0)
    
    # Create a rotation vector
    rotation_vector = np.array([0, 0, rotation])
    
    # Add the rotation vector to the ugrid
    ugrid.point_data['rotation'] = np.tile(rotation_vector, (len(positions_np), 1))
    
    glyphs = ugrid.glyph(geom=glyph, scale="sizes", orient="rotation")

    if isCol:    
        # Add the glyphs to the provided plotter
        plotter.add_mesh(glyphs, scalars="colors", rgb=True, opacity=1.0)
    else:
        plotter.add_mesh(glyphs, color = 'white', opacity=1.0)



def visualise_tree_outlines(plotter, multiblock):
        
    def get_color_index(is_precolonial, control):
        base_color = int(is_precolonial) * 4  # 0 or 4
        control_offset = {'reserve-tree': 0, 'street-tree': 2, 'park-tree': 1}[control]
        shade = base_color + control_offset
        return shade

    # Create an initial empty PolyData object to store all outlines
    all_outlines = pv.PolyData()

    for block in multiblock:
        if block is not None:
            # Create a wireframe cube that matches the block's bounds
            outline = pv.Box(bounds=block.bounds)

            # Get color index
            color_index = get_color_index(block.field_data['isPrecolonial'][0],
                                          block.field_data['_Control'][0])

            # Set the color index as an attribute of the outline
            outline.point_data['color_index'] = [color_index] * len(outline.points)

            # Append the outline to the all_outlines PolyData
            all_outlines = all_outlines.merge(outline)

    # Add the combined PolyData to the plotter
    #plotter.add_mesh(all_outlines, scalars='color_index', cmap=plt.cm.tab20c, style='wireframe', line_width=5)
    plotter.add_mesh(all_outlines, scalars='color_index',cmap = 'tab20c', clim=[0, 19], line_width = 4, style = 'wireframe', show_scalar_bar=False)


@staticmethod
def visualise_block_outlines(plotter, multiblock, scalar_field_name, cmap):
    # Create an initial empty PolyData object to store all outlines
    all_outlines = pv.PolyData()

    for block in multiblock:
        if block is not None:
            # Create a wireframe cube that matches the block's bounds
            bounds = block.bounds
            outline = pv.Box(bounds=block.bounds)

            # Transfer all attributes from field_data to each block's point data
            for key, value in block.field_data.items():
                outline.point_data[key] = [value[0]] * len(outline.points)

            # Append the outline to the all_outlines PolyData
            all_outlines = all_outlines.merge(outline)

    # Add the combined PolyData to the plotter
    plotter.add_mesh(all_outlines, scalars=scalar_field_name, cmap=cmap, style='wireframe', line_width=5)



def main():
    # Load the site_polydata from a VTK file
    site = 'city'
    vtk_path = f'data/{site}/flattened-{site}.vtk'
    site_polydata = pv.read(vtk_path)

    

    plotter = pv.Plotter()
    # Use add_mesh_rgba to both add site_polydata to the plotter and show the plot
    #add_mesh_rgba(p, site_polydata.points, 1.0, site_polydata.point_data["RGB"])
    
    cube = pv.Cube()
    glyphs = site_polydata.glyph(geom=cube, scale=False, orient=False, factor=1)
    plotter.add_mesh(glyphs, scalars='material', cmap='viridis')

    
    #add_mesh_scalar(p, site_polydata.points, 1.0, site_polydata.point_data["open_space_HA"], 'viridis')

    # Settings for better visualization
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    plotter.show()

"""multihue
    YlGn
    YlGnBu
    GnBu
    BuGn
    PuBuGn
    PuBu
    BuPu
    RdPu
    PuRd
    OrRd
    YlOrRd
    YlOrBr
singlehue
    Purples
    Blues
    Greens
    Oranges
    Reds
    Greys
"""

@timeit
def plotSite(plotter, multi_block, translation_amount):
    
    print(f'translation amount is {translation_amount}')
    trees = multi_block.get('trees')
    if trees is not None:
        add_trees_to_plotter(plotter, trees, translation_amount)
    
    segmented = multi_block.get('segmentedB')
    if segmented is not None:
        add_segmented_data(plotter, segmented, translation_amount)

    habitatPotential = multi_block.get('habitat potential')
    habitatPotentialTree = multi_block.get('canopy habitat potential')

    if habitatPotential is not None and habitatPotentialTree is not None:
        #visualise_habitat_potential2(plotter, habitatPotentialTree, habitatPotential, translation_amount,'turbo')
        #visualise_habitat_potential2(plotter, habitatPotentialTree, habitatPotential, translation_amount,'Pastel1')
        visualise_habitat_potential3(plotter, habitatPotentialTree, habitatPotential, translation_amount,'Pastel1')


    rest = multi_block.get('rest of points')
    if rest is not None:
        plotRestofSitePoints(plotter, rest, translation_amount, True)

    """restCanopy = multi_block.get('rest of canopy')
    if rest is not None:
        visualise_white(plotter, restCanopy, translation_amount)"""


    """if habitatPotential is not None:
        customCmap = customcmaps.create_custom_colormap(['YlGnBu','PuBuGn','YlOrRd'])
        visualise_habitat_potential(plotter, habitatPotential, translation_amount,'Set2')

    habitatPotentialTree = multi_block.get('canopy habitat potential')
    if habitatPotentialTree is not None:
        customCmap = customcmaps.create_custom_colormap(['Reds','Blues','Greens'])
        visualise_habitat_potential(plotter, habitatPotentialTree, translation_amount, 'Set1')"""
    

    
def generate_color_dict(search_names):
    print(f'getting colours for segments {search_names}')
    cmap = plt.cm.get_cmap('Set1', len(search_names))
    color_dict = {name: cmap(i) for i, name in enumerate(search_names)}
    return color_dict

@timeit

def visualise_white(plotter, segmented_poly_data, translation_amount):


    # Create a custom colormap
    cube = pv.Cube()  # Create a cube geometry for glyphing

    if segmented_poly_data.n_points > 0:

        segmented_poly_data.points += translation_amount  # Shift in x, y, and z directions
        glyphs = segmented_poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
        plotter.add_mesh(glyphs, color = 'white', scalars='segmentedCols',  show_edges = True)
         # Set clim to match the range of your data
    else:
        print(f"No points found for habitat potential")


def visualise_habitat_potential3(plotter, segmentedHabitat, segmentedStructures, translation_amount, cmapName):

    segmented_poly_data = segmentedStructures

    print(f"range of searches are {np.unique(segmented_poly_data['searchName'])}")

    # Create a custom colormap
    cube = pv.Cube()  # Create a cube geometry for glyphing

    if segmented_poly_data.n_points > 0:

        print(type(segmented_poly_data.point_data['segmentedCols']))
        
        segmented_poly_data.points += translation_amount  # Shift in x, y, and z directions
        glyphs = segmented_poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
        #plotter.add_mesh(glyphs, cmap=cmapName, scalars='segmentedCols',  show_edges = True, clim=[0, cmapName.N - 1])
        #plotter.add_mesh(glyphs, cmap=cmapName, scalars='searchName',  show_edges = True, edge_color='white')
        #plotter.add_mesh(glyphs, cmap=cmapName, scalars='searchName', opacity = .5)
        plotter.add_mesh(glyphs, cmap=cmapName, scalars='searchName', opacity = 1)
        #plotter.add_mesh(glyphs, cmap=cmapName, scalars='searchName', opacity = .5,line_width = 2, style = 'wireframe')
        #plotter.add_mesh(glyphs, color='grey', opacity = .2,line_width = 4, style = 'wireframe')
        
         # Set clim to match the range of your data
    else:
        print(f"No points found for habitat potential")





def visualise_habitat_potential2(plotter, segmentedHabitat, segmentedStructures, translation_amount, cmapName):

    segmented_poly_data = segmentedStructures + segmentedHabitat

    print(f"range of searches are {np.unique(segmented_poly_data['searchName'])}")

    # Create a custom colormap
    cube = pv.Cube()  # Create a cube geometry for glyphing

    if segmented_poly_data.n_points > 0:

        print(type(segmented_poly_data.point_data['segmentedCols']))
        
        segmented_poly_data.points += translation_amount  # Shift in x, y, and z directions
        glyphs = segmented_poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
        #plotter.add_mesh(glyphs, cmap=cmapName, scalars='segmentedCols',  show_edges = True, clim=[0, cmapName.N - 1])
        #plotter.add_mesh(glyphs, cmap=cmapName, scalars='searchName',  show_edges = True, edge_color='white')
        #plotter.add_mesh(glyphs, cmap=cmapName, scalars='searchName', opacity = .5)
        plotter.add_mesh(glyphs, cmap=cmapName, scalars='searchName', opacity = 1)
        #plotter.add_mesh(glyphs, cmap=cmapName, scalars='searchName', opacity = .5,line_width = 2, style = 'wireframe')
        #plotter.add_mesh(glyphs, color='grey', opacity = .2,line_width = 4, style = 'wireframe')
        
         # Set clim to match the range of your data
    else:
        print(f"No points found for habitat potential")



def visualise_habitat_potential(plotter, segmented_poly_data, translation_amount, cmapName):
    # Get the range of values within 'segmentedCols'
    min_value = min(segmented_poly_data.point_data['segmentedCols'])
    max_value = max(segmented_poly_data.point_data['segmentedCols'])
    
    print(f'Range of segmentedCols: {min_value} to {max_value}')

    # Create a custom colormap
    cube = pv.Cube()  # Create a cube geometry for glyphing

    if segmented_poly_data.n_points > 0:

        print(type(segmented_poly_data.point_data['segmentedCols']))


        min_value = min(segmented_poly_data.point_data['segmentedCols'])
        max_value = max(segmented_poly_data.point_data['segmentedCols'])
    
        print(f'Range of segmentedCols: {min_value} to {max_value}')

        segmented_poly_data.points += translation_amount  # Shift in x, y, and z directions
        glyphs = segmented_poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
        #plotter.add_mesh(glyphs, cmap=cmapName, scalars='segmentedCols',  show_edges = True, clim=[0, cmapName.N - 1])
        plotter.add_mesh(glyphs, cmap=cmapName, scalars='searchName',  show_edges = True, edge_color='white')
         # Set clim to match the range of your data
    else:
        print(f"No points found for habitat potential")

@timeit
def add_segmented_data(plotter, multi_block, translation_amount):
    import pyvista as pv
    import glyphs as glyphMapper
    
    color_dict = generate_color_dict(multi_block.keys())
    
    # Create or extend a multi_block    

    cube = pv.Cube()  # Create a cube geometry for glyphing

    for block_name in multi_block.keys():
        segmented_poly_data = multi_block[block_name]
        if segmented_poly_data.n_points > 0:
            color = color_dict[block_name]
            # Translate the segmented_poly_data points
            segmented_poly_data.points += translation_amount  # Shift in x, y, and z directions
            
            #adjust_colour(baseline_site_polydata, .5, .3)

            glyphs = segmented_poly_data.glyph(geom=cube, scale=False, orient=False, factor=1)
            plotter.add_mesh(glyphs, color=color)
        else:
            print(f"No points found for {block_name}")

    # Translate the rest_of_points
    
    
    
@timeit
def plotRestofSitePoints(plotter, rest_of_points, translation_amount, isColour=True, voxelSize = 1):
    rest_of_points.points += translation_amount

    if "RGBdesat" not in rest_of_points.point_data:
        adjust_colour(rest_of_points, .5, .3)

    add_mesh_rgba(plotter, rest_of_points.points, voxelSize, rest_of_points.point_data["RGBdesat"], rotation=70, isCol=isColour)


def extract_isosurface_from_polydata(polydata: pv.PolyData, spacing: tuple[float, float, float], isovalue: float = 1.0) -> pv.PolyData:
    """
    Generate an isosurface from a PolyData object using the Marching Cubes algorithm.

    Parameters:
    - polydata: A PyVista PolyData object containing the points to be used for surface extraction.
    - spacing: A tuple (dx, dy, dz) specifying the spacing between grid points.
    - isovalue: The scalar value at which to extract the surface (default is 1.0).

    Returns:
    - A PyVista PolyData object representing the extracted isosurface.
    """
    if polydata is not None and polydata.n_points > 0:
        print(f'PolyData has {polydata.n_points} points')
        
        # Extract points from the PolyData
        points = polydata.points
        x, y, z = points.T

        # Create the bounds and dimensions for the grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

        # Calculate grid dimensions
        dims = (
            int((x_max - x_min) / spacing[0]) + 1,
            int((y_max - y_min) / spacing[1]) + 1,
            int((z_max - z_min) / spacing[2]) + 1
        )

        # Create an ImageData grid
        grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=(x_min, y_min, z_min))

        # Assign scalar values to the grid (initially all zeros)
        scalars = np.zeros(grid.n_points)

        # Find the closest grid point for each PolyData point and assign a scalar value of 2
        for i, (px, py, pz) in enumerate(points):
            ix = int((px - x_min) / spacing[0])
            iy = int((py - y_min) / spacing[1])
            iz = int((pz - z_min) / spacing[2])
            grid_idx = ix + iy * dims[0] + iz * dims[0] * dims[1]
            scalars[grid_idx] = 2

        grid.point_data['values'] = scalars

        # Apply the contour filter (Marching Cubes algorithm)
        isosurface = grid.contour(isosurfaces=[isovalue], scalars='values')

        return isosurface

    else:
        print('PolyData is empty or None')
        return None

    
@timeit
def add_trees_to_plotter(plotter, multi_block: pv.MultiBlock, shift=None, showSite=False):
    def apply_shift(poly_data):        
        if shift is not None:
            poly_data.points += shift  # Shift in x, y, and z directions

    @timeit
    def processBranches(multi_block: pv.MultiBlock):
        # BRANCHES
        branchesPolydata = multi_block.get('branches')

        apply_shift(branchesPolydata)

        #print(f'branch polydata has the following properties {branchesPolydata.point_data}')


        # Assume branchesPolydata is your original data
        resource = branchesPolydata.point_data['resource']
        
        # Count and print the occurrences of each resource type
        dead_branch_count = np.count_nonzero(resource == 'dead branch')
        peeling_bark_count = np.count_nonzero(resource == 'peeling bark')
        perchable_branch_count = np.count_nonzero(resource == 'perchable branch')
        nonresource_count = np.count_nonzero(resource == 'other')

        print(f"'dead branch' count: {dead_branch_count}")
        print(f"'peeling bark' count: {peeling_bark_count}")
        print(f"'perchable branch' count: {perchable_branch_count}")
        print(f"'other' count: {nonresource_count}")

        # Assuming branchesPolydata is your original data
        resource = branchesPolydata.point_data['resource']

        # Create boolean masks for each category
        perchable_mask = resource == 'perchable branch'
        peeling_mask = resource == 'peeling bark'
        dead_mask = resource == 'dead branch'
        
        #resource_mask = np.logical_or(resource == 'dead branch', resource == 'peeling bark')
        non_resource_mask = resource == 'other'

        # Use the masks to extract points for each category
        perchableBranches = branchesPolydata.extract_points(perchable_mask)
        peelingBranches = branchesPolydata.extract_points(peeling_mask)
        deadBranches = branchesPolydata.extract_points(dead_mask)
        #resourceBranches = branchesPolydata.extract_points(resource_mask)
        nonResourceBranches = branchesPolydata.extract_points(non_resource_mask)

        return nonResourceBranches, perchableBranches, peelingBranches, deadBranches
    
    @timeit    
    def plotBranches(nonResourceBranches, perchableBranches, peelingBranches, deadBranches):
        non_resource_glyphs = nonResourceBranches.glyph(geom=cube, scale=False, orient=False, factor=0.1)
        #resource_glyphs = resourceBranches.glyph(geom=cube, scale=False, orient=False, factor=0.5)
        perchable_glyphs = perchableBranches.glyph(geom=cube, scale=False, orient=False, factor=0.2)
        peeling_glyphs = peelingBranches.glyph(geom=cube, scale=False, orient=False, factor=0.2)
        dead_glyphs = deadBranches.glyph(geom=cube, scale=False, orient=False, factor=0.2)

        voxelsize = [.3,.3,.3]

        # Add the meshes to the plotter
        # Check if glyphs are not None before adding to plotter
        # Check if glyphs are not empty before adding to plotter
        if non_resource_glyphs is not None and non_resource_glyphs.n_points > 0:
            #plotter.add_mesh(non_resource_glyphs, color = 'white', silhouette=dict(color='white', line_width=4.0))
            #plotter.add_mesh(non_resource_glyphs, color = 'white')

            isosurface = extract_isosurface_from_polydata(nonResourceBranches, [.2,.2,.2])
            if isosurface.n_points > 0:
                # Add the surface mesh to the plotter with only silhouette
                plotter.add_mesh(isosurface, 
                                opacity=0.5,  # Set opacity to 0 to make the fill invisible
                                color = '#e8e8e8',
                                silhouette=dict(color='#e8e8e8', line_width=2, feature_angle=10),
                                show_edges=False)  # Ensure no edges are shown
            
            #plotter.add_mesh(non_resource_glyphs, cmap='gray', scalars = 'ScanZ', show_scalar_bar=False, clim=[0, 20])
        else:
            print("non_resource_glyphs is None or empty")

        
        if perchable_glyphs is not None and perchable_glyphs.n_points > 0:
            isosurface = extract_isosurface_from_polydata(perchableBranches, voxelsize)
            if isosurface.n_points > 0:
                # Add the surface mesh to the plotter with only silhouette
                plotter.add_mesh(isosurface, 
                                opacity=0.1,  # Set opacity to 0 to make the fill invisible
                                color = '#d1d1d1',
                                silhouette=dict(color='#d1d1d1', line_width=2, feature_angle=10),
                                show_edges=False)  # Ensure no edges are shown
            
            #plotter.add_mesh(perchable_glyphs, cmap= cmaps.greys, scalars = 'ScanZ', show_scalar_bar=False, clim=[0, 20], silhouette=dict(color='grey', line_width=4.0))
            #plotter.add_mesh(perchable_glyphs, cmap='YlGn', scalars = 'ScanZ', show_scalar_bar=False, clim=[0, 20], silhouette=dict(color='green', line_width=4.0))

        else:
            print("perchable_glyphs is None or empty")

        if peeling_glyphs is not None and peeling_glyphs.n_points > 0:
            
            isosurface = extract_isosurface_from_polydata(peelingBranches, voxelsize)
            if isosurface.n_points > 0:
                # Add the surface mesh to the plotter with only silhouette
                plotter.add_mesh(isosurface, 
                                opacity=0.1,  # Set opacity to 0 to make the fill invisible
                                color = 'orange',
                                silhouette=dict(color='orange', line_width=2, feature_angle=10),
                                show_edges=False)  # Ensure no edges are shown
            else:
                print('Isosurface extraction resulted in an empty mesh')
            
            
            #plotter.add_mesh(peeling_glyphs, cmap='YlOrBr', scalars = 'ScanZ', show_scalar_bar=False, clim=[0, 20], silhouette=dict(color='brown', line_width=4.0))
            #plotter.add_mesh(peeling_glyphs, cmap=cmaps.oranges, scalars = 'ScanZ', show_scalar_bar=False, clim=[0, 20], silhouette=dict(color='orange', line_width=4.0))

        else:
            print("peeling_glyphs is None or empty")

        if dead_glyphs is not None and dead_glyphs.n_points > 0:
            
            isosurface = extract_isosurface_from_polydata(deadBranches, voxelsize)
            if isosurface.n_points > 0:
                # Add the surface mesh to the plotter with only silhouette
                plotter.add_mesh(isosurface, 
                                opacity=0.1,  # Set opacity to 0 to make the fill invisible
                                color = 'purple',
                                silhouette=dict(color='purple', line_width=2, feature_angle=10),
                                show_edges=False)  # Ensure no edges are shown
            else:
                print('Isosurface extraction resulted in an empty mesh')
            
            #plotter.add_mesh(dead_glyphs, cmap='RdPu', scalars = 'ScanZ', show_scalar_bar=False, clim=[0, 20], silhouette=dict(color='purple', line_width=4.0))
            #plotter.add_mesh(dead_glyphs, cmap=cmaps.purples, scalars = 'ScanZ', show_scalar_bar=False, clim=[-10, 20], silhouette=dict(color='purple', line_width=4.0))
        else:
            print("dead_glyphs is None or empty")

    @timeit
    def plotCanopyResources(multi_block: pv.MultiBlock):
        # CANOPY RESOURCES
        canopy_polydata = multi_block.get('canopy resources')
        apply_shift(canopy_polydata)
        resource = canopy_polydata.point_data['resource']

        hollow_mask = resource == 'hollow'
        epiphytes_mask = resource == 'epiphyte'

        hollow_polydata = canopy_polydata.extract_points(hollow_mask)
        epiphytes_polydata = canopy_polydata.extract_points(epiphytes_mask)

        if hollow_polydata is not None and hollow_polydata.n_points > 0:
            print(f'hollows has {hollow_polydata.n_points} points')
              # Apply shift to canopy_polydata
            isosurface = extract_isosurface_from_polydata(hollow_polydata, [1,1,1])
            if isosurface.n_points > 0:
                # Add the surface mesh to the plotter with only silhouette
                plotter.add_mesh(isosurface, 
                                opacity=1,  # Set opacity to 0 to make the fill invisible
                                color = 'magenta',
                                silhouette=dict(color='white', line_width=2, feature_angle=10),
                                show_edges=False)  # Ensure no edges are shown
            else:
                print('Isosurface extraction resulted in an empty mesh')

        if epiphytes_mask is not None and epiphytes_polydata.n_points > 0:
            print(f'epiphytes has {epiphytes_polydata.n_points} points')
              # Apply shift to canopy_polydata
            isosurface = extract_isosurface_from_polydata(epiphytes_polydata, [.8,.8,.8])
            if isosurface.n_points > 0:
                # Add the surface mesh to the plotter with only silhouette
                plotter.add_mesh(isosurface, 
                                opacity=1,  # Set opacity to 0 to make the fill invisible
                                color = 'cyan',
                                silhouette=dict(color='white', line_width=2, feature_angle=10),
                                show_edges=False)  # Ensure no edges are shown
            else:
                print('Isosurface extraction resulted in an empty mesh')

        else:
            print('no canopy points')



    def plotCanopyResources2(multi_block: pv.MultiBlock):
        # CANOPY RESOURCES
        canopy_polydata = multi_block.get('canopy resources')

        
        if canopy_polydata is not None and canopy_polydata.n_points > 0:
            print(f'canopy has {canopy_polydata.n_points} points')
            apply_shift(canopy_polydata)  # Apply shift to canopy_polydata
            canopyGlyphs = canopy_polydata.glyph(geom=cube, scale=False, orient=False, factor=1.5)
            plotter.add_mesh(canopyGlyphs, scalars='resource',  cmap=['magenta', 'cyan'], silhouette=dict(color='cyan', line_width=4.0), show_scalar_bar=False)

        else:
            print('no canopy points')




    def plotGreenVolume(multi_block: pv.MultiBlock):
        leaf_polydata = multi_block.get('green biovolume')
        if leaf_polydata is not None and leaf_polydata.n_points > 0:
            print(f'green biovolume has {leaf_polydata.n_points} points')
            
            # Apply shift to leaf_polydata (assuming apply_shift is defined elsewhere)
            apply_shift(leaf_polydata)

            isosurface = extract_isosurface_from_polydata(leaf_polydata, [1,1,1])

            if isosurface.n_points > 0:
                # Add the surface mesh to the plotter with only silhouette
                plotter.add_mesh(isosurface, 
                                color = '#d5ffd1',
                                opacity=0.1,  # Set opacity to 0 to make the fill invisible
                                #silhouette=dict(color='green', line_width=2, feature_angle=10),
                                show_edges=False)  # Ensure no edges are shown
            else:
                print('Isosurface extraction resulted in an empty mesh')
        else:
            print('No green biovolume points')

  

    def plotGreenVolume0(multi_block: pv.MultiBlock):
        leaf_polydata = multi_block.get('green biovolume')
        if leaf_polydata is not None and leaf_polydata.n_points > 0:
            print(f'green biovolume has {leaf_polydata.n_points} points')
            
            # Apply shift to leaf_polydata (assuming apply_shift is defined elsewhere)
            apply_shift(leaf_polydata)

             # Attach the scalar field to the polydata object
            leaf_polydata.point_data['values'] = 2

            # Apply the contour filter (Marching Cubes algorithm) directly on the polydata object
            isosurface = leaf_polydata.contour(isosurfaces=[1], scalars='values', method='marching_cubes')

            # Extract the surface
            # Add the surface mesh to the plotter with only silhouette
            plotter.add_mesh(isosurface, 
                            opacity=0.1,  # Set opacity to 0 to make the fill invisible
                            silhouette=dict(color='green', line_width=2, feature_angle=30),
                            show_edges=False)  # Ensure no edges are shown
        else:
            print('no green biovolume points')




    @timeit
    def plotGroundPoints(multi_block: pv.MultiBlock):
        # GROUND RESOURCES
        ground_cover_polydata = multi_block.get('ground resources')
        ground_cover_polydata.points[:, 2] += 1  # Adds 1 to the z-coordinate of all points

        if ground_cover_polydata is not None and ground_cover_polydata.n_points > 0:
            print(f'ground has {ground_cover_polydata.n_points} points')

            resource = ground_cover_polydata.point_data['resource']
        
            # Count and print the occurrences of each resource type
            fallen_log_count = np.count_nonzero(resource == 'fallen log')
            leaf_litter_count = np.count_nonzero(resource == 'leaf litter')

            print(f"'fallen log' count: {fallen_log_count}")
            print(f"'leaf litter' count: {leaf_litter_count}")


            # Create a mask for 'fallen log'
            fallen_logs_mask = ground_cover_polydata['resource'] == 'fallen log'
            fallen_logs_polydata = ground_cover_polydata.extract_points(fallen_logs_mask)

            # Create a mask for 'leaf litter'
            leaf_litter_mask = ground_cover_polydata['resource'] == 'leaf litter'
            leaf_litter_polydata = ground_cover_polydata.extract_points(leaf_litter_mask)

            # Apply shifts or any other operations to each polydata if needed
            apply_shift(fallen_logs_polydata)
            apply_shift(leaf_litter_polydata)

            # Create glyphs and add them to the plotter

            fallen_logs_glyphs = fallen_logs_polydata.glyph(geom=cube, scale=False, orient=False, factor=1)
            leaf_litter_glyphs = leaf_litter_polydata.glyph(geom=cube, scale=False, orient=False, factor=0.25)

            if fallen_logs_glyphs is not None and fallen_logs_glyphs.n_points > 0:
                plotter.add_mesh(fallen_logs_glyphs, color = 'plum', silhouette=dict(color='plum', line_width=4.0))
            else:
                print("fallen_logs_glyphs is None or empty")

            if leaf_litter_glyphs is not None and leaf_litter_glyphs.n_points > 0:
                plotter.add_mesh(leaf_litter_glyphs, color = 'peachpuff', silhouette=dict(color='peachpuff', line_width=4.0))
            else:
                print("leaf_litter_glyphs is None or empty")
            
        else:
            print('no ground points')

            


    def plotGroundPointsOLD(multi_block: pv.MultiBlock):
        # GROUND RESOURCES



        ground_cover_polydata = multi_block.get('ground resources')
        ground_cover_polydata.points[:, 2] += 1  # Adds 1 to the z-coordinate of all points
        #print(f'ground data has point data {ground_cover_polydata.point_data}')

        if ground_cover_polydata is not None and ground_cover_polydata.n_points > 0:
            print(f'ground has {ground_cover_polydata.n_points} points')
            apply_shift(ground_cover_polydata)  # Apply shift to ground_cover_polydata
            glyphs = ground_cover_polydata.glyph(geom=cube, scale=False, orient=False, factor=0.25)
            plotter.add_mesh(glyphs, scalars='resource', cmap=['peachpuff', 'plum'], show_scalar_bar=False, silhouette=dict(color='peachpuff', line_width=4.0))
            
        else:
            print ('no ground points')
        
    cube = pv.Cube()
    nonResourceBranches, perchableBranches, peelingBranches, deadBranches = processBranches(multi_block)
    plotBranches(nonResourceBranches, perchableBranches, peelingBranches, deadBranches)
    plotCanopyResources(multi_block)
    plotGroundPoints(multi_block)
    #plotGreenVolume(multi_block)


def add_trees_to_plotter2(plotter, multi_block: pv.MultiBlock, shift=None, showSite=False):
    def apply_shift(poly_data):
        
        if shift is not None:
            poly_data.points += shift  # Shift in x, y, and z directions

    cube = pv.Cube()  # Declare once to be used in multiple places

    # BRANCHES
    branchesPolydata = multi_block.get('branches')

    apply_shift(branchesPolydata)

    #print(f'branch polydata has the following properties {branchesPolydata.point_data}')


    # Assume branchesPolydata is your original data
    resource = branchesPolydata.point_data['resource']
    
    # Count and print the occurrences of each resource type
    dead_branch_count = np.count_nonzero(resource == 'dead branch')
    peeling_bark_count = np.count_nonzero(resource == 'peeling bark')
    perchable_branch_count = np.count_nonzero(resource == 'perchable branch')

    print(f"'dead branch' count: {dead_branch_count}")
    print(f"'peeling bark' count: {peeling_bark_count}")
    print(f"'perchable branch' count: {perchable_branch_count}")

    # Assuming branchesPolydata is your original data
    resource = branchesPolydata.point_data['resource']

    # Create boolean masks for each category
    perchable_mask = resource == 'perchable branch'
    peeling_mask = resource == 'peeling bark'
    dead_mask = resource == 'dead branch'
    
    #resource_mask = np.logical_or(resource == 'dead branch', resource == 'peeling bark')
    non_resource_mask = resource == 'other'

    # Use the masks to extract points for each category
    perchableBranches = branchesPolydata.extract_points(perchable_mask)
    peelingBranches = branchesPolydata.extract_points(peeling_mask)
    deadBranches = branchesPolydata.extract_points(dead_mask)
    #resourceBranches = branchesPolydata.extract_points(resource_mask)
    nonResourceBranches = branchesPolydata.extract_points(non_resource_mask)



    # Define geometry for glyphs
    cube = pv.Cube()

    # Create glyphs
    non_resource_glyphs = nonResourceBranches.glyph(geom=cube, scale=False, orient=False, factor=0.1)
    #resource_glyphs = resourceBranches.glyph(geom=cube, scale=False, orient=False, factor=0.5)
    perchable_glyphs = perchableBranches.glyph(geom=cube, scale=False, orient=False, factor=0.5)
    peeling_glyphs = peelingBranches.glyph(geom=cube, scale=False, orient=False, factor=0.5)
    dead_glyphs = deadBranches.glyph(geom=cube, scale=False, orient=False, factor=0.5)

    # Define the color map for resources
    #color_map = {'dead branch': 'red', 'perchable branch': 'green', 'peeling bark': 'orange'}

    """if resource_glyphs is not None and resource_glyphs.n_points > 0:
        plotter.add_mesh(resource_glyphs, cmap='Set2', scalars = 'resource')
    else:
        print("resource_glyphs is None or empty")"""



    # Add the meshes to the plotter
    # Check if glyphs are not None before adding to plotter
    # Check if glyphs are not empty before adding to plotter
    if non_resource_glyphs is not None and non_resource_glyphs.n_points > 0:
        #plotter.add_mesh(non_resource_glyphs, cmap='gray', show_scalar_bar=False, clim=[0, 1])
        plotter.add_mesh(non_resource_glyphs, scalars = 'ScanZ', cmap='gray', show_scalar_bar=False, clim=[.2, 1])
    else:
        print("non_resource_glyphs is None or empty")

    if perchable_glyphs is not None and perchable_glyphs.n_points > 0:
        plotter.add_mesh(perchable_glyphs, cmap='YlGn', scalars = 'ScanZ', show_scalar_bar=False, clim=[0, 20])
    else:
        print("perchable_glyphs is None or empty")

    if peeling_glyphs is not None and peeling_glyphs.n_points > 0:
        plotter.add_mesh(peeling_glyphs, cmap='YlOrBr', scalars = 'ScanZ', show_scalar_bar=False, clim=[0, 20])
    else:
        print("peeling_glyphs is None or empty")

    if dead_glyphs is not None and dead_glyphs.n_points > 0:
        plotter.add_mesh(dead_glyphs, cmap='RdPu', scalars = 'ScanZ', show_scalar_bar=False, clim=[0, 20])
    else:
        print("dead_glyphs is None or empty")


    # CANOPY RESOURCES
    canopy_polydata = multi_block.get('canopy resources')

    
    if canopy_polydata is not None and canopy_polydata.n_points > 0:
        print(f'canopy has {canopy_polydata.n_points} points')
        apply_shift(canopy_polydata)  # Apply shift to canopy_polydata
        canopyGlyphs = canopy_polydata.glyph(geom=cube, scale=False, orient=False, factor=1.5)
        plotter.add_mesh(canopyGlyphs, scalars='resource',  cmap=['magenta', 'cyan'], show_scalar_bar=False)

    else:
        print('no canopy points')

    # GROUND RESOURCES
    ground_cover_polydata = multi_block.get('ground resources')
    ground_cover_polydata.points[:, 2] += 1  # Adds 1 to the z-coordinate of all points
    #print(f'ground data has point data {ground_cover_polydata.point_data}')

    if ground_cover_polydata is not None and ground_cover_polydata.n_points > 0:
        print(f'ground has {ground_cover_polydata.n_points} points')
        apply_shift(ground_cover_polydata)  # Apply shift to ground_cover_polydata
        cubeGround = pv.Cube().triangulate().subdivide(2).clean().extract_surface()  # Smoother cube
        glyphs = ground_cover_polydata.glyph(geom=cubeGround, scale=False, orient=False, factor=0.25)
        plotter.add_mesh(glyphs, scalars='resource', cmap=['peachpuff', 'plum'], show_scalar_bar=False)
        
    else:
        print ('no ground points')

if __name__ == '__main__':
    main()
