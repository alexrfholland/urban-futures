from modules import getColorsAndAttributes, cullpoints, flattenmultiblock, stamp_cloud, stamp_shapefile, helper_functions

from scipy.spatial import cKDTree
import numpy as np
import pyvista as pv

def combine_blocks_and_visualize(vtkBlocks: dict, site: str, save = False) -> None:
    # Initialize a new MultiBlock to store the combined blocks
    combined_multi_block = pv.MultiBlock()
    
    def rename_attributes(block, block_type):
        for key in list(block.point_data.keys()):  # Use list to avoid RuntimeError due to size change
            if key not in ['x', 'y', 'z', 'RGB']:
                # Get the old attribute values
                old_values = block.point_data[key].copy()
                # Create a new attribute with the updated name and assign the old attribute's values to it
                block.point_data[f'{block_type}-{key}'] = old_values
                # Remove the old attribute
                block.point_data.remove(key)


    # Iterate through each block in the input dictionary
    for block_type, block in vtkBlocks.items():
        print(f"Processing block type: {block_type}")
        
        if isinstance(block, pv.MultiBlock):
            print(f"Number of sub_blocks in {block_type}: {len(block)}")
            
            for idx, sub_block in enumerate(block):
                print(f"Processing sub_block {idx+1} in {block_type}...")
                
                rename_attributes(sub_block, block_type)
                
                # Add a 'blocktype' attribute to all points in the sub_block
                sub_block.point_data['blocktype'] = [block_type] * sub_block.n_points
                
                # Add an 'id' attribute to all points in the sub_block
                sub_block.point_data['id'] = [idx + 1] * sub_block.n_points
                
                # Add the modified sub_block to the combined MultiBlock
                combined_multi_block.append(sub_block)
        
        elif isinstance(block, pv.PolyData):
            rename_attributes(block, block_type)
            
            # Add a 'blocktype' attribute to all points in the block
            block.point_data[f'blocktype'] = [block_type] * block.n_points
            
            # Add an 'id' attribute to all points in the block
            block.point_data['id'] = [1] * block.n_points
            
            # Add the modified block to the combined MultiBlock
            combined_multi_block.append(block)
        else:
            print(f"Unsupported data structure for block type: {block_type}")

    if save:
        # Visualize the combined MultiBlock
        plotter = pv.Plotter()
        for idx, block in enumerate(combined_multi_block):
            plotter.add_mesh(block, color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)
            plotter.add_text("This is a Sphere", position='upper_left', font_size=24, color='blue')
        
        plotter.show()

        # Save the combined MultiBlock
        combined_multi_block.save(f'data/{site}/{site}.vtm')
    
    return combined_multi_block

import glob

def find_shapefiles(folder_name):
    shapefile_paths = glob.glob(f"{folder_name}/*.shp")
    return shapefile_paths

def getBlocks(site):
    vtkBlocks = {}
    cullBlocks = []
    stampFilesPaths = []

    try:
        buildings = getColorsAndAttributes.transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("buildings", 10, site)[0]
        vtkBlocks['buildings'] = buildings
        cullBlocks.append(buildings)
        print(f'loaded {site} buildings')
    except Exception as e:
        print(f"An error for {site} occurred while processing buildings: {e}")

    try:
        #TODO: change the VTK files from 'street-furniture' to 'road_types'
        roadtypes = getColorsAndAttributes.transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("street-furniture", 10, site)[0]
        vtkBlocks['road_types'] = roadtypes
        cullBlocks.append(roadtypes)
        print(f'loaded {site} road_types')


    except Exception as e:
        print(f"An error for {site} occurred while processing road_types: {e}")
        stampFilesPaths.append('road_types')

    #get topo for where there are no other points
    try:
        topoResults = getColorsAndAttributes.transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("topography", 10, site)
        topo = topoResults[0]
        print(f'loaded {site} topo')

        try: 
            editedTopo = cullpoints.mask_and_filter_multi_block(topo, cullBlocks, 1, nearerKeep=False, ignoreZ=True)
            vtkBlocks['topo'] = editedTopo
            print(f'culled {site} and made edited topo')
            """plotter = pv.Plotter()
            plotter.add_mesh(editedTopo, point_size=5.0, render_points_as_spheres=True)
            plotter.view_xy()
            plotter.show()"""

        except Exception as e:
            print(f"An error for {site} occurred while culling topography: {e}")



    except Exception as e:
        print(f"An error for {site} occurred while processing topography: {e}")



    try:
        powerlines = getColorsAndAttributes.transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("powerlines", 100, site)[0]
        vtkBlocks['powerlines'] = powerlines
    except Exception as e:
        print(f"An error for {site} occurred while processing powerlines: {e}")

    print(f'data for {site} is {vtkBlocks}')

    multiBlock = combine_blocks_and_visualize(vtkBlocks, site)
    polyData = flattenmultiblock.flattenBlock(multiBlock)

    # Stamp errors VTK file - TODO: flatten typo and do on that first
    
    print(f'site is {site} and missing shapefiles are {stampFilesPaths}')
    for name in stampFilesPaths:
        paths = find_shapefiles(f'data/gh-shapefiles/{name}/')
        print(f"Found shapefiles to fill: {paths}")
        stamp_shapefile.read_and_plot(site, polyData, paths, 2000)
        #stamp_shapefile.read_and_plot(site, polyData, paths, 500)


    print('assinging other GIS info...')
    extraInfo = ['intensive_green_roof', 'extensive_green_roof', 'open_space', 'street_items', 'parking', 'canopy']
    for name in extraInfo:
        paths = find_shapefiles(f'data/shapefiles/{name}/')
        print(f"Found shapefiles to fill: {paths}")
        stamp_shapefile.read_and_plot(site, polyData, paths, 2000)
       # stamp_shapefile.read_and_plot(site, polyData, paths, 500)


    print('assigning extended cloud points...')
    polyData = stamp_cloud.stampCloud(site, polyData)

    print(f'calculatingElevation')
    polyData = stamp_cloud.getElev(topoResults[1], polyData)

    return polyData

if __name__ == "__main__":
    for site in ['parade', 'street']:
    #for site in ['parade']:
    #for site in ['city']:
        
        polydata = getBlocks(site)

        # Visualization logic remains the same
        plotter = pv.Plotter()

        polydata.point_data['isSynthesised'] = np.full(polydata.points.shape[0], False)

        #move road points down
        searchVals = {'roads': {'road_types-material' : ['HMA']}}
        roadIndexes = helper_functions.filter_points(polydata, searchVals)
        polydata.points[roadIndexes[1], 2] -= 2

        #move road points down
        searchVals = {'footpaths': {'road_types-type' : ['Footway']}}
        footIndexes = helper_functions.filter_points(polydata, searchVals)
        polydata.points[footIndexes[1], 2] += 1

        polydata.save(f'data/{site}/flattened-{site}.vtk')
        print(f'saved {site}')

        # Add the point clouds to the plotter
        #plotter.add_mesh(polydata, scalars='material', cmap='rainbow', point_size=5.0, render_points_as_spheres=True)
        """plotter.add_mesh(polydata, scalars='iscanopy', cmap='rainbow', point_size=5.0, render_points_as_spheres=True)
        plotter.view_xy()
        plotter.show()"""






