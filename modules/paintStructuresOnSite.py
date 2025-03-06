import glyphs, cameraSetUpRevised, packtoMultiblock
import painter.paint as paint

import numpy as np
import time

import os
import sys
import pyvista as pv
import deployable_manager
import defensive_structure_manager

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

@timeit
def main(sites, states):

    @timeit
    def loadSite(path):
        siteMultiBlock = pv.read(path)
        return siteMultiBlock

    plotter = pv.Plotter()

    # Usage:
    for siteNo, site in enumerate(sites):

        if (site != 'parade'):
            gridDist = 300
        else:
            gridDist = 800

        for stateNo, state in enumerate(states):
            print(f'processing {site} - {state}...')

            #siteMultiBlock = pv.read(f'data/{site}/combined-{site}-{state}.vtm')
            siteMultiBlock = loadSite(f'data/{site}/combined-{site}-potential.vtm')
            print(f'{site} - {state} loaded')
            print_block_info(siteMultiBlock)


            potential = siteMultiBlock.get('habitat potential')
            print(f'potential attributes are \n{potential.point_data}')

            norms = potential.point_data['buildings-Normals']
            print(f'unique norms are {np.unique(norms)}')
            print(f'building normal values are {norms}')


            
            import pandas as pd

            # Extract x, y, z components
            x, y, z = zip(*potential.points)

            # Create DataFrame
            df = pd.DataFrame({
                'urban system': potential.point_data['action'],
                'type': potential.point_data['actionType'],
                'subtype': potential.point_data['actionSubtype'],
                'nx': potential.point_data['nx'],
                'ny': potential.point_data['ny'],
                'nz': potential.point_data['nz'],
                'dip (degrees)' : potential.point_data['nz'],
                'dip direction (degrees)' : potential.point_data['nz'],
                'X': x,
                'Y': y,
                'Z': z
            })

        


            z_translation = siteMultiBlock.get('rest of points').bounds[4]
            translation_amount = np.array([gridDist * stateNo, gridDist * siteNo, z_translation])

            


            #get greeen roofs and walls
            #vtk_path = f'data/{site}/updated-{site}.vtk'
            vtk_path = f'data/{site}/offesnivePrep-{site}.vtk'
            poly_data = pv.read(vtk_path)

            mask = poly_data.point_data['fortifiedStructures'] != 'unassigned'


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
            poly_data.points += shifts

            #poly_data = defensive_structure_manager.defensiveStructureManager(poly_data)
            poly_data.points += translation_amount

            # Extract only the points where the mask is True
            habitatStructuresPOLY = pv.PolyData(poly_data.points[mask])

            # Assign the corresponding 'fortifiedStructures' data to these points
            habitatStructuresPOLY.point_data['fortifiedStructures'] = poly_data.point_data['fortifiedStructures'][mask]


            
            from utilities.getCmaps import create_colormaps     
            colormaps = create_colormaps()
       
            siteStateMULTIBLOCK = pv.read(f'data/{site}/combined-{site}-now.vtm')
            restofPointsPOLY = siteStateMULTIBLOCK.get('rest of points')
            glyphs.plotRestofSitePoints(plotter, restofPointsPOLY, translation_amount, voxelSize=1)


             # Visualization code
            cube = pv.Cube()  # Create a cube geometry for glyphing
            glyphsClass = habitatStructuresPOLY.glyph(geom=cube, scale=False, orient=False, factor=1)

            # Settings for better visualization
            #plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
            #light2 = pv.Light(light_type='cameralight', intensity=.5)
            #light2.specular = 0.1  # Reduced specular reflection
            #plotter.add_light(light2)


            cameraSetUpRevised.setup_camera(plotter, gridDist, 600)



            plotter.enable_eye_dome_lighting()
            plotter.add_mesh(glyphsClass, scalars='fortifiedStructures', cmap=colormaps['discrete-1-7-section-muted'],silhouette=dict(color='grey', line_width=4.0))
            

            #plotter.add_mesh(glyphsClass, scalars='fortifiedStructures', cmap='tab10')

            #plotter.show()


            #habitatStructureMultiBlocks = paint.distributeStructuresAndResources(df)
            #glyphs.add_trees_to_plotter(plotter, habitatStructureMultiBlocks, translation_amount)


            #glyphs.plotSite(plotter, siteMultiBlock, translation_amount)


            ##get deployables
            deployablesMULTIBLOCK = deployable_manager.createDeployables(site, plotter)
            glyphs.add_trees_to_plotter(plotter, deployablesMULTIBLOCK, translation_amount)

            #add trees
            #PrefSiteMultiBlock = pv.read(f'data/{site}/combined-{site}-preferable.vtm')

            treePOLYMULTIBLOCK = siteStateMULTIBLOCK.get('trees')
            
            glyphs.add_trees_to_plotter(plotter, treePOLYMULTIBLOCK, translation_amount)
            

            
            treePOLYMULTIBLOCK
            deployablesMULTIBLOCK

            multiblock = pv.MultiBlock()
            multiblock['branches'] = treePOLYMULTIBLOCK['branches'] + deployablesMULTIBLOCK['branches'] 
            multiblock['ground resources'] = treePOLYMULTIBLOCK['ground resources'] + deployablesMULTIBLOCK['ground resources']
            multiblock['canopy resources'] = treePOLYMULTIBLOCK['canopy resources'] + deployablesMULTIBLOCK['canopy resources']
            multiblock['rest of points'] = restofPointsPOLY
            multiblock['structure voxels'] = habitatStructuresPOLY

            multiblock.save(f'data/{site}/structures-{site}-{state}.vtm')

            


        
            ##trees
                        
            print(f'added to plotter: {site} - {state}')

        

    # Additional settings for better visualization
    """plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.4))
    light2 = pv.Light(light_type='cameralight', intensity=.4)
    light2.specular = 0.3  # Reduced specular reflection
    plotter.add_light(light2)
    plotter.enable_eye_dome_lighting()
    """
    




    plotter.show()


if __name__ == "__main__":
    #sites = ['city', 'street', 'trimmed-parade']  # List of sites
    #sites = ['city', 'trimmed-parade']
    #sites = ['parade']
    #state = ['baseline', 'now', 'trending']
    sites = ['street', 'city','trimmed-parade']
    #sites = ['city']
    #sites = ['city']

    #sites = ['street']
    #sites = ['parade']
    
    state = ['now']
    #state = ['baseline']
    main(sites, state)


