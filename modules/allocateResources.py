import glyphs
import packtoMultiblock
import pyvista as pv
import numpy as np
import cameraSetUp
import time

import os
import sys
import pyvista as pv

import boundary



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
            siteMultiBlock = loadSite(f'data/{site}/combined-{site}-{state}.vtm')
            print(f'{site} - {state} loaded')
            print_block_info(siteMultiBlock)


            potential = siteMultiBlock.get('habitat potential')




            # Assign the action, actionType and actionSubtype arrays to the point_data of poly_data
            action_array= potential.point_data['action']
            actionType_array = potential.point_data['actionType']
            actionSubtype_array = potential.point_data['actionSubtype']

            
            import pandas as pd




            z_translation = siteMultiBlock.get('rest of points').bounds[4]
            translation_amount = np.array([gridDist * stateNo, gridDist * siteNo, z_translation])

            glyphs.plotSite(plotter, siteMultiBlock, translation_amount)
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
    #sites = ['city', 'street', 'trimmed-parade']  # List of sites
    #sites = ['city', 'trimmed-parade']
    #sites = ['parade']
    #state = ['baseline', 'now', 'trending']
    #sites = ['city', 'street', 'trimmed-parade']
    sites = ['street']
    
    state = ['potential']
    #state = ['baseline']
    main(sites, state)


