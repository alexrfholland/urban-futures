
import pyvista as pv
import numpy as np

def getSite(sites, states):

    polys = []
    multis = []

    for siteNo, site in enumerate(sites):

        for stateNo, state in enumerate(states):
            print(f'processing {site} - {state}...')

            #vtk_path = f'data/{site}/updated-{site}.vtk'
            vtk_path = f'data/{site}/defensive-{site}.vtk'
            poly_data = pv.read(vtk_path)


            siteMultiBlock = pv.read(f'data/{site}/combined-{site}-now.vtm')

            polys.append(poly_data)
            multis.append(siteMultiBlock)
            
    return (polys, multis)