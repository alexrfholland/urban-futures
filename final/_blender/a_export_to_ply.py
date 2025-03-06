import pyvista as pv
import a_resource_distributor_dataframes
import pandas as pd
import os
import f_vtk_to_ply_surfaces

def get_scenario_polydata(site, voxelSize, resolution, year):
    # Assuming processedDF is already loaded with tree instance data
    print(f'loading treeDF from {filePATH}/{site}_{voxelSize}_treeDF_{year}.csv')
    filepath = f'data/revised/final/{site}/{site}_{voxelSize}_treeDF_{year}.csv'
    treeDF = pd.read_csv(filepath)
    print(f'processing treeDF')
    locationDF, resourceDF = a_resource_distributor_dataframes.process_all_trees(treeDF, voxel_size=resolution)
    print(f'rotating resource structures')
    resourceDF = a_resource_distributor_dataframes.rotate_resource_structures(locationDF, resourceDF)
    print(f'converting to polydata')
    poly = a_resource_distributor_dataframes.convertToPoly(resourceDF)
    return poly

def get_base_polydata(site):
    sitePath = f'data/revised/final/{site}-siteVoxels-masked.vtk'
    roadPath = f'data/revised/final/{site}-roadVoxels-coloured.vtk'

    print(f'loading site polydata from {sitePath}')
    sitePoly = pv.read(sitePath)
    print(f'loading road polydata from {roadPath}')
    roadPoly = pv.read(roadPath)
    return sitePoly, roadPoly


site = 'trimmed-parade'
voxel_size = 1
years = [0, 10, 30, 60, 180]
resolutions = [0, .25]
filePATH = f'data/revised/final/{site}'
#check if ply path exists
# Create the ply directory if it doesn't exist
os.makedirs(os.path.dirname(f'{filePATH}/ply'), exist_ok=True)

sitePoly, roadPoly = get_base_polydata(site)

print(f'saving base polydata to {filePATH}/ply/{site}_base.ply')
#save just with pyvista ply
sitePoly.save(f'{filePATH}/ply/{site}_base.ply', texture='colors')
roadPoly.save(f'{filePATH}/ply/{site}_road.ply', texture='colors')

print(f'done')
"""
#years = [-1]
for year in years:

    sitePoly, roadPoly = get_base_polydata(site)


    if year == -1:
        vtkPath = f'{filePATH}/{site}_{voxel_size}_scenarioInitialPoly.vtk'
        print(f'loading initial polydata from {vtkPath}')

    else:
        vtkPath = f'{filePATH}/{site}_{voxel_size}_scenarioYR{year}.vtk'
        print(f'loading polydata from {vtkPath}')



    poly = pv.read(vtkPath)
    print(f'loaded polydata from {vtkPath}')
    polyExportPath = f'{filePATH}/ply/{site}_{voxel_size}_scenarioYR{year}.ply'
    f_vtk_to_ply_surfaces.export_polydata_to_ply(poly, polyExportPath)
    print(f'saved polydata to {polyExportPath}')
    plyPath = f'{filePATH}/ply/{site}_{voxel_size}_scenarioYR{year}.ply'


    if year != -1:
        #resource high resolution file
        print('getting high resolution polydata')
        for resolution in resolutions:
            print(f'getting high resolution polydata for resolution {resolution}')
            poly = get_scenario_polydata(site, voxel_size, resolution, year)
            polyfilePath = f'{filePATH}/ply/{site}_resources_scenarioYR{year}_res{resolution}.ply' 
            print(f'saving polydata to {polyfilePath}')
            f_vtk_to_ply_surfaces.export_polydata_to_ply(poly, polyfilePath)
            print(f'saved polydata to {polyfilePath}')
"""


