import a_voxeliser, a_logDistributor, a_create_resistance_grid, a_rewilding, a_urban_forest_parser, a_helper_functions
import os
import pandas as pd
import xarray as xr
import numpy as np
#choose site and scenario

sites = ['trimmed-parade', 'uni', 'city']
#sites = ['trimmed-parade']
sites = ['uni']
#sites = ['city','uni']
voxel_size = 1
stages = ['original','resistance']
var_number = int(input("\nWhat processing stage? 0 or 1: "))
stage = stages[var_number]

#sites = ['city']

for site in sites: # Replace with actual site identifiers
    
    
    print(f'processing {site} with voxel size {voxel_size}')
    filePATH = f'data/revised/final/{site}'
    os.makedirs(filePATH, exist_ok=True)


    if stage != 'resistance':
        #STEP 1: Create xarray of site from site_vtk and ground_vtk (voxel size 2.5)
        print(f'initial voxelisation with size {voxel_size}')
        ds = a_voxeliser.voxelize_polydata_and_create_xarray(site, voxel_size)
        ds.to_netcdf(f'{filePATH}/{site}_{voxel_size}_voxelArray_initial.nc')
        print(f'xarray saved to {filePATH}/{site}_{voxel_size}_voxelArray_initial.nc')

        #STEP 2: Add trees and poles to xarray
        print('adding trees and poles to xarray')
        ds, treeDF, poleDF = a_urban_forest_parser.get_resource_dataframe(site, ds, filePATH)
        ds.to_netcdf(f'{filePATH}/{site}_{voxel_size}_voxelArray_withTreesAndPoleLocations.nc')

        print(f"TEST: STEP 2: Add trees and poles to xarray, Unique values in xarray_dataset['poles_pole_number']: {np.unique(ds['poles_pole_number'].values)}")
        treeDF.to_csv(f'{filePATH}/{site}_{voxel_size}_treeDF.csv')
        poleDF.to_csv(f'{filePATH}/{site}_{voxel_size}_poleDF.csv')
        print(f'xarray saved to {filePATH}/{site}_{voxel_size}_voxelArray_withTreesAndPoleLocations.nc')

        print(f'converting xarray to polydata')
        polydata = a_helper_functions.convert_xarray_into_polydata(ds)
        polydata.save(f'{filePATH}/{site}_{voxel_size}_polydata.vtk')
        print(f'polydata saved to {filePATH}/{site}_{voxel_size}_voxelArray_withTreesAndPoleLocations.vtk')

        print('distributing logs')

        logLibraryPath = f'data/revised/trees/logLibraryStats.csv'
        logLibraryDF = pd.read_csv(logLibraryPath)

        #STEP 3: LOG NODE DISTRIBUTOR
        ds, grouped_roof_info, roof_info_df, logDF = a_logDistributor.process_roof_logs(ds, logLibraryDF)
        #save grouped_roof_info to filePath with name {site}_{voxel_size}_grouped_roof_info.csv
        grouped_roof_info.to_csv(f'{filePATH}/{site}_{voxel_size}_grouped_roof_info.csv')  
        #save log_info_df to filePath with name {site}_{voxel_size}_logInfo.csv
        logDF.to_csv(f'{filePATH}/{site}_{voxel_size}_logDF.csv')
        #save xarray to filePath with name {site}_{voxel_size}_voxelArray_final.nc
        ds.to_netcdf(f'{filePATH}/{site}_{voxel_size}_voxelArray_withLogs.nc')
        print(f"TEST: After STEP 3: LOG NODE DISTRIBUTOR, Unique values in xarray_dataset['poles_pole_number']: {np.unique(ds['poles_pole_number'].values)}")


    elif stage == 'resistance':
        print(f'skipping to resistance grid creation for {site}')
        print(f'loading xarray from {filePATH}/{site}_{voxel_size}_voxelArray_withLogs.nc')
        ds = xr.open_dataset(f'{filePATH}/{site}_{voxel_size}_voxelArray_withLogs.nc')

        #load treeDF, poleDF, logDF
        treeDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_treeDF.csv')
        poleDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_poleDF.csv')
        logDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_logDF.csv')
        print(f'loaded treeDF, poleDF, logDF')

    #STEP 4: CREATE RESISTANCE GRID
    

    print(f'STEP 4: CREATE RESISTANCE GRID, creating resistance grid for {site}')
    ds, subsetDS, updatedDFs = a_create_resistance_grid.get_resistance(ds, treeDF, poleDF, logDF, site)
    
    #SAVE RESULTS
    print(f'Saving {subsetDS} to {filePATH}/{site}_{voxel_size}_voxelArray_withResistanceSUBSET.nc')
    subsetDS.to_netcdf(f'{filePATH}/{site}_{voxel_size}_voxelArray_withResistanceSUBSET.nc')
    print(f'Saving {ds} to {filePATH}/{site}_{voxel_size}_voxelArray_withResistance.nc')
    ds.to_netcdf(f'{filePATH}/{site}_{voxel_size}_voxelArray_withResistance.nc')
    


    #STEP 5: GENERATE REWILDING NODES
    print(f'STEP 5: GETTING REWILDING NODES FOR SITE, creating resistance grid for {site}')
    ds, updatedDFs = a_rewilding.GetRewildingNodes(ds,updatedDFs['treeDF'],updatedDFs['poleDF'],updatedDFs['logDF'])
    ds.to_netcdf(f'{filePATH}/{site}_{voxel_size}_voxelArray_RewildingNodes.nc')

    for df in updatedDFs:
        print(f'Saving {df} to {filePATH}/{site}_{voxel_size}_{df}.csv')
        updatedDFs[df].to_csv(f'{filePATH}/{site}_{voxel_size}_{df}.csv', index=False)

    print(f'Rewilding nodes for {site} complete!')

    """print(f'creating rewilding distributions for {site}')
    ds = a_rewilding.get_rewilding(ds, site)
    ds.to_netcdf(f'{filePATH}/{site}_{voxel_size}_voxelArray_withRewilding.nc')"""

        