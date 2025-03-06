import final.tree_processing.combine_resource_treeMeshGenerator as combine_resource_treeMeshGenerator
import pandas as pd
import pyvista as pv
import numpy as np
import os
from pathlib import Path

# Load log library
logLibrary = pd.read_pickle('data/treeOutputs/logLibrary.pkl')

# Filter out specific log numbers and add resource column
logLibraryDF = logLibrary[~logLibrary['logNo'].isin([1, 2, 3, 4])]
logLibraryDF['resource'] = 'fallen log'

# Get one row per log number
logInfo = logLibraryDF.groupby('logNo').first().reset_index()

# Process each log
for _, row in logInfo.iterrows():
    size = row['logSize']
    logID = row['logNo']
    
    # Set isosurface parameters
    isoSize = (0.15, 0.15, 0.15)
    
    # Get points for this log
    log_points = logLibraryDF[logLibraryDF['logNo'] == logID][['X', 'Y', 'Z']]

    print(f'processing log {logID}, size {size}, with {len(log_points)} points')
    
    # Create polydata from points
    polyLog = pv.PolyData(log_points.values)
    
    # Convert to mesh using isosurface
    mesh = combine_resource_treeMeshGenerator.extract_isosurface_from_polydata(
        polyLog,
        spacing=isoSize,
        resource_name='fallen log'
    )

    # Add resource point data to mesh
    mesh.point_data['resource'] = np.full(mesh.n_points, 'fallen log')

    print(f'mesh point data: {mesh.point_data}')

    if mesh is not None:
        mesh = combine_resource_treeMeshGenerator.fill_holes_in_mesh(mesh, hole_size=10000)
    
        if mesh.n_points > 0:
            #mesh.plot()

            # Create output directories
            output_dir = Path('data/revised/final/logMeshes')
            
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save mesh
            filename = f'size.{size}.log.{logID}.vtk'
            output_path = output_dir / filename
            mesh.save(str(output_path))
            print(f'saved {filename}')
