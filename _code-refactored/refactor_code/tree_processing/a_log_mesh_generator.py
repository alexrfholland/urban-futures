import sys
from pathlib import Path

TREE_PROCESSING_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = REPO_ROOT / "_code-refactored"
FINAL_DIR = REPO_ROOT / "final"

for import_root in (TREE_PROCESSING_DIR, FINAL_DIR, CODE_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import combine_resource_treeMeshGenerator
import pandas as pd
import pyvista as pv
import numpy as np
import os
from refactor_code.paths import log_mesh_vtk_dir

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
            output_dir = log_mesh_vtk_dir()
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save mesh
            filename = f'size.{size}.log.{logID}.vtk'
            output_path = output_dir / filename
            mesh.save(str(output_path))
            print(f'saved {filename}')
