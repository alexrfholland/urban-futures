import pandas as pd
import pyvista as pv
import sys
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Get the absolute path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from final.tree_processing import aa_tree_helper_functions
#load the templateDF

#extract the trees for comparison

#get the 




predictions = pd.read_csv('data/csvs/branchPredictions - adjusted.csv')
#group predictions by 'Tree.ID'
predictions_grouped = predictions.groupby('Tree.ID')

# Load templates once
output_dir = Path('data/revised/trees')
combined_templates = pd.read_pickle(output_dir / 'combined_templateDF.pkl')


largeTree = aa_tree_helper_functions.get_template(combined_templates, True, 'large', 'reserve-tree', 4)

mediumTree = aa_tree_helper_functions.get_template(combined_templates, True, 'medium', 'reserve-tree', 0)

highResFolder = Path('data/revised/final/stanislav')

largeTreeHighRes = pv.read(highResFolder / 'Tree 13 - Trunk.ply')
largeTreeHighResCanopy = pv.read(highResFolder / 'Tree 13 - Leaf.ply')
mediumTreeHighRes = pv.read(highResFolder / 'Tree 6 - Trunk.ply')
mediumTreeHighResCanopy = pv.read(highResFolder / 'Tree 6 - Leaf.ply')

canopies = {6 : mediumTreeHighResCanopy, 13 : largeTreeHighResCanopy}
trunks = {6 : mediumTreeHighRes, 13 : largeTreeHighRes}


for tree in [largeTree, mediumTree]:
    tree_id = tree['tree_id']   
    print(f"\nProcessing Tree ID: {tree_id}")
    
    mask = predictions['Tree.ID'] == tree_id
    tree_predictions = predictions.loc[mask]
    print(f"Found {len(tree_predictions)} predictions for Tree {tree_id}")
    
    treeDF = tree['template']
    print(f"\nTemplate columns available: {list(treeDF.columns)}")

    # Get the transform values
    transform_values = treeDF.iloc[0][['transformX', 'transformY', 'transformZ']]
    print(f"\nTransform values: x={transform_values['transformX']:.2f}, y={transform_values['transformY']:.2f}, z={transform_values['transformZ']:.2f}")
    
    # Create a 4x4 transformation matrix for translation
    transform_matrix = np.eye(4)  # Create 4x4 identity matrix
    transform_matrix[0:3, 3] = [transform_values['transformX'], 
                               transform_values['transformY'], 
                               transform_values['transformZ']]
    print("\nTransformation matrix created")
    
    # Apply the transformation matrix
    treeHighRes = trunks[tree_id].transform(transform_matrix)
    canopyHighRes = canopies[tree_id].transform(transform_matrix)
    print(f"\nApplied transformation to high-res models")
    print(f"Trunk points: {treeHighRes.n_points}")
    print(f"Canopy points: {canopyHighRes.n_points}")
    
    prediction_coords = tree_predictions[['x', 'y', 'z']].values
    ckd_tree = cKDTree(prediction_coords)
    print(f"\nBuilt KD-tree with {len(prediction_coords)} prediction coordinates")
    
    # Get points from the PolyData object
    points = treeHighRes.points  # This gives us the x, y, z coordinates
    
    # Find nearest predictions for each point
    distances, indices = ckd_tree.query(points, k=1)
    print(f"\nFound nearest neighbors for {len(points)} points")
    print(f"Average distance to nearest prediction: {np.mean(distances):.3f}")
    
    # Add all columns except x,y,z as point data arrays
    data_columns = [col for col in tree_predictions.columns if col not in ['x', 'y', 'z']]
    print(f"\nAdding {len(data_columns)} data columns to point data:")
    for column in data_columns:
        treeHighRes.point_data[column] = tree_predictions[column].values[indices]
        print(f"Added column: {column}")

    treeDFPoly = aa_tree_helper_functions.convertToPoly(treeDF)

    # Get dimensions using bounds
    bounds = treeDFPoly.bounds
    dims = [
        bounds[1] - bounds[0],  # x dimension
        bounds[3] - bounds[2],  # y dimension
        bounds[5] - bounds[4]   # z dimension
    ]
    print(f"\nTree {tree_id} dimensions: x={dims[0]:.2f}, y={dims[1]:.2f}, z={dims[2]:.2f}")


    """plotter = pv.Plotter()
    plotter.add_mesh(treeDFPoly, color='red')
    plotter.add_mesh(treeHighRes, scalars='indEst', cmap='viridis', clim=[1,1.5])
    plotter.add_mesh(canopyHighRes, color='green')
    plotter.show()"""

    #save the treeHighRes and the canopyHighRes as vtk
    treeHighRes.save(highResFolder / f'Tree {tree_id} - Trunk.vtk')
    canopyHighRes.save(highResFolder / f'Tree {tree_id} - Leaf.vtk')
    print(f"\nSaved VTK files for Tree {tree_id}")
    print("----------------------------------------")

  




