import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import f_SiteCoordinates, f_resource_distributor_meshes, f_vtk_to_ply_surfaces
import faiss
import os


# Set a fixed seed for reproducibility
SEED = 1
rng = np.random.default_rng(SEED)

def transfer_colours(site_vtk, treePolydata):    
    # Extract the RGB data from site_vtk
    print("Extracting RGB data from site_vtk...")
    rgb_data = site_vtk.point_data['colors']
    print(f"RGB data extracted: {rgb_data.shape}")

    # Build a KD-tree with the site_vtk points
    print("Building KD-tree from site_vtk points...")
    site_tree = cKDTree(site_vtk.points)
    print("KD-tree built successfully.")

    # Assign the RGB color of the nearest site_vtk point
    print("Assigning colours to tree voxels")
    print(f'finding for {treePolydata.points.shape[0]} points')
    distances, indices = site_tree.query(treePolydata.points)
    nearest_colors = rgb_data[indices]
    treePolydata.point_data['colors'] = nearest_colors

    return treePolydata

import hnswlib
import numpy as np
import pyvista as pv


###HNSWLIB
def transfer_colours_hnswlib(site_vtk, treePolydata, ef=100, M=12, batch_size=1000000, index_path=None):
    """
    Assign colors to tree voxels using HNSWLIB for approximate nearest neighbor search,
    with simple print statements for progress updates.

    Parameters:
    - site_vtk: PyVista PolyData containing site voxel data with 'colors'.
    - treePolydata: PyVista PolyData containing tree voxel points.
    - ef: Size of the dynamic list for the nearest neighbors (controls accuracy/speed).
    - M: Maximum number of outgoing connections in the graph (controls accuracy/speed).
    - batch_size: Number of points to process in each batch.
    - index_path: (Optional) Path to save/load the HNSWLIB index.

    Returns:
    - Modified treePolydata with assigned 'colors'.
    """
    # Convert points to float32 for HNSWLIB
    print("Converting site and tree points to float32...")
    site_points = site_vtk.points.astype(np.float32)
    tree_points = treePolydata.points.astype(np.float32)

    # Extract RGB data
    print("Extracting RGB data from site_vtk...")
    rgb_data = site_vtk.point_data['colors']
    print(f"RGB data extracted with shape: {rgb_data.shape}")

    # Dimensionality
    dim = site_points.shape[1]
    print(f"Dimensionality of points: {dim}")

    # Initialize HNSWLIB index
    num_elements = site_points.shape[0]
    print(f"Initializing HNSWLIB index with {num_elements} elements...")
    p = hnswlib.Index(space='l2', dim=dim)  # 'l2' for Euclidean distance

    # If index_path is provided and the file exists, load the index
    if index_path and os.path.isfile(index_path):
        try:
            print(f"Loading existing HNSWLIB index from '{index_path}'...")
            p.load_index(index_path, max_elements=num_elements)
            print("Index loaded successfully.")
        except Exception as e:
            print(f"Failed to load index from '{index_path}'. Initializing a new index. Error: {e}")
            initialize_and_add(p, site_points, num_elements, ef, M)
            # Save the newly created index if index_path is provided
            if index_path:
                save_index(p, index_path)
    else:
        # Initialize a new index and add items
        initialize_and_add(p, site_points, num_elements, ef, M)
        # Save the newly created index if index_path is provided
        if index_path:
            save_index(p, index_path)

    # Set ef parameter (controls query accuracy and speed)
    p.set_ef(ef)
    print(f"Set ef parameter to {ef}.")

    # Prepare to store nearest colors
    total_points = tree_points.shape[0]
    print(f"Total tree points to process: {total_points}")
    nearest_colors = np.empty((total_points, rgb_data.shape[1]), dtype=rgb_data.dtype)

    # Calculate the number of batches
    num_batches = int(np.ceil(total_points / batch_size))
    print(f"Starting nearest neighbor search in {num_batches} batches...")

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_points)
        batch = tree_points[start_idx:end_idx]

        print(f"Processing batch {batch_num + 1}/{num_batches}: Points {start_idx} to {end_idx - 1}")

        # Perform k-nearest neighbor search (k=1)
        labels, distances = p.knn_query(batch, k=1)

        # Assign nearest colors
        nearest_colors[start_idx:end_idx] = rgb_data[labels.flatten()]

        print(f"Completed batch {batch_num + 1}/{num_batches}")

    print("Nearest neighbor search completed.")

    # Assign the nearest colors to treePolydata
    print("Assigning colors to treePolydata...")
    treePolydata.point_data['colors'] = nearest_colors
    print("Color assignment to tree voxels completed.")

    return treePolydata

def initialize_and_add(index, site_points, num_elements, ef, M):
    """
    Initialize the HNSWLIB index and add site points.

    Parameters:
    - index: hnswlib.Index object.
    - site_points: NumPy array of site points.
    - num_elements: Total number of elements.
    - ef: ef_construction parameter.
    - M: M parameter.
    """
    print("Initializing the index...")
    index.init_index(max_elements=num_elements, ef_construction=ef, M=M)
    print("Index initialized.")

    print("Adding site points to the index...")
    index.add_items(site_points, np.arange(num_elements))
    print("Site points added to the index.")

def save_index(index, index_path):
    """
    Save the HNSWLIB index to the specified path.

    Parameters:
    - index: hnswlib.Index object.
    - index_path: Path to save the index.
    """
    # Ensure the directory exists
    index_dir = os.path.dirname(index_path)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
        print(f"Created directory '{index_dir}' for saving the index.")

    print(f"Saving index to '{index_path}'...")
    index.save_index(index_path)
    print("Index saved successfully.")
##

def get_tree_id(is_precolonial, tree_size):
    """
    Get a random tree_id based on the precolonial status and tree size.
    """
    tree_id = np.select(
        [
            (is_precolonial == False) & (tree_size == 'small'),
            (is_precolonial == False) & (tree_size == 'medium'),
            (is_precolonial == False) & (tree_size == 'large'),
            (is_precolonial == True) & (tree_size == 'small'),
            (is_precolonial == True) & (tree_size == 'medium'),
            (is_precolonial == True) & (tree_size == 'large')
        ],
        [
            rng.integers(4, 7, size=len(is_precolonial)),  # False, small
            rng.integers(1, 4, size=len(is_precolonial)),  # False, medium
            rng.integers(7, 15, size=len(is_precolonial)),  # False, large
            rng.integers(1, 5, size=len(is_precolonial)),  # True, small
            rng.integers(5, 11, size=len(is_precolonial)),  # True, medium
            rng.integers(11, 17, size=len(is_precolonial))  # True, large
        ],
        default=None
    )
    return tree_id

def run(site, index_path=None):
    # Load the dataframe
    print(f"Loading data for {site}...")
    resourceDF = pd.read_csv(f'data/revised/{site}-tree-locations.csv')
    print(f"Data loaded: {resourceDF.shape}")

    # Get the transform vector
    centre = f_SiteCoordinates.get_site_coordinates(site)
    midpoint = np.array([centre[0], centre[1], 0])
    translation_vector = -midpoint
    print(f"Translation vector for {site}: {translation_vector}")

    # Create new columns 'eastings', 'northings', 'elevation' and apply translation to original x, y, z
    resourceDF['eastings'] = resourceDF['x']
    resourceDF['northings'] = resourceDF['y']
    resourceDF['elevation'] = resourceDF['z']
    resourceDF[['x', 'y', 'z']] += translation_vector

    # Dictionary to store scenarios
    scenarios = {}
    scenarioNames = ['improved', 'native', 'native_old', 'degrade']

    # Create scenarios
    for scenarioName in scenarioNames:
        scenarios[scenarioName] = resourceDF.copy()

    # Modify each scenario using vectorized operations
    # Scenario 1: 'improved'
    scenarios['improved']['control'] = 'improved-tree'

    # Scenario 2: 'native'
    native_large_idx = (scenarios['native']['size'] == 'large') & (scenarios['native']['precolonial'] == False)
    scenarios['native'].loc[native_large_idx, 'size'] = 'small'
    scenarios['native'].loc[native_large_idx, 'precolonial'] = True
    scenarios['native'].loc[native_large_idx, 'treeID'] = get_tree_id(
        scenarios['native']['precolonial'][native_large_idx],
        scenarios['native']['size'][native_large_idx]
    )

    # Scenario 3: 'native_old'
    scenarios['native_old']['precolonial'] = True

    # Scenario 4: 'degrade'
    np.random.seed(SEED)
    degraded_df = scenarios['degrade']
    large_trees_idx = degraded_df['size'] == 'large'
    degraded_sample = degraded_df[large_trees_idx].sample(frac=1, random_state=SEED)
    n = len(degraded_sample)
    degraded_df.loc[degraded_sample.index[:n//4], 'size'] = 'senescing'
    degraded_df.loc[degraded_sample.index[n//4:n//2], 'size'] = 'snag'
    degraded_df.loc[degraded_sample.index[n//2:n//2 + n//10], 'size'] = 'fallen'
    degraded_df.loc[degraded_sample.index[n//2 + n//10:], 'size'] = 'propped'

    # Save each modified dataframe
    for scenarioName, scenarioDF in scenarios.items():
        scenarioDF.to_csv(f'data/revised/{site}-{scenarioName}-tree-locations.csv', index=False)
        print(f"Saved {site} scenario: {scenarioName}")

    # Load the VTK files for the current site
    print("Loading site VTK...")
    site_vtk = pv.read(f'data/revised/{site}-siteVoxels-masked.vtk')
    print("VTK files loaded successfully.")

    # Parameters for HNSWLIB
    ef = 100          # Reduced for faster processing
    M = 12            # Reduced for faster indexing
    batch_size = 1000000  # Adjust based on your system's memory
    index_path = f"indexes/{site}-hnsw_index.bin"  # Specify your desired index path

    # Process each scenario and save the VTK and PLY files
    for scenarioName, scenarioDF in scenarios.items():
        print(f"Processing scenario: {scenarioName} for {site}...")
        resourceVoxels = f_resource_distributor_meshes.distribute_meshes(scenarioDF)

        # Use the HNSWLIB-based function with progress monitoring
        resourceVoxels = transfer_colours_hnswlib(
            site_vtk,
            resourceVoxels,
            ef=ef,
            M=M,
            batch_size=batch_size,
            index_path=index_path  # Pass the index path if you want to save/load the index
        )

        # Save as VTK
        resourceVoxels.save(f'data/revised/{site}-{scenarioName}-treeVoxels.vtk')
        print(f"VTK saved for {site} scenario: {scenarioName}")

        # Save as PLY
        attributes = ['resource', 'radius', 'tree_number']
        filepath = f"data/revised/{site}-{scenarioName}-treeVoxels.ply"
        f_vtk_to_ply_surfaces.export_polydata_to_ply(resourceVoxels, filepath, attributes)
        print(f"PLY saved for {site} scenario: {scenarioName}")

    print(f"All scenarios processed for {site}.")

def run2(site):
    # Load the dataframe
    print(f"Loading data for {site}...")
    resourceDF = pd.read_csv(f'data/revised/{site}-tree-locations.csv')
    print(f"Data loaded: {resourceDF.shape}")

    # Get the transform vector
    centre = f_SiteCoordinates.get_site_coordinates(site)
    midpoint = np.array([centre[0], centre[1], 0])
    translation_vector = -midpoint
    print(f"Translation vector for {site}: {translation_vector}")

    # Create new columns 'eastings', 'northings', 'elevation' and apply translation to original x, y, z
    resourceDF['eastings'] = resourceDF['x']
    resourceDF['northings'] = resourceDF['y']
    resourceDF['elevation'] = resourceDF['z']
    resourceDF[['x', 'y', 'z']] += translation_vector

    # Dictionary to store scenarios
    scenarios = {}
    scenarioNames = ['improved', 'native', 'native_old', 'degrade']

    # Create scenarios
    for scenarioName in scenarioNames:
        scenarios[scenarioName] = resourceDF.copy()

    # Modify each scenario using vectorized operations
    # Scenario 1: 'improved'
    scenarios['improved']['control'] = 'improved-tree'

    # Scenario 2: 'native'
    native_large_idx = (scenarios['native']['size'] == 'large') & (scenarios['native']['precolonial'] == False)
    scenarios['native'].loc[native_large_idx, 'size'] = 'small'
    scenarios['native'].loc[native_large_idx, 'precolonial'] = True
    scenarios['native'].loc[native_large_idx, 'treeID'] = get_tree_id(
        scenarios['native']['precolonial'][native_large_idx],
        scenarios['native']['size'][native_large_idx]
    )

    # Scenario 3: 'native_old'
    scenarios['native_old']['precolonial'] = True

    # Scenario 4: 'degrade'
    np.random.seed(SEED)
    degraded_df = scenarios['degrade']
    large_trees_idx = degraded_df['size'] == 'large'
    degraded_sample = degraded_df[large_trees_idx].sample(frac=1, random_state=SEED)
    n = len(degraded_sample)
    degraded_df.loc[degraded_sample.index[:n//4], 'size'] = 'senescing'
    degraded_df.loc[degraded_sample.index[n//4:n//2], 'size'] = 'snag'
    degraded_df.loc[degraded_sample.index[n//2:n//2 + n//10], 'size'] = 'fallen'
    degraded_df.loc[degraded_sample.index[n//2 + n//10:], 'size'] = 'propped'

    # Save each modified dataframe
    for scenarioName, scenarioDF in scenarios.items():
        scenarioDF.to_csv(f'data/revised/{site}-{scenarioName}-tree-locations.csv', index=False)
        print(f"Saved {site} scenario: {scenarioName}")

    # Load the VTK files for the current site
    print("Loading site VTK...")
    site_vtk = pv.read(f'data/revised/{site}-siteVoxels-masked.vtk')
    print("VTK files loaded successfully.")

    # Process each scenario and save the VTK and PLY files
    for scenarioName, scenarioDF in scenarios.items():
        print(f"Processing scenario: {scenarioName} for {site}...")
        resourceVoxels = f_resource_distributor_meshes.distribute_meshes(scenarioDF)
        #resourceVoxels = transfer_colours(site_vtk, resourceVoxels)
        #resourceVoxels = transfer_colours_faiss_approx(site_vtk, resourceVoxels)
        #resourceVoxels = transfer_colours_hnswlib(site_vtk, resourceVoxels)
        
        # Save as VTK
        resourceVoxels.save(f'data/revised/{site}-{scenarioName}-treeVoxels.vtk')
        print(f"VTK saved for {site} scenario: {scenarioName}")

        # Save as PLY
        attributes = ['resource', 'radius', 'tree_number']
        filepath = f"data/revised/{site}-{scenarioName}-treeVoxels.ply"
        f_vtk_to_ply_surfaces.export_polydata_to_ply(resourceVoxels, filepath, attributes)
        print(f"PLY saved for {site} scenario: {scenarioName}")

def main():
    sites = ['city', 'trimmed-parade', 'uni']
    for site in sites:
        run(site)

if __name__ == '__main__':
    main()
