import numpy as np
import CSF
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import pyvista as pv
import open3d as o3d  # Add this import

# Downsample the point cloud by taking every nth point
def downsample(coords, n=10):
    print(f"Downsampling the point cloud, taking every {n}th point...")
    downsampled_indices = np.arange(0, coords.shape[0], n)
    return coords[downsampled_indices], downsampled_indices

# Classify ground using CSF
def ClassifyGround(polydata):
    print(f"Classifying ground using CSF...")

    coords = np.array(polydata.points)
    xyz = np.vstack((coords[:, 0], coords[:, 1], coords[:, 2])).transpose()

    csf = CSF.CSF()
    csf.setPointCloud(xyz)

    csf.params.bSloopSmooth = True  # Perform slope post-processing
    csf.params.cloth_resolution = .5  # Cloth resolution
    csf.params.class_threshold = 0.5  # Classification threshold
    csf.params.height_diff_threshold = 0.3  # Height difference threshold
    csf.params.time_step = 0.65  # Time step
    csf.params.rigidness = 3  # Rigidness
    csf.params.iterations = 500  # Maximum number of iterations

    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    ground_indices = np.array(ground)
    non_ground_indices = np.array(non_ground)

    classification = np.zeros(coords.shape[0], dtype=int)
    classification[ground_indices] = 2
    classification[non_ground_indices] = 1

    polydata.point_data['Classification'] = classification
    return polydata

# Compute Height Above Ground (HAG)
def computeHAG(polydata):
    print("Starting Height Above Ground (HAG) calculation using KD-tree nearest neighbors...")

    coords = np.array(polydata.points)
    classification = polydata.point_data['Classification']

    ground_mask = classification == 2
    ground_points = coords[ground_mask]
    non_ground_points = coords[~ground_mask]

    if len(ground_points) == 0:
        print("No ground points available for HAG calculation.")
        return polydata

    tree = cKDTree(ground_points[:, :2])
    distances, indices = tree.query(non_ground_points[:, :2], k=5)
    avg_ground_elevation = np.mean(ground_points[indices, 2], axis=1)
    height_above_ground = non_ground_points[:, 2] - avg_ground_elevation

    hag = np.zeros(coords.shape[0])
    hag[~ground_mask] = height_above_ground

    polydata.point_data['HeightAboveGround'] = hag
    return polydata

# Apply DBSCAN clustering
def apply_dbscan(polydata, eps=50, min_samples=500, downsample_factor=10):
    """
    Apply DBSCAN clustering to the point cloud.
    
    Parameters:
    - eps: float, default=2
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is a critical parameter
        that determines the maximum spatial distance between points to be
        considered part of the same cluster.
    
    - min_samples: int, default=50
        The number of samples in a neighborhood for a point to be considered
        as a core point. This includes the point itself. It determines the
        minimum cluster size and helps distinguish between core points and
        noise points.
    
    - downsample_factor: int, default=10
        Factor by which to downsample the point cloud before clustering.
    """
    print("Applying DBSCAN clustering...")

    coords = np.array(polydata.points)
    height_above_ground = polydata.point_data['HeightAboveGround']

    non_ground_mask = height_above_ground > 1.5
    non_ground_points = coords[non_ground_mask]

    # Downsample the non-ground points
    downsampled_points, downsampled_indices = downsample(non_ground_points, n=downsample_factor)
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit(downsampled_points).labels_
    print(f"DBSCAN clustering resulted in {len(np.unique(cluster_labels))} clusters.")

    # Map cluster labels back to the full-resolution point cloud
    full_cluster_id = np.full(coords.shape[0], -1)
    if len(downsampled_points) > 0:
        tree = cKDTree(downsampled_points)
        _, full_indices = tree.query(non_ground_points, k=1)
        full_cluster_id[non_ground_mask] = cluster_labels[full_indices]

    polydata.point_data['ClusterID'] = full_cluster_id

    # Preview full-resolution points with Open3D, colored by ClusterID
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    colors = np.zeros((coords.shape[0], 3))
    unique_clusters = np.unique(full_cluster_id)
    for i, cluster_id in enumerate(unique_clusters):
        if cluster_id == -1:
            continue
        cluster_mask = full_cluster_id == cluster_id
        colors[cluster_mask] = np.random.rand(3)  # Assign random color to each cluster
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="ClusterID Preview")

    return polydata

# Process VTK meshes
def process_vtk_meshes(vtk_meshes):
    processed_meshes = []

    for polydata in vtk_meshes:
        # Step 1: Classify ground using CSF
        polydata = ClassifyGround(polydata)
        print("Ground classification completed.")
        #polydata.plot(scalars='Classification')

        # Step 2: Compute HAG using KD-Tree nearest neighbors
        polydata = computeHAG(polydata)
        print("HAG computation completed.")
        #polydata.plot(scalars='HeightAboveGround')

        # Step 3: Apply DBSCAN clustering
        """polydata = apply_dbscan(polydata)
        print("DBSCAN clustering completed.")
        polydata.plot(scalars='ClusterID')"""

        processed_meshes.append(polydata)

    return processed_meshes

if __name__ == "__main__":
    #vtk_file_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/obj/_converted/C4-17-0.25.vtk"
    vtk_file_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/obj/_converted/B4-22-0.25.vtk"
    mesh = pv.read(vtk_file_path)
    processed_meshes = process_vtk_meshes([mesh])
    print("Processing completed.")