import open3d as o3d
import numpy as np

# Function to compute the bounding box and add it to a list of geometries
def add_bounding_box(node, node_info, geometries):
    # Create the axis-aligned bounding box
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=node_info.origin,
        max_bound=node_info.origin + node_info.size
    )
    
    # Set color for the bounding box
    bbox.color = (0, 1, 0)  # Green color for bounding boxes

    print(f"Adding bounding box at depth {node_info.depth}")
    
    # Add the bounding box to the list of geometries
    geometries.append(bbox)
    
    return True  # Continue traversal

# Create a random point cloud
def generate_random_point_cloud(num_points=10000):
    # Using more points to force better subdivision
    points = np.random.rand(num_points, 3) * 10  # Spread out the points more
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

# Create a random octree and convert it from a point cloud
def create_random_octree(max_depth=6, num_points=10000):
    point_cloud = generate_random_point_cloud(num_points)
    octree = o3d.geometry.Octree(max_depth=max_depth)
    # Using a smaller size_expand to ensure all points fit within the octree
    octree.convert_from_point_cloud(point_cloud, size_expand=0.5)
    return octree, point_cloud

# Create a list to hold all bounding boxes and other geometries
geometries = []

# Create random octree
octree, point_cloud = create_random_octree(max_depth=6, num_points=10000)

# Add the point cloud to the geometries list for visualization
geometries.append(point_cloud)

# Traverse the octree and add bounding boxes for all nodes (including internal nodes)
def traverse_all_nodes(node, node_info):
    # Add bounding box for both internal and leaf nodes
    add_bounding_box(node, node_info, geometries)
    return True

# Start traversal
octree.traverse(traverse_all_nodes)

print('all done')

# Visualize the point cloud and bounding boxes
o3d.visualization.draw(geometries)
