import trimesh
import numpy as np
from scipy.spatial import KDTree
import pyvista as pv

def ToEastingsAndNorthings(vertices, model_point, real_point):
    """
    Transform vertices from model space to eastings and northings.
    
    Args:
    vertices (np.array): Array of vertices to transform.
    model_point (np.array): Reference point in model space.
    real_point (np.array): Corresponding point in eastings and northings.
    
    Returns:
    np.array: Transformed vertices.
    """
    # Calculate the translation vector
    translation = real_point - model_point
    
    # Apply the translation to all vertices
    transformed_vertices = vertices + translation
    
    return transformed_vertices

def inspect_scene_and_search(file_path, search_radius, point=None):
    # Load the COLLADA file
    print(f"Loading COLLADA file from {file_path}...")
    scene = trimesh.load(file_path)
    
    # Define the reference points for transformation
    model_point = np.array([-192.93200684, 1883.12744141, 0.0])
    real_point = np.array([320901.57699435, 5814373.65543407, 0.0])
    
    if isinstance(scene, trimesh.Trimesh):
        print("Processing single mesh...")
        # Transform vertices to eastings and northings
        scene.vertices = ToEastingsAndNorthings(scene.vertices, model_point, real_point)
        vertices = scene.vertices
        faces = scene.faces
        face_array = np.hstack([np.full((faces.shape[0], 1), 3), faces])
        pyvista_mesh = pv.PolyData(vertices, face_array.flatten())
        plotter = pv.Plotter()
        plotter.add_mesh(pyvista_mesh, color='lightblue')
        plotter.show()
    
    else:
        # If it's a scene, print out the key properties
        print("Processing scene...")
        print(f"Number of geometries: {len(scene.geometry)}")
        print(f"Number of nodes: {len(scene.graph.nodes)}")

        # Extract the centroids for all geometries and transform them
        valid_geometries = []
        centroids = []
        for geom in scene.geometry.values():
            if hasattr(geom, 'centroid'):
                # Transform vertices to eastings and northings
                geom.vertices = ToEastingsAndNorthings(geom.vertices, model_point, real_point)
                valid_geometries.append(geom)
                centroids.append(geom.centroid)
        
        centroids = np.array(centroids)

        print(f"Centroids extracted for {len(valid_geometries)} valid geometries.")
        
        # Print the first 10 centroids
        print("First 10 centroids:")
        print(centroids[:10])
        
        # Compute and print the bounding box of centroids
        min_bound = centroids.min(axis=0)
        max_bound = centroids.max(axis=0)
        print("Bounding box of centroids (min, max):")
        print(f"Min: {min_bound}")
        print(f"Max: {max_bound}")

        # Calculate and print the dimensions of the bounding box
        dimensions = max_bound - min_bound
        print(f"Bounding box dimensions (X, Y, Z):")
        print(f"X: {dimensions[0]}")
        print(f"Y: {dimensions[1]}")
        print(f"Z: {dimensions[2]}")

        # Compute and print the centroid of all centroids
        centroid_of_centroids = centroids.mean(axis=0)
        print("Centroid of all centroids:")
        print(centroid_of_centroids)
        
        if point is None:
            # Use the centroid of centroids as the search point
            search_point = centroid_of_centroids[:2]  # Only x, y for 2D search
        else:
            search_point = point[:2]  # Use only x, y coordinates of the provided point

        # Build a KD-Tree for fast spatial search
        kdtree = KDTree(centroids[:, :2])  # Use x, y for 2D search
        
        print(f"Searching for geometries within {search_radius} units of point {search_point}...")
        indices = kdtree.query_ball_point(search_point, search_radius)
        
        if indices:
            print(f"Found {len(indices)} geometries within {search_radius} units of point {search_point}.")
        else:
            print(f"No geometries found within {search_radius} units of point {search_point}.")
        
        # Optionally visualize the geometries
        plotter = pv.Plotter()
        
        # Collect all vertices from all geometries
        all_vertices = []
        all_meshes = []
        
        # Plot only the geometries within the search radius
        for i in indices:
            geometry = valid_geometries[i]
            vertices = geometry.vertices
            faces = geometry.faces
            face_array = np.hstack([np.full((faces.shape[0], 1), 3), faces])
            pyvista_mesh = pv.PolyData(vertices, face_array.flatten())
            plotter.add_mesh(pyvista_mesh, color='lightblue')
            all_vertices.extend(vertices)
            all_meshes.append(pyvista_mesh)

        print(f"Plotted {len(indices)} geometries.")

        # Create a KDTree with all vertices
        vertex_tree = KDTree(np.array(all_vertices))

        # Define a callback function for point picking
        def callback(point, picker):
            if picker.GetActor() is None:
                print("No mesh picked")
                return

            print(f"Raw picked point: {point}")
            
            # Get the picked position on the mesh surface
            picked_position = picker.GetPickPosition()
            print(f"Picked position on mesh: {picked_position}")
            
            # Find the nearest vertex
            distance, index = vertex_tree.query(picked_position)
            nearest_vertex = all_vertices[index]
            
            print(f"Nearest vertex: {nearest_vertex}")

            # Add a text label at the picked point
            plotter.add_point_labels([picked_position], [f"Picked: {picked_position}"], point_size=20, font_size=10)
            
            plotter.render()

        # Enable point picking with increased tolerance
        plotter.enable_point_picking(
            callback=callback,
            show_message=True,
            tolerance=0.01,
            use_picker=True,
            pickable_window=False,  # Only pick on the mesh surface
            show_point=True,
            point_size = 20,
            picker = 'point',
            font_size = 20  # Show the picked point for debugging
        )
        
        # Add instructions text
        plotter.add_text("Click on the mesh to add a red sphere at the nearest vertex", position='upper_left')
        
        plotter.show()

# Example usage
file_path = "data/revised/experimental/DevelopmentActivityModel-trimmed-metric.glb"
point = [320901.57699435, 5814373.65543407, 0.0]  # Example point in eastings and northings
inspect_scene_and_search(file_path, 1000, point)
