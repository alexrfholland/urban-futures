import trimesh
import numpy as np
from scipy.spatial import KDTree, cKDTree
import pyvista as pv
import noise

def to_eastings_and_northings(vertices, model_point, real_point):
    print('translating model from model space to eastings and northings')
    translation = real_point - model_point
    return vertices + translation

def get_valid_buildings(file_path, easting, northing, eastings_dim, northings_dim):
    print(f"Loading COLLADA file from {file_path}...")
    scene = trimesh.load(file_path)
    
    model_point = np.array([-192.93200684, 1883.12744141, 0.0])
    real_point = np.array([320901.57699435, 5814373.65543407, 0.0])
    
    valid_geometries = []
    centroids = []
    
    if isinstance(scene, trimesh.Trimesh):
        scene.vertices = to_eastings_and_northings(scene.vertices, model_point, real_point)
        valid_geometries = [scene]
        centroids = [scene.centroid]
    else:
        for geom in scene.geometry.values():
            if hasattr(geom, 'centroid'):
                geom.vertices = to_eastings_and_northings(geom.vertices, model_point, real_point)
                valid_geometries.append(geom)
                centroids.append(geom.centroid)
    
    centroids = np.array(centroids)
    
    print(f"Centroids extracted for {len(valid_geometries)} valid geometries.")
    print("First 10 centroids:")
    print(centroids[:10])
    
    min_bound = centroids.min(axis=0)
    max_bound = centroids.max(axis=0)
    print("Bounding box of centroids (min, max):")
    print(f"Min: {min_bound}")
    print(f"Max: {max_bound}")
    
    dimensions = max_bound - min_bound
    print(f"Bounding box dimensions (X, Y, Z):")
    print(f"X: {dimensions[0]}")
    print(f"Y: {dimensions[1]}")
    print(f"Z: {dimensions[2]}")
    
    centroid_of_centroids = centroids.mean(axis=0)
    print("Centroid of all centroids:")
    print(centroid_of_centroids)

    # Replace KDTree search with rectangular bounding box check
    print(f"Searching for geometries within a rectangle of {eastings_dim}x{northings_dim} units centered at {easting}, {northing}...")
    half_eastings = eastings_dim / 2
    half_northings = northings_dim / 2
    min_eastings, min_northings = easting - half_eastings, northing - half_northings
    max_eastings, max_northings = easting + half_eastings, northing + half_northings
    
    indices = []
    for i, centroid in enumerate(centroids):
        if (min_eastings <= centroid[0] <= max_eastings and
            min_northings <= centroid[1] <= max_northings):
            indices.append(i)
    
    if indices:
        print(f"Found {len(indices)} geometries within the rectangular area.")
    else:
        print(f"No geometries found within the rectangular area.")
    
    return [valid_geometries[i] for i in indices]


def adjust_building_heights(terrain_mesh, valid_buildings):
    print("Starting building height adjustment...")
    
    # Convert terrain_mesh to trimesh if it's not already
    if isinstance(terrain_mesh, pv.PolyData):
        terrain_vertices = terrain_mesh.points
    else:
        terrain_vertices = terrain_mesh.vertices
    
    print(f"Terrain mesh has {len(terrain_vertices)} vertices")

    # Create a KD-tree for efficient nearest neighbor searches
    print("Creating KD-tree for terrain vertices...")
    kdtree = cKDTree(terrain_vertices[:, :2])  # Only use x and y coordinates
    print(f"KD-tree created with {len(terrain_vertices)} points")

    adjusted_buildings = []
    for i, building in enumerate(valid_buildings):
        if i % 100 == 0:  # Print progress every 100 buildings
            print(f"Processing building {i+1}/{len(valid_buildings)}...")
        try:
            # Get the centroid and minimum z-coordinate of the building
            centroid = building.centroid
            min_z = np.min(building.vertices[:, 2])

            # Find the nearest point on the terrain
            _, index = kdtree.query(centroid[:2])
            terrain_z = terrain_vertices[index][2]

            # Calculate the vertical translation
            translation = terrain_z - min_z

            # Create a copy of the building and adjust its height
            new_building = building.copy()
            new_building.vertices[:, 2] += translation
            adjusted_buildings.append(new_building)

            if i % 1000 == 0:  # Print detailed info every 1000 buildings
                print(f"  Building {i+1}:")
                print(f"    Centroid: {centroid}")
                print(f"    Nearest terrain point: {terrain_vertices[index]}")
                print(f"    Vertical translation: {translation}")

        except Exception as e:
            print(f"Error processing building {i+1}: {e}")

    print(f"Number of valid buildings: {len(valid_buildings)}")
    print(f"Number of adjusted buildings: {len(adjusted_buildings)}")

    return adjusted_buildings

def combine_buildings_to_mesh(valid_buildings):
    all_vertices = []
    all_faces = []
    all_building_ids = []  # List to store building IDs for each face
    total_vertices = 0
    building_id = 0  # Initialize the building ID counter

    for geometry in valid_buildings:
        vertices = geometry.vertices
        faces = geometry.faces
        
        all_vertices.append(vertices)
        faces_offset = faces + total_vertices
        all_faces.append(np.column_stack((np.full(len(faces), 3), faces_offset)))
        
        # Assign the current building ID to all faces of this building
        all_building_ids.extend([building_id] * len(faces))  # Assign the same ID to all faces of the current building
        
        total_vertices += len(vertices)
        building_id += 1  # Increment the building ID for the next building
    
    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces).flatten()

    # Create a PyVista PolyData object
    polydataMesh = pv.PolyData(combined_vertices, combined_faces.reshape(-1, 4))

    # Add the building ID as cell data to the combined mesh
    polydataMesh.cell_data['building_ID'] = np.array(all_building_ids)

    print(f"Created combined mesh with {polydataMesh.n_points} points and {polydataMesh.n_cells} cells.")
    return polydataMesh


def process_buildings(file_path, easting, northing, eastings_dim, northings_dim, terrain_mesh, buffer=None):
    print(f"Starting process_buildings with dimensions: {eastings_dim}x{northings_dim}")

    if buffer is not None:
        eastings_dim = eastings_dim + buffer
        northings_dim = northings_dim + buffer
        print(f"Applying buffer of {buffer} meters, eastings dim")
    
    valid_buildings = get_valid_buildings(file_path, easting, northing, eastings_dim, northings_dim)
    print(f"Number of valid buildings before adjustment: {len(valid_buildings)}")
    
    print(f"Terrain mesh type: {type(terrain_mesh)}")
    
    # Convert PyVista grid to trimesh
    if isinstance(terrain_mesh, pv.UnstructuredGrid) or isinstance(terrain_mesh, pv.StructuredGrid):
        print("Converting UnstructuredGrid/StructuredGrid to PolyData...")
        terrain_polydata = terrain_mesh.extract_surface()
        
        print("Converting PolyData to trimesh...")
        terrain_trimesh = trimesh.Trimesh(vertices=terrain_polydata.points, 
                                          faces=terrain_polydata.faces.reshape(-1, 4)[:, 1:])
    elif isinstance(terrain_mesh, pv.PolyData):
        print("Converting PolyData to trimesh...")
        terrain_trimesh = trimesh.Trimesh(vertices=terrain_mesh.points, 
                                          faces=terrain_mesh.faces.reshape(-1, 4)[:, 1:])
    else:
        print("Using provided terrain mesh as is (assuming it's already a trimesh)")
        terrain_trimesh = terrain_mesh

    print(f"Terrain trimesh created with {len(terrain_trimesh.vertices)} vertices and {len(terrain_trimesh.faces)} faces")

    print("Adjusting building heights...")
    adjusted_buildings = adjust_building_heights(terrain_trimesh, valid_buildings)
    print(f"Number of buildings after height adjustment: {len(adjusted_buildings)}")

    print("Combining adjusted buildings into a single mesh...")
    combined_mesh = combine_buildings_to_mesh(adjusted_buildings)
    print(f"Final combined mesh created with {combined_mesh.n_points} points and {combined_mesh.n_cells} cells")

    return combined_mesh

if __name__ == "__main__":
    # Test the script independently
    file_path = "data/revised/experimental/DevelopmentActivityModel-trimmed-metric.glb"
    eastings_dim = 2000  # 2000 meters in eastings direction
    northings_dim = 2000  # 2000 meters in northings direction
    point = np.array([320757.79029528715, 5814438.136253171, 0.0])  # Example point for Melbourne Connect site
    
    # Create a terrain mesh covering the rectangular area
    x_min, y_min = point[:2] - np.array([eastings_dim/2, northings_dim/2])
    x_max, y_max = point[:2] + np.array([eastings_dim/2, northings_dim/2])
    
    resolution = 100
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    x, y = np.meshgrid(x, y)
    
    # Generate Perlin noise for the terrain
    scale = 1000.0  # Adjust this to change the scale of the terrain features
    octaves = 6  # Number of layers of noise
    persistence = 0.5  # How much each octave contributes to the overall shape
    lacunarity = 2.0  # How much detail is added or removed at each octave
    
    z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            z[i][j] = noise.pnoise2(x[i][j]/scale, 
                                    y[i][j]/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=resolution, 
                                    repeaty=resolution, 
                                    base=0)

    # Normalize and scale the terrain
    z = (z - z.min()) / (z.max() - z.min())
    base_height = 0  # Lowest point of the terrain
    height_range = 100  # Difference between highest and lowest points
    z = base_height + z * height_range
    
    # Create a trimesh terrain mesh
    terrain_vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    terrain_faces = []
    for i in range(resolution-1):
        for j in range(resolution-1):
            v1 = i * resolution + j
            v2 = i * resolution + (j + 1)
            v3 = (i + 1) * resolution + (j + 1)
            v4 = (i + 1) * resolution + j
            terrain_faces.append([v1, v2, v3])
            terrain_faces.append([v1, v3, v4])
    terrain_mesh = trimesh.Trimesh(vertices=terrain_vertices, faces=terrain_faces)
    
    # Convert trimesh terrain to PyVista PolyData
    terrain_pv = pv.PolyData(terrain_vertices, np.column_stack((np.full(len(terrain_faces), 3), terrain_faces)))
    
    combined_mesh = process_buildings(file_path, eastings_dim, northings_dim, terrain_mesh, point)
    
    # Visualize the results using PyVista
    plotter = pv.Plotter()
    plotter.add_mesh(combined_mesh, color='lightblue', opacity=0.7, show_edges=True)
    plotter.add_mesh(terrain_pv, color='green', opacity=0.5)
    plotter.add_point_labels([point], ["Search Point"], point_size=20, font_size=16)
    plotter.show()