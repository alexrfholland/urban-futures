import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pyproj import CRS, Transformer
import requests
import csv
from io import StringIO
import pyvista as pv
import numpy as np
# geospatial_module.py
from shapely.geometry import Polygon, MultiPolygon

import requests
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pyproj import CRS, Transformer
from io import StringIO
import csv
import pandas as pd
import json

from scipy.spatial import KDTree


def get_site_coordinates(site_name):
    csv_data = """Site,Latitude,Longitude,Easting,Northing,Name
Arden street oval,-37.798176,144.940516,318678.57675011235,5814579.653000373,park
CBD laneway,-37.810233,144.97079,321373.3041194739,5813300.055663307,city
Royal Parade,-37.7884085,144.9590711,320288.7124645042,5815699.3579083355,parade
Kensington street,-37.79325,144.934198,318110.1824055008,5815113.993545745,street
Royal Parade Trimmed,-1,-1,320266.26,5815638.74,trimmed-parade"""

    csv_data = """Site,Latitude,Longitude,Easting,Northing,Name
Arden street oval,-37.798176,144.940516,318678.57675011235,5814579.653000373,park
CBD laneway,-37.810233,144.97079,321373.3041194739,5813300.055663307,city
Royal Parade,-37.7884085,144.9590711,320288.7124645042,5815699.3579083355,parade
Kensington street,-37.79325,144.934198,318110.1824055008,5815113.993545745,street
Royal Parade Trimmed,-1,-1,320266.26,5815638.74,trimmed-parade
Melbourne Connect (Site),-37.799861111111106,144.96408333333332,320757.79029528715,5814438.136253171,uni"""

    sites = list(csv.DictReader(StringIO(csv_data)))
    target_site = next(site for site in sites if site['Name'] == site_name)
    return float(target_site['Easting']), float(target_site['Northing'])

def retrieve_geospatial_data(easting, northing, radius_meters, base_url, dataset_name):
    transformer = Transformer.from_crs("EPSG:28355", "EPSG:4326", always_xy=True)
    center_lon, center_lat = transformer.transform(easting, northing)
    
    all_results = []
    offset = 0
    limit = 100
    
    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "select": "*",
        }
        
        if dataset_name == "trees":
            params["where"] = f"distance(geolocation, geom'POINT({center_lon} {center_lat})', {radius_meters}m)"
        else:
            params["where"] = f"distance(geo_point_2d, geom'POINT({center_lon} {center_lat})', {radius_meters}m)"
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            all_results.extend(results)
            
            if len(results) < limit:
                break
            
            offset += limit
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        print(f"Retrieved {len(all_results)} {dataset_name} so far...")
    
    print(f"Total {dataset_name} retrieved: {len(all_results)}")
    return all_results


def plot_geospatial_data(gdf, easting, northing, radius_meters, title):
    search_area = Point(easting, northing).buffer(radius_meters)
    search_area_gdf = gpd.GeoDataFrame({'geometry': [search_area]}, crs=CRS.from_epsg(28355))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    search_area_gdf.plot(ax=ax, color='none', edgecolor='red', linewidth=2)
    gdf.plot(ax=ax, color='blue', alpha=0.5)
    plt.title(title)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.plot(easting, northing, 'ro', markersize=10, label='Target Site')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Number of features within {radius_meters}m: {len(gdf)}")
    print(f"Total area of features within {radius_meters}m: {gdf.geometry.area.sum():.2f} square meters")

def export_geojson(data, output_file):
    """
    Export the retrieved data as GeoJSON.
    """
    print(f'exporting geojson')
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": feature["geo_shape"]["geometry"],
                "properties": {k: v for k, v in feature.items() if k != "geo_shape"}
            }
            for feature in data
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(geojson, f)
    
    print(f"Original GeoJSON exported to {output_file}")



def handle_building_data(easting, northing, radius_meters):
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/2023-building-footprints/records"
    buildings = retrieve_geospatial_data(easting, northing, radius_meters, base_url, "buildings")
    
    print(f'buildings: {buildings}')
    if buildings:
        # Export the original data as GeoJSON
        path = f'data/revised/outputs/buildings.geojson'
        export_geojson(buildings, "original_buildings.geojson")
        gdf = gpd.GeoDataFrame.from_features([
            {
                "type": "Feature",
                "geometry": building["geo_shape"]["geometry"],
                "properties": {k: v for k, v in building.items() if k != "geo_shape"}
            }
            for building in buildings
        ])
        
        
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)


        
        # Print all columns (layers) in the GeoDataFrame
        print("Columns in the GeoDataFrame:")
        print(gdf.columns.tolist())

        # Print unique footprint_type values
        print("Unique footprint_type values:")
        print(gdf['footprint_type'].unique().tolist())
        
        #plot_geospatial_data(gdf, easting, northing, radius_meters, f"Building Footprints within {radius_meters}m")

        return gdf
    else:
        print("Failed to retrieve or process building data.")
        


### CODE THAT STILL ERRORS CAPS BUT LOOKS OK
from scipy.spatial import ConvexHull

def is_clockwise(coords):
    """
    Determine if a polygon is clockwise using the shoelace formula.
    """
    return sum((x2 - x1) * (y2 + y1) for ((x1, y1), (x2, y2)) in zip(coords, coords[1:] + [coords[0]])) > 0

def correct_point_order(coords):
    """
    Correct the order of points in a polygon to ensure they form a proper sequence.
    """
    # Use ConvexHull to get a proper ordering of points
    hull = ConvexHull(coords)
    ordered_coords = coords[hull.vertices]
    
    # Ensure the polygon has the same orientation as the original
    if is_clockwise(coords) != is_clockwise(ordered_coords):
        ordered_coords = ordered_coords[::-1]
    
    return ordered_coords

def create_3d_building_visualizationNEATERRORS(gdf):
    print("Available columns in the GeoDataFrame:")
    print(gdf.columns.tolist())

    polygons = pv.PolyData()
    
    for idx, row in gdf.iterrows():
        try:
            print(f"\nProcessing building {idx}")
            
            geom = row.geometry
            
            # Handling MultiPolygons
            if isinstance(geom, MultiPolygon):
                print(f"Building {idx} is a MultiPolygon. Selecting largest polygon.")
                geom = max(geom, key=lambda x: x.area)
            
            if isinstance(geom, Polygon):
                # Get exterior coordinates
                exterior_coords = np.array(geom.exterior.coords)
                
                # Ensuring Closed Loops
                if np.allclose(exterior_coords[0], exterior_coords[-1]):
                    print(f"Building {idx} is a closed loop. Removing duplicate end point.")
                    exterior_coords = exterior_coords[:-1]
                
                if len(exterior_coords) < 3:
                    print(f"Skipping building {idx}: Not enough points ({len(exterior_coords)})")
                    continue
                
                # Correct point order
                exterior_coords = correct_point_order(exterior_coords)
                
                print(f"Building {idx} has {len(exterior_coords)} points")
                
                # Selecting the extrusion value
                if 'structure_extrusion' in row and row['structure_extrusion'] is not None:
                    height = row['structure_extrusion']
                elif 'footprint_extrusion' in row and row['footprint_extrusion'] is not None:
                    height = row['footprint_extrusion']
                elif 'structure_max_elevation' in row and 'structure_min_elevation' in row:
                    height = row['structure_max_elevation'] - row['structure_min_elevation']
                else:
                    print(f"Skipping building {idx}: No valid extrusion data")
                    continue
                
                print(f"Building {idx} height: {height}")
                
                # Create points for the building (only top face and sides)
                bottom = np.column_stack((exterior_coords, np.zeros(len(exterior_coords))))
                top = np.column_stack((exterior_coords, np.full(len(exterior_coords), height)))
                
                points = np.vstack((bottom, top))
                
                num_points = len(exterior_coords)
                faces = []
                
                # Top face
                faces.extend([num_points] + list(range(num_points, 2*num_points)))
                print(f"Building {idx}: Top face created")
                
                # Side faces
                for i in range(num_points):
                    faces.extend([4, i, (i+1)%num_points, num_points+(i+1)%num_points, num_points+i])
                print(f"Building {idx}: Side faces created")
                
                building = pv.PolyData(points, faces)
                polygons += building
                print(f"Building {idx} added to the visualization")
            else:
                print(f"Skipping building {idx}: Not a Polygon or MultiPolygon")
        
        except Exception as e:
            print(f"Error processing building {idx}:")
            print(f"Geometry type: {type(row.geometry)}")
            print(f"Geometry: {row.geometry}")
            print(f"Available columns: {row.index.tolist()}")
            print(f"Error: {e}")
            continue

    print(f"\nTotal buildings processed: {len(polygons.points) // 2}")
    return polygons

####

def create_3d_building_visualizationSIMPLE(gdf):
    print("Available columns in the GeoDataFrame:")
    print(gdf.columns.tolist())

    polygons = pv.PolyData()
    
    for idx, row in gdf.iterrows():
        try:
            print(f"\nProcessing building {idx}")
            
            geom = row.geometry
            
            # Handling MultiPolygons
            if isinstance(geom, MultiPolygon):
                print(f"Building {idx} is a MultiPolygon. Selecting largest polygon.")
                geom = max(geom, key=lambda x: x.area)
            
            if isinstance(geom, Polygon):
                # Consistent Point Ordering
                exterior_coords = np.array(geom.exterior.coords)
                
                # Ensuring Closed Loops
                if np.allclose(exterior_coords[0], exterior_coords[-1]):
                    print(f"Building {idx} is a closed loop. Removing duplicate end point.")
                    exterior_coords = exterior_coords[:-1]
                
                if len(exterior_coords) < 3:
                    print(f"Skipping building {idx}: Not enough points ({len(exterior_coords)})")
                    continue
                
                print(f"Building {idx} has {len(exterior_coords)} points")
                
                # Selecting the extrusion value
                if 'structure_extrusion' in row and row['structure_extrusion'] is not None:
                    height = row['structure_extrusion']
                elif 'footprint_extrusion' in row and row['footprint_extrusion'] is not None:
                    height = row['footprint_extrusion']
                elif 'structure_max_elevation' in row and 'structure_min_elevation' in row:
                    height = row['structure_max_elevation'] - row['structure_min_elevation']
                else:
                    print(f"Skipping building {idx}: No valid extrusion data")
                    continue
                
                print(f"Building {idx} height: {height}")
                
                bottom = np.column_stack((exterior_coords, np.zeros(len(exterior_coords))))
                top = np.column_stack((exterior_coords, np.full(len(exterior_coords), height)))
                
                points = np.vstack((bottom, top))
                
                num_points = len(exterior_coords)
                faces = []
                
                # Correct Normal Direction for bottom face
                #faces.extend([num_points] + list(range(num_points))[::-1])
                print(f"Building {idx}: Bottom face created with reversed order for correct normal direction")
                
                faces.extend([num_points] + list(range(num_points, 2*num_points)))
                print(f"Building {idx}: Top face created")
                
                for i in range(num_points):
                    faces.extend([4, i, (i+1)%num_points, num_points+(i+1)%num_points, num_points+i])
                print(f"Building {idx}: Side faces created")
                
                building = pv.PolyData(points, faces)
                
                polygons += building
                print(f"Building {idx} added to the visualization")
            else:
                print(f"Skipping building {idx}: Not a Polygon or MultiPolygon")
        
        except Exception as e:
            print(f"Error processing building {idx}:")
            print(f"Geometry type: {type(row.geometry)}")
            print(f"Geometry: {row.geometry}")
            print(f"Available columns: {row.index.tolist()}")
            print(f"Error: {e}")
            continue

    print(f"\nTotal buildings processed: {len(polygons.points) // 2}")
    return polygons


import numpy as np
import pyvista as pv
from shapely.geometry import Polygon, MultiPolygon
import pandas as pd

def analyze_top_cap(points, face, bottom_coords):
    """Analyze the top cap and compare it with the building's footprint."""
    top_coords = points[face[1:]]
    
    # Calculate area of top face and bottom face
    top_area = polygon_area(top_coords)
    bottom_area = polygon_area(bottom_coords)
    
    # Calculate centroid of top face and bottom face
    top_centroid = np.mean(top_coords, axis=0)
    bottom_centroid = np.mean(bottom_coords, axis=0)
    
    # Calculate the maximum distance between any two points on the top face
    max_distance = max(np.linalg.norm(p1 - p2) for p1 in top_coords for p2 in top_coords)
    
    return {
        "top_area": top_area,
        "bottom_area": bottom_area,
        "area_ratio": top_area / bottom_area if bottom_area > 0 else float('inf'),
        "centroid_offset": np.linalg.norm(top_centroid[:2] - bottom_centroid[:2]),
        "max_distance": max_distance,
        "num_points": len(top_coords)
    }

def polygon_area(points):
    """Calculate the area of a polygon."""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area

def create_3d_building_visualization(gdf):
    print("Available columns in the GeoDataFrame:")
    print(gdf.columns.tolist())

    polygons = pv.PolyData()
    building_data = []
    building_meshes = {}  # New dictionary to store individual building meshes
    
    for idx, row in gdf.iterrows():
        try:
            print(f"\nProcessing building {idx}")
            
            geom = row.geometry
            
            # Handling MultiPolygons
            if isinstance(geom, MultiPolygon):
                print(f"Building {idx} is a MultiPolygon. Selecting largest polygon.")
                geom = max(geom, key=lambda x: x.area)
            
            if isinstance(geom, Polygon):
                # Consistent Point Ordering
                exterior_coords = np.array(geom.exterior.coords)
                
                # Ensuring Closed Loops
                if np.allclose(exterior_coords[0], exterior_coords[-1]):
                    print(f"Building {idx} is a closed loop. Removing duplicate end point.")
                    exterior_coords = exterior_coords[:-1]
                
                if len(exterior_coords) < 3:
                    print(f"Skipping building {idx}: Not enough points ({len(exterior_coords)})")
                    continue
                
                print(f"Building {idx} has {len(exterior_coords)} points")
                
                # Selecting the extrusion value
                if 'structure_extrusion' in row and row['structure_extrusion'] is not None:
                    height = row['structure_extrusion']
                elif 'footprint_extrusion' in row and row['footprint_extrusion'] is not None:
                    height = row['footprint_extrusion']
                elif 'structure_max_elevation' in row and 'structure_min_elevation' in row:
                    height = row['structure_max_elevation'] - row['structure_min_elevation']
                else:
                    print(f"Skipping building {idx}: No valid extrusion data")
                    continue
                
                print(f"Building {idx} height: {height}")
                
                bottom = np.column_stack((exterior_coords, np.zeros(len(exterior_coords))))
                top = np.column_stack((exterior_coords, np.full(len(exterior_coords), height)))
                points = np.vstack((bottom, top))
                
                num_points = len(exterior_coords)
                faces = []
                
                # Top face
                top_face = [num_points] + list(range(num_points, 2*num_points))
                faces.extend(top_face)
                
                # Analyze top cap
                top_cap_data = analyze_top_cap(points, top_face, bottom)
                building_data.append((idx, height, top_cap_data))
                
                # Side faces
                for i in range(num_points):
                    faces.extend([4, i, (i+1)%num_points, num_points+(i+1)%num_points, num_points+i])
                
                building = pv.PolyData(points, faces)
                building_meshes[idx] = building  # Store individual building mesh
                polygons += building
                print(f"Building {idx} added to the visualization")
            else:
                print(f"Skipping building {idx}: Not a Polygon or MultiPolygon")
        
        except Exception as e:
            print(f"Error processing building {idx}:")
            print(f"Geometry type: {type(row.geometry)}")
            print(f"Geometry: {row.geometry}")
            print(f"Available columns: {row.index.tolist()}")
            print(f"Error: {e}")
            continue

    print(f"\nTotal buildings processed: {len(polygons.points) // 2}")
    
    # Create DataFrame
    df = pd.DataFrame(building_data, columns=['building_id', 'height', 'top_cap_data'])
    df = pd.concat([df, df['top_cap_data'].apply(pd.Series)], axis=1)
    df = df.drop('top_cap_data', axis=1)
    
    return polygons, df, building_meshes




def visualize_buildings(gdf):
    # Create a plotter
    plotter = pv.Plotter()
    print('Running 3D building visualization')

    # Create visualization
    polygons, df, building_meshes = create_3d_building_visualization(gdf)

    # Add mesh to the plot
    plotter.add_mesh(polygons, color='lightblue', show_edges=True)

    # Collect all vertices from all geometries
    all_vertices = []
    for mesh in building_meshes.values():
        all_vertices.extend(mesh.points)

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
        point_size=20,
        picker='point',
        font_size=20  # Show the picked point for debugging
    )
    
    # Add instructions text
    plotter.add_text("Click on the mesh to add a label at the nearest vertex", position='upper_left')

    # Display the plot interactively
    plotter.show()

    return plotter

def handle_road_segments(easting, northing, radius_meters):
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/road-segments-with-surface-type/records"
    road_segments = retrieve_geospatial_data(easting, northing, radius_meters, base_url, "road segments")
    
    if road_segments:
        gdf = gpd.GeoDataFrame.from_features([
            {
                "type": "Feature",
                "geometry": segment["geo_shape"]["geometry"],
                "properties": {k: v for k, v in segment.items() if k != "geo_shape"}
            }
            for segment in road_segments
        ])
        
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)
        
        plot_geospatial_data(gdf, easting, northing, radius_meters, f"Road Segments within {radius_meters}m")
        print(f"Total length of road segments: {gdf.geometry.length.sum():.2f} meters")
    else:
        print("Failed to retrieve or process road segment data.")

def handle_road_corridors(easting, northing, radius_meters):
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/road-corridors/records"
    road_corridors = retrieve_geospatial_data(easting, northing, radius_meters, base_url, "road corridors")
    
    if road_corridors:
        gdf = gpd.GeoDataFrame.from_features([
            {
                "type": "Feature",
                "geometry": corridor["geo_shape"]["geometry"],
                "properties": {k: v for k, v in corridor.items() if k != "geo_shape"}
            }
            for corridor in road_corridors
        ])
        
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)
        
        plot_geospatial_data(gdf, easting, northing, radius_meters, f"Road Corridors within {radius_meters}m")
        print(f"Total area of road corridors: {gdf.geometry.area.sum():.2f} square meters")
    else:
        print("Failed to retrieve or process road corridor data.")

def handle_trees(easting, northing, radius_meters):
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/trees-with-species-and-dimensions-urban-forest/records"
    trees = retrieve_geospatial_data(easting, northing, radius_meters, base_url, "trees")
    
    if trees:
        gdf = gpd.GeoDataFrame([
            {
                **tree,
                'geometry': Point(tree['longitude'], tree['latitude'])
            }
            for tree in trees
        ])
        
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)
        
        plot_geospatial_data(gdf, easting, northing, radius_meters, f"Trees within {radius_meters}m")
        print(f"Number of trees: {len(gdf)}")
        if 'diameter_breast_height' in gdf.columns:
            # Filter out null values and convert to numeric
            dbh_values = pd.to_numeric(gdf['diameter_breast_height'], errors='coerce').dropna()
            if not dbh_values.empty:
                print(f"Average diameter at breast height: {dbh_values.mean():.2f} cm")
            else:
                print("No valid diameter at breast height data available.")
        
        # Additional statistics
        print("\nTop 5 most common tree species:")
        print(gdf['common_name'].value_counts().head())
        
        print("\nAge distribution:")
        print(gdf['age_description'].value_counts())
        
        print("\nLocation distribution:")
        print(gdf['located_in'].value_counts())
        
    else:
        print("Failed to retrieve or process tree data.")


def mainExperiment(gdf):
    # Create 3D building visualization
    polygons, df, building_meshes = create_3d_building_visualization(gdf)
    
    # Identify potential issues
    df['potential_issue'] = (
        (df['area_ratio'] < 0.9) | (df['area_ratio'] > 1.1) |
        (df['centroid_offset'] > 1) |
        (df['max_distance'] > 100)  # Adjust this threshold based on your data
    )
    
    problematic_buildings = df[df['potential_issue']]['building_id'].tolist()
    
    # Print summary
    print(f"Total buildings: {len(df)}")
    print(f"Potentially problematic buildings: {len(problematic_buildings)}")
    
    # Visualize
    if polygons.n_points > 0:
        plotter = pv.Plotter()
        
        # Add all buildings, coloring problematic ones red
        for idx, mesh in building_meshes.items():
            if idx in problematic_buildings:
                color = 'red'
                opacity = 0.5
            else:
                color = 'lightblue'
                opacity = 1.0
            plotter.add_mesh(mesh, color=color, opacity=opacity, show_edges=True)
        
        # Collect all vertices from all geometries
        all_vertices = []
        for mesh in building_meshes.values():
            all_vertices.extend(mesh.points)

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
            distance, index = vertex_tree.query(picked_position)  # Use all three coordinates
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
            point_size=20,
            picker='point',
            font_size=20  # Show the picked point for debugging
        )
        
        # Add instructions text
        plotter.add_text("Click on the mesh to add a label at the nearest vertex", position='upper_left')
        
        plotter.show()
    else:
        print("No buildings to visualize")


def main():
    #site_name = 'trimmed-parade'
    site_name = 'uni'
    easting, northing = get_site_coordinates(site_name)
    print(f"Target site coordinates: Easting = {easting}, Northing = {northing}")
    
    radius_meters = 500
    
    print('getting building data')
    #gdf = handle_building_data(easting, northing, radius_meters)
    #mainExperiment(gdf)
    
    #print('creating 3d building visualization')
    
    
    
    #visualize_buildings(gdf)


    #handle_road_segments(easting, northing, radius_meters)
    #handle_road_corridors(easting, northing, radius_meters)
    handle_trees(easting, northing, radius_meters)


if __name__ == "__main__":
    main()