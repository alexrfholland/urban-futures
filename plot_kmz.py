from pykml import parser
import trimesh
import numpy as np
from zipfile import ZipFile
import io

def process_kmz(kmz_path):
    # Open the KMZ file and extract the KML content
    with ZipFile(kmz_path, 'r') as kmz:
        kml_file = next(f for f in kmz.namelist() if f.endswith('.kml'))
        kml_content = kmz.read(kml_file)
    
    # Parse the KML content
    root = parser.fromstring(kml_content)
    
    # Collect all coordinates
    all_coords = []
    
    # Process all placemarks
    def process_features(element):
        for placemark in element.findall('.//{http://www.opengis.net/kml/2.2}Placemark'):
            polygon = placemark.find('.//{http://www.opengis.net/kml/2.2}Polygon')
            if polygon is not None:
                coords_text = polygon.find('.//{http://www.opengis.net/kml/2.2}coordinates').text
                coords = np.array([list(map(float, coord.split(','))) for coord in coords_text.strip().split()])
                all_coords.append(coords)
            else:
                print(f"Found a Placemark without a Polygon: {placemark.find('.//{http://www.opengis.net/kml/2.2}name').text}")

    process_features(root)
    
    if not all_coords:
        print("No polygons found in the KMZ file.")
        return
    
    # Combine all coordinates
    combined_coords = np.vstack(all_coords)
    
    # Calculate and print bounding box
    min_coords = np.min(combined_coords, axis=0)
    max_coords = np.max(combined_coords, axis=0)
    print(f"Bounding Box:")
    print(f"Min Longitude: {min_coords[0]}")
    print(f"Min Latitude: {min_coords[1]}")
    print(f"Min Altitude: {min_coords[2]}")
    print(f"Max Longitude: {max_coords[0]}")
    print(f"Max Latitude: {max_coords[1]}")
    print(f"Max Altitude: {max_coords[2]}")
    
    # Create a trimesh scene
    scene = trimesh.Scene()
    
    # Add each polygon to the scene
    for coords in all_coords:
        # Create a path for the polygon
        path = trimesh.path.Path3D(coords[:, :3])  # Use only x, y, z coordinates
        # Add the path to the scene
        scene.add_geometry(path)
    
    # Show the scene
    scene.show()

if __name__ == "__main__":
    kmz_path = 'data/revised/experimental/DevelopmentActivityModel.kmz'
    process_kmz(kmz_path)