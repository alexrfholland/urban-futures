import numpy as np
import xml.etree.ElementTree as ET
import rasterio
from shapely.geometry import Polygon, Point
import os

def parse_kml(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    
    tiles = {}
    for placemark in root.findall(".//{http://www.opengis.net/kml/2.2}Placemark"):
        name = placemark.find("{http://www.opengis.net/kml/2.2}name").text
        coords_text = placemark.find(".//{http://www.opengis.net/kml/2.2}coordinates").text
        coords = [tuple(map(float, coord.split(','))) for coord in coords_text.strip().split()]
        polygon = Polygon(coords)
        tiles[name] = polygon
    
    return tiles

def find_tile(target, tiles):
    point = Point(target[0], target[1])
    for name, polygon in tiles.items():
        if polygon.contains(point):
            return name
    return None

def load_geotiff(tile_name, tiff_folder):
    tiff_path = os.path.join(tiff_folder, tile_name)
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"GeoTIFF file not found: {tiff_path}")
    
    with rasterio.open(tiff_path) as src:
        return src.read(1), src.transform

def get_geotiff_for_target(target, kml_path, tiff_folder):
    tiles = parse_kml(kml_path)
    tile_name = find_tile(target, tiles)
    
    if tile_name is None:
        raise ValueError(f"No tile found for target coordinates: {target}")
    
    print(f"Target coordinates {target} fall within tile: {tile_name}")
    return load_geotiff(tile_name, tiff_folder)



if __name__ == "__main__":
    # Example usage
    gda_target = np.array([320757.79029528715, 5814438.136253171])
    kml_path = "data/revised/dem/DSM_Tile_Index.KML"
    tiff_folder = "data/revised/dem/tiffs/"
    
    try:
        geotiff_data, transform = get_geotiff_for_target(gda_target, kml_path, tiff_folder)
        print("GeoTIFF loaded successfully.")
        print(f"GeoTIFF shape: {geotiff_data.shape}")
        print(f"GeoTIFF transform: {transform}")
    except Exception as e:
        print(f"Error: {str(e)}")