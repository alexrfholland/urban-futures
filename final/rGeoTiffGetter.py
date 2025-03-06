import numpy as np
import xml.etree.ElementTree as ET
import rasterio
from rasterio.plot import show
from shapely.geometry import Polygon, Point
import os
import csv
from io import StringIO
from pyproj import Transformer
import matplotlib.pyplot as plt

def get_site_coordinates(site_name):
    csv_data = """Site,Latitude,Longitude,Easting,Northing,Name
Arden street oval,-37.798176,144.940516,318678.57675011235,5814579.653000373,park
CBD laneway,-37.810233,144.97079,321373.3041194739,5813300.055663307,city
Royal Parade,-37.7884085,144.9590711,320288.7124645042,5815699.3579083355,parade
Kensington street,-37.79325,144.934198,318110.1824055008,5815113.993545745,street
Royal Parade Trimmed,-1,-1,320266.26,5815638.74,trimmed-parade
Melbourne Connect (Site),-37.799861111111106,144.96408333333332,320757.79029528715,5814438.136253171,uni"""

    sites = list(csv.DictReader(StringIO(csv_data)))
    target_site = next(site for site in sites if site['Name'] == site_name)
    print(f"Retrieved coordinates for site: {site_name}")
    return float(target_site['Easting']), float(target_site['Northing'])

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

def convert_to_latlon(easting, northing):
    transformer = Transformer.from_crs("EPSG:28355", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon

def find_nearest_tile(target, tiles):
    point = Point(target[1], target[0])  # lon, lat
    nearest_tile = None
    min_distance = float('inf')
    
    for name, polygon in tiles.items():
        distance = polygon.exterior.distance(point)
        if distance < min_distance:
            min_distance = distance
            nearest_tile = name
    
    if nearest_tile:
        print(f"Nearest tile found: {nearest_tile} (distance: {min_distance:.6f} degrees)")
    else:
        print("No tiles found")
    
    return nearest_tile

def load_geotiff(tile_name, tiff_folder):
    tiff_path = os.path.join(tiff_folder, tile_name)
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"GeoTIFF file not found: {tiff_path}")
    
    with rasterio.open(tiff_path) as src:
        return src.read(1), src.transform

def get_geotiff_for_target(target, kml_path, tiff_folder):
    tiles = parse_kml(kml_path)
    lat, lon = convert_to_latlon(target[0], target[1])
    tile_name = find_nearest_tile((lat, lon), tiles)
    
    if tile_name is None:
        raise ValueError(f"No tile found for target coordinates: {target}")
    
    print(f"Target coordinates {target} fall within tile: {tile_name}")
    return load_geotiff(tile_name, tiff_folder)

def plot_geotiff(tile_name, tiff_folder):
    tiff_path = os.path.join(tiff_folder, tile_name)
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        
        # Clamp the data range to 0-100
        data = np.clip(data, 0, 100)
        
        fig, ax = plt.subplots(figsize=(12, 12))
        show(data, ax=ax, cmap='terrain', title=f"Digital Surface Model - {tile_name}")
        plt.colorbar(ax.images[0], label='Elevation (m)')
        plt.show()

def find_tiles_within_radius(target, tiles, radius_meters):
    point = Point(target[1], target[0])  # lon, lat
    tiles_within_radius = []
    
    for name, polygon in tiles.items():
        distance = polygon.exterior.distance(point) * 111000  # Approximate conversion to meters
        if distance <= radius_meters:
            tiles_within_radius.append((name, distance))
    
    tiles_within_radius.sort(key=lambda x: x[1])
    
    if tiles_within_radius:
        print(f"Found {len(tiles_within_radius)} tiles within {radius_meters} meters:")
        for name, distance in tiles_within_radius:
            print(f"  {name} (distance: {distance:.2f} meters)")
    else:
        print(f"No tiles found within {radius_meters} meters")
    
    return [name for name, _ in tiles_within_radius]

def plot_multiple_geotiffs(tile_names, tiff_folder):
    fig, ax = plt.subplots(figsize=(15, 15))
    
    for tile_name in tile_names:
        tiff_path = os.path.join(tiff_folder, tile_name)
        with rasterio.open(tiff_path) as src:
            data = src.read(1)
            data = np.clip(data, 0, 100)
            show(data, ax=ax, cmap='terrain', alpha=0.7)
    
    ax.set_title(f"Digital Surface Model - Multiple Tiles")
    plt.colorbar(ax.images[0], label='Elevation (m)')
    plt.show()

if __name__ == "__main__":
    # Example usage
    site_name = 'uni'  # Melbourne Connect site
    easting, northing = get_site_coordinates(site_name)
    radius_meters = 500  # Adjust this value as needed
    gda_target = np.array([easting, northing])
    kml_path = "data/revised/dem/CoM_DSM_2018/DSM_Tile_Index.KML"
    tiff_folder = "data/revised/dem/CoM_DSM_2018"
    
    try:
        tiles = parse_kml(kml_path)
        lat, lon = convert_to_latlon(easting, northing)
        tile_names = find_tiles_within_radius((lat, lon), tiles, radius_meters)
        
        if not tile_names:
            raise ValueError(f"No tiles found within {radius_meters} meters of the target coordinates: {gda_target}")
        
        # Plot the GeoTIFFs
        plot_multiple_geotiffs(tile_names, tiff_folder)
        
    except Exception as e:
        print(f"Error: {str(e)}")