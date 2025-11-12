import geopandas as gpd
import numpy as np
import pyvista as pv
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString, shape
import requests
from io import StringIO
import csv
import pandas as pd
from pyproj import CRS, Transformer
import shapely.ops
from shapely.ops import linemerge
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from f_SiteCoordinates import get_site_coordinates
import math
from shapely.geometry import box

import dask
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely import points

import numpy as np
import pandas as pd
import geopandas as gpd
import pyvista as pv
from shapely.geometry import Point
from dask import delayed, compute
import dask.array as da
import warnings



def retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url, isPoint=False):
    # Calculate the radius that encompasses the entire rectangular area
    radius_meters = math.sqrt((eastings_dim/2)**2 + (northings_dim/2)**2)

    print(f"Retrieving data for point ({easting}, {northing}) with dimensions {eastings_dim}x{northings_dim}m...")
    print(f"Using calculated radius of {radius_meters:.2f}m for the query")

    transformer = Transformer.from_crs("EPSG:28355", "EPSG:4326", always_xy=True)
    center_lon, center_lat = transformer.transform(easting, northing)
    
    print(f"Center point (EPSG:4326): Lon: {center_lon}, Lat: {center_lat}")

    all_results = []
    offset = 0
    limit = 100

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "select": "*",
        }
        if isPoint:
            params["where"] = f"distance(geolocation, geom'POINT({center_lon} {center_lat})', {radius_meters}m)"
        else:
            params["where"] = f"distance(geo_point_2d, geom'POINT({center_lon} {center_lat})', {radius_meters}m)"
        
        print(f"Making API request with offset: {offset}, limit: {limit}")
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            all_results.extend(results)
            
            print(f"Retrieved {len(results)} new results. Total: {len(all_results)}")
            
            if len(results) < limit:
                print(f"Received {len(results)} results, which is less than the limit of {limit}. Finished retrieving data.")
                break
            
            offset += limit
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    print(f"Total retrieved: {len(all_results)}")

    # Print the keys of the first result to inspect the structure
    if all_results:
        print("Keys in the first result:", all_results[0].keys())

    return all_results


def plot_trimmed_and_untrimmed_gdfs(gdf, trimmed_gdf, title="Comparison of Untrimmed and Trimmed GeoDataFrames"):
    """
    Plot the untrimmed and trimmed GeoDataFrames for comparison.
    
    Parameters:
    - gdf: The original untrimmed GeoDataFrame.
    - trimmed_gdf: The GeoDataFrame after trimming to the bounding box.
    - title: The title of the plot.
    """
    # Create a plot with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the original untrimmed GeoDataFrame
    gdf.plot(ax=ax, color='lightblue', edgecolor='blue', label='Original (Untrimmed)', alpha=0.5)
    
    # Plot the trimmed GeoDataFrame
    trimmed_gdf.plot(ax=ax, color='salmon', edgecolor='red', label='Trimmed', alpha=0.7)
    
    # Add plot title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    
    # Add legend
    ax.legend()
    
    # Show the plot
    plt.show()

def trim_gdf_to_bounding_box(gdf, easting, northing, eastings_dim, northings_dim, target_crs=28355):
    # Ensure the GeoDataFrame is in the target CRS (EPSG:28355)
    if gdf.crs != f"EPSG:{target_crs}":
        gdf = gdf.to_crs(epsg=target_crs)
    
    # Calculate the bounding box from the center point and dimensions
    min_easting = easting - eastings_dim / 2
    max_easting = easting + eastings_dim / 2
    min_northing = northing - northings_dim / 2
    max_northing = northing + northings_dim / 2

    # Create the bounding box geometry in the target CRS
    bounding_box = box(min_easting, min_northing, max_easting, max_northing)

    # Print statements with details before and after the trimming
    original_bounds = gdf.total_bounds
    print(f"Original bounding box dimensions: Easting_dim = {original_bounds[2] - original_bounds[0]:.1f}m, Northing_dim = {original_bounds[3] - original_bounds[1]:.1f}m")
    print(f"Desired bounding box dimensions: Easting_dim = {eastings_dim:.1f}m, Northing_dim = {northings_dim:.1f}m")
    print(f" - Calculated bounding box: MinEasting = {min_easting:.1f}, MaxEasting = {max_easting:.1f}, MinNorthing = {min_northing:.1f}, MaxNorthing = {max_northing:.1f}")
    
    # Clip the GeoDataFrame to the bounding box
    trimmed_gdf = gdf.clip(bounding_box)

    # Get the bounds of the trimmed GeoDataFrame
    trimmed_bounds = trimmed_gdf.total_bounds
    print(f"Trimmed bounding box dimensions: Easting_dim = {trimmed_bounds[2] - trimmed_bounds[0]:.1f}m, Northing_dim = {trimmed_bounds[3] - trimmed_bounds[1]:.1f}m")

    #plot_trimmed_and_untrimmed_gdfs(gdf, trimmed_gdf)

    return trimmed_gdf

def results_to_gdf(all_results, dataset_name, easting, northing, eastings_dim, northings_dim):
    # Extract geometries and attributes, handling points and polygons differently
    geometries = []
    attributes = []
    
    for result in all_results:
        if "geo_shape" in result:
            geometries.append(shape(result["geo_shape"]["geometry"]))
        elif "latitude" in result and "longitude" in result:
            # If it's point data, create Point geometry
            geometries.append(Point(result["longitude"], result["latitude"]))
        else:
            continue
        
        # Prepend dataset name to attribute keys and add the attributes
        attributes.append({f"{dataset_name}_{k}": v for k, v in result.items() if k not in ["geo_shape", "latitude", "longitude"]})

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries)
    
    # Set CRS and reproject
    gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)

    print(f"Successfully created GeoDataFrame with {len(gdf)} entries")

    # Trim the GeoDataFrame to the bounding box
    gdf = trim_gdf_to_bounding_box(gdf, easting, northing, eastings_dim, northings_dim)

    # Print all column names (attribute headings)
    print("Attribute headings:")
    print(gdf.columns.tolist())

    return gdf

###DASK
import numpy as np
import pandas as pd
import geopandas as gpd
import pyvista as pv
from shapely.geometry import Point
from dask import delayed, compute
import warnings

# Suppress potential warnings for cleaner output
warnings.filterwarnings("ignore")

def identify_unhashable_columns(gdf):
    """
    Identify columns in the GeoDataFrame that contain unhashable types (e.g., dicts).
    
    Parameters:
    - gdf: GeoDataFrame to inspect.
    
    Returns:
    - List of column names with unhashable types.
    """
    unhashable_cols = []
    for col in gdf.columns:
        # Check the type of the first non-null entry
        first_valid = gdf[col].dropna().iloc[0] if not gdf[col].dropna().empty else None
        if isinstance(first_valid, dict):
            unhashable_cols.append(col)
    return unhashable_cols

def encode_categorical_columns(gdf):
    """
    Encode categorical and object columns in the GeoDataFrame as integer codes.
    
    Parameters:
    - gdf: GeoDataFrame containing polygon attributes.
    
    Returns:
    - Tuple of (encoded GeoDataFrame, category mappings dictionary).
    """
    category_mappings = {}
    gdf_encoded = gdf.copy()
    
    for col in gdf_encoded.select_dtypes(include=['object', 'category']).columns:
        gdf_encoded[col] = gdf_encoded[col].astype('category')
        category_mappings[col] = dict(enumerate(gdf_encoded[col].cat.categories))
    
    return gdf_encoded, category_mappings

def initialize_dask_arrays(num_points, gdf_encoded):
    """
    Initialize Dask arrays for each column in the encoded GeoDataFrame, excluding the geometry column.
    
    Parameters:
    - num_points: Number of points to initialize the arrays for.
    - gdf_encoded: Encoded GeoDataFrame containing the attributes.
    
    Returns:
    - A dictionary of Dask arrays initialized with default values.
    """
    dask_arrays = {}
    
    for col in gdf_encoded.columns:
        if col == 'geometry':
            print(f"Skipping geometry column {col}.")
            continue
        
        col_dtype = gdf_encoded[col].dtype
        
        # Handle categorical and object dtypes as integer codes
        if pd.api.types.is_categorical_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype):
            print(f"Initializing integer array for encoded column {col}.")
            dask_arrays[col] = delayed(lambda: np.full(num_points, -1, dtype=np.int32))()
        
        # Handle numeric columns (floats or integers)
        elif col_dtype.kind in 'f':  # Floating point types
            print(f"Initializing float array for column {col}.")
            dask_arrays[col] = delayed(lambda: np.full(num_points, np.nan, dtype=col_dtype))()
        
        elif col_dtype.kind in 'i':  # Integer types
            print(f"Initializing integer array for column {col} with default value -1.")
            dask_arrays[col] = delayed(lambda: np.full(num_points, -1, dtype=col_dtype))()
        
        else:
            raise TypeError(f"Unsupported column type for {col}: {col_dtype}")
    
    return dask_arrays

def convert_points_to_geodataframe(points_array, gdf):
    """
    Efficiently convert a Nx2 array of points (x, y) to a GeoDataFrame using vectorized operations.
    
    Parameters:
    - points_array: A NumPy array of shape (N, 2), where N is the number of points and each point has (x, y) coordinates.
    - gdf: Existing GeoDataFrame from which to extract the CRS.
    
    Returns:
    - A GeoDataFrame with the points as geometries and the same CRS as the input GeoDataFrame.
    """
    crs = gdf.crs  # Extract CRS from the input GeoDataFrame
    print(f"Using CRS: {crs} for the new GeoDataFrame.")
    
    # Use GeoPandas' optimized points_from_xy function
    geometries = gpd.points_from_xy(points_array[:, 0], points_array[:, 1])
    
    # Create a GeoDataFrame with the same CRS
    df = pd.DataFrame(points_array, columns=['x', 'y'])
    points_gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=crs)
    
    return points_gdf

def process_chunk_with_max_len(points_chunk, gdf_encoded, binaryMask=None):
    """
    Process a chunk of points by performing a spatial join and extracting attribute values or binary mask.
    
    Parameters:
    - points_chunk: GeoDataFrame of the points to process.
    - gdf_encoded: Encoded GeoDataFrame containing the polygon geometries and attributes.
    - binaryMask: Name of the binary mask attribute (if applicable).
    
    Returns:
    - A dictionary containing processed values for each column or binary mask.
    """
    print(f"Processing chunk with {len(points_chunk)} points.")
    
    # Perform a spatial join (vectorized operation) to find which polygons the points fall within
    joined = gpd.sjoin(points_chunk, gdf_encoded, how="left", predicate="within")
    
    if binaryMask is not None:
        # Assign TRUE/FALSE based on whether the point is within any geometry
        mask = joined['index_right'].notna().astype(np.int32)  # 1 for True, 0 for False
        result = {'values': {binaryMask: mask.values}}
    else:
        # Prepare results with attributes, skipping unhashable columns
        result = {'values': {}}
        for col in gdf_encoded.columns:
            if col == 'geometry':
                continue
            result['values'][col] = joined[col].values
    
    return result

def assign_gdf_attributes_to_polydata(polydata, gdf, attribute_prefix, chunk_size=5000000, binaryMask=None):
    """
    Assign attributes from a GeoDataFrame (gdf) to the points in a PyVista PolyData object.
    
    Parameters:
    - polydata: PyVista PolyData object containing points and point data.
    - gdf: GeoDataFrame containing polygon geometries and associated attributes.
    - attribute_prefix: Prefix to prepend to each attribute name when adding back to PolyData.
    - chunk_size: Size of the chunks to process in parallel (default: 50000).
    - binaryMask: Name of the binary mask attribute. If not None, only a boolean mask is assigned.
    
    Returns:
    - Updated PyVista PolyData object with attributes added back to point_data.
    """
    # Step 1: Extract the (x, y) coordinates from the PyVista PolyData object as a NumPy array
    points = polydata.points[:, :2]  # Extract only x and y coordinates
    num_points = len(points)
    
    # Step 2: Identify unhashable columns
    unhashable_columns = identify_unhashable_columns(gdf)
    if unhashable_columns and binaryMask is None:
        print(f"Columns with unhashable types: {unhashable_columns}")
        # Skip unhashable columns by dropping them
        gdf = gdf.drop(columns=unhashable_columns)
    elif unhashable_columns and binaryMask is not None:
        print(f"Columns with unhashable types: {unhashable_columns} will be ignored due to binaryMask.")
        # When binaryMask is used, all attributes are ignored, so no need to drop
    else:
        print("No unhashable columns detected.")
    
    # Step 3: Encode categorical and object columns (only if binaryMask is None)
    if binaryMask is None:
        gdf_encoded, category_mappings = encode_categorical_columns(gdf)
    else:
        gdf_encoded = gdf.copy()  # No encoding needed
        category_mappings = {}
    
    # Step 4: Convert points to a GeoDataFrame with the same CRS as the gdf
    points_gdf = convert_points_to_geodataframe(points, gdf)
    
    # Step 5: Initialize Dask arrays (only if binaryMask is None)
    if binaryMask is None:
        dask_arrays = initialize_dask_arrays(num_points, gdf_encoded)
    
    # Step 6: Split points into GeoDataFrame chunks
    if num_points == 0:
        print("No points to process.")
        return polydata
    elif num_points <= chunk_size:
        point_chunks = [points_gdf]
        print(f"Processing 1 chunk of size {num_points} points.")
    else:
        point_chunks = [points_gdf.iloc[i:i + chunk_size] for i in range(0, num_points, chunk_size)]
        print(f"Processing {len(point_chunks)} chunks of size up to {chunk_size} points each.")
    
    # Step 7: Process the chunks in parallel using Dask delayed
    tasks = [
        delayed(process_chunk_with_max_len)(
            chunk, 
            gdf_encoded, 
            binaryMask=binaryMask
        ) 
        for chunk in point_chunks
    ]
    
    # Step 8: Compute the results in parallel
    print("Performing spatial joins in parallel...")
    results = compute(*tasks)
    print("Spatial joins completed.")
    
    # Step 9: Accumulate results
    if binaryMask is not None:
        # Only one attribute: the binary mask
        accumulated_mask = []
        for res in results:
            accumulated_mask.append(res['values'][binaryMask])
        
        # Concatenate all masks
        final_mask = np.concatenate(accumulated_mask, axis=0).astype(bool)
        
        # Step 10: Assign the binary mask to the PolyData's point_data
        attribute_name = f"{attribute_prefix}{binaryMask}"
        polydata.point_data[attribute_name] = final_mask
        true_count = np.sum(final_mask)
        print(f"Assigned binary mask '{attribute_name}' to polydata. {true_count} out of {len(final_mask)} points are True.")
        
    else:
        # Accumulate attribute results
        accumulated_results = {col: [] for col in gdf_encoded.columns if col != 'geometry'}
        for res in results:
            for col in accumulated_results.keys():
                accumulated_results[col].append(res['values'][col])
        
        # Step 11: Concatenate results for each column
        final_attributes = {}
        for col, lists in accumulated_results.items():
            # Concatenate all chunk results for the column
            concatenated = np.concatenate(lists, axis=0)
            
            if col in category_mappings:
                print(f"Decoding categorical column {col}.")
                # Decode integer codes back to original string categories
                decoded = np.array([
                    category_mappings[col].get(code, '') if code != -1 else '' for code in concatenated
                ])
                final_attributes[col] = decoded
            else:
                final_attributes[col] = concatenated
        
        # Step 12: Assign attributes to the PolyData's point_data
        for col, array in final_attributes.items():
            attribute_name = f"{attribute_prefix}{col}"
            polydata.point_data[attribute_name] = array
    
    print("Attribute assignment completed successfully.")
    return polydata


###END DASK



def handle_road_segments(easting, northing, eastings_dim, northings_dim):
    nickname = 'roadInfo'
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/road-segments-with-surface-type/records"
    
    # Retrieve the GeoDataFrame and all results
    all_results = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url)

    # Create the GeoDataFrame using the generic results_to_gdf
    gdf = results_to_gdf(all_results, nickname, easting, northing, eastings_dim, northings_dim)

    return gdf

def handle_road_corridors(easting, northing, eastings_dim, northings_dim):
    nickname = "roadCorridors"
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/road-corridors/records"
    
    # Retrieve the GeoDataFrame and all results
    all_results = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url)
    
    # Create the GeoDataFrame using the generic results_to_gdf
    gdf = results_to_gdf(all_results, nickname, easting, northing, eastings_dim, northings_dim)

    return gdf

def handle_tree_canopies(easting, northing, eastings_dim, northings_dim):
    nickname = "canopy"
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/tree-canopies-2021-urban-forest/records"
    
    # Retrieve the GeoDataFrame and all results
    all_results = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url)
    
    # Create the GeoDataFrame using the generic results_to_gdf
    gdf = results_to_gdf(all_results, nickname, easting, northing, eastings_dim, northings_dim)

    return gdf


def handle_urban_forest(easting, northing, eastings_dim, northings_dim):
    nickname = "urbanForest"
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/trees-with-species-and-dimensions-urban-forest/records"
    
    # Retrieve the GeoDataFrame and all results (using isPointData=True)
    all_results = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url, isPoint=True)
    
    # Create the GeoDataFrame using the generic results_to_gdf
    gdf = results_to_gdf(all_results, nickname, easting, northing, eastings_dim, northings_dim)
            
    return gdf

def handle_building_footprints(easting, northing, eastings_dim, northings_dim):
    nickname = "buildingFootprints"
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/2023-building-footprints/records"
    
    # Retrieve the GeoDataFrame and all results (using isPointData=True)
    all_results = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url)
    
    # Create the GeoDataFrame using the generic results_to_gdf
    gdf = results_to_gdf(all_results, nickname, easting, northing, eastings_dim, northings_dim)
            
    return gdf



    

###LOCAL SHAPEFILE STUFF
def retrieve_local_shapefile(file_path, easting, northing, eastings_dim, northings_dim):
    print(f"Retrieving local shapefile data from {file_path}")
    gdf = gpd.read_file(file_path)

    print(f'gdf.crs: {gdf.crs}')
    
    # Ensure the CRS is set to EPSG:28355
    if gdf.crs is None or gdf.crs.to_epsg() != 28355:
        gdf = gdf.to_crs(epsg=28355)
        print(f'gdf.crs after conversion: {gdf.crs}')

    
    
    # Create a bounding box from easting, northing, eastings_dim, and northings_dim
    min_easting, min_northing = easting - eastings_dim / 2, northing - northings_dim / 2
    max_easting, max_northing = easting + eastings_dim / 2, northing + northings_dim / 2
    bounding_box = box(min_easting, min_northing, max_easting, max_northing)
    
    # Filter features within the specified bounding box
    gdf_filtered = gdf[gdf.geometry.intersects(bounding_box)]
    
    print(f"Retrieved {len(gdf_filtered)} features within the bounding box")
    
    return gdf_filtered

def handle_parking(easting, northing, eastings_dim, northings_dim):
    parkingFilePath = 'data/revised/shapefiles/parking/on-street-parking-bays.shp'
    parkingMedians3mFilepath  = 'data/deployables/shapefiles/parkingmedian/parking_median_buffer.shp'

    gdfParking = retrieve_local_shapefile(parkingFilePath, easting, northing, eastings_dim, northings_dim)
    gdfParkingMedian3mBuffer = retrieve_local_shapefile(parkingMedians3mFilepath, easting, northing, eastings_dim, northings_dim)
    
    return gdfParking, gdfParkingMedian3mBuffer

    
def handle_other_road_info(easting, northing, eastings_dim, northings_dim):
    filePathLittleStreets = 'data/shapefiles/little_streets/little_streets.shp'
    filePathLaneways  = 'data/shapefiles/laneways/laneways-greening.shp'
    filePathOpenSpace = 'data/shapefiles/open_space/open-space.shp'
    filePathPrivateEmptySpace = 'data/shapefiles/private/deployables_private_empty_space.shp'

    gdfLittleStreets = retrieve_local_shapefile(filePathLittleStreets, easting, northing, eastings_dim, northings_dim)
    gdfLaneways = retrieve_local_shapefile(filePathLaneways, easting, northing, eastings_dim, northings_dim)
    gdfOpenSpace = retrieve_local_shapefile(filePathOpenSpace, easting, northing, eastings_dim, northings_dim)
    gdfPrivateEmptySpace = retrieve_local_shapefile(filePathPrivateEmptySpace, easting, northing, eastings_dim, northings_dim)
    
    return gdfLittleStreets, gdfLaneways, gdfOpenSpace, gdfPrivateEmptySpace

def handle_poles(easting, northing, eastings_dim, northings_dim):
    filepathPylons = 'data/revised/shapefiles/pylons/pylons.shp'
    filepathStreetlights = 'data/revised/shapefiles/pylons/streetlights.shp'

    gdfPylons = retrieve_local_shapefile(filepathPylons, easting, northing, eastings_dim, northings_dim)
    gdfStreetlights = retrieve_local_shapefile(filepathStreetlights, easting, northing, eastings_dim, northings_dim)
    return gdfPylons, gdfStreetlights

def handle_green_roofs(easting, northing, eastings_dim, northings_dim):
    
    green_roof_intensiveFilePath = 'data/revised/shapefiles/GreenRooftopPolygonLayers/mga55_gda94_green_roof_intensive.shp'
    brown_roof_extensive_filepath = 'data/revised/shapefiles/GreenRooftopPolygonLayers/mga55_gda94_green_roof_extensive.shp'

    gdfGreenRoof = retrieve_local_shapefile(green_roof_intensiveFilePath, easting, northing, eastings_dim, northings_dim)
    gdfBrownRoof = retrieve_local_shapefile(brown_roof_extensive_filepath, easting, northing, eastings_dim, northings_dim)
    
    # Define the ratings and the corresponding integer values
    ratings = ["Very Poor", "Poor", "Moderate", "Good", "Excellent"]
    rating_map = {rating: i + 1 for i, rating in enumerate(ratings)}
    
    # Function to map ratings to integers
    def map_rating_to_int(rating):
        return rating_map.get(rating, 0)  # 0 for unassessed or any string not in ratings

    # Convert 'RATING' to ordered categories and create 'ratingInt'
    for gdf in [gdfGreenRoof, gdfBrownRoof]:
        gdf['RATING'] = pd.Categorical(gdf['RATING'], categories=ratings, ordered=True)
        gdf['ratingInt'] = gdf['RATING'].apply(map_rating_to_int)

        # **Convert the 'ratingInt' categorical column to integer before saving**
        gdf['ratingInt'] = gdf['ratingInt'].astype(int)
        gdf['RATING'] = gdf['RATING'].astype(str)  # Convert 'RATING' back to string
    
    #plot_gdf(gdfGreenRoof, 'ratingInt')
    #plot_gdf(gdfBrownRoof, 'ratingInt')
    
    return gdfGreenRoof, gdfBrownRoof



def handle_contours(easting, northing, eastings_dim, northings_dim):
    file_path = "data/revised/shapefiles/contours/EL_CONTOUR_1TO5M.shp"
    print(f"Retrieving contour data from {file_path}")
    gdf = gpd.read_file(file_path)
    
    # Ensure the CRS is set to EPSG:28355
    if gdf.crs is None or gdf.crs.to_epsg() != 28355:
        gdf = gdf.to_crs(epsg=28355)
    
    # Create a rectangle from easting, northing, eastings_dim, and northings_dim
    min_easting, min_northing = easting - eastings_dim / 2, northing - northings_dim / 2
    max_easting, max_northing = easting + eastings_dim / 2, northing + northings_dim / 2
    
    print(f'min_easting: {max_easting}, min_northing{min_northing}, max_easting{max_easting}, max_northing{max_northing}')
    
    rectangle = Polygon([(min_easting, min_northing), (max_easting, min_northing), 
                         (max_easting, max_northing), (min_easting, max_northing)])
    
    # Filter features that intersect with the rectangle
    gdf_filtered = gdf[gdf.geometry.intersects(rectangle)]
    
    print(f"Retrieved {len(gdf_filtered)} contour lines that intersect with the rectangle")
    
    # Check if ALTITUDE column exists
    if 'ALTITUDE' not in gdf_filtered.columns:
        raise ValueError("ALTITUDE column not found. Available columns: " + ", ".join(gdf_filtered.columns))
    
    # Extract points and elevations from the contour lines
    points = []
    elevations = []
    for _, row in gdf_filtered.iterrows():
        if isinstance(row.geometry, LineString):
            points.extend(list(row.geometry.coords))
            elevations.extend([float(row['ALTITUDE'])] * len(row.geometry.coords))
        elif isinstance(row.geometry, MultiLineString):
            for line in row.geometry.geoms:
                points.extend(list(line.coords))
                elevations.extend([float(row['ALTITUDE'])] * len(line.coords))
    
    points = np.array(points)
    elevations = np.array(elevations)
    
    # Define a constant 1-meter spacing for the grid
    grid_spacing = 1.0  # 1 meter spacing
    
    # Extend the range slightly beyond the dims to fit the 1-meter spacing
    x_min, x_max = min_easting, max_easting
    y_min, y_max = min_northing, max_northing
    
    # Adjust the limits to ensure full coverage with 1-meter spacing
    x_max = x_min + (np.ceil((x_max - x_min) / grid_spacing) * grid_spacing)
    y_max = y_min + (np.ceil((y_max - y_min) / grid_spacing) * grid_spacing)
    
    # Create a grid with 1-meter spacing
    grid_x, grid_y = np.mgrid[x_min:x_max:grid_spacing, y_min:y_max:grid_spacing]
    
    # Interpolate the elevations
    grid_z = griddata(points[:, :2], elevations, (grid_x, grid_y), method='linear')
    
    # Replace NaN values with the minimum elevation
    min_elevation = np.min(elevations)
    grid_z = np.nan_to_num(grid_z, nan=min_elevation)
    
    # Create a PyVista structured grid
    grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
    
    # Add the elevation data as a scalar field
    grid.point_data["Elevation"] = grid_z.ravel()
    
    # Create a boundary box
    boundary_box = pv.Box(bounds=(x_min, x_max, y_min, y_max, grid_z.min(), grid_z.max()))
    
    # Clip the mesh with the boundary box
    clipped_grid = grid.clip_box(boundary_box, invert=False)
    
    print("Created 3D terrain mesh from contour data")
    return clipped_grid, gdf_filtered



import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from rtree import index



#########

import geopandas as gpd
import numpy as np
from rasterio import features
from affine import Affine
from shapely.geometry import box
import pandas as pd
from scipy.spatial import cKDTree

def rasterize_shapefile(gdf, transform, out_shape):
    """Rasterize the shapefile with unique IDs for each geometry."""
    shapes = [(geom, i) for i, geom in enumerate(gdf.geometry)]
    raster = features.rasterize(shapes, out_shape=out_shape, transform=transform, fill=-1, all_touched=True)
    return raster

def raster_to_points(raster, affine):
    """Convert raster to 2D points, keeping only filled areas."""
    rows, cols = np.where(raster != -1)
    points = np.vstack((cols, rows)).T
    # Convert pixel coordinates to real-world coordinates
    real_coords = affine * points
    return real_coords, raster[rows, cols]

def assign_attributes(points, ids, gdf):
    """Assign attributes from the GeoDataFrame to the points."""
    # Create a DataFrame for the points with their IDs
    points_df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1], 'id': ids})
    
    # Merge attributes based on ID
    merged_df = points_df.merge(gdf.reset_index(), left_on='id', right_index=True, how='left')
    
    # Drop unnecessary columns and return
    return merged_df.drop(['geometry', 'id'], axis=1)

def process_shapefile(shapefile_path, bbox, resolution=0.25):
    """Process a shapefile: read, crop, rasterize, extract points, and assign attributes."""
    # Read and crop shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf_cropped = gdf[gdf.geometry.intersects(bbox)]
    
    if gdf_cropped.empty:
        print(f"No geometries from {shapefile_path} intersect with the bounding box.")
        return None, None

    # Prepare for rasterization
    bounds = bbox.bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = Affine.translation(bounds[0], bounds[1]) * Affine.scale(resolution, resolution)
    
    # Rasterize
    raster = rasterize_shapefile(gdf_cropped, transform, (height, width))
    
    # Extract points
    points, ids = raster_to_points(raster, ~transform)
    
    # Assign attributes
    if len(points) > 0:
        attributes = assign_attributes(points, ids, gdf_cropped)
        return points, attributes
    else:
        print(f"No points extracted from {shapefile_path} after rasterization.")
        return None, None

def create_subset_dataset(ds, variables, attributes):
    """
    Create a subset of the xarray Dataset by dropping all variables not in the specified list
    and keeping only the specified attributes.
    
    Parameters:
    ds (xarray.Dataset): The original Dataset
    variables (list): List of variable names to keep
    attributes (list): List of attribute names to keep
    
    Returns:
    xarray.Dataset: A new Dataset with only the specified variables and attributes
    """
    # Create a copy of the original dataset
    subset_ds = ds.copy(deep=True)
    
    # Get the list of variables to drop
    vars_to_drop = [var for var in subset_ds.variables if var not in variables and var not in subset_ds.dims]
    
    # Drop the variables not in the list
    subset_ds = subset_ds.drop_vars(vars_to_drop)
    
    # Keep only the specified attributes
    subset_ds.attrs = {k: v for k, v in subset_ds.attrs.items() if k in attributes}
    
    return subset_ds

def plot_gdf(gdf, attribute=None):
    """
    Plot the GeoDataFrame with an optional attribute for coloring.
    
    Parameters:
    - gdf: The GeoDataFrame to plot.
    - attribute: Optional; the name of the attribute to use for coloring.
    """
    # Create a plot with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the GeoDataFrame with or without an attribute
    if attribute and attribute in gdf.columns:
        gdf.plot(ax=ax, column=attribute, legend=True, cmap='viridis', edgecolor='blue', alpha=0.5)
    else:
        gdf.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.5)
    
    # Add plot title and labels
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    
    # Show the plot
    plt.show()


import os
# Example usage
if __name__ == "__main__":
    site_name = 'trimmed-parade'  # Melbourne Connect site
    easting, northing = get_site_coordinates(site_name)
    eastings_dim = 1000
    northings_dim = 1000

    parkingGdf, parkingMedian3mBufferGdf = handle_parking(easting, northing, eastings_dim, northings_dim)
    plot_gdf(parkingGdf)
    plot_gdf(parkingMedian3mBufferGdf)


    #gdfLittleStreets, gdfLaneways, gdfOpenSpace, gdfPrivateEmptySpace = handle_other_road_info(easting, northing, eastings_dim, northings_dim)
    #gdfRoadCorridors = handle_road_corridors(easting, northing, eastings_dim, northings_dim)
"""def efficient_point_in_polygon(polygon_gdf, points):
    # Create spatial index for the polygons
    spatial_index = index.Index()
    for idx, geometry in enumerate(polygon_gdf.geometry):
        spatial_index.insert(idx, geometry.bounds)

    # Function to check if a point is in any polygon
    def check_point(point):
        potential_matches_idx = list(spatial_index.intersection(point.coords[0]))
        return any(polygon_gdf.iloc[idx].geometry.contains(point) for idx in potential_matches_idx)

    # Apply the check to all points
    results = np.array([check_point(Point(x, y)) for x, y in points])
    
    return results

def plot_points_and_polygons(polygon_gdf, points, results):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot polygons
    polygon_gdf.plot(ax=ax, alpha=0.5, edgecolor='black')
    
    # Separate points inside and outside
    points_inside = points[results]
    points_outside = points[~results]
    
    # Plot points
    ax.scatter(points_inside[:, 0], points_inside[:, 1], c='green', label='Inside', alpha=0.6, s=10)
    ax.scatter(points_outside[:, 0], points_outside[:, 1], c='red', label='Outside', alpha=0.6, s=10)
    
    ax.set_title('Points Inside and Outside Polygons')
    ax.legend()
    plt.tight_layout()
    plt.show()




###
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from rtree import index
import time
import psutil

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def create_polygon_index(gdf):
    print("Starting to create spatial index for polygons...")
    start_time = time.time()
    idx = index.Index()
    for i, geometry in enumerate(gdf.geometry):
        idx.insert(i, geometry.bounds)
        if (i + 1) % 1000 == 0:
            print(f"Indexed {i + 1} polygons...")
    print(f"Spatial index created in {time.time() - start_time:.2f} seconds")
    log_memory_usage()
    return idx

def generate_grid_chunks(minx, miny, maxx, maxy, spacing, chunk_size=1000000):
    print(f"Generating grid chunks with spacing {spacing} and chunk size {chunk_size}")
    x_chunks = np.arange(minx, maxx, spacing * chunk_size)
    total_chunks = len(x_chunks)
    print(f"Total number of chunks to process: {total_chunks}")
    for i, x_start in enumerate(x_chunks):
        print(f"Generating chunk {i + 1}/{total_chunks}")
        x_end = min(x_start + spacing * chunk_size, maxx)
        x = np.arange(x_start, x_end, spacing)
        y = np.arange(miny, maxy, spacing)
        xx, yy = np.meshgrid(x, y)
        yield xx.ravel(), yy.ravel()

def process_large_grid(gdf, easting, northing, eastings_dim, northings_dim, spacing=0.25):
    print(f"Initializing grid processing...")
    print(f"Grid dimensions: {eastings_dim}x{northings_dim}, spacing: {spacing}")
    print(f"Center point: Easting {easting}, Northing {northing}")
    start_time = time.time()
    
    minx, miny = easting - eastings_dim/2, northing - northings_dim/2
    maxx, maxy = easting + eastings_dim/2, northing + northings_dim/2
    print(f"Grid bounds: ({minx}, {miny}) to ({maxx}, {maxy})")
    
    polygon_index = create_polygon_index(gdf)
    
    interior_points = []
    total_points = 0
    chunks_processed = 0
    
    print("Starting to process grid points...")
    for chunk_x, chunk_y in generate_grid_chunks(minx, miny, maxx, maxy, spacing):
        chunk_start_time = time.time()
        chunk_points = np.column_stack((chunk_x, chunk_y))
        chunk_size = len(chunk_points)
        total_points += chunk_size
        chunks_processed += 1
        
        print(f"\nProcessing chunk {chunks_processed}")
        print(f"Chunk size: {chunk_size} points")
        print(f"Chunk bounds: ({chunk_points[:, 0].min()}, {chunk_points[:, 1].min()}) to ({chunk_points[:, 0].max()}, {chunk_points[:, 1].max()})")
        
        chunk_interior_points = []
        for i, point in enumerate(chunk_points):
            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1}/{chunk_size} points in current chunk...")
            potential_matches_idx = list(polygon_index.intersection(point))
            if any(gdf.iloc[idx].geometry.contains(Point(point)) for idx in potential_matches_idx):
                chunk_interior_points.append(point)
        
        interior_points.extend(chunk_interior_points)
        
        chunk_time = time.time() - chunk_start_time
        print(f"Chunk {chunks_processed} completed in {chunk_time:.2f} seconds.")
        print(f"Points in this chunk: {chunk_size}")
        print(f"Interior points in this chunk: {len(chunk_interior_points)}")
        print(f"Total points processed so far: {total_points}")
        print(f"Total interior points found so far: {len(interior_points)}")
        print(f"Current processing rate: {chunk_size / chunk_time:.0f} points/second")
        print(f"Percentage of points inside (this chunk): {len(chunk_interior_points) / chunk_size * 100:.2f}%")
        print(f"Overall percentage of points inside: {len(interior_points) / total_points * 100:.2f}%")
        log_memory_usage()
        print("--------------------")
    
    total_time = time.time() - start_time
    print("\nProcessing completed!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total chunks processed: {chunks_processed}")
    print(f"Total points processed: {total_points}")
    print(f"Total interior points found: {len(interior_points)}")
    print(f"Overall processing rate: {total_points / total_time:.0f} points/second")
    print(f"Final percentage of points inside: {len(interior_points) / total_points * 100:.2f}%")
    log_memory_usage()
    
    return np.array(interior_points)


# Example usage
if __name__ == "__main__":
    site_name = 'uni'  # Melbourne Connect site
    easting, northing = get_site_coordinates(site_name)
    eastings_dim = 1000
    northings_dim = 1000
    
    # Retrieve tree canopy data
    gdf = handle_road_segments(easting, northing, eastings_dim, northings_dim)
    
    # Process the grid
    print("\nStarting grid processing...")
    interior_grid = process_large_grid(gdf, easting, northing, eastings_dim, northings_dim)
    
    print(f"\nFinal Results:")
    print(f"Total number of interior points: {len(interior_grid)}")
    print(f"Bounding box of interior points:")
    print(f"  Min X: {interior_grid[:, 0].min()}, Max X: {interior_grid[:, 0].max()}")
    print(f"  Min Y: {interior_grid[:, 1].min()}, Max Y: {interior_grid[:, 1].max()}")
    
    print("\nGenerating visualization...")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf.plot(ax=ax, alpha=0.5, edgecolor='black')
    sample_size = min(10000, len(interior_grid))
    sample_points = interior_grid[np.random.choice(len(interior_grid), sample_size, replace=False)]
    ax.scatter(sample_points[:, 0], sample_points[:, 1], c='red', s=1, alpha=0.5)
    plt.title(f'Sample of Interior Grid Points (showing {sample_size} out of {len(interior_grid)})')
    plt.tight_layout()
    print("Saving visualization to 'grid_visualization.png'...")
    plt.savefig('grid_visualization.png', dpi=300)
    plt.close()
    
    print("\nProgram completed successfully!")
    log_memory_usage()"""






def plot_outline_data(gdf, plotter, color='white', opacity=0.5, line_width=2, label=None):
    print("Creating 3D outline visualization...")
    all_points = []
    all_faces = []
    total_points = 0
    extrusion_height = 1  # 1 meter extrusion

    for _, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, (Polygon, MultiPolygon)):
            if isinstance(geom, MultiPolygon):
                polygons = geom.geoms
            else:
                polygons = [geom]

            for polygon in polygons:
                exterior_coords = np.array(polygon.exterior.coords)
                if np.allclose(exterior_coords[0], exterior_coords[-1]):
                    exterior_coords = exterior_coords[:-1]

                if len(exterior_coords) < 3:
                    continue

                bottom = np.column_stack((exterior_coords, np.zeros(len(exterior_coords))))
                top = np.column_stack((exterior_coords, np.full(len(exterior_coords), extrusion_height)))
                points = np.vstack((bottom, top))

                num_points = len(exterior_coords)
                faces = []
                for i in range(num_points):
                    faces.extend([4, total_points + i, total_points + (i+1)%num_points, 
                                  total_points + num_points + (i+1)%num_points, total_points + num_points + i])

                all_points.append(points)
                all_faces.extend(faces)
                total_points += len(points)

                # Add interior rings (holes) if any
                for interior in polygon.interiors:
                    interior_coords = np.array(interior.coords)
                    if np.allclose(interior_coords[0], interior_coords[-1]):
                        interior_coords = interior_coords[:-1]

                    bottom = np.column_stack((interior_coords, np.zeros(len(interior_coords))))
                    top = np.column_stack((interior_coords, np.full(len(interior_coords), extrusion_height)))
                    points = np.vstack((bottom, top))

                    num_points = len(interior_coords)
                    faces = []
                    for i in range(num_points):
                        faces.extend([4, total_points + i, total_points + (i+1)%num_points, 
                                      total_points + num_points + (i+1)%num_points, total_points + num_points + i])

                    all_points.append(points)
                    all_faces.extend(faces)
                    total_points += len(points)

        elif isinstance(geom, (LineString, MultiLineString)):
            if isinstance(geom, MultiLineString):
                lines = geom.geoms
            else:
                lines = [geom]

            for line in lines:
                coords = np.array(line.coords)
                bottom = np.column_stack((coords, np.zeros(len(coords))))
                top = np.column_stack((coords, np.full(len(coords), extrusion_height)))
                points = np.vstack((bottom, top))

                num_points = len(coords)
                faces = []
                for i in range(num_points - 1):
                    faces.extend([4, total_points + i, total_points + i + 1, 
                                  total_points + num_points + i + 1, total_points + num_points + i])

                all_points.append(points)
                all_faces.extend(faces)
                total_points += len(points)

    # Combine all points and faces into a single mesh
    combined_points = np.vstack(all_points)
    combined_mesh = pv.PolyData(combined_points, np.array(all_faces))

    # Add the mesh to the plotter
    plotter.add_mesh(combined_mesh, color=color, opacity=opacity, show_edges=True, line_width=line_width, label=label)

    
    
    print(f"Created 3D visualization for {len(gdf)} geometries")

def plot_point_data(gdf, plotter, color='green', point_size=5, label=None):
    points = np.array([[point.x, point.y, 0] for point in gdf.geometry])
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(point_cloud, color=color, point_size=point_size, render_points_as_spheres=True, label=label)
    

def plot_terrain_mesh(terrain_mesh, plotter, cmap='terrain', opacity=1, show_edges=False, label=None):
    plotter.add_mesh(terrain_mesh, scalars="Elevation", cmap=cmap, opacity=opacity, show_edges=show_edges, label=label)


def main2():
    # Set up the parameters
    site_name = 'uni'  # Melbourne Connect site
    easting, northing = get_site_coordinates(site_name)
    eastings_dim = 1000
    northings_dim = 1000

    print(f"Testing for site: {site_name}")
    print(f"Easting: {easting}, Northing: {northing}")
    
    try:
        # Retrieve contour data and create terrain mesh
        terrain_mesh, contours_gdf = handle_contours(easting, northing, eastings_dim, northings_dim)

        plotter = pv.Plotter()

        if terrain_mesh is not None and "Elevation" in terrain_mesh.point_data:
            plot_terrain_mesh(terrain_mesh, plotter, label='Terrain')
        else:
            print("Warning: Terrain mesh could not be created or lacks elevation data.")

        if contours_gdf is not None:
            plot_outline_data(contours_gdf, plotter, color='black', opacity=1, line_width=1, label='Contour Lines')

        # Plot the site center
        center_point = pv.PolyData(np.array([[easting, northing, terrain_mesh.points[:, 2].mean()]]))
        plotter.add_mesh(center_point, color='red', point_size=10, render_points_as_spheres=True, label='Site Center')
        
        # Add text for site center
        plotter.add_point_labels(center_point, ['Site Center'], font_size=10, point_size=1)
        
        # Set up the camera
        plotter.camera_position = [
            (easting + eastings_dim / 2, northing + northings_dim / 2, max(eastings_dim, northings_dim)),  # Camera position
            (easting, northing, terrain_mesh.points[:, 2].mean()),  # Focal point
            (0, 0, 1)  # View up direction
        ]
        
        # Add a colorbar
        plotter.add_scalar_bar('Elevation', vertical=True)
        
        # Add a legend
        plotter.add_legend()
        
        print("Displaying visualization...")
        plotter.show()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()




def create_3d_building_visualization(gdf):
    print("Creating 3D building visualization...")
    building_data = []
    all_points = []
    all_faces = []
    total_points = 0
    
    for idx, row in gdf.iterrows():
        try:
            geom = row.geometry
            if isinstance(geom, MultiPolygon):
                geom = max(geom, key=lambda x: x.area)
            
            if isinstance(geom, Polygon):
                exterior_coords = np.array(geom.exterior.coords)
                if np.allclose(exterior_coords[0], exterior_coords[-1]):
                    exterior_coords = exterior_coords[:-1]
                
                if len(exterior_coords) < 3:
                    continue
                
                height = row.get('structure_extrusion') or row.get('footprint_extrusion') or (row.get('structure_max_elevation', 0) - row.get('structure_min_elevation', 0))
                
                bottom = np.column_stack((exterior_coords, np.zeros(len(exterior_coords))))
                top = np.column_stack((exterior_coords, np.full(len(exterior_coords), height)))
                points = np.vstack((bottom, top))
                
                num_points = len(exterior_coords)
                faces = []
                faces.extend([num_points] + list(range(total_points + num_points, total_points + 2*num_points)))
                for i in range(num_points):
                    faces.extend([4, total_points + i, total_points + (i+1)%num_points, 
                                  total_points + num_points + (i+1)%num_points, total_points + num_points + i])
                
                all_points.append(points)
                all_faces.extend(faces)
                total_points += len(points)
                
                top_cap_data = analyze_top_cap(points, faces[:num_points+1], bottom)
                building_data.append((idx, height, top_cap_data))
        except Exception as e:
            print(f"Error processing building {idx}: {e}")
    
    # Combine all points and faces into a single mesh
    combined_points = np.vstack(all_points)
    combined_mesh = pv.PolyData(combined_points, np.array(all_faces))
    
    df = pd.DataFrame(building_data, columns=['building_id', 'height', 'top_cap_data'])
    df = pd.concat([df, df['top_cap_data'].apply(pd.Series)], axis=1)
    df = df.drop('top_cap_data', axis=1)
    
    print(f"Created 3D visualization for {len(building_data)} buildings")
    return combined_mesh, df

def analyze_top_cap(points, face, bottom_coords):
    top_coords = points[face[1:]]
    top_area = polygon_area(top_coords)
    bottom_area = polygon_area(bottom_coords)
    top_centroid = np.mean(top_coords, axis=0)
    bottom_centroid = np.mean(bottom_coords, axis=0)
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
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0
