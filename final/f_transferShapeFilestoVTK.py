# f_GeospatialModule.py

import numpy as np
import pandas as pd
import geopandas as gpd
import pyvista as pv
from shapely.geometry import Point, Polygon
from dask import delayed, compute
import warnings
import copy

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
        # Convert to categorical type
        gdf_encoded[col] = gdf_encoded[col].astype('category')
        # Create category mappings
        category_mappings[col] = dict(enumerate(gdf_encoded[col].cat.categories))
        # Replace column values with integer codes
        gdf_encoded[col] = gdf_encoded[col].cat.codes

        # Validation: Ensure the column is now of integer type
        assert pd.api.types.is_integer_dtype(gdf_encoded[col]), \
            f"Encoding failed for column {col}. Expected integer type."

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
            print(f"Skipping geometry column '{col}'.")
            continue

        col_dtype = gdf_encoded[col].dtype

        # Handle categorical and object dtypes as integer codes
        if pd.api.types.is_categorical_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype):
            print(f"Initializing integer array for encoded column '{col}'.")
            dask_arrays[col] = delayed(lambda: np.full(num_points, -1, dtype=np.int32))()

        # Handle numeric columns (floats or integers)
        elif col_dtype.kind in 'f':  # Floating point types
            print(f"Initializing float array for column '{col}'.")
            dask_arrays[col] = delayed(lambda: np.full(num_points, np.nan, dtype=col_dtype))()

        elif col_dtype.kind in 'i':  # Integer types
            print(f"Initializing integer array for column '{col}' with default value -1.")
            dask_arrays[col] = delayed(lambda: np.full(num_points, -1, dtype=col_dtype))()

        else:
            raise TypeError(f"Unsupported column type for '{col}': {col_dtype}")

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
            if pd.api.types.is_integer_dtype(gdf_encoded[col].dtype):
                # Replace NaN with -1 for integer columns and cast to int
                values = joined[col].fillna(-1).astype(int).values
            else:
                values = joined[col].values
            result['values'][col] = values

    return result


def assign_gdf_attributes_to_polydata(polydata, gdf, attribute_prefix, chunk_size=5000000, binaryMask=None):
    """
    Assign attributes from a GeoDataFrame (gdf) to the points in a PyVista PolyData object.

    Parameters:
    - polydata: PyVista PolyData object containing points and point data.
    - gdf: GeoDataFrame containing polygon geometries and associated attributes.
    - attribute_prefix: Prefix to prepend to each attribute name when adding back to PolyData.
    - chunk_size: Size of the chunks to process in parallel (default: 5000000).
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
                print(f"Decoding categorical column '{col}' for prefix '{attribute_prefix}'.")

                # Create a lookup array for decoding
                max_code = max(category_mappings[col].keys())
                lookup = np.empty(max_code + 1, dtype=object)
                for code, category in category_mappings[col].items():
                    lookup[code] = category

                # Ensure 'concatenated' is of integer type
                if not np.issubdtype(concatenated.dtype, np.integer):
                    concatenated = concatenated.astype(int)

                # Replace -1 with max_code + 1 for safe indexing
                mask_invalid = concatenated == -1
                concatenated_safe = concatenated.copy()
                concatenated_safe[mask_invalid] = max_code + 1

                # Append the extra slot to lookup
                lookup = np.append(lookup, '')  # Add an extra slot for -1

                # Perform vectorized indexing
                decoded = lookup[concatenated_safe]

                # Assign empty string to invalid entries
                decoded[mask_invalid] = ''

                final_attributes[col] = decoded
            else:
                # Ensure integer columns remain integers
                if pd.api.types.is_integer_dtype(gdf_encoded[col].dtype):
                    concatenated = concatenated.astype(np.int32)
                final_attributes[col] = concatenated

        # Step 12: Assign attributes to the PolyData's point_data
        for col, array in final_attributes.items():
            attribute_name = f"{attribute_prefix}{col}"
            polydata.point_data[attribute_name] = array

    print("Attribute assignment completed successfully.")
    return polydata


def getSpatials(siteVTK, greenRoofGDF, brownRoofGDF):
    """
    Assign attributes from green and brown roof GeoDataFrames to the PyVista PolyData object.

    Parameters:
    - siteVTK: PyVista PolyData object containing the point cloud.
    - greenRoofGDF: GeoDataFrame containing green roof polygons and attributes.
    - brownRoofGDF: GeoDataFrame containing brown roof polygons and attributes.

    Returns:
    - Updated PyVista PolyData object with green and brown roof attributes.
    """
    print('Transferring green roofs...')
    siteVTK = assign_gdf_attributes_to_polydata(
        polydata=siteVTK,
        gdf=greenRoofGDF,
        attribute_prefix='greenRoof_'
    )
    print('Green roof transfer complete!')

    print('Transferring brown roofs...')
    siteVTK = assign_gdf_attributes_to_polydata(
        polydata=siteVTK,
        gdf=brownRoofGDF,
        attribute_prefix='brownRoof_'
    )
    print('Brown roof transfer complete!')

    return siteVTK


# Simplified Systematic Checks
def systematic_checks(polydata, attribute_prefixes=None):
    """
    Perform systematic checks on the assigned attributes.

    Parameters:
    - polydata: PyVista PolyData object with assigned attributes.
    - attribute_prefixes: List of prefixes to filter relevant attributes. If None, all attributes are checked.
    """
    print("\nSystematic Checks:")

    # If specific prefixes are provided, filter attributes by these prefixes
    if attribute_prefixes is not None:
        relevant_attributes = [attr for attr in polydata.point_data.keys() 
                               if any(attr.startswith(prefix) for prefix in attribute_prefixes)]
    else:
        relevant_attributes = list(polydata.point_data.keys())

    for attr in relevant_attributes:
        # Skip binary masks or non-ratingInt attributes if necessary
        if not attr.endswith('ratingInt'):
            continue  # Adjust this condition based on your attribute naming conventions

        print(f"\nChecking attribute: {attr}")

        ratings = polydata.point_data[attr]
        assigned = np.sum(ratings != -1)
        unassigned = np.sum(ratings == -1)
        print(f"  Assigned: {assigned}")
        print(f"  Unassigned: {unassigned}")

        # Count per category
        unique, counts = np.unique(ratings[ratings != -1], return_counts=True)
        rating_counts = dict(zip(unique, counts))
        print("  Ratings Distribution:")
        for code, count in rating_counts.items():
            print(f"    Code {code}: {count} points")


if __name__ == "__main__":
    # Example usage with actual data

    # Define the site
    site = 'city'
    site_vtk_path = f'data/revised/{site}-siteVoxels.vtk'

    # Read the PyVista PolyData object
    print(f"Loading PolyData from {site_vtk_path}...")
    siteVTK = pv.read(site_vtk_path)
    print("PolyData loaded successfully.")

    # Import necessary modules
    import f_GeospatialModule
    import f_SiteCoordinates

    # Get spatial dimensions and center
    easting, northing, eastingsDim, northingsDim = f_SiteCoordinates.get_center_and_dims([siteVTK])

    # Handle green and brown roofs (Assuming handle_green_roofs returns two GeoDataFrames)
    greenRoofGDF, brownRoofGDF = f_GeospatialModule.handle_green_roofs(easting, northing, eastingsDim, northingsDim)

    # Assign spatial attributes
    filtered_site_voxels = getSpatials(siteVTK, greenRoofGDF, brownRoofGDF)

    # Perform systematic checks
    systematic_checks(filtered_site_voxels, attribute_prefixes=['greenRoof_', 'brownRoof_'])

    # Sequentially plot each attribute
    for attr in filtered_site_voxels.point_data.keys():
        if 'ratingInt' in attr:
            print(f"Plotting attribute: {attr}")
            filtered_site_voxels.plot(scalars=attr, cmap='viridis', show_scalar_bar=True)
