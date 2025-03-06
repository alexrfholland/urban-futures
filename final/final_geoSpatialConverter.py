import itertools
import numpy as np
import pyvista as pv
import geopandas as gpd
from shapely import speedups
from shapely.geometry import shape
from f_GeospatialModule import retrieve_geospatial_data, get_site_coordinates

speedups.disable()


def handle_road_segments(easting, northing, radius_meters):
    dataset_name = "road-segments-with-surface-type"
    road_segments = retrieve_geospatial_data(easting, northing, radius_meters, dataset_name)
    
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
        print(f"Created GeoDataFrame with {len(gdf)} road segments")
        return gdf
    else:
        print("Failed to retrieve or process road segment data.")
        return None

def create_line_vtk(gdf):
    lineTubes = {}

    for index, row in gdf.iterrows():
        cellSec = []
        linePointSec = []

        z = row.get('elevation', 0)  # Use 'elevation' if available, otherwise 0
        zipObject = zip(row.geometry.xy[0], row.geometry.xy[1], itertools.repeat(z))
        for linePoint in zipObject:
            linePointSec.append(linePoint)

        nPoints = len(list(row.geometry.coords))
        cellSec = [nPoints] + list(range(nPoints))

        cellSecArray = np.array(cellSec)
        cellTypeArray = np.array([4])
        linePointArray = np.array(linePointSec)

        partialLineUgrid = pv.UnstructuredGrid(cellSecArray, cellTypeArray, linePointArray)
        
        # Add attributes if available
        for column in gdf.columns:
            if column != 'geometry':
                partialLineUgrid.cell_arrays[column] = row[column]

        lineTubes[str(index)] = partialLineUgrid

    lineBlocks = pv.MultiBlock(lineTubes)
    return lineBlocks.combine()

def main():
    # Set up the parameters
    site_name = 'uni'  # Melbourne Connect site
    easting, northing = get_site_coordinates(site_name)
    radius_meters = 200

    print(f"Processing data for site: {site_name}")
    print(f"Easting: {easting}, Northing: {northing}")
    
    # Retrieve and process road segments data
    road_segments_gdf = handle_road_segments(easting, northing, radius_meters)
    
    if road_segments_gdf is not None:
        print(f"Converting {len(road_segments_gdf)} road segments to VTK format...")
        road_segments_vtk = create_line_vtk(road_segments_gdf)
        
        # Visualize the results
        plotter = pv.Plotter()
        plotter.add_mesh(road_segments_vtk, color='yellow', line_width=2, label='Road Segments')
        plotter.add_point_labels([[easting, northing, 0]], ["Site Center"], point_size=20, font_size=16)
        plotter.add_legend()
        
        print("Displaying 3D visualization...")
        plotter.show()
        
        # Save the VTK file
        output_file = "road_segments.vtk"
        road_segments_vtk.save(output_file)
        print(f"Saved road segments to {output_file}")
    else:
        print("No road segments data to process.")

if __name__ == "__main__":
    main()