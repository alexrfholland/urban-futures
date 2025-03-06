import pyvista as pv
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import pandas as pd
import math
isChatGpt = True

def getEastingAndNorthing(site):
    site_coord = pd.read_csv('data/site projections.csv')
    print(site)
    #print(site_coord)

    # Extract easting and northing offsets for the site
    easting_offset = site_coord.loc[site_coord['Name'].str.contains(site, case=False), 'Easting'].values[0]
    northing_offset = site_coord.loc[site_coord['Name'].str.contains(site, case=False), 'Northing'].values[0]

    print(f'easting and northing offsets for {site} is {easting_offset}, {northing_offset}')
    
    return easting_offset, northing_offset


def getBoundsofPolydata(site):

    # Step 1: Load Site Coordinates and Calculate Offsets
    easting_offset, northing_offset = getEastingAndNorthing(site)

    # Step 2: Load Polydata and Apply Offsets
    # This is a placeholder. Replace it with your actual method of loading and handling polydata.

    vtk_path = f'data/{site}/updated-{site}.vtk'
    poly_data = pv.read(vtk_path)

    minx, maxx, miny, maxy = poly_data.bounds[0:4]
    print(f'polydata bounds are {poly_data.bounds[0:4]}')


    # Adjusting the polydata bounds with the offsets. Ceiling to avoid off by 1 pixel errors
    adjusted_minx = math.ceil(easting_offset + minx)
    adjusted_maxx = math.ceil(easting_offset + maxx)
    adjusted_miny = math.ceil(northing_offset + miny)
    adjusted_maxy = math.ceil(northing_offset + maxy)


    # Creating a bounding box for the area of interest
    bbox = box(adjusted_minx, adjusted_miny, adjusted_maxx, adjusted_maxy)

    # Convert the bounding box to a GeoDataFrame
    gdfBbox = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs="EPSG:28355")

    return gdfBbox


def cropAndSaveGeoTiff(site, file_path, name):
    gdfBbox = getBoundsofPolydata(site)
    with rasterio.open(file_path) as src:
        # Mask the area outside the bounding box
        out_image, out_transform = mask(src, gdfBbox.geometry, crop=True)
        out_meta = src.meta.copy()

         # Update the metadata to reflect the new shape and transform of the image
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        # Save the cropped GeoTIFF
    
    #outfile = f'data/{site}/geotiff-{name}.tiff'
    outfile = f'data/temp/geotiff-{site}-{name}.tif'
    with rasterio.open(outfile, "w", **out_meta) as dest:
        dest.write(out_image)

        print(outfile)

    print(f'{name} geotiff saved at {outfile}')


    #return out_image, out_transform, gdfBbox
    return outfile

def cropGeoTiff(site, file_path):
    gdfBbox = getBoundsofPolydata(site)
    print(f'bounds of bbox are: {gdfBbox.bounds}')

    
    with rasterio.open(file_path) as src:
        out_image, out_transform = mask(src, gdfBbox.geometry, crop=True)
        out_meta = src.meta.copy()

        # Update the metadata to reflect the new shape and transform of the image
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Calculate the bounds of the cropped image
        out_bounds = rasterio.transform.array_bounds(
            out_image.shape[1], out_image.shape[2], out_transform
        )
    print(f'bounds of geotiff are {out_bounds}')

    return out_image, out_transform, out_bounds



if __name__ == "__main__":

    file_path = 'data/deployables/raster_canopy_distance.tif'
    name = 'canopy-proximity'
    name = cropGeoTiff('parade', file_path, name)
    



