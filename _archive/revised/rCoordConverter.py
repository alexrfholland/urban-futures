from pyproj import Proj, transform

# Convert from DMS to Decimal Degrees
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + minutes / 60 + seconds / 3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

# Input coordinates in DMS
lat_dms = (37, 47, 59.5, 'S')
lon_dms = (144, 57, 50.7, 'E')

# Convert to Decimal Degrees
lat_decimal = dms_to_decimal(*lat_dms)
lon_decimal = dms_to_decimal(*lon_dms)

# Define the projection for GDA94 / MGA zone 55
proj_gda94 = Proj("epsg:28355")

# Convert from geographic (lat, lon) to UTM (Easting, Northing)
easting, northing = proj_gda94(lon_decimal, lat_decimal)

lat_decimal, lon_decimal, easting, northing

print(lat_decimal, lon_decimal, easting, northing)
