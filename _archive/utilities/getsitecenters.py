# Import required libraries
import pandas as pd
import pyproj

#create pandas dataframe from csv file
df = pd.read_csv('data/site projections.csv')

# Initialize coordinate transformation objects
in_proj = pyproj.Proj(proj='latlong', datum='WGS84', init='epsg:4326')
out_proj = pyproj.Proj(init='epsg:28355')

# Initialize Transformer object for coordinate transformation
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:28355", always_xy=True)

# Update function to use the new Transformer object
def calculate_easting_northing_updated(row):
    easting, northing = transformer.transform(row['Longitude'], row['Latitude'])
    return pd.Series([easting, northing], index=['Easting', 'Northing'])

# Apply the updated calculation to each row
df[['Easting', 'Northing']] = df.apply(calculate_easting_northing_updated, axis=1)

print(df)

# Save the updated dataframe to the same csv file
df.to_csv('data/site projections.csv', index=False)

