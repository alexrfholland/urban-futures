import pyvista as pv
import pandas as pd

if __name__ == "__main__":
    sites = ['trimmed-parade', 'street', 'city']

    # Dictionary to store data for each site
    data = {site: [] for site in sites}

    for site in sites:
        vtk_path = f'data/{site}/updated-{site}.vtk'
        poly_data = pv.read(vtk_path)
        print(poly_data.point_data)

        # Extract each poly_data.point_data column names
        data[site] = list(poly_data.point_data.keys())

    # Find the maximum number of columns
    max_cols = max(len(columns) for columns in data.values())

    # Pad columns to have the same number of rows
    for site in sites:
        data[site].extend([None] * (max_cols - len(data[site])))

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data)

    # Save the DataFrame to a CSV file

    df.to_csv('data/site_col_names.csv', index=False)
