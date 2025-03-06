import pyvista as pv
import numpy as np

# Function to log the count of ispylons == True
def log_ispylons_count(poly_data, label):
    ispylons_array = poly_data.point_data['ispylons']
    ispylons_count = np.sum(ispylons_array == True)
    print(f'Number of points with ispylons == True in {label}: {ispylons_count}')



# Load the PolyData from a file
file_path = "data/street/updated-street.vtk"  # Replace with the actual path to your VTK file
poly_data = pv.read(file_path)

# Log the number of ispylons == True in the original PolyData
log_ispylons_count(poly_data, 'poly_data')

# Create a boolean mask where ispylons == True
ispylons_mask = poly_data.point_data['ispylons'] == True

# Log the sum of True values in ispylons_mask
ispylons_mask_sum = np.sum(ispylons_mask)
print(f"Sum of True values in ispylons_mask: {ispylons_mask_sum}")


# Extract points using the boolean mask
extracted_polydata = poly_data.extract_points(ispylons_mask)

# Log the number of ispylons == True in the extracted PolyData
log_ispylons_count(extracted_polydata, 'extracted_polydata')
