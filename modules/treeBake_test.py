import numpy as np
import pyvista as pv

def extract_surface_from_polydata(polydata: pv.PolyData, scalar_function, isovalue: float = 0.5) -> pv.PolyData:
    """
    Extract a surface from a scalar field using the Marching Cubes algorithm.

    Parameters:
    - polydata: A PyVista PolyData object containing the points of the grid.
    - scalar_function: A function that takes x, y, z arrays and returns the scalar values.
    - isovalue: The scalar value at which to extract the surface (default is 0.5).

    Returns:
    - A PyVista PolyData object representing the extracted isosurface.
    """

    # Extract the points from the polydata object
    x, y, z = polydata.points.T

    # Calculate the scalar values using the provided scalar function
    values = scalar_function(x, y, z)

    # Attach the scalar field to the polydata object
    polydata.point_data['values'] = values

    # Apply the contour filter (Marching Cubes algorithm) directly on the polydata object
    isosurface = polydata.contour(isosurfaces=[isovalue], scalars='values', method='marching_cubes')

    return isosurface

# Example usage:
def example_scalar_function(x, y, z):
    return np.sin(np.sqrt(x**2 + y**2 + z**2))

# Assuming you have a polydata object:
# polydata = pv.PolyData(points)

# Extract the isosurface
# isosurface = extract_surface_from_polydata(polydata, example_scalar_function, isovalue=0.5)
# isosurface.plot(scalars='values', smooth_shading=True, cmap="viridis", show_scalar_bar=False)
