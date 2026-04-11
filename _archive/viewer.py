import pyvista as pv
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import modules.getColorsAndAttributes as getColorsAndAttributes    
    
street = getColorsAndAttributes.transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("data/street-furniture.vtm", "data/city-points.ply", 10)
plotter = pv.Plotter()
for idx, block in enumerate(street):
    plotter.add_mesh(block, color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)
plotter.show()
