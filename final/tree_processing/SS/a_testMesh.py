import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyvista as pv
import f_vtk_to_ply_surfaces

path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/treeMeshes/resolution_original/True_large_park-tree_True_15.vtk"
poly = pv.read(path)

polyExportPath = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/trees/test.ply"
f_vtk_to_ply_surfaces.export_polydata_to_ply(poly, polyExportPath)
print(f'saved polydata to {polyExportPath}')