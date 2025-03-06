import pyvista as pv
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def flatten_multiblock_to_polydata(multi_blocks):

    all_points = []
    for multi in multi_blocks:
       points = np.vstack([block.points for block in multi])
       all_points.append(points)
        
    return np.vstack(all_points)


def mask_and_filter_multi_block(target_multi_block, multi_reference_block, distance_value, nearerKeep=True, ignoreZ = False):
    
    reference_block = flatten_multiblock_to_polydata(multi_reference_block)
    new_multi_block = pv.MultiBlock()
    
    for idx, target_block in enumerate(target_multi_block):        
        if ignoreZ:
            kd_tree = cKDTree(reference_block[:, :2])
            distances, _ = kd_tree.query(target_block.points[:, :2])
        else:
            kd_tree = cKDTree(reference_block)
            distances, _ = kd_tree.query(target_block.points)

        if nearerKeep:
            mask = distances < distance_value
        else:
            mask = distances >= distance_value


        points_df = pd.DataFrame(target_block.points, columns=['x', 'y', 'z'])
        attributes_df = pd.DataFrame()

        for name in target_block.point_data.keys():
            attr_data = target_block.point_data[name]
            if len(attr_data.shape) == 1:
                attributes_df[name] = attr_data
            else:
                for i in range(attr_data.shape[1]):
                    attributes_df[f"{name}_{i}"] = attr_data[:, i]

        full_df = pd.concat([points_df, attributes_df], axis=1)
        filtered_df = full_df[mask]
        new_points = filtered_df[['x', 'y', 'z']].to_numpy()
        new_block = pv.PolyData(new_points)

        for name in filtered_df.columns:
            if name not in ['x', 'y', 'z']:
                if '_' in name:
                    original_name, index = name.split('_')
                    if original_name not in new_block.point_data:
                        original_data_shape = target_block.point_data[original_name].shape[1]
                        new_block.point_data[original_name] = np.zeros((new_block.n_points, original_data_shape))
                    new_block.point_data[original_name][:, int(index)] = filtered_df[name].values
                else:
                    new_block.point_data[name] = filtered_df[name].values
        new_multi_block.append(new_block)

    return new_multi_block

if __name__ == "__main__":
    import getColorsAndAttributes as getColorsAndAttributes

    site = 'city'

    #np.random.seed(0)
    #points = np.random.rand(1000, 3) * 100
    #reference_block = pv.PolyData(points)
    topo = getColorsAndAttributes.transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("topography", 10, site)
    street = getColorsAndAttributes.transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("street-furniture", 10, site)
    buildings = getColorsAndAttributes.transfer_and_filter_attributes_from_raw_pointcloud_to_vtm_and_visualize("buildings", 10, site)


    
    stampFilesPaths = []

    reference_block = [street, buildings]

    new_topo = mask_and_filter_multi_block(topo, reference_block, 3, nearerKeep=False, ignoreZ=True)

    
    plotter = pv.Plotter()
    for idx, block in enumerate(new_topo):
        plotter.add_mesh(block, color=None, rgb=True, point_size=5.0, render_points_as_spheres=True)
    plotter.show()


