import pickle
import pyvista as pv
import cameraSetUpRevised
import glyphs

def create_multiblock_from_df(df):
    print(f'creating multiblock')

    def create_polydata(df):
        points = df[['x', 'y', 'z']].values
        polydata = pv.PolyData(points)

        for col in df.columns.difference(['x', 'y', 'z']):
            sanitized_data = []
            for val in df[col]:
                if isinstance(val, str):
                    sanitized_val = val.encode('ascii', 'ignore').decode()
                    sanitized_data.append(sanitized_val)
                else:
                    sanitized_data.append(val)
            polydata.point_data[col] = sanitized_data

        polydata.point_data['ScanZ'] = df['z']
        return polydata

    branches_df = df[df['resource'].isin(['perchable branch', 'peeling bark', 'dead branch','other'])]
    ground_resources_df = df[df['resource'].isin(['fallen log', 'leaf litter'])]
    print(f'ground resource df is {ground_resources_df}')
    canopy_resources_df = df[df['resource'].isin(['epiphyte', 'hollow'])]
    leaf_cluster_df = df[df['resource'].isin(['leaf cluster'])]

    print('creating branches polydata')
    branches_polydata = create_polydata(branches_df)

    print('creating ground polydata')
    ground_resources_polydata = create_polydata(ground_resources_df)

    print('creating canopy polydata')
    canopy_resources_polydata = create_polydata(canopy_resources_df)

    print('creating leaf clusters')
    leaf_clusters_polydata = create_polydata(leaf_cluster_df)
    print(f'leaf cluster has {leaf_clusters_polydata.n_points} points')

    multiblock = pv.MultiBlock()
    multiblock['branches'] = branches_polydata
    multiblock['ground resources'] = ground_resources_polydata
    multiblock['canopy resources'] = canopy_resources_polydata
    multiblock['green biovolume'] = leaf_clusters_polydata

    return multiblock
def plot_example_trees(tree_samples_with_voxels, tree_size, precolonial, tree_id):
    controls = ['reserve-tree', 'park-tree', 'street-tree']
    plotter = pv.Plotter(shape=(1, 3))

    plotter.enable_eye_dome_lighting()

    for i, control in enumerate(controls):
        key = (tree_size, precolonial, control, tree_id)
        if key not in tree_samples_with_voxels:
            print(f"No tree found with key: {key}")
            continue

        df = tree_samples_with_voxels[key]
        
        # Create multiblock
        multiblock = create_multiblock_from_df(df)

        # Set the current subplot
        plotter.subplot(0, i)


        
        # Add trees to the current subplot
        glyphs.add_trees_to_plotter(plotter, multiblock, (0, 0, 0))

        # Set up the camera for all subplots
        cameraSetUpRevised.setup_camera(plotter, 50, 600)
        
        # Optionally, add a title to the subplot
        plotter.add_title(f"{control}")





    # Link views and show the plot
    plotter.link_views()
    plotter.show()

if __name__ == "__main__":
    # Load the pickle file
    with open('data/treeSim.pkl', 'rb') as f:
        tree_sim_data = pickle.load(f)

    tree_voxel_templates = tree_sim_data['tree_voxel_templates']

    # Plot example trees
    plot_example_trees(tree_voxel_templates, 'large', False, 13)
    # Uncomment the following lines to plot additional wsk
    # plot_example_trees(tree_voxel_templates, 'large', False, 11)
    # plot_example_trees(tree_voxel_templates, 'small', False, 1)