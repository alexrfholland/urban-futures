import trimesh
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



def create_cylinder_mesh_no_caps(start_point, end_point, radius, resolution=16):
    vector = end_point - start_point
    length = np.linalg.norm(vector)
    direction = vector / length
    
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=resolution, cap_ends=False)
    cylinder.apply_translation([0, 0, length / 2])
    rotation_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
    cylinder.apply_transform(rotation_matrix)
    cylinder.apply_translation(start_point)
    
    return cylinder

def create_mesh_from_csv_no_caps(csv_file):
    df = pd.read_csv(csv_file, delimiter=';')
    cylinders = []

    # Print column names to debug
    print("Columns in the CSV:", df.columns)
    
    # Print the number of rows in the DataFrame
    print(f"Number of rows in the CSV: {len(df)}")
    
    # Debugging loop iteration
    for index, row in df.iterrows():
        # Print the current row number
        if index % 100 == 0:  # Adjust this number as needed to reduce the amount of output
            print(f"Processing row {index}")

        start_point = np.array([row['startX'], row['startY'], row['startZ']])
        end_point = np.array([row['endX'], row['endY'], row['endZ']])
        radius = row['radius']
        
        # Print the start and end points and radius for debugging
        print(f"Start point: {start_point}, End point: {end_point}, Radius: {radius}")
        
        try:
            cylinder = create_cylinder_mesh_no_caps(start_point, end_point, radius)
        except Exception as e:
            print(f"Error creating cylinder for row {index}: {e}")
            continue  # Skip this row and continue with the next
        
        cylinders.append(cylinder)
    
    # Print the number of cylinders created
    print(f"Number of cylinders created: {len(cylinders)}")

    combined_mesh = trimesh.util.concatenate(cylinders)
    
    return combined_mesh, df  # Return both the mesh and the DataFrame


def voxelize_mesh(mesh, voxel_size=0.5):
    """
    Voxelize the mesh using trimesh.
    """
    voxels = mesh.voxelized(pitch=voxel_size)
    return voxels

def extract_voxel_centroids(voxelized_mesh):
    """
    Extract the centroids of the voxels in the VoxelGrid.
    """
    centroids = voxelized_mesh.points
    return centroids

def visualize_voxel_centroids(centroids):
    """
    Visualize the voxel centroids using PyVista.
    """
    # Create a PyVista point cloud from the centroids
    point_cloud = pv.PolyData(centroids)
    
    # Visualize the point cloud
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color='red', point_size=5, render_points_as_spheres=True)
    plotter.show()

def match_properties_to_voxels(centroids, original_df):
    # Strip whitespace from column names
    original_df.columns = original_df.columns.str.strip()
    
    original_points = original_df[['startX', 'startY', 'startZ']].values
    tree = KDTree(original_points)
    
    distances, indices = tree.query(centroids)
    
    voxel_properties = pd.DataFrame(centroids, columns=['centroid_x', 'centroid_y', 'centroid_z'])

    voxel_properties['resource'] = 'other'

    
    # Add columns conditionally based on availability
    for column in ['branch_ID', 'branch_order', 'segment_ID', 'parent_segment_ID',
                   'growth_volume', 'growth_length', 'ID', 'length_to_leave',
                   'inverse_branch_order', 'length_of_segment', 'branch_order_cum',
                   'radius', 'length']:
        if column in original_df.columns:
            voxel_properties[column] = original_df[column].values[indices]
        else:
            print(f"Warning: Column '{column}' not found in the original DataFrame.")
    
    return voxel_properties

def convert_df_to_pyvista_polydata(voxel_properties_df):
    points = voxel_properties_df[['centroid_x', 'centroid_y', 'centroid_z']].values
    polydata = pv.PolyData(points)
    
    for column in voxel_properties_df.columns:
        if column not in ['centroid_x', 'centroid_y', 'centroid_z']:
            polydata.point_data[column] = voxel_properties_df[column].values
    
    return polydata


def create_random_colormap(num_colors):
    """Create a colormap with a specified number of random colors."""
    colors = np.random.rand(num_colors, 3)  # Generate random colors
    cmap = ListedColormap(colors)
    return cmap

def visualize_point_cloud(polydata, scalar_field='parent_segment_ID'):
    # Extract the point data for the specified scalar field
    scalar_data = polydata.point_data[scalar_field]
    
    
    # Determine the number of unique values in the scalar field
    num_unique_values = len(np.unique(scalar_data))
    
    # Create a random colormap with the number of unique values
    cmap = create_random_colormap(num_unique_values)
    
    # Visualization with the custom random colormap
    plotter = pv.Plotter()
    plotter.add_mesh(
        polydata,
        render_points_as_spheres=True,
        point_size=10,
        scalars=scalar_field,
        cmap = 'tab20'
        #cmap=cmap,  # Applying the custom random colormap
    )
    plotter.show()


def visualize_linked_views(polydata, scalar_properties, max_columns=4):
    n_views = len(scalar_properties)
    n_columns = min(n_views, max_columns)
    n_rows = (n_views + n_columns - 1) // n_columns  # Calculate the number of rows needed
    
    plotter = pv.Plotter(shape=(n_rows, n_columns), border=False, lighting="light_kit")

    for i, scalar in enumerate(scalar_properties):
        row = i // n_columns
        col = i % n_columns
        plotter.subplot(row, col)
        
        # Make a copy of the PolyData to ensure independent handling of scalar data
        polydata_copy = polydata.copy(deep=True)
        
        plotter.add_mesh(
            polydata_copy,
            render_points_as_spheres=True,
            point_size=10,
            scalars=scalar,
            cmap="Set1",
            show_scalar_bar=True,
            scalar_bar_args={'title': scalar, 'n_labels': len(np.unique(polydata_copy[scalar]))}
        )
        plotter.add_text(scalar, font_size=10, color="white")

    plotter.link_views()  # Link the views
    plotter.show()


def visualize_voxelized_mesh_as_boxes(voxelized_mesh, plotter=None):
    """
    Visualize the voxelized mesh as boxes using PyVista.
    """
    # Convert the voxel grid to a mesh of boxes
    box_mesh = voxelized_mesh.as_boxes()
    
    # Wrap the box mesh with PyVista
    voxel_pv = pv.wrap(box_mesh)
    
    # Add the voxelized boxes to the plotter
    if plotter is None:
        plotter = pv.Plotter()
    plotter.add_mesh(voxel_pv, color='cyan', show_edges=True)
    return plotter


"""if __name__ == "__main__":
    index = 13
    csv_file_path = f'data/treeInputs/trunks/{index}.csv'
    
    mesh, original_df = create_mesh_from_csv_no_caps(csv_file_path)
    
    voxelized_mesh = voxelize_mesh(mesh, voxel_size=0.1)
    
    centroids = extract_voxel_centroids(voxelized_mesh)
    
    voxel_properties_df = match_properties_to_voxels(centroids, original_df)
    
    polydata = convert_df_to_pyvista_polydata(voxel_properties_df)
    
    scalar_properties = [
        'branch_ID', 'branch_order', 'segment_ID', 'parent_segment_ID',
        'growth_volume', 'growth_length', 'ID', 'length_to_leave',
        'inverse_branch_order', 'length_of_segment', 'branch_order_cum',
        'radius', 'length'
    ]
    
    visualize_point_cloud(polydata, scalar_field='dead branch')
    
    #visualize_linked_views(polydata, scalar_properties)
"""

if __name__ == "__main__2":
    index = 13
    csv_file_path = f'data/treeInputs/trunks/{index}.csv'
    
    mesh, original_df = create_mesh_from_csv_no_caps(csv_file_path)
    
    voxelized_mesh = voxelize_mesh(mesh, voxel_size=0.1)
    
    centroids = extract_voxel_centroids(voxelized_mesh)
    
    voxel_properties_df = match_properties_to_voxels(centroids, original_df)
    
    polydata = convert_df_to_pyvista_polydata(voxel_properties_df)
    
    visualize_point_cloud(polydata)

#CHATGPT TO DO
## Voxel_properties_df is a dataframe of coordinates of voxels describing branches in a tree. In this df, create a new column called 'resource' and initialise it with 'other'
## for a given resource, look up the resource_dict and obtain a percentage
## calculate how many voxels in voxel_properties_df need to change from 'other' to achieve this percentage
## write and call the clustering conversion function. We want the voxels to be converted in logical clusters (ie. a side of a tree), rather than random. To do so, this function should take a resource, quantity, patchiness level and the df as arguments. and convert 'other' voxels into the resource
    #using the segment_id column of the df, create buckets that group the the segment_id into patchiness_level groups. Calculate the number of voxels each group. Then, if all voxels in the group have 'other' as their resource column, keep converting groups so each rows 'other' resource is now the 'resource' argument. Keep count and check that you are not about to exceed the number of voxels to convert by converting an entire next group. For this last conversion, order this group by inverse_branch_order and convert the first n rows so the number of voxels is correctly converted
#make sure you print informative debug statements saying the resource that will convert, the percentage that should be converted for that resource, how many voxels to convert for that resource, each group converted, the final amount of voxels converted and the % of all voxels are now that resource. 
#make performant but keep to the standard numpy/pandas libraries
#do a test where we get an example dictionary that has 'dead branch : 30' as input dictionary and do a patch level of 20


import numpy as np
def convert_voxels_to_resource(voxel_properties_df, resource_dict, patchiness_level, exclude=None):
    # Calculate the total number of voxels
    total_voxels = len(voxel_properties_df)
    
    # Determine the range of segment IDs and create groups based on the patchiness level
    min_segment_id = voxel_properties_df['segment_ID'].min()
    max_segment_id = voxel_properties_df['segment_ID'].max()
    segment_id_range = max_segment_id - min_segment_id + 1
    
    # Create bins for the segment IDs based on the patchiness level
    bins = np.linspace(min_segment_id, max_segment_id + 1, patchiness_level + 1)
    voxel_properties_df['segment_group'] = np.digitize(voxel_properties_df['segment_ID'], bins) - 1
    
    for resource, percentage in resource_dict.items():
        quantity_to_convert = int(np.ceil((percentage / 100) * total_voxels))
        converted_voxels = 0
        
        # Group by the newly created segment_group
        group_clusters = voxel_properties_df.groupby('segment_group')
        
        # Shuffle the order of the groups randomly
        group_ids = np.random.permutation(group_clusters.size().index.tolist())
        
        for group_id in group_ids:
            group = group_clusters.get_group(group_id)
            group_size = len(group)
            
            # Check exclusion criteria if provided
            if exclude:
                excluded = False
                for col, conditions in exclude.items():
                    if col in group.columns and group[col].isin(conditions).any():
                        excluded = True
                        print(f"Skipping group {group_id} due to exclusion condition on column '{col}' with conditions {conditions}.")
                        break
                if excluded:
                    continue
            
            # Proceed with conversion if no exclusions apply
            if group['resource'].eq('other').all():
                if converted_voxels + group_size <= quantity_to_convert:
                    voxel_properties_df.loc[group.index, 'resource'] = resource
                    converted_voxels += group_size
                    print(f"Converted entire group with segment_group {group_id}: {group_size} voxels")
                else:
                    remaining_voxels = quantity_to_convert - converted_voxels
                    sorted_group = group.sort_values(by='inverse_branch_order', ascending=True)
                    voxel_properties_df.loc[sorted_group.index[:remaining_voxels], 'resource'] = resource
                    converted_voxels += remaining_voxels
                    print(f"Partially converted group with segment_group {group_id}: {remaining_voxels} voxels")
                    break

        final_percentage = (converted_voxels / total_voxels) * 100
        print(f"Total converted voxels for {resource}: {converted_voxels} ({final_percentage:.2f}%)")

    return voxel_properties_df

# Example usage with exclusion
if __name__ == "__main__":
    index = 11


    
    csv_file_path = f'data/treeInputs/trunks/orig/{index}.csv'
    output_template_path = 'data/treeInputs/trunks/initial_templates/'
    
    # Generate the initial mesh from the CSV file
    mesh, original_df = create_mesh_from_csv_no_caps(csv_file_path)
    
    # Voxelize the mesh
    voxelized_mesh = voxelize_mesh(mesh, voxel_size=0.1)

    visualize_voxelized_mesh_as_boxes(voxelized_mesh)
    
    # Extract voxel centroids
    centroids = extract_voxel_centroids(voxelized_mesh)

    
    # Match properties to voxels
    voxel_properties_df = match_properties_to_voxels(centroids, original_df)
    
    # Define the resource dictionary and exclusion criteria
    resource_dict = {'dead branch': 30}
    exclude = {'branch_order': [0]}  # Example exclusion
    
    # Perform the conversion with a patchiness level of 20 and exclusion criteria
    voxel_properties_df = convert_voxels_to_resource(voxel_properties_df, resource_dict, patchiness_level=20, exclude=exclude)
    
    # Convert to PyVista PolyData
    polydata = convert_df_to_pyvista_polydata(voxel_properties_df)
    
    # Visualize the result
    visualize_point_cloud(polydata, 'resource')


"""
if __name__ == "__main2__":
    index = 13
    csv_file_path = f'data/treeInputs/trunks/{index}.csv'
    
    # Generate the initial mesh from the CSV file
    mesh = create_mesh_from_csv_no_caps(csv_file_path)
    
    # Voxelize the mesh
    voxelized_mesh = voxelize_mesh(mesh, voxel_size=0.1)

    # Create a plotter and visualize the voxelized mesh as boxes
    plotter = pv.Plotter()
    plotter = visualize_voxelized_mesh_as_boxes(voxelized_mesh, plotter=plotter)
    plotter.show()

    # Extract voxel centroids
   #centroids = extract_voxel_centroids(voxelized_mesh)
    
    # Visualize the voxel centroids
    #visualize_voxel_centroids(centroids)
"""