import pyvista as pv
import numpy as np
import pandas as pd
from pathlib import Path



def create_resource_column(voxelDF):
    # Define resource priorities (higher number = higher priority)
    resource_priorities = {
        'other': 0,
        'dead branch': 1, 
        'peeling bark': 2,
        'perch branch': 3,
        'epiphyte': 4,
        'fallen log': 5,
        'hollow': 6
    }
    
    # Initialize the 'resource' column with 'other'
    voxelDF['resource'] = 'other'
    
    # Get resource columns and map them to resource names
    resource_cols = [col for col in voxelDF.columns if col.startswith('resource_') and col != 'resource']
    col_to_resource = {col: col.split('resource_')[1] for col in resource_cols}
    
    # Verify the mappings before processing
    print("\nColumn to Resource Mapping:")
    print(col_to_resource)
    
    # Process columns in order of descending priority
    for col in sorted(resource_cols, key=lambda col: -resource_priorities[col_to_resource[col]]):
        resource_name = col_to_resource[col]
        # Assign the resource where the column is 1 and the current resource is still 'other'
        mask = (voxelDF[col] == 1) & (voxelDF['resource'] == 'other')
        voxelDF.loc[mask, 'resource'] = resource_name
        print(f"\nResource '{resource_name}' assigned to {mask.sum()} rows.")
    
    # Verify the result for specific critical resources
    for resource_name, col in col_to_resource.items():
        if resource_name in ['epiphyte', 'hollow']:
            mismatches = voxelDF[(voxelDF[col] == 1) & (voxelDF['resource'] != resource_name)]
            print(f"\nVerification for '{resource_name}':")
            print(f"Rows with {col}=1 but not assigned '{resource_name}': {len(mismatches)}")
    
    # Create column 'resource_other' and set all values to 1 if resource is 'other' else 0
    voxelDF['resource_other'] = (voxelDF['resource'] == 'other').astype(int)
    
    return voxelDF




def check_for_duplicates(df):
       # Check for duplicates
        duplicate_check = df.groupby(
            ['precolonial', 'size', 'control', 'tree_id']
        ).size()
        duplicates = duplicate_check[duplicate_check > 1]
        
        if not duplicates.empty:
            error_msg = "Found duplicate combinations:\n"
            error_msg += duplicates.to_string()
            raise ValueError(error_msg)


def resource_names():
    return [
        'resource_hollow',
        'resource_epiphyte', 
        'resource_dead branch',
        'resource_perch branch',
        'resource_peeling bark',
        'resource_fallen log',
        'resource_other'
    ]

def resource_names_no_columns():
    return [
        'other',
        'dead branch', 
        'peeling bark',
        'perch branch',
        'epiphyte',
        'fallen log',
        'hollow'
    ]

def validate_resource_quantities(template, precolonial, size, control, tree_id=-1):
  
    print(template.head())
    resourceDFPath = 'data/revised/trees/resource_dicDF.csv'
    resourceDF = pd.read_csv(resourceDFPath)

    resourceForTemplate = resourceDF[
        (resourceDF['precolonial'] == precolonial) &
        (resourceDF['size'] == size) &
        (resourceDF['control'] == control)
    ]

    for resource in resource_names_no_columns():
        if resource in ['other', 'peeling bark', 'perch branch', 'hollow', 'epiphyte', 'fallen log']:
            continue

        if size in ['senescing']:
            continue

        if control in ['improved-tree']:
            continue

        columnName = f'resource_{resource}'  # For template DataFrame
        
        #get canopy template that is templte where resource_fallen log = 0
        template = template[template['resource_fallen log'] == 0]

        total_branches = len(template)

        # Final count and statistics
        final_converted = template[columnName].sum()
        actual_percentage = (final_converted / total_branches) * 100
        target_percentage = resourceForTemplate[resource].iloc[0]  # Use resource without prefix for resourceDF
        percentage_difference = abs(actual_percentage - target_percentage)

        print(f"\nFinal results:")
        print(f"Final voxels converted to {resource}: {final_converted}")
        print(f"{resource} target was: {int(target_percentage)}")
        print(f"{resource} actual percentage: {actual_percentage:.2f}%")
        print(f"{resource} Target percentage: {target_percentage:.2f}%")
        print(f"Percentage difference: {percentage_difference:.2f}%")

        # Print distribution of True/False values from template
        value_counts = template[columnName].value_counts()  # Use columnName (with prefix) for template
        print(f"\nDistribution of {columnName}:")
        print(value_counts)

        # Validate final percentage is within 5% of target
        if percentage_difference > 5:
            print(f"ERROR with precolonial={precolonial}, size={size}, control={control}, tree_id={tree_id}")
            print(f"{resource} final percentage ({actual_percentage:.2f}%) differs from target "
                f"({target_percentage:.2f}%) by more than 5% "
                f"({resource} difference: {percentage_difference:.2f}%)")
            
            poly = convertToPoly(template)
            poly.plot(scalars=columnName)

            raise ValueError(
                f"ERROR with precolonial={precolonial}, size={size}, control={control}, tree_id={tree_id}"
                f"{resource} final percentage ({actual_percentage:.2f}%) differs from target "
                f"({target_percentage:.2f}%) by more than 5% "
                f"({resource} difference: {percentage_difference:.2f}%)"
            )
        else:
            print(f"SUCCESS with precolonial={precolonial}, size={size}, control={control}, tree_id={tree_id}")
            print(f"{resource} final percentage: {actual_percentage:.2f}% is within 5% of target")
            print(f"{resource} target was: {target_percentage:.2f}%")
            print(f"{resource} difference: {percentage_difference:.2f}%")
        

def verify_resources_columns(template):
    for resource in resource_names():
        if resource not in template.columns:
            template[resource] = 0
    return template


def convertToPoly(voxelDF):

    points = voxelDF[['x', 'y', 'z']].values
    poly = pv.PolyData(points)

    # Add all columns as point data attributes
    for col in voxelDF.columns:
        if col not in ['x', 'y', 'z']:  # Skip coordinate columns
            poly.point_data[col] = voxelDF[col].values

    return poly


def create_color_mapping(poly, attribute_name):
    # Ensure the attribute exists in point or cell data
    if attribute_name not in poly.point_data and attribute_name not in poly.cell_data:
        raise ValueError(f"Attribute '{attribute_name}' not found in point or cell data.")

    # Decide whether the attribute is point or cell-based
    if attribute_name in poly.point_data:
        data_array = poly.point_data[attribute_name]
    else:
        data_array = poly.cell_data[attribute_name]

    # Handle both string and numerical data
    if isinstance(data_array[0], (str, np.str_)):
        # For string data, create numerical mapping
        unique_values = np.unique(data_array)
        value_to_index = {value: i for i, value in enumerate(unique_values)}
        numerical_data = np.array([value_to_index[val] for val in data_array])
        n_clusters = len(unique_values)
    else:
        # For numerical data, use as is
        numerical_data = data_array
        n_clusters = int(np.max(numerical_data) + 1)

    # Generate random colors
    cluster_colors = np.random.rand(n_clusters, 3)  # Random RGB colors

    # Map colors based on the numerical indices
    color_array = cluster_colors[numerical_data]

    # Assign the colors to the appropriate data set
    if attribute_name in poly.point_data:
        poly.point_data['colors'] = color_array
    else:
        poly.cell_data['colors'] = color_array

    print(f'poly has {poly.n_points} points')
    print(f'color array has shape: {color_array.shape}')

    return poly

def get_template(templates, precolonial, size, control, offset=0):
    # Add debug prints
    print(f"\nSearching for:")
    print(f"Precolonial: {precolonial}")
    print(f"Size: {size}")
    print(f"Control: {control}")
    
    # Print unique values in templates
    print("\nAvailable combinations in data:")
    print("Precolonial values:", templates['precolonial'].unique())
    print("Size values:", templates['size'].unique())
    print("Control values:", templates['control'].unique())
    
    mask = (templates['precolonial'] == precolonial) & \
           (templates['size'] == size) & \
           (templates['control'] == control)
    
    matches = templates[mask]
    print(f"\nFound {len(matches)} matching templates")
    print(matches)
    
    if len(matches) == 0:
        # Print examples of close matches to help debug
        print("\nClose matches:")
        print(templates[templates['precolonial'] == precolonial].head())
    
    if offset >= len(matches):
        return matches.iloc[0] if not matches.empty else None
    if offset < 0:
        return matches
    return matches.iloc[offset]

def interactive_template_viewer():
    """
    Interactive function to visualize tree templates with user-selected parameters.
    """
    # Load templates once
    output_dir = Path('data/revised/trees')
    combined_templates = pd.read_pickle(output_dir / 'combined_templateDF.pkl')
    
    def get_user_selections():
        # Get user input for template parameters
        print("\nSelect template parameters:")
        
        # Precolonial selection
        print("\nPrecolonial options:")
        print("(1) True")
        print("(2) False")
        precolonial_choice = int(input("Select option (1-2): "))
        precolonial = precolonial_choice == 1
        
        # Size selection
        size_options = ['small', 'medium', 'large', 'senescing', 'snag', 'fallen']
        print("\nSize options:")
        for i, size in enumerate(size_options, 1):
            print(f"({i}) {size}")
        size_choice = int(input(f"Select option (1-{len(size_options)}): "))
        size = size_options[size_choice - 1]
        
        # Control selection
        control_options = ['street-tree', 'park-tree', 'reserve-tree', 'improved-tree']
        print("\nControl options:")
        for i, control in enumerate(control_options, 1):
            print(f"({i}) {control}")
        control_choice = int(input(f"Select option (1-{len(control_options)}): "))
        control = control_options[control_choice - 1]
        
        return precolonial, size, control

    def select_and_show_template(precolonial, size, control):
        # Get matching templates
        matches = get_template(combined_templates, precolonial, size, control, offset=-1)
        
        # Display available templates
        print("\nAvailable templates:")
        for i, (idx, row) in enumerate(matches.iterrows(), 1):
            print(f"{i}: precolonial: {row['precolonial']}, size: {row['size']}, "
                  f"control: {row['control']}, tree_id: {row['tree_id']}")
        
        # Get user selection for template
        template_choice = int(input("\nSelect template number: ")) - 1
        return matches.iloc[template_choice]['template']

    def select_scalar_field(poly):
        # Display available attributes
        print("\nAvailable attributes:")
        attributes = list(poly.point_data.keys())
        for i, attr in enumerate(attributes, 1):
            print(f"{i}: {attr}")
        
        # Get user selection for attribute
        attr_choice = int(input("\nSelect attribute number to visualize: ")) - 1
        return attributes[attr_choice]

    def get_colormap():
        default_cmap = 'viridis'
        cmap = input(f"\nEnter colormap name (press Enter for default '{default_cmap}'): ")
        return cmap if cmap else default_cmap

    def visualize_template(template, scalar_field=None, cmap=None):
        poly = convertToPoly(template)
        if scalar_field is None:
            scalar_field = select_scalar_field(poly)
        if cmap is None:
            cmap = get_colormap()
            
        plotter = pv.Plotter()
        
        def new_scalar():  # Removed flag parameter
            plotter.close()
            visualize_template(template, select_scalar_field(poly), cmap)
            
        def new_tree():    # Removed flag parameter
            plotter.close()
            precolonial, size, control = get_user_selections()
            new_template = select_and_show_template(precolonial, size, control)
            visualize_template(new_template)
            
        plotter.add_key_event('n', new_scalar)  # 'n' for new scalar
        plotter.add_key_event('m', new_tree)    # 'm' for new tree
        
        print("\nKeyboard shortcuts:")
        print("'n': Select new scalar field")
        print("'m': Select new tree")
        print("'q': Quit")
        
        plotter.add_mesh(poly, scalars=scalar_field, render_points_as_spheres=True, cmap=cmap)
        plotter.enable_eye_dome_lighting()
        plotter.view_isometric()
        plotter.show()

    # Initial run
    precolonial, size, control = get_user_selections()
    template = select_and_show_template(precolonial, size, control)
    visualize_template(template)

if __name__ == "__main__":
    interactive_template_viewer()