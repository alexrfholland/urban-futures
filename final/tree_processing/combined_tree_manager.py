
    # ------------------------------------------------------------------------------------------------------------
    # Structure and Keys of `tree_resources_dict`:
    # ------------------------------------------------------------------------------------------------------------
    # The `tree_resources_dict` is a dictionary where:
    # - The **keys** are tuples containing four elements:
    #   1. **is_precolonial**: A boolean value (`True` or `False`) indicating whether the tree is precolonial.
    #   2. **size**: A string representing the size of the tree (`'small'`, `'medium'`, or `'large'`).
    #   3. **control**: A string representing the control category (`'street-tree'`, `'park-tree'`, or `'reserve-tree'`).
    #   4. **improvement**: A boolean value (`True` or `False`) indicating whether the improvement logic has been applied.
    # 
    # - The **values** are dictionaries where:
    #   - The keys are resource names (`'peeling bark'`, `'dead branch'`, `'fallen log'`, `'leaf litter'`, `'hollow'`, `'epiphyte'`).
    #   - The values are the computed resource counts for that specific tree configuration.
    # ------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------
    # Structure and Keys of 'data/revised/trees/'updated_tree_dict.pkl' (the old Eucalypt template dictionary)
    # ------------------------------------------------------------------------------------------------------------
 
    #The structure of the tuples in the dictionary follows this pattern: (precolonial, size, control, tree_id)
    #Where:
    # 1. **precolonial**: A boolean value (True/False) indicating if the tree is precolonial
    # 2. **size**: String - 'small', 'medium', 'large', 'senescing', 'snag', 'fallen', or 'propped'
    # 3. **control**: String - 'reserve-tree', 'park-tree', 'street-tree', or 'improved-tree'
    # 4. **tree_id**: Integer - unique identifier for the tree
    # ------------------------------------------------------------------------------------------------------------
    
    #Creation of updated_trees_dict.pkl
    #The file is created in final/tree_processing/euc_convertPickle.py. 
    # This script gets file revised_tree_dict.pkl, and does the following:
    #Filters out leaf litter
    #Renames coordinate columns to lowercase
    #Converts the single 'resource' column into multiple boolean columns through aggregate_data():type
    #Creates VTK visualization files for each tree:

    #Creation of revised_tree_dict.pkl.
    #Script f_temp_adjustTreeDict creates revised_tree_dict.pkl.
    #It combines and reformats trees from two source files:
    # 1. data/treeOutputs/adjusted_tree_templates.pkl
    # 2. data/treeOutputs/fallen_trees_dict.pkl
    #The script:
    # - Loads both pickle files
    # - Processes each tree key to match the new 4-tuple structure (precolonial, size, control, treeID) by:
    #   - Converting 5-tuple keys to 4-tuples by merging isImproved into the control field
    #   - Converting 3-tuple keys by adding 'improved-tree' as the control
    # - Saves the combined dictionary as data/revised/revised_tree_dict.pkl

    # For 5-tuple keys:
    #(isPreColonial, size, control, isImproved, treeID) -> (isPreColonial, size, control, treeID)
    # where if isImproved is True, control becomes 'improved-tree'

    # For snag/senescing/fallen keys:
    #(isPreColonial, treeID, size) -> (isPreColonial, size, 'improved-tree', treeID)

    #adjusted_tree_templates.pkl is created in modules/treeBake_recreateLogs.py.
    #fallen_trees_dict.pkl is created in modules/treeBake_treeAging.py
    # ------------------------------------------------------------------------------------------------------------



import combined_redoSnags
import combined_voxelise_dfs
import aa_tree_helper_functions
import pandas as pd
import pickle
import json
from pathlib import Path
import networkx as nx
import aa_io
import pyvista as pv
import combine_resource_treeMeshGenerator
def load_files():
    """Load all necessary files and prepare graph dictionary."""
    print('Loading templates and graphs')
    
    # Define paths
    template_dir = Path('data/revised/trees')
    elm_path = template_dir / 'elm_tree_dict.pkl'
    euc_path = template_dir / 'updated_tree_dict.pkl'
    graph_dir = Path('data/revised/lidar scans/elm/adtree/processedGraph')
    
    # File mapping dictionary
    point_cloud_files = {
        "Small A_skeleton.ply": 4,
        "Small B_skeleton.ply": 5,
        "Small C_skeleton.ply": 6,
        "Med A 1 mil_skeleton.ply": 1,
        "Med B 1 mil_skeleton.ply": 2,
        "Med C 1 mil_skeleton.ply": 3,
        "ElmL1_skeleton.ply": 7,
        "Elm L3_skeleton.ply": 9,
        "Elm L4_skeleton.ply": 10,
        "Elm L5_skeleton.ply": 11,
        "Large Elm A 1 mil_skeleton.ply": 12,
        "Large Elm B - 1 mil_skeleton.ply": 13,
        "Large Elm C 1 mil_skeleton.ply": 14
    }
    
    # Create filename dictionary
    filename_dict = {tree_id: filename.split('_')[0] for filename, tree_id in point_cloud_files.items()}
    
    # Load templates
    print(f'Loading elm templates from {elm_path}')
    elm_tree_templates = pd.read_pickle(elm_path)
    print(f'Loading euc templates from {euc_path}')
    euc_tree_templates = pickle.load(open(euc_path, 'rb'))
    
    # Create graph dictionary
    graph_dict = {}
    print(f'\nLooking for graphs in {graph_dir}')
    
    for tree_id in [7, 8, 9, 10, 11, 12, 13, 14]:
        if tree_id not in filename_dict:
            print(f'Warning: No filename mapping for tree {tree_id}')
            continue
            
        filename = filename_dict[tree_id]
        graph_path = graph_dir / f'{filename}_processedGraph.graphml'
        print(f'Loading graph from: {graph_path}')
        
        if graph_path.exists():
            graph = nx.read_graphml(graph_path)
            graph_dict[tree_id] = graph
            print(f'Loaded graph for tree {tree_id}')
        else:
            print(f'Warning: No graph found at {graph_path}')
    
    print(f'\nLoaded {len(graph_dict)} graphs')
    print(f'Graph dictionary keys: {list(graph_dict.keys())}')
    
    # Load resource DataFrame
    resourceDFPath = 'data/revised/trees/resource_dicDF.csv'
    resourceDF = pd.read_csv(resourceDFPath)
    
    return elm_tree_templates, euc_tree_templates, graph_dict, resourceDF

def update_template_files(eucalyptus_templates, elm_templates, updated_snags, regenerated_snags, use_original=False):
    """
    Combines eucalyptus and elm templates into a single DataFrame and adds snag templates.

    Args:
        eucalyptus_templates (dict): Dictionary with tuple keys and template DataFrame values
        elm_templates (pd.DataFrame): DataFrame with template DataFrames in 'template' column
        updated_snags (dict): Dictionary of updated snag templates
        regenerated_snags (dict): Dictionary of regenerated snag templates
        use_original (bool, optional): If True, use updated_snags; if False, use regenerated_snags. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - eucalyptus_templates (dict): The original eucalyptus templates dictionary
            - combined_templates (pd.DataFrame): Combined DataFrame with all templates including snags
    """
    print('\nUpdating template dictionaries with new snags')

    # Convert eucalyptus dictionary to DataFrame format
    eucalyptus_template_rows = []
    for template_key, template_data in eucalyptus_templates.items():
        precolonial, tree_size, tree_control, tree_id = template_key

        if precolonial:
            template_row = {
                'precolonial': precolonial,
                'size': tree_size,
                'control': tree_control,
                'tree_id': tree_id,
                'template': template_data
            }
            eucalyptus_template_rows.append(template_row)
    
    eucalyptus_dataframe = pd.DataFrame(eucalyptus_template_rows)
    
    # Combine eucalyptus and elm templates
    combined_templates = pd.concat([elm_templates, eucalyptus_dataframe], ignore_index=True)
    
    # Add snag templates
    selected_snag_templates = updated_snags if use_original else regenerated_snags
    snag_template_rows = []
    
    for snag_tree_id, snag_template_data in selected_snag_templates.items():
        print(f'Adding snag template for tree {snag_tree_id}')
        snag_row = {
            'precolonial': False,
            'size': 'snag',
            'control': 'improved-tree',
            'tree_id': snag_tree_id,
            'template': snag_template_data
        }
        snag_template_rows.append(snag_row)
    
    snag_dataframe = pd.DataFrame(snag_template_rows)
    combined_templates = pd.concat([combined_templates, snag_dataframe], ignore_index=True)
    
    print(f'Added {len(snag_template_rows)} snag templates to the combined templates')

    aa_tree_helper_functions.check_for_duplicates(combined_templates)
    return combined_templates

def save_snags_and_originalVTKS(regenerated_snags, tree_templates, output_folder):
    # Create paths
    tree_VTKpts_path = Path(output_folder) / 'tree_VTKpts'
    regenerated_snags_path = Path(output_folder) / 'regenerated_snags'
    snagVTKs_path = Path(output_folder) / 'snagVTKs'

    print(tree_templates['size'].value_counts())
    
    # Create directories if they don't exist
    tree_VTKpts_path.mkdir(parents=True, exist_ok=True)
    regenerated_snags_path.mkdir(parents=True, exist_ok=True)
    snagVTKs_path.mkdir(parents=True, exist_ok=True)

    # Make sure 'control' is a categorical value
    tree_templates['control'] = pd.Categorical(
        tree_templates['control'], 
        categories=['street-tree', 'park-tree', 'improved-tree', 'reserve-tree']
    )
    
    # Filter for large or snag trees
    filtered_templates = tree_templates[tree_templates['size'].isin(['large', 'snag'])].copy()
    
    # Group and sort within groups
    grouped = filtered_templates.groupby(['tree_id', 'precolonial', 'size']).apply(
        lambda x: x.sort_values('control')
    ).reset_index(drop=True)
    
    # Take first row of each group
    final_templates = grouped.groupby(['tree_id', 'precolonial', 'size']).first().reset_index()

    # Process each template
    for _, row in final_templates.iterrows():
        filename = f"precolonial.{row['precolonial']}_size.{row['size']}_control.{row['control']}_id.{row['tree_id']}.vtk"
        print(f"Generating output for tree {row['tree_id']} to {filename}")
        
        # Save original VTK
        polyName = Path(tree_VTKpts_path) / filename
        poly = aa_tree_helper_functions.convertToPoly(row['template'])
        poly = aa_tree_helper_functions.create_color_mapping(poly, 'cluster_id')
        poly.save(polyName)
        print(f"Saved tree mesh for tree {row['tree_id']} to {polyName}")

        # Process regenerated snags
        if row['size'] == 'snag' and not row['precolonial']:
            if row['tree_id'] not in regenerated_snags:
                print(f"No regenerated snag template for tree {row['tree_id']}")
                continue
            
            snag_row = row.copy()
            snag_row['template'] = regenerated_snags[row['tree_id']]
            snagPoly = aa_tree_helper_functions.convertToPoly(snag_row['template'])
            snagPoly = aa_tree_helper_functions.create_color_mapping(snagPoly, 'cluster_id')
            snagVTKFilepath = Path(snagVTKs_path) / filename
            snagPoly.save(snagVTKFilepath)

            print(f"Saved regenerated snag VTK for tree {row['tree_id']} to {snagVTKFilepath}")
           
            mesh = combine_resource_treeMeshGenerator.process_template_row(
                snag_row, 
                regenerated_snags_path
            )
        
            print(f"Saved snag mesh for tree {row['tree_id']} to {regenerated_snags_path}")

def get_template_stats(templates_df):
    """
    Create a statistics DataFrame from a templates DataFrame using vectorized operations.
    """
    # Create initial stats_df with metadata columns
    metadata_columns = ['precolonial', 'size', 'control', 'tree_id']
    stats_df = templates_df[metadata_columns].copy()
    
    # Explicitly specify resource columns
    resource_columns = [
        'resource_hollow',
        'resource_epiphyte',
        'resource_dead branch',
        'resource_perch branch',
        'resource_peeling bark',
        'resource_fallen log',
        'resource_other'
    ]
    
    print("\nTemplate counts by size and control:")
    print(pd.crosstab(templates_df['control'], templates_df['size']))
    
    # Calculate total voxels
    stats_df['total_voxels'] = templates_df['template'].apply(len)
    
    def sum_resource(row, resource_col):
        """Helper function to sum resource columns and print warnings"""
        template = row['template']
        
        # Case 1: Column doesn't exist
        if resource_col not in template.columns:
            print(f"Missing column {resource_col:20} - precolonial={row['precolonial']}, "
                  f"size={row['size']:8}, control={row['control']:12}, tree_id={row['tree_id']}")
            return 0
        
        # Case 2: Column exists but sum is 0
        resource_sum = template[resource_col].sum()
        if resource_sum == 0:
            print(f"Zero count for {resource_col:20} - precolonial={row['precolonial']}, "
                  f"size={row['size']:8}, control={row['control']:12}, tree_id={row['tree_id']}")
        
        return resource_sum
    
    print("\nChecking resources in templates...")
    # Calculate resource sums
    for resource_col in resource_columns:
        print(f"\n{'=' * 80}\nProcessing {resource_col}:")
        stats_df[resource_col] = templates_df.apply(
            lambda row: sum_resource(row, resource_col), 
            axis=1
        )
    
    return stats_df

def print_template_stats(stats_df):
    """
    Print matrix statistics for templates, split by precolonial status.
    Shows mean (min-max) for each control x size combination.
    """
    # Specify metrics we want to analyze
    metrics = [
        'total_voxels',
        'resource_hollow',
        'resource_epiphyte',
        'resource_dead branch',
        'resource_perch branch',
        'resource_peeling bark',
        'resource_fallen log',
        'resource_other'
    ]
    
    # Define ordered categories
    size_order = ['small', 'medium', 'large', 'senescing', 'snag', 'fallen']
    control_order = ['street-tree', 'park-tree', 'reserve-tree', 'improved-tree']

    def format_number(value):
        """Format number with thousand separators and round to nearest integer"""
        return f"{int(round(value)):,}"
    
    def format_cell(subset):
        """Format cell with mean (min-max) and thousand separators"""
        if not subset.empty:
            mean_val = subset.mean()
            min_val = subset.min()
            max_val = subset.max()
            return f"{format_number(mean_val)} ({format_number(min_val)}-{format_number(max_val)})"
        return "N/A"

    # For each metric
    for metric in metrics:
        print(f"\n{'=' * 80}")
        print(f"{metric.upper()}")
        print("=" * 80)
        
        # Print precolonial and non-precolonial matrices side by side
        for is_precolonial in [True, False]:
            df_subset = stats_df[stats_df['precolonial'] == is_precolonial]
            
            print(f"\n{'Precolonial' if is_precolonial else 'Non-precolonial'} trees:")
            print("-" * 80)
            
            # Get existing categories in specified order
            sizes = [size for size in size_order if size in df_subset['size'].unique()]
            controls = [control for control in control_order if control in df_subset['control'].unique()]
            
            # Create the matrix
            matrix_rows = []
            for control in controls:
                row_values = []
                for size in sizes:
                    subset = df_subset[
                        (df_subset['control'] == control) & 
                        (df_subset['size'] == size)
                    ][metric]
                    row_values.append(format_cell(subset))
                matrix_rows.append(row_values)
            
            # Convert to DataFrame for pretty printing
            matrix_df = pd.DataFrame(matrix_rows, index=controls, columns=sizes)
            print(matrix_df.to_string())
            print()

def analyze_all_resource_counts(templates_df):
    """
    Analyze counts for all resource types simultaneously.
    Shows -1 if resource does not exist in template.
    Exports results to CSV file.
    """
    # Define ordered categories
    size_order = ['small', 'medium', 'large', 'senescing', 'snag', 'fallen']
    control_order = ['street-tree', 'park-tree', 'reserve-tree', 'improved-tree']
    
    # Define resource columns
    resource_columns = [
        'resource_hollow',
        'resource_epiphyte',
        'resource_dead branch',
        'resource_perch branch',
        'resource_peeling bark',
        'resource_fallen log',
        'resource_other'
    ]
    
    # Create base DataFrame with metadata
    resource_counts = pd.DataFrame({
        'precolonial': templates_df['precolonial'],
        'size': templates_df['size'],
        'control': templates_df['control'],
        'tree_id': templates_df['tree_id'],
        'voxel_count': templates_df['template'].apply(len),  # Add voxel count
        'canopy_count': templates_df['template'].apply(
            lambda x: len(x[x['resource_fallen log'] == 0]) if 'resource_fallen log' in x.columns else len(x)
        )  # Add canopy count
    })
    
    # Add counts for each resource type
    for column in resource_columns:
        resource_counts[column] = templates_df['template'].apply(
            lambda x: x[column].sum() if column in x.columns else -1
        )
    
    # Sort categories
    resource_counts['size'] = pd.Categorical(resource_counts['size'], categories=size_order, ordered=True)
    resource_counts['control'] = pd.Categorical(resource_counts['control'], categories=control_order, ordered=True)
    
    # Sort the DataFrame
    result = resource_counts.sort_values(['precolonial', 'size', 'control', 'tree_id'])
    
    # Print results
    print("\nRESOURCE COUNTS AND PERCENTAGES")
    print("=" * 150)  # Widened separator for better readability
    
    for precolonial in [True, False]:
        print(f"\n{'Precolonial' if precolonial else 'Non-precolonial'} Trees:")
        print("-" * 150)
        
        subset = result[result['precolonial'] == precolonial]
        base_columns = ['size', 'control', 'tree_id', 'voxel_count', 'canopy_count']
        
        # Create a formatted DataFrame for display
        display_df = subset[base_columns].copy()
        
        # Add count and percentage columns for each resource
        for resource in resource_columns:
            # Add count column
            display_df[f"{resource}_count"] = subset[resource]
            
            # Add percentage column (avoiding division by zero)
            display_df[f"{resource}_pct"] = subset.apply(
                lambda row: (row[resource] / row['canopy_count'] * 100) 
                if row[resource] >= 0 and row['canopy_count'] > 0 
                else -1, 
                axis=1
            )
            
            # Format the columns
            display_df[f"{resource}"] = display_df.apply(
                lambda row: f"{int(row[f'{resource}_count']):,d} ({row[f'{resource}_pct']:.1f}%)"
                if row[f'{resource}_count'] >= 0
                else "N/A",
                axis=1
            )
        
        # Keep only the formatted columns for display
        display_columns = base_columns + resource_columns
        display_df = display_df[display_columns]
        
        # Print the formatted DataFrame
        print(display_df.to_string())

    # Export to CSV
    output_path = Path('data/revised/final/stats/arboreal-future-stats/data/template_resource_counts.csv')
    result.to_csv(output_path, index=False)
    print(f"\nResource counts exported to {output_path}")

    return result

def main():
    # Get user input for processing choice
    print("\nChoose processing option:")
    print("1) Generate templates from beginning") 
    print("2) Load existing voxelised templates and run statistics")
    choice = input("Enter choice (1 or 2): ")

    voxel_size = 1


    if choice == "1":
        #1. LOAD FILES
        print('loading files...')
        elm_templates, euc_templates, graph_dict, resourceDF = load_files()
        aa_tree_helper_functions.check_for_duplicates(elm_templates)
        
        #2. PROCESS SNAGS
        print('processing snags...')
        updated_snags, regenerated_snags = combined_redoSnags.process_snags(euc_templates, elm_templates, graph_dict, resourceDF)
        
        print(f'snag keys:')
        print(updated_snags.keys())
        
        #3. COMBINE TEMPLATES
        print('combining into single template file...')
        combined_templates = update_template_files(euc_templates, elm_templates, updated_snags, regenerated_snags, use_original=True)
        #remove propped size
        combined_templates = combined_templates[combined_templates['size'] != 'propped']
        
        #4. VOXELISE TEMPLATES
        print('\nVoxelizing templates...')
        voxelised_templates, adjustment_summary = combined_voxelise_dfs.process_trees(combined_templates, voxel_size, resetCount=True)

        #5. SAVE OUTPUTS
        print(f'saving')
        output_dir = Path('data/revised/trees') 
        output_dir.mkdir(parents=True, exist_ok=True)

        outputName = f'combined_templateDF.pkl'
        output_path = output_dir / outputName
        combined_templates.to_pickle(output_path)
        print(f'Combined templates dataframe saved at {output_path}')

        #save regenerated snags
        output_dir = Path('data/revised/trees')
        outputName = f'regenerated_snags.pkl'
        output_path = output_dir / outputName
        #regenerated snags is a dictionary, save it as a pickle
        with open(output_path, 'wb') as f:
            pickle.dump(regenerated_snags, f)
        print(f'Regenerated snags saved at {output_path}')

        voxelisedName = f'combined_voxelSize_{voxel_size}_templateDF.pkl'
        output_path = output_dir / voxelisedName
        voxelised_templates.to_pickle(output_path)
        print(f'Voxelized templates dataframe saved at {output_path}')

        summary_name = f'{voxel_size}_combined_voxel_adjustment_summary.csv'
        adjustment_summary.to_csv(output_dir / summary_name, index=False)
        print(f"\nAdjustment summary saved to {output_dir / summary_name}")
        
    else:
        # Load existing voxelised templates
        print('\nLoading existing templates...')
        output_dir = Path('data/revised/trees')
        combined_templates = pd.read_pickle(output_dir / f'combined_templateDF.pkl')
        voxelised_templates = pd.read_pickle(output_dir / f'combined_voxelSize_{voxel_size}_templateDF.pkl')
        with open(output_dir / f'regenerated_snags.pkl', 'rb') as f:
            regenerated_snags = pickle.load(f)
            # Convert keys to integers
            regenerated_snags = {int(k): v for k, v in regenerated_snags.items()}

    
    """templateDF = aa_tree_helper_functions.get_template(combined_templates, True, 'large', 'park-tree')['template']
    poly = aa_tree_helper_functions.convertToPoly(templateDF)
    plotter = pv.Plotter()
    plotter.add_mesh(poly, scalars='resource', render_points_as_spheres=True, cmap='Set1')
    plotter.enable_eye_dome_lighting()
    plotter.view_isometric
    plotter.show()"""
    
    ##VALIDATE RESOURCES
    """resourceDFPath = 'data/revised/trees/resource_dicDF.csv'
    resourceDF = pd.read_csv(resourceDFPath)

    # Merge templates with resource targets based on matching columns
    matching_templates = combined_templates.merge(
        resourceDF,
        on=['precolonial', 'size', 'control'],
        how='left',
        indicator=True
    )

    # Process only templates that have matching resource targets
    matched = matching_templates[matching_templates['_merge'] == 'both']
    unmatched = matching_templates[matching_templates['_merge'] == 'left_only']

    # Print summary
    print(f"\nFound {len(matched)} templates with matching resource targets")
    print(f"Found {len(unmatched)} templates without matching resource targets")

    # Validate each matched template
    for _, row in matched.iterrows():
        print(f"\nValidating template: precolonial={row['precolonial']}, size={row['size']}, control={row['control']}")
        aa_tree_helper_functions.validate_resource_quantities(
            row['template'],
            row['precolonial'],
            row['size'],
            row['control'],
            row['tree_id']
        )

    analyze_all_resource_counts(voxelised_templates)
    """
    #manual edits
    allTreeEdits = pd.read_pickle('data/revised/trees/combined_editsDF.pkl')
    justEditsDF = pd.DataFrame(columns=combined_templates.columns)  # Create empty DataFrame with same structure
    
    for _, row in allTreeEdits.iterrows():
        #get mask for row and apply this to combined_templates
        mask = ((combined_templates['precolonial'] == row['precolonial']) 
                & (combined_templates['size'] == row['size'])
                & (combined_templates['control'] == row['control'])
                & (combined_templates['tree_id'] == row['tree_id']))
        
        # Get and modify template
        template = combined_templates.loc[mask, 'template'].iloc[0]  # Get the template value
        editsDf = row['edits']

        choice = {'resource_hollow' : 'replace',
                'resource_epiphyte' : 'replace', 
                'resource_dead branch' : 'adjust',
                'resource_perch branch' : 'adjust',
                'resource_peeling bark' : 'adjust',
                'resource_fallen log' : 'adjust',
                'resource_other' : 'adjust' 
                }
        
        # Apply edits to template
        for resource, action in choice.items():
            if action == 'replace':
                template[resource] = 0
            print(f'{resource} edits:')
            print(editsDf[resource].value_counts())
            valid_edits = editsDf[editsDf[resource] != -1]
            
            if not valid_edits.empty:
                resource_map = valid_edits.set_index('cluster_id')[resource]
                template.loc[template['cluster_id'].isin(resource_map.index), resource] = \
                    template.loc[template['cluster_id'].isin(resource_map.index), 'cluster_id'].map(resource_map)
        template = template.drop(columns=[resource])
        template = aa_tree_helper_functions.create_resource_column(template)
        
        # Visualize if needed
        """poly = aa_tree_helper_functions.convertToPoly(template)
        plotter = pv.Plotter()
        plotter.add_mesh(poly, scalars='resource', render_points_as_spheres=True, cmap='Set1')
        plotter.enable_eye_dome_lighting()
        plotter.view_isometric
        plotter.show()
        """
        # Update combined_templates
        combined_templates = combined_templates.copy()  # Create a copy to avoid SettingWithCopyWarning
        mask_index = combined_templates[mask].index[0]  # Get the exact index
        combined_templates.at[mask_index, 'template'] = template



        # Verify the update worked
        print("After update:")
        print(combined_templates.loc[mask])
        
        # Create a copy of the row and update its template, then add to justEditsDF
        edited_row = combined_templates[mask].iloc[0].copy()  # Get single row as Series
        edited_row['template'] = template  # Update template
        justEditsDF = pd.concat([justEditsDF, pd.DataFrame([edited_row])], ignore_index=True)



    print(justEditsDF.head())
    # Save both DataFrames
    combined_templates.to_pickle('data/revised/trees/edited_combined_templateDF.pkl')
    justEditsDF.to_pickle('data/revised/trees/just_edits_templateDF.pkl')

    #save snags and original VTKS
    save_snags_and_originalVTKS(regenerated_snags, combined_templates, 'data/revised/final/stanislav')
    print('saved snags and original VTKS')


    # TODO: 
    # - peeling bark not completely accurate
    # - senescing for precolonial false does not recalibrate (ie.too many dead branches)
    # - logs in non-precolonial are small
    # - perch branches overassigned (maybe just do on dead branches)

    #TODO: implement 'quantity mode'
    # - do a resource check after voxelisation and assign the correct percentages



if __name__ == '__main__':
    main()