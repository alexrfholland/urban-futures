import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path

def load_vtk_file(site, voxel_size, file_type, timestep=None, scenario=None):
    """Load VTK file for a specific timestep or baseline"""
    if file_type == 'baseline':
        baseline_path = f'data/revised/final/baselines/{site}_baseline_combined_{voxel_size}_urban_features.vtk'
        return pv.read(baseline_path)    
    elif file_type == 'timestep':
        path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{timestep}_urban_features.vtk'
        return pv.read(path)
    
def vtk_to_dataframe(vtk_data, file_type, scenario=None, timestep=None):
    """Convert VTK point data to a pandas DataFrame"""
    # Create a label for the file type
    if file_type == 'baseline':
        file_label = 'baseline'
    else:
        file_label = f"{scenario}_{timestep}"
    
    # Find the grouping variable (tree ID)
    id_field = None
    for field in ['forest_tree_number']:
        if field in vtk_data.point_data:
            id_field = field
            break
    
    if id_field is None:
        print(f"No tree ID field found in {file_label} VTK data")
        return pd.DataFrame()
    
    # Create a dictionary to store all point data
    data_dict = {'file': [file_label] * vtk_data.n_points}
    
    # Add the tree ID field
    data_dict['tree_id'] = vtk_data.point_data[id_field]
    
    # Add precolonial, size, and control fields if available
    for field_name, alt_fields in [
        ('precolonial', ['forest_precolonial', 'scenario_precolonial']),
        ('size', ['forest_size', 'scenario_size']),
        ('control', ['forest_control', 'scenario_control'])
    ]:
        # Try the primary field name
        if field_name in vtk_data.point_data:
            data_dict[field_name] = vtk_data.point_data[field_name]
        else:
            # Try alternative field names
            field_found = False
            for alt_field in alt_fields:
                if alt_field in vtk_data.point_data:
                    data_dict[field_name] = vtk_data.point_data[alt_field]
                    field_found = True
                    break
            
            # If no field found, use 'unknown'
            if not field_found:
                data_dict[field_name] = ['unknown'] * vtk_data.n_points
    
    # Add all stat_ fields
    for key in vtk_data.point_data.keys():
        if key.startswith('stat_'):
            data = vtk_data.point_data[key]
            
            if np.issubdtype(data.dtype, np.number):
                # For numeric data, use as is
                data_dict[key] = data > 0  # Convert to boolean for counting non-zero values
            else:
                # For string data, convert to boolean for counting non-'none' values
                data_dict[key] = (data != 'none') & (data != '') & (data != 'nan')
    
    # Add all resource_ fields
    for key in vtk_data.point_data.keys():
        if key.startswith('resource_'):
            data = vtk_data.point_data[key]
            data_dict[key] = data
    
    # Convert to DataFrame
    df = pd.DataFrame(data_dict)
    
    # Filter out rows with 'none' or empty tree_id
    df = df[~df['tree_id'].isin(['none', ''])]
    
    return df

def aggregate_by_tree_id(df):
    """Group DataFrame by tree_id and aggregate data"""
    if df.empty:
        return df
    
    # Define aggregation functions
    agg_funcs = {
        'file': 'first',
        'precolonial': lambda x: x.value_counts().index[0],
        'size': lambda x: x.value_counts().index[0],
        'control': lambda x: x.value_counts().index[0]
    }
    
    # Add sum aggregation for all stat_ columns
    for col in df.columns:
        if col.startswith('stat_'):
            agg_funcs[col] = 'sum'
        # Add appropriate aggregation for resource_ columns
        elif col.startswith('resource_'):
            if df[col].dtype == np.dtype('O'):  # Object/string type
                agg_funcs[col] = lambda x: x.value_counts().index[0] if not x.empty else None
            else:  # Numeric type
                agg_funcs[col] = 'sum'  # Use sum for numeric resource values since they are counts
    
    # Group by tree_id and aggregate
    grouped_df = df.groupby('tree_id').agg(agg_funcs).reset_index()
    
    return grouped_df

def compare_stats_by_tree(site, voxel_size, timestep):
    """Compare stat counts grouped by tree ID between baseline and scenarios"""
    # Load baseline VTK
    baseline_vtk = load_vtk_file(site, voxel_size, 'baseline')
    if baseline_vtk is None:
        print("Could not load baseline VTK")
        return
    
    # Convert baseline VTK to DataFrame and aggregate by tree_id
    baseline_df = vtk_to_dataframe(baseline_vtk, 'baseline')
    baseline_grouped = aggregate_by_tree_id(baseline_df)
    
    # Define scenarios to process
    scenarios = ['trending', 'positive']
    scenario_grouped_dfs = []
    
    # Process each scenario
    for scenario in scenarios:
        scenario_vtk = load_vtk_file(site, voxel_size, 'timestep', timestep, scenario)
        if scenario_vtk is not None:
            # Convert scenario VTK to DataFrame and aggregate by tree_id
            scenario_df = vtk_to_dataframe(scenario_vtk, 'timestep', scenario, timestep)
            scenario_grouped = aggregate_by_tree_id(scenario_df)
            scenario_grouped_dfs.append(scenario_grouped)
    
    # Combine all grouped DataFrames
    if not scenario_grouped_dfs:
        print("No scenario data could be loaded")
        return baseline_grouped
    
    combined_df = pd.concat([baseline_grouped] + scenario_grouped_dfs, ignore_index=True)
    
    # Display the DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    
    print(f"\nTree ID-based comparison for site '{site}', timestep {timestep}:")
    print(f"Showing data for Baseline, Trending {timestep}, and Positive {timestep}")
    print("-" * 80)
    print(combined_df)
    
    # Save to CSV
    output_dir = Path('data/revised/final/stats')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{site}_tree_stats_baseline_vs_scenarios_year{timestep}.csv'
    combined_df.to_csv(output_path, index=False)
    
    return combined_df

def main():
    """Main function to get user inputs and run comparison"""
    # Default values
    default_site = 'trimmed-parade'
    default_voxel_size = 1
    default_timestep = 60
    
    # Ask for site
    site_input = input(f"Enter site (default '{default_site}'): ")
    site = site_input.strip() if site_input else default_site
    
    # Ask for voxel size
    voxel_size_input = input(f"Enter voxel size (default {default_voxel_size}): ")
    voxel_size = int(voxel_size_input) if voxel_size_input else default_voxel_size
    
    # Ask for timestep
    timestep_input = input(f"Enter timestep (default {default_timestep}): ")
    timestep = int(timestep_input) if timestep_input else default_timestep
    
    # Run the comparison
    compare_stats_by_tree(site, voxel_size, timestep)

if __name__ == "__main__":
    main() 