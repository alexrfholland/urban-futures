import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
from scipy.spatial import cKDTree

def get_capability_id(layer_name, capabilities_info):
    """Get capability ID from layer name using the capabilities info dataframe."""
    if layer_name in capabilities_info['layer_name'].values:
        return capabilities_info[capabilities_info['layer_name'] == layer_name]['capability_id'].iloc[0]
    return 'NA'  # Return 'NA' if not found

def collect_capability_stats(polydata, capabilities_info, site, scenario, timestep, voxel_size):
    """
    Collect statistics on capabilities by copying capabilities_info and adding counts
    
    Args:
        polydata: The polydata with capability information
        capabilities_info: DataFrame with capability mapping information
        site: Site name
        scenario: Scenario name
        timestep: Current timestep/year
        voxel_size: Voxel size used
        
    Returns:
        DataFrame with capabilities info and counts
    """
    # Make a copy of the capabilities_info DataFrame
    stats_df = capabilities_info.copy()
    
    # Add context columns
    stats_df['site'] = site
    stats_df['scenario'] = scenario
    stats_df['timestep'] = str(timestep)
    stats_df['voxel_size'] = voxel_size
    
    # Initialize count column
    stats_df['count'] = 0
    
    # Fill in counts for each capability
    for idx, row in stats_df.iterrows():
        layer_name = row['layer_name']        
        data = polydata.point_data[layer_name]
        count = np.sum(data)
        stats_df.at[idx, 'count'] = count
    
    return stats_df

def converted_urban_element_counts(site, scenario, year, polydata, tree_df=None, capabilities_info=None):
    """Generate counts of converted urban elements for different capabilities
    
    Args:
        site (str): Site name
        scenario (str): Scenario name
        year (int): Year/timestep
        polydata (pyvista.UnstructuredGrid): polydata with capability information
        tree_df (pd.DataFrame, optional): Tree dataframe for df-based counts
        capabilities_info (pd.DataFrame): DataFrame with capability mapping information
        
    Returns:
        pd.DataFrame: DataFrame containing all counts with columns
    """
    # Initialize empty list to store count records
    count_records = []
    
    # Convert year to string for consistency
    year_str = str(year)
    
    # Define common urban element categories used throughout the function
    URBAN_ELEMENT_TYPES = [
        'open space', 'green roof', 'brown roof', 'facade', 
        'roadway', 'busy roadway', 'existing conversion', 
        'other street potential', 'parking'
    ]
    
    # Helper function to create count records from capabilities_info row
    def create_count_record(base_row, countname, countelement, count):
        # Create a copy of the base row and update relevant fields
        record = base_row.copy()
        record['site'] = site
        record['scenario'] = scenario
        record['timestep'] = year_str
        record['countname'] = countname
        record['countelement'] = countelement.replace(' ', '_')
        record['count'] = int(count)
        
        return record
    
    # Helper function to handle boolean or string data fields
    def get_boolean_mask(data_field, condition=None):
        if np.issubdtype(data_field.dtype, np.bool_):
            if condition is None:
                return data_field
            else:
                return data_field & condition
        else:
            if condition is None:
                return (data_field != 'none') & (data_field != '')
            else:
                return ((data_field != 'none') & (data_field != '')) & condition
    
    # Helper function to process counts by urban element types
    def count_by_urban_elements(mask_data, urban_data, base_row, countname):
        element_records = []
        for element_type in URBAN_ELEMENT_TYPES:
            element_mask = urban_data == element_type
            combined_mask = get_boolean_mask(mask_data, element_mask)
            count = np.sum(combined_mask)
            
            # Use the element_type directly
            element_name = element_type  # Will be converted to underscore format in create_count_record
            element_records.append(create_count_record(base_row, countname, element_name, count))
        return element_records
    
    # Helper function to process counts by control levels
    def count_by_control_levels(mask_data, control_data, base_row, countname):
        control_records = []
        control_levels = {
            'high': ['street-tree'],
            'medium': ['park-tree'],
            'low': ['reserve-tree', 'improved-tree']
        }
        
        for level, control_types in control_levels.items():
            level_mask = np.zeros(len(control_data), dtype=bool)
            for control_type in control_types:
                level_mask |= (control_data == control_type)
            
            combined_mask = get_boolean_mask(mask_data, level_mask)
            count = np.sum(combined_mask)
            control_records.append(create_count_record(base_row, countname, level, count))
        return control_records
    
    # Helper function to process points near a feature using KDTree
    def count_near_features(feature_mask, urban_data, base_row, countname, distance=5.0):
        near_feature_records = []
        all_points = polydata.points
        feature_points = all_points[feature_mask]
        
        if len(feature_points) > 0:
            feature_tree = cKDTree(feature_points)
            distances, _ = feature_tree.query(all_points, k=1, distance_upper_bound=distance)
            near_feature_mask = distances < distance
            
            for element_type in URBAN_ELEMENT_TYPES:
                element_mask = urban_data == element_type
                count = np.sum(near_feature_mask & element_mask)
                # Use the element_type directly without prefix
                element_name = element_type  # Will be converted to underscore format in create_count_record
                near_feature_records.append(create_count_record(base_row, countname, element_name, count))
        return near_feature_records
    
    # Check if we have the necessary data for capability processing
    if capabilities_info is None:
        raise ValueError("capabilities_info is required but was not provided")
    
    # Check if urban elements data is available
    has_urban_elements = 'search_urban_elements' in polydata.point_data
    
    #---------------------------------------------------------------------------
    # 1. BIRD CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    
    # 1.1 Bird socialise
    # 1.1.1 Urban element / design action: Canopy volume across control levels: high, medium, low
    capability_id = '1.1.1'
    countname = 'canopy_volume'
    
    bird_socialise_mask = capabilities_info['capability_id'] == capability_id
    if any(bird_socialise_mask):
        # Check required fields
        field_name = 'capabilities_bird_socialise_perch-branch'
            
        base_row = capabilities_info[bird_socialise_mask].iloc[0]
        perch_branch = polydata.point_data[field_name]
        forest_control = polydata.point_data['forest_control']
        
        # Count by control levels (high, medium, low)
        control_level_records = count_by_control_levels(perch_branch, forest_control, base_row, countname)
        count_records.extend(control_level_records)
    
    # 1.2 Bird feed
    # 1.2.1 Urban element / design action: Artificial bark installed on branches, utility poles
    capability_id = '1.2.1'
    countname = 'artificial_bark'
    
    bird_feed_mask = capabilities_info['capability_id'] == capability_id
    if any(bird_feed_mask):
        field_name = 'capabilities_bird_feed_peeling-bark'
            
        base_row = capabilities_info[bird_feed_mask].iloc[0]
        peeling_bark = polydata.point_data[field_name]
        precolonial_mask = polydata.point_data['forest_precolonial']
        
        # Only count non-precolonial bark (artificial installations)
        artificial_bark_mask = get_boolean_mask(peeling_bark) & (~precolonial_mask)
        count = np.sum(artificial_bark_mask)
        bark_record = create_count_record(base_row, countname, 'installed', count)
        count_records.append(bark_record)
        
        # Additionally break down by urban elements if available
        if has_urban_elements:
            urban_elements = polydata.point_data['search_urban_elements']
            bark_element_records = count_by_urban_elements(artificial_bark_mask, urban_elements, base_row, f"{countname}_by_element")
            count_records.extend(bark_element_records)
    
    # 1.3 Bird raise young
    # 1.3.1 Urban element / design action: Artificial hollows installed on branches, utility poles
    capability_id = '1.3.1'
    countname = 'artificial_hollows'
    
    bird_raise_young_mask = capabilities_info['capability_id'] == capability_id
    if any(bird_raise_young_mask):
        field_name = 'capabilities_bird_raise-young_hollow'
        base_row = capabilities_info[bird_raise_young_mask].iloc[0]
        hollow = polydata.point_data[field_name]
        precolonial_mask = polydata.point_data['forest_precolonial']
        
        # Only count non-precolonial hollows (artificial installations)
        artificial_hollow_mask = get_boolean_mask(hollow) & (~precolonial_mask)
        count = np.sum(artificial_hollow_mask)
        hollow_record = create_count_record(base_row, countname, 'installed', count)
        count_records.append(hollow_record)
        
        # Additionally break down by urban elements if available
        if has_urban_elements:
            urban_elements = polydata.point_data['search_urban_elements']
            hollow_element_records = count_by_urban_elements(artificial_hollow_mask, urban_elements, base_row, f"{countname}_by_element")
            count_records.extend(hollow_element_records)
    
    #---------------------------------------------------------------------------
    # 2. REPTILE CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    
    # 2.1 Reptile traverse
    # 2.1.1 Urban element / design action: Count of site voxels converted from urban elements
    capability_id = '2.1.1'
    countname = 'urban_conversion'
    
    reptile_traverse_mask = capabilities_info['capability_id'] == capability_id
    if any(reptile_traverse_mask):
        field_name = 'capabilities_reptile_traverse_traversable'
            
        base_row = capabilities_info[reptile_traverse_mask].iloc[0]
        reptile_traverse = polydata.point_data[field_name]
        urban_elements = polydata.point_data['search_urban_elements']
        
        # Count traversable points by urban element type
        traverse_records = count_by_urban_elements(reptile_traverse, urban_elements, base_row, countname)
        count_records.extend(traverse_records)
    
    # 2.2 Reptile forage
    # 2.2.1 Urban element / design action: Count of voxels converted from urban elements (low vegetation)
    capability_id = '2.2.1'
    countname = 'low_veg'
    
    reptile_forage_low_veg_mask = capabilities_info['capability_id'] == capability_id
    if any(reptile_forage_low_veg_mask):
        field_name = 'capabilities_reptile_forage_ground-cover'
            
        base_row = capabilities_info[reptile_forage_low_veg_mask].iloc[0]
        low_veg = polydata.point_data[field_name]
        urban_elements = polydata.point_data['search_urban_elements']
        
        # Count low vegetation points by urban element type
        low_veg_records = count_by_urban_elements(low_veg, urban_elements, base_row, countname)
        count_records.extend(low_veg_records)
    
    # 2.2.2 Urban element / design action: Dead branch volume across control levels
    capability_id = '2.2.2'
    countname = 'dead_branch'
    
    reptile_forage_dead_branch_mask = capabilities_info['capability_id'] == capability_id
    if any(reptile_forage_dead_branch_mask):
        field_name = 'capabilities_reptile_forage_dead-branch'
            
        base_row = capabilities_info[reptile_forage_dead_branch_mask].iloc[0]
        dead_branch = polydata.point_data[field_name]
        forest_control = polydata.point_data['forest_control']
        
        # Count dead branch points by control level
        dead_branch_records = count_by_control_levels(dead_branch, forest_control, base_row, countname)
        count_records.extend(dead_branch_records)
    
    # 2.2.3 Urban element / design action: Number of epiphytes installed (mistletoe)
    capability_id = '2.2.3'
    countname = 'mistletoe'
    
    reptile_forage_epiphyte_mask = capabilities_info['capability_id'] == capability_id
    if any(reptile_forage_epiphyte_mask):
        field_name = 'capabilities_reptile_forage_epiphyte'
            
        base_row = capabilities_info[reptile_forage_epiphyte_mask].iloc[0]
        epiphyte = polydata.point_data[field_name]
        precolonial_mask = polydata.point_data['forest_precolonial']
        
        # Only count non-precolonial epiphytes (installed)
        artificial_epiphyte_mask = get_boolean_mask(epiphyte) & (~precolonial_mask)
        count = np.sum(artificial_epiphyte_mask)
        epiphyte_record = create_count_record(base_row, countname, 'installed', count)
        count_records.append(epiphyte_record)
    
    # 2.3 Reptile shelter
    if has_urban_elements:
        urban_elements = polydata.point_data['search_urban_elements']
        
        # 2.3.1 Urban element / design action: Count of ground elements supporting fallen logs
        capability_id = '2.3.1'
        countname = 'near_fallen_5m'
        
        reptile_shelter_fallen_log_mask = capabilities_info['capability_id'] == capability_id
        if any(reptile_shelter_fallen_log_mask):
            field_name = 'capabilities_reptile_shelter_fallen-log'
                
            base_row = capabilities_info[reptile_shelter_fallen_log_mask].iloc[0]
            fallen_log = polydata.point_data[field_name]
            fallen_log_mask = get_boolean_mask(fallen_log)
            
            # Count urban elements within 5m of fallen logs
            fallen_log_records = count_near_features(fallen_log_mask, urban_elements, base_row, countname)
            count_records.extend(fallen_log_records)
        
        # 2.3.2 Urban element / design action: Count of ground elements supporting fallen trees
        capability_id = '2.3.2'
        countname = 'near_fallen_5m'
        
        reptile_shelter_fallen_tree_mask = capabilities_info['capability_id'] == capability_id
        if any(reptile_shelter_fallen_tree_mask):
            field_name = 'capabilities_reptile_shelter_fallen-tree'
                
            base_row = capabilities_info[reptile_shelter_fallen_tree_mask].iloc[0]
            fallen_tree = polydata.point_data[field_name]
            fallen_tree_mask = get_boolean_mask(fallen_tree)
            
            # Count urban elements within 5m of fallen trees
            fallen_tree_records = count_near_features(fallen_tree_mask, urban_elements, base_row, countname)
            count_records.extend(fallen_tree_records)
    
    #---------------------------------------------------------------------------
    # 3. TREE CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    
    # 3.1 Tree grow
    # 3.1.1 Urban element / design action: Count of number of trees planted this timestep
    capability_id = '3.1.1'
    countname = 'trees_planted'
    
    tree_grow_mask = capabilities_info['capability_id'] == capability_id
    if any(tree_grow_mask):
            
        base_row = capabilities_info[tree_grow_mask].iloc[0]
        print(tree_df)
        total_trees_planted = tree_df['number_of_trees_to_plant'].sum()
        tree_planted_record = create_count_record(base_row, countname, 'total', total_trees_planted)
        count_records.append(tree_planted_record)
    
    # 3.2 Tree age
    # 3.2.1 Urban element / design action: Count of AGE-IN-PLACE actions
    capability_id = '3.2.1'
    countname = 'AGE-IN-PLACE_actions'
    
    tree_age_mask = capabilities_info['capability_id'] == capability_id
    if any(tree_age_mask):
            
        base_row = capabilities_info[tree_age_mask].iloc[0]
        
        # Define rewilding action types
        rewilding_types = ['footprint-depaved', 'exoskeleton', 'node-rewilded']
        
        # Count occurrences of each rewilding type
        for rwild_type in rewilding_types:
            count = sum(tree_df['rewilded'] == rwild_type)
            age_record = create_count_record(base_row, countname, rwild_type, count)
            count_records.append(age_record)
    
    # 3.3 Tree persist
    # 3.3.3 Urban element / design action: Count of site voxels converted from urban elements (eligible soil)
    capability_id = '3.3.3'
    countname = 'eligible_soil'
    
    tree_persist_mask = capabilities_info['capability_id'] == capability_id
    if any(tree_persist_mask):
        field_name = 'scenario_rewildingPlantings'
            
        base_row = capabilities_info[tree_persist_mask].iloc[0]
        rewilding_plantings = polydata.point_data[field_name]
        urban_elements = polydata.point_data['search_urban_elements']
        plantings_mask = rewilding_plantings >= 1

        # Count eligible soil points by urban element type
        eligible_soil_records = count_by_urban_elements(plantings_mask, urban_elements, base_row, countname)
        count_records.extend(eligible_soil_records)
    
    # Convert all records to DataFrame
    counts_df = pd.DataFrame(count_records) if count_records else pd.DataFrame()
    
    # Ensure capability_id is string type to prevent floating point conversion in CSV
    if not counts_df.empty and 'capability_id' in counts_df.columns:
        counts_df['capability_id'] = counts_df['capability_id'].astype(str)
    
    return counts_df

def main():
    """Main function to extract capability counts from generated VTK files"""
    # Load capabilities info
    capabilities_info_path = Path('data/revised/final/stats/arboreal-future-stats/data/capabilities_info.csv')

    
    capabilities_info = pd.read_csv(capabilities_info_path)
    print(f"Loaded capabilities info from {capabilities_info_path}")
    
    #--------------------------------------------------------------------------
    # GATHER USER INPUTS
    #--------------------------------------------------------------------------
    # Default values
    default_sites = ['trimmed-parade']
    default_scenarios = ['baseline', 'positive', 'trending']
    default_years = [0, 10, 30, 60, 180]
    default_voxel_size = 1
    
    # Ask for sites
    sites_input = input(f"Enter site(s) to process (comma-separated) or press Enter for default {default_sites}: ")
    sites = sites_input.split(',') if sites_input else default_sites
    sites = [site.strip() for site in sites]
    
    # Ask for scenarios
    print("\nAvailable scenarios: baseline, positive, trending")
    scenarios_input = input(f"Enter scenario(s) to process (comma-separated) or press Enter for default {default_scenarios}: ")
    scenarios = scenarios_input.split(',') if scenarios_input else default_scenarios
    scenarios = [scenario.strip() for scenario in scenarios]
    
    # Check if baseline is included
    include_baseline = 'baseline' in scenarios
    if include_baseline:
        scenarios.remove('baseline')
    
    # Ask for years/trimesters
    years_input = input(f"Enter years to process (comma-separated) or press Enter for default {default_years}: ")
    years = [int(year.strip()) for year in years_input.split(',')] if years_input else default_years
    
    # Ask for voxel size
    voxel_size_input = input(f"Enter voxel size (default {default_voxel_size}): ")
    voxel_size = int(voxel_size_input) if voxel_size_input else default_voxel_size
    
    # Print summary of selected options
    print("\n===== Processing with the following parameters =====")
    print(f"Sites: {sites}")
    print(f"Scenarios: {scenarios}")
    print(f"Process baseline: {include_baseline}")
    print(f"Years/Trimesters: {years}")
    print(f"Voxel Size: {voxel_size}")
    
    # Confirm proceeding
    confirm = input("\nProceed with these settings? (yes/no, default yes): ")
    if confirm.lower() in ['no', 'n']:
        print("Operation cancelled.")
        return
    
    #--------------------------------------------------------------------------
    # EXTRACT CAPABILITIES COUNTS
    #--------------------------------------------------------------------------
    print("\n===== EXTRACTING CAPABILITY COUNTS =====")
    
    # Create output directory
    output_dir = Path('data/revised/final/stats/arboreal-future-stats/data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {output_dir}")
    
    # Process each site
    for site in sites:
        print(f"\n=== Processing site: {site} ===")
        
        # Initialize combined dataframes for this site
        all_capabilities_counts = []
        all_urban_elements_counts = []
        
        # Process baseline if requested
        if include_baseline:
            print(f"Processing baseline for site: {site}")
            baseline_path = f'data/revised/final/{site}/{site}_baseline_resources_{voxel_size}_with_capabilities.vtk'
        
            baseline_polydata = pv.read(baseline_path)
            print(f"  Reading capability data from baseline...")
            
            # Collect capability statistics
            baseline_stats_df = collect_capability_stats(
                baseline_polydata, 
                capabilities_info,
                site=site,
                scenario='baseline',
                timestep='baseline',
                voxel_size=voxel_size
            )
            all_capabilities_counts.append(baseline_stats_df)
            
            # Generate urban element counts
            urban_element_counts = converted_urban_element_counts(
                site=site,
                scenario='baseline',
                year='baseline',
                polydata=baseline_polydata,
                tree_df=None,
                capabilities_info=capabilities_info
            )
            
            # Add to combined urban elements counts
            if not urban_element_counts.empty:
                all_urban_elements_counts.append(urban_element_counts)
        
        # Process each scenario
        for scenario in scenarios:
            print(f"\n=== Processing scenario: {scenario} for site: {site} ===")
            
            # Process each year
            for year in years:
                print(f"Processing year {year} for site: {site}, scenario: {scenario}")
                
                # Load VTK file with capabilities
                vtk_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_capabilities.vtk'
                
                if not Path(vtk_path).exists():
                    print(f"Warning: VTK file {vtk_path} not found. Skipping.")
                    continue
                    
                polydata = pv.read(vtk_path)
                
                # Load tree dataframe if it exists
                tree_df_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv'
                tree_df = None
                if Path(tree_df_path).exists():
                    tree_df = pd.read_csv(tree_df_path)
                    print(f"  Loaded tree dataframe from {tree_df_path}")
                
                # Collect capability statistics
                year_stats_df = collect_capability_stats(
                    polydata, 
                    capabilities_info,
                    site=site,
                    scenario=scenario,
                    timestep=year,
                    voxel_size=voxel_size
                )
                all_capabilities_counts.append(year_stats_df)
                
                # Generate urban element counts
                urban_element_counts = converted_urban_element_counts(
                    site=site,
                    scenario=scenario,
                    year=year,
                    polydata=polydata,
                    tree_df=tree_df,
                    capabilities_info=capabilities_info
                )
                
                # Add to combined urban elements counts
                if not urban_element_counts.empty:
                    all_urban_elements_counts.append(urban_element_counts)
        
        # Convert to dataframes
        capabilities_count_df = pd.concat(all_capabilities_counts, ignore_index=True)
        capabilities_count_df['capability_id'] = capabilities_count_df['capability_id'].astype(str)
        
        # Save capabilities counts for this site (all scenarios combined)
        capabilities_path = output_dir / f'{site}_all_scenarios_{voxel_size}_capabilities_counts.csv'
        capabilities_count_df.to_csv(capabilities_path, index=False)
        print(f"Saved capability statistics to {capabilities_path}")
        
        # Save urban element counts for this site (all scenarios combined)
        if all_urban_elements_counts:
            urban_elements_count_df = pd.concat(all_urban_elements_counts, ignore_index=True)
            counts_path = output_dir / f'{site}_all_scenarios_{voxel_size}_urban_element_counts.csv'
            urban_elements_count_df.to_csv(counts_path, index=False)
            print(f"Saved urban element counts to {counts_path}")
    
    print("\n===== All processing completed =====")

if __name__ == "__main__":
    main()