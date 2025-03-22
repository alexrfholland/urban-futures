import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
from scipy.spatial import cKDTree

# Mapping for urban elements counts to capabilityID
URBAN_ELEMENT_ID_MAP = {
    # Bird
    ('bird', 'socialise', 'canopy_volume'): '1.1.2',  # Canopy volume across control levels
    ('bird', 'feed', 'artificial_bark'): '1.2.1',     # Artificial bark installed
    ('bird', 'raise-young', 'artificial_hollows'): '1.3.1',  # Artificial hollows
    
    # Reptile
    ('reptile', 'traverse', 'urban_conversion'): '2.1.1',  # Urban element conversion
    ('reptile', 'forage', 'mistletoe'): '2.2.3',      # Number of epiphytes installed
    ('reptile', 'forage', 'low_veg'): '2.2.1',        # Count of voxels converted
    ('reptile', 'forage', 'dead_branch'): '2.2.2',    # Dead branch volume
    ('reptile', 'shelter', 'near_fallen_5m'): '2.3.1', # Supporting fallen logs
    
    # Tree
    ('tree', 'grow', 'trees_planted'): '3.1.1',       # Trees planted
    ('tree', 'age', 'AGE-IN-PLACE_actions'): '3.2.1', # AGE-IN-PLACE actions
    ('tree', 'persist', 'eligible_soil'): '3.3.3'     # Eligible soil in urban elements
}

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
    
    # Helper function to create a count record dict
    def create_count_record(persona, capability, countname, countelement, count, capability_id=None):
        # Use the capabilities dataframe to look up capability_id if not provided
        if capability_id is None:
            capability_id = URBAN_ELEMENT_ID_MAP.get((persona, capability, countname), 'NA')
        
        # Find hierarchical positions from capabilities_info if available
        hpos = -1
        capability_no = -1
        indicator_no = -1
        
        if capabilities_info is not None:
            # Get the hierarchical positions for this persona/capability
            persona_mask = capabilities_info['persona'] == persona
            if any(persona_mask):
                hpos = capabilities_info[persona_mask]['hpos'].iloc[0]
                
                capability_mask = persona_mask & (capabilities_info['capability'] == capability)
                if any(capability_mask):
                    capability_no = capabilities_info[capability_mask]['capability_no'].iloc[0]
                    
                    # For indicator, look for specific countname or closest match
                    indicator_mask = capability_mask & (capabilities_info['numeric_indicator'] != '')
                    if any(indicator_mask):
                        indicator_no = capabilities_info[indicator_mask]['indicator_no'].iloc[0]
            
        return {
            'site': site,
            'scenario': scenario,
            'timestep': year_str,
            'persona': persona,
            'capability': capability,
            'countname': countname,
            'countelement': countelement.replace(' ', '_'),
            'count': int(count),
            'capabilityID': capability_id,
            'hpos': hpos,
            'capability_no': capability_no,
            'indicator_no': indicator_no
        }
    
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
    def count_by_urban_elements(mask_data, urban_data, count_name, persona, capability, capability_id):
        element_counts = []
        for element_type in URBAN_ELEMENT_TYPES:
            element_mask = urban_data == element_type
            combined_mask = get_boolean_mask(mask_data, element_mask)
            count = np.sum(combined_mask)
            
            # Use the element_type directly
            element_name = element_type  # Will be converted to underscore format in create_count_record
            element_counts.append(create_count_record(persona, capability, count_name, element_name, count, capability_id))
        return element_counts
    
    # Helper function to process counts by control levels
    def count_by_control_levels(mask_data, control_data, count_name, persona, capability, capability_id):
        control_counts = []
        control_levels = {
            'high': ['street-tree'],
            'medium': ['park-tree'],
            'low': ['reserve-tree', 'improved-tree']
        }
        
        for level, control_types in control_levels.items():
            for control_type in control_types:
                combined_mask = get_boolean_mask(mask_data, control_data == control_type)
                count = np.sum(combined_mask)
                control_counts.append(create_count_record(persona, capability, count_name, level, count, capability_id))
        return control_counts
    
    # Helper function to process points near a feature using KDTree
    def count_near_features(feature_mask, urban_data, count_name, prefix, persona, capability, capability_id, distance=5.0):
        near_feature_counts = []
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
                near_feature_counts.append(create_count_record(persona, capability, count_name, element_name, count, capability_id))
        return near_feature_counts
    
    #---------------------------------------------------------------------------
    # 1. BIRD CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    
    # 1.1 Bird socialise
    # 1.1.1 Canopy volume across control levels
    capability_id = '1.1.1'
    count_element = 'canopy_volume'
    
    if 'maskForTrees' in polydata.point_data and 'forest_control' in polydata.point_data:
        mask_for_trees = polydata.point_data['maskForTrees']
        forest_control = polydata.point_data['forest_control']
        
        # Create a mask for tree voxels
        tree_mask = (mask_for_trees == 1)
        control_level_counts = count_by_control_levels(tree_mask, forest_control, 'canopy_volume', 
                               'bird', 'socialise', '1.1.1')
        count_records.extend(control_level_counts)
        print(f"Added {len(control_level_counts)} bird socialise canopy volume records")
    
    # 1.2 Bird feed
    # 1.2.1 Artificial bark installed
    capability_id = '1.2.1'
    count_element = 'artificial_bark'
    if 'capabilities_bird_feed_peeling-bark' in polydata.point_data and 'precolonial' in polydata.point_data:
        peeling_bark = polydata.point_data['capabilities_bird_feed_peeling-bark']        
        precolonial_mask = polydata.point_data['precolonial']
        artificial_bark_mask = get_boolean_mask(peeling_bark) & (~precolonial_mask)
        
        count = np.sum(artificial_bark_mask)
        bark_record = create_count_record('bird', 'feed', 'artificial_bark', 'installed', count, '1.2.1')
        count_records.append(bark_record)
        print(f"Added bird feed artificial bark record")
        
        # Additional: If you also want to break down by urban elements
        if 'search_urban_elements' in polydata.point_data:
            urban_elements = polydata.point_data['search_urban_elements']
            bark_element_counts = count_by_urban_elements(artificial_bark_mask, urban_elements, 
                                 'artificial_bark_by_element', 'bird', 'feed', '1.2.1')
            count_records.extend(bark_element_counts)
            print(f"Added {len(bark_element_counts)} bird feed bark by urban element records")
    
    # 1.3 Bird raise young
    # 1.3.1 Artificial hollows installed
    capability_id = '1.3.1'
    count_element = 'artificial_hollows'
    if 'capabilities_bird_raise-young_hollow' in polydata.point_data and 'precolonial' in polydata.point_data:
        hollow = polydata.point_data['capabilities_bird_raise-young_hollow']
        precolonial_mask = polydata.point_data['precolonial']
        artificial_hollow_mask = get_boolean_mask(hollow) & (~precolonial_mask)
        
        count = np.sum(artificial_hollow_mask)
        hollow_record = create_count_record('bird', 'raise-young', 'artificial_hollows', 'installed', count, '1.3.1')
        count_records.append(hollow_record)
        print(f"Added bird raise-young artificial hollows record")
        
        # Additional: If you also want to break down by urban elements
        if 'search_urban_elements' in polydata.point_data:
            urban_elements = polydata.point_data['search_urban_elements']
            hollow_element_counts = count_by_urban_elements(artificial_hollow_mask, urban_elements, 
                                   'artificial_hollows_by_element', 'bird', 'raise-young', '1.3.1')
            count_records.extend(hollow_element_counts)
            print(f"Added {len(hollow_element_counts)} bird raise-young hollows by urban element records")
    
    #---------------------------------------------------------------------------
    # 2. REPTILE CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    
    # 2.1 Reptile traverse
    # 2.1.1 Count of site voxels converted from urban elements
    capability_id = '2.1.1'
    count_element = 'urban_conversion'
    if 'capabilities_reptile_traverse_traversable' in polydata.point_data and 'search_urban_elements' in polydata.point_data:
        reptile_traverse = polydata.point_data['capabilities_reptile_traverse_traversable']
        urban_elements = polydata.point_data['search_urban_elements']
        
        traverse_counts = count_by_urban_elements(reptile_traverse, urban_elements, 'urban_conversion', 
                                                'reptile', 'traverse', '2.1.1')
        count_records.extend(traverse_counts)
        print(f"Added {len(traverse_counts)} reptile traverse by urban element records")
    
    # 2.2 Reptile forage
    # 2.2.1 Count of voxels converted from urban elements (low vegetation)
    capability_id = '2.2.1'
    count_element = 'low_veg'
    if 'capabilities_reptile_forage_ground-cover' in polydata.point_data and 'search_urban_elements' in polydata.point_data:
        low_veg = polydata.point_data['capabilities_reptile_forage_ground-cover']
        urban_elements = polydata.point_data['search_urban_elements']
        
        low_veg_counts = count_by_urban_elements(low_veg, urban_elements, 'low_veg', 
                                               'reptile', 'forage', '2.2.1')
        count_records.extend(low_veg_counts)
        print(f"Added {len(low_veg_counts)} reptile forage low vegetation records")
    
    # 2.2.2 Dead branch volume across control levels
    capability_id = '2.2.2'
    count_element = 'dead_branch'
    if 'capabilities_reptile_forage_dead-branch' in polydata.point_data and 'forest_control' in polydata.point_data:
        dead_branch = polydata.point_data['capabilities_reptile_forage_dead-branch']
        forest_control = polydata.point_data['forest_control']
        
        dead_branch_counts = count_by_control_levels(dead_branch, forest_control, 'dead_branch', 
                                                   'reptile', 'forage', '2.2.2')
        count_records.extend(dead_branch_counts)
        print(f"Added {len(dead_branch_counts)} reptile forage dead branch by control level records")
    
    # 2.2.3 Number of epiphytes installed (mistletoe)
    capability_id = '2.2.3'
    count_element = 'mistletoe'
    if 'capabilities_reptile_forage_epiphyte' in polydata.point_data and 'precolonial' in polydata.point_data:
        epiphyte = polydata.point_data['capabilities_reptile_forage_epiphyte']
        precolonial_mask = polydata.point_data['precolonial']
        epiphyte_mask = get_boolean_mask(epiphyte) & (~precolonial_mask)
        
        count = np.sum(epiphyte_mask)
        epiphyte_record = create_count_record('reptile', 'forage', 'mistletoe', 'installed', count, '2.2.3')
        count_records.append(epiphyte_record)
        print(f"Added reptile forage epiphyte record")
    
    # 2.3 Reptile shelter
    
    if 'search_urban_elements' in polydata.point_data:
        urban_elements = polydata.point_data['search_urban_elements']
        
        # 2.3.1 Count of ground elements supporting fallen logs
        capability_id = '2.3.1'
        count_element = 'fallen_log'
        if 'capabilities_reptile_shelter_fallen-log' in polydata.point_data:
            fallen_log = polydata.point_data['capabilities_reptile_shelter_fallen-log']
            fallen_log_mask = get_boolean_mask(fallen_log)
            
            fallen_log_counts = count_near_features(fallen_log_mask, urban_elements, 'near_fallen_5m', 
                                                 '', 'reptile', 'shelter', '2.3.1')
            count_records.extend(fallen_log_counts)
            print(f"Added {len(fallen_log_counts)} reptile shelter fallen log proximity records")
        
        # 2.3.2 Count of ground elements supporting fallen trees
        capability_id = '2.3.2'
        count_element = 'fallen_tree'
        if 'capabilities_reptile_shelter_fallen-tree' in polydata.point_data:
            fallen_tree = polydata.point_data['capabilities_reptile_shelter_fallen-tree']
            fallen_tree_mask = get_boolean_mask(fallen_tree)
            
            fallen_tree_counts = count_near_features(fallen_tree_mask, urban_elements, 'near_fallen_5m', 
                                                  '', 'reptile', 'shelter', '2.3.2')
            count_records.extend(fallen_tree_counts)
            print(f"Added {len(fallen_tree_counts)} reptile shelter fallen tree proximity records")
    
    #---------------------------------------------------------------------------
    # 3. TREE CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    # 3.1 Tree grow
    # 3.1.1 Count of number of trees planted this timestep
    capability_id = '3.1.1'
    count_element = 'trees_planted'
    if tree_df is not None and 'number_of_trees_to_plant' in tree_df.columns:
        total_trees_planted = tree_df['number_of_trees_to_plant'].sum()
        tree_planted_record = create_count_record('tree', 'grow', 'trees_planted', 'total', total_trees_planted, '3.1.1')
        count_records.append(tree_planted_record)
        print(f"Added tree grow planted trees record")
    
    # 3.2 Tree age
    # 3.2.1 Count of AGE-IN-PLACE actions
    capability_id = '3.2.1'
    count_element = 'AGE-IN-PLACE_actions'
    if tree_df is not None and 'rewilded' in tree_df.columns:
        # Define rewilding action types
        rewilding_types = ['footprint-depaved', 'exoskeleton', 'node-rewilded']
        
        # Count occurrences of each rewilding type
        age_in_place_records = []
        for rwild_type in rewilding_types:
            count = sum(tree_df['rewilded'] == rwild_type)
            age_record = create_count_record('tree', 'age', 'AGE-IN-PLACE_actions', rwild_type, count, '3.2.1')
            age_in_place_records.append(age_record)
        
        count_records.extend(age_in_place_records)
        print(f"Added {len(age_in_place_records)} tree age age-in-place records")
    
    # 3.3 Tree persist
    # 3.3.3 Count of site voxels converted from urban elements (eligible soil)
    capability_id = '3.3.3'
    count_element = 'eligible_soil'
    if 'scenario_rewildingPlantings' in polydata.point_data and 'search_urban_elements' in polydata.point_data:
        rewilding_plantings = polydata.point_data['scenario_rewildingPlantings']
        urban_elements = polydata.point_data['search_urban_elements']
        
        # Handle numeric/non-numeric rewilding plantings data
        if np.issubdtype(rewilding_plantings.dtype, np.number):
            plantings_mask = rewilding_plantings >= 1
        else:
            # Simple string conversion without extensive error handling
            plantings_mask = np.zeros(polydata.n_points, dtype=bool)
            for i, val in enumerate(rewilding_plantings):
                if val not in ['none', '', 'nan']:
                    try:
                        plantings_mask[i] = float(val) >= 1
                    except (ValueError, TypeError):
                        pass
        
        eligible_soil_counts = count_by_urban_elements(plantings_mask, urban_elements, 'eligible_soil', 
                                                     'tree', 'persist', '3.3.3')
        count_records.extend(eligible_soil_counts)
        print(f"Added {len(eligible_soil_counts)} tree persist eligible soil records")
    
    # Print summary of records collected
    print(f"\nTotal records collected: {len(count_records)}")
    
    # Convert records to DataFrame
    counts_df = pd.DataFrame(count_records) if count_records else pd.DataFrame()
    
    # Ensure capabilityID is string type to prevent floating point conversion in CSV
    if not counts_df.empty:
        counts_df['capabilityID'] = counts_df['capabilityID'].astype(str)
    
    return counts_df

def add_hierarchical_info(capabilities_df, capabilities_info):
    """Add hierarchical position information to the capabilities DataFrame"""
    # Initialize new columns
    capabilities_df['hpos'] = -1
    capabilities_df['capability_no'] = -1 
    capabilities_df['indicator_no'] = -1
    
    # Process each capability row
    for idx, row in capabilities_df.iterrows():
        capability = row['capability']
        
        # Split the capability into parts to help with lookup
        parts = capability.split('_')
        if len(parts) >= 2:
            persona = parts[1]  # e.g., 'bird' from 'capabilities_bird_socialise'
            
            # Find persona in capabilities_info
            persona_mask = capabilities_info['persona'] == persona
            if any(persona_mask):
                capabilities_df.at[idx, 'hpos'] = capabilities_info[persona_mask]['hpos'].iloc[0]
                
            # If more parts, get capability information
            if len(parts) >= 3:
                capability_name = parts[2]  # e.g., 'socialise' from 'capabilities_bird_socialise'
                
                # Find capability in capabilities_info
                capability_mask = persona_mask & (capabilities_info['capability'] == capability_name)
                if any(capability_mask):
                    capabilities_df.at[idx, 'capability_no'] = capabilities_info[capability_mask]['capability_no'].iloc[0]
                    
                # If more parts, get indicator information
                if len(parts) >= 4:
                    indicator_name = parts[3]  # e.g., 'perch-branch' from 'capabilities_bird_socialise_perch-branch'
                    
                    # Find indicator in capabilities_info
                    indicator_mask = capability_mask & (capabilities_info['numeric_indicator'] == indicator_name)
                    if any(indicator_mask):
                        capabilities_df.at[idx, 'indicator_no'] = capabilities_info[indicator_mask]['indicator_no'].iloc[0]
    
    return capabilities_df

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
        
        # Convert to dataframes and add hierarchical position information
        capabilities_count_df = pd.concat(all_capabilities_counts, ignore_index=True)
        capabilities_count_df['capabilityID'] = capabilities_count_df['capabilityID'].astype(str)
        
        # Add hierarchical columns using capabilities_info
        capabilities_count_df = add_hierarchical_info(capabilities_count_df, capabilities_info)
        
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