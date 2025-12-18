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
        print(f'{layer_name} count: {count}')
        stats_df.at[idx, 'count'] = count
    
    return stats_df

def get_site_conditions(polydata, site, voxel_size):
    """
    Count the unique values in the 'search_urban_elements' column.
    
    Args:
        polydata: The polydata with urban elements information
        site: Site name
        voxel_size: Voxel size used
        
    Returns:
        DataFrame with urban elements counts
    """
    if 'search_urban_elements' not in polydata.point_data:
        print(f"Warning: 'search_urban_elements' column not found in the polydata for site {site}")
        return pd.DataFrame()
    
    # Extract the urban elements column
    urban_elements = polydata.point_data['search_urban_elements']
    
    # Convert to pandas Series for easier counting
    urban_series = pd.Series(urban_elements)
    
    # Count unique values
    counts = urban_series.value_counts().reset_index()
    counts.columns = ['urban_element', 'count']
    
    # Add site and voxel_size columns
    counts['site'] = site
    counts['voxel_size'] = voxel_size
    
    return counts


def converted_urban_element_counts(site, scenario, year, polydata, tree_df=None, capabilities_info=None):
    """
    Generate counts of converted urban elements for different capabilities.
    
    For each capability (based on capabilities_info) the function creates count
    records for each design action. Note that for some actions (e.g. bird feed and 
    bird raise young) no breakdown by urban element is applied – only an overall count.
    
    Args:
        site (str): Site name
        scenario (str): Scenario name
        year (int): Year/timestep
        polydata (pyvista.UnstructuredGrid): Polydata with capability information
        tree_df (pd.DataFrame, optional): Tree dataframe for df-based counts
        capabilities_info (pd.DataFrame): DataFrame with capability mapping information
        
    Returns:
        pd.DataFrame: DataFrame containing all counts with columns.
    """
    count_records = []
    year_str = str(year)
    URBAN_ELEMENT_TYPES = [
        'open space', 'green roof', 'brown roof', 'facade', 
        'roadway', 'busy roadway', 'existing conversion', 
        'other street potential', 'parking'
    ]
    def create_count_record(base_row, countname, countelement, count):
        record = base_row.copy()
        record['site'] = site
        record['scenario'] = scenario
        record['timestep'] = year_str
        record['countname'] = countname
        record['countelement'] = countelement  # use as is
        record['count'] = int(count)
        return record

    def get_boolean_mask(data_field, condition=None):
        data_array = np.array(data_field)
        if np.issubdtype(data_array.dtype, np.bool_):
            return data_array if condition is None else data_array & condition
        else:
            base_mask = (data_array != 'none') & (data_array != '')
            return base_mask if condition is None else base_mask & condition

    def count_by_urban_elements(mask_data, urban_data, base_row, countname):
        element_records = []
        for element_type in URBAN_ELEMENT_TYPES:
            element_mask = (urban_data == element_type)
            combined_mask = get_boolean_mask(mask_data, element_mask)
            count = np.sum(combined_mask)
            element_records.append(create_count_record(base_row, countname, element_type, count))
        return element_records

    def count_points_near_senescent_trees(polydata, base_row, countname, distance=1.0):
        """
        Count points near senescent trees (large, senescing, snag, fallen) by rewilded status.
        
        Args:
            polydata: The polydata containing point attributes
            base_row: Base row data to use for all records
            countname: Name of the count metric
            distance: Distance threshold in meters (default 1.0)
            
        Returns:
            List of count records for each rewilded type near senescent trees
        """
        count_records = []
        
        # Identify senescent tree points (either forest_size == [senescing, snag, fallen] or forest_useful_life_expectancy < 10)
        forest_size = polydata.point_data['forest_size']
        forest_ule = polydata.point_data['forest_useful_life_expectancy']
        
        senescent_mask = np.zeros(len(forest_size), dtype=bool)
        senescent_stages = ['senescing', 'snag', 'fallen']
        
        for stage in senescent_stages:
            senescent_mask |= (forest_size == stage)
            
        # Add trees with low useful life expectancy
        senescent_mask |= (forest_ule < 10)
        
        # Get coordinates of all points and senescent points
        all_points = polydata.points
        senescent_points = all_points[senescent_mask]
        
        # If no senescent points found, return empty records
        if len(senescent_points) == 0:
            return count_records
        
        # Extract only x,y coordinates (ignore z) for distance calculation
        all_points_2d = all_points[:, :2]
        senescent_points_2d = senescent_points[:, :2]
        
        # Build KDTree with 2D coordinates of senescent points
        tree = cKDTree(senescent_points_2d)
        
        # Find all points within distance of any senescent point
        distances, _ = tree.query(all_points_2d, k=1, distance_upper_bound=distance)
        near_senescent_mask = distances <= distance
        
        # Mask out existing tree points using maskForTrees
        tree_mask = polydata.point_data['maskforTrees']
        non_tree_mask = ~tree_mask
        
        # Combined mask: points near senescent trees that are not trees themselves
        eligible_points_mask = near_senescent_mask & non_tree_mask
        
        # Count points by rewilded status
        scenario_rewilded = polydata.point_data['scenario_rewilded']
        rewilded_types = ['exoskeleton', 'footprint-depaved', 'rewilded']
        
        # Count for each rewilded type
        for rwild_type in rewilded_types:
            rwild_mask = (scenario_rewilded == rwild_type)
            combined_mask = eligible_points_mask & rwild_mask
            count = np.sum(combined_mask)
            count_records.append(create_count_record(base_row, countname, rwild_type, count))
        
        # Count 'none' category (all eligible points not in the rewilded types or node-rewilded)
        none_mask = eligible_points_mask
        for rwild_type in rewilded_types:
            none_mask &= (scenario_rewilded != rwild_type)
        # Also exclude node-rewilded points
        none_mask &= (scenario_rewilded != 'node-rewilded')
        
        count = np.sum(none_mask)
        count_records.append(create_count_record(base_row, countname, 'none', count))
        
        return count_records
        

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
    
    def count_by_lifecycle_stage(polydata, base_row, countname):

        lifecycle_records = []
        forest_size = polydata.point_data['forest_size']
        lifecycle_stages = ['small', 'medium', 'large', 'senescing', 'snag', 'fallen']
        countelements = {
            'small':'sprout',
            'medium':'mid-phase',
            'large':'large',
            'senescing':'senescent',
            'snag':'snag',
            'fallen':'fallen'
        }
        for stage in lifecycle_stages:
            # Create a mask for this stage
            stage_mask = (forest_size == stage)
            # Count the number of points matching this stage
            count = np.sum(stage_mask)
            # Create a record for this stage
            lifecycle_records.append(create_count_record(base_row, countname, countelements[stage], count))
        
        return lifecycle_records
    
    def count_near_features(feature_mask, urban_data, base_row, countname, distance=5.0):
        near_feature_records = []
        all_points = polydata.points
        feature_points = all_points[feature_mask]
        if len(feature_points) > 0:
            feature_tree = cKDTree(feature_points)
            distances, _ = feature_tree.query(all_points, k=1, distance_upper_bound=distance)
            near_feature_mask = distances < distance
            for element_type in URBAN_ELEMENT_TYPES:
                element_mask = (urban_data == element_type)
                count = np.sum(near_feature_mask & element_mask)
                near_feature_records.append(create_count_record(base_row, countname, element_type, count))
        return near_feature_records

    if capabilities_info is None:
        raise ValueError("capabilities_info is required but was not provided")
    
    has_urban_elements = 'search_urban_elements' in polydata.point_data

    # 1. BIRD CAPABILITIES
    # 1.1 Socialise: Count by control levels from 'capabilities_bird_socialise_perch-branch'
    capability_id = '1.1.1'
    countname = 'canopy_volume'
    bird_socialise_mask = capabilities_info['capability_id'] == capability_id
    if any(bird_socialise_mask):
        field_name = 'capabilities_bird_socialise_perch-branch'
        base_row = capabilities_info[bird_socialise_mask].iloc[0]
        perch_branch = polydata.point_data[field_name]
        forest_control = polydata.point_data['forest_control']
        control_level_records = count_by_control_levels(perch_branch, forest_control, base_row, countname)
        count_records.extend(control_level_records)

    # 1.2 Feed: Count of 'capabilities_bird_feed_peeling-bark' where precolonial is False.
    capability_id = '1.2.1'
    countname = 'artificial_bark'
    bird_feed_mask = capabilities_info['capability_id'] == capability_id
    if any(bird_feed_mask):
        field_name = 'capabilities_bird_feed_peeling-bark'
        base_row = capabilities_info[bird_feed_mask].iloc[0]
        peeling_bark_mask = get_bool_mask(field_name, polydata, True)
        non_precolonial_mask = get_bool_mask('forest_precolonial', polydata, False)
        artificial_bark_mask = peeling_bark_mask & non_precolonial_mask
        count = np.sum(artificial_bark_mask)

        #if year is >175, reduce bark_record to 10%
        if year > 175:
            count = count * 0.1
        
        bark_record = create_count_record(base_row, countname, 'installed', count)
        
        count_records.append(bark_record)

    # 1.3 Raise Young: Count of 'capabilities_bird_raise-young_hollow' where precolonial is False.
    capability_id = '1.3.1'
    countname = 'artificial_hollows'
    bird_raise_young_mask = capabilities_info['capability_id'] == capability_id
    if any(bird_raise_young_mask):
        field_name = 'capabilities_bird_raise-young_hollow'
        base_row = capabilities_info[bird_raise_young_mask].iloc[0]
        hollow_mask = get_bool_mask(field_name, polydata, True)
        non_precolonial_mask = get_bool_mask('forest_precolonial', polydata, False)
        artificial_hollow_mask = hollow_mask & non_precolonial_mask
        count = np.sum(artificial_hollow_mask)
        hollow_record = create_count_record(base_row, countname, 'installed', count)
        #if year is >175, reduce bark_record to 10%
        if year > 175:
            count = count * 0.1
        
        hollow_record = create_count_record(base_row, countname, 'installed', count)
        count_records.append(hollow_record)

    # 2. REPTILE CAPABILITIES
    # 2.1 Traverse: Count by urban elements from 'capabilities_reptile_traverse_traversable'
    capability_id = '2.1.1'
    countname = 'urban_conversion'
    reptile_traverse_mask = capabilities_info['capability_id'] == capability_id
    if any(reptile_traverse_mask):
        field_name = 'capabilities_reptile_traverse_traversable'
        base_row = capabilities_info[reptile_traverse_mask].iloc[0]
        reptile_traverse = polydata.point_data[field_name]
        urban_elements = polydata.point_data['search_urban_elements']
        traverse_records = count_by_urban_elements(reptile_traverse, urban_elements, base_row, countname)
        count_records.extend(traverse_records)

    # 2.2 Forage:
    # 2.2.1 Low Vegetation: Count by urban elements from 'capabilities_reptile_forage_ground-cover'
    capability_id = '2.2.1'
    countname = 'low_veg'
    reptile_forage_low_veg_mask = capabilities_info['capability_id'] == capability_id
    if any(reptile_forage_low_veg_mask):
        field_name = 'capabilities_reptile_forage_ground-cover'
        base_row = capabilities_info[reptile_forage_low_veg_mask].iloc[0]
        low_veg = polydata.point_data[field_name]
        urban_elements = polydata.point_data['search_urban_elements']
        low_veg_records = count_by_urban_elements(low_veg, urban_elements, base_row, countname)
        count_records.extend(low_veg_records)

    # 2.2.2 Dead Branch: Count by control levels from 'capabilities_reptile_forage_dead-branch'
    capability_id = '2.2.2'
    countname = 'dead_branch'
    reptile_forage_dead_branch_mask = capabilities_info['capability_id'] == capability_id
    if any(reptile_forage_dead_branch_mask):
        field_name = 'capabilities_reptile_forage_dead-branch'
        base_row = capabilities_info[reptile_forage_dead_branch_mask].iloc[0]
        dead_branch = polydata.point_data[field_name]
        forest_control = polydata.point_data['forest_control']
        dead_branch_records = count_by_control_levels(dead_branch, forest_control, base_row, countname)
        count_records.extend(dead_branch_records)

    # 2.2.3 Epiphyte: Count of 'capabilities_reptile_forage_epiphyte' where precolonial is False.
    capability_id = '2.2.3'
    countname = 'mistletoe'
    reptile_forage_epiphyte_mask = capabilities_info['capability_id'] == capability_id
    if any(reptile_forage_epiphyte_mask):
        field_name = 'capabilities_reptile_forage_epiphyte'
        base_row = capabilities_info[reptile_forage_epiphyte_mask].iloc[0]
        epiphyte_mask = get_bool_mask(field_name, polydata, True)
        non_precolonial_mask = get_bool_mask('forest_precolonial', polydata, False)
        artificial_epiphyte_mask = epiphyte_mask & non_precolonial_mask
        count = np.sum(artificial_epiphyte_mask)
        epiphyte_record = create_count_record(base_row, countname, 'installed', count)
        count_records.append(epiphyte_record)

    # 2.3 Shelter:
    if has_urban_elements:
        urban_elements = polydata.point_data['search_urban_elements']
        # 2.3.1 Fallen Log: Count by urban elements using KDTree on 'capabilities_reptile_shelter_fallen-log'
        capability_id = '2.3.1'
        countname = 'near_fallen_5m'
        reptile_shelter_fallen_log_mask = capabilities_info['capability_id'] == capability_id
        if any(reptile_shelter_fallen_log_mask):
            field_name = 'capabilities_reptile_shelter_fallen-log'
            base_row = capabilities_info[reptile_shelter_fallen_log_mask].iloc[0]
            fallen_log = polydata.point_data[field_name]
            fallen_log_mask = get_boolean_mask(fallen_log)
            fallen_log_records = count_near_features(fallen_log_mask, urban_elements, base_row, countname)
            count_records.extend(fallen_log_records)
        # 2.3.2 Fallen Tree: Count by urban elements using KDTree on 'capabilities_reptile_shelter_fallen-tree'
        capability_id = '2.3.2'
        countname = 'near_fallen_5m'
        reptile_shelter_fallen_tree_mask = capabilities_info['capability_id'] == capability_id
        if any(reptile_shelter_fallen_tree_mask):
            field_name = 'capabilities_reptile_shelter_fallen-tree'
            base_row = capabilities_info[reptile_shelter_fallen_tree_mask].iloc[0]
            fallen_tree = polydata.point_data[field_name]
            fallen_tree_mask = get_boolean_mask(fallen_tree)
            fallen_tree_records = count_near_features(fallen_tree_mask, urban_elements, base_row, countname)
            count_records.extend(fallen_tree_records)

    # 3. TREE CAPABILITIES
    #3.1.1 Grow:  Canopy across life cycle stages
    # search criteria: Count of 'forest_control' per control class 
    capability_id = '3.1.1'
    countname = 'lifecycle_stage'
    tree_grow_mask = capabilities_info['capability_id'] == capability_id
    if any(tree_grow_mask):
        base_row = capabilities_info[tree_grow_mask].iloc[0]
        points = polydata.point_data['forest_control']
        forest_control = polydata.point_data['forest_control']
        control_level_records = count_by_control_levels(points, forest_control, base_row, countname)
        count_records.extend(control_level_records)

    # 3.2.1 Age: Count AGE-IN-PLACE actions from the dataframe
    capability_id = '3.2.1'
    countname = 'AGE-IN-PLACE_actions'
    tree_age_mask = capabilities_info['capability_id'] == capability_id

    if any(tree_grow_mask):
        base_row = capabilities_info[tree_age_mask].iloc[0]
        age_records = count_points_near_senescent_trees(polydata, base_row, countname, distance=1.0)
        count_records.extend(age_records)
    
    """if any(tree_age_mask):
        base_row = capabilities_info[tree_age_mask].iloc[0]
        rewilding_types = ['paved', 'footprint-depaved', 'exoskeleton', 'node-rewilded']
        for rwild_type in rewilding_types:
            count = np.sum(tree_df['rewilded'] == rwild_type)
            if year <60 and rwild_type == 'node-rewilded':
                count = count * 0.1
            age_record = create_count_record(base_row, countname, rwild_type, count)
            count_records.append(age_record)"""

    # 3.3.1 Persist: Count eligible soil points by urban elements
    capability_id = '3.3.3'
    countname = 'eligible_soil'
    tree_persist_mask = capabilities_info['capability_id'] == capability_id
    if any(tree_persist_mask):
        field_name = 'scenario_rewildingPlantings'
        base_row = capabilities_info[tree_persist_mask].iloc[0]
        rewilding_plantings = polydata.point_data[field_name]
        urban_elements = polydata.point_data['search_urban_elements']
        plantings_mask = rewilding_plantings >= 1
        eligible_soil_records = count_by_urban_elements(plantings_mask, urban_elements, base_row, countname)
        count_records.extend(eligible_soil_records)
    
    counts_df = pd.DataFrame(count_records) if count_records else pd.DataFrame()
    if not counts_df.empty and 'capability_id' in counts_df.columns:
        counts_df['capability_id'] = counts_df['capability_id'].astype(str)
    return counts_df


def get_bool_mask(attribute_name, polydata, value=True):
    """
    Creates a boolean mask from a polydata attribute that may contain True, False, or NaN values.
    
    Args:
        attribute_name (str): Name of the attribute in polydata.point_data
        polydata (pyvista.UnstructuredGrid): The polydata containing point attributes
        value (bool, default=True): Whether to return mask for True or False values
                                    If True, returns points where attribute is True
                                    If False, returns points where attribute is False
                                    NaN values are always excluded (set to False in return mask)
    
    Returns:
        np.ndarray: Boolean mask where True indicates points matching the requested value
    """
    # Get the attribute data
    if attribute_name not in polydata.point_data:
        raise ValueError(f"Attribute '{attribute_name}' not found in polydata")
    
    attr_data = polydata.point_data[attribute_name]
    
    # Convert to numpy array to work with it
    attr_array = np.array(attr_data)
    
    # Create mask for NaN values
    if np.issubdtype(attr_array.dtype, np.floating):
        nan_mask = np.isnan(attr_array)
    else:
        # For non-float types, create empty mask
        nan_mask = np.zeros(attr_array.shape, dtype=bool)
    
    # Create mask for True values
    if np.issubdtype(attr_array.dtype, np.bool_):
        true_mask = attr_array
    else:
        # Handle string or other types by checking equality with 'true' or '1'
        true_mask = np.isin(attr_array, [True, 'true', 'True', 1, '1'])
    
    # Return appropriate mask based on the requested value
    if value:
        # Return True for points where attribute is True and not NaN
        return true_mask & (~nan_mask)
    else:
        # Return True for points where attribute is False and not NaN
        return (~true_mask) & (~nan_mask)

def add_empty_rows(count_df, capabilities_info):
    """
    Given the combined urban elements counts DataFrame (across all time steps),
    group by persona and numeric indicator (countname), then for each group determine
    the union of countelements present in the overall dataset for that countname.
    For any group missing one of these countelements, add a row using the group's base row,
    setting countelement to the missing value and count to 0.
    
    Args:
        count_df (pd.DataFrame): Combined urban element counts DataFrame.
        capabilities_info (pd.DataFrame): Base capabilities info DataFrame.
        
    Returns:
        pd.DataFrame: Updated DataFrame with zero-valued rows added.
    """
    # First, determine the overall expected countelements per countname.
    expected_by_countname = {}
    for countname, group in count_df.groupby('countname'):
        expected_by_countname[countname] = set(group['countelement'].unique())
    
    new_rows = []
    # Group by persona and countname. If "persona" does not exist, use "capability" instead.
    group_columns = ['persona', 'countname'] if 'persona' in count_df.columns else ['capability', 'countname']
    for keys, group in count_df.groupby(group_columns):
        # keys = (persona, countname) or (capability, countname)
        countname_val = keys[1] if len(keys) > 1 else keys[0]
        expected = expected_by_countname.get(countname_val, set())
        present = set(group['countelement'].unique())
        missing = expected - present
        for missing_element in missing:
            base_row = group.iloc[0].copy()
            base_row['countelement'] = missing_element
            base_row['count'] = 0
            new_rows.append(base_row)
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        count_df = pd.concat([count_df, new_df], ignore_index=True)
    return count_df


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
    output_dir = Path('data/revised/final/stats/arboreal-future-stats/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    site_conditions_dataframes = []
    
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

        site_conditions_df = None
        
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
                print(f'loaded vtk file: {vtk_path}')
                
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

                if site_conditions_df is None:
                    site_conditions_df = get_site_conditions(polydata, site, voxel_size)
                    site_conditions_dataframes.append(site_conditions_df)
        

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
            urban_elements_count_df = add_empty_rows(urban_elements_count_df, capabilities_info)
            counts_path = output_dir / f'{site}_all_scenarios_{voxel_size}_urban_element_counts.csv'
            urban_elements_count_df.to_csv(counts_path, index=False)
            print(f"Saved urban element counts to {counts_path}")
        
        # Save site conditions for this site (all scenarios combined)
        #all_site_conditions_df = pd.concat(site_conditions_dataframes, ignore_index=True)
        site_conditions_path = output_dir / f'{site}_{voxel_size}_site_conditions.csv'
        site_conditions_df.to_csv(site_conditions_path, index=False)
        print(f"Saved site conditions to {site_conditions_path}")
    
    print("\n===== All processing completed =====")

if __name__ == "__main__":
    main()