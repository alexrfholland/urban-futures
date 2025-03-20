import numpy as np
import pandas as pd
import pyvista as pv
import argparse
from pathlib import Path
from scipy.spatial import cKDTree

"""
CAPABILITIES INFO    
1.  Bird capabilities:

1.1. Socialise: Points where 'stat_perch branch' > 0
   - Birds need branches to perch on for social activities
   
   Numeric indicators:
   - 1.1.1 bird_socialise: Total voxels where stat_perch branch > 0 : 'perch branch'
        -label for graph: 'Perchable canopy volume'
   
   Urban element / design action: 
   - 1.1.2 Canopy volume across control levels: high, medium, low
        # search criteria: Total voxels in polydata['maskForTrees'] == 1, broken down by their ['forest_control'], where high == 'street-tree', medium == 'park-tree', low == 'reserve-tree' OR 'improved-tree'

1.2 Feed: Points where 'stat_peeling bark' > 0
   - Birds feed on insects found under peeling bark
   
   Numeric indicators:
   - 1.2.1 bird_feed: Total voxels where stat_peeling bark > 0 : 'peeling bark'
        - label for graph: 'Peeling bark volume'
   
   Urban element / design action:
   - 1.2.1 Artificial bark installed on branches, utility poles
        #search criteria: Total voxels where polydata['stat_epiphyte'] > 0 & polydata['precolonial'] == False
        TO DO: could include eucs

1.3. Raise Young: Points where 'stat_hollow' > 0
   - Birds need hollows in trees to nest and raise their young
   
   Numeric indicators:
   - 1.3.1 bird_raise_young: Total voxels where stat_hollow > 0 : 'hollow'
        - label for graph: 'Hollow count'

   Urban element / design action:
   - 1.3.1 Artificial hollows installed on branches, utility poles
        #search criteria: Total voxels where polydata['stat_hollow'] > 0 & polydata['precolonial'] == False
        TO DO: could include eucs


2. Reptile capabilities:

2.1. Traverse: Points where 'search_bioavailable' != 'none'
   - Reptiles can move through any bioavailable space
   
   Numeric indicators:
   - 2.1.1 reptile_traverse: Total voxels where search_bioavailable != 'none' : traversable
        - label for graph: 'Non-paved surface area'

   Urban element / design action: 
   - 2.1.1 Count of site voxels converted from: car parks, roads, green roofs, brown roofs, facades
        # search criteria: Total voxels where polydata['capabilities-reptile-traverse'] == 'traversable', 
        # broken down by the defined urban element catagories in polydata['search_urban_elements']


2.2 Foraige: Points where any of the following conditions are met:
   - 'search_bioavailable' == 'low-vegetation' (areas reptiles can move through)
   - 'stat_dead branch' > 0 (dead branches in canopy generate coarse woody debris)
   - 'stat_epiphyte' > 0 (epiphytes in canopy generate fallen leaves)
   
   Numeric indicators:
   - 2.2.1 reptile_forage_low_veg: Voxels where search_bioavailable == 'low-vegetation' : 'ground cover'
         - label for graph: 'Low vegetation surface area'
   - 2.2.2 reptile_forage_dead_branch: Voxels where stat_dead branch > 0 : 'dead branch'
         - label for graph: 'Canopy dead branch volume'
   - 2.2.3 reptile_forage_epiphyte: Voxels where stat_epiphyte > 0 : 'epiphyte'
        - label for graph: 'Epiphyte count'

   Urban element / design action:
   - 2.2.1 Count of voxels converted from  : car parks, roads, green roofs, brown roofs, facades
        # search criteria: Count of 'reptile_forage_low_veg', broken down by the defined urban element catagories in polydata['search_urban_elements']
   - 2.2.2 Dead branch volume across control levels: high, medium, low
        # search criteria: Count of 'reptile_forage_dead_branch', broken down by their ['forest_control'], where high == 'street-tree', medium == 'park-tree', low == 'reserve-tree' OR 'improved-tree'
   - 2.2.3 Number of epiphytes installed in elms
        # search criteria: Count of 'reptile_forage_epiphyte' where 'forest_precolonial' == False

2.3. Shelter: Points where any of the following conditions are met:
   - 'stat_fallen log' > 0 (fallen logs provide shelter)
   - 'forest_size' == 'fallen' (fallen trees provide shelter)
   
   Numeric indicators:
   - 2.3.1 reptile_shelter_fallen_log: Voxels where stat_fallen log > 0 : 'fallen log'
        - label for graph: 'Nurse log volume'
   - 2.3.2 reptile_shelter_fallen_tree: Voxels where forest_size == 'fallen' : 'fallen tree'
        - label for graph: 'Fallen tree volume'

   Urban element / design action:
   - 2.3.1 Count of ground elements supporting fallen logs  : car parks, roads, green roofs, brown roofs, facades within 5m of fallen trees and logs
        #search criteria. Use a ckdTree to find points within 5m where 'reptile_shelter_fallen_log' == True 
        #break these down by the defined urban element catagories in polydata['search_urban_elements']
   - 2.3.2 Count of ground elements supporting fallen trees  : car parks, roads, green roofs, brown roofs, facades within 5m of fallen trees and logs
        #search criteria. Use a ckdTree to find points within 5m where 'reptile_shelter_fallen_tree' == True 
        #break these down by the defined urban element catagories in polydata['search_urban_elements']

3. Tree capabilities:

3.1. Grow: Points where 'stat_other' > 0
   - Areas where trees can grow and establish
   
   Numeric indicators:
   - 3.1.1 tree_grow: Total voxels where stat_other > 0 : 'volume'
        - label for graph: 'Forest biovolume'

   #Urban element / design action: 
   - 3.1.1 count of number of trees planted this timestep
        # search criteria: sum of df['number_of_trees_to_plant']      

3.2. Age: Points where 'search_design_action' == 'improved-tree' OR 'forest_control' == 'reserve-tree'
   - Areas where trees are protected and can mature
   
   Numeric indicators:
   - 3.2.1 tree_age: Total voxels where search_design_action == 'improved-tree' : 'improved tree'
        - label for graph: 'Canopy volume supported by humans'
   - 3.2.2 tree_age: Total voxels where forest_control == 'reserve-tree' : 'reserve tree'
        - label for graph: 'Canopy volume autonomous'

   #Urban element / design action: 
   -3.2.1 count of AGE-IN-PLACE actions: exoskeletons, habitat islands, depaved areas
        # search criteria:  counts of df['rewilded'] == 'footprint-depaved','exoskeleton','node-rewilded'
     -3.2.1 count of AGE-IN-PLACE actions: exoskeletons, habitat islands, depaved areas
        # search criteria:  counts of df['rewilded'] == 'footprint-depaved','exoskeleton','node-rewilded'

3.3. Persist: Terrain points eligible for new tree plantings (ie. depaved and unmanaged land away from trees)

    Numeric indicator:
    - 3.3.3 scenario_rewildingPlantings >= 1 : 'eligble soil'
        - label for graph: 'Ground area for tree recruitment'

    #Urban element / design action:
    -3.3.3/ Count of site voxels converted from: car parks, roads, etc
        #count of subset of polydata['scenario_rewildingPlantings'] >= 1, broken down by the defined urban element catagories in polydata['search_urban_elements']

"""



def process_capabilities(site, scenario, voxel_size, years=None, include_baseline=True):
    """Process capabilities for all years and baseline"""
    if years is None:
        years = [0, 10, 30, 60, 180]
    
    # Dictionary to store all statistics
    all_stats = {}
    
    # Dictionary to track zero counts
    zero_counts = {}
    
    # Flag to track if any data was processed
    data_processed = False
    
    # DataFrame to collect converted urban element counts
    all_converted_urban_element_counts = []
    
    # Process baseline if requested
    if include_baseline:
        print(f"\n=== Processing baseline for site: {site} ===")

        baseline_path = f'data/revised/final/baselines/{site}_baseline_combined_{voxel_size}_urban_features.vtk'
        baseline_vtk = None

        try:
            if Path(baseline_path).exists():
                print(f"Found baseline file: {baseline_path}")
                baseline_vtk = pv.read(baseline_path)
                print(f"Loaded baseline VTK from {baseline_path}")
        except Exception as e:
            print(f"Could not load baseline from {baseline_path}: {e}")
        
        if baseline_vtk is not None:
            # Create capabilities for baseline
            baseline_vtk = create_bird_capabilities(baseline_vtk)
            baseline_vtk = create_reptile_capabilities(baseline_vtk)
            baseline_vtk = create_tree_capabilities(baseline_vtk)
            
            # Collect statistics
            baseline_stats = collect_capability_stats(baseline_vtk)
            all_stats['baseline'] = baseline_stats
            data_processed = True
            
            # Try to load baseline tree dataframe
            tree_df_path = f'data/revised/final/{site}/{site}_baseline_{voxel_size}_treeDF_0.csv'
            try:
                if Path(tree_df_path).exists():
                    baseline_tree_df = pd.read_csv(tree_df_path)
                    print(f"Loaded baseline tree dataframe from {tree_df_path}")
                else:
                    baseline_tree_df = None
                    print(f"Baseline tree dataframe not found at {tree_df_path}")
            except Exception as e:
                baseline_tree_df = None
                print(f"Could not load baseline tree dataframe: {e}")
            
            # Note: We don't collect urban elements counts for baseline
            
            # Check for zero counts
            for key, count in baseline_stats.items():
                if count == 0:
                    print(f"  ##WARNING## Zero count for {key} in baseline")
                    zero_counts.setdefault('baseline', []).append(key)
            
            # Save updated baseline with capabilities
            baseline_output_path = f'data/revised/final/{site}/{site}_baseline_resources_{voxel_size}_with_capabilities.vtk'
            baseline_vtk.save(baseline_output_path)
            print(f"Saved baseline with capabilities to {baseline_output_path}")
        else:
            print(f"Error: Could not find baseline VTK file for {site}")
            # List all VTK files in the baselines directory
            baseline_dir = Path('data/revised/final/baselines')
            if baseline_dir.exists():
                baseline_files = list(baseline_dir.glob(f"{site}*.vtk"))
                if baseline_files:
                    print(f"Available baseline files in {baseline_dir}:")
                    for file_path in baseline_files:
                        print(f"  - {file_path}")
                else:
                    print(f"No baseline files found for {site} in {baseline_dir}")
    
    # Process each year
    for year in years:
        print(f"\n=== Processing year {year} for site: {site}, scenario: {scenario} ===")
        
        # Load VTK file
        vtk_data = load_vtk_file(site, scenario, voxel_size, year)
        if vtk_data is None:
            continue
        
        # Load tree dataframe
        tree_df_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv'
        try:
            if Path(tree_df_path).exists():
                tree_df = pd.read_csv(tree_df_path)
                print(f"Loaded tree dataframe from {tree_df_path}")
            else:
                tree_df = None
                print(f"Tree dataframe not found at {tree_df_path}")
        except Exception as e:
            tree_df = None
            print(f"Could not load tree dataframe: {e}")
        
        # Note: Log and pole DFs are commented out as per instructions
        # log_df = None
        # pole_df = None
        
        # Create capabilities
        vtk_data = create_bird_capabilities(vtk_data)
        vtk_data = create_reptile_capabilities(vtk_data)
        vtk_data = create_tree_capabilities(vtk_data)
        
        # Collect statistics
        year_stats = collect_capability_stats(vtk_data)
        # Store year as string to ensure consistent key types
        all_stats[str(year)] = year_stats
        data_processed = True
        
        # Generate converted urban element counts
        year_counts = converted_urban_element_counts(
            site=site,
            scenario=scenario,
            year=year,
            vtk_data=vtk_data,
            tree_df=tree_df
        )
        
        all_converted_urban_element_counts.append(year_counts)
        
        # Check for zero counts
        for key, count in year_stats.items():
            if count == 0:
                print(f"  ##WARNING## Zero count for {key} in year {year}")
                zero_counts.setdefault(str(year), []).append(key)
        
        # Save updated VTK file
        output_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_capabilities.vtk'
        vtk_data.save(output_path)
        print(f"Saved updated VTK file to {output_path}")
    
    # Check if any data was processed
    if not data_processed:
        print(f"\n### ERROR: No capability statistics were processed for site {site}, scenario {scenario} ###")
        print("Please check that the VTK files exist and are in the expected locations.")
        return None
    
    # Print summary of all statistics
    print("\n=== Capability Statistics Summary ===")
    
    # Group by persona and capability
    personas = ['bird', 'reptile', 'tree']
    for persona in personas:
        print(f"\n{persona.upper()} CAPABILITIES:")
        
        # Get all keys for this persona
        first_time_point = list(all_stats.keys())[0]
        persona_keys = [key for key in all_stats[first_time_point].keys() 
                       if key.startswith(f'capabilities-{persona}-') and key.count('-') == 2]
        
        for capability_key in sorted(persona_keys):
            capability = capability_key.split('-')[2]
            print(f"  {capability.upper()}:")
            
            # Define a custom sort order for time points
            def sort_key(time_point):
                if time_point == 'baseline':
                    return -1  # Baseline comes first
                return int(time_point)  # Convert other time points to integers for sorting
            
            # Print counts for each year, with baseline first, then years in numerical order
            for time_point in sorted(all_stats.keys(), key=sort_key):
                count = all_stats[time_point].get(capability_key, 0)
                print(f"    {time_point}: {count:,}")
                
                # Print component counts if available
                component_keys = [k for k in all_stats[time_point].keys() 
                                if k.startswith(f'{capability_key}-') or 
                                (k.startswith(capability_key) and '_' in k)]
                
                for comp_key in sorted(component_keys):
                    comp_count = all_stats[time_point].get(comp_key, 0)
                    comp_name = comp_key.split('-')[-1] if '-' in comp_key else comp_key.split('_')[-1]
                    print(f"      {comp_name}: {comp_count:,}")
    
    # Print summary of zero counts
    if zero_counts:
        print("\n=== Zero Count Summary ===")
        # Define the same custom sort order for time points
        def sort_key(time_point):
            if time_point == 'baseline':
                return -1  # Baseline comes first
            return int(time_point)  # Convert other time points to integers for sorting
        
        for time_point, keys in sorted(zero_counts.items(), key=lambda x: sort_key(x[0])):
            print(f"\n{time_point}:")
            for key in sorted(keys):
                print(f"  {key}")
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(all_stats)
    stats_dir = Path('data/revised/final/stats')
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f'{site}_{scenario}_capabilities_raw.csv'
    stats_df.to_csv(stats_path)
    print(f"\nSaved raw capability statistics to {stats_path}")
    
    # Combine all converted urban element counts and save to CSV
    if all_converted_urban_element_counts:
        combined_counts = pd.concat(all_converted_urban_element_counts, ignore_index=True)
        counts_path = stats_dir / f'{site}_{scenario}_converted_urban_element_counts.csv'
        combined_counts.to_csv(counts_path, index=False)
        print(f"Saved converted urban element counts to {counts_path}")
    
    return all_stats

def main():
    """Main function to process capabilities for sites and scenarios"""
    #--------------------------------------------------------------------------
    # STEP 1: GATHER USER INPUTS
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
    try:
        years = [int(year.strip()) for year in years_input.split(',')] if years_input else default_years
    except ValueError:
        print("Invalid input for years. Using default values.")
        years = default_years
    
    # Ask for voxel size
    voxel_size_input = input(f"Enter voxel size (default {default_voxel_size}): ")
    try:
        voxel_size = int(voxel_size_input) if voxel_size_input else default_voxel_size
    except ValueError:
        print("Invalid input for voxel size. Using default value.")
        voxel_size = default_voxel_size
    
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
    # STEP 2: PROCESS CAPABILITIES
    #--------------------------------------------------------------------------
    print("\n===== PROCESSING CAPABILITIES =====")
    
    # Process each site
    for site in sites:
        # Process each scenario
        for scenario in scenarios:
            print(f"\n=== Processing site: {site}, scenario: {scenario} ===")
            
            # Process capabilities
            process_capabilities(
                site=site,
                scenario=scenario,
                voxel_size=voxel_size,
                years=years,
                include_baseline=include_baseline
            )
        
        # If baseline was requested but no scenarios, process it separately
        if include_baseline and not scenarios:
            print(f"\n=== Processing baseline only for site: {site} ===")
            process_capabilities(
                site=site,
                scenario="baseline",  # Dummy value, not used for baseline
                voxel_size=voxel_size,
                years=years,
                include_baseline=True
            )
    
    print("\n===== All processing completed =====")

#search criteria for many of the Urban element / design action count 'urban elements'. Use these:
##polydata['search_urban_elements'] catagories:
'open space'
'green roof'
'brown roof'
'facade'
'roadway'
'busy roadway'
'existing conversion'
'other street potential'
'parking'

def create_bird_capabilities(vtk_data):
    print("  Creating bird capability layers...")
    
    # Initialize capability arrays
    bird_socialise = np.zeros(vtk_data.n_points, dtype=bool)
    bird_feed = np.zeros(vtk_data.n_points, dtype=bool)
    bird_raise_young = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize aggregate capability array with 'none'
    capabilities_bird = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Bird Socialise: points in stat_perch_branch > 0
    if 'stat_perch branch' in vtk_data.point_data:
        perch_data = vtk_data.point_data['stat_perch branch']
        if np.issubdtype(perch_data.dtype, np.number):
            bird_socialise_mask = perch_data > 0
        else:
            bird_socialise_mask = (perch_data != 'none') & (perch_data != '') & (perch_data != 'nan')
        
        bird_socialise |= bird_socialise_mask
        capabilities_bird[bird_socialise_mask] = 'socialise'
        print(f"    Bird socialise points: {np.sum(bird_socialise_mask):,}")
    else:
        print("    'stat_perch branch' not found in point data")
    
    # Bird Feed: points in stat_peeling bark > 0
    if 'stat_peeling bark' in vtk_data.point_data:
        bark_data = vtk_data.point_data['stat_peeling bark']
        if np.issubdtype(bark_data.dtype, np.number):
            bird_feed_mask = bark_data > 0
        else:
            bird_feed_mask = (bark_data != 'none') & (bark_data != '') & (bark_data != 'nan')
        
        bird_feed |= bird_feed_mask
        # Override any existing values (later capabilities take precedence)
        capabilities_bird[bird_feed_mask] = 'feed'
        print(f"    Bird feed points: {np.sum(bird_feed_mask):,}")
    else:
        print("    'stat_peeling bark' not found in point data")
    
    # Bird Raise Young: points in stat_hollow > 0
    if 'stat_hollow' in vtk_data.point_data:
        hollow_data = vtk_data.point_data['stat_hollow']
        if np.issubdtype(hollow_data.dtype, np.number):
            bird_raise_young_mask = hollow_data > 0
        else:
            bird_raise_young_mask = (hollow_data != 'none') & (hollow_data != '') & (hollow_data != 'nan')
        
        bird_raise_young |= bird_raise_young_mask
        # Override any existing values (later capabilities take precedence)
        capabilities_bird[bird_raise_young_mask] = 'raise-young'
        print(f"    Bird raise-young points: {np.sum(bird_raise_young_mask):,}")
    else:
        print("    'stat_hollow' not found in point data")
    
    # Add bird capability layers to vtk_data
    vtk_data.point_data['capabilities-bird-socialise'] = bird_socialise
    vtk_data.point_data['capabilities-bird-feed'] = bird_feed
    vtk_data.point_data['capabilities-bird-raise-young'] = bird_raise_young
    vtk_data.point_data['capabilities-bird'] = capabilities_bird
    
    return vtk_data

def create_reptile_capabilities(vtk_data):
    print("  Creating reptile capability layers...")
    
    # Initialize capability arrays
    reptile_traverse = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_forage = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_shelter = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize component arrays
    reptile_forage_low_veg = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_forage_dead_branch = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_forage_epiphyte = np.zeros(vtk_data.n_points, dtype=bool)
    
    reptile_shelter_fallen_log = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_shelter_fallen_tree = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize string arrays for detailed capability types
    capabilities_reptile = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_reptile_traverse = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_reptile_forage = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_reptile_shelter = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Reptile Traverse: points in 'search_bioavailable' != 'none'
    if 'search_bioavailable' in vtk_data.point_data:
        bioavailable_data = vtk_data.point_data['search_bioavailable']
        reptile_traverse_mask = bioavailable_data != 'none'
        
        capabilities_reptile_traverse[reptile_traverse_mask] = 'traversable'
        capabilities_reptile[reptile_traverse_mask] = 'traverse'
        print(f"    Reptile traverse points: {np.sum(reptile_traverse_mask):,}")
    else:
        print("    'search_bioavailable' not found in point data")
    
    # Reptile Forage: points in 'search_bioavailable' == 'low-vegetation' OR points in 'stat_dead branch' > 0 OR stat_epiphyte > 0
    reptile_forage_mask = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Check low-vegetation points
    if 'search_bioavailable' in vtk_data.point_data:
        low_veg_mask = vtk_data.point_data['search_bioavailable'] == 'low-vegetation'
        reptile_forage_low_veg |= low_veg_mask
        reptile_forage_mask |= low_veg_mask
        capabilities_reptile_forage[low_veg_mask] = 'ground cover'
        print(f"    Reptile forage low-vegetation points: {np.sum(low_veg_mask):,}")
    
    # Check dead branch points
    if 'stat_dead branch' in vtk_data.point_data:
        dead_branch_data = vtk_data.point_data['stat_dead branch']
        if np.issubdtype(dead_branch_data.dtype, np.number):
            dead_branch_mask = dead_branch_data > 0
        else:
            dead_branch_mask = (dead_branch_data != 'none') & (dead_branch_data != '') & (dead_branch_data != 'nan')
        
        reptile_forage_dead_branch |= dead_branch_mask
        reptile_forage_mask |= dead_branch_mask
        capabilities_reptile_forage[dead_branch_mask] = 'dead branch'
        print(f"    Reptile forage dead branch points: {np.sum(dead_branch_mask):,}")
    else:
        print("    'stat_dead branch' not found in point data")
    
    # Check epiphyte points
    if 'stat_epiphyte' in vtk_data.point_data:
        epiphyte_data = vtk_data.point_data['stat_epiphyte']
        if np.issubdtype(epiphyte_data.dtype, np.number):
            epiphyte_mask = epiphyte_data > 0
        else:
            epiphyte_mask = (epiphyte_data != 'none') & (epiphyte_data != '') & (epiphyte_data != 'nan')
        
        reptile_forage_epiphyte |= epiphyte_mask
        reptile_forage_mask |= epiphyte_mask
        capabilities_reptile_forage[epiphyte_mask] = 'epiphyte'
        print(f"    Reptile forage epiphyte points: {np.sum(epiphyte_mask):,}")
    else:
        print("    'stat_epiphyte' not found in point data")
    
    # Update aggregate forage mask
    reptile_forage |= reptile_forage_mask
    
    # Override any existing values (later capabilities take precedence)
    capabilities_reptile[reptile_forage_mask] = 'forage'
    print(f"    Reptile forage total points: {np.sum(reptile_forage_mask):,}")
    
    # Reptile Shelter: points in 'stat_fallen log' > 0 OR forest_size == 'fallen'
    reptile_shelter_mask = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Check fallen log points
    if 'stat_fallen log' in vtk_data.point_data:
        fallen_log_data = vtk_data.point_data['stat_fallen log']
        if np.issubdtype(fallen_log_data.dtype, np.number):
            fallen_log_mask = fallen_log_data > 0
        else:
            fallen_log_mask = (fallen_log_data != 'none') & (fallen_log_data != '') & (fallen_log_data != 'nan')
        
        reptile_shelter_fallen_log |= fallen_log_mask
        reptile_shelter_mask |= fallen_log_mask
        capabilities_reptile_shelter[fallen_log_mask] = 'fallen log'
        print(f"    Reptile shelter fallen log points: {np.sum(fallen_log_mask):,}")
    else:
        print("    'stat_fallen log' not found in point data")
    
    # Check fallen tree points
    if 'forest_size' in vtk_data.point_data:
        forest_size = vtk_data.point_data['forest_size']
        if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':  # String types
            fallen_tree_mask = (forest_size == 'fallen')
        else:
            fallen_tree_mask = np.zeros(vtk_data.n_points, dtype=bool)  # No fallen trees if numeric
        
        reptile_shelter_fallen_tree |= fallen_tree_mask
        reptile_shelter_mask |= fallen_tree_mask
        capabilities_reptile_shelter[fallen_tree_mask] = 'fallen tree'
        print(f"    Reptile shelter fallen tree points: {np.sum(fallen_tree_mask):,}")
    else:
        print("    'forest_size' not found in point data")
    
    # Update aggregate shelter mask
    reptile_shelter |= reptile_shelter_mask
    
    # Override any existing values (later capabilities take precedence)
    capabilities_reptile[reptile_shelter_mask] = 'shelter'
    print(f"    Reptile shelter total points: {np.sum(reptile_shelter_mask):,}")
    
    # Add detailed component layers
    vtk_data.point_data['capabilities-reptile-forage-low-veg'] = reptile_forage_low_veg
    vtk_data.point_data['capabilities-reptile-forage-dead-branch'] = reptile_forage_dead_branch
    vtk_data.point_data['capabilities-reptile-forage-epiphyte'] = reptile_forage_epiphyte
    
    vtk_data.point_data['capabilities-reptile-shelter-fallen-log'] = reptile_shelter_fallen_log
    vtk_data.point_data['capabilities-reptile-shelter-fallen-tree'] = reptile_shelter_fallen_tree
    
    # Add capability layers
    vtk_data.point_data['capabilities-reptile-traverse'] = capabilities_reptile_traverse
    vtk_data.point_data['capabilities-reptile-forage'] = capabilities_reptile_forage
    vtk_data.point_data['capabilities-reptile-shelter'] = capabilities_reptile_shelter
    
    # Add overall aggregate layer
    vtk_data.point_data['capabilities-reptile'] = capabilities_reptile
    
    return vtk_data

def create_tree_capabilities(vtk_data):
    print("  Creating tree capability layers...")
    
    # Initialize capability arrays
    tree_grow = np.zeros(vtk_data.n_points, dtype=bool)
    tree_age = np.zeros(vtk_data.n_points, dtype=bool)
    tree_persist = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize component arrays
    tree_persist_near_medium = np.zeros(vtk_data.n_points, dtype=bool)
    tree_persist_near_large = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize string arrays for detailed capability types
    capabilities_tree = np.full(vtk_data.n_points, 'none', dtype='<U20')
    tree_grow_attr = np.full(vtk_data.n_points, 'none', dtype='<U20')
    tree_age_attr = np.full(vtk_data.n_points, 'none', dtype='<U20')
    tree_persist_attr = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Tree Grow: points in 'stat_other' > 0
    if 'stat_other' in vtk_data.point_data:
        other_data = vtk_data.point_data['stat_other']
        if np.issubdtype(other_data.dtype, np.number):
            tree_grow_mask = other_data > 0
        else:
            tree_grow_mask = (other_data != 'none') & (other_data != '') & (other_data != 'nan')
        
        tree_grow |= tree_grow_mask
        tree_grow_attr[tree_grow_mask] = 'volume'
        capabilities_tree[tree_grow_mask] = 'grow'
        print(f"    Tree grow points: {np.sum(tree_grow_mask):,}")
    else:
        print("    'stat_other' not found in point data")
    
    # Tree Age: points in 'improved-tree' OR 'reserve-tree'
    if 'search_design_action' in vtk_data.point_data:
        design_action = vtk_data.point_data['search_design_action']
        tree_age_mask = design_action == 'improved-tree'
        
        tree_age |= tree_age_mask
        tree_age_attr[tree_age_mask] = 'improved tree'
        # Override any existing values (later capabilities take precedence)
        capabilities_tree[tree_age_mask] = 'age'
        print(f"    Tree age points from improved-tree: {np.sum(tree_age_mask):,}")
    else:
        print("    'search_design_action' not found in point data")
    
    # Add the second condition: forest_control == 'reserve-tree'
    if 'forest_control' in vtk_data.point_data:
        forest_control = vtk_data.point_data['forest_control']
        reserve_tree_mask = forest_control == 'reserve-tree'
        
        tree_age |= reserve_tree_mask
        tree_age_attr[reserve_tree_mask] = 'reserve tree'
        # Override any existing values (later capabilities take precedence)
        capabilities_tree[reserve_tree_mask] = 'age'
        print(f"    Tree age points from reserve-tree: {np.sum(reserve_tree_mask):,}")
    else:
        print("    'forest_control' not found in point data")
    
    # Changed approach for tree persist - now using scenario_rewildingPlantings
    if 'scenario_rewildingPlantings' in vtk_data.point_data:
        rewilding_data = vtk_data.point_data['scenario_rewildingPlantings']
        if np.issubdtype(rewilding_data.dtype, np.number):
            tree_persist_mask = rewilding_data >= 1
        else:
            # If not numeric, try to handle string representation
            try:
                # Convert string representation to numbers if possible
                numeric_values = np.zeros_like(rewilding_data, dtype=float)
                for i, val in enumerate(rewilding_data):
                    try:
                        if val not in ['none', '', 'nan']:
                            numeric_values[i] = float(val)
                    except ValueError:
                        pass
                tree_persist_mask = numeric_values >= 1
            except:
                # Fallback if conversion fails
                tree_persist_mask = np.zeros(vtk_data.n_points, dtype=bool)
                print("    Unable to convert scenario_rewildingPlantings to numeric values")
        
        tree_persist |= tree_persist_mask
        tree_persist_attr[tree_persist_mask] = 'eligible soil'
        # Override any existing values (later capabilities take precedence)
        capabilities_tree[tree_persist_mask] = 'persist'
        print(f"    Tree persist points from eligible soil: {np.sum(tree_persist_mask):,}")
    else:
        print("    'scenario_rewildingPlantings' not found in point data")
        
        # Fallback to old method if scenario_rewildingPlantings is not available
        print("    Falling back to proximity-based persist calculation")
        
        if 'search_bioavailable' in vtk_data.point_data and 'forest_size' in vtk_data.point_data:
            bioavailable = vtk_data.point_data['search_bioavailable']
            forest_size = vtk_data.point_data['forest_size']
            
            # Find traversable points
            traversable_mask = bioavailable == 'traversable'
            traversable_points = np.where(traversable_mask)[0]
            
            if len(traversable_points) > 0:
                # Find medium and large tree points
                if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':  # String types
                    medium_tree_mask = (forest_size == 'medium')
                    large_tree_mask = (forest_size == 'large')
                else:  # Numeric types
                    medium_tree_mask = np.zeros(vtk_data.n_points, dtype=bool)
                    large_tree_mask = np.zeros(vtk_data.n_points, dtype=bool)
                
                medium_tree_points = np.where(medium_tree_mask)[0]
                large_tree_points = np.where(large_tree_mask)[0]
                
                # Get coordinates of all points
                points = vtk_data.points
                
                # Process medium trees if any exist
                if len(medium_tree_points) > 0:
                    # Build KDTree for medium tree points
                    medium_tree_coords = points[medium_tree_points]
                    medium_tree = cKDTree(medium_tree_coords)
                    
                    # Query traversable points against medium tree points
                    traversable_coords = points[traversable_points]
                    distances, _ = medium_tree.query(traversable_coords, k=1, distance_upper_bound=1.0)
                    
                    # Find traversable points within 1m of medium trees
                    near_medium_indices = traversable_points[distances < 1.0]
                    tree_persist_near_medium[near_medium_indices] = True
                    tree_persist_mask = np.zeros(vtk_data.n_points, dtype=bool)
                    tree_persist_mask[near_medium_indices] = True
                    tree_persist |= tree_persist_mask
                    tree_persist_attr[near_medium_indices] = 'near medium tree'
                    print(f"    Tree persist points near medium trees: {len(near_medium_indices):,}")
                
                # Process large trees if any exist
                if len(large_tree_points) > 0:
                    # Build KDTree for large tree points
                    large_tree_coords = points[large_tree_points]
                    large_tree = cKDTree(large_tree_coords)
                    
                    # Query traversable points against large tree points
                    traversable_coords = points[traversable_points]
                    distances, _ = large_tree.query(traversable_coords, k=1, distance_upper_bound=1.0)
                    
                    # Find traversable points within 1m of large trees
                    near_large_indices = traversable_points[distances < 1.0]
                    tree_persist_near_large[near_large_indices] = True
                    tree_persist_mask = np.zeros(vtk_data.n_points, dtype=bool)
                    tree_persist_mask[near_large_indices] = True
                    tree_persist |= tree_persist_mask
                    tree_persist_attr[near_large_indices] = 'near large tree'
                    print(f"    Tree persist points near large trees: {len(near_large_indices):,}")
            else:
                print("    No traversable points found for tree persist capability")
        else:
            print("    'search_bioavailable' or 'forest_size' not found in point data")
    
    # Add tree capability layers to vtk_data
    vtk_data.point_data['capabilities-tree-grow'] = tree_grow
    vtk_data.point_data['capabilities-tree-age'] = tree_age
    vtk_data.point_data['capabilities-tree-persist'] = tree_persist
    
    # Add detailed component layers
    vtk_data.point_data['capabilities-tree-persist-near-medium'] = tree_persist_near_medium
    vtk_data.point_data['capabilities-tree-persist-near-large'] = tree_persist_near_large
    
    # Add string attribute layers
    vtk_data.point_data['capabilities-tree-grow-attr'] = tree_grow_attr
    vtk_data.point_data['capabilities-tree-age-attr'] = tree_age_attr
    vtk_data.point_data['capabilities-tree-persist-attr'] = tree_persist_attr
    
    vtk_data.point_data['capabilities-tree'] = capabilities_tree
    
    return vtk_data

def collect_capability_stats(vtk_data):
    """Collect statistics on capabilities"""
    stats = {}
    
    # Find all capability layers
    capability_keys = [key for key in vtk_data.point_data.keys() if key.startswith('capabilities-')]
    
    # Collect statistics for each capability layer
    for key in capability_keys:
        data = vtk_data.point_data[key]
        if np.issubdtype(data.dtype, np.bool_):
            # Boolean array
            true_count = np.sum(data)
            stats[key] = true_count
        else:
            # String array
            non_none_count = np.sum(data != 'none')
            stats[key] = non_none_count
            
            # Collect counts for each unique value
            values, counts = np.unique(data[data != 'none'], return_counts=True)
            for value, count in zip(values, counts):
                stats[f"{key}-{value}"] = count
    
    return stats

def load_vtk_file(site, scenario, voxel_size, year):
    """Load VTK file with error handling"""
    path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk'
    try:
        if Path(path).exists():
            return pv.read(path)
        else:
            print(f"VTK file not found: {path}")
            return None
    except Exception as e:
        print(f"Error loading VTK file {path}: {e}")
        return None

def converted_urban_element_counts(site, scenario, year, vtk_data, tree_df=None):
    """Generate counts of converted urban elements for different capabilities
    
    Args:
        site (str): Site name
        scenario (str): Scenario name
        year (str): Year/timestep
        vtk_data (pyvista.UnstructuredGrid): VTK data with capability information
        tree_df (pd.DataFrame, optional): Tree dataframe for df-based counts
        
    Returns:
        pd.DataFrame: DataFrame containing all counts with columns:
        [site, scenario, timestep, persona, capability, countname, countelement, count]
    """
    # Initialize empty list to store count records
    count_records = []
    
    # Convert year to string for consistency
    year_str = str(year)
    
    #---------------------------------------------------------------------------
    # 1. BIRD CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    
    # 1.1 Bird socialise: canopy volume across control levels
    if 'maskForTrees' in vtk_data.point_data and 'forest_control' in vtk_data.point_data:
        mask_for_trees = vtk_data.point_data['maskForTrees']
        forest_control = vtk_data.point_data['forest_control']
        
        # Define the control level mappings
        control_levels = {
            'high': ['street-tree'],
            'medium': ['park-tree'],
            'low': ['reserve-tree', 'improved-tree']
        }
        
        # Count voxels for each control level
        for level, control_types in control_levels.items():
            for control_type in control_types:
                mask = (mask_for_trees == 1) & (forest_control == control_type)
                count = np.sum(mask)
                
                count_records.append({
                    'site': site,
                    'scenario': scenario,
                    'timestep': year_str,
                    'persona': 'bird',
                    'capability': 'socialise',
                    'countname': 'canopy_volume',
                    'countelement': level,
                    'count': int(count)
                })
    
    # 1.2 Bird feed: artificial bark installations
    if 'stat_epiphyte' in vtk_data.point_data and 'precolonial' in vtk_data.point_data:
        stat_epiphyte = vtk_data.point_data['stat_epiphyte']
        precolonial = vtk_data.point_data['precolonial']
        
        # Handle numeric/non-numeric epiphyte data
        if np.issubdtype(stat_epiphyte.dtype, np.number):
            epiphyte_mask = stat_epiphyte > 0
        else:
            epiphyte_mask = (stat_epiphyte != 'none') & (stat_epiphyte != '') & (stat_epiphyte != 'nan')
        
        # Handle boolean/non-boolean precolonial data
        if precolonial.dtype.kind in ['b', 'B']:
            precolonial_mask = precolonial
        else:
            precolonial_mask = np.array([str(x).lower() == 'true' for x in precolonial])
        
        # Count artificial epiphytes (not precolonial)
        mask = epiphyte_mask & (~precolonial_mask)
        count = np.sum(mask)
        
        count_records.append({
            'site': site,
            'scenario': scenario,
            'timestep': year_str,
            'persona': 'bird',
            'capability': 'feed',
            'countname': 'artificial_bark',
            'countelement': 'installed',
            'count': int(count)
        })
    
    # 1.3 Bird raise young: artificial hollows
    if 'stat_hollow' in vtk_data.point_data and 'precolonial' in vtk_data.point_data:
        stat_hollow = vtk_data.point_data['stat_hollow']
        precolonial = vtk_data.point_data['precolonial']
        
        # Handle numeric/non-numeric hollow data
        if np.issubdtype(stat_hollow.dtype, np.number):
            hollow_mask = stat_hollow > 0
        else:
            hollow_mask = (stat_hollow != 'none') & (stat_hollow != '') & (stat_hollow != 'nan')
        
        # Handle boolean/non-boolean precolonial data
        if precolonial.dtype.kind in ['b', 'B']:
            precolonial_mask = precolonial
        else:
            precolonial_mask = np.array([str(x).lower() == 'true' for x in precolonial])
        
        # Count artificial hollows (not precolonial)
        mask = hollow_mask & (~precolonial_mask)
        count = np.sum(mask)
        
        count_records.append({
            'site': site,
            'scenario': scenario,
            'timestep': year_str,
            'persona': 'bird',
            'capability': 'raise-young',
            'countname': 'artificial_hollows',
            'countelement': 'installed',
            'count': int(count)
        })
    
    #---------------------------------------------------------------------------
    # 2. REPTILE CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    
    # 2.1 Reptile traverse: urban elements conversion
    if 'capabilities-reptile-traverse' in vtk_data.point_data and 'search_urban_elements' in vtk_data.point_data:
        reptile_traverse = vtk_data.point_data['capabilities-reptile-traverse']
        urban_elements = vtk_data.point_data['search_urban_elements']
        
        # Define urban element categories
        urban_element_types = [
            'open space', 'green roof', 'brown roof', 'facade', 
            'roadway', 'busy roadway', 'existing conversion', 
            'other street potential', 'parking'
        ]
        
        # Count traversable voxels for each urban element type
        for element_type in urban_element_types:
            mask = (reptile_traverse == 'traversable') & (urban_elements == element_type)
            count = np.sum(mask)
            
            count_records.append({
                'site': site,
                'scenario': scenario,
                'timestep': year_str,
                'persona': 'reptile',
                'capability': 'traverse',
                'countname': 'urban_conversion',
                'countelement': element_type.replace(' ', '_'),
                'count': int(count)
            })
    
    # 2.2 Reptile forage: mistletoe installations (same as bird feed artificial bark)
    if 'stat_epiphyte' in vtk_data.point_data and 'precolonial' in vtk_data.point_data:
        stat_epiphyte = vtk_data.point_data['stat_epiphyte']
        precolonial = vtk_data.point_data['precolonial']
        
        # Handle numeric/non-numeric epiphyte data
        if np.issubdtype(stat_epiphyte.dtype, np.number):
            epiphyte_mask = stat_epiphyte > 0
        else:
            epiphyte_mask = (stat_epiphyte != 'none') & (stat_epiphyte != '') & (stat_epiphyte != 'nan')
        
        # Handle boolean/non-boolean precolonial data
        if precolonial.dtype.kind in ['b', 'B']:
            precolonial_mask = precolonial
        else:
            precolonial_mask = np.array([str(x).lower() == 'true' for x in precolonial])
        
        # Count mistletoe installations (not precolonial)
        mask = epiphyte_mask & (~precolonial_mask)
        count = np.sum(mask)
        
        count_records.append({
            'site': site,
            'scenario': scenario,
            'timestep': year_str,
            'persona': 'reptile',
            'capability': 'forage',
            'countname': 'mistletoe',
            'countelement': 'installed',
            'count': int(count)
        })
    
    # 2.3 Reptile shelter: proximity to fallen logs/trees
    if ('search_urban_elements' in vtk_data.point_data and 
        ('stat_fallen log' in vtk_data.point_data or 'forest_size' in vtk_data.point_data)):
        
        # Get urban elements data
        urban_elements = vtk_data.point_data['search_urban_elements']
        
        # Define urban element categories
        urban_element_types = [
            'open space', 'green roof', 'brown roof', 'facade', 
            'roadway', 'busy roadway', 'existing conversion', 
            'other street potential', 'parking'
        ]
        
        # Create a mask for fallen logs and trees
        fallen_mask = np.zeros(vtk_data.n_points, dtype=bool)
        
        # Check for fallen logs
        if 'stat_fallen log' in vtk_data.point_data:
            fallen_log = vtk_data.point_data['stat_fallen log']
            if np.issubdtype(fallen_log.dtype, np.number):
                fallen_mask |= (fallen_log > 0)
            else:
                fallen_mask |= ((fallen_log != 'none') & (fallen_log != '') & (fallen_log != 'nan'))
        
        # Check for fallen trees
        if 'forest_size' in vtk_data.point_data:
            forest_size = vtk_data.point_data['forest_size']
            if forest_size.dtype.kind in ['U', 'S']:  # String types
                fallen_mask |= (forest_size == 'fallen')
        
        # If we have any fallen logs/trees, find nearby points
        if np.any(fallen_mask):
            # Get coordinates of fallen log/tree points and all points
            all_points = vtk_data.points
            fallen_points = all_points[fallen_mask]
            
            # Build KDTree for fallen points
            if len(fallen_points) > 0:
                fallen_tree = cKDTree(fallen_points)
                
                # Find points within 5m of fallen logs/trees
                distances, _ = fallen_tree.query(all_points, k=1, distance_upper_bound=5.0)
                near_fallen_mask = distances < 5.0
                
                # Count nearby points for each urban element type
                for element_type in urban_element_types:
                    element_mask = urban_elements == element_type
                    count = np.sum(near_fallen_mask & element_mask)
                    
                    count_records.append({
                        'site': site,
                        'scenario': scenario,
                        'timestep': year_str,
                        'persona': 'reptile',
                        'capability': 'shelter',
                        'countname': 'near_fallen_5m',
                        'countelement': element_type.replace(' ', '_'),
                        'count': int(count)
                    })
    
    #---------------------------------------------------------------------------
    # 3. TREE CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    
    # 3.1 Tree grow: tree planting count
    if tree_df is not None and 'number_of_trees_to_plant' in tree_df.columns:
        total_trees_planted = tree_df['number_of_trees_to_plant'].sum()
        
        count_records.append({
            'site': site,
            'scenario': scenario,
            'timestep': year_str,
            'persona': 'tree',
            'capability': 'grow',
            'countname': 'trees_planted',
            'countelement': 'total',
            'count': int(total_trees_planted)
        })
    
    # 3.2 Tree age: AGE-IN-PLACE actions
    if tree_df is not None and 'rewilded' in tree_df.columns:
        # Define rewilding action types
        rewilding_types = ['footprint-depaved', 'exoskeleton', 'node-rewilded']
        
        # Count occurrences of each rewilding type
        for rwild_type in rewilding_types:
            count = sum(tree_df['rewilded'] == rwild_type)
            
            count_records.append({
                'site': site,
                'scenario': scenario,
                'timestep': year_str,
                'persona': 'tree',
                'capability': 'age',
                'countname': 'AGE-IN-PLACE_actions',
                'countelement': rwild_type,
                'count': int(count)
            })
    
    # 3.3 Tree persist: eligible soil in urban elements
    if 'scenario_rewildingPlantings' in vtk_data.point_data and 'search_urban_elements' in vtk_data.point_data:
        rewilding_plantings = vtk_data.point_data['scenario_rewildingPlantings']
        urban_elements = vtk_data.point_data['search_urban_elements']
        
        # Handle numeric/non-numeric rewilding plantings data
        if np.issubdtype(rewilding_plantings.dtype, np.number):
            plantings_mask = rewilding_plantings >= 1
        else:
            # Simple string conversion without extensive error handling
            plantings_mask = np.zeros(vtk_data.n_points, dtype=bool)
            for i, val in enumerate(rewilding_plantings):
                if val not in ['none', '', 'nan']:
                    try:
                        plantings_mask[i] = float(val) >= 1
                    except (ValueError, TypeError):
                        pass
        
        # Define urban element categories
        urban_element_types = [
            'open space', 'green roof', 'brown roof', 'facade', 
            'roadway', 'busy roadway', 'existing conversion', 
            'other street potential', 'parking'
        ]
        
        # Count eligible soil voxels for each urban element type
        for element_type in urban_element_types:
            mask = plantings_mask & (urban_elements == element_type)
            count = np.sum(mask)
            
            count_records.append({
                'site': site,
                'scenario': scenario,
                'timestep': year_str,
                'persona': 'tree',
                'capability': 'persist',
                'countname': 'eligible_soil',
                'countelement': element_type.replace(' ', '_'),
                'count': int(count)
            })
    
    # Convert records to DataFrame
    counts_df = pd.DataFrame(count_records)
    
    return counts_df

if __name__ == "__main__":
    main()