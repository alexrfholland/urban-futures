import numpy as np
import pandas as pd
import pyvista as pv
import argparse
from pathlib import Path
from scipy.spatial import cKDTree

"""
STRUCTURE OF CAPABILITIES:
    # Numeric indicator layers
    # name in polydata is f'capabilities_{persona}_{capability}_{numeric indicator}'
    # possible values are 0 and 1
    # example:
        polydata.point_data['capabilities_reptile_shelter_fallen-log'] can be 0 or 1

    # Capability layers
    # name in polydata is f'capabilities_{persona}_{capability}'
    # possible values are all the nummeric indicators for that capability
    # example:
        polydata.point_data['capabilities_reptile_shelter'] can be 'none', 'fallen-log', 'fallen-tree'
    
    # Persona aggregate capabilities layer
    # name in polydata is f'capabilities_{persona}'
    # possible values are all the capability layers for that persona
    # example:
        polydata.point_data['capabilities_reptile'] can be 'none', 'traverse', 'forage', 'shelter'

"""
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

"""
CAPABILITIES INFO    
1.  Bird capabilities:

1.1. Socialise: Points where 'stat_perch branch' > 0
   - Birds need branches to perch on for social activities
   
   Numeric indicators:
   - 1.1.1 bird_socialise: Total voxels where stat_perch branch > 0 : 'perch branch'
        -label for graph: 'Perchable canopy volume'
   
   Urban element / design action: 
   - 1.1.1 Canopy volume across control levels: high, medium, low
        # search criteria: Count of 'capabilities_bird_socialise_perch-branch', broken down by ['forest_control'], where high == 'street-tree', medium == 'park-tree', low == 'reserve-tree' OR 'improved-tree'

1.2 Feed: Points where 'stat_peeling bark' > 0
   - Birds feed on insects found under peeling bark
   
   Numeric indicators:
   - 1.2.1 bird_feed: Total voxels where stat_peeling bark > 0 : 'peeling bark'
        - label for graph: 'Peeling bark volume'
   
   Urban element / design action:
   - 1.2.1 Artificial bark installed on branches, utility poles
        #search criteria: Count of 'capabilities_bird_feed_peeling-bark' where polydata['precolonial'] == False
        TO DO: could include eucs

1.3. Raise Young: Points where 'stat_hollow' > 0
   - Birds need hollows in trees to nest and raise their young
   
   Numeric indicators:
   - 1.3.1 bird_raise_young: Total voxels where stat_hollow > 0 : 'hollow'
        - label for graph: 'Hollow count'

   Urban element / design action:
   - 1.3.1 Artificial hollows installed on branches, utility poles
        #search criteria: Count of 'capabilities_bird_raise-young_hollow' where polydata['precolonial'] == False
        TO DO: could include eucs


2. Reptile capabilities:

2.1. Traverse: Points where 'search_bioavailable' != 'none'
   - Reptiles can move through any bioavailable space
   
   Numeric indicators:
   - 2.1.1 reptile_traverse: Total voxels where search_bioavailable != 'none' : traversable
        - label for graph: 'Non-paved surface area'

   Urban element / design action: 
   - 2.1.1 Count of site voxels converted from: car parks, roads, green roofs, brown roofs, facades
        # search criteria: Total voxels where polydata['capabilities_reptile_traverse'] == 'traversable', 
        # broken down by the defined urban element catagories in polydata['search_urban_elements']


2.2 Foraige: Points where any of the following conditions are met:
   - 'search_bioavailable' == 'low-vegetation' (areas reptiles can move through)
   - 'stat_dead branch' > 0 (dead branches in canopy generate coarse woody debris)
   - 'stat_epiphyte' > 0 (epiphytes in canopy generate fallen leaves)
   
   Numeric indicators:
   - 2.2.1 reptile_forage_low_veg: Voxels where search_bioavailable == 'low-vegetation' : 'ground-cover'
         - label for graph: 'Low vegetation surface area'
   - 2.2.2 reptile_forage_dead_branch: Voxels where stat_dead branch > 0 : 'dead-branch'
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
   - 2.3.1 reptile_shelter_fallen_log: Voxels where stat_fallen log > 0 : 'fallen-log'
        - label for graph: 'Nurse log volume'
   - 2.3.2 reptile_shelter_fallen_tree: Voxels where forest_size == 'fallen' : 'fallen-tree'
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
   - 3.1.1 count of urban forest canopy co
        # search criteria: sum of df['number_of_trees_to_plant']      

3.2. Age: Points where 'forest_control' == 'improved-tree' OR 'forest_control' == 'reserve-tree'
   - Areas where trees are protected and can mature
   
   Numeric indicators:
   - 3.2.1 tree_age: Total voxels where forest_control == 'improved-tree' : 'improved-tree'
        - label for graph: 'Canopy volume supported by humans'
   - 3.2.2 tree_age: Total voxels where forest_control == 'reserve-tree' : 'reserve-tree'
        - label for graph: 'Canopy volume autonomous'

   #Urban element / design action: 
   -3.2.1 count of AGE-IN-PLACE actions: exoskeletons, habitat islands, depaved areas
        # search criteria:  counts of df['rewilded'] == 'footprint-depaved','exoskeleton','node-rewilded'
     -3.2.1 count of AGE-IN-PLACE actions: exoskeletons, habitat islands, depaved areas
        # search criteria:  counts of df['rewilded'] == 'footprint-depaved','exoskeleton','node-rewilded'

3.3. Persist: Terrain points eligible for new tree plantings (ie. depaved and unmanaged land away from trees)

    Numeric indicator:
    - 3.3.3 scenario_rewildingPlantings >= 1 : 'eligible-soil'
        - label for graph: 'Ground area for tree recruitment'

    #Urban element / design action:
    -3.3.3/ Count of site voxels converted from: car parks, roads, etc
        #count of subset of polydata['scenario_rewildingPlantings'] >= 1, broken down by the defined urban element catagories in polydata['search_urban_elements']

"""

"""
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
"""

# Add capability ID mapping dictionaries
CAPABILITY_ID_MAP = {
    # Bird (1.x.x)
    'capabilities_bird': '1',
    'capabilities_bird_socialise': '1.1',
    'capabilities_bird_socialise_perch-branch': '1.1.1',
    'capabilities_bird_feed': '1.2',
    'capabilities_bird_feed_peeling-bark': '1.2.1',
    'capabilities_bird_raise-young': '1.3',
    'capabilities_bird_raise-young_hollow': '1.3.1',
    
    # Reptile (2.x.x)
    'capabilities_reptile': '2',
    'capabilities_reptile_traverse': '2.1',
    'capabilities_reptile_traverse_traversable': '2.1.1',
    'capabilities_reptile_forage': '2.2',
    'capabilities_reptile_forage_ground-cover': '2.2.1',
    'capabilities_reptile_forage_low-veg': '2.2.1',
    'capabilities_reptile_forage_dead-branch': '2.2.2',
    'capabilities_reptile_forage_epiphyte': '2.2.3',
    'capabilities_reptile_shelter': '2.3',
    'capabilities_reptile_shelter_fallen-log': '2.3.1',
    'capabilities_reptile_shelter_fallen-tree': '2.3.2',
    
    # Tree (3.x.x)
    'capabilities_tree': '3',
    'capabilities_tree_grow': '3.1',
    'capabilities_tree_grow_volume': '3.1.1',
    'capabilities_tree_age': '3.2',
    'capabilities_tree_age_improved-tree': '3.2.1',
    'capabilities_tree_age_reserve-tree': '3.2.2',
    'capabilities_tree_persist': '3.3',
    'capabilities_tree_persist_eligible-soil': '3.3.3'
}

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
    # STEP 2: PROCESS CAPABILITIES AND SAVE RESULTS
    #--------------------------------------------------------------------------
    print("\n===== PROCESSING CAPABILITIES =====")
    
    # Create directory structure if it doesn't exist
    stats_dir = Path('data/revised/final/stats/arboreal-future-stats/data')
    raw_dir = stats_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {raw_dir}")
    
    # Process each site
    for site in sites:
        print(f"\n=== Processing site: {site} ===")
        
        # Initialize combined dataframes for this site
        all_capabilities_counts = []
        all_urban_elements_counts = []
        
        # Process baseline if requested
        if include_baseline:
            print(f"Processing baseline for site: {site}")
            baseline_path = f'data/revised/final/baselines/{site}_baseline_combined_{voxel_size}_urban_features.vtk'
            
            baseline_vtk = pv.read(baseline_path)
            print(f"  Creating capability layers for baseline...")

            # Make a scenario_rewildingPlantings in the baseline
            bioavailable = baseline_vtk.point_data['search_bioavailable']
            eligible_soil_mask = bioavailable == 'low-vegetation'
            baseline_vtk.point_data['scenario_rewildingPlantings'] = eligible_soil_mask

            # Create capabilities for baseline
            baseline_vtk = create_bird_capabilities(baseline_vtk)
            baseline_vtk = create_reptile_capabilities(baseline_vtk)
            baseline_vtk = create_tree_capabilities(baseline_vtk)
            
            # Collect statistics
            baseline_stats = collect_capability_stats(baseline_vtk)
            
            # Convert to dataframe rows
            for capability, count in baseline_stats.items():
                capability_id = CAPABILITY_ID_MAP.get(capability, 'NA')
                all_capabilities_counts.append({
                    'site': site,
                    'scenario': 'baseline',
                    'voxel_size': voxel_size,
                    'capability': capability,
                    'timestep': 'baseline',
                    'count': count,
                    'capabilityID': capability_id
                })
            
            # Save updated baseline with capabilities
            baseline_output_path = f'data/revised/final/{site}/{site}_baseline_resources_{voxel_size}_with_capabilities.vtk'
            baseline_vtk.save(baseline_output_path)
            print(f"  Saved baseline with capabilities to {baseline_output_path}")
        
        # Process each scenario
        for scenario in scenarios:
            print(f"\n=== Processing scenario: {scenario} for site: {site} ===")
            
            # Process each year
            for year in years:
                print(f"Processing year {year} for site: {site}, scenario: {scenario}")
                
                # Load VTK file
                vtk_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk'
                vtk_data = pv.read(vtk_path)
                
                # Load tree dataframe
                tree_df_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv'
                tree_df = None
                if Path(tree_df_path).exists():
                    tree_df = pd.read_csv(tree_df_path)
                    print(f"  Loaded tree dataframe from {tree_df_path}")
                
                # Create capabilities
                print(f"  Creating capability layers...")
                vtk_data = create_bird_capabilities(vtk_data)
                vtk_data = create_reptile_capabilities(vtk_data)
                vtk_data = create_tree_capabilities(vtk_data)
                
                # Collect capability statistics
                year_capability_stats = collect_capability_stats(vtk_data)
                
                # Convert to dataframe rows
                for capability, count in year_capability_stats.items():
                    capability_id = CAPABILITY_ID_MAP.get(capability, 'NA')
                    all_capabilities_counts.append({
                        'site': site,
                        'scenario': scenario,
                        'voxel_size': voxel_size,
                        'capability': capability,
                        'timestep': str(year),
                        'count': count,
                        'capabilityID': capability_id
                    })
                
                # Generate urban element counts
                urban_element_counts = converted_urban_element_counts(
                    site=site,
                    scenario=scenario,
                    year=year,
                    vtk_data=vtk_data,
                    tree_df=tree_df
                )
                
                # Add to combined urban elements counts
                if not urban_element_counts.empty:
                    all_urban_elements_counts.append(urban_element_counts)
                
                # Save updated VTK file
                output_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_capabilities.vtk'
                vtk_data.save(output_path)
                print(f"  Saved VTK with capabilities to {output_path}")
        
        # Convert to dataframes and save for this site
        capabilities_count_df = pd.DataFrame(all_capabilities_counts)
        capabilities_count_df['capabilityID'] = capabilities_count_df['capabilityID'].astype(str)
        
        # Save capabilities counts for this site (all scenarios combined)
        capabilities_path = raw_dir / f'{site}_all_scenarios_{voxel_size}_capabilities_counts.csv'
        capabilities_count_df.to_csv(capabilities_path, index=False)
        print(f"Saved capability statistics to {capabilities_path}")
        
        # Save urban element counts for this site (all scenarios combined)
        if all_urban_elements_counts:
            urban_elements_count_df = pd.concat(all_urban_elements_counts, ignore_index=True)
            counts_path = raw_dir / f'{site}_all_scenarios_{voxel_size}_urban_element_counts.csv'
            urban_elements_count_df.to_csv(counts_path, index=False)
            print(f"Saved urban element counts to {counts_path}")
    
    print("\n===== All processing completed =====")


def collect_capability_stats(vtk_data):
    """Collect statistics on capabilities"""
    stats = {}
    
    # Find all capability layers
    capability_keys = [key for key in vtk_data.point_data.keys() if key.startswith('capabilities_')]
    
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
                # Make sure to create a normalized key for mapping
                if isinstance(value, str):
                    stats[f"{key}_{value}"] = count
    
    return stats

def create_bird_capabilities(vtk_data):
    print("  Creating bird capability layers...")
    
    # Initialize capability_numeric_indicator layers (boolean arrays)
    bird_socialise_perch_branch = np.zeros(vtk_data.n_points, dtype=bool)
    bird_feed_peeling_bark = np.zeros(vtk_data.n_points, dtype=bool)
    bird_raise_young_hollow = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize individual_capability layers (string arrays)
    capabilities_bird_socialise = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_bird_feed = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_bird_raise_young = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Initialize persona_aggregate_capabilities layer
    capabilities_bird = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # 1.1. Bird Socialise: points in stat_perch_branch > 0
    # - Birds need branches to perch on for social activities
    perch_data = vtk_data.point_data['stat_perch branch']
    if np.issubdtype(perch_data.dtype, np.number):
        perch_branch_mask = perch_data > 0
    else:
        perch_branch_mask = (perch_data != 'none') & (perch_data != '') & (perch_data != 'nan')
    
    # 1.1.1 capability_numeric_indicator: perch branch
    bird_socialise_perch_branch |= perch_branch_mask
    capabilities_bird_socialise[perch_branch_mask] = 'perch-branch'
    capabilities_bird[perch_branch_mask] = 'socialise'
    print(f"    Bird socialise points: {np.sum(perch_branch_mask):,}")
    
    # 1.2. Bird Feed: points in stat_peeling bark > 0
    # - Birds feed on insects found under peeling bark
    bark_data = vtk_data.point_data['stat_peeling bark']
    if np.issubdtype(bark_data.dtype, np.number):
        peeling_bark_mask = bark_data > 0
    else:
        peeling_bark_mask = (bark_data != 'none') & (bark_data != '') & (bark_data != 'nan')
    
    # 1.2.1 capability_numeric_indicator: peeling bark
    bird_feed_peeling_bark |= peeling_bark_mask
    capabilities_bird_feed[peeling_bark_mask] = 'peeling-bark'
    # Override any existing values (later capabilities take precedence)
    capabilities_bird[peeling_bark_mask] = 'feed'
    print(f"    Bird feed points: {np.sum(peeling_bark_mask):,}")
    
    # 1.3. Bird Raise Young: points in stat_hollow > 0
    # - Birds need hollows in trees to nest and raise their young
    hollow_data = vtk_data.point_data['stat_hollow']
    if np.issubdtype(hollow_data.dtype, np.number):
        hollow_mask = hollow_data > 0
    else:
        hollow_mask = (hollow_data != 'none') & (hollow_data != '') & (hollow_data != 'nan')
    
    # 1.3.1 capability_numeric_indicator: hollow
    bird_raise_young_hollow |= hollow_mask
    capabilities_bird_raise_young[hollow_mask] = 'hollow'
    # Override any existing values (later capabilities take precedence)
    capabilities_bird[hollow_mask] = 'raise-young'
    print(f"    Bird raise-young points: {np.sum(hollow_mask):,}")
    
    # Add capability_numeric_indicator layers to vtk_data
    vtk_data.point_data['capabilities_bird_socialise_perch-branch'] = bird_socialise_perch_branch
    vtk_data.point_data['capabilities_bird_feed_peeling-bark'] = bird_feed_peeling_bark
    vtk_data.point_data['capabilities_bird_raise-young_hollow'] = bird_raise_young_hollow
    
    # Add individual_capability layers to vtk_data
    vtk_data.point_data['capabilities_bird_socialise'] = capabilities_bird_socialise
    vtk_data.point_data['capabilities_bird_feed'] = capabilities_bird_feed
    vtk_data.point_data['capabilities_bird_raise-young'] = capabilities_bird_raise_young
    
    # Add persona_aggregate_capabilities layer to vtk_data
    vtk_data.point_data['capabilities_bird'] = capabilities_bird
    
    return vtk_data

def create_reptile_capabilities(vtk_data):
    print("  Creating reptile capability layers...")
    
    # Initialize capability_numeric_indicator layers (boolean arrays)
    reptile_traverse_traversable = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_forage_ground_cover = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_forage_dead_branch = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_forage_epiphyte = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_shelter_fallen_log = np.zeros(vtk_data.n_points, dtype=bool)
    reptile_shelter_fallen_tree = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize individual_capability layers (string arrays)
    capabilities_reptile_traverse = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_reptile_forage = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_reptile_shelter = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Initialize persona_aggregate_capabilities layer
    capabilities_reptile = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # 2.1. Reptile Traverse: points in 'search_bioavailable' != 'none'
    # - Reptiles can move through any bioavailable space
    bioavailable_data = vtk_data.point_data['search_bioavailable']
    traversable_mask = bioavailable_data != 'none'
    
    # 2.1.1 capability_numeric_indicator: traversable
    reptile_traverse_traversable |= traversable_mask
    capabilities_reptile_traverse[traversable_mask] = 'traversable'
    capabilities_reptile[traversable_mask] = 'traverse'
    print(f"    Reptile traverse points: {np.sum(traversable_mask):,}")
    
    # 2.2. Reptile Forage: points with multiple conditions
    
    # 2.2.1 Low vegetation points - ground cover
    ground_cover_mask = vtk_data.point_data['search_bioavailable'] == 'low-vegetation'
    reptile_forage_ground_cover |= ground_cover_mask
    capabilities_reptile_forage[ground_cover_mask] = 'ground-cover'
    # Override any existing values in capabilities_reptile if needed
    capabilities_reptile[ground_cover_mask] = 'forage'
    print(f"    Reptile forage ground cover points: {np.sum(ground_cover_mask):,}")
    
    # 2.2.2 Dead branch points
    dead_branch_data = vtk_data.point_data['stat_dead branch']
    if np.issubdtype(dead_branch_data.dtype, np.number):
        dead_branch_mask = dead_branch_data > 0
    else:
        dead_branch_mask = (dead_branch_data != 'none') & (dead_branch_data != '') & (dead_branch_data != 'nan')
    
    reptile_forage_dead_branch |= dead_branch_mask
    capabilities_reptile_forage[dead_branch_mask] = 'dead-branch'
    # Override any existing values in capabilities_reptile if needed
    capabilities_reptile[dead_branch_mask] = 'forage'
    print(f"    Reptile forage dead branch points: {np.sum(dead_branch_mask):,}")
    
    # 2.2.3 Epiphyte points
    epiphyte_data = vtk_data.point_data['stat_epiphyte']
    if np.issubdtype(epiphyte_data.dtype, np.number):
        epiphyte_mask = epiphyte_data > 0
    else:
        epiphyte_mask = (epiphyte_data != 'none') & (epiphyte_data != '') & (epiphyte_data != 'nan')
    
    reptile_forage_epiphyte |= epiphyte_mask
    capabilities_reptile_forage[epiphyte_mask] = 'epiphyte'
    # Override any existing values in capabilities_reptile if needed
    capabilities_reptile[epiphyte_mask] = 'forage'
    print(f"    Reptile forage epiphyte points: {np.sum(epiphyte_mask):,}")
    
    # 2.3.1 Fallen log points
    fallen_log_data = vtk_data.point_data['stat_fallen log']
    if np.issubdtype(fallen_log_data.dtype, np.number):
        fallen_log_mask = fallen_log_data > 0
    else:
        fallen_log_mask = (fallen_log_data != 'none') & (fallen_log_data != '') & (fallen_log_data != 'nan')
    
    reptile_shelter_fallen_log |= fallen_log_mask
    capabilities_reptile_shelter[fallen_log_mask] = 'fallen-log'
    # Override any existing values in capabilities_reptile if needed
    capabilities_reptile[fallen_log_mask] = 'shelter'
    print(f"    Reptile shelter fallen log points: {np.sum(fallen_log_mask):,}")
    
    # 2.3.2 Fallen tree points
    forest_size = vtk_data.point_data['forest_size']
    if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':  # String types
        fallen_tree_mask = (forest_size == 'fallen')
    else:
        fallen_tree_mask = np.zeros(vtk_data.n_points, dtype=bool)  # No fallen trees if numeric
    
    reptile_shelter_fallen_tree |= fallen_tree_mask
    capabilities_reptile_shelter[fallen_tree_mask] = 'fallen-tree'
    # Override any existing values in capabilities_reptile if needed
    capabilities_reptile[fallen_tree_mask] = 'shelter'
    print(f"    Reptile shelter fallen tree points: {np.sum(fallen_tree_mask):,}")
    
    # Add capability_numeric_indicator layers to vtk_data
    vtk_data.point_data['capabilities_reptile_traverse_traversable'] = reptile_traverse_traversable
    vtk_data.point_data['capabilities_reptile_forage_ground-cover'] = reptile_forage_ground_cover
    vtk_data.point_data['capabilities_reptile_forage_dead-branch'] = reptile_forage_dead_branch
    vtk_data.point_data['capabilities_reptile_forage_epiphyte'] = reptile_forage_epiphyte
    vtk_data.point_data['capabilities_reptile_shelter_fallen-log'] = reptile_shelter_fallen_log
    vtk_data.point_data['capabilities_reptile_shelter_fallen-tree'] = reptile_shelter_fallen_tree
    
    # Add individual_capability layers to vtk_data
    vtk_data.point_data['capabilities_reptile_traverse'] = capabilities_reptile_traverse
    vtk_data.point_data['capabilities_reptile_forage'] = capabilities_reptile_forage
    vtk_data.point_data['capabilities_reptile_shelter'] = capabilities_reptile_shelter
    
    # Add persona_aggregate_capabilities layer to vtk_data
    vtk_data.point_data['capabilities_reptile'] = capabilities_reptile
    
    return vtk_data

def create_tree_capabilities(vtk_data):
    print("  Creating tree capability layers...")
    
    # Initialize capability_numeric_indicator layers (boolean arrays)
    tree_grow_volume = np.zeros(vtk_data.n_points, dtype=bool)
    tree_age_improved_tree = np.zeros(vtk_data.n_points, dtype=bool)
    tree_age_reserve_tree = np.zeros(vtk_data.n_points, dtype=bool)
    tree_persist_eligible_soil = np.zeros(vtk_data.n_points, dtype=bool)
    
    # Initialize individual_capability layers (string arrays)
    capabilities_tree_grow = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_tree_age = np.full(vtk_data.n_points, 'none', dtype='<U20')
    capabilities_tree_persist = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # Initialize persona_aggregate_capabilities layer
    capabilities_tree = np.full(vtk_data.n_points, 'none', dtype='<U20')
    
    # 3.1. Tree Grow: points in 'stat_other' > 0
    # - Areas where trees can grow and establish
    other_data = vtk_data.point_data['stat_other']
    if np.issubdtype(other_data.dtype, np.number):
        volume_mask = other_data > 0
    else:
        volume_mask = (other_data != 'none') & (other_data != '') & (other_data != 'nan')
    
    # 3.1.1 capability_numeric_indicator: volume
    tree_grow_volume |= volume_mask
    capabilities_tree_grow[volume_mask] = 'volume'
    capabilities_tree[volume_mask] = 'grow'
    print(f"    Tree grow points: {np.sum(volume_mask):,}")
    
    # 3.2. Tree Age: points in 'improved-tree' OR 'reserve-tree'
    
    # 3.2.1 capability_numeric_indicator: improved tree
    design_action = vtk_data.point_data['forest_control']
    improved_tree_mask = design_action == 'improved-tree'
    
    tree_age_improved_tree |= improved_tree_mask
    capabilities_tree_age[improved_tree_mask] = 'improved-tree'
    # Override any existing values (later capabilities take precedence)
    capabilities_tree[improved_tree_mask] = 'age'
    print(f"    Tree age points from improved-tree: {np.sum(improved_tree_mask):,}")
    
    # 3.2.2 capability_numeric_indicator: reserve tree
    forest_control = vtk_data.point_data['forest_control']
    reserve_tree_mask = forest_control == 'reserve-tree'
    
    tree_age_reserve_tree |= reserve_tree_mask
    capabilities_tree_age[reserve_tree_mask] = 'reserve-tree'
    # Override any existing values (later capabilities take precedence)
    capabilities_tree[reserve_tree_mask] = 'age'
    print(f"    Tree age points from reserve-tree: {np.sum(reserve_tree_mask):,}")
    
    # 3.3.3 capability_numeric_indicator: eligible soil
    # Use rewilding plantings
    rewilding_data = vtk_data.point_data['scenario_rewildingPlantings']
    eligible_soil_mask = rewilding_data >= 1
    
    tree_persist_eligible_soil |= eligible_soil_mask
    capabilities_tree_persist[eligible_soil_mask] = 'eligible-soil'
    # Override any existing values (later capabilities take precedence)
    capabilities_tree[eligible_soil_mask] = 'persist'
    print(f"    Tree persist points from eligible soil (scenario): {np.sum(eligible_soil_mask):,}")
    
    # Add capability_numeric_indicator layers to vtk_data
    vtk_data.point_data['capabilities_tree_grow_volume'] = tree_grow_volume
    vtk_data.point_data['capabilities_tree_age_improved-tree'] = tree_age_improved_tree
    vtk_data.point_data['capabilities_tree_age_reserve-tree'] = tree_age_reserve_tree
    vtk_data.point_data['capabilities_tree_persist_eligible-soil'] = tree_persist_eligible_soil
    
    # Add individual_capability layers to vtk_data
    vtk_data.point_data['capabilities_tree_grow'] = capabilities_tree_grow
    vtk_data.point_data['capabilities_tree_age'] = capabilities_tree_age
    vtk_data.point_data['capabilities_tree_persist'] = capabilities_tree_persist
    
    # Add persona_aggregate_capabilities layer to vtk_data
    vtk_data.point_data['capabilities_tree'] = capabilities_tree
    
    return vtk_data


def converted_urban_element_counts(site, scenario, year, vtk_data, tree_df=None):
    """Generate counts of converted urban elements for different capabilities
    
    Args:
        site (str): Site name
        scenario (str): Scenario name
        year (int): Year/timestep
        vtk_data (pyvista.UnstructuredGrid): VTK data with capability information
        tree_df (pd.DataFrame, optional): Tree dataframe for df-based counts
        
    Returns:
        pd.DataFrame: DataFrame containing all counts with columns:
        [site, scenario, timestep, persona, capability, countname, countelement, count, capabilityID]
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
    def create_count_record(persona, capability, countname, countelement, count, capability_id):
        return {
            'site': site,
            'scenario': scenario,
            'timestep': year_str,
            'persona': persona,
            'capability': capability,
            'countname': countname,
            'countelement': countelement.replace(' ', '_'),
            'count': int(count),
            'capabilityID': capability_id
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
        all_points = vtk_data.points
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
    
    # 1.1 Bird socialise - Canopy volume across control levels
    if 'maskForTrees' in vtk_data.point_data and 'forest_control' in vtk_data.point_data:
        mask_for_trees = vtk_data.point_data['maskForTrees']
        forest_control = vtk_data.point_data['forest_control']
        
        # Create a mask for tree voxels
        tree_mask = (mask_for_trees == 1)
        control_level_counts = count_by_control_levels(tree_mask, forest_control, 'canopy_volume', 
                               'bird', 'socialise', '1.1.2')
        count_records.extend(control_level_counts)
        print(f"Added {len(control_level_counts)} bird socialise canopy volume records")
    
    # 1.2 Bird feed - Artificial bark installed
    if 'capabilities_bird_feed_peeling-bark' in vtk_data.point_data and 'precolonial' in vtk_data.point_data:
        peeling_bark = vtk_data.point_data['capabilities_bird_feed_peeling-bark']        
        precolonial_mask = vtk_data.point_data['precolonial']
        artificial_bark_mask = get_boolean_mask(peeling_bark) & (~precolonial_mask)
        
        count = np.sum(artificial_bark_mask)
        bark_record = create_count_record('bird', 'feed', 'artificial_bark', 'installed', count, '1.2.1')
        count_records.append(bark_record)
        print(f"Added bird feed artificial bark record")
        
        # Additional: If you also want to break down by urban elements
        if 'search_urban_elements' in vtk_data.point_data:
            urban_elements = vtk_data.point_data['search_urban_elements']
            bark_element_counts = count_by_urban_elements(artificial_bark_mask, urban_elements, 
                                 'artificial_bark_by_element', 'bird', 'feed', '1.2.1')
            count_records.extend(bark_element_counts)
            print(f"Added {len(bark_element_counts)} bird feed bark by urban element records")
    
    # 1.3 Bird raise young - Artificial hollows installed
    if 'capabilities_bird_raise-young_hollow' in vtk_data.point_data and 'precolonial' in vtk_data.point_data:
        hollow = vtk_data.point_data['capabilities_bird_raise-young_hollow']
        precolonial_mask = vtk_data.point_data['precolonial']
        artificial_hollow_mask = get_boolean_mask(hollow) & (~precolonial_mask)
        
        count = np.sum(artificial_hollow_mask)
        hollow_record = create_count_record('bird', 'raise-young', 'artificial_hollows', 'installed', count, '1.3.1')
        count_records.append(hollow_record)
        print(f"Added bird raise-young artificial hollows record")
        
        # Additional: If you also want to break down by urban elements
        if 'search_urban_elements' in vtk_data.point_data:
            urban_elements = vtk_data.point_data['search_urban_elements']
            hollow_element_counts = count_by_urban_elements(artificial_hollow_mask, urban_elements, 
                                   'artificial_hollows_by_element', 'bird', 'raise-young', '1.3.1')
            count_records.extend(hollow_element_counts)
            print(f"Added {len(hollow_element_counts)} bird raise-young hollows by urban element records")
    
    #---------------------------------------------------------------------------
    # 2. REPTILE CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    
    # 2.1 Reptile traverse - Count of site voxels converted from urban elements
    if 'capabilities_reptile_traverse_traversable' in vtk_data.point_data and 'search_urban_elements' in vtk_data.point_data:
        reptile_traverse = vtk_data.point_data['capabilities_reptile_traverse_traversable']
        urban_elements = vtk_data.point_data['search_urban_elements']
        
        traverse_counts = count_by_urban_elements(reptile_traverse, urban_elements, 'urban_conversion', 
                                                'reptile', 'traverse', '2.1.1')
        count_records.extend(traverse_counts)
        print(f"Added {len(traverse_counts)} reptile traverse by urban element records")
    
    # 2.2 Reptile forage
    
    # 2.2.1 Count of voxels converted from urban elements (low vegetation)
    if 'capabilities_reptile_forage_ground-cover' in vtk_data.point_data and 'search_urban_elements' in vtk_data.point_data:
        low_veg = vtk_data.point_data['capabilities_reptile_forage_ground-cover']
        urban_elements = vtk_data.point_data['search_urban_elements']
        
        low_veg_counts = count_by_urban_elements(low_veg, urban_elements, 'low_veg', 
                                               'reptile', 'forage', '2.2.1')
        count_records.extend(low_veg_counts)
        print(f"Added {len(low_veg_counts)} reptile forage low vegetation records")
    
    # 2.2.2 Dead branch volume across control levels
    if 'capabilities_reptile_forage_dead-branch' in vtk_data.point_data and 'forest_control' in vtk_data.point_data:
        dead_branch = vtk_data.point_data['capabilities_reptile_forage_dead-branch']
        forest_control = vtk_data.point_data['forest_control']
        
        dead_branch_counts = count_by_control_levels(dead_branch, forest_control, 'dead_branch', 
                                                   'reptile', 'forage', '2.2.2')
        count_records.extend(dead_branch_counts)
        print(f"Added {len(dead_branch_counts)} reptile forage dead branch by control level records")
    
    # 2.2.3 Number of epiphytes installed (mistletoe)
    if 'capabilities_reptile_forage_epiphyte' in vtk_data.point_data and 'precolonial' in vtk_data.point_data:
        epiphyte = vtk_data.point_data['capabilities_reptile_forage_epiphyte']
        precolonial_mask = vtk_data.point_data['precolonial']
        epiphyte_mask = get_boolean_mask(epiphyte) & (~precolonial_mask)
        
        count = np.sum(epiphyte_mask)
        epiphyte_record = create_count_record('reptile', 'forage', 'mistletoe', 'installed', count, '2.2.3')
        count_records.append(epiphyte_record)
        print(f"Added reptile forage epiphyte record")
    
    # 2.3 Reptile shelter
    # 2.3.1 and 2.3.2 Ground elements supporting fallen logs/trees
    if 'search_urban_elements' in vtk_data.point_data:
        urban_elements = vtk_data.point_data['search_urban_elements']
        
        # 2.3.1 Count of ground elements supporting fallen logs
        if 'capabilities_reptile_shelter_fallen-log' in vtk_data.point_data:
            fallen_log = vtk_data.point_data['capabilities_reptile_shelter_fallen-log']
            fallen_log_mask = get_boolean_mask(fallen_log)
            
            fallen_log_counts = count_near_features(fallen_log_mask, urban_elements, 'near_fallen_5m', 
                                                 '', 'reptile', 'shelter', '2.3.1')
            count_records.extend(fallen_log_counts)
            print(f"Added {len(fallen_log_counts)} reptile shelter fallen log proximity records")
        
        # 2.3.2 Count of ground elements supporting fallen trees
        if 'capabilities_reptile_shelter_fallen-tree' in vtk_data.point_data:
            fallen_tree = vtk_data.point_data['capabilities_reptile_shelter_fallen-tree']
            fallen_tree_mask = get_boolean_mask(fallen_tree)
            
            fallen_tree_counts = count_near_features(fallen_tree_mask, urban_elements, 'near_fallen_5m', 
                                                  '', 'reptile', 'shelter', '2.3.2')
            count_records.extend(fallen_tree_counts)
            print(f"Added {len(fallen_tree_counts)} reptile shelter fallen tree proximity records")
    
    #---------------------------------------------------------------------------
    # 3. TREE CAPABILITY COUNTS
    #---------------------------------------------------------------------------
    
    # 3.1 Tree grow - Count of number of trees planted this timestep
    if tree_df is not None and 'number_of_trees_to_plant' in tree_df.columns:
        total_trees_planted = tree_df['number_of_trees_to_plant'].sum()
        tree_planted_record = create_count_record('tree', 'grow', 'trees_planted', 'total', total_trees_planted, '3.1.1')
        count_records.append(tree_planted_record)
        print(f"Added tree grow planted trees record")
    
    # 3.2 Tree age - Count of AGE-IN-PLACE actions
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
    
    # 3.3 Tree persist - Count of site voxels converted from urban elements (eligible soil)
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

if __name__ == "__main__":
    main()