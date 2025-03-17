import numpy as np
import pandas as pd
import pyvista as pv
import argparse
from pathlib import Path
from scipy.spatial import cKDTree

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
    
    # Process baseline if requested
    if include_baseline:
        print(f"\n=== Processing baseline for site: {site} ===")
        # Try both possible baseline paths
        baseline_paths = [
            f'data/revised/final/baselines/{site}_baseline_combined_{voxel_size}.vtk',
            f'data/revised/final/{site}/{site}_baseline_resources_{voxel_size}.vtk',
            f'data/revised/final/{site}/{site}_baseline_{voxel_size}.vtk',
            f'data/revised/final/baselines/{site}_baseline_{voxel_size}.vtk'
        ]
        
        baseline_vtk = None
        for baseline_path in baseline_paths:
            try:
                if Path(baseline_path).exists():
                    print(f"Found baseline file: {baseline_path}")
                    baseline_vtk = pv.read(baseline_path)
                    print(f"Loaded baseline VTK from {baseline_path}")
                    break
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
        
        # Create capabilities
        vtk_data = create_bird_capabilities(vtk_data)
        vtk_data = create_reptile_capabilities(vtk_data)
        vtk_data = create_tree_capabilities(vtk_data)
        
        # Collect statistics
        year_stats = collect_capability_stats(vtk_data)
        # Store year as string to ensure consistent key types
        all_stats[str(year)] = year_stats
        data_processed = True
        
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

def create_bird_capabilities(vtk_data):
    """    Bird capabilities:
    1. Socialise: Points where 'stat_perch branch' > 0
       - Birds need branches to perch on for social activities
       
       Numeric indicators:
       - bird_socialise: Total voxels where stat_perch branch > 0 : 'perch branch'
    
    2. Feed: Points where 'stat_peeling bark' > 0
       - Birds feed on insects found under peeling bark
       
       Numeric indicators:
       - bird_feed: Total voxels where stat_peeling bark > 0 : 'peeling bark'
    
    3. Raise Young: Points where 'stat_hollow' > 0
       - Birds need hollows in trees to nest and raise their young
       
       Numeric indicators:
       - bird_raise_young: Total voxels where stat_hollow > 0 : 'hollow'
    """
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
    """Create capability layers for reptiles
    
    Reptile capabilities:
    1. Traverse: Points where 'search_bioavailable' != 'none'
       - Reptiles can move through any bioavailable space
       
       Numeric indicators:
       - reptile_traverse: Total voxels where search_bioavailable != 'none' : traversable
    
    2. Foraige: Points where any of the following conditions are met:
       - 'search_bioavailable' == 'low-vegetation' (areas reptiles can move through)
       - 'stat_dead branch' > 0 (dead branches provide foraging opportunities)
       - 'stat_epiphyte' > 0 (epiphytes provide habitat for prey)
       
       Numeric indicators:
       - reptile_forage_low_veg: Voxels where search_bioavailable == 'low-vegetation' : 'ground cover'
       - reptile_forage_dead_branch: Voxels where stat_dead branch > 0 : 'dead branch'
       - reptile_forage_epiphyte: Voxels where stat_epiphyte > 0 : 'epiphyte'
    
    3. Shelter: Points where any of the following conditions are met:
       - 'stat_fallen log' > 0 (fallen logs provide shelter)
       - 'forest_size' == 'fallen' (fallen trees provide shelter)
       
       Numeric indicators:
       - reptile_shelter_fallen_log: Voxels where stat_fallen log > 0 : 'fallen log'
       - reptile_shelter_fallen_tree: Voxels where forest_size == 'fallen' : 'fallen tree'
    """
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
    """Create capability layers for trees
    
    Tree capabilities:
    1. Grow: Points where 'stat_other' > 0
       - Areas where trees can grow and establish
       
       Numeric indicators:
       - tree_grow: Total voxels where stat_other > 0 : 'volume'
    
    2. Age: Points where 'search_design_action' == 'improved-tree' OR 'forset_control' == 'reserve-tree'
       - Areas where trees are protected and can mature
       
       Numeric indicators:
       - tree_age: Total voxels where search_design_action == 'improved-tree' : 'improved tree'
       - tree_age: Total voxels where forest_control == 'reserve-tree' : 'reserve tree'
    
    #change persist to: 
    3. Persist: Terrain points eligle for new tree plantings (ie. depaved and unmanaged land away from trees)
    
    Numeric indicator:
    -scenario_rewildingPlantings >= 1 : 'eligble soil'

    ARCHIVE: 3. Persist: Points where both conditions are met:
       - 'search_bioavailable' == 'traversable' (suitable growing conditions)
       - Within 1m of points where 'forest_size' == 'medium' OR 'forest_size' == 'large'
         (proximity to mature trees that can reproduce)
       
       Numeric indicators:
       - tree_persist_near_medium: Traversable voxels within 1m of medium trees : 'medium tree'
       - tree_persist_near_large: Traversable voxels within 1m of large trees : 'large tree'
    """
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
    """Load VTK file for a specific year"""
    output_path = f'data/revised/final/{site}'
    
    # Try different file patterns
    file_patterns = [
        # Standard patterns
        f'{output_path}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk',
        f'{output_path}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk',
        f'{output_path}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_features.vtk',
        
        # Additional patterns
        f'{output_path}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_urban_features.vtk',
        f'{output_path}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_features.vtk',
        
        # Patterns from a_scenario_urban_elements_count.py
        f'{output_path}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_capabilities.vtk',
        f'{output_path}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_search.vtk',
        
        # Try without voxel size in filename
        f'{output_path}/{site}_{scenario}_scenarioYR{year}.vtk',
        f'{output_path}/{site}_{scenario}_scenarioYR{year}_urban_features.vtk'
    ]
    
    print(f"Searching for VTK files for site: {site}, scenario: {scenario}, year: {year}")
    
    # Check if files exist before trying to load them
    existing_files = []
    for vtk_path in file_patterns:
        if Path(vtk_path).exists():
            existing_files.append(vtk_path)
    
    if existing_files:
        print(f"Found {len(existing_files)} matching VTK files:")
        for file_path in existing_files:
            print(f"  - {file_path}")
    else:
        print(f"No matching VTK files found in {output_path}")
        # List all VTK files in the directory
        all_vtk_files = list(Path(output_path).glob(f"{site}_{scenario}*{year}*.vtk"))
        if all_vtk_files:
            print(f"Available VTK files for {site}_{scenario} year {year}:")
            for file_path in all_vtk_files:
                print(f"  - {file_path}")
        else:
            # List all VTK files for this scenario
            all_scenario_files = list(Path(output_path).glob(f"{site}_{scenario}*.vtk"))
            if all_scenario_files:
                print(f"Available VTK files for {site}_{scenario}:")
                for file_path in all_scenario_files:
                    print(f"  - {file_path}")
            else:
                print(f"No VTK files found for {site}_{scenario} in {output_path}")
    
    # Try to load each file
    for vtk_path in file_patterns:
        try:
            if Path(vtk_path).exists():
                vtk_data = pv.read(vtk_path)
                print(f"Successfully loaded VTK file from {vtk_path}")
                
                # Print available point data keys
                print(f"Available point data keys in {Path(vtk_path).name}:")
                for key in sorted(vtk_data.point_data.keys()):
                    print(f"  - {key}")
                
                return vtk_data
        except Exception as e:
            print(f"Error loading {vtk_path}: {e}")
    
    print(f"ERROR: Could not find or load any VTK file for site {site}, scenario {scenario}, year {year}")
    return None

if __name__ == "__main__":
    main()