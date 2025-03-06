"""
Simulation Workflow and Statistics Generation

The simulation processes trees through several stages at each timestep:

1. AGING:
   - Trees grow based on growth_factor and years_passed
   - DBH (diameter at breast height) increases
   - Tree size updates (small->medium->large based on DBH thresholds)
   - Useful life expectancy decreases

2. SENESCENCE CHECK:
   - Trees with low life expectancy may become senescent
   - Senescent trees either:
     a) AGE-IN-PLACE (if CanopyResistance < ageInPlaceThreshold)
     b) REPLACE (if CanopyResistance >= ageInPlaceThreshold)

3. AGING-IN-PLACE PROGRESSION:
   - Trees marked for AGE-IN-PLACE become 'senescing'
   - May progress to 'snag' based on snagChance
   - May progress to 'fallen' based on collapseChance
   - Control changes to 'improved-tree'

4. REWILDING STATUS:
   - AGE-IN-PLACE trees are assigned status based on CanopyResistance:
     * 'node-rewilded' (lowest resistance)
     * 'footprint-depaved' (medium resistance)
     * 'exoskeleton' (highest resistance)

5. NEW TREE PLANTING:
   - New trees planted in rewilded areas
   - Two types of planting:
     a) Node-based: Around existing trees
     b) Turn-based: In areas meeting resistance/turns criteria

The resulting statistics table describes:
- Key identifiers: scenario, timestep, size, control, precolonial status
- Tree counts: total, removed, planted, aging-in-place
- Rewilding counts: exoskeletons, depaved areas
- Health metrics: critical lifespan count (<10 years)
- Resistance metrics: average and quartile boundaries of canopy resistance

Each row represents a unique combination of scenario (trending/positive), 
timestep, size category, control type, and precolonial status, providing 
a comprehensive view of the urban forest's evolution over time.
"""

site = 'trimmed-parade'
import pandas as pd
import numpy as np
import os

def create_aggregate_stats(site, voxel_size=1):
    # Define paths and scenarios
    base_path = f'data/revised/final/{site}'
    scenarios = ['trending', 'positive']
    timesteps = [0, 10, 30, 60, 180]
    
    # Initialize list to store all stats
    all_stats = []
    
    for scenario in scenarios:
        for timestep in timesteps:
            # Load the nodeDF for this timestep and scenario
            df_path = f'{base_path}/{site}_{scenario}_{voxel_size}_nodeDF_{timestep}.csv'
            if not os.path.exists(df_path):
                print(f"Warning: File not found: {df_path}")
                continue
                
            df = pd.read_csv(df_path)
            
            # Group by the key columns
            groupby_cols = ['size', 'control', 'precolonial']
            groups = df.groupby(groupby_cols)
            
            for group_key, group_df in groups:
                size, control, precolonial = group_key
                
                stats = {
                    'scenario': scenario,
                    'timestep': timestep,
                    'size': size,
                    'control': control,
                    'precolonial': precolonial,
                    'total_trees': len(group_df),
                    'total_removed': len(group_df[group_df['action'] == 'REPLACE']),
                    'total_planted': len(group_df[group_df['isNewTree'] == True]),
                    'total_age_in_place': len(group_df[group_df['action'] == 'AGE-IN-PLACE']),
                    'total_exoskeletons': len(group_df[group_df['rewilded'] == 'exoskeleton']),
                    'total_depaved': len(group_df[group_df['rewilded'] == 'footprint-depaved']),
                    'total_critical_lifespan': len(group_df[group_df['useful_life_expectancy'] < 10]),
                    'avg_canopy_resistance': group_df['CanopyResistance'].mean(),
                    'canopy_resistance_25': group_df['CanopyResistance'].quantile(0.25),
                    'canopy_resistance_75': group_df['CanopyResistance'].quantile(0.75)
                }
                
                all_stats.append(stats)
    
    # Create DataFrame from all stats
    stats_df = pd.DataFrame(all_stats)
    
    # Create stats directory if it doesn't exist
    stats_dir = 'data/revised/final/stats'
    os.makedirs(stats_dir, exist_ok=True)
    
    # Save to CSV
    output_path = f'{stats_dir}/{site}-scenario-stats.csv'
    stats_df.to_csv(output_path, index=False)
    
    print(f"\nStats for {site}:")
    print(stats_df)
    print(f"\nSaved to: {output_path}")
    
    return stats_df

# Process each site
sites = ['trimmed-parade']
for site in sites:
    create_aggregate_stats(site)