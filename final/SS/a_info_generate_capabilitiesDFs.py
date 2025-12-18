import pandas as pd
import numpy as np
from pathlib import Path

def generate_capability_dataframe(stats_df, selected_sites=None, selected_scenarios=None):
    """Generate a dataframe tracking capabilities across timesteps"""
    # Filter data if sites or scenarios are specified
    if selected_sites:
        stats_df = stats_df[stats_df['site'].isin(selected_sites)]
    if selected_scenarios:
        stats_df = stats_df[stats_df['scenario'].isin(selected_scenarios)]
    
    # Define the expected capabilities and their column names in the stats dataframe
    expected_indicators = {
        'Bird': {
            'Socialise': [('perch-branch', 'capabilities_bird_socialise_perch-branch')],
            'Feed': [('peeling-bark', 'capabilities_bird_feed_peeling-bark')],
            'Raise Young': [('hollow', 'capabilities_bird_raise-young_hollow')]
        },
        'Reptile': {
            'Traverse': [('traversable', 'capabilities_reptile_traverse_traversable')],
            'Forage': [
                ('ground-cover', 'capabilities_reptile_forage_ground-cover'),
                ('dead-branch', 'capabilities_reptile_forage_dead-branch'),
                ('epiphyte', 'capabilities_reptile_forage_epiphyte')
            ],
            'Shelter': [
                ('fallen-log', 'capabilities_reptile_shelter_fallen-log'),
                ('fallen-tree', 'capabilities_reptile_shelter_fallen-tree')
            ]
        },
        'Tree': {
            'Grow': [('volume', 'capabilities_tree_grow_volume')],
            'Age': [
                ('improved-tree', 'capabilities_tree_age_improved-tree'),
                ('reserve-tree', 'capabilities_tree_age_reserve-tree')
            ],
            'Persist': [('eligible-soil', 'capabilities_tree_persist_eligible-soil')]
        }
    }
    
    # Create a mapping from capability ID to capability details
    capability_map = {}
    for persona, capabilities in expected_indicators.items():
        capability_numbers = {cap: i for i, cap in enumerate(capabilities.keys())}
        
        for capability_name, indicators in capabilities.items():
            for indicator_name, capability_id in indicators:
                capability_map[capability_id] = {
                    'Persona': persona,
                    'Capability': capability_name,
                    'CapabilityNo': capability_numbers[capability_name],
                    'NumericIndicator': indicator_name
                }
    
    # Process each row in the original dataframe
    rows = []
    for _, row in stats_df.iterrows():
        capability_id = row['capability']
        
        # Skip if this capability ID is not in our expected mapping
        if capability_id not in capability_map:
            continue
        
        # Get the mapped values and create a new row
        mapped_values = capability_map[capability_id]
        rows.append({
            'Persona': mapped_values['Persona'],
            'Capability': mapped_values['Capability'],
            'CapabilityNo': mapped_values['CapabilityNo'],
            'NumericIndicator': mapped_values['NumericIndicator'],
            'Scenario': row['scenario'],
            'Site': row['site'],
            'is_dummy': False,
            'timestep': row['timestep'],
            'count': row['count'],
            'capabilityID': row.get('capabilityID', ''),
            'voxel_size': row.get('voxel_size', 1),
            'capability': row['capability']
        })
    
    # Create dataframe
    capability_df = pd.DataFrame(rows)
    
    # Add IndicatorNo - a unique count for each unique NumericIndicator within each persona
    capability_df['IndicatorNo'] = -1
    
    # Process each persona separately to assign IndicatorNo
    for persona in capability_df['Persona'].unique():
        mask = capability_df['Persona'] == persona
        unique_indicators = capability_df.loc[mask, 'NumericIndicator'].unique()
        indicator_map = pd.Series(range(len(unique_indicators)), index=unique_indicators)
        capability_df.loc[mask, 'IndicatorNo'] = capability_df.loc[mask, 'NumericIndicator'].map(indicator_map)
    
    # Add hpos - a unique number for each persona
    persona_order = ['Tree', 'Bird', 'Reptile']
    persona_map = {persona: i for i, persona in enumerate(persona_order) if persona in capability_df['Persona'].unique()}
    capability_df['hpos'] = capability_df['Persona'].map(persona_map)
    
    # Ensure types
    capability_df['IndicatorNo'] = capability_df['IndicatorNo'].astype(int)
    
    return capability_df

def generate_urban_elements_dataframe(counts_df, capability_df, selected_sites=None, selected_scenarios=None):
    """Generate a dataframe tracking urban elements counts across timesteps"""
    # Filter data if sites or scenarios are specified
    if selected_sites:
        counts_df = counts_df[counts_df['site'].isin(selected_sites)]
    if selected_scenarios:
        counts_df = counts_df[counts_df['scenario'].isin(selected_scenarios)]
    
    # Create a mapping from capabilityID to IndicatorNo, NumericIndicator, and hpos
    capability_mapping = {}
    for _, row in capability_df.iterrows():
        if row['capabilityID'] and not pd.isna(row['capabilityID']):
            capability_mapping[row['capabilityID']] = {
                'IndicatorNo': row['IndicatorNo'],
                'NumericIndicator': row['NumericIndicator'],
                'hpos': row['hpos']
            }
    
    # Create a list to store all rows
    all_rows = []
    
    # Process each row in the counts_df
    for _, row in counts_df.iterrows():
        capabilityID = row['capabilityID']
        
        # Get the mapped values if available
        indicator_no = -1
        numeric_indicator = None
        hpos = -1
        
        if capabilityID in capability_mapping:
            mapping = capability_mapping[capabilityID]
            indicator_no = mapping['IndicatorNo']
            numeric_indicator = mapping['NumericIndicator']
            hpos = mapping['hpos']
        
        # Create a new row with the original data plus the mapped values
        new_row = {
            'site': row['site'],
            'scenario': row['scenario'],
            'timestep': row['timestep'],
            'persona': row['persona'],
            'capability': row['capability'],
            'countname': row['countname'],
            'countelement': row['countelement'],
            'count': row['count'],
            'capabilityID': capabilityID,
            'hpos': hpos,
            'IndicatorNo': indicator_no,
            'NumericIndicator': numeric_indicator if numeric_indicator else f"{row['countname']} ({row['countelement']})"
        }
        
        all_rows.append(new_row)
    
    urban_df = pd.DataFrame(all_rows)
    
    # For any rows where IndicatorNo or hpos is still -1, assign values based on the original data
    
    # Assign hpos for any row that didn't get mapped
    if any(urban_df['hpos'] == -1):
        # Create a mapping of personas to their position (lowercase to match raw data)
        persona_order = ['tree', 'bird', 'reptile']
        persona_map = {persona: i for i, persona in enumerate(persona_order)}
        urban_df.loc[urban_df['hpos'] == -1, 'hpos'] = urban_df.loc[urban_df['hpos'] == -1, 'persona'].map(persona_map)
    
    # Assign IndicatorNo for any row that didn't get mapped
    # Process each persona separately
    for persona in urban_df['persona'].unique():
        # Get all rows for this persona
        persona_mask = urban_df['persona'] == persona
        
        # Get the max existing IndicatorNo for this persona
        existing_max = urban_df.loc[persona_mask & (urban_df['IndicatorNo'] != -1), 'IndicatorNo'].max()
        existing_max = -1 if pd.isna(existing_max) else existing_max
        
        # Find rows with missing IndicatorNo
        missing_mask = (persona_mask) & (urban_df['IndicatorNo'] == -1)
        
        # Group by capability and countname to assign the same IndicatorNo to related rows
        unique_combos = urban_df.loc[missing_mask, ['capability', 'countname']].drop_duplicates()
        combo_map = {(row['capability'], row['countname']): i + existing_max + 1 
                     for i, (_, row) in enumerate(unique_combos.iterrows())}
        
        # Apply the mapping
        for idx, row in urban_df[missing_mask].iterrows():
            combo_key = (row['capability'], row['countname'])
            if combo_key in combo_map:
                urban_df.at[idx, 'IndicatorNo'] = combo_map[combo_key]
    
    # Ensure IndicatorNo is integer type
    urban_df['IndicatorNo'] = urban_df['IndicatorNo'].astype(int)
    urban_df['hpos'] = urban_df['hpos'].astype(int)
    
    return urban_df

def process_capabilities_and_elements(sites, scenarios, voxel_size=1):
    """Process capability and urban elements data for selected sites and scenarios"""
    # Create the output directory structure
    output_dir = Path('data/revised/final/stats/arboreal-future-stats/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lists to store dataframes for each site
    all_capability_dfs = []
    all_urban_dfs = []
    
    # Process each site
    for site in sites:
        print(f"Processing site: {site}")
        
        # Load capabilities data
        capabilities_path = Path(f'data/revised/final/stats/arboreal-future-stats/data/raw/{site}_all_scenarios_{voxel_size}_capabilities_counts.csv')
        
        try:
            capabilities_df = pd.read_csv(capabilities_path)
            capability_df = generate_capability_dataframe(capabilities_df, [site], scenarios)
            all_capability_dfs.append(capability_df)
            
            # Load urban elements data
            urban_path = Path(f'data/revised/final/stats/arboreal-future-stats/data/raw/{site}_all_scenarios_{voxel_size}_urban_element_counts.csv')
            urban_df = pd.read_csv(urban_path)
            
            # Generate urban elements dataframe
            processed_urban_df = generate_urban_elements_dataframe(urban_df, capability_df, [site], scenarios)
            all_urban_dfs.append(processed_urban_df)
        except FileNotFoundError as e:
            print(f"  File not found: {e.filename}")
    
    # Combine and save capability dataframes
    combined_capability_df = pd.concat(all_capability_dfs, ignore_index=True)
    
    # Save per-site capability dataframes
    for site in sites:
        site_df = combined_capability_df[combined_capability_df['Site'] == site]
        output_path = output_dir / f'{site}_{voxel_size}.csv'
        site_df.to_csv(output_path, index=False)
    
    # Save combined capability dataframe
    all_output_path = output_dir / 'all_capabilities.csv'
    combined_capability_df.to_csv(all_output_path, index=False)
    
    # Combine and save urban elements dataframes
    combined_urban_df = pd.concat(all_urban_dfs, ignore_index=True)
    
    # Select and reorder columns
    columns_order = ['site', 'scenario', 'timestep', 'persona', 'capability', 'NumericIndicator',
                     'countname', 'countelement', 'count', 'capabilityID', 'hpos', 'IndicatorNo']
    
    # Filter to only include the requested columns that are present
    existing_columns = [col for col in columns_order if col in combined_urban_df.columns]
    combined_urban_df = combined_urban_df[existing_columns]
    
    # Save urban elements dataframe
    urban_output_path = output_dir / 'all_urban_elements.csv'
    combined_urban_df.to_csv(urban_output_path, index=False)
    
    return output_dir

def main():
    """Main function to process and reorganize capabilities and urban elements data"""
    # Default values
    default_sites = ['trimmed-parade']
    default_scenarios = ['baseline', 'positive', 'trending']
    default_voxel_size = 1
    
    # Ask for inputs
    sites_input = input(f"Enter site(s) (default {default_sites}): ")
    sites = sites_input.split(',') if sites_input else default_sites
    sites = [site.strip() for site in sites]
    
    scenarios_input = input(f"Enter scenario(s) (default {default_scenarios}): ")
    scenarios = scenarios_input.split(',') if scenarios_input else default_scenarios
    scenarios = [scenario.strip() for scenario in scenarios]
    
    voxel_size_input = input(f"Voxel size (default {default_voxel_size}): ")
    voxel_size = int(voxel_size_input) if voxel_size_input and voxel_size_input.isdigit() else default_voxel_size
    
    # Process data
    output_dir = process_capabilities_and_elements(sites, scenarios, voxel_size)
    
    print(f"Processing completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()