import pandas as pd
import numpy as np
from pathlib import Path

def generate_capability_dataframe(stats_df, site, scenario):
    """Generate a dataframe tracking capabilities across timesteps"""
    # Define the expected capabilities and their column names in the stats dataframe
    expected_indicators = {
        'Bird': {
            'Socialise': [('perch branch', 'capabilities-bird-socialise')],
            'Feed': [('peeling bark', 'capabilities-bird-feed')],
            'Raise Young': [('hollow', 'capabilities-bird-raise-young')]
        },
        'Reptile': {
            'Traverse': [('traversable', 'capabilities-reptile-traverse')],
            'Forage': [
                ('ground cover', 'capabilities-reptile-forage-low-veg'),
                ('dead branch', 'capabilities-reptile-forage-dead-branch'),
                ('epiphyte', 'capabilities-reptile-forage-epiphyte')
            ],
            'Shelter': [
                ('fallen log', 'capabilities-reptile-shelter-fallen-log'),
                ('fallen tree', 'capabilities-reptile-shelter-fallen-tree')
            ]
        },
        'Tree': {
            'Grow': [('volume', 'capabilities-tree-grow')],
            'Age': [
                ('improved tree', 'capabilities-tree-age'),
                ('reserve tree', 'capabilities-tree-age')
            ],
            'Persist': [
                ('medium tree', 'capabilities-tree-persist-near-medium'),
                ('large tree', 'capabilities-tree-persist-near-large')
            ]
        }
    }
    
    # Create a list to store rows for the dataframe
    rows = []
    
    # Get all years (timesteps) from the stats dataframe
    years = [col for col in stats_df.columns if col != 'Unnamed: 0' and col != 'capabilityID']
    
    # For each persona, capability, and numeric indicator, extract counts for all years
    for persona, capabilities in expected_indicators.items():
        # Assign a capability number (0, 1, 2) to each capability for this persona
        capability_numbers = {capability: i for i, capability in enumerate(capabilities.keys())}
        
        for capability, indicators in capabilities.items():
            for indicator_name, column_name in indicators:
                # Create a row with persona, capability, capability number, numeric indicator, and scenario
                row = {
                    'Persona': persona,
                    'Capability': capability,
                    'CapabilityNo': capability_numbers[capability],
                    'NumericIndicator': indicator_name,
                    'Scenario': scenario,
                    'Site': site,
                    'is_dummy': False  # Default value
                }
                
                # Add counts for each year
                for year in years:
                    if column_name in stats_df.index:
                        row[str(year)] = stats_df.at[column_name, year]
                    else:
                        row[str(year)] = 0
                
                # Add capabilityID if available
                if column_name in stats_df.index and 'capabilityID' in stats_df.columns:
                    row['capabilityID'] = stats_df.at[column_name, 'capabilityID']
                else:
                    row['capabilityID'] = ''
                
                # Add to rows
                rows.append(row)
    
    # Create dataframe
    capability_df = pd.DataFrame(rows)
    
    # Add IndicatorNo - a unique count for each unique NumericIndicator within each persona
    capability_df['IndicatorNo'] = -1  # Initialize with placeholder
    
    # Process each persona separately
    for persona in capability_df['Persona'].unique():
        # Create mask for current persona
        mask = capability_df['Persona'] == persona
        # Get unique indicators for this persona
        unique_indicators = capability_df.loc[mask, 'NumericIndicator'].unique()
        # Create mapping series with indicator numbers
        indicator_map = pd.Series(range(len(unique_indicators)), index=unique_indicators)
        # Apply mapping to this persona's rows
        capability_df.loc[mask, 'IndicatorNo'] = capability_df.loc[mask, 'NumericIndicator'].map(indicator_map)
    
    # Ensure IndicatorNo is integer type
    capability_df['IndicatorNo'] = capability_df['IndicatorNo'].astype(int)
    
    # Add hpos - a unique number for each persona (Tree=0, Bird=1, etc.)
    # Create a mapping of personas to their position
    persona_order = ['Tree', 'Bird', 'Reptile']  # Define the desired order
    persona_map = {persona: i for i, persona in enumerate(persona_order) if persona in capability_df['Persona'].unique()}
    # Apply the mapping
    capability_df['hpos'] = capability_df['Persona'].map(persona_map)
    
    return capability_df

def generate_urban_elements_dataframe(counts_df, stats_df, site, scenario):
    """Generate a dataframe tracking urban elements counts across timesteps"""
    if counts_df is None or counts_df.empty:
        return None
    
    # Generate the capability dataframe first to map capabilityIDs
    capability_df = None
    if stats_df is not None:
        capability_df = generate_capability_dataframe(stats_df, site, scenario)
    
    # Create a mapping from capabilityID to IndicatorNo, NumericIndicator, and hpos
    capability_mapping = {}
    if capability_df is not None:
        for _, row in capability_df.iterrows():
            if row['capabilityID'] and not pd.isna(row['capabilityID']):
                # Store IndicatorNo, NumericIndicator, and hpos in the mapping
                capability_mapping[row['capabilityID']] = {
                    'IndicatorNo': row['IndicatorNo'],
                    'NumericIndicator': row['NumericIndicator'],
                    'hpos': row['hpos']
                }
    
    # Create a list to store all rows in their original format
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
            'IndicatorNo': indicator_no
        }
        
        # Use the mapped NumericIndicator if available, otherwise create one from countname and countelement
        if numeric_indicator is not None:
            new_row['NumericIndicator'] = numeric_indicator
        else:
            new_row['NumericIndicator'] = f"{row['countname']} ({row['countelement']})"
        
        all_rows.append(new_row)
    
    # Create dataframe
    if not all_rows:
        return None
    
    urban_df = pd.DataFrame(all_rows)
    
    # For any rows where IndicatorNo or hpos is still -1, assign values based on the original data
    
    # Assign hpos for any row that didn't get mapped
    if any(urban_df['hpos'] == -1):
        # Create a mapping of personas to their position (lowercase to match raw data)
        persona_order = ['tree', 'bird', 'reptile']
        persona_map = {persona: i for i, persona in enumerate(persona_order)}
        urban_df.loc[urban_df['hpos'] == -1, 'hpos'] = urban_df.loc[urban_df['hpos'] == -1, 'persona'].map(persona_map)
    
    # Assign IndicatorNo for any row that didn't get mapped
    if any(urban_df['IndicatorNo'] == -1):
        # Process each persona separately
        for persona in urban_df['persona'].unique():
            # Get all rows for this persona
            persona_mask = urban_df['persona'] == persona
            
            # Get the max existing IndicatorNo for this persona
            existing_max = -1
            if any(persona_mask) and any(urban_df.loc[persona_mask, 'IndicatorNo'] != -1):
                existing_max = urban_df.loc[persona_mask, 'IndicatorNo'].max()
            
            # Find rows with missing IndicatorNo
            missing_mask = (persona_mask) & (urban_df['IndicatorNo'] == -1)
            
            if any(missing_mask):
                # Group by capability and countname to assign the same IndicatorNo to related rows
                # This mirrors how we'd expect the capability dataframe to be structured
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

def process_all_sites_scenarios(sites, scenarios, voxel_size=1):
    """Process all sites and scenarios to generate capability dataframes"""
    # Create lists to store all dataframes
    all_capability_dfs = []
    all_urban_dfs = []
    
    # Create the output directory structure
    output_dir = Path('data/revised/final/stats/arboreal-future-stats/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for site in sites:
        for scenario in scenarios:
            print(f"Processing {site}, scenario {scenario}")
            
            # Load capability statistics directly
            stats_path = Path(f'data/revised/final/stats/arboreal-future-stats/data/raw/{site}_{scenario}_{voxel_size}_capabilities_raw.csv')
            if stats_path.exists():
                stats_df = pd.read_csv(stats_path, index_col=0)
                
                # Generate capability dataframe
                capability_df = generate_capability_dataframe(stats_df, site, scenario)
                all_capability_dfs.append(capability_df)
                
                # Load urban elements counts directly
                counts_path = Path(f'data/revised/final/stats/arboreal-future-stats/data/raw/{site}_{scenario}_{voxel_size}_converted_urban_element_counts.csv')
                if counts_path.exists():
                    counts_df = pd.read_csv(counts_path)
                    
                    # Generate urban elements dataframe
                    urban_df = generate_urban_elements_dataframe(counts_df, stats_df, site, scenario)
                    if urban_df is not None:
                        all_urban_dfs.append(urban_df)
    
    # Process capability dataframes
    combined_capability_df = None
    if all_capability_dfs:
        combined_capability_df = pd.concat(all_capability_dfs, ignore_index=True)
        
        # Convert year columns to integers in capability dataframe
        year_columns = [col for col in combined_capability_df.columns if col.isdigit()]
        for col in year_columns:
            combined_capability_df[col] = combined_capability_df[col].astype(int)
        
        # Save combined capability dataframe with the requested naming convention
        for site in sites:
            site_df = combined_capability_df[combined_capability_df['Site'] == site]
            if not site_df.empty:
                output_path = output_dir / f'{site}_{voxel_size}.csv'
                site_df.to_csv(output_path, index=False)
                print(f"Saved capability data for site {site} to {output_path}")
        
        # Save the complete combined capability dataframe
        capability_output_path = output_dir / 'all_capabilities.csv'
        combined_capability_df.to_csv(capability_output_path, index=False)
        print(f"Saved combined capability data to {capability_output_path}")
    
    # Process urban elements dataframes
    combined_urban_df = None
    if all_urban_dfs:
        combined_urban_df = pd.concat(all_urban_dfs, ignore_index=True)
        
        # Select and reorder columns
        columns_order = ['site', 'scenario', 'timestep', 'persona', 'capability', 'NumericIndicator',
                          'countname', 'countelement', 'count', 'capabilityID', 'hpos', 'IndicatorNo']
        
        # Filter to only include the requested columns that are present
        existing_columns = [col for col in columns_order if col in combined_urban_df.columns]
        combined_urban_df = combined_urban_df[existing_columns]
        
        # Save the complete combined urban elements dataframe
        urban_output_path = output_dir / 'all_urban_elements.csv'
        combined_urban_df.to_csv(urban_output_path, index=False)
        print(f"Saved urban elements data to {urban_output_path}")
    
    return combined_capability_df, combined_urban_df

def main():
    """Main function to generate capability dataframes"""
    # Default values
    default_sites = ['trimmed-parade']
    default_scenarios = ['baseline', 'positive', 'trending']
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
    
    # Ask for voxel size
    voxel_size_input = input(f"Enter voxel size (default {default_voxel_size}): ")
    try:
        voxel_size = int(voxel_size_input) if voxel_size_input else default_voxel_size
    except ValueError:
        voxel_size = default_voxel_size
    
    # Print summary and confirm
    print(f"\nProcessing sites: {sites}")
    print(f"Scenarios: {scenarios}")
    print(f"Voxel Size: {voxel_size}")
    
    confirm = input("\nProceed? (yes/no, default yes): ")
    if confirm.lower() in ['no', 'n']:
        print("Operation cancelled.")
        return
    
    # Process all sites and scenarios
    capability_df, urban_df = process_all_sites_scenarios(sites, scenarios, voxel_size)
    
    if capability_df is not None and urban_df is not None:
        print("\nProcessing completed successfully.")

if __name__ == "__main__":
    main()