import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import os

def load_capability_stats(site, scenario):
    """Load capability statistics from CSV file"""
    # Use the same naming convention as in a_info_generate_capabilitiesVTKs.py
    stats_path = Path(f'data/revised/final/stats/{site}_{scenario}_capabilities_raw.csv')
    
    if not stats_path.exists():
        print(f"Error: Statistics file not found at {stats_path}")
        return None
    
    # Load statistics
    stats_df = pd.read_csv(stats_path, index_col=0)
    print(f"Loaded capability statistics from {stats_path}")
    print(f"  Shape: {stats_df.shape}")
    
    return stats_df

def generate_capability_dataframe(stats_df, site, scenario):
    """Generate a dataframe tracking capabilities across timesteps"""
    print("\nGenerating capability dataframe...")
    
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
    years = [col for col in stats_df.columns if col != 'Unnamed: 0']
    print(f"  Processing years/timesteps: {years}")
    
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
                        print(f"  Warning: {column_name} not found in stats dataframe")
                        row[str(year)] = 0
                
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
    
    # Print the dataframe
    print("\nCapability dataframe:")
    print(capability_df.head())
    print(f"Total rows: {len(capability_df)}")
    
    return capability_df

def process_all_sites_scenarios(sites, scenarios, voxel_size=1):
    """Process all sites and scenarios to generate capability dataframes"""
    # Create a list to store all dataframes
    all_dfs = []
    
    for site in sites:
        for scenario in scenarios:
            print(f"\n=== Processing site: {site}, scenario: {scenario} ===")
            
            # Load capability statistics
            stats_df = load_capability_stats(site, scenario)
            if stats_df is None:
                continue
            
            # Generate capability dataframe
            capability_df = generate_capability_dataframe(stats_df, site, scenario)
            
            # Add to list
            all_dfs.append(capability_df)
    
    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Convert year columns to integers
        year_columns = ['0', '10', '30', '60', '180']
        for col in year_columns:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].astype(int)
        
        # Create the output directory structure
        output_dir = Path('data/revised/final/stats/arboreal-future-stats/data')

        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined dataframe with the requested naming convention
        for site in sites:
            site_df = combined_df[combined_df['Site'] == site]
            if not site_df.empty:
                output_path = output_dir / f'{site}_{voxel_size}.csv'
                site_df.to_csv(output_path, index=False)
                print(f"\nSaved capability dataframe for site {site} to {output_path}")
        
        # Also save the complete combined dataframe
        all_output_path = output_dir / 'all_capabilities.csv'
        combined_df.to_csv(all_output_path, index=False)
        print(f"\nSaved combined capability dataframe to {all_output_path}")
        print(f"Columns in final dataframe: {combined_df.columns.tolist()}")
        
        return combined_df
    else:
        print("\nNo capability dataframes were generated.")
        return None

def main():
    """Main function to generate capability dataframes"""
    #--------------------------------------------------------------------------
    # STEP 1: GATHER USER INPUTS
    #--------------------------------------------------------------------------
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
        print("Invalid input for voxel size. Using default value.")
        voxel_size = default_voxel_size
    
    # Print summary of selected options
    print("\n===== Processing with the following parameters =====")
    print(f"Sites: {sites}")
    print(f"Scenarios: {scenarios}")
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
    
    # Process all sites and scenarios
    combined_df = process_all_sites_scenarios(sites, scenarios, voxel_size)
    
    if combined_df is not None:
        print("\nSuccessfully generated capability dataframes.")
    else:
        print("\nFailed to generate capability dataframes.")

if __name__ == "__main__":
    main() 