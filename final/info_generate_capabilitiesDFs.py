import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import os

def load_capability_stats(site, scenario):
    """Load capability statistics from CSV file"""
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
            'Traverse': [('traversable', 'capabilities-reptile-traverse_traversable')],
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
                ('improved tree', 'capabilities-tree-age-improved-tree'),
                ('reserve tree', 'capabilities-tree-age-reserve-tree')
            ],
            'Persist': [
                ('medium tree', 'capabilities-tree-persist-near-medium'),
                ('large tree', 'capabilities-tree-persist-near-large')
            ]
        }
    }
    
    # Create a list to store rows for the dataframe
    rows = []
    
    # Get all years (timesteps) from the stats dataframe, including baseline if present
    years = sorted([col for col in stats_df.columns if col != 'baseline' and col != 'Unnamed: 0'])
    if 'baseline' in stats_df.columns:
        years = ['baseline'] + years
    
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
                    'hpos': 0,  # Default value
                    'is_dummy': False  # Default value
                }
                
                # Add counts for each year
                for year in years:
                    if year in stats_df.columns:
                        if column_name in stats_df.index:
                            row[str(year)] = stats_df.at[column_name, year]
                        else:
                            # Try alternative column names
                            alt_names = [
                                column_name,
                                column_name.replace('-', '_'),
                                f"{column_name}_bool"
                            ]
                            
                            found = False
                            for alt_name in alt_names:
                                if alt_name in stats_df.index:
                                    row[str(year)] = stats_df.at[alt_name, year]
                                    found = True
                                    break
                            
                            if not found:
                                print(f"  Warning: {column_name} not found in stats dataframe")
                                row[str(year)] = 0
                    else:
                        row[str(year)] = 0
                
                # Add to rows
                rows.append(row)
    
    # Create dataframe
    capability_df = pd.DataFrame(rows)
    
    # Print the dataframe
    print("\nCapability dataframe:")
    print(capability_df.head())
    print(f"Total rows: {len(capability_df)}")
    
    return capability_df

def process_all_sites_scenarios(sites, scenarios):
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
        
        # Save combined dataframe
        output_dir = Path('data/revised/final/stats')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'all_capabilities.csv'
        combined_df.to_csv(output_path, index=False)
        print(f"\nSaved combined capability dataframe to {output_path}")
        
        return combined_df
    else:
        print("\nNo capability dataframes were generated.")
        return None

def main():
    """Main function to generate capability dataframes"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate capability dataframes from statistics')
    parser.add_argument('--sites', nargs='+', default=['trimmed-parade'], 
                        help='Sites to process (default: [trimmed-parade])')
    parser.add_argument('--scenarios', nargs='+', default=['positive', 'trending'], 
                        help='Scenarios to process (default: [positive, trending])')
    
    args = parser.parse_args()
    
    # Process all sites and scenarios
    combined_df = process_all_sites_scenarios(args.sites, args.scenarios)
    
    if combined_df is not None:
        print("\nSuccessfully generated capability dataframes.")
    else:
        print("\nFailed to generate capability dataframes.")

if __name__ == "__main__":
    main() 