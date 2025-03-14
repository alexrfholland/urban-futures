import pyvista as pv
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

def load_vtk_file(site, scenario, voxel_size, year):
    """Load VTK file with features for given parameters"""
    filepath = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_with_features.vtk'
    try:
        return pv.read(filepath)
    except Exception as e:
        print(f"Error loading VTK file for year {year}: {e}")
        return None

def count_search_variables(vtk_data):
    """Count occurrences of each search variable category"""
    counts = {}
    
    # Print all point data keys to help debug
    print("\nAll point data keys in VTK file:")
    for key in sorted(vtk_data.point_data.keys()):
        if key.startswith('capabilities-'):
            print(f"  {key}")
    
    # Count bioavailable points by category
    if 'search_bioavailable' in vtk_data.point_data:
        bioavailable = vtk_data.point_data['search_bioavailable']
        
        # Count total bioavailable points (non-'none' values)
        counts['bioavailable_total'] = np.sum(bioavailable != 'none')
        
        # Count each category separately
        categories, category_counts = np.unique(bioavailable[bioavailable != 'none'], return_counts=True)
        for category, count in zip(categories, category_counts):
            counts[f'bioavailable_{category}'] = count
    
    # Count design actions
    if 'search_design_action' in vtk_data.point_data:
        design_actions = vtk_data.point_data['search_design_action']
        actions, action_counts = np.unique(design_actions[design_actions != 'none'], return_counts=True)
        for action, count in zip(actions, action_counts):
            counts[f'design_action_{action}'] = count
    
    # Count urban elements
    if 'search_urban_elements' in vtk_data.point_data:
        urban_elements = vtk_data.point_data['search_urban_elements']
        elements, element_counts = np.unique(urban_elements[urban_elements != 'none'], return_counts=True)
        for element, count in zip(elements, element_counts):
            counts[f'urban_element_{element}'] = count
    
    # Count capabilities for each persona
    for persona in ['bird', 'reptile', 'tree']:
        # Count total capabilities (non-'none' values)
        persona_key = f'capabilities-{persona}'
        if persona_key in vtk_data.point_data:
            capabilities = vtk_data.point_data[persona_key]
            counts[f'capabilities_{persona}_total'] = np.sum(capabilities != 'none')
            
            # Count each capability separately
            cap_types, cap_counts = np.unique(capabilities[capabilities != 'none'], return_counts=True)
            for cap_type, count in zip(cap_types, cap_counts):
                counts[f'capabilities_{persona}_{cap_type}'] = count
            
            print(f"  Found {len(cap_types)} capability types for {persona}: {cap_types}")
        else:
            print(f"  '{persona_key}' not found in point data")
        
        # Count individual boolean capability layers
        for capability in ['socialise', 'feed', 'raise-young'] if persona == 'bird' else \
                         ['traverse', 'foraige', 'shelter'] if persona == 'reptile' else \
                         ['grow', 'age', 'persist']:
            key = f'capabilities-{persona}-{capability}'
            if key in vtk_data.point_data:
                counts[f'capabilities_{persona}_{capability}_bool'] = np.sum(vtk_data.point_data[key])
                print(f"  Found boolean layer '{key}' with {counts[f'capabilities_{persona}_{capability}_bool']:,} True values")
            else:
                print(f"  Boolean layer '{key}' not found in point data")
    
    # Count resources (all point_data starting with resource_)
    for key in vtk_data.point_data.keys():
        if key.startswith('resource_'):
            # For numeric resources, count non-zero/non-NaN values
            data = vtk_data.point_data[key]
            if np.issubdtype(data.dtype, np.number):
                # Count non-zero and non-NaN values
                valid_count = np.sum(~np.isnan(data) & (data != 0))
                if valid_count > 0:
                    counts[key] = valid_count
            else:
                # For non-numeric resources, count non-empty/non-'none' values
                if data.dtype.kind == 'U' or data.dtype.kind == 'S':  # String types
                    valid_count = np.sum((data != '') & (data != 'none') & (data != 'nan'))
                    if valid_count > 0:
                        counts[key] = valid_count
                else:
                    # For boolean or other types
                    valid_count = np.sum(data)
                    if valid_count > 0:
                        counts[key] = valid_count
    
    return counts

def process_all_statistics(site='trimmed-parade', years=None, scenario='positive', voxel_size=1):
    """Process all statistics in a single pass through the data"""
    if years is None:
        years = [0, 10, 30, 60, 180]
    
    # Variables to exclude
    exclude_variables = ['design_action_node-rewilded']
    exclude_urban_elements = []
    
    # Define custom order for tree elements
    tree_element_order = ['tree_small', 'tree_medium', 'tree_large', 'tree_senescing', 'tree_snag', 'tree_fallen']
    
    # Initialize data structures
    main_stats_data = {}
    variable_names = set()
    
    # For design action tables
    all_design_actions = set()
    all_urban_elements = set()
    design_action_data = {}
    
    # First pass - collect all unique values across all timesteps
    print("\nFirst pass - collecting all unique values...")
    for year in years:
        print(f"  Scanning year {year}...")
        
        # Load VTK file
        vtk_data = load_vtk_file(site, scenario, voxel_size, year)
        if vtk_data is None:
            continue
        
        # Collect main statistics variable names
        counts = count_search_variables(vtk_data)
        variable_names.update(counts.keys())
        
        # Collect unique design actions and urban elements
        if 'search_design_action' in vtk_data.point_data and 'search_urban_elements' in vtk_data.point_data:
            design_actions = vtk_data.point_data['search_design_action']
            urban_elements = vtk_data.point_data['search_urban_elements']
            
            # Get unique design actions (excluding 'none' and excluded variables)
            unique_actions = np.unique(design_actions)
            for action in unique_actions:
                if action != 'none' and f'design_action_{action}' not in exclude_variables:
                    all_design_actions.add(action)
            
            # Get unique urban elements (excluding 'none' and excluded elements)
            unique_elements = np.unique(urban_elements)
            for element in unique_elements:
                if element != 'none' and element not in exclude_urban_elements:
                    all_urban_elements.add(element)
            
            # Add this debugging code
            if 'search_urban_elements' in vtk_data.point_data:
                urban_elements = vtk_data.point_data['search_urban_elements']
                unique_elements = np.unique(urban_elements)
                print(f"  Year {year} unique urban elements: {[e for e in unique_elements if e != 'none']}")
    
    # Remove excluded variables
    variable_names = variable_names - set(exclude_variables)
    
    # Convert sets to sorted lists with custom ordering for tree elements
    all_design_actions = sorted(all_design_actions)
    
    # Custom sort for urban elements - tree elements in specified order, others alphabetically
    tree_elements = [e for e in all_urban_elements if e in tree_element_order]
    other_elements = [e for e in all_urban_elements if e not in tree_element_order]
    
    # Sort tree elements according to the defined order
    tree_elements.sort(key=lambda x: tree_element_order.index(x) if x in tree_element_order else 999)
    
    # Sort other elements alphabetically
    other_elements.sort()
    
    # Combine the sorted lists
    all_urban_elements = other_elements + tree_elements
    
    print(f"\nFound {len(variable_names)} variables, {len(all_design_actions)} design actions, and {len(all_urban_elements)} urban elements")
    
    # Second pass - collect actual data
    print("\nSecond pass - collecting data...")
    for year in years:
        print(f"  Processing year {year}...")
        
        # Load VTK file
        vtk_data = load_vtk_file(site, scenario, voxel_size, year)
        if vtk_data is None:
            continue
        
        # Get main statistics counts
        counts = count_search_variables(vtk_data)
        main_stats_data[year] = counts
        
        # Process design action tables
        if 'search_design_action' in vtk_data.point_data and 'search_urban_elements' in vtk_data.point_data:
            design_actions = vtk_data.point_data['search_design_action']
            urban_elements = vtk_data.point_data['search_urban_elements']
            
            # For each design action, count urban elements
            for action in all_design_actions:
                action_key = f'design_action_{action}'
                
                # Skip if in exclude list
                if action_key in exclude_variables:
                    continue
                
                # Get mask for this design action
                action_mask = design_actions == action
                action_count = np.sum(action_mask)
                
                # Initialize data structure if needed
                if action_key not in design_action_data:
                    design_action_data[action_key] = {}
                
                if year not in design_action_data[action_key]:
                    design_action_data[action_key][year] = {'total': 0}
                
                # Set total count
                design_action_data[action_key][year]['total'] = action_count
                
                # Count urban elements for this action
                for element in all_urban_elements:
                    element_key = f'urban_element_{element}'
                    
                    # Count points that have both this action and this element
                    count = np.sum((design_actions == action) & (urban_elements == element))
                    
                    # Store count (even if zero)
                    design_action_data[action_key][year][element_key] = count
            
            # For improved-tree action, print the unique urban elements
            if 'improved-tree' in all_design_actions:
                improved_tree_mask = design_actions == 'improved-tree'
                if np.any(improved_tree_mask):
                    improved_tree_elements = urban_elements[improved_tree_mask]
                    unique_elements = np.unique(improved_tree_elements)
                    print(f"  Year {year} improved-tree urban elements: {[e for e in unique_elements if e != 'none']}")
    
    # Create main statistics DataFrame
    main_stats_df = None
    if main_stats_data:
        # Sort variable names
        variable_names = sorted(variable_names)
        
        # Initialize DataFrame with zeros
        main_stats_df = pd.DataFrame(0, 
                                    index=variable_names,
                                    columns=sorted(main_stats_data.keys()))
        
        # Fill in values
        for year, counts in main_stats_data.items():
            for var, count in counts.items():
                if var not in exclude_variables:
                    main_stats_df.at[var, year] = count
    
    # Create design action DataFrames
    design_tables = {}
    for action_key, year_data in design_action_data.items():
        if year_data:
            # Get all element keys
            tree_element_keys = [f'urban_element_{e}' for e in tree_elements]
            other_element_keys = [f'urban_element_{e}' for e in other_elements]
            element_keys = ['total'] + other_element_keys + tree_element_keys
            
            # Create DataFrame with zeros
            df = pd.DataFrame(0, 
                             index=element_keys,
                             columns=sorted(year_data.keys()))
            
            # Fill in values
            for year, counts in year_data.items():
                for element_key in element_keys:
                    if element_key in counts:
                        df.at[element_key, year] = counts[element_key]
            
            design_tables[action_key] = df
    
    return main_stats_df, design_tables

def create_design_action_plots(design_tables, site, scenario):
    """Create plots for each design action table"""
    print("\nCreating design action plots...")
    
    # Create temp directory for plots
    temp_dir = f'data/revised/final/{site}/temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    for action, table in design_tables.items():
        # Skip tables with no data
        if table.empty:
            continue
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot data
        table.T.plot(kind='bar', stacked=True, ax=plt.gca())
        
        # Add labels and title
        plt.title(f'{action.replace("design_action_", "")} - Urban Elements')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.legend(title='Urban Element', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save figure
        output_path = f'{temp_dir}/{site}_{scenario}_{action}_plot.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  Plot saved to: {output_path}")

def create_combined_design_action_plot(design_tables, site, scenario):
    """Create a combined plot of all design actions"""
    print("\nCreating combined design action plot...")
    
    # Create temp directory for plots
    temp_dir = f'data/revised/final/{site}/temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Skip if no tables
    if not design_tables:
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Extract total counts for each design action
    totals = {}
    for action, table in design_tables.items():
        if 'total' in table.index:
            action_name = action.replace('design_action_', '')
            totals[action_name] = table.loc['total']
    
    # Create DataFrame from totals
    if totals:
        totals_df = pd.DataFrame(totals)
        
        # Plot data
        totals_df.plot(kind='bar', ax=plt.gca())
        
        # Add labels and title
        plt.title('Design Actions - Total Counts')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.legend(title='Design Action', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save figure
        output_path = f'{temp_dir}/{site}_{scenario}_combined_design_actions_plot.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  Combined plot saved to: {output_path}")

def create_specific_plots(design_tables, site, scenario):
    """Create specific plots with pseudolog y-axis"""
    print("\nCreating specific plots...")
    
    # Create temp directory for plots
    temp_dir = f'data/revised/final/{site}/temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Skip if no tables
    if not design_tables:
        return
    
    # Check if we have the improved-tree design action
    if 'design_action_improved-tree' in design_tables:
        table = design_tables['design_action_improved-tree']
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot data with pseudolog y-axis
        ax = plt.gca()
        table.T.plot(kind='bar', stacked=True, ax=ax)
        ax.set_yscale('symlog')
        
        # Add labels and title
        plt.title('Improved Tree - Urban Elements (Log Scale)')
        plt.xlabel('Year')
        plt.ylabel('Count (log scale)')
        plt.legend(title='Urban Element', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save figure
        output_path = f'{temp_dir}/{site}_{scenario}_improved_tree_log_plot.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  Improved tree log plot saved to: {output_path}")

def generate_capability_dataframe(main_stats_df):
    """Generate a dataframe tracking capabilities across timesteps"""
    print("\nGenerating capability dataframe...")
    
    # Initialize lists to store data
    data = []
    
    # Track which capabilities were found
    found_capabilities = set()
    expected_capabilities = {
        'bird': ['socialise', 'feed', 'raise-young'],
        'reptile': ['traverse', 'foraige', 'shelter'],
        'tree': ['grow', 'age', 'persist']
    }
    
    # Process each timestep
    for year in main_stats_df.columns:
        # Get data for this year
        year_data = main_stats_df[year]
        
        # Process each persona and capability
        for persona in ['bird', 'reptile', 'tree']:
            # Get all capability keys for this persona
            capability_keys = [key for key in year_data.index if key.startswith(f'capabilities_{persona}_') and 
                              not key.endswith('_total')]
            
            print(f"  Year {year}, {persona.title()} capability keys found: {len(capability_keys)}")
            
            # Add data for each capability
            for key in capability_keys:
                # Extract capability name from key
                capability = key.replace(f'capabilities_{persona}_', '')
                
                # Track found capability
                found_capabilities.add(f"{persona}_{capability}")
                
                # Add to data list
                data.append({
                    'Year': year,
                    'Persona': persona.title(),
                    'Capability': capability.replace('-', ' ').title(),
                    'Count': year_data.get(key, 0)  # Use get with default 0 to handle missing keys
                })
    
    # Report on missing capabilities
    all_expected = set()
    for persona, capabilities in expected_capabilities.items():
        for capability in capabilities:
            all_expected.add(f"{persona}_{capability}")
    
    missing = all_expected - found_capabilities
    if missing:
        print("\n  Missing capabilities:")
        for item in sorted(missing):
            persona, capability = item.split('_', 1)
            print(f"    - {persona.title()}: {capability.replace('-', ' ').title()}")
    
    # Create dataframe
    capability_df = pd.DataFrame(data)
    
    # Check if dataframe is empty
    if capability_df.empty:
        print("  No capability data found in statistics")
        # Return an empty dataframe with the expected columns
        return pd.DataFrame(columns=['Persona', 'Capability', 'Year', 'Count'])
    
    # Check if 'Count' column exists
    if 'Count' not in capability_df.columns:
        print("  No count data found for capabilities")
        capability_df['Count'] = 0
    
    # Print the raw dataframe
    print("\n  Raw capability dataframe:")
    print(capability_df.head(10))
    print(f"  Total rows: {len(capability_df)}")
    
    # Pivot to get the format: persona, capability, timestep
    try:
        pivot_df = capability_df.pivot_table(
            index=['Persona', 'Capability'], 
            columns='Year', 
            values='Count',
            fill_value=0
        ).reset_index()
        
        print("\n  Pivoted capability dataframe:")
        print(pivot_df.head(10))
        print(f"  Total rows: {len(pivot_df)}")
        
        return pivot_df
    except Exception as e:
        print(f"  Error creating pivot table: {e}")
        print(f"  Dataframe columns: {capability_df.columns}")
        print(f"  Dataframe sample: {capability_df.head()}")
        # Return the unpivoted dataframe as a fallback
        return capability_df

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate statistics for search variables across years')
    parser.add_argument('--sites', nargs='+', default=['trimmed-parade'], 
                        help='Sites to process (default: [trimmed-parade])')
    parser.add_argument('--years', nargs='+', type=int, help='Years to process (default: [0, 10, 30, 60, 180])')
    parser.add_argument('--scenarios', nargs='+', default=['positive', 'trending'], 
                        help='Scenarios to process (default: [positive, trending])')
    parser.add_argument('--voxel-size', type=int, default=1, help='Voxel size (default: 1)')
    
    args = parser.parse_args()
    
    # Process all sites and scenarios
    for site in args.sites:
        print(f"\n=== Processing site: {site} ===")
        
        for scenario in args.scenarios:
            print(f"\n--- Processing scenario: {scenario} ---")
            
            # Process all statistics in a single pass
            main_stats_df, design_tables = process_all_statistics(
                site=site,
                years=args.years,
                scenario=scenario,
                voxel_size=args.voxel_size
            )
            
            # Create temp directory for outputs
            temp_dir = f'data/revised/final/{site}/temp'
            os.makedirs(temp_dir, exist_ok=True)
            
            # Print and save main statistics
            if main_stats_df is not None and not main_stats_df.empty:
                print("\nSearch Variables Statistics:")
                
                # Print capability-related columns first
                capability_cols = [col for col in main_stats_df.index if 'capabilities_' in col]
                if capability_cols:
                    print("\nCapability Statistics:")
                    capability_subset = main_stats_df.loc[capability_cols]
                    print(capability_subset)
                
                # Print full dataframe
                print("\nFull Statistics:")
                print(main_stats_df)
                
                # Save to CSV
                output_path = f'{temp_dir}/{site}_{scenario}_search_variables_stats.csv'
                main_stats_df.to_csv(output_path)
                print(f"\nStatistics saved to: {output_path}")
                
                # Generate and save capability dataframe
                print("\n=== GENERATING CAPABILITY DATAFRAME ===")
                capability_df = generate_capability_dataframe(main_stats_df)
                capability_output_path = f'{temp_dir}/{site}_{scenario}_capabilities_by_persona.csv'
                capability_df.to_csv(capability_output_path)
                print(f"Capability statistics saved to: {capability_output_path}")
            else:
                print(f"\nNo main statistics were processed for site {site}, scenario {scenario}.")
            
            # Print and save design action tables
            if design_tables:
                print("\nDesign Action Tables:")
                for action, table in design_tables.items():
                    print(f"\n{action}:")
                    print(table)
                    
                    # Save to CSV
                    output_path = f'{temp_dir}/{site}_{scenario}_{action}_urban_elements.csv'
                    table.to_csv(output_path)
                    print(f"Table saved to: {output_path}")
                
                # Create plots for design action tables
                create_design_action_plots(design_tables, site, scenario)
                
                # Create combined design action plot
                create_combined_design_action_plot(design_tables, site, scenario)
                
                # Create specific plots with pseudolog y-axis
                create_specific_plots(design_tables, site, scenario)
            else:
                print(f"\nNo design action tables were created for site {site}, scenario {scenario}.")

if __name__ == "__main__":
    main() 



#Persona	Capability	Search Criteria

#Bird	Socialise:	points in resource_perch branch	> 0
#Bird	Feed: points in resource_peeling bark	> 0 	
#Bird	Raise_young	points in resource_hollow > 0



#Reptile    Traverse: points in 'search_bioavailable' != 'none'
#Repitle    Foraige: points in 'search_bioavailable' == 'traversable' OR points in 'resource_dead branch' > 0 OR resource_epiphyte > 0
#Reptile    Shelter: points in 'resource_fallen log' > 0 OR points in 'forest_size' == 'fallen'


#Tree    Grow: points in 'resource_other' > 0
#Tree    Age: points in 'improved-tree'
#Tree    Persist: points in 'search_bioavailable' == 'traversable' WITHIN 5m of points in analysis_nodeType == 'tree'


