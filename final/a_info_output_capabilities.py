"""
a_infov3_output_capabilities.py
===============================
Combines indicator and action counts from all sites into unified CSVs.

USAGE:
    python a_infov3_output_capabilities.py --site all

This script:
    1. Loads indicator_counts.csv and action_counts.csv from each site
    2. Combines them into unified all_sites files
    3. Optionally generates debug analysis outputs

ESSENTIAL OUTPUTS (per site, from a_infov3_gather_capabilities.py):
    - {site}_{voxel_size}_indicator_counts.csv
    - {site}_{voxel_size}_action_counts.csv

COMBINED OUTPUTS (from this script):
    - all_sites_{voxel_size}_indicator_counts.csv
    - all_sites_{voxel_size}_action_counts.csv

DEBUG OUTPUTS (optional, use --debug flag):
    See a_infov3_debuglog.py for analysis outputs
"""

import pandas as pd
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path('data/revised/final/output/csv')
SITES = ['trimmed-parade', 'city', 'uni']


# =============================================================================
# DATA LOADING
# =============================================================================

def load_indicator_counts(site, voxel_size=1):
    """Load indicator counts CSV for a site."""
    path = OUTPUT_DIR / f'{site}_{voxel_size}_indicator_counts.csv'
    if not path.exists():
        path = Path(f'data/revised/final/{site}/{site}_{voxel_size}_indicator_counts.csv')
    if not path.exists():
        raise FileNotFoundError(f"Indicator counts not found: {path}")
    return pd.read_csv(path)


def load_action_counts(site, voxel_size=1):
    """Load action counts CSV for a site."""
    path = OUTPUT_DIR / f'{site}_{voxel_size}_action_counts.csv'
    if not path.exists():
        path = Path(f'data/revised/final/{site}/{site}_{voxel_size}_action_counts.csv')
    if not path.exists():
        return None
    return pd.read_csv(path)


# =============================================================================
# COMBINE SITES
# =============================================================================

def combine_sites(sites=None, voxel_size=1):
    """
    Combine indicator and action data from multiple sites into unified CSVs.
    
    Parameters:
        sites: List of site names (default: all sites)
        voxel_size: Voxel size (default 1)
    
    Returns:
        tuple: (combined_indicators_df, combined_actions_df)
    """
    if sites is None:
        sites = SITES
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# COMBINING {len(sites)} SITES")
    print(f"{'#'*60}")
    
    all_indicators = []
    all_actions = []
    
    for site in sites:
        try:
            indicator_df = load_indicator_counts(site, voxel_size)
            all_indicators.append(indicator_df)
            print(f"  Loaded {site}: {len(indicator_df)} indicator records")
        except FileNotFoundError:
            print(f"  Skipping {site}: no indicator counts")
            continue
        
        action_df = load_action_counts(site, voxel_size)
        if action_df is not None:
            all_actions.append(action_df)
    
    combined_indicators = None
    combined_actions = None
    
    if all_indicators:
        combined_indicators = pd.concat(all_indicators, ignore_index=True)
        combined_path = OUTPUT_DIR / f'all_sites_{voxel_size}_indicator_counts.csv'
        combined_indicators.to_csv(combined_path, index=False)
        print(f"\nSaved: {combined_path}")
        print(f"  {len(combined_indicators)} total indicator records")
    
    if all_actions:
        combined_actions = pd.concat(all_actions, ignore_index=True)
        actions_path = OUTPUT_DIR / f'all_sites_{voxel_size}_action_counts.csv'
        combined_actions.to_csv(actions_path, index=False)
        print(f"Saved: {actions_path}")
        print(f"  {len(combined_actions)} total action records")
    
    return combined_indicators, combined_actions


def print_summary(sites=None, voxel_size=1):
    """Print a summary of indicator counts for all sites."""
    if sites is None:
        sites = SITES
    
    print(f"\n{'='*80}")
    print("INDICATOR COUNTS SUMMARY")
    print(f"{'='*80}")
    
    for site in sites:
        try:
            df = load_indicator_counts(site, voxel_size)
        except FileNotFoundError:
            continue
        
        print(f"\n--- {site.upper()} ---")
        
        for scenario in ['positive', 'trending']:
            scenario_df = df[df['scenario'] == scenario]
            if scenario_df.empty:
                continue
            
            print(f"\n{scenario.capitalize()}:")
            years = sorted(scenario_df['year'].unique())
            
            # Header
            header = f"  {'Indicator':<30}" + "".join([f"{int(y):>8}" for y in years])
            print(header)
            
            # By persona
            for persona in ['Bird', 'Lizard', 'Tree']:
                persona_df = scenario_df[scenario_df['persona'] == persona]
                for ind_id in persona_df['indicator_id'].unique():
                    ind_data = persona_df[persona_df['indicator_id'] == ind_id]
                    row = f"  {ind_id:<30}"
                    for year in years:
                        val = ind_data[ind_data['year'] == year]['count'].values
                        val = val[0] if len(val) > 0 else 0
                        row += f"{int(val):>8,}"
                    print(row)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Combine capability data from all sites'
    )
    parser.add_argument('--site', type=str, default='all',
                        help='Site name, or "all" to combine all sites')
    parser.add_argument('--voxel-size', type=int, default=1)
    parser.add_argument('--summary', action='store_true',
                        help='Print summary table')
    parser.add_argument('--debug', action='store_true',
                        help='Generate debug analysis outputs')
    
    args = parser.parse_args()
    
    if args.site == 'all':
        combine_sites(SITES, args.voxel_size)
    else:
        # For single site, just print summary
        print_summary([args.site], args.voxel_size)
    
    if args.summary:
        print_summary(SITES if args.site == 'all' else [args.site], args.voxel_size)
    
    if args.debug:
        from a_infov3_debuglog import generate_debug_outputs
        sites = SITES if args.site == 'all' else [args.site]
        for site in sites:
            generate_debug_outputs(site, args.voxel_size)


if __name__ == '__main__':
    main()
