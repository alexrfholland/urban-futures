"""
a_infov3_debuglog.py
====================
Debug and analysis utilities for capability indicator data.

This module contains functions for generating analysis outputs that are useful
for debugging and detailed analysis but are NOT required for normal operation.

USAGE:
    from a_infov3_debuglog import generate_debug_outputs
    generate_debug_outputs(site, voxel_size=1)

All debug outputs are saved to: data/revised/final/output/debug/
"""

import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

DEBUG_DIR = Path('data/revised/final/output/debug')

INDICATOR_METADATA = {
    'Bird.self.peeling':           {'color': '#FF8A65', 'order': 1},
    'Bird.others.perch':           {'color': '#FFAB91', 'order': 2},
    'Bird.generations.hollow':     {'color': '#FFCCBC', 'order': 3},
    'Lizard.self.grass':           {'color': '#81C784', 'order': 4},
    'Lizard.self.dead':            {'color': '#A5D6A7', 'order': 5},
    'Lizard.self.epiphyte':        {'color': '#C8E6C9', 'order': 6},
    'Lizard.others.notpaved':      {'color': '#4CAF50', 'order': 7},
    'Lizard.generations.nurse-log':{'color': '#388E3C', 'order': 8},
    'Lizard.generations.fallen-tree':{'color': '#1B5E20', 'order': 9},
    'Tree.self.senescent':         {'color': '#64B5F6', 'order': 10},
    'Tree.others.notpaved':        {'color': '#42A5F5', 'order': 11},
    'Tree.generations.grassland':  {'color': '#1E88E5', 'order': 12},
}

PERSONA_ORDER = {'Bird': 0, 'Lizard': 1, 'Tree': 2}
CAPABILITY_ORDER = {'self': 0, 'others': 1, 'generations': 2}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_indicator_counts(site, voxel_size=1):
    """
    Load indicator counts CSV for a site.
    
    Tries output/csv folder first, then falls back to site folder.
    """
    path = Path(f'data/revised/final/output/csv/{site}_{voxel_size}_indicator_counts.csv')
    if not path.exists():
        path = Path(f'data/revised/final/{site}/{site}_{voxel_size}_indicator_counts.csv')
    if not path.exists():
        raise FileNotFoundError(f"Indicator counts not found: {path}")
    return pd.read_csv(path)


def load_action_counts(site, voxel_size=1):
    """
    Load action counts CSV for a site.
    
    Tries output/csv folder first, then falls back to site folder.
    """
    path = Path(f'data/revised/final/output/csv/{site}_{voxel_size}_action_counts.csv')
    if not path.exists():
        path = Path(f'data/revised/final/{site}/{site}_{voxel_size}_action_counts.csv')
    if not path.exists():
        return None
    return pd.read_csv(path)


# =============================================================================
# DEBUG ANALYSIS FUNCTIONS
# =============================================================================

def create_timeseries_df(indicator_df):
    """
    DEBUG: Pivot indicator counts to create a time-series DataFrame.
    
    Creates a wide-format table with years as columns for easy visualization.
    Output columns: site, scenario, indicator_id, persona, capability, label, 
                    color, order, year_0, year_10, year_30, year_60, year_180
    """
    pivot = indicator_df.pivot_table(
        index=['site', 'scenario', 'indicator_id', 'persona', 'capability', 'label'],
        columns='year',
        values='count',
        aggfunc='sum'
    ).reset_index()
    
    year_cols = [c for c in pivot.columns if isinstance(c, (int, float))]
    rename_map = {y: f'year_{int(y)}' for y in year_cols}
    pivot = pivot.rename(columns=rename_map)
    
    pivot['color'] = pivot['indicator_id'].map(lambda x: INDICATOR_METADATA.get(x, {}).get('color', '#999999'))
    pivot['order'] = pivot['indicator_id'].map(lambda x: INDICATOR_METADATA.get(x, {}).get('order', 99))
    pivot['persona_order'] = pivot['persona'].map(PERSONA_ORDER)
    pivot['capability_order'] = pivot['capability'].map(CAPABILITY_ORDER)
    
    pivot = pivot.sort_values(['persona_order', 'capability_order', 'order'])
    pivot = pivot.drop(columns=['persona_order', 'capability_order'])
    
    return pivot


def create_comparison_df(indicator_df, scenario_a='positive', scenario_b='trending'):
    """
    DEBUG: Compare indicator values between two scenarios.
    
    Calculates the difference and percentage difference between scenarios
    for each indicator at each year.
    """
    df_a = indicator_df[indicator_df['scenario'] == scenario_a].copy()
    df_b = indicator_df[indicator_df['scenario'] == scenario_b].copy()
    
    if df_a.empty or df_b.empty:
        return pd.DataFrame()
    
    merged = df_a.merge(
        df_b[['site', 'year', 'indicator_id', 'count']],
        on=['site', 'year', 'indicator_id'],
        suffixes=(f'_{scenario_a}', f'_{scenario_b}')
    )
    
    merged = merged.rename(columns={
        f'count_{scenario_a}': f'{scenario_a}_count',
        f'count_{scenario_b}': f'{scenario_b}_count'
    })
    
    merged['difference'] = merged[f'{scenario_a}_count'] - merged[f'{scenario_b}_count']
    merged['pct_difference'] = np.where(
        merged[f'{scenario_b}_count'] > 0,
        merged['difference'] / merged[f'{scenario_b}_count'] * 100,
        0
    )
    
    cols = ['site', 'year', 'indicator_id', 'persona', 'capability', 'label',
            f'{scenario_a}_count', f'{scenario_b}_count', 'difference', 'pct_difference']
    
    return merged[[c for c in cols if c in merged.columns]]


def create_change_df(indicator_df, scenario='positive'):
    """
    DEBUG: Calculate change from baseline (year 0) for a scenario.
    
    Shows absolute and percentage change from year 0 for each indicator.
    """
    df = indicator_df[indicator_df['scenario'] == scenario].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    baseline = df[df['year'] == 0].set_index('indicator_id')['count'].to_dict()
    
    records = []
    for _, row in df.iterrows():
        ind_id = row['indicator_id']
        baseline_val = baseline.get(ind_id, 0)
        current_val = row['count']
        abs_change = current_val - baseline_val
        pct_change = (abs_change / baseline_val * 100) if baseline_val > 0 else 0
        
        records.append({
            'site': row['site'],
            'scenario': scenario,
            'year': row['year'],
            'indicator_id': ind_id,
            'persona': row['persona'],
            'capability': row['capability'],
            'label': row['label'],
            'baseline_count': baseline_val,
            'current_count': current_val,
            'absolute_change': abs_change,
            'pct_change': pct_change
        })
    
    return pd.DataFrame(records)


def create_action_summary_df(action_df):
    """
    DEBUG: Aggregate action counts into a summary DataFrame.
    
    Groups actions by type and value to show totals.
    """
    if action_df is None or action_df.empty:
        return pd.DataFrame()
    
    summary = action_df.groupby([
        'site', 'scenario', 'year', 'indicator_id', 'action_type', 'action_value'
    ])['count'].sum().reset_index()
    
    return summary


def print_summary_table(indicator_df, scenario='positive'):
    """
    DEBUG: Print a formatted summary table of indicators to console.
    """
    df = indicator_df[indicator_df['scenario'] == scenario].copy()
    
    if df.empty:
        print(f"No data for scenario: {scenario}")
        return
    
    print(f"\n{'='*70}")
    print(f"INDICATOR SUMMARY - {scenario.upper()}")
    print(f"{'='*70}")
    
    years = sorted(df['year'].unique())
    
    header = f"{'Indicator':<35}" + "".join([f"{int(y):>10}" for y in years])
    print(header)
    print("-" * 70)
    
    for persona in ['Bird', 'Lizard', 'Tree']:
        persona_df = df[df['persona'] == persona]
        if persona_df.empty:
            continue
        
        print(f"\n{persona}")
        
        for ind_id in persona_df['indicator_id'].unique():
            ind_data = persona_df[persona_df['indicator_id'] == ind_id]
            row = f"  {ind_id:<33}"
            for year in years:
                val = ind_data[ind_data['year'] == year]['count'].values
                val = val[0] if len(val) > 0 else 0
                row += f"{int(val):>10,}"
            print(row)


# =============================================================================
# MAIN DEBUG OUTPUT FUNCTION
# =============================================================================

def generate_debug_outputs(site, voxel_size=1, print_summary=True):
    """
    Generate all debug analysis outputs for a site.
    
    Outputs are saved to: data/revised/final/output/debug/
    
    Parameters:
        site: Site name ('trimmed-parade', 'city', 'uni')
        voxel_size: Voxel size (default 1)
        print_summary: Whether to print summary tables to console
    
    Returns:
        dict with generated DataFrames
    """
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# GENERATING DEBUG OUTPUTS: {site}")
    print(f"{'#'*60}")
    
    # Load data
    indicator_df = load_indicator_counts(site, voxel_size)
    action_df = load_action_counts(site, voxel_size)
    
    print(f"\nLoaded {len(indicator_df)} indicator records")
    if action_df is not None:
        print(f"Loaded {len(action_df)} action records")
    
    results = {}
    
    # Timeseries
    timeseries = create_timeseries_df(indicator_df)
    timeseries_path = DEBUG_DIR / f'{site}_{voxel_size}_indicators_timeseries.csv'
    timeseries.to_csv(timeseries_path, index=False)
    print(f"\nSaved: {timeseries_path.name}")
    results['timeseries'] = timeseries
    
    # Scenario comparison
    comparison = create_comparison_df(indicator_df)
    if not comparison.empty:
        comparison_path = DEBUG_DIR / f'{site}_{voxel_size}_scenario_comparison.csv'
        comparison.to_csv(comparison_path, index=False)
        print(f"Saved: {comparison_path.name}")
        results['comparison'] = comparison
    
    # Change from baseline
    for scenario in ['positive', 'trending']:
        if scenario in indicator_df['scenario'].unique():
            change_df = create_change_df(indicator_df, scenario)
            change_path = DEBUG_DIR / f'{site}_{voxel_size}_{scenario}_change.csv'
            change_df.to_csv(change_path, index=False)
            print(f"Saved: {change_path.name}")
            results[f'{scenario}_change'] = change_df
    
    # Action summary
    if action_df is not None:
        action_summary = create_action_summary_df(action_df)
        if not action_summary.empty:
            action_path = DEBUG_DIR / f'{site}_{voxel_size}_actions_summary.csv'
            action_summary.to_csv(action_path, index=False)
            print(f"Saved: {action_path.name}")
            results['actions'] = action_summary
    
    # Print summary tables
    if print_summary:
        for scenario in ['positive', 'trending']:
            if scenario in indicator_df['scenario'].unique():
                print_summary_table(indicator_df, scenario)
    
    return results


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate debug analysis outputs for capability data'
    )
    parser.add_argument('--site', type=str, default='trimmed-parade')
    parser.add_argument('--voxel-size', type=int, default=1)
    parser.add_argument('--all-sites', action='store_true',
                        help='Process all sites')
    
    args = parser.parse_args()
    
    if args.all_sites:
        for site in ['trimmed-parade', 'city', 'uni']:
            generate_debug_outputs(site, args.voxel_size)
    else:
        generate_debug_outputs(args.site, args.voxel_size)

