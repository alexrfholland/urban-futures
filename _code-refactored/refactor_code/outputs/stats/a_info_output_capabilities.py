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
import sys

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_code-refactored")
REPO_ROOT = CODE_ROOT.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.paths import statistics_csv_dir


# =============================================================================
# CONFIGURATION
# =============================================================================

SITES = ['trimmed-parade', 'city', 'uni']


def format_voxel_size(voxel_size):
    """Normalize voxel size for file naming (e.g. 1.0 -> '1')."""
    try:
        value = float(voxel_size)
        if value.is_integer():
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return str(voxel_size)


def output_dir(output_mode: str | None = None) -> Path:
    return statistics_csv_dir(output_mode)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_indicator_counts(site, voxel_size=1):
    """Load indicator counts CSV for a site."""
    voxel = format_voxel_size(voxel_size)
    path = output_dir() / f'{site}_{voxel}_indicator_counts.csv'
    if not path.exists():
        raise FileNotFoundError(f"Indicator counts not found: {path}")
    return pd.read_csv(path)


def load_action_counts(site, voxel_size=1):
    """Load action counts CSV for a site."""
    voxel = format_voxel_size(voxel_size)
    path = output_dir() / f'{site}_{voxel}_action_counts.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_proposal_opportunities(site, voxel_size=1):
    """Load proposal opportunities CSV for a site."""
    voxel = format_voxel_size(voxel_size)
    path = output_dir() / f'{site}_{voxel}_proposal_opportunities.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_proposal_interventions(site, voxel_size=1):
    """Load proposal interventions CSV for a site."""
    voxel = format_voxel_size(voxel_size)
    path = output_dir() / f'{site}_{voxel}_proposal_interventions.csv'
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
    voxel = format_voxel_size(voxel_size)
    
    current_output_dir = output_dir()
    current_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# COMBINING {len(sites)} SITES")
    print(f"{'#'*60}")
    
    all_indicators = []
    all_actions = []
    all_proposal_opportunities = []
    all_proposal_interventions = []
    
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

        proposal_opp_df = load_proposal_opportunities(site, voxel_size)
        if proposal_opp_df is not None:
            all_proposal_opportunities.append(proposal_opp_df)

        proposal_int_df = load_proposal_interventions(site, voxel_size)
        if proposal_int_df is not None:
            all_proposal_interventions.append(proposal_int_df)
    
    combined_indicators = None
    combined_actions = None
    
    if all_indicators:
        combined_indicators = pd.concat(all_indicators, ignore_index=True)
        trending_counts = combined_indicators[
            combined_indicators['scenario'] == 'trending'
        ][['site', 'year', 'indicator_id', 'voxel_size', 'count']].rename(
            columns={'count': 'trending_count'}
        )
        combined_indicators = combined_indicators.merge(
            trending_counts,
            on=['site', 'year', 'indicator_id', 'voxel_size'],
            how='left'
        )
        combined_indicators['increase from trending'] = pd.NA
        non_baseline_mask = combined_indicators['scenario'] != 'baseline'
        combined_indicators.loc[non_baseline_mask, 'increase from trending'] = (
            combined_indicators.loc[non_baseline_mask, 'count']
            / combined_indicators.loc[non_baseline_mask, 'trending_count']
        )
        combined_indicators = combined_indicators.drop(columns=['trending_count'])
        combined_path = current_output_dir / f'all_sites_{voxel}_indicator_counts.csv'
        combined_indicators.to_csv(combined_path, index=False)
        print(f"\nSaved: {combined_path}")
        print(f"  {len(combined_indicators)} total indicator records")
    
    if all_actions:
        combined_actions = pd.concat(all_actions, ignore_index=True)
        actions_path = current_output_dir / f'all_sites_{voxel}_action_counts.csv'
        combined_actions.to_csv(actions_path, index=False)
        print(f"Saved: {actions_path}")
        print(f"  {len(combined_actions)} total action records")

    if all_proposal_opportunities:
        combined_proposal_opportunities = pd.concat(all_proposal_opportunities, ignore_index=True)
        proposal_opp_path = current_output_dir / f'all_sites_{voxel}_proposal_opportunities.csv'
        combined_proposal_opportunities.to_csv(proposal_opp_path, index=False)
        print(f"Saved: {proposal_opp_path}")
        print(f"  {len(combined_proposal_opportunities)} total proposal opportunity records")

    if all_proposal_interventions:
        combined_proposal_interventions = pd.concat(all_proposal_interventions, ignore_index=True)
        proposal_int_path = current_output_dir / f'all_sites_{voxel}_proposal_interventions.csv'
        combined_proposal_interventions.to_csv(proposal_int_path, index=False)
        print(f"Saved: {proposal_int_path}")
        print(f"  {len(combined_proposal_interventions)} total proposal intervention records")
    
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

    print(f"\n{'='*80}")
    print("PROPOSAL METRICS SUMMARY")
    print(f"{'='*80}")
    for site in sites:
        proposal_df = load_proposal_interventions(site, voxel_size)
        if proposal_df is None or proposal_df.empty:
            continue
        print(f"\n--- {site.upper()} ---")
        if {'proposal_id', 'support_label', 'count'}.issubset(proposal_df.columns):
            # Backward compatibility for older/alternate schemas
            grouped = proposal_df.groupby(['proposal_id', 'support_label'])['count'].sum().reset_index()
            for _, row in grouped.iterrows():
                print(f"  {row['proposal_id']} / {row['support_label']}: {int(row['count']):,}")
        elif {'proposal_id', 'support_label', 'supported_voxel_count'}.issubset(proposal_df.columns):
            grouped = proposal_df.groupby(['proposal_id', 'support_label'])['supported_voxel_count'].sum(min_count=1).reset_index()
            for _, row in grouped.iterrows():
                value = row['supported_voxel_count']
                if pd.isna(value):
                    print(f"  {row['proposal_id']} / {row['support_label']}: n/a")
                else:
                    print(f"  {row['proposal_id']} / {row['support_label']}: {int(value):,} voxels")


def generate_capability_totals(sites=None, voxel_size=1, include_baseline=False):
    """
    Generate capability totals from indicator counts.

    Outputs:
    - totals_by_site_persona_pathway_{voxel}.csv
    - totals_by_pathway_persona_{voxel}.csv
    - totals_by_pathway_{voxel}.csv
    """
    if sites is None:
        sites = SITES
    voxel = format_voxel_size(voxel_size)

    all_frames = []
    for site in sites:
        try:
            df = load_indicator_counts(site, voxel_size)
            all_frames.append(df)
        except FileNotFoundError:
            print(f"Skipping {site}: no indicator counts")

    if not all_frames:
        print("No indicator data found; capability totals not generated.")
        return None, None, None

    combined = pd.concat(all_frames, ignore_index=True)
    combined['count'] = pd.to_numeric(combined['count'], errors='coerce').fillna(0)

    if not include_baseline:
        combined = combined[combined['scenario'] != 'baseline'].copy()

    by_site_persona_pathway = (
        combined
        .groupby(['site', 'persona', 'scenario'], as_index=False)['count']
        .sum()
        .rename(columns={'scenario': 'pathway', 'count': 'total_count'})
        .sort_values(['site', 'persona', 'pathway'])
    )

    by_pathway_persona = (
        combined
        .groupby(['scenario', 'persona'], as_index=False)['count']
        .sum()
        .rename(columns={'scenario': 'pathway', 'count': 'total_count'})
        .sort_values(['pathway', 'persona'])
    )

    by_pathway = (
        combined
        .groupby(['scenario'], as_index=False)['count']
        .sum()
        .rename(columns={'scenario': 'pathway', 'count': 'total_count'})
        .sort_values(['pathway'])
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    site_persona_path = OUTPUT_DIR / f'totals_by_site_persona_pathway_{voxel}.csv'
    pathway_persona_path = OUTPUT_DIR / f'totals_by_pathway_persona_{voxel}.csv'
    pathway_path = OUTPUT_DIR / f'totals_by_pathway_{voxel}.csv'

    by_site_persona_pathway.to_csv(site_persona_path, index=False)
    by_pathway_persona.to_csv(pathway_persona_path, index=False)
    by_pathway.to_csv(pathway_path, index=False)

    print(f"\nSaved: {site_persona_path}")
    print(f"Saved: {pathway_persona_path}")
    print(f"Saved: {pathway_path}")

    print(f"\n{'='*80}")
    print("TOTALS BY SITE + PERSONA + PATHWAY")
    print(f"{'='*80}")
    for _, row in by_site_persona_pathway.iterrows():
        print(f"{row['site']:<16} {row['persona']:<8} {row['pathway']:<9} {int(row['total_count']):>14,}")

    print(f"\n{'='*80}")
    print("TOTALS BY PATHWAY + PERSONA")
    print(f"{'='*80}")
    for _, row in by_pathway_persona.iterrows():
        print(f"{row['pathway']:<9} {row['persona']:<8} {int(row['total_count']):>14,}")

    print(f"\n{'='*80}")
    print("TOTALS BY PATHWAY")
    print(f"{'='*80}")
    for _, row in by_pathway.iterrows():
        print(f"{row['pathway']:<9} {int(row['total_count']):>14,}")

    return by_site_persona_pathway, by_pathway_persona, by_pathway


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
    parser.add_argument('--voxel-size', type=float, default=1)
    parser.add_argument('--summary', action='store_true',
                        help='Print summary table')
    parser.add_argument('--debug', action='store_true',
                        help='Generate debug analysis outputs')
    parser.add_argument('--totals', action='store_true',
                        help='Generate totals by site/persona/pathway and pathway summaries')
    parser.add_argument('--include-baseline', action='store_true',
                        help='Include baseline records when generating totals')
    
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

    if args.totals:
        sites = SITES if args.site == 'all' else [args.site]
        generate_capability_totals(sites, args.voxel_size, include_baseline=args.include_baseline)


if __name__ == '__main__':
    main()
