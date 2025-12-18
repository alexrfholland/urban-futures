"""
a_infov3_graphs.py
==================
Generate stream graphs (stacked area charts) from indicator counts.

USAGE:
    python final/a_infov3_graphs.py
    python final/a_infov3_graphs.py --sites trimmed-parade,city,uni
    python final/a_infov3_graphs.py --sites city --color-by persona

OUTPUT:
    plots/stream_graph_1_{sites}.html  (interactive HTML file)

SETTINGS:
    All adjustable settings are in the CONFIGURATION section below.
    Modify and re-run the script to see changes.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.interpolate import UnivariateSpline, PchipInterpolator


# =============================================================================
# CONFIGURATION - ADJUST THESE SETTINGS
# =============================================================================

# --- PATHS ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data' / 'revised' / 'final' / 'output' / 'csv'
PLOT_DIR = SCRIPT_DIR.parent / 'data' / 'revised' / 'final' / 'output' / 'plots'

# --- GRAPH DIMENSIONS ---
GRAPH_WIDTH = 800               # Total width in pixels
GRAPH_HEIGHT_PER_SITE = 800     # Height per site subplot in pixels

# --- BASELINE SETTINGS ---
BASELINE_YEAR = -50             # X position of baseline on graph
BASELINE_PCT = 100              # Baseline value (100%)

# --- SMOOTHING (for PCHIP interpolation, not currently used but kept for reference) ---
SMOOTHING_FACTOR = None         # None = use PCHIP (monotonic, no overshoot)

# --- STREAM AREA STYLING ---
FILL_ALPHA = 0.4                # Opacity of stream areas (0-1)

# --- INDICATOR SEPARATOR LINES (between indicators within a persona) ---
SEP_LINE_WIDTH = 0.5
SEP_LINE_COLOR = 'rgba(255,255,255,0.8)'  # White, slightly transparent
SEP_LINE = dict(width=SEP_LINE_WIDTH, color=SEP_LINE_COLOR)

# --- PERSONA SEPARATOR LINES (between Bird/Lizard/Tree groups) ---
PERSONA_SEP_WIDTH = 1.5
PERSONA_SEP_COLOR = 'rgba(0,0,0,0.8)'     # Black
PERSONA_SEP_LINE = dict(width=PERSONA_SEP_WIDTH, color=PERSONA_SEP_COLOR)

# --- CAPABILITY COLORS (when color-by=capability) ---
CAPABILITY_COLORS = {
    'self': '#1B9E77',        # teal
    'others': '#D95F02',      # orange
    'generations': '#7570B3', # purple
}

# --- PERSONA COLORS (when color-by=persona) ---
PERSONA_COLORS = {
    'Bird': '#E41A1C',    # red
    'Lizard': '#4DAF4A',  # green
    'Tree': '#377EB8',    # blue
}


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_indicator_data(sites, voxel_size=1):
    """Load indicator counts for specified sites."""
    all_data = []
    
    for site in sites:
        path = DATA_DIR / f'{site}_{voxel_size}_indicator_counts.csv'
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue
        
        df = pd.read_csv(path)
        all_data.append(df)
    
    if not all_data:
        raise FileNotFoundError(f"No indicator count files found in {DATA_DIR}")
    
    return pd.concat(all_data, ignore_index=True)


def prepare_plot_data(raw_data, sites):
    """
    Prepare data for plotting:
    - Filter to specified sites and scenarios
    - Create stream IDs
    - Add LOESS support points (baseline anchors)
    """
    # Filter data
    df = raw_data[
        (raw_data['site'].isin(sites)) & 
        (raw_data['scenario'].isin(['trending', 'positive']))
    ].copy()
    
    df['year'] = pd.to_numeric(df['year'])
    df['indicator_site'] = df['indicator_id'] + '-' + df['site']
    df['stream_id'] = df['indicator_id']
    
    # Create key columns for grouping
    keys = ['site', 'scenario', 'capability', 'indicator_id', 'indicator_site', 'stream_id']
    
    # Get unique combinations for baseline rows
    unique_combos = df[keys].drop_duplicates()
    
    # Create baseline points for every year from -50 to -45 for smooth transition
    baseline_rows_list = []
    for year in range(BASELINE_YEAR, BASELINE_YEAR + 6):  # -50 to -45
        baseline_year_rows = unique_combos.copy()
        baseline_year_rows['year'] = year
        baseline_year_rows['pct_of_baseline'] = BASELINE_PCT
        baseline_rows_list.append(baseline_year_rows)
    
    # Combine all data
    plot_data = pd.concat([df] + baseline_rows_list, ignore_index=True)
    
    # Remove duplicates (keep first)
    plot_data = plot_data.drop_duplicates(subset=keys + ['year'], keep='first')
    
    return plot_data


# =============================================================================
# LOESS SMOOTHING
# =============================================================================

def smooth_series(df, smoothing=None):
    """
    Apply smoothing to full data range including baseline transition.
    
    Uses PCHIP interpolation for smooth monotonic curves.
    
    Parameters:
        df: DataFrame with 'year' and 'pct_of_baseline' columns
        smoothing: Not used with PCHIP
    
    Returns:
        DataFrame with year and pct_smooth columns
    """
    df = df.dropna(subset=['year', 'pct_of_baseline']).sort_values('year')
    
    if len(df) < 3:
        return pd.DataFrame({
            'year': df['year'].values,
            'pct_smooth': df['pct_of_baseline'].values
        })
    
    # Remove duplicates by averaging
    df_unique = df.groupby('year')['pct_of_baseline'].mean().reset_index()
    x = df_unique['year'].values.astype(float)
    y = df_unique['pct_of_baseline'].values.astype(float)
    
    # Generate smooth points for full range (from min to max year)
    # Include transition zone from baseline to year 0
    x_min = x.min()
    x_max = x.max()
    
    # Create points: every year from min to max
    x_smooth = np.arange(int(x_min), int(x_max) + 1, 1).astype(float)
    
    # Use PCHIP for smooth monotonic interpolation
    try:
        pchip = PchipInterpolator(x, y)
        y_smooth = pchip(x_smooth)
    except Exception as e:
        print(f"PCHIP failed, using linear: {e}")
        y_smooth = np.interp(x_smooth, x, y)
    
    # Clamp: no negative values
    y_smooth = np.maximum(y_smooth, 0)
    
    return pd.DataFrame({
        'year': x_smooth,
        'pct_smooth': y_smooth
    })


def apply_loess_smoothing(plot_data, smoothing=None):
    """Apply smoothing to all indicator series."""
    
    group_cols = ['site', 'scenario', 'capability', 'indicator_id', 'indicator_site', 'stream_id']
    
    results = []
    
    for name, group in plot_data.groupby(group_cols):
        smoothed = smooth_series(group, smoothing=smoothing)
        
        # Add group info back
        for i, col in enumerate(group_cols):
            smoothed[col] = name[i]
        
        results.append(smoothed)
    
    loess_data = pd.concat(results, ignore_index=True)
    
    # Calculate y_plot (mirror trending below zero)
    loess_data['y_plot'] = np.where(
        loess_data['scenario'] == 'trending',
        -loess_data['pct_smooth'],
        loess_data['pct_smooth']
    )
    
    # Create hover text
    loess_data['hover'] = (
        'indicator-site: ' + loess_data['indicator_site'] +
        '<br>indicator_id: ' + loess_data['indicator_id'] +
        '<br>capability: ' + loess_data['capability'] +
        '<br>scenario: ' + loess_data['scenario'] +
        '<br>year: ' + loess_data['year'].round(0).astype(int).astype(str) +
        '<br>pct_of_baseline: ' + loess_data['pct_smooth'].round(2).astype(str)
    )
    
    return loess_data


# =============================================================================
# STREAM GRAPH CREATION
# =============================================================================

def create_site_stream(loess_data, site_name, row_num, fig, color_by='capability'):
    """
    Create stream graph traces for a single site.
    
    Positive scenario stacks upward, trending stacks downward (mirrored).
    """
    df = loess_data[loess_data['site'] == site_name].sort_values('year')
    
    # Get color mapping
    if color_by == 'capability':
        color_map = CAPABILITY_COLORS
        color_col = 'capability'
    else:
        color_map = PERSONA_COLORS
        color_col = 'persona'
    
    # Get consistent stream_id order (sorted) - SAME order for both pos and neg
    stream_ids = sorted(df['stream_id'].unique())
    
    # Determine persona for each stream_id (first part before the dot)
    def get_persona(sid):
        return sid.split('.')[0] if '.' in sid else sid
    
    # Find persona boundaries - mark the LAST indicator of each persona
    personas = [get_persona(sid) for sid in stream_ids]
    persona_end_indices = []
    for i in range(len(personas) - 1):
        if personas[i] != personas[i+1]:
            persona_end_indices.append(i)
    # Don't include the very last one - no line needed at outer edge
    
    # --- POSITIVE SCENARIO ---
    df_pos = df[df['scenario'] == 'positive']
    
    # Build pivot table for cumulative sum calculation
    pos_pivot = df_pos.pivot_table(index='year', columns='stream_id', values='y_plot', aggfunc='first')
    pos_pivot = pos_pivot.reindex(columns=stream_ids).fillna(0)
    
    for i, sid in enumerate(stream_ids):
        dd = df_pos[df_pos['stream_id'] == sid].sort_values('year')
        if len(dd) == 0:
            continue
        color_key = dd[color_col].iloc[0] if color_col in dd.columns else 'self'
        base_color = color_map.get(color_key, '#999999')
        
        fill_color = f'rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, {FILL_ALPHA})'
        
        fig.add_trace(
            go.Scatter(
                x=dd['year'],
                y=dd['y_plot'],
                mode='lines',
                stackgroup='pos',
                fill='tonexty',
                line=SEP_LINE,
                fillcolor=fill_color,
                text=dd['hover'],
                hoverinfo='text',
                name=sid,
                showlegend=False
            ),
            row=row_num, col=1
        )
    
    # --- TRENDING SCENARIO ---
    df_neg = df[df['scenario'] == 'trending']
    
    neg_pivot = df_neg.pivot_table(index='year', columns='stream_id', values='y_plot', aggfunc='first')
    neg_pivot = neg_pivot.reindex(columns=stream_ids).fillna(0)
    
    for i, sid in enumerate(stream_ids):
        dd = df_neg[df_neg['stream_id'] == sid].sort_values('year')
        if len(dd) == 0:
            continue
        color_key = dd[color_col].iloc[0] if color_col in dd.columns else 'self'
        base_color = color_map.get(color_key, '#999999')
        
        fill_color = f'rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, {FILL_ALPHA})'
        
        fig.add_trace(
            go.Scatter(
                x=dd['year'],
                y=dd['y_plot'],
                mode='lines',
                stackgroup='neg',
                fill='tonexty',
                line=SEP_LINE,
                fillcolor=fill_color,
                text=dd['hover'],
                hoverinfo='text',
                name=sid,
                showlegend=False
            ),
            row=row_num, col=1
        )
    
    # --- ADD PERSONA SEPARATOR LINES ON TOP ---
    for end_idx in persona_end_indices:
        # Compute cumulative sum up to and including this index
        cols_to_sum = stream_ids[:end_idx + 1]
        
        # Positive scenario
        if len(pos_pivot) > 0:
            cumsum_pos = pos_pivot[cols_to_sum].sum(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=cumsum_pos.index,
                    y=cumsum_pos.values,
                    mode='lines',
                    line=PERSONA_SEP_LINE,
                    hoverinfo='skip',
                    showlegend=False
                ),
                row=row_num, col=1
            )
        
        # Negative scenario
        if len(neg_pivot) > 0:
            cumsum_neg = neg_pivot[cols_to_sum].sum(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=cumsum_neg.index,
                    y=cumsum_neg.values,
                    mode='lines',
                    line=PERSONA_SEP_LINE,
                    hoverinfo='skip',
                    showlegend=False
                ),
                row=row_num, col=1
            )
    
    return fig


def create_stream_graph(loess_data, sites, color_by='capability'):
    """
    Create faceted stream graph with one subplot per site.
    
    Positive scenarios stack upward, trending scenarios stack downward.
    """
    n_sites = len(sites)
    
    fig = make_subplots(
        rows=n_sites, cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.05,
        subplot_titles=sites
    )
    
    for i, site in enumerate(sites):
        create_site_stream(loess_data, site, row_num=i+1, fig=fig, color_by=color_by)
    
    # Define tick values and labels - show baseline, 0, 10, 30, 60, 180
    tickvals = [BASELINE_YEAR, 0, 10, 30, 60, 180]
    ticktext = ['baseline', '0', '10', '30', '60', '180']
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Indicator Stream Graph (smoothed % of baseline)',
            x=0.5
        ),
        height=GRAPH_HEIGHT_PER_SITE * n_sites,
        width=GRAPH_WIDTH,
        autosize=False,  # Force fixed size, don't auto-resize
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        hovermode='closest',
        showlegend=False,
        margin=dict(l=60, r=20, t=80, b=40)
    )
    
    # Update all x-axes - hide default ticks and grid
    for i in range(1, n_sites + 1):
        fig.update_xaxes(
            showticklabels=False,  # Hide default tick labels (we add our own at y=0)
            showline=False,
            zeroline=False,
            showgrid=False,  # Hide grid lines
            row=i, col=1
        )
        
        # Add thick vertical line at x=0
        fig.add_vline(
            x=0, 
            line_width=2, 
            line_color='black',
            row=i, col=1
        )
    
    # Update all y-axes - no labels, lines, or grid
    for i in range(1, n_sites + 1):
        fig.update_yaxes(
            zeroline=False,        # We'll add our own on top
            showline=False,
            showticklabels=False,
            showgrid=False,  # Hide grid lines
            row=i, col=1
        )
        
        # Add thick horizontal line at y=0 ON TOP of traces
        fig.add_hline(
            y=0,
            line_width=2,
            line_color='black',
            layer='above',  # Draw on top of traces
            row=i, col=1
        )
    
    # Get y-range to scale tick marks appropriately
    y_max = loess_data['y_plot'].abs().max()
    tick_height = y_max * 0.03  # 3% of range for tick height
    
    # Add x-axis labels and tick marks at y=0
    for i, site in enumerate(sites):
        for xval, label in zip(tickvals, ticktext):
            # Add tick mark (small vertical line crossing y=0)
            fig.add_shape(
                type='line',
                x0=xval, x1=xval,
                y0=-tick_height, y1=tick_height,
                line=dict(color='black', width=1.5),
                layer='above',
                row=i+1, col=1
            )
            
            # Labels removed per user request
            # if label:
            #     fig.add_annotation(
            #         x=xval,
            #         y=0,
            #         text=label,
            #         showarrow=False,
            #         font=dict(size=12, color='black'),
            #         xanchor='center',
            #         yanchor='bottom',
            #         yshift=10,
            #         row=i+1, col=1
            #     )
    
    # Add legend for capabilities
    for cap, color in (CAPABILITY_COLORS if color_by == 'capability' else PERSONA_COLORS).items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=cap,
                showlegend=True
            )
        )
    
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    return fig


def create_combined_stream_graph(loess_data, sites, color_by='capability'):
    """
    Create a single combined stream graph with all sites stacked together.
    
    Indicators are ordered by: site > persona > indicator
    Site boundaries are marked with thicker lines.
    """
    from plotly.subplots import make_subplots
    
    df = loess_data.copy()
    
    # Get color mapping
    if color_by == 'capability':
        color_map = CAPABILITY_COLORS
        color_col = 'capability'
    else:
        color_map = PERSONA_COLORS
        color_col = 'persona'
    
    # Create ordered stream_ids: site > persona > indicator
    # indicator_site format is "{indicator}-{site}" e.g. "Bird.self.peeling-trimmed-parade"
    # Order: all trimmed-parade first, then city, then uni
    # Within each site: Bird, then Lizard, then Tree
    PERSONA_ORDER = {'Bird': 0, 'Lizard': 1, 'Tree': 2}
    
    def parse_indicator_site(indicator_site):
        """Parse indicator_site into (indicator, site) by checking known site names."""
        for site in sites:
            suffix = f'-{site}'
            if indicator_site.endswith(suffix):
                indicator = indicator_site[:-len(suffix)]
                return indicator, site
        return indicator_site, ''
    
    def sort_key(indicator_site):
        indicator, site = parse_indicator_site(indicator_site)
        persona = indicator.split('.')[0] if '.' in indicator else ''
        site_order = sites.index(site) if site in sites else 999
        persona_order = PERSONA_ORDER.get(persona, 999)
        return (site_order, persona_order, indicator)
    
    stream_ids = sorted(df['indicator_site'].unique(), key=sort_key)
    
    print(f"Stream order (first 6): {stream_ids[:6]}")  # Debug
    
    # Find site and persona boundaries
    def get_site(sid):
        indicator, site = parse_indicator_site(sid)
        return site
    
    def get_persona(sid):
        indicator, site = parse_indicator_site(sid)
        return indicator.split('.')[0] if '.' in indicator else ''
    
    site_end_indices = []
    persona_end_indices = []
    for i in range(len(stream_ids) - 1):
        if get_site(stream_ids[i]) != get_site(stream_ids[i+1]):
            site_end_indices.append(i)
        elif get_persona(stream_ids[i]) != get_persona(stream_ids[i+1]):
            persona_end_indices.append(i)
    
    # Create single subplot
    fig = go.Figure()
    
    # --- POSITIVE SCENARIO ---
    df_pos = df[df['scenario'] == 'positive']
    pos_pivot = df_pos.pivot_table(index='year', columns='indicator_site', values='y_plot', aggfunc='first')
    pos_pivot = pos_pivot.reindex(columns=stream_ids).fillna(0)
    
    for i, sid in enumerate(stream_ids):
        dd = df_pos[df_pos['indicator_site'] == sid].sort_values('year')
        if len(dd) == 0:
            continue
        color_key = dd[color_col].iloc[0] if color_col in dd.columns else 'self'
        base_color = color_map.get(color_key, '#999999')
        
        fill_color = f'rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, {FILL_ALPHA})'
        
        fig.add_trace(
            go.Scatter(
                x=dd['year'],
                y=dd['y_plot'],
                mode='lines',
                stackgroup='pos',
                fill='tonexty',
                line=SEP_LINE,
                fillcolor=fill_color,
                text=dd['hover'],
                hoverinfo='text',
                name=sid,
                showlegend=False
            )
        )
    
    # --- TRENDING SCENARIO ---
    df_neg = df[df['scenario'] == 'trending']
    neg_pivot = df_neg.pivot_table(index='year', columns='indicator_site', values='y_plot', aggfunc='first')
    neg_pivot = neg_pivot.reindex(columns=stream_ids).fillna(0)
    
    for i, sid in enumerate(stream_ids):
        dd = df_neg[df_neg['indicator_site'] == sid].sort_values('year')
        if len(dd) == 0:
            continue
        color_key = dd[color_col].iloc[0] if color_col in dd.columns else 'self'
        base_color = color_map.get(color_key, '#999999')
        
        fill_color = f'rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, {FILL_ALPHA})'
        
        fig.add_trace(
            go.Scatter(
                x=dd['year'],
                y=dd['y_plot'],
                mode='lines',
                stackgroup='neg',
                fill='tonexty',
                line=SEP_LINE,
                fillcolor=fill_color,
                text=dd['hover'],
                hoverinfo='text',
                name=sid,
                showlegend=False
            )
        )
    
    # --- ADD SITE BOUNDARY LINES (thick) ---
    SITE_SEP_LINE = dict(width=2, color='rgba(0,0,0,0.9)')
    for end_idx in site_end_indices:
        cols_to_sum = stream_ids[:end_idx + 1]
        
        if len(pos_pivot) > 0:
            cumsum_pos = pos_pivot[cols_to_sum].sum(axis=1)
            fig.add_trace(go.Scatter(x=cumsum_pos.index, y=cumsum_pos.values,
                mode='lines', line=SITE_SEP_LINE, hoverinfo='skip', showlegend=False))
        
        if len(neg_pivot) > 0:
            cumsum_neg = neg_pivot[cols_to_sum].sum(axis=1)
            fig.add_trace(go.Scatter(x=cumsum_neg.index, y=cumsum_neg.values,
                mode='lines', line=SITE_SEP_LINE, hoverinfo='skip', showlegend=False))
    
    # --- ADD PERSONA BOUNDARY LINES (thin) ---
    for end_idx in persona_end_indices:
        cols_to_sum = stream_ids[:end_idx + 1]
        
        if len(pos_pivot) > 0:
            cumsum_pos = pos_pivot[cols_to_sum].sum(axis=1)
            fig.add_trace(go.Scatter(x=cumsum_pos.index, y=cumsum_pos.values,
                mode='lines', line=PERSONA_SEP_LINE, hoverinfo='skip', showlegend=False))
        
        if len(neg_pivot) > 0:
            cumsum_neg = neg_pivot[cols_to_sum].sum(axis=1)
            fig.add_trace(go.Scatter(x=cumsum_neg.index, y=cumsum_neg.values,
                mode='lines', line=PERSONA_SEP_LINE, hoverinfo='skip', showlegend=False))
    
    # Define tick values
    tickvals = [BASELINE_YEAR, 0, 10, 30, 60, 180]
    ticktext = ['baseline', '0', '10', '30', '60', '180']
    
    # Update layout
    fig.update_layout(
        title=dict(text='Combined Stream Graph (all sites)', x=0.5),
        height=GRAPH_HEIGHT_PER_SITE * 2,  # Double height for combined view
        width=GRAPH_WIDTH,
        autosize=False,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        hovermode='closest',
        showlegend=False,
        margin=dict(l=60, r=20, t=80, b=40)
    )
    
    fig.update_xaxes(showticklabels=False, showline=False, zeroline=False, showgrid=False)
    fig.update_yaxes(zeroline=False, showline=False, showticklabels=False, showgrid=False)
    
    # Add y=0 line
    fig.add_hline(y=0, line_width=2, line_color='black', layer='above')
    # Add x=0 line  
    fig.add_vline(x=0, line_width=2, line_color='black')
    
    # Get y-range for tick marks
    y_max = df['y_plot'].abs().max()
    tick_height = y_max * 0.03
    
    # Add tick marks at y=0 (labels removed per user request)
    for xval, label in zip(tickvals, ticktext):
        fig.add_shape(type='line', x0=xval, x1=xval, y0=-tick_height, y1=tick_height,
            line=dict(color='black', width=1.5), layer='above')
        # Labels removed
        # if label:
        #     fig.add_annotation(x=xval, y=0, text=label, showarrow=False,
        #         font=dict(size=12, color='black'), xanchor='center', yanchor='bottom', yshift=10)
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def generate_stream_graph(sites=None, voxel_size=1, color_by='capability', save=True, show=True,
                          width=None, height_per_site=None, combined=False):
    """
    Generate stream graph for specified sites.
    
    Parameters:
        sites: List of site names (default: all three sites)
        voxel_size: Voxel size (default: 1)
        color_by: 'capability' or 'persona'
        save: Whether to save HTML file
        show: Whether to display in browser
        width: Graph width in pixels (default: GRAPH_WIDTH)
        height_per_site: Height per site subplot in pixels (default: GRAPH_HEIGHT_PER_SITE)
    
    Returns:
        plotly Figure object
    """
    global GRAPH_WIDTH, GRAPH_HEIGHT_PER_SITE
    
    if width is not None:
        GRAPH_WIDTH = width
    if height_per_site is not None:
        GRAPH_HEIGHT_PER_SITE = height_per_site
    
    if sites is None:
        sites = ['trimmed-parade', 'city', 'uni']
    
    print(f"Loading data for sites: {sites}")
    raw_data = load_indicator_data(sites, voxel_size)
    
    print("Preparing plot data with baseline anchors...")
    plot_data = prepare_plot_data(raw_data, sites)
    
    print("Applying smoothing spline...")
    loess_data = apply_loess_smoothing(plot_data, smoothing=SMOOTHING_FACTOR)
    
    if combined:
        print("Creating combined stream graph...")
        fig = create_combined_stream_graph(loess_data, sites, color_by=color_by)
        graph_height = GRAPH_HEIGHT_PER_SITE * 2  # Double height for combined
    else:
        print("Creating faceted stream graph...")
        fig = create_stream_graph(loess_data, sites, color_by=color_by)
        graph_height = GRAPH_HEIGHT_PER_SITE * len(sites)
    
    if save:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        sites_str = '-'.join(sites)
        import time
        timestamp = int(time.time())
        mode = 'combined' if combined else 'faceted'
        filename = f'stream_graph_{mode}_{voxel_size}_{sites_str}_{timestamp}.html'
        filepath = PLOT_DIR / filename
        # Manually set width/height in div style to force fixed size
        html_content = fig.to_html(
            config={'responsive': False},
            include_plotlyjs=True,
            full_html=True,
            default_width=f'{GRAPH_WIDTH}px',
            default_height=f'{graph_height}px'
        )
        with open(filepath, 'w') as f:
            f.write(html_content)
        print(f"Saved HTML: {filepath}")
        
        # Also save as PNG at double resolution (2x DPI)
        png_filename = f'stream_graph_{mode}_{voxel_size}_{sites_str}_{timestamp}.png'
        png_filepath = PLOT_DIR / png_filename
        try:
            # Use original dimensions with scale=2 for 2x DPI (keeps visual proportions)
            fig.write_image(str(png_filepath), width=GRAPH_WIDTH, height=graph_height, scale=2)
            actual_width = GRAPH_WIDTH * 2
            actual_height = graph_height * 2
            print(f"Saved PNG ({actual_width}x{actual_height}px @ 2x DPI): {png_filepath}")
        except Exception as e:
            print(f"Could not save PNG (kaleido may not be installed): {e}")
    
    if show:
        fig.show()
    
    return fig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate stream graphs from indicator counts'
    )
    parser.add_argument('--sites', type=str, default='trimmed-parade,city,uni',
                        help='Comma-separated list of sites')
    parser.add_argument('--voxel-size', type=int, default=1)
    parser.add_argument('--color-by', choices=['capability', 'persona'], default='capability')
    parser.add_argument('--width', type=int, default=None,
                        help=f'Graph width in pixels (default: {GRAPH_WIDTH})')
    parser.add_argument('--height', type=int, default=None,
                        help=f'Height per site subplot in pixels (default: {GRAPH_HEIGHT_PER_SITE})')
    parser.add_argument('--no-show', action='store_true', help='Do not open in browser')
    parser.add_argument('--no-save', action='store_true', help='Do not save HTML file')
    parser.add_argument('--combined', action='store_true', 
                        help='Combine all sites into one stream graph (default: faceted by site)')
    
    args = parser.parse_args()
    
    sites = [s.strip() for s in args.sites.split(',')]
    
    generate_stream_graph(
        sites=sites,
        voxel_size=args.voxel_size,
        color_by=args.color_by,
        save=not args.no_save,
        show=not args.no_show,
        width=args.width,
        height_per_site=args.height,
        combined=args.combined
    )


if __name__ == '__main__':
    main()

