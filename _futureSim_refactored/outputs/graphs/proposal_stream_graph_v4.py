"""
V4 proposal intervention stream graph generator.

Reads per-state V4 intervention CSVs and produces stacked stream graphs of
voxel counts per intervention, grouped by proposal family. Two modes:

- absolute: raw voxel counts on the y-axis
- relative: each proposal family's streams divided by that family's global
  peak stack total across all (site, scenario, year) combinations. On the
  relative chart, the tallest state for any given proposal reads as 1.0.

Three scopes per mode:

- combined: all sites stacked in one figure
- per-site: one figure per site
- per-proposal: one figure per proposal family (scoped to all-sites combined)

USAGE:
    uv run python _futureSim_refactored/outputs/graphs/proposal_stream_graph_v4.py \
        --root _data-refactored/model-outputs/generated-states/4.9
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import PchipInterpolator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _futureSim_refactored.sim.setup.constants import (
    COLONISE_FULL_ENVELOPE,
    COLONISE_FULL_GROUND,
    COLONISE_PARTIAL_ENVELOPE,
    DECAY_FULL,
    DECAY_PARTIAL,
    DEPLOY_FULL_LOG,
    DEPLOY_FULL_POLE,
    DEPLOY_FULL_UPGRADE,
    INTERVENTION_SUPPORT,
    RECRUIT_FULL,
    RECRUIT_PARTIAL,
    RELEASECONTROL_FULL,
    RELEASECONTROL_PARTIAL,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SITES_DEFAULT = ["trimmed-parade", "city", "uni"]
SCENARIOS = ["positive", "trending"]
SCENARIO_LABELS = {
    "positive": "nonhuman-led (positive)",
    "trending": "human-led (trending)",
}
YEAR_TICKS = [0, 10, 30, 60, 90, 120, 150, 180]
X_RANGE_PAD = 4

# Proposal display order (v3 convention)
PROPOSAL_ORDER = [
    "deploy-structure",
    "decay",
    "recruit",
    "colonise",
    "release-control",
]

# Intervention stacking order within each proposal family
INTERVENTION_ORDER_BY_PROPOSAL = {
    "deploy-structure": [DEPLOY_FULL_POLE, DEPLOY_FULL_LOG, DEPLOY_FULL_UPGRADE],
    "decay": [DECAY_FULL, DECAY_PARTIAL],
    "recruit": [RECRUIT_FULL, RECRUIT_PARTIAL],
    "colonise": [COLONISE_FULL_GROUND, COLONISE_FULL_ENVELOPE, COLONISE_PARTIAL_ENVELOPE],
    "release-control": [RELEASECONTROL_FULL, RELEASECONTROL_PARTIAL],
}

# Family colours taken verbatim from the V4 compositor
# (_futureSim_refactored/blender/compositor/scripts/_set_proposal_colors.py).
PROPOSAL_FAMILY_COLORS = {
    "colonise": "#FF8C00",          # (1.00, 0.55, 0.00)
    "decay": "#E62626",              # (0.90, 0.15, 0.15)
    "deploy-structure": "#2659FF",  # (0.15, 0.35, 1.00)
    "recruit": "#26BF26",            # (0.15, 0.75, 0.15)
    "release-control": "#8C1AD9",   # (0.55, 0.10, 0.85)
}

# Every intervention is a lightened shade of its family colour. Partials are
# lightened further so they read as a softer version of the same family.
SUPPORT_LIGHTNESS = {
    "full": 0.25,       # 25% toward white
    "partial": 0.60,    # 60% toward white
}

GRAPH_WIDTH = 1100
GRAPH_HEIGHT_COMBINED = 900
GRAPH_HEIGHT_PER_SITE = 520
GRAPH_HEIGHT_PER_PROPOSAL = 420

Y_HEADROOM_FACTOR = 1.08
# Thin line between consecutive interventions within the same family
# (analogous to the persona boundary lines in the indicator stream graph).
INTERVENTION_SEPARATOR = dict(width=0.8, color="rgba(0, 0, 0, 0.55)")
# Thicker line between proposal families (analogous to the site boundary lines).
PROPOSAL_SEPARATOR = dict(width=1.8, color="rgba(0, 0, 0, 0.92)")


def lighten_hex(hex_color: str, amount: float) -> str:
    """Blend ``hex_color`` toward white by ``amount`` (0..1)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(round(r + (255 - r) * amount))
    g = int(round(g + (255 - g) * amount))
    b = int(round(b + (255 - b) * amount))
    return f"#{r:02x}{g:02x}{b:02x}"


def stream_fill_hex(proposal: str, support: str) -> str:
    base = PROPOSAL_FAMILY_COLORS.get(proposal, "#888888")
    return lighten_hex(base, SUPPORT_LIGHTNESS.get(support, 0.0))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_intervention_csvs(root: Path, sites: list[str]) -> pd.DataFrame:
    stats_dir = root / "output" / "stats" / "per-state"
    frames: list[pd.DataFrame] = []
    for site in sites:
        site_dir = stats_dir / site
        if not site_dir.exists():
            print(f"Warning: missing stats dir {site_dir}")
            continue
        for csv_path in sorted(site_dir.glob(f"{site}_*_v4_interventions.csv")):
            frames.append(pd.read_csv(csv_path))
    if not frames:
        raise FileNotFoundError(f"No V4 intervention CSVs found under {stats_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(float)
    df = df[df["scenario"].isin(SCENARIOS)].copy()
    df = df[df["year"].isin(YEAR_TICKS)].copy()
    return df


# =============================================================================
# STREAM PREPARATION
# =============================================================================

def proposal_order_key(proposal: str) -> int:
    try:
        return PROPOSAL_ORDER.index(proposal)
    except ValueError:
        return len(PROPOSAL_ORDER)


def intervention_order_key(proposal: str, intervention: str) -> int:
    order = INTERVENTION_ORDER_BY_PROPOSAL.get(proposal, [])
    try:
        return order.index(intervention)
    except ValueError:
        return len(order)


def build_streams(scope_df: pd.DataFrame, proposals: list[str]) -> list[dict]:
    """Build one stream per (proposal, intervention) present in scope_df."""
    streams: list[dict] = []
    for proposal in proposals:
        family_df = scope_df[scope_df["proposal"] == proposal]
        if family_df.empty:
            continue
        for intervention in INTERVENTION_ORDER_BY_PROPOSAL.get(proposal, []):
            inter_df = family_df[family_df["intervention"] == intervention]
            if inter_df.empty:
                continue
            aggregated = (
                inter_df.groupby(["scenario", "year"], as_index=False)["count"].sum()
            )
            year_series = {
                scenario: aggregated[aggregated["scenario"] == scenario]
                    .set_index("year")["count"]
                    .reindex(YEAR_TICKS, fill_value=0.0)
                for scenario in SCENARIOS
            }
            support = INTERVENTION_SUPPORT.get(intervention, "full")
            streams.append(
                {
                    "proposal": proposal,
                    "intervention": intervention,
                    "support": support,
                    "year_series": year_series,
                    "order": (proposal_order_key(proposal), intervention_order_key(proposal, intervention)),
                    "color": stream_fill_hex(proposal, support),
                    "opacity": 1.0,
                }
            )
    streams.sort(key=lambda s: s["order"])
    return streams


def compute_global_proposal_peaks(raw_df: pd.DataFrame) -> dict[str, float]:
    """For each proposal family, return the max stack-total after summing across
    sites. Using the summed-across-sites aggregation means every family tops
    out at 1.0 in the combined (all-sites) view, so families are directly
    comparable on a single chart. Per-site charts then show each site's share
    of that same all-sites peak (always ≤1.0)."""
    totals = (
        raw_df.groupby(["proposal", "scenario", "year"], as_index=False)["count"].sum()
    )
    return {
        proposal: max(float(group["count"].max()), 1.0)
        for proposal, group in totals.groupby("proposal")
    }


def scale_streams(streams: list[dict], peaks: dict[str, float]) -> list[dict]:
    """Return a deep copy of streams with each year_series divided by its family's peak."""
    scaled: list[dict] = []
    for stream in streams:
        scale = peaks.get(stream["proposal"], 1.0) or 1.0
        copy = dict(stream)
        copy["year_series"] = {
            scenario: stream["year_series"][scenario].astype(float) / scale
            for scenario in SCENARIOS
        }
        scaled.append(copy)
    return scaled


# =============================================================================
# SMOOTHING
# =============================================================================

def smooth_series(years: list[int], values: list[float]) -> tuple[list[float], list[float]]:
    x = np.asarray(years, dtype=float)
    y = np.asarray([max(0.0, v) for v in values], dtype=float)
    if len(x) < 3:
        padded_x = [x[0] - X_RANGE_PAD] + x.tolist() + [x[-1] + X_RANGE_PAD]
        padded_y = [float(y[0])] + y.tolist() + [float(y[-1])]
        return padded_x, padded_y

    x_smooth = np.arange(int(x.min()), int(x.max()) + 1, 1).astype(float)
    try:
        y_smooth = PchipInterpolator(x, y)(x_smooth)
    except Exception:
        y_smooth = np.interp(x_smooth, x, y)
    y_smooth = np.maximum(y_smooth, 0.0)
    return (
        [x_smooth[0] - X_RANGE_PAD] + x_smooth.tolist() + [x_smooth[-1] + X_RANGE_PAD],
        [float(y_smooth[0])] + y_smooth.astype(float).tolist() + [float(y_smooth[-1])],
    )


def max_stack_total(streams: list[dict], scenario: str) -> float:
    if not streams:
        return 0.0
    total = None
    for stream in streams:
        series = stream["year_series"][scenario].astype(float)
        total = series if total is None else total.add(series, fill_value=0.0)
    return float(total.max()) if total is not None else 0.0


def paired_half_range(streams: list[dict]) -> float:
    peak = max(
        max_stack_total(streams, "positive"),
        max_stack_total(streams, "trending"),
        1.0,
    )
    return (peak / 2.0) * Y_HEADROOM_FACTOR


def mirrored_half_range(streams: list[dict]) -> float:
    """Half-range when positive stacks above zero and trending stacks below.

    Unlike ``paired_half_range`` the full stack height lives on one side of
    zero, so we do not divide the peak in half.
    """
    peak = max(
        max_stack_total(streams, "positive"),
        max_stack_total(streams, "trending"),
        1.0,
    )
    return peak * Y_HEADROOM_FACTOR


# =============================================================================
# FIGURE BUILDING
# =============================================================================

def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def format_hover_value(value: float, mode: str) -> str:
    if mode == "relative":
        return f"{value * 100:.1f}% of family peak"
    return f"{int(round(value)):,} voxels"


def build_hover_texts(
    stream: dict,
    dense_years: list[float],
    smooth_values: list[float],
    scenario: str,
    mode: str,
) -> list[str]:
    label_header = f"<b>{stream['proposal']}</b><br>{stream['intervention']} ({stream['support']})"
    scenario_label = SCENARIO_LABELS.get(scenario, scenario)
    texts: list[str] = []
    for year, value in zip(dense_years, smooth_values):
        texts.append(
            f"{label_header}"
            f"<br>scenario: {scenario_label}"
            f"<br>year: {int(round(year))}"
            f"<br>{format_hover_value(max(0.0, float(value)), mode)}"
        )
    return texts


def add_hover_line(
    fig: go.Figure,
    years: list[float],
    upper: list[float],
    hover_texts: list[str],
    row: int | None,
) -> None:
    """Invisible line trace along the upper edge of a band that carries
    per-year tooltips. Uses a zero-width transparent line so it doesn't
    appear visually but is still pickable by Plotly's hover system."""
    trace = go.Scatter(
        x=years,
        y=upper,
        mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        text=hover_texts,
        hoverinfo="text",
        showlegend=False,
    )
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=1)


def separator_line_for(stream: dict, next_stream: dict | None) -> dict | None:
    """Pick the separator line spec to draw on top of ``stream``.

    - No next stream: no separator (we don't cap the outermost edge).
    - Same family: thin intervention separator.
    - Different family: thicker proposal-family separator.
    """
    if next_stream is None:
        return None
    if next_stream["proposal"] != stream["proposal"]:
        return PROPOSAL_SEPARATOR
    return INTERVENTION_SEPARATOR


def add_stream_band(
    fig: go.Figure,
    row: int,
    years: list[float],
    lower: list[float],
    upper: list[float],
    stream: dict,
    showlegend: bool,
    separator_line: dict | None,
) -> None:
    fill_color = rgba(stream["color"], stream["opacity"])
    legend_label = f"{stream['proposal']} / {stream['intervention']} ({stream['support']})"
    fig.add_trace(
        go.Scatter(
            x=years + years[::-1],
            y=upper + lower[::-1],
            mode="none",
            fill="toself",
            fillcolor=fill_color,
            hoverinfo="skip",
            name=legend_label,
            legendgroup=legend_label,
            showlegend=showlegend,
        ),
        row=row,
        col=1,
    )
    if separator_line is not None:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=upper,
                mode="lines",
                line=separator_line,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=1,
        )


def build_two_panel_figure(
    streams: list[dict],
    title: str,
    height: int,
    mode: str,
    half_range: float | None = None,
    proposal_separators: bool = False,
) -> go.Figure:
    """Two rows: trending on top, positive on bottom. Symmetric centered stack."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[SCENARIO_LABELS["trending"], SCENARIO_LABELS["positive"]],
    )

    if half_range is None:
        half_range = paired_half_range(streams)

    for row_index, scenario in enumerate(["trending", "positive"], start=1):
        scenario_series = [
            (stream, stream["year_series"][scenario].astype(float).tolist())
            for stream in streams
        ]
        totals = [sum(values[i] for _, values in scenario_series) for i in range(len(YEAR_TICKS))]
        dense_years, smooth_totals = smooth_series(YEAR_TICKS, totals)
        baseline = [-(t / 2.0) for t in smooth_totals]

        for stream_index, (stream, values) in enumerate(scenario_series):
            _, smooth_values = smooth_series(YEAR_TICKS, values)
            lower = list(baseline)
            upper = [baseline[i] + smooth_values[i] for i in range(len(smooth_values))]
            baseline = upper
            next_stream = scenario_series[stream_index + 1][0] if stream_index + 1 < len(scenario_series) else None
            add_stream_band(
                fig=fig,
                row=row_index,
                years=dense_years,
                lower=lower,
                upper=upper,
                stream=stream,
                showlegend=(row_index == 1),
                separator_line=separator_line_for(stream, next_stream),
            )
            hover_texts = build_hover_texts(stream, dense_years, smooth_values, scenario, mode)
            add_hover_line(fig, dense_years, upper, hover_texts, row=row_index)

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, font=dict(size=16, color="#27496d")),
        width=GRAPH_WIDTH,
        height=height,
        margin=dict(l=50, r=30, t=90, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Aptos, Arial, sans-serif", size=11, color="#1f2933"),
        legend=dict(
            orientation="h",
            y=-0.14,
            x=0.5,
            xanchor="center",
            title_text="",
            font=dict(size=10, color="#111111"),
        ),
        hovermode="closest",
        showlegend=True,
    )
    for annotation in fig.layout.annotations:
        annotation.font.size = 12

    for row_index in [1, 2]:
        fig.update_xaxes(
            row=row_index,
            col=1,
            range=[YEAR_TICKS[0] - X_RANGE_PAD, YEAR_TICKS[-1] + X_RANGE_PAD],
            tickmode="array",
            tickvals=[YEAR_TICKS[0], YEAR_TICKS[-1]],
            ticktext=[str(YEAR_TICKS[0]), str(YEAR_TICKS[-1])],
            showgrid=False,
            zeroline=False,
            showline=(row_index == 2),
            linecolor="#111111",
            linewidth=1.2,
            ticks="outside" if row_index == 2 else "",
            ticklen=7 if row_index == 2 else 0,
            tickwidth=1.2,
            tickcolor="#111111",
            showticklabels=(row_index == 2),
            title_text="Year" if row_index == 2 else None,
        )
        fig.update_yaxes(
            row=row_index,
            col=1,
            range=[-half_range, half_range],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False,
            title_text="",
        )
    return fig


def build_mirrored_figure(
    streams: list[dict],
    title: str,
    height: int,
    mode: str,
    half_range: float | None = None,
    proposal_separators: bool = False,
) -> go.Figure:
    """Single panel: positive stacked above zero, trending mirrored below zero.

    Matches the visual convention of the V4 indicator stream graph — one
    shared x-axis with a horizontal baseline at y=0, positive-scenario streams
    stacking up, trending-scenario streams stacking down.
    """
    fig = go.Figure()

    if half_range is None:
        half_range = mirrored_half_range(streams)

    for scenario, sign in [("positive", 1.0), ("trending", -1.0)]:
        scenario_series = [
            (stream, stream["year_series"][scenario].astype(float).tolist())
            for stream in streams
        ]
        baseline: list[float] | None = None
        for stream_index, (stream, values) in enumerate(scenario_series):
            dense_years, smooth_values = smooth_series(YEAR_TICKS, values)
            if baseline is None:
                baseline = [0.0] * len(dense_years)
            lower = list(baseline)
            upper = [baseline[i] + sign * smooth_values[i] for i in range(len(smooth_values))]
            baseline = upper

            fill_color = rgba(stream["color"], stream["opacity"])
            legend_label = f"{stream['proposal']} / {stream['intervention']} ({stream['support']})"
            fig.add_trace(
                go.Scatter(
                    x=dense_years + dense_years[::-1],
                    y=upper + lower[::-1],
                    mode="none",
                    fill="toself",
                    fillcolor=fill_color,
                    hoverinfo="skip",
                    name=legend_label,
                    legendgroup=legend_label,
                    showlegend=(scenario == "positive"),
                )
            )

            next_stream = scenario_series[stream_index + 1][0] if stream_index + 1 < len(scenario_series) else None
            sep_line = separator_line_for(stream, next_stream)
            if sep_line is not None:
                fig.add_trace(
                    go.Scatter(
                        x=dense_years,
                        y=upper,
                        mode="lines",
                        line=sep_line,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            hover_texts = build_hover_texts(stream, dense_years, smooth_values, scenario, mode)
            add_hover_line(fig, dense_years, upper, hover_texts, row=None)

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, font=dict(size=16, color="#27496d")),
        width=GRAPH_WIDTH,
        height=height,
        margin=dict(l=50, r=30, t=90, b=100),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Aptos, Arial, sans-serif", size=11, color="#1f2933"),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0.5,
            xanchor="center",
            title_text="",
            font=dict(size=10, color="#111111"),
        ),
        hovermode="closest",
        showlegend=True,
    )
    fig.add_hline(y=0, line_width=1.5, line_color="#111111")
    fig.update_xaxes(
        range=[YEAR_TICKS[0] - X_RANGE_PAD, YEAR_TICKS[-1] + X_RANGE_PAD],
        tickmode="array",
        tickvals=[YEAR_TICKS[0], YEAR_TICKS[-1]],
        ticktext=[str(YEAR_TICKS[0]), str(YEAR_TICKS[-1])],
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="#111111",
        linewidth=1.2,
        ticks="outside",
        ticklen=7,
        tickwidth=1.2,
        tickcolor="#111111",
        showticklabels=True,
        title_text="Year",
    )
    fig.update_yaxes(
        range=[-half_range, half_range],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        showline=False,
        title_text="",
    )
    return fig


# =============================================================================
# OUTPUT
# =============================================================================

def save_figure(fig: go.Figure, out_dir: Path, stem: str, height: int) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{stem}.html"
    html_path.write_text(
        fig.to_html(
            config={"responsive": False},
            include_plotlyjs=True,
            full_html=True,
            default_width=f"{GRAPH_WIDTH}px",
            default_height=f"{height}px",
        )
    )
    print(f"Saved HTML: {html_path}")
    png_path = out_dir / f"{stem}.png"
    try:
        fig.write_image(str(png_path), width=GRAPH_WIDTH, height=height, scale=2)
        print(f"Saved PNG:  {png_path}")
    except Exception as exc:
        print(f"PNG export skipped for {png_path.name}: {exc}")
        png_path = None
    return html_path, png_path


def generate(root: Path, sites: list[str]) -> None:
    print(f"Loading intervention CSVs from {root}")
    raw_df = load_intervention_csvs(root, sites)

    available_proposals = [p for p in PROPOSAL_ORDER if p in raw_df["proposal"].unique()]
    if not available_proposals:
        raise ValueError("No known proposals found in the loaded intervention data")

    peaks = compute_global_proposal_peaks(raw_df)
    print(f"Global proposal peaks: {peaks}")

    root_name = root.name
    ts = int(time.time())
    plots_base = root / "output" / "plots" / "proposal-intervention-streamgraphs"
    filename_prefix = f"proposal_stream_graph_v4_{root_name}"

    # Scopes: ("all-sites-combined", full raw_df) + one per site
    scopes: list[tuple[str, pd.DataFrame]] = [("all-sites-combined", raw_df)]
    scopes.extend([(f"site-{site}", raw_df[raw_df["site"] == site].copy()) for site in sites])

    # Precompute absolute + relative streams per scope
    scope_streams_abs: dict[str, list[dict]] = {}
    scope_streams_rel: dict[str, list[dict]] = {}
    for slug, scope_df in scopes:
        abs_streams = build_streams(scope_df, available_proposals)
        scope_streams_abs[slug] = abs_streams
        scope_streams_rel[slug] = scale_streams(abs_streams, peaks)

    # Shared y-ranges so different scopes/figures stay comparable
    abs_combined_range = max(
        (paired_half_range(scope_streams_abs[slug]) for slug, _ in scopes),
        default=1.0,
    )
    rel_combined_range = max(
        (paired_half_range(scope_streams_rel[slug]) for slug, _ in scopes),
        default=1.0,
    )
    abs_mirrored_range = mirrored_half_range(scope_streams_abs["all-sites-combined"])
    rel_mirrored_range = mirrored_half_range(scope_streams_rel["all-sites-combined"])

    # Per-proposal ranges use the full dataset (all sites)
    per_proposal_abs_range = 0.0
    per_proposal_rel_range = 0.0
    per_proposal_streams_abs: dict[str, list[dict]] = {}
    per_proposal_streams_rel: dict[str, list[dict]] = {}
    for proposal in available_proposals:
        streams_abs = build_streams(raw_df, [proposal])
        streams_rel = scale_streams(streams_abs, peaks)
        per_proposal_streams_abs[proposal] = streams_abs
        per_proposal_streams_rel[proposal] = streams_rel
        per_proposal_abs_range = max(per_proposal_abs_range, paired_half_range(streams_abs))
        per_proposal_rel_range = max(per_proposal_rel_range, paired_half_range(streams_rel))
    per_proposal_abs_range = max(per_proposal_abs_range, 1.0)
    per_proposal_rel_range = max(per_proposal_rel_range, 1.0)

    # ---- Emit figures ------------------------------------------------------
    for mode, all_streams_by_scope, combined_range, mirrored_range, per_proposal_streams, per_proposal_range in [
        ("absolute", scope_streams_abs, abs_combined_range, abs_mirrored_range, per_proposal_streams_abs, per_proposal_abs_range),
        ("relative", scope_streams_rel, rel_combined_range, rel_mirrored_range, per_proposal_streams_rel, per_proposal_rel_range),
    ]:
        mode_dir = plots_base / mode

        # Combined across all sites (all proposals)
        combined_streams = all_streams_by_scope["all-sites-combined"]
        if combined_streams:
            title = f"{root_name} — {mode} — all sites combined, all proposals"
            fig = build_two_panel_figure(
                combined_streams,
                title=title,
                height=GRAPH_HEIGHT_COMBINED,
                mode=mode,
                half_range=combined_range,
                proposal_separators=True,
            )
            save_figure(
                fig,
                mode_dir / "combined",
                f"{filename_prefix}_combined_all-sites_{ts}",
                height=GRAPH_HEIGHT_COMBINED,
            )

            # Mirrored single-panel variant (indicator-stream-graph style):
            # positive stacks above zero, trending mirrors below zero.
            title_mirror = f"{root_name} — {mode} — all sites combined, all proposals (mirrored)"
            fig_mirror = build_mirrored_figure(
                combined_streams,
                title=title_mirror,
                height=GRAPH_HEIGHT_COMBINED,
                mode=mode,
                half_range=mirrored_range,
                proposal_separators=True,
            )
            save_figure(
                fig_mirror,
                mode_dir / "combined-mirrored",
                f"{filename_prefix}_combined-mirrored_all-sites_{ts}",
                height=GRAPH_HEIGHT_COMBINED,
            )

        # Per-site (all proposals), both scenarios
        for slug, _ in scopes[1:]:
            streams = all_streams_by_scope[slug]
            if not streams:
                continue
            title = f"{root_name} — {mode} — {slug}, all proposals"
            fig = build_two_panel_figure(
                streams,
                title=title,
                height=GRAPH_HEIGHT_PER_SITE,
                mode=mode,
                half_range=combined_range,
                proposal_separators=True,
            )
            save_figure(
                fig,
                mode_dir / "per-site",
                f"{filename_prefix}_per-site_{slug}_{ts}",
                height=GRAPH_HEIGHT_PER_SITE,
            )

        # Per-proposal (all sites combined), both scenarios
        for proposal in available_proposals:
            streams = per_proposal_streams[proposal]
            if not streams:
                continue
            title = f"{root_name} — {mode} — all sites combined, proposal={proposal}"
            fig = build_two_panel_figure(
                streams,
                title=title,
                height=GRAPH_HEIGHT_PER_PROPOSAL,
                mode=mode,
                half_range=per_proposal_range,
                proposal_separators=False,
            )
            save_figure(
                fig,
                mode_dir / "per-proposal",
                f"{filename_prefix}_per-proposal_{proposal}_{ts}",
                height=GRAPH_HEIGHT_PER_PROPOSAL,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V4 proposal intervention stream graphs.")
    parser.add_argument("--root", required=True, help="Run output root, e.g. _data-refactored/model-outputs/generated-states/4.9")
    parser.add_argument("--sites", type=str, default=",".join(SITES_DEFAULT))
    args = parser.parse_args()

    root = Path(args.root)
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    generate(root, sites)


if __name__ == "__main__":
    main()
