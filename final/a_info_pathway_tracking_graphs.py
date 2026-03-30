"""
a_info_pathway_tracking_graphs.py
=================================
Generate proposal/intervention stream graphs from the raw intervention statistics.

The script reads:
    - _statistics-refactored/raw/{site}/interventions.csv

It writes PNG output families under:
    1. proposal-intervention-streamgraphs/relative/{combined,per-site,per-proposal}
    2. proposal-intervention-streamgraphs/absolute/{combined,per-site,per-proposal}

Each figure has two scenario panels:
    - nonhuman-led (positive)
    - human-led (trending)

Within each panel, one stream is stacked per intervention using a single
canonical measure for that intervention stream. Voxel-like measures are
preferred when available.

Usage:
    python final/a_info_pathway_tracking_graphs.py
    python final/a_info_pathway_tracking_graphs.py --sites city,uni
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent
RAW_DIR = REPO_DIR / "_statistics-refactored" / "raw"
DEFAULT_OUTPUT_DIR = (
    REPO_DIR
    / "_statistics-refactored"
    / "plots"
    / "pathway_tracking"
    / "proposal-intervention-streamgraphs"
)

PAGE_WIDTH_PT = 45 * 12 + 4.252
PAGE_MARGIN_PT = 3 * 12 + 2.608
CONTENT_WIDTH_PT = PAGE_WIDTH_PT - (2 * PAGE_MARGIN_PT)
EXPORT_DPI = 600

YEAR_TICKS = [0, 10, 30, 60, 90, 120, 150, 180]
X_RANGE_PAD = 4
SCENARIOS = ["positive", "trending"]
SCENARIO_LABELS = {
    "positive": "nonhuman-led (positive)",
    "trending": "human-led (trending)",
}
SCENARIO_COLORS = {
    "positive": "#2f6f9f",
    "trending": "#cf7b32",
}

PROPOSAL_ORDER = [
    "Deploy-Structure",
    "Decay",
    "Recruit",
    "Colonise",
    "Release-Control",
]

STREAM_ORDER_BY_PROPOSAL = {
    "Deploy-Structure": ["Adapt-Utility-Pole", "Upgrade-Feature"],
    "Decay": ["Buffer-Feature", "Brace-Feature"],
    "Recruit": ["Buffer-Feature", "Rewild-Ground"],
    "Colonise": ["Rewild-Ground", "Enrich-Envelope", "Roughen-Envelope"],
    "Release-Control": ["Buffer-Feature", "Brace-Feature"],
}

CANONICAL_MEASURE_PRIORITY = {
    ("Deploy-Structure", "Adapt-Utility-Pole", "full"): [
        "artificial canopy voxels",
        "utility poles",
    ],
    ("Deploy-Structure", "Upgrade-Feature", "full"): [
        "upgraded-feature voxels",
    ],
    ("Decay", "Buffer-Feature", "full"): [
        "buffer-feature voxels",
        "senescing trees",
    ],
    ("Decay", "Brace-Feature", "partial"): [
        "brace-feature voxels",
        "senescing trees",
    ],
    ("Recruit", "Buffer-Feature", "partial"): [
        "recruit grassland voxels",
    ],
    ("Recruit", "Rewild-Ground", "full"): [
        "recruit grassland voxels",
    ],
    ("Colonise", "Rewild-Ground", "mixed"): [
        "rewilded ground voxels",
    ],
    ("Colonise", "Enrich-Envelope", "full"): [
        "green roof voxels",
        "enabled rooftop logs",
    ],
    ("Colonise", "Roughen-Envelope", "partial"): [
        "roughened envelope voxels",
    ],
    ("Release-Control", "Buffer-Feature", "full"): [
        "arboreal voxels",
    ],
    ("Release-Control", "Brace-Feature", "partial"): [
        "arboreal voxels",
    ],
}

STREAM_COLORS = {
    "Adapt-Utility-Pole": "#1f6f8b",
    "Upgrade-Feature": "#7c5ba6",
    "Buffer-Feature": "#d97a3a",
    "Brace-Feature": "#c0a02d",
    "Rewild-Ground": "#4f8b57",
    "Enrich-Envelope": "#8a63d2",
    "Roughen-Envelope": "#8e6b52",
}

PROPOSAL_BASE_COLORS = {
    "Deploy-Structure": "#0f766e",
    "Decay": "#d97706",
    "Recruit": "#65a30d",
    "Colonise": "#7c3aed",
    "Release-Control": "#2563eb",
}

FIGURE_WIDTH_PX = round((CONTENT_WIDTH_PT / 72.0) * EXPORT_DPI)
FIGURE_HEIGHT_BASE_PX = 420
FIGURE_HEIGHT_PER_STREAM_PX = 28
Y_HEADROOM_FACTOR = 2.00


def slugify(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
    )


def discover_sites() -> list[str]:
    sites = [path.parent.name for path in sorted(RAW_DIR.glob("*/interventions.csv"))]
    ordered = [site for site in ["trimmed-parade", "city", "uni"] if site in sites]
    ordered.extend([site for site in sites if site not in ordered])
    return ordered


def load_raw_interventions(sites: list[str] | None = None) -> pd.DataFrame:
    if sites:
        paths = [RAW_DIR / site / "interventions.csv" for site in sites]
    else:
        paths = sorted(RAW_DIR.glob("*/interventions.csv"))

    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["site"] = frame["site"].astype(str)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No raw intervention tables found under {RAW_DIR}")

    df = pd.concat(frames, ignore_index=True)
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df = df[df["scenario"].isin(SCENARIOS)].copy()
    df = df[df["year"].isin(YEAR_TICKS)].copy()
    df["year"] = df["year"].astype(int)
    return df


def normalize_site_filter(all_sites: list[str], requested: str | None) -> list[str]:
    if not requested:
        return all_sites
    wanted = [item.strip() for item in requested.split(",") if item.strip()]
    available = set(all_sites)
    return [site for site in wanted if site in available]


def normalize_proposal_filter(all_proposals: list[str], requested: str | None) -> list[str]:
    if not requested:
        return all_proposals
    wanted = [item.strip() for item in requested.split(",") if item.strip()]
    available = set(all_proposals)
    ordered = [proposal for proposal in wanted if proposal in available]
    return ordered


def canonical_measure_for_stream(
    proposal_label: str,
    intervention: str,
    support: str,
    measures: pd.Series,
) -> str:
    unique_measures: list[str] = []
    for measure in measures.dropna().astype(str).tolist():
        measure = measure.strip()
        if measure and measure not in unique_measures:
            unique_measures.append(measure)

    priority = CANONICAL_MEASURE_PRIORITY.get((proposal_label, intervention, support), [])
    for preferred in priority:
        if preferred in unique_measures:
            return preferred

    voxel_like = [measure for measure in unique_measures if "voxel" in measure.lower()]
    if voxel_like:
        return voxel_like[0]

    if unique_measures:
        return unique_measures[0]

    raise ValueError(
        f"No measure candidates found for {proposal_label} / {intervention} / {support}"
    )


def proposal_stream_order(proposal_label: str, intervention: str, support: str) -> tuple[int, int]:
    stream_order = STREAM_ORDER_BY_PROPOSAL.get(proposal_label, [])
    intervention_rank = stream_order.index(intervention) if intervention in stream_order else len(stream_order)
    support_rank = {"full": 0, "mixed": 1, "partial": 2}.get(support, 9)
    return intervention_rank, support_rank


def prepare_streams(scope_df: pd.DataFrame, proposal_label: str) -> list[dict]:
    proposal_df = scope_df[
        (scope_df["proposal_label"] == proposal_label)
        & (scope_df["scenario"].isin(SCENARIOS))
    ].copy()

    if proposal_df.empty:
        return []

    streams: list[dict] = []
    for (intervention, support), stream_df in proposal_df.groupby(["intervention", "support"], sort=False):
        canonical_measure = canonical_measure_for_stream(
            proposal_label,
            intervention,
            support,
            stream_df["measure"],
        )
        canonical_df = stream_df[stream_df["measure"] == canonical_measure].copy()
        if canonical_df.empty:
            continue

        aggregated = (
            canonical_df.groupby(["scenario", "year"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "series_value"})
        )

        year_series = {
            scenario: (
                aggregated[aggregated["scenario"] == scenario]
                .set_index("year")["series_value"]
                .reindex(YEAR_TICKS, fill_value=0.0)
            )
            for scenario in SCENARIOS
        }

        streams.append(
            {
                "proposal_label": proposal_label,
                "intervention": intervention,
                "support": support,
                "canonical_measure": canonical_measure,
                "legend_label": intervention,
                "display_label": f"{proposal_label} / {intervention} ({support})",
                "year_series": year_series,
                "order_key": proposal_stream_order(proposal_label, intervention, support),
            }
        )

    streams.sort(key=lambda item: item["order_key"])
    return streams


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def mix_hex(hex_color: str, target_hex: str, amount: float) -> str:
    source = hex_color.lstrip("#")
    target = target_hex.lstrip("#")
    sr, sg, sb = int(source[0:2], 16), int(source[2:4], 16), int(source[4:6], 16)
    tr, tg, tb = int(target[0:2], 16), int(target[2:4], 16), int(target[4:6], 16)
    r = round(sr + (tr - sr) * amount)
    g = round(sg + (tg - sg) * amount)
    b = round(sb + (tb - sb) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def combined_stream_color(stream: dict) -> str:
    base = PROPOSAL_BASE_COLORS.get(stream["proposal_label"], "#5c6f82")
    support = stream.get("support", "")
    intervention_rank = int(stream.get("order_key", (0, 0))[0]) if isinstance(stream.get("order_key"), tuple) else 0

    if support == "full":
        color = mix_hex(base, "#000000", 0.06)
    elif support == "mixed":
        color = mix_hex(base, "#ffffff", 0.16)
    else:
        color = mix_hex(base, "#ffffff", 0.34)

    if intervention_rank > 0:
        tweak = min(0.10, 0.04 * intervention_rank)
        color = mix_hex(color, "#ffffff", tweak)
    return color


def scenario_fill_color(base_color: str, scenario: str) -> str:
    if scenario == "positive":
        return mix_hex(base_color, "#ffffff", 0.12)
    return mix_hex(base_color, "#000000", 0.06)


def max_stack_total(streams: list[dict], scenario: str) -> float:
    if not streams:
        return 0.0
    total = None
    for stream in streams:
        series = stream["year_series"][scenario].astype(float)
        total = series if total is None else total.add(series, fill_value=0.0)
    return float(total.max()) if total is not None else 0.0


def smooth_series(years: list[int], values: list[float]) -> tuple[list[float], list[float]]:
    dense_years = [float(year) for year in years]
    dense_values = [max(0.0, float(value)) for value in values]
    if not dense_years:
        return dense_years, dense_values

    # Add constant endcaps beyond the first/last timestep so filled bands do not
    # terminate flush against the plot boundary and appear horizontally clipped.
    return (
        [dense_years[0] - X_RANGE_PAD] + dense_years + [dense_years[-1] + X_RANGE_PAD],
        [dense_values[0]] + dense_values + [dense_values[-1]],
    )


def add_stream_band(
    fig: go.Figure,
    row: int,
    years: list[float],
    lower: list[float],
    upper: list[float],
    mid: list[float],
    values: list[float],
    stream: dict,
    scenario: str,
    showlegend: bool,
    separator: bool = False,
) -> None:
    color = stream.get("color") or STREAM_COLORS.get(stream["intervention"], "#5c6f82")
    legend_label = f"{stream['legend_label']} — {stream['canonical_measure']}"
    fill_color = scenario_fill_color(color, scenario)

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

    if separator:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=upper,
                mode="lines",
                line=dict(width=1.0, color="rgba(0, 0, 0, 0.78)", shape="linear"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=1,
        )


def build_proposal_figure(
    streams: list[dict],
    proposal_label: str,
    scope_label: str,
    height_scale: float = 1.0,
    compact: bool = False,
    proposal_separators: bool = False,
    half_range_override: float | None = None,
) -> go.Figure:
    panel_titles = [
        SCENARIO_LABELS["trending"],
        SCENARIO_LABELS["positive"],
    ]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.16,
        subplot_titles=panel_titles,
    )

    panel_peak_total = max(
        max_stack_total(streams, "positive"),
        max_stack_total(streams, "trending"),
        1.0,
    )
    half_range = half_range_override if half_range_override is not None else (panel_peak_total / 2.0) * 1.08

    for row_index, scenario in enumerate(["trending", "positive"], start=1):
        scenario_series = []
        for stream in streams:
            values = stream["year_series"][scenario].astype(float).tolist()
            scenario_series.append((stream, values))

        totals = [sum(values[i] for _, values in scenario_series) for i in range(len(YEAR_TICKS))]
        dense_years, smooth_totals = smooth_series(YEAR_TICKS, totals)
        baseline = [-(total / 2.0) for total in smooth_totals]

        for stream_index, (stream, values) in enumerate(scenario_series):
            _, smooth_values = smooth_series(YEAR_TICKS, values)
            lower = list(baseline)
            upper = [baseline[i] + smooth_values[i] for i in range(len(smooth_values))]
            mid = [lower[i] + (smooth_values[i] / 2.0) for i in range(len(smooth_values))]
            baseline = upper
            next_proposal = None
            if proposal_separators and stream_index < len(scenario_series) - 1:
                next_proposal = scenario_series[stream_index + 1][0]["proposal_label"]
            add_stream_band(
                fig=fig,
                row=row_index,
                years=dense_years,
                lower=lower,
                upper=upper,
                mid=mid,
                values=smooth_values,
                stream=stream,
                scenario=scenario,
                showlegend=(row_index == 1),
                separator=proposal_separators and next_proposal is not None and next_proposal != stream["proposal_label"],
            )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=""
            if compact
            else (
                f"{proposal_label} intervention mini streams: {scope_label}"
                f"<br><sup>Centered stream graphs. Canonical stream measures are listed in the legend.</sup>"
            ),
            x=0.5,
            font=dict(size=14 if compact else 22, color="#27496d"),
        ),
        width=FIGURE_WIDTH_PX,
        height=max(220 if compact else 180, int((FIGURE_HEIGHT_BASE_PX + len(streams) * FIGURE_HEIGHT_PER_STREAM_PX) * height_scale)),
        margin=dict(
            l=36 if compact else 44,
            r=18 if compact else 24,
            t=14 if compact else 96,
            b=18 if compact else 84,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Aptos, Arial, sans-serif", size=9 if compact else 11, color="#1f2933"),
        legend=dict(
            orientation="h",
            y=-0.16 if not compact else -0.28,
            x=0.5,
            xanchor="center",
            title_text="",
            font=dict(size=8 if compact else 10, color="#111111"),
        ),
        hovermode="closest",
        showlegend=not compact,
    )

    for annotation in fig.layout.annotations:
        annotation.font.size = 7 if compact else 12

    for row_index in [1, 2]:
        fig.update_xaxes(
            row=row_index,
            col=1,
            range=[YEAR_TICKS[0] - X_RANGE_PAD, YEAR_TICKS[-1] + X_RANGE_PAD],
            tickmode="array",
            tickvals=YEAR_TICKS,
            ticktext=[str(year) for year in YEAR_TICKS],
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            title_text=None if compact else ("Year" if row_index == 2 else None),
        )
        fig.update_yaxes(
            row=row_index,
            col=1,
            title_text="",
            range=[-half_range, half_range],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False,
        )
    return fig


def build_single_scenario_figure(
    streams: list[dict],
    proposal_label: str,
    scope_label: str,
    scenario: str,
    height_scale: float = 1.0,
    compact: bool = False,
    proposal_separators: bool = False,
    half_range_override: float | None = None,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.0,
        subplot_titles=[SCENARIO_LABELS[scenario]],
    )

    panel_peak_total = max(max_stack_total(streams, scenario), 1.0)
    half_range = half_range_override if half_range_override is not None else (panel_peak_total / 2.0) * 1.08

    scenario_series = []
    for stream in streams:
        values = stream["year_series"][scenario].astype(float).tolist()
        scenario_series.append((stream, values))

    totals = [sum(values[i] for _, values in scenario_series) for i in range(len(YEAR_TICKS))]
    dense_years, smooth_totals = smooth_series(YEAR_TICKS, totals)
    baseline = [-(total / 2.0) for total in smooth_totals]

    for stream_index, (stream, values) in enumerate(scenario_series):
        _, smooth_values = smooth_series(YEAR_TICKS, values)
        lower = list(baseline)
        upper = [baseline[i] + smooth_values[i] for i in range(len(smooth_values))]
        mid = [lower[i] + (smooth_values[i] / 2.0) for i in range(len(smooth_values))]
        baseline = upper
        next_proposal = None
        if proposal_separators and stream_index < len(scenario_series) - 1:
            next_proposal = scenario_series[stream_index + 1][0]["proposal_label"]
        add_stream_band(
            fig=fig,
            row=1,
            years=dense_years,
            lower=lower,
            upper=upper,
            mid=mid,
            values=smooth_values,
            stream=stream,
            scenario=scenario,
            showlegend=not compact,
            separator=proposal_separators and next_proposal is not None and next_proposal != stream["proposal_label"],
        )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=""
            if compact
            else (
                f"{proposal_label} intervention mini streams: {scope_label}"
                f"<br><sup>{SCENARIO_LABELS[scenario]}. Canonical stream measures are listed in the legend.</sup>"
            ),
            x=0.5,
            font=dict(size=14 if compact else 22, color="#27496d"),
        ),
        width=FIGURE_WIDTH_PX,
        height=max(140 if compact else 140, int((260 + len(streams) * FIGURE_HEIGHT_PER_STREAM_PX) * height_scale)),
        margin=dict(
            l=36 if compact else 44,
            r=18 if compact else 24,
            t=14 if compact else 72,
            b=18 if compact else 64,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Aptos, Arial, sans-serif", size=9 if compact else 11, color="#1f2933"),
        legend=dict(
            orientation="h",
            y=-0.18 if not compact else -0.34,
            x=0.5,
            xanchor="center",
            title_text="",
            font=dict(size=8 if compact else 10, color="#111111"),
        ),
        hovermode="closest",
        showlegend=not compact,
    )

    for annotation in fig.layout.annotations:
        annotation.font.size = 7 if compact else 12

    fig.update_xaxes(
        row=1,
        col=1,
        range=[YEAR_TICKS[0] - X_RANGE_PAD, YEAR_TICKS[-1] + X_RANGE_PAD],
        tickmode="array",
        tickvals=YEAR_TICKS,
        ticktext=[str(year) for year in YEAR_TICKS],
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        title_text=None if compact else "Year",
    )
    fig.update_yaxes(
        row=1,
        col=1,
        title_text="",
        range=[-half_range, half_range],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        showline=False,
    )
    return fig


def prepare_all_proposal_streams(scope_df: pd.DataFrame) -> list[dict]:
    streams: list[dict] = []
    for proposal_label in PROPOSAL_ORDER:
        proposal_streams = prepare_streams(scope_df, proposal_label)
        for stream in proposal_streams:
            stream_copy = dict(stream)
            stream_copy["legend_label"] = f"{proposal_label} / {stream['intervention']}"
            stream_copy["display_label"] = f"{proposal_label} / {stream['intervention']} ({stream['support']})"
            stream_copy["combined_order"] = (
                proposal_order_key(proposal_label),
                stream["order_key"][0],
                stream["order_key"][1],
            )
            stream_copy["color"] = combined_stream_color(stream_copy)
            streams.append(stream_copy)
    streams.sort(key=lambda item: item["combined_order"])
    return streams


def normalize_all_proposal_streams(streams: list[dict]) -> list[dict]:
    return normalize_all_proposal_streams_with_reference(streams, streams)


def normalize_all_proposal_streams_with_reference(
    streams: list[dict],
    reference_streams: list[dict],
) -> list[dict]:
    proposal_peaks: dict[str, float] = {}
    for proposal_label in PROPOSAL_ORDER:
        proposal_streams = [stream for stream in reference_streams if stream["proposal_label"] == proposal_label]
        if not proposal_streams:
            continue
        peak = max(
            max_stack_total(proposal_streams, "positive"),
            max_stack_total(proposal_streams, "trending"),
            0.0,
        )
        proposal_peaks[proposal_label] = peak if peak > 0 else 1.0

    normalized: list[dict] = []
    for stream in streams:
        scale = proposal_peaks.get(stream["proposal_label"], 1.0)
        copy = dict(stream)
        copy["year_series"] = {
            scenario: stream["year_series"][scenario].astype(float) / scale
            for scenario in SCENARIOS
        }
        copy["canonical_measure"] = f"{stream['canonical_measure']} (normalized)"
        normalized.append(copy)
    return normalized


def compute_global_proposal_peaks_by_site(raw_df: pd.DataFrame) -> dict[str, float]:
    selected_rows = []
    for proposal_label in PROPOSAL_ORDER:
        for (intervention, support), stream_df in raw_df[raw_df["proposal_label"] == proposal_label].groupby(
            ["intervention", "support"],
            sort=False,
        ):
            canonical_measure = canonical_measure_for_stream(
                proposal_label,
                intervention,
                support,
                stream_df["measure"],
            )
            canonical_df = stream_df[stream_df["measure"] == canonical_measure].copy()
            if canonical_df.empty:
                continue
            selected_rows.append(
                canonical_df[
                    ["site", "scenario", "year", "proposal_label", "value"]
                ].copy()
            )

    if not selected_rows:
        return {}

    selected = pd.concat(selected_rows, ignore_index=True)
    totals = (
        selected.groupby(["proposal_label", "site", "scenario", "year"], as_index=False)["value"]
        .sum()
    )
    peaks = totals.groupby("proposal_label")["value"].max().to_dict()
    return {str(key): float(value) for key, value in peaks.items()}


def normalize_all_proposal_streams_with_peak_map(
    streams: list[dict],
    peak_map: dict[str, float],
) -> list[dict]:
    normalized: list[dict] = []
    for stream in streams:
        scale = peak_map.get(stream["proposal_label"], 1.0)
        if scale <= 0:
            scale = 1.0
        copy = dict(stream)
        copy["year_series"] = {
            scenario: stream["year_series"][scenario].astype(float) / scale
            for scenario in SCENARIOS
        }
        copy["canonical_measure"] = f"{stream['canonical_measure']} (normalized)"
        normalized.append(copy)
    return normalized


def build_all_proposals_figure(
    streams: list[dict],
    scope_label: str,
    normalized: bool = False,
    height_scale: float = 1.0,
    compact: bool = False,
    half_range_override: float | None = None,
) -> go.Figure:
    fig = build_proposal_figure(
        streams=streams,
        proposal_label="All Proposals",
        scope_label=scope_label,
        height_scale=height_scale,
        compact=compact,
        proposal_separators=True,
        half_range_override=half_range_override,
    )
    fig.update_layout(
        title=dict(
            text=""
            if compact
            else (
                f"All proposals intervention streams: {scope_label}"
                f"<br><sup>{'Each proposal normalized to its own peak total; ' if normalized else ''}Interventions coloured by proposal; full support richer, partial lighter.</sup>"
            ),
            x=0.5,
            font=dict(size=14 if compact else 22, color="#27496d"),
        )
    )
    return fig


def relative_streams(streams: list[dict]) -> list[dict]:
    return normalize_all_proposal_streams_with_reference(streams, streams)


def paired_half_range(streams: list[dict]) -> float:
    return (max(max_stack_total(streams, "positive"), max_stack_total(streams, "trending"), 1.0) / 2.0) * Y_HEADROOM_FACTOR


def single_half_range(streams: list[dict], scenario: str) -> float:
    return (max(max_stack_total(streams, scenario), 1.0) / 2.0) * Y_HEADROOM_FACTOR


def variant_dir(output_dir: Path, *parts: str) -> Path:
    target_dir = output_dir
    for part in parts:
        if part:
            target_dir = target_dir / part
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def save_png_figure(fig: go.Figure, target_dir: Path, stem: str) -> Path | None:
    png_path = target_dir / f"{stem}.png"
    try:
        fig.write_image(
            str(png_path),
            width=FIGURE_WIDTH_PX,
            height=int(fig.layout.height or 800),
            scale=1,
        )
        return png_path
    except Exception as exc:
        print(f"PNG export skipped for {png_path.name}: {exc}")
        return None


def proposal_order_key(proposal_label: str) -> int:
    try:
        return PROPOSAL_ORDER.index(proposal_label)
    except ValueError:
        return len(PROPOSAL_ORDER)


def generate_outputs(
    raw_df: pd.DataFrame,
    sites: list[str],
    proposals: list[str],
    output_dir: Path,
) -> list[Path]:
    generated: list[Path] = []
    proposal_labels = normalize_proposal_filter(
        sorted(raw_df["proposal_label"].dropna().astype(str).unique().tolist(), key=proposal_order_key),
        proposals,
    )
    global_peak_map = compute_global_proposal_peaks_by_site(raw_df)
    scopes: list[tuple[str, pd.DataFrame]] = [("all-sites-combined", raw_df)]
    scopes.extend([(f"site-{site}", raw_df[raw_df["site"] == site].copy()) for site in sites])

    abs_combined_range = 0.0
    rel_combined_range = 0.0
    abs_per_site_range = 0.0
    rel_per_site_range = 0.0
    abs_single_scenario_site_range = 0.0
    rel_single_scenario_site_range = 0.0
    abs_per_proposal_range = 0.0
    rel_per_proposal_range = 0.0

    prepared_scope_streams: dict[str, dict[str, list[dict]]] = {}
    prepared_scope_relative_global: dict[str, list[dict]] = {}
    prepared_scope_proposal_streams: dict[tuple[str, str], list[dict]] = {}
    prepared_scope_proposal_relative: dict[tuple[str, str], list[dict]] = {}

    for scope_slug, scope_df in scopes:
        all_streams = prepare_all_proposal_streams(scope_df)
        rel_all_streams = relative_streams(all_streams) if all_streams else []
        rel_all_global_streams = normalize_all_proposal_streams_with_peak_map(all_streams, global_peak_map) if all_streams else []
        prepared_scope_streams[scope_slug] = {
            "absolute": all_streams,
            "relative": rel_all_streams,
        }
        prepared_scope_relative_global[scope_slug] = rel_all_global_streams
        if all_streams:
            abs_combined_range = max(abs_combined_range, paired_half_range(all_streams))
            rel_combined_range = max(rel_combined_range, paired_half_range(rel_all_streams))
            if scope_slug != "all-sites-combined":
                abs_per_site_range = max(abs_per_site_range, paired_half_range(all_streams))
                rel_per_site_range = max(rel_per_site_range, paired_half_range(rel_all_global_streams))
                for scenario in SCENARIOS:
                    abs_single_scenario_site_range = max(
                        abs_single_scenario_site_range,
                        single_half_range(all_streams, scenario),
                    )
                    rel_single_scenario_site_range = max(
                        rel_single_scenario_site_range,
                        single_half_range(rel_all_global_streams, scenario),
                    )

        for proposal_label in proposal_labels:
            streams = prepare_streams(scope_df, proposal_label)
            prepared_scope_proposal_streams[(scope_slug, proposal_label)] = streams
            rel_streams = relative_streams(streams) if streams else []
            rel_global_streams = normalize_all_proposal_streams_with_peak_map(streams, global_peak_map) if streams else []
            prepared_scope_proposal_relative[(scope_slug, proposal_label)] = rel_global_streams
            if streams:
                abs_per_proposal_range = max(abs_per_proposal_range, paired_half_range(streams))
                rel_per_proposal_range = max(rel_per_proposal_range, paired_half_range(rel_global_streams))

    abs_combined_range = max(abs_combined_range, 1.0)
    rel_combined_range = max(rel_combined_range, 1.0)
    abs_per_site_range = max(abs_per_site_range, 1.0)
    rel_per_site_range = max(rel_per_site_range, 1.0)
    abs_single_scenario_site_range = max(abs_single_scenario_site_range, 1.0)
    rel_single_scenario_site_range = max(rel_single_scenario_site_range, 1.0)
    abs_per_proposal_range = max(abs_per_proposal_range, 1.0)
    rel_per_proposal_range = max(rel_per_proposal_range, 1.0)

    # Combined across all sites: one graph per mode, both scenarios together, extra tall.
    combined_slug = "all-sites-combined"
    combined_mode_streams = {
        "absolute": prepared_scope_streams[combined_slug]["absolute"],
        "relative": prepared_scope_relative_global[combined_slug],
    }
    for mode, mode_streams in combined_mode_streams.items():
        if not mode_streams:
            continue
        combined_fig = build_all_proposals_figure(
            mode_streams,
            combined_slug,
            normalized=(mode == "relative"),
            height_scale=4.5,
            compact=False,
            half_range_override=abs_combined_range if mode == "absolute" else rel_combined_range,
        )
        target_dir = variant_dir(output_dir, mode, "combined", "combined")
        png_path = save_png_figure(combined_fig, target_dir, combined_slug)
        if png_path is not None:
            generated.append(png_path)
            print(f"Saved PNG:  {png_path}")

    # Per-site: both scenarios together plus one graph per scenario.
    for scope_slug, _scope_df in scopes[1:]:
        absolute_all_streams = prepared_scope_streams[scope_slug]["absolute"]
        relative_all_streams = prepared_scope_relative_global[scope_slug]

        for mode, mode_streams, paired_range, single_range in [
            ("absolute", absolute_all_streams, abs_per_site_range, abs_single_scenario_site_range),
            ("relative", relative_all_streams, rel_per_site_range, rel_single_scenario_site_range),
        ]:
            if not mode_streams:
                continue

            for height_name, height_scale, compact in [
                ("full-height", 1.0, False),
                ("half-height", 0.5, True),
            ]:
                paired_fig = build_all_proposals_figure(
                    mode_streams,
                    scope_slug,
                    normalized=(mode == "relative"),
                    height_scale=height_scale,
                    compact=compact,
                    half_range_override=paired_range,
                )
                target_dir = variant_dir(output_dir, mode, "per-site", height_name)
                paired_path = save_png_figure(paired_fig, target_dir, scope_slug)
                if paired_path is not None:
                    generated.append(paired_path)
                    print(f"Saved PNG:  {paired_path}")

                for scenario in SCENARIOS:
                    single_fig = build_single_scenario_figure(
                        mode_streams,
                        "All Proposals",
                        f"{scope_slug}-just_{scenario}",
                        scenario=scenario,
                        height_scale=height_scale,
                        compact=compact,
                        proposal_separators=True,
                        half_range_override=single_range,
                    )
                    stem = f"{scope_slug}-just_{scenario}"
                    single_path = save_png_figure(single_fig, target_dir, stem)
                    if single_path is not None:
                        generated.append(single_path)
                        print(f"Saved PNG:  {single_path}")

    # Per-proposal: full-height only, both scenarios together, shared y-axis across proposals.
    for proposal_label in proposal_labels:
        for scope_slug, _scope_df in scopes:
            absolute_streams = prepared_scope_proposal_streams[(scope_slug, proposal_label)]
            relative_streams_for_scope = prepared_scope_proposal_relative[(scope_slug, proposal_label)]
            for mode, mode_streams, shared_range in [
                ("absolute", absolute_streams, abs_per_proposal_range),
                ("relative", relative_streams_for_scope, rel_per_proposal_range),
            ]:
                if not mode_streams:
                    continue
                fig = build_proposal_figure(
                    mode_streams,
                    proposal_label,
                    scope_slug,
                    height_scale=1.0,
                    compact=False,
                    half_range_override=shared_range,
                )
                target_dir = variant_dir(output_dir, mode, "per-proposal", "full-height")
                stem = f"{slugify(proposal_label)}_{scope_slug}"
                png_path = save_png_figure(fig, target_dir, stem)
                if png_path is not None:
                    generated.append(png_path)
                    print(f"Saved PNG:  {png_path}")

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate proposal mini stream graphs from raw intervention statistics."
    )
    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Comma-separated site list. Default: all sites in _statistics-refactored/raw.",
    )
    parser.add_argument(
        "--proposals",
        type=str,
        default=None,
        help="Comma-separated proposal list for a smaller export sample. Default: all proposals.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    available_sites = discover_sites()
    sites = normalize_site_filter(available_sites, args.sites)
    if not sites:
        raise ValueError("No sites available after applying the requested site filter.")

    raw_df = load_raw_interventions(sites)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"Writing outputs to: {output_dir}")
    print(f"Sites: {', '.join(sites)}")
    generated = generate_outputs(
        raw_df=raw_df,
        sites=sites,
        proposals=args.proposals,
        output_dir=output_dir,
    )

    print("Completed graph generation.")
    print("Output files:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
