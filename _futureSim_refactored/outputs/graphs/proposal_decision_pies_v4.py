"""
V4 proposal decision pie charts.

Reads `{root}/output/stats/proposal-decisions/proposal_decisions.csv` (produced
by `proposal_decisions_metrics_v4.py`) and emits pie-chart grids showing the
per-state breakdown of proposal decisions into rejected / accepted-full /
accepted-partial.

Grid layout (default):

                       yr 1                       yr 30
                positive   trending    |   positive   trending
    decay          pie        pie      |     pie        pie
    release-ctrl   pie        pie      |     pie        pie
    recruit        pie        pie      |     pie        pie

Two modes:

- absolute: each pie's radius scales with its total count (per-row normalised
  so the biggest total in a row fills the slot). Lets you read "this state has
  many more decisions than that state".
- relative: all pies the same size, just showing percentage breakdown.

USAGE:
    uv run python _futureSim_refactored/outputs/graphs/proposal_decision_pies_v4.py \
        --root _data-refactored/model-outputs/generated-states/4.9 \
        --site trimmed-parade
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

PROPOSALS = ["decay", "release-control", "recruit", "colonise", "deploy-structure"]
SCENARIOS = ["positive", "trending"]
YEARS = [1, 10, 30, 60, 90, 120, 150, 180]
SITES = ["trimmed-parade", "city", "uni"]

SCENARIO_LABELS = {
    "positive": "nonhuman-led",
    "trending": "human-led",
}
PROPOSAL_LABELS = {
    "decay": "Decay",
    "release-control": "Release control",
    "recruit": "Recruit",
    "colonise": "Colonise",
    "deploy-structure": "Deploy structure",
}

# Minimum visible pie size for zero-count states, as a fraction of the
# row-max area. 0.01 = 1% area, which is ~10% of the full radius.
# This makes "nothing happened here" obvious at a glance without hiding the
# slot entirely.
ZERO_AREA_FRACTION = 0.01

# V4 compositor family colours (from
# _futureSim_refactored/blender/compositor/scripts/_set_proposal_colors.py)
PROPOSAL_FAMILY_COLORS = {
    "colonise": "#FF8C00",
    "decay": "#E62626",
    "deploy-structure": "#2659FF",
    "recruit": "#26BF26",
    "release-control": "#8C1AD9",
}

# Lightness-by-support: accepted-full is the family hue, accepted-partial is a
# lighter version, rejected is a dark neutral grey.
SUPPORT_LIGHTNESS = {
    "accepted_full": 0.15,     # mildly lightened
    "accepted_partial": 0.55,  # substantially lightened
}
REJECTED_COLOR = "#3a3a3a"

CATEGORY_ORDER = ["rejected", "accepted_full", "accepted_partial"]
CATEGORY_LABELS = {
    "rejected": "rejected",
    "accepted_full": "accepted (full)",
    "accepted_partial": "accepted (partial)",
}

GRID_WIDTH = 1900
GRID_HEIGHT = 2400

SEPARATOR_LINE = dict(color="#000000", width=1.2)
EMPTY_PIE_COLOR = "#d8d8d8"


def lighten_hex(hex_color: str, amount: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(round(r + (255 - r) * amount))
    g = int(round(g + (255 - g) * amount))
    b = int(round(b + (255 - b) * amount))
    return f"#{r:02x}{g:02x}{b:02x}"


def slice_colors_for(proposal: str) -> list[str]:
    base = PROPOSAL_FAMILY_COLORS[proposal]
    return [
        REJECTED_COLOR,
        lighten_hex(base, SUPPORT_LIGHTNESS["accepted_full"]),
        lighten_hex(base, SUPPORT_LIGHTNESS["accepted_partial"]),
    ]


# =============================================================================
# DATA
# =============================================================================


def load_decisions(root: Path) -> pd.DataFrame:
    """Load decision counts from per-state v4_decisions CSVs.

    Falls back to the legacy aggregated proposal_decisions.csv if per-state
    CSVs are not found.
    """
    per_state_dir = root / "output" / "stats" / "per-state"
    frames = []
    for csv_path in sorted(per_state_dir.rglob("*_v4_decisions.csv")):
        frames.append(pd.read_csv(csv_path))
    if frames:
        return pd.concat(frames, ignore_index=True)

    # Legacy fallback
    legacy = root / "output" / "stats" / "proposal-decisions" / "proposal_decisions.csv"
    if legacy.exists():
        return pd.read_csv(legacy)
    raise FileNotFoundError(
        f"No v4_decisions CSVs under {per_state_dir} and no legacy {legacy}.\n"
        "Run: uv run python -m _futureSim_refactored.outputs.stats.vtk_to_stat_counts --root <root>"
    )


def load_subgroups(root: Path) -> pd.DataFrame | None:
    """Load decision subgroup counts from per-state v4_decision_subgroups CSVs.

    Falls back to the legacy aggregated proposal_decision_subgroups.csv.
    """
    per_state_dir = root / "output" / "stats" / "per-state"
    frames = []
    for csv_path in sorted(per_state_dir.rglob("*_v4_decision_subgroups.csv")):
        frames.append(pd.read_csv(csv_path))
    if frames:
        return pd.concat(frames, ignore_index=True)

    # Legacy fallback
    legacy = root / "output" / "stats" / "proposal-decisions" / "proposal_decision_subgroups.csv"
    if legacy.exists():
        return pd.read_csv(legacy)
    return None


# Families that use the subgroup breakdown instead of the flat full/partial slices.
# Each family defines the subgroup display order (left-to-right around the pie
# within each support bucket).
SUBGROUP_FAMILIES = {
    "colonise": ["ground", "green roof", "brown roof", "biofacade"],
    "recruit": ["open", "roads", "street potential", "other"],
    "deploy-structure": ["pole", "log", "upgrade"],
}


def get_subgroup_counts(
    df_sub: pd.DataFrame,
    sites: str | list[str],
    proposal: str,
    year: int,
    scenario: str,
) -> list[tuple[str, str, int]]:
    """Return [(support_bucket, subgroup, count), ...] in display order.

    Order: full/partial x family's subgroup order.
    """
    if isinstance(sites, str):
        sites = [sites]
    sub = df_sub[
        (df_sub["site"].isin(sites))
        & (df_sub["proposal"] == proposal)
        & (df_sub["year"] == year)
        & (df_sub["scenario"] == scenario)
    ]
    out: list[tuple[str, str, int]] = []
    order = SUBGROUP_FAMILIES.get(proposal, [])
    for bucket in ("full", "partial"):
        for sg in order:
            match = sub[(sub["support_bucket"] == bucket) & (sub["subgroup"] == sg)]
            cnt = int(match["count"].sum()) if not match.empty else 0
            out.append((bucket, sg, cnt))
    return out


def get_slice_counts(
    df: pd.DataFrame,
    sites: str | list[str],
    proposal: str,
    year: int,
    scenario: str,
) -> dict[str, int]:
    if isinstance(sites, str):
        sites = [sites]
    sub = df[
        (df["site"].isin(sites))
        & (df["proposal"] == proposal)
        & (df["year"] == year)
        & (df["scenario"] == scenario)
    ]
    if sub.empty:
        return {c: 0 for c in CATEGORY_ORDER}
    return {
        "rejected": int(sub["rejected"].sum()),
        "accepted_full": int(sub["accepted_full"].sum()),
        "accepted_partial": int(sub["accepted_partial"].sum()),
    }


def get_unit(df: pd.DataFrame, proposal: str) -> str:
    sub = df[df["proposal"] == proposal]
    if sub.empty:
        return ""
    return str(sub.iloc[0]["unit"])


# =============================================================================
# FIGURE
# =============================================================================


def column_titles() -> list[str]:
    return [f"year {y}" for y in YEARS]


def row_label(proposal: str, scenario: str, unit: str) -> str:
    return (
        f"<b>{PROPOSAL_LABELS[proposal]}</b><br>"
        f"<span style='font-size:12px;color:#444'>{SCENARIO_LABELS[scenario]}</span><br>"
        f"<span style='font-size:10px;color:#888'>({unit})</span>"
    )


def data_rows() -> list[tuple[str, str]]:
    """Return (proposal, scenario) pairs in display order."""
    rows: list[tuple[str, str]] = []
    for proposal in PROPOSALS:
        for scenario in SCENARIOS:
            rows.append((proposal, scenario))
    return rows


def build_pie_trace(
    counts: dict[str, int],
    proposal: str,
    label: str,
    hole: float = 0.35,
) -> go.Pie:
    """Pie trace for a state with at least one decision."""
    total = sum(counts.values())
    colors = slice_colors_for(proposal)

    values = [counts[c] for c in CATEGORY_ORDER]
    labels = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER]
    hover = [
        f"<b>{label}</b><br>{CATEGORY_LABELS[c]}: {counts[c]:,}<br>"
        f"{(counts[c] / total * 100):.1f}% of total"
        for c in CATEGORY_ORDER
    ]

    return go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=SEPARATOR_LINE),
        textinfo="percent",
        textfont=dict(size=10, color="white"),
        insidetextorientation="radial",
        hoverinfo="text",
        hovertext=hover,
        sort=False,
        direction="clockwise",
        showlegend=False,
        hole=hole,
    )


def build_subgroup_pie_trace(
    rejected: int,
    subgroups: list[tuple[str, str, int]],
    proposal: str,
    label: str,
    hole: float = 0.35,
) -> go.Pie:
    """Pie trace that expands accepted_full/partial into per-subgroup wedges.

    Colouring: each subgroup within a support bucket keeps the family's
    full/partial colour as-is, so the pie still reads as full-vs-partial at a
    glance. Wedge boundaries (black lines) visually separate neighbouring
    subgroups, and the subgroup label is shown inside each slice.
    """
    base = PROPOSAL_FAMILY_COLORS[proposal]
    full_color = lighten_hex(base, SUPPORT_LIGHTNESS["accepted_full"])
    partial_color = lighten_hex(base, SUPPORT_LIGHTNESS["accepted_partial"])

    labels: list[str] = []
    values: list[int] = []
    colors: list[str] = []
    hover: list[str] = []
    text_inside: list[str] = []

    total = rejected + sum(c for _, _, c in subgroups)

    if rejected > 0:
        labels.append("rejected")
        values.append(rejected)
        colors.append(REJECTED_COLOR)
        hover.append(
            f"<b>{label}</b><br>rejected: {rejected:,}<br>"
            f"{(rejected / total * 100):.1f}% of total"
        )
        text_inside.append("")

    for bucket, sg, count in subgroups:
        if count <= 0:
            continue
        labels.append(f"{bucket} · {sg}")
        values.append(count)
        colors.append(full_color if bucket == "full" else partial_color)
        pct = (count / total * 100) if total > 0 else 0
        hover.append(
            f"<b>{label}</b><br>{bucket} — {sg}: {count:,}<br>"
            f"{pct:.1f}% of total"
        )
        text_inside.append(sg)

    return go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=SEPARATOR_LINE),
        text=text_inside,
        textinfo="text",
        textfont=dict(size=9, color="white"),
        insidetextorientation="radial",
        hoverinfo="text",
        hovertext=hover,
        sort=False,
        direction="clockwise",
        showlegend=False,
        hole=hole,
    )


def build_empty_pie_trace(
    counts: dict[str, int],
    label: str,
    hole: float = 0.35,
) -> go.Pie:
    """Placeholder pie for a state with zero decisions (grey, no slices)."""
    return go.Pie(
        labels=["no decisions"],
        values=[1],
        marker=dict(colors=[EMPTY_PIE_COLOR], line=SEPARATOR_LINE),
        textinfo="none",
        hoverinfo="text",
        hovertext=[f"<b>{label}</b><br>no decisions recorded"],
        sort=False,
        showlegend=False,
        hole=hole,
    )


def shrink_domain(trace, fraction: float) -> None:
    """Shrink a pie trace's domain symmetrically around its centre."""
    cur = trace.domain
    x0, x1 = cur.x
    y0, y1 = cur.y
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    half_w = (x1 - x0) / 2 * fraction
    half_h = (y1 - y0) / 2 * fraction
    trace.domain = dict(
        x=[cx - half_w, cx + half_w],
        y=[cy - half_h, cy + half_h],
    )


def build_grid_figure(
    df: pd.DataFrame,
    site: str,
    mode: str,
    df_sub: pd.DataFrame | None = None,
) -> go.Figure:
    rows = data_rows()
    n_rows = len(rows)
    n_cols = len(YEARS)

    specs = [[{"type": "domain"}] * n_cols for _ in range(n_rows)]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        column_titles=column_titles(),
        horizontal_spacing=0.04,
        vertical_spacing=0.05,
    )

    # Compute per-proposal max totals across BOTH scenarios (so cross-scenario
    # contrast is visible — e.g. trending-decay pies look small next to
    # positive-decay pies).
    proposal_max: dict[str, int] = {}
    for proposal in PROPOSALS:
        totals = []
        for scenario in SCENARIOS:
            for year in YEARS:
                c = get_slice_counts(df, site, proposal, year, scenario)
                totals.append(sum(c.values()))
        proposal_max[proposal] = max(totals) if totals else 0

    for r, (proposal, scenario) in enumerate(rows, start=1):
        row_max = proposal_max[proposal]
        use_subgroups = df_sub is not None and proposal in SUBGROUP_FAMILIES
        for c, year in enumerate(YEARS, start=1):
            counts = get_slice_counts(df, site, proposal, year, scenario)
            total = sum(counts.values())
            label = (
                f"{PROPOSAL_LABELS[proposal]} · yr{year} · "
                f"{SCENARIO_LABELS[scenario]}"
            )

            if total == 0:
                trace = build_empty_pie_trace(counts, label)
            elif use_subgroups:
                subgroups = get_subgroup_counts(df_sub, site, proposal, year, scenario)
                trace = build_subgroup_pie_trace(counts["rejected"], subgroups, proposal, label)
            else:
                trace = build_pie_trace(counts, proposal, label)
            fig.add_trace(trace, row=r, col=c)

            # Absolute mode: shrink domain so pie AREA ∝ total count
            # (ggplot2 scale_size_area() convention — Stevens' Power Law).
            # Zero totals clamp to ZERO_AREA_FRACTION so the slot still shows
            # a tiny grey marker meaning "nothing happened here".
            if mode == "absolute" and row_max > 0:
                area_frac = max(total / row_max, ZERO_AREA_FRACTION)
                radius_frac = area_frac ** 0.5
                shrink_domain(fig.data[-1], radius_frac)

    # Row-label annotations on the left side
    for r, (proposal, scenario) in enumerate(rows):
        unit = get_unit(df, proposal)
        y_top = 1.0 - (r / n_rows)
        y_mid = y_top - (1.0 / n_rows) / 2
        fig.add_annotation(
            text=row_label(proposal, scenario, unit),
            x=-0.015,
            y=y_mid,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=13, color="#222"),
            xanchor="right",
            align="right",
        )

    # Thicker horizontal separator shapes between proposal groups
    # (after every 2 rows since each proposal has 2 scenarios).
    for r in range(1, len(PROPOSALS)):
        y_line = 1.0 - (2 * r) / n_rows
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=-0.18,
            x1=1.0,
            y0=y_line,
            y1=y_line,
            line=dict(color="#000000", width=1.6),
        )

    mode_label = (
        "absolute — pie area ∝ total count (per-proposal scale, across both scenarios)"
        if mode == "absolute"
        else "relative — equal-size pies, showing % breakdown only"
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Proposal decisions @ {site}</b><br>"
                f"<span style='font-size:12px;color:#666'>{mode_label}</span>"
            ),
            x=0.5,
            xanchor="center",
        ),
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        margin=dict(l=200, r=40, t=120, b=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # Build a simple legend with one dummy scatter per category+proposal combo
    # Too noisy — instead emit a single shared legend with 3 entries using
    # neutral swatches (exact colour varies by family).
    legend_entries = [
        ("rejected", REJECTED_COLOR),
        ("accepted (full)", "#888888"),
        ("accepted (partial)", "#cccccc"),
    ]
    for name, color in legend_entries:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=14, color=color, line=dict(color="black", width=1)),
                name=name,
                showlegend=True,
            )
        )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# =============================================================================
# MAIN
# =============================================================================


def generate(root: Path, sites: list[str]) -> None:
    df = load_decisions(root)
    df_sub = load_subgroups(root)
    if df_sub is None:
        print(
            "  (no proposal_decision_subgroups.csv found — colonise/recruit "
            "will render without subgroup breakdown)"
        )
    out_dir = root / "output" / "graphs" / "proposal-decision-pies"
    out_dir.mkdir(parents=True, exist_ok=True)

    for site in sites:
        for mode in ["absolute", "relative"]:
            fig = build_grid_figure(df, site, mode, df_sub=df_sub)
            html_path = out_dir / f"{site}_proposal_decision_pies_{mode}.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            print(f"  wrote {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render V4 proposal decision pie charts.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--sites", default=",".join(SITES), help="Comma-separated site list")
    args = parser.parse_args()

    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    generate(Path(args.root), sites)


if __name__ == "__main__":
    main()
