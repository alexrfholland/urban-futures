"""
V4 stream graph generator.

Reads per-state V4 indicator CSVs, computes pct_of_baseline per indicator,
builds composites as mean of sub-indicator percentages, and produces stacked
stream graphs (positive up, trending down).

USAGE:
    # From repo root:
    uv run python _futureSim_refactored/outputs/graphs/stream_graph_v4.py \
        --root _data-refactored/model-outputs/generated-states/v4.6

    # Subset of sites:
    uv run python ... --sites trimmed-parade,city

    # Color by persona instead of capability:
    uv run python ... --color-by persona
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.interpolate import PchipInterpolator


# =============================================================================
# CONFIGURATION
# =============================================================================

GRAPH_WIDTH = 800
GRAPH_HEIGHT_PER_SITE = 640

BASELINE_YEAR = -50
BASELINE_PCT = 100

FILL_ALPHA = 0.4

SEP_LINE = dict(width=0.5, color='rgba(255,255,255,0.8)')
PERSONA_SEP_LINE = dict(width=1.5, color='rgba(0,0,0,0.8)')

CAPABILITY_COLORS = {
    'acquire': '#1B9E77',        # teal  (was 'self')
    'communicate': '#D95F02',    # orange (was 'others')
    'reproduce': '#7570B3',      # purple (was 'generations')
}

# Legacy mapping for reference:
# self -> acquire, others -> communicate, generations -> reproduce

PERSONA_COLORS = {
    'Bird': '#E41A1C',
    'Lizard': '#4DAF4A',
    'Tree': '#377EB8',
}

# Indicators excluded from stream graphs.
# Composites (two-part names like Lizard.acquire) are always excluded —
# the graph builds its own composites as mean-of-sub-indicator-percentages.
# Additional exclusions go here (baseline=0 indicators that produce INF%).
EXCLUDE_INDICATORS = {
    "Tree.acquire.moderated",
    "Tree.reproduce.smaller-patches-rewild",
}

# Human-readable descriptions for hover tooltips.
INDICATOR_DESCRIPTIONS = {
    "Bird.acquire.peeling-bark": "Peeling bark volume (foraging habitat)",
    "Bird.communicate.perch-branch": "Perch branches on senescing/snag/artificial trees",
    "Bird.reproduce.hollow": "Tree hollows (nesting sites)",
    "Lizard.acquire.grass": "Low vegetation ground cover",
    "Lizard.acquire.dead-branch": "Dead branch volume",
    "Lizard.acquire.epiphyte": "Epiphyte volume",
    "Lizard.communicate.not-paved": "Accessible ground (not paved/roadway)",
    "Lizard.reproduce.nurse-log": "Fallen log volume (nurse logs)",
    "Lizard.reproduce.fallen-tree": "Fallen/decayed tree volume",
    "Tree.acquire.autonomous": "Canopy pruning eliminated (autonomous growth)",
    "Tree.communicate.snag": "Standing dead trees (snags)",
    "Tree.communicate.fallen": "Fallen trees",
    "Tree.communicate.decayed": "Decayed trees",
    "Tree.reproduce.larger-patches-rewild": "Ground recruitable for rewilding (>1.5m from trees)",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_v4_indicator_csvs(root: str | Path, sites: list[str]) -> pd.DataFrame:
    """Load all per-state V4 indicator CSVs under a run root.

    Expected layout:
        {root}/output/stats/per-state/{site}/{site}_{scenario}_1_yr{year}_v4_indicators.csv
        {root}/output/stats/per-state/{site}/{site}_baseline_1_v4_indicators.csv
    """
    root = Path(root)
    stats_dir = root / "output" / "stats" / "per-state"
    frames = []

    for site in sites:
        site_dir = stats_dir / site
        if not site_dir.exists():
            print(f"Warning: missing stats dir {site_dir}")
            continue
        for csv_path in sorted(site_dir.glob(f"{site}_*_v4_indicators.csv")):
            df = pd.read_csv(csv_path)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No V4 indicator CSVs found under {stats_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = pd.to_numeric(combined["year"])
    return combined


def is_composite(indicator: str) -> bool:
    """Composite indicators have exactly two dot-separated parts (Persona.capability)."""
    return indicator.count(".") == 1


def parse_indicator(indicator: str) -> tuple[str, str, str]:
    """Return (persona, capability, sub) from an indicator name.

    Composites like 'Lizard.acquire' return ('Lizard', 'acquire', '').
    """
    parts = indicator.split(".")
    persona = parts[0] if len(parts) >= 1 else ""
    capability = parts[1] if len(parts) >= 2 else ""
    sub = parts[2] if len(parts) >= 3 else ""
    return persona, capability, sub


def prepare_stream_data(
    raw: pd.DataFrame,
    sites: list[str],
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    """Prepare indicator data for stream plotting.

    1. Drop composite rows and excluded indicators.
    2. Compute pct_of_baseline per indicator per site.
    3. Build composite capability streams as mean of sub-indicator percentages.
    4. Add baseline anchor points.
    """
    if exclude is None:
        exclude = EXCLUDE_INDICATORS

    # Drop composites and exclusions
    df = raw[~raw["indicator"].apply(is_composite)].copy()
    df = df[~df["indicator"].isin(exclude)].copy()

    # Split into baseline and scenarios
    baseline = df[df["scenario"] == "baseline"].copy()
    scenarios = df[df["scenario"].isin(["positive", "trending"])].copy()

    # Build baseline lookup: (site, indicator) -> count
    baseline_lookup = baseline.groupby(["site", "indicator"])["count"].first().to_dict()

    # Compute pct_of_baseline
    scenarios["baseline_count"] = scenarios.apply(
        lambda r: baseline_lookup.get((r["site"], r["indicator"]), 0), axis=1
    )
    scenarios["pct_of_baseline"] = np.where(
        scenarios["baseline_count"] > 0,
        (scenarios["count"] / scenarios["baseline_count"] * 100).round(2),
        np.nan,
    )

    # Drop indicators with no valid baseline
    scenarios = scenarios.dropna(subset=["pct_of_baseline"])

    # Parse persona/capability
    scenarios[["persona", "capability", "sub"]] = pd.DataFrame(
        scenarios["indicator"].apply(parse_indicator).tolist(),
        index=scenarios.index,
    )
    scenarios["stream_id"] = scenarios["indicator"]

    # ----- Normalise: divide each sub-indicator by sub-count so every -----
    # capability group stacks to the same height (its mean percentage).
    # This means Bird.acquire (1 sub) and Lizard.acquire (3 subs) occupy
    # equal vertical space at the same composite percentage.
    sub_counts = (
        scenarios[["site", "persona", "capability", "indicator"]]
        .drop_duplicates()
        .groupby(["site", "persona", "capability"])
        .size()
        .reset_index(name="n_subs")
    )
    scenarios = scenarios.merge(sub_counts, on=["site", "persona", "capability"], how="left")
    scenarios["pct_of_baseline"] = scenarios["pct_of_baseline"] / scenarios["n_subs"]
    scenarios.drop(columns=["n_subs"], inplace=True)

    # No separate composite band — the normalised subs already sum to
    # the composite (mean) percentage per capability group.
    combined = scenarios.copy()

    # ----- Baseline anchor points at x = BASELINE_YEAR -----
    keys = ["site", "scenario", "persona", "capability", "stream_id", "indicator", "sub"]
    unique_combos = combined[keys].drop_duplicates()

    # Build sub-count lookup for normalising baseline anchors
    anchor_sub_counts = (
        combined[["site", "persona", "capability", "indicator"]]
        .drop_duplicates()
        .groupby(["site", "persona", "capability"])
        .size()
        .reset_index(name="n_subs")
    )

    anchor_rows = []
    for yr in range(BASELINE_YEAR, BASELINE_YEAR + 6):
        rows = unique_combos.copy()
        rows["year"] = yr
        rows = rows.merge(anchor_sub_counts, on=["site", "persona", "capability"], how="left")
        rows["pct_of_baseline"] = BASELINE_PCT / rows["n_subs"]
        rows.drop(columns=["n_subs"], inplace=True)
        anchor_rows.append(rows)

    combined = pd.concat([combined] + anchor_rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=keys + ["year"], keep="first")

    return combined


# =============================================================================
# SMOOTHING
# =============================================================================

def smooth_series(years, values):
    """PCHIP-smooth a single series. Returns (x_smooth, y_smooth)."""
    df = pd.DataFrame({"year": years, "val": values}).dropna().sort_values("year")
    df = df.groupby("year")["val"].mean().reset_index()
    x, y = df["year"].values.astype(float), df["val"].values.astype(float)

    if len(x) < 3:
        return x, y

    x_smooth = np.arange(int(x.min()), int(x.max()) + 1, 1).astype(float)
    try:
        y_smooth = PchipInterpolator(x, y)(x_smooth)
    except Exception:
        y_smooth = np.interp(x_smooth, x, y)
    return x_smooth, np.maximum(y_smooth, 0)


def apply_smoothing(data: pd.DataFrame) -> pd.DataFrame:
    """Smooth all streams and compute y_plot (trending mirrored below zero)."""
    group_cols = ["site", "scenario", "stream_id"]
    results = []

    for name, group in data.groupby(group_cols):
        x_s, y_s = smooth_series(group["year"].values, group["pct_of_baseline"].values)
        out = pd.DataFrame({"year": x_s, "pct_smooth": y_s})
        for i, col in enumerate(group_cols):
            out[col] = name[i]

        # Carry over persona/capability from group
        out["persona"] = group["persona"].iloc[0]
        out["capability"] = group["capability"].iloc[0]
        out["indicator"] = group["indicator"].iloc[0]
        results.append(out)

    smoothed = pd.concat(results, ignore_index=True)
    smoothed["y_plot"] = np.where(
        smoothed["scenario"] == "trending",
        -smoothed["pct_smooth"],
        smoothed["pct_smooth"],
    )
    smoothed["description"] = smoothed["indicator"].map(INDICATOR_DESCRIPTIONS).fillna("")
    smoothed["hover"] = (
        "<b>" + smoothed["site"] + "</b>"
        + "<br><b>" + smoothed["persona"] + "." + smoothed["capability"] + "</b>"
        + "<br>indicator: " + smoothed["indicator"]
        + "<br>measures: " + smoothed["description"]
        + "<br>scenario: " + smoothed["scenario"]
        + "<br>year: " + smoothed["year"].round(0).astype(int).astype(str)
        + "<br>% of baseline: " + smoothed["pct_smooth"].round(1).astype(str) + "%"
    )
    return smoothed


# =============================================================================
# STREAM GRAPH (combined single panel, all sites stacked)
# =============================================================================

SITE_SEP_LINE = dict(width=2, color="rgba(0,0,0,0.9)")


def create_combined_stream_graph(smoothed: pd.DataFrame, sites: list[str], color_by: str = "capability"):
    """Single combined stream graph with all sites stacked together.

    Stream order: site > persona > indicator.
    Site boundaries get thick separator lines, persona boundaries get thinner ones.
    """
    df = smoothed.copy()

    if color_by == "capability":
        color_map, color_col = CAPABILITY_COLORS, "capability"
    else:
        color_map, color_col = PERSONA_COLORS, "persona"

    PERSONA_ORDER = {"Bird": 0, "Lizard": 1, "Tree": 2}

    # Build indicator_site key for ordering: "{stream_id}-{site}"
    df["indicator_site"] = df["stream_id"] + "-" + df["site"]

    def parse_indicator_site(indicator_site):
        for site in sites:
            suffix = f"-{site}"
            if indicator_site.endswith(suffix):
                return indicator_site[: -len(suffix)], site
        return indicator_site, ""

    def sort_key(indicator_site):
        indicator, site = parse_indicator_site(indicator_site)
        parts = indicator.split(".")
        persona = parts[0]
        capability = parts[1] if len(parts) >= 2 else ""
        sub = parts[2] if len(parts) >= 3 else ""
        is_comp = 1 if sub == "" else 0
        site_order = sites.index(site) if site in sites else 999
        return (site_order, PERSONA_ORDER.get(persona, 99), capability, is_comp, sub)

    stream_ids = sorted(df["indicator_site"].unique(), key=sort_key)

    # Find site and persona boundary indices
    def get_site(sid):
        _, site = parse_indicator_site(sid)
        return site

    def get_persona(sid):
        indicator, _ = parse_indicator_site(sid)
        return indicator.split(".")[0]

    site_end_indices = []
    persona_end_indices = []
    for i in range(len(stream_ids) - 1):
        if get_site(stream_ids[i]) != get_site(stream_ids[i + 1]):
            site_end_indices.append(i)
        elif get_persona(stream_ids[i]) != get_persona(stream_ids[i + 1]):
            persona_end_indices.append(i)

    fig = go.Figure()

    for scenario_label, stack_group in [("positive", "pos"), ("trending", "neg")]:
        df_sc = df[df["scenario"] == scenario_label]
        pivot = df_sc.pivot_table(index="year", columns="indicator_site", values="y_plot", aggfunc="first")
        pivot = pivot.reindex(columns=stream_ids).fillna(0)

        for sid in stream_ids:
            dd = df_sc[df_sc["indicator_site"] == sid].sort_values("year")
            if len(dd) == 0:
                continue
            key = dd[color_col].iloc[0]
            base = color_map.get(key, "#999999")
            r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
            fill = f"rgba({r},{g},{b},{FILL_ALPHA})"

            fig.add_trace(go.Scatter(
                x=dd["year"], y=dd["y_plot"],
                mode="lines", stackgroup=stack_group, fill="tonexty",
                line=SEP_LINE, fillcolor=fill,
                text=dd["hover"], hoverinfo="text",
                name=sid, showlegend=False,
            ))

        # Site boundary lines (thick)
        for end_idx in site_end_indices:
            cols = stream_ids[: end_idx + 1]
            if len(pivot) > 0:
                cumsum = pivot[cols].sum(axis=1)
                fig.add_trace(go.Scatter(
                    x=cumsum.index, y=cumsum.values,
                    mode="lines", line=SITE_SEP_LINE,
                    hoverinfo="skip", showlegend=False,
                ))

        # Persona boundary lines (thin)
        for end_idx in persona_end_indices:
            cols = stream_ids[: end_idx + 1]
            if len(pivot) > 0:
                cumsum = pivot[cols].sum(axis=1)
                fig.add_trace(go.Scatter(
                    x=cumsum.index, y=cumsum.values,
                    mode="lines", line=PERSONA_SEP_LINE,
                    hoverinfo="skip", showlegend=False,
                ))

    # Layout
    graph_height = GRAPH_HEIGHT_PER_SITE * 2
    tickvals = [BASELINE_YEAR, 0, 10, 30, 60, 180]

    fig.update_layout(
        title=dict(text="V4 Combined Stream Graph (all sites)", x=0.5),
        height=graph_height,
        width=GRAPH_WIDTH,
        autosize=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        showlegend=False,
        margin=dict(l=60, r=20, t=80, b=40),
    )

    fig.update_xaxes(showticklabels=False, showline=False, zeroline=False, showgrid=False)
    fig.update_yaxes(zeroline=False, showline=False, showticklabels=False, showgrid=False)
    fig.add_hline(y=0, line_width=2, line_color="black", layer="above")
    fig.add_vline(x=0, line_width=2, line_color="black")

    y_max = df["y_plot"].abs().max()
    tick_height = y_max * 0.03
    for xval in tickvals:
        fig.add_shape(
            type="line", x0=xval, x1=xval, y0=-tick_height, y1=tick_height,
            line=dict(color="black", width=1.5), layer="above",
        )

    return fig, graph_height


# =============================================================================
# MAIN
# =============================================================================

def generate_stream_graph(
    root: str | Path,
    sites: list[str] | None = None,
    color_by: str = "capability",
    save: bool = True,
    show: bool = True,
):
    if sites is None:
        sites = ["trimmed-parade", "city", "uni"]

    print(f"Loading V4 indicator CSVs from {root}")
    raw = load_v4_indicator_csvs(root, sites)

    print("Preparing stream data (pct_of_baseline, composites)...")
    prepared = prepare_stream_data(raw, sites)

    print("Smoothing...")
    smoothed = apply_smoothing(prepared)

    print("Creating combined stream graph...")
    fig, graph_height = create_combined_stream_graph(smoothed, sites, color_by=color_by)

    if save:
        root = Path(root)
        plot_dir = root / "output" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        import time
        ts = int(time.time())
        sites_str = "-".join(sites)
        root_name = root.name

        html_path = plot_dir / f"v4_stream_graph_{root_name}_{sites_str}_{ts}.html"
        html_content = fig.to_html(
            config={"responsive": False},
            include_plotlyjs=True,
            full_html=True,
            default_width=f"{GRAPH_WIDTH}px",
            default_height=f"{graph_height}px",
        )
        with open(html_path, "w") as f:
            f.write(html_content)
        print(f"Saved HTML: {html_path}")

        png_path = plot_dir / f"v4_stream_graph_{root_name}_{sites_str}_{ts}.png"
        try:
            fig.write_image(str(png_path), width=GRAPH_WIDTH, height=graph_height, scale=2)
            actual_w, actual_h = GRAPH_WIDTH * 2, graph_height * 2
            print(f"Saved PNG ({actual_w}x{actual_h}px @ 2x DPI): {png_path}")
        except Exception as e:
            print(f"Could not save PNG: {e}")

    if show:
        fig.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate V4 stream graphs")
    parser.add_argument("--root", required=True, help="Run output root (e.g. _data-refactored/model-outputs/generated-states/v4.6)")
    parser.add_argument("--sites", type=str, default="trimmed-parade,city,uni")
    parser.add_argument("--color-by", choices=["capability", "persona"], default="capability")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    generate_stream_graph(
        root=args.root,
        sites=[s.strip() for s in args.sites.split(",")],
        color_by=args.color_by,
        save=not args.no_save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
