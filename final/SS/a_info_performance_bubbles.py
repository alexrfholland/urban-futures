"""
a_info_performance_bubbles.py
=============================
Generate bubble charts comparing positive vs trending performance over time.

Each bubble represents either:
    - a combined capability outcome (site / persona / capability), or
    - a raw indicator (site / persona / indicator)

The chart uses:
    x-axis: simulation year
    y-axis: relative performance (positive / trending)
    bubble size: positive scenario % of baseline

Bubble packing uses an anchored beeswarm layout, so circles stay close to their
true coordinates while spreading laterally to avoid collisions.

USAGE:
    ./.venv/bin/python final/a_info_performance_bubbles.py --no-show
    ./.venv/bin/python final/a_info_performance_bubbles.py --level indicator --sites trimmed-parade
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "revised" / "final" / "output" / "csv"
PLOT_DIR = SCRIPT_DIR.parent / "data" / "revised" / "final" / "output" / "plots"

DEFAULT_SITES = ["trimmed-parade", "city", "uni"]
GRAPH_WIDTH = 1400
GRAPH_HEIGHT = 880
DEFAULT_Y_SCALE = "log"
DEFAULT_CUTOFF = 25.0
PSEUDO_LOG_FACTOR = 2.0

MIN_MARKER_DIAMETER = 12
MAX_MARKER_DIAMETER = 56
BUBBLE_PADDING_PX = 2
MAX_X_JITTER_YEARS = 20
YEAR_GAP_SHARE = 0.62
MIN_X_JITTER_SHARE = 0.42
SEARCH_STEP_PX = 1.5
RELAXATION_PASSES_LINEAR = 220
RELAXATION_PASSES_LOG = 120
FINAL_CLEANUP_PASSES = 60

PLOT_MARGIN = dict(l=90, r=30, t=110, b=85)

SITE_LABELS = {
    "trimmed-parade": "Parade",
    "city": "City",
    "uni": "Street",
}

CAPABILITY_LABELS = {
    "self": "Acquire Resources",
    "others": "Communicate",
    "generations": "Reproduce",
}

CAPABILITY_COLORS = {
    "self": "#1B9E77",
    "others": "#D95F02",
    "generations": "#7570B3",
}

PERSONA_COLORS = {
    "Bird": "#C83F49",
    "Lizard": "#3D8F5B",
    "Tree": "#2B6CB0",
}

PERSONA_ORDER = ["Bird", "Lizard", "Tree"]
CAPABILITY_ORDER = ["self", "others", "generations"]
AXIS_TICK_CANDIDATES = [1, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


# =============================================================================
# DATA
# =============================================================================

def format_voxel_size(voxel_size: int | float | str) -> str:
    """Normalize voxel size for file naming."""
    try:
        value = float(voxel_size)
        if value.is_integer():
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return str(voxel_size)


def load_indicator_data(sites: list[str], voxel_size: int = 1) -> pd.DataFrame:
    """Load and combine indicator count CSVs for the requested sites."""
    frames = []
    voxel = format_voxel_size(voxel_size)

    all_sites_path = DATA_DIR / f"all_sites_{voxel}_indicator_counts.csv"
    if all_sites_path.exists():
        df = pd.read_csv(all_sites_path)
        df = df[df["site"].isin(sites)].copy()
        if not df.empty:
            return df

    for site in sites:
        path = DATA_DIR / f"{site}_{voxel}_indicator_counts.csv"
        if not path.exists():
            raise FileNotFoundError(f"Indicator counts not found: {path}")
        frames.append(pd.read_csv(path))

    return pd.concat(frames, ignore_index=True)


def build_capability_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw indicators to site/persona/capability outcomes."""
    baseline = (
        df[df["scenario"] == "baseline"]
        .groupby(["site", "persona", "capability", "voxel_size"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "baseline_count"})
    )

    scenario = (
        df[df["scenario"].isin(["positive", "trending"])]
        .groupby(
            ["site", "scenario", "year", "persona", "capability", "voxel_size"],
            as_index=False,
        )["count"]
        .sum()
    )

    merged = scenario.merge(
        baseline,
        on=["site", "persona", "capability", "voxel_size"],
        how="left",
    )
    merged["pct_of_baseline"] = np.where(
        merged["baseline_count"] > 0,
        merged["count"] / merged["baseline_count"] * 100,
        np.nan,
    )
    merged["series_id"] = merged["persona"] + "." + merged["capability"] + ".combined"
    merged["measure_label"] = merged["capability"].map(CAPABILITY_LABELS)
    merged["detail_label"] = (
        merged["persona"] + " / " + merged["capability"].map(CAPABILITY_LABELS)
    )
    return merged


def build_indicator_level(df: pd.DataFrame) -> pd.DataFrame:
    """Keep raw indicator series and ensure baseline counts are attached."""
    baseline = (
        df[df["scenario"] == "baseline"][
            ["site", "indicator_id", "persona", "capability", "label", "voxel_size", "count"]
        ]
        .rename(columns={"count": "baseline_count"})
        .drop_duplicates()
    )

    scenario = df[df["scenario"].isin(["positive", "trending"])].copy()
    merged = scenario.merge(
        baseline,
        on=["site", "indicator_id", "persona", "capability", "label", "voxel_size"],
        how="left",
    )
    merged["pct_of_baseline"] = np.where(
        merged["baseline_count"] > 0,
        merged["count"] / merged["baseline_count"] * 100,
        np.nan,
    )
    merged["series_id"] = merged["indicator_id"]
    merged["measure_label"] = merged["label"]
    merged["detail_label"] = merged["persona"] + " / " + merged["label"]
    return merged


def prepare_performance_data(
    raw_df: pd.DataFrame,
    level: str = "capability",
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Compute positive/trending percentages and relative performance."""
    if years is None:
        years = sorted(
            raw_df.loc[
                raw_df["scenario"].isin(["positive", "trending"]) & (raw_df["year"] >= 0),
                "year",
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

    if level == "capability":
        series_df = build_capability_level(raw_df)
    else:
        series_df = build_indicator_level(raw_df)

    series_df = series_df[series_df["year"].isin(years)].copy()

    pivot = (
        series_df.pivot_table(
            index=[
                "site",
                "year",
                "persona",
                "capability",
                "series_id",
                "measure_label",
                "detail_label",
            ],
            columns="scenario",
            values=["count", "pct_of_baseline"],
            aggfunc="first",
        )
        .reset_index()
    )

    pivot.columns = [
        "_".join([part for part in col if part]).strip("_")
        if isinstance(col, tuple)
        else col
        for col in pivot.columns
    ]

    pivot = pivot.rename(
        columns={
            "count_positive": "positive_count",
            "count_trending": "trending_count",
            "pct_of_baseline_positive": "positive_pct",
            "pct_of_baseline_trending": "trending_pct",
        }
    )

    for col in ["positive_count", "trending_count", "positive_pct", "trending_pct"]:
        if col not in pivot:
            pivot[col] = np.nan

    pivot["site_label"] = pivot["site"].map(SITE_LABELS).fillna(pivot["site"])
    pivot["capability_label"] = (
        pivot["capability"].map(CAPABILITY_LABELS).fillna(pivot["capability"])
    )
    pivot["ratio"] = np.where(
        pivot["trending_pct"] > 0,
        pivot["positive_pct"] / pivot["trending_pct"],
        np.nan,
    )
    pivot["zero_denominator"] = (pivot["trending_pct"] <= 0) & (pivot["positive_pct"] > 0)
    pivot["both_zero"] = (pivot["trending_pct"] <= 0) & (pivot["positive_pct"] <= 0)
    pivot = pivot[~pivot["both_zero"]].copy()

    pivot["ratio_display"] = pivot["ratio"]
    pivot["plottable"] = pivot["ratio_display"].replace([np.inf, -np.inf], np.nan).notna()

    positive_max = pivot["positive_pct"].fillna(0).max()
    positive_max = positive_max if positive_max > 0 else 1.0
    scaled = np.sqrt(pivot["positive_pct"].clip(lower=0) / positive_max)
    pivot["marker_size_px"] = np.where(
        pivot["positive_pct"] > 0,
        MIN_MARKER_DIAMETER + scaled * (MAX_MARKER_DIAMETER - MIN_MARKER_DIAMETER),
        0,
    )

    pivot["capability_color"] = pivot["capability"].map(CAPABILITY_COLORS).fillna("#666666")
    pivot["persona_color"] = pivot["persona"].map(PERSONA_COLORS).fillna("#444444")
    pivot["ratio_label"] = np.where(
        pivot["trending_pct"] > 0,
        pivot["ratio"].map(lambda value: f"{value:.1f}x"),
        "n/a",
    )
    pivot["comparison_label"] = np.where(
        pivot["trending_pct"] > 0,
        pivot.apply(
            lambda row: (
                f"{row['ratio']:.1f}x "
                f"({row['positive_pct']:.1f}% vs {row['trending_pct']:.1f}%)"
            ),
            axis=1,
        ),
        pivot.apply(
            lambda row: f"n/a ({row['positive_pct']:.1f}% vs 0.0%)",
            axis=1,
        ),
    )

    return pivot.sort_values(
        ["site", "persona", "capability", "series_id", "year"]
    ).reset_index(drop=True)


# =============================================================================
# PACKING
# =============================================================================

def year_padding_limits(
    year: int,
    years: list[int],
    x_scale_px_per_year: float,
) -> tuple[float, float]:
    """Return the local horizontal packing bounds in pixels for a year."""
    ordered_years = sorted(set(years))
    index = ordered_years.index(year)
    hard_limit_px = MAX_X_JITTER_YEARS * x_scale_px_per_year
    min_limit_px = hard_limit_px * MIN_X_JITTER_SHARE

    left_limit = 0.0
    right_limit = 0.0

    if index > 0:
        left_gap = year - ordered_years[index - 1]
        left_limit = min(
            max(left_gap * x_scale_px_per_year * YEAR_GAP_SHARE, min_limit_px),
            hard_limit_px,
        )
    elif len(ordered_years) > 1:
        left_limit = 0.0
    if index < len(ordered_years) - 1:
        right_gap = ordered_years[index + 1] - year
        right_limit = min(
            max(right_gap * x_scale_px_per_year * YEAR_GAP_SHARE, min_limit_px),
            hard_limit_px,
        )
    elif len(ordered_years) > 1:
        right_limit = 0.0

    if index == 0:
        right_limit = hard_limit_px
    if index == len(ordered_years) - 1:
        left_limit = hard_limit_px

    return (-left_limit, right_limit)


def transform_y_value(value: float, y_scale: str) -> float:
    """Map a ratio onto the requested axis transform."""
    value = max(float(value), 0.0)
    if y_scale == "log":
        return math.log10(max(value, 1.0))
    if y_scale == "sqrt":
        return math.sqrt(value)
    if y_scale == "pseudo_log":
        return math.asinh(value / PSEUDO_LOG_FACTOR)
    return value


def inverse_transform_y_value(value: float, y_scale: str) -> float:
    """Invert a transformed y value back to the original ratio."""
    value = float(value)
    if y_scale == "log":
        return 10 ** value
    if y_scale == "sqrt":
        return max(value, 0.0) ** 2
    if y_scale == "pseudo_log":
        return math.sinh(value) * PSEUDO_LOG_FACTOR
    return value


def plot_y_value(value: float, y_scale: str) -> float:
    """Return the coordinate used on the Plotly y-axis."""
    if y_scale in {"sqrt", "pseudo_log"}:
        return transform_y_value(value, y_scale)
    return float(value)


def apply_horizontal_packing(
    df: pd.DataFrame,
    years: list[int],
    graph_width: int,
    graph_height: int,
    y_scale: str = DEFAULT_Y_SCALE,
    plottable_col: str = "plottable",
    ratio_min: float | None = None,
    ratio_max: float | None = None,
) -> pd.DataFrame:
    """Pack bubbles around their anchors using an anchored beeswarm layout."""
    packed = df.copy()
    plot_width_px = graph_width - PLOT_MARGIN["l"] - PLOT_MARGIN["r"]
    plot_height_px = graph_height - PLOT_MARGIN["t"] - PLOT_MARGIN["b"]
    x_min, x_max = min(years), max(years)
    x_scale_px_per_year = plot_width_px / max(x_max - x_min, 1)

    plottable = packed[packed[plottable_col]].copy()
    if plottable.empty:
        packed["x_offset_px"] = 0.0
        packed["x_offset_years"] = 0.0
        packed["x_packed"] = packed["year"].astype(float)
        packed["y_packed"] = packed["ratio_display"]
        packed["y_plot"] = packed["ratio_display"]
        return packed

    if y_scale == "log":
        y_min = max(float(ratio_min) if ratio_min is not None else float(plottable["ratio_display"].min()), 1.0)
        y_max = float(ratio_max) if ratio_max is not None else max(float(plottable["ratio_display"].max()) * 1.08, y_min * 1.05)
    else:
        y_min = max(1.0, float(ratio_min) if ratio_min is not None else 1.0)
        y_max = float(ratio_max) if ratio_max is not None else max(float(plottable["ratio_display"].max()) * 1.05, 1.0)

    scaled_y_min = transform_y_value(y_min, y_scale)
    scaled_y_max = transform_y_value(y_max, y_scale)

    def to_y_px(value: float) -> float:
        scaled = (transform_y_value(float(value), y_scale) - scaled_y_min) / max(scaled_y_max - scaled_y_min, 1e-9)
        return scaled * plot_height_px

    def from_y_px(value_px: float) -> float:
        clamped_px = float(np.clip(value_px, 0.0, plot_height_px))
        scaled_value = scaled_y_min + (clamped_px / max(plot_height_px, 1e-9)) * (scaled_y_max - scaled_y_min)
        return inverse_transform_y_value(scaled_value, y_scale)

    packed["x_offset_px"] = 0.0
    packed["x_offset_years"] = 0.0
    packed["x_packed"] = packed["year"].astype(float)
    packed["y_packed"] = packed["ratio_display"].astype(float)
    packed["y_plot"] = packed["ratio_display"].astype(float)

    if plottable.empty:
        return packed

    sim = plottable.copy()
    sim["anchor_x_px"] = (sim["year"].astype(float) - x_min) * x_scale_px_per_year
    sim["anchor_y_px"] = sim["ratio_display"].map(to_y_px)
    sim["x_px"] = sim["anchor_x_px"]
    sim["y_px"] = sim["anchor_y_px"]
    sim["radius_px"] = sim["marker_size_px"] / 2.0 + BUBBLE_PADDING_PX + 0.5
    sim["left_limit_px"] = 0.0
    sim["right_limit_px"] = 0.0

    for idx in sim.index:
        left_limit, right_limit = year_padding_limits(
            int(sim.at[idx, "year"]),
            years,
            x_scale_px_per_year,
        )
        sim.at[idx, "left_limit_px"] = left_limit
        sim.at[idx, "right_limit_px"] = right_limit

    def x_bounds(row_index: int) -> tuple[float, float]:
        anchor_x = float(sim.at[row_index, "anchor_x_px"])
        radius = float(sim.at[row_index, "radius_px"])
        left_bound = anchor_x + float(sim.at[row_index, "left_limit_px"])
        right_bound = anchor_x + float(sim.at[row_index, "right_limit_px"])
        min_x = max(radius, left_bound)
        max_x = min(plot_width_px - radius, right_bound)
        if min_x > max_x:
            midpoint = (min_x + max_x) / 2.0
            min_x = midpoint
            max_x = midpoint
        return (min_x, max_x)

    def has_overlap(candidate_x: float, y_px: float, radius_px: float, placed: list[int]) -> bool:
        for other_idx in placed:
            other_x = float(sim.at[other_idx, "x_px"])
            other_y = float(sim.at[other_idx, "y_px"])
            other_radius = float(sim.at[other_idx, "radius_px"])
            if math.hypot(candidate_x - other_x, y_px - other_y) < (radius_px + other_radius):
                return True
        return False

    def candidate_offsets(limit_left: float, limit_right: float) -> list[float]:
        max_offset = max(abs(limit_left), abs(limit_right))
        offsets = [0.0]
        steps = int(max_offset / SEARCH_STEP_PX) + 2
        for step in range(1, steps):
            delta = step * SEARCH_STEP_PX
            if delta <= limit_right:
                offsets.append(delta)
            if -delta >= limit_left:
                offsets.append(-delta)
        return offsets

    order = sim.sort_values(
        ["marker_size_px", "anchor_y_px", "detail_label"],
        ascending=[False, True, True],
    ).index.tolist()

    placed: list[int] = []
    for idx in order:
        anchor_x = float(sim.at[idx, "anchor_x_px"])
        anchor_y = float(sim.at[idx, "anchor_y_px"])
        radius = float(sim.at[idx, "radius_px"])
        min_x, max_x = x_bounds(idx)

        candidate_xs = [float(np.clip(anchor_x, min_x, max_x))]
        for other_idx in placed:
            other_y = float(sim.at[other_idx, "y_px"])
            combined_radius = radius + float(sim.at[other_idx, "radius_px"])
            dy = anchor_y - other_y
            if abs(dy) >= combined_radius:
                continue
            dx = math.sqrt(max(combined_radius ** 2 - dy ** 2, 0.0))
            other_x = float(sim.at[other_idx, "x_px"])
            candidate_xs.extend([other_x - dx, other_x + dx])

        limit_left = min_x - anchor_x
        limit_right = max_x - anchor_x
        for offset in candidate_offsets(limit_left, limit_right):
            candidate_xs.append(anchor_x + offset)

        best_x = float(np.clip(anchor_x, min_x, max_x))
        best_penalty = float("inf")
        for candidate_x in sorted(set(round(value, 3) for value in candidate_xs), key=lambda value: (abs(value - anchor_x), value)):
            candidate_x = float(np.clip(candidate_x, min_x, max_x))
            if has_overlap(candidate_x, anchor_y, radius, placed):
                continue
            penalty = abs(candidate_x - anchor_x)
            if penalty < best_penalty:
                best_penalty = penalty
                best_x = candidate_x
                if penalty <= 0.5:
                    break

        sim.at[idx, "x_px"] = best_x
        sim.at[idx, "y_px"] = anchor_y
        placed.append(idx)

    if y_scale == "linear":
        relax_passes = RELAXATION_PASSES_LINEAR
        spring_x = 0.055
        spring_y = 0.11
        cleanup_passes = FINAL_CLEANUP_PASSES
    elif y_scale == "sqrt":
        relax_passes = RELAXATION_PASSES_LINEAR
        spring_x = 0.06
        spring_y = 0.12
        cleanup_passes = FINAL_CLEANUP_PASSES + 40
    elif y_scale == "pseudo_log":
        relax_passes = RELAXATION_PASSES_LINEAR
        spring_x = 0.065
        spring_y = 0.125
        cleanup_passes = FINAL_CLEANUP_PASSES + 20
    else:
        relax_passes = RELAXATION_PASSES_LOG
        spring_x = 0.07
        spring_y = 0.14
        cleanup_passes = FINAL_CLEANUP_PASSES

    for _ in range(relax_passes):
        max_overlap = 0.0
        for i, idx_i in enumerate(order):
            for idx_j in order[i + 1:]:
                xi = float(sim.at[idx_i, "x_px"])
                yi = float(sim.at[idx_i, "y_px"])
                ri = float(sim.at[idx_i, "radius_px"])
                xj = float(sim.at[idx_j, "x_px"])
                yj = float(sim.at[idx_j, "y_px"])
                rj = float(sim.at[idx_j, "radius_px"])

                dx = xj - xi
                dy = yj - yi
                distance = math.hypot(dx, dy)
                min_distance = ri + rj

                if distance < 1e-6:
                    dx = 0.001 * (((idx_i + idx_j) % 3) - 1)
                    dy = 1.0
                    distance = math.hypot(dx, dy)

                if distance >= min_distance:
                    continue

                overlap = min_distance - distance
                max_overlap = max(max_overlap, overlap)
                ux = dx / distance
                uy = dy / distance
                push = overlap * 0.54

                xi -= ux * push * 1.08
                yi -= uy * push * 0.82
                xj += ux * push * 1.08
                yj += uy * push * 0.82

                min_x_i, max_x_i = x_bounds(idx_i)
                min_x_j, max_x_j = x_bounds(idx_j)
                sim.at[idx_i, "x_px"] = float(np.clip(xi, min_x_i, max_x_i))
                sim.at[idx_i, "y_px"] = float(np.clip(yi, ri, plot_height_px - ri))
                sim.at[idx_j, "x_px"] = float(np.clip(xj, min_x_j, max_x_j))
                sim.at[idx_j, "y_px"] = float(np.clip(yj, rj, plot_height_px - rj))

        for idx in order:
            anchor_x = float(sim.at[idx, "anchor_x_px"])
            anchor_y = float(sim.at[idx, "anchor_y_px"])
            current_x = float(sim.at[idx, "x_px"])
            current_y = float(sim.at[idx, "y_px"])
            radius = float(sim.at[idx, "radius_px"])
            min_x, max_x = x_bounds(idx)
            pulled_x = current_x + (anchor_x - current_x) * spring_x
            pulled_y = current_y + (anchor_y - current_y) * spring_y
            sim.at[idx, "x_px"] = float(np.clip(pulled_x, min_x, max_x))
            sim.at[idx, "y_px"] = float(np.clip(pulled_y, radius, plot_height_px - radius))

        if max_overlap < 0.15:
            break

    for _ in range(cleanup_passes):
        max_overlap = 0.0
        for i, idx_i in enumerate(order):
            for idx_j in order[i + 1:]:
                xi = float(sim.at[idx_i, "x_px"])
                yi = float(sim.at[idx_i, "y_px"])
                ri = float(sim.at[idx_i, "radius_px"])
                xj = float(sim.at[idx_j, "x_px"])
                yj = float(sim.at[idx_j, "y_px"])
                rj = float(sim.at[idx_j, "radius_px"])

                dx = xj - xi
                dy = yj - yi
                distance = math.hypot(dx, dy)
                min_distance = ri + rj

                if distance < 1e-6:
                    dx = 0.001 * (((idx_i + idx_j) % 3) - 1)
                    dy = 1.0
                    distance = math.hypot(dx, dy)

                if distance >= min_distance:
                    continue

                overlap = min_distance - distance
                max_overlap = max(max_overlap, overlap)
                ux = dx / distance
                uy = dy / distance
                push = overlap * 0.58

                xi -= ux * push * 1.10
                yi -= uy * push * 0.90
                xj += ux * push * 1.10
                yj += uy * push * 0.90

                min_x_i, max_x_i = x_bounds(idx_i)
                min_x_j, max_x_j = x_bounds(idx_j)
                sim.at[idx_i, "x_px"] = float(np.clip(xi, min_x_i, max_x_i))
                sim.at[idx_i, "y_px"] = float(np.clip(yi, ri, plot_height_px - ri))
                sim.at[idx_j, "x_px"] = float(np.clip(xj, min_x_j, max_x_j))
                sim.at[idx_j, "y_px"] = float(np.clip(yj, rj, plot_height_px - rj))

        if max_overlap < 0.06:
            break

    for idx in sim.index:
        x_px = float(sim.at[idx, "x_px"])
        y_px = float(sim.at[idx, "y_px"])
        anchor_x = float(sim.at[idx, "anchor_x_px"])
        packed.at[idx, "x_offset_px"] = x_px - anchor_x
        packed.at[idx, "x_offset_years"] = (x_px - anchor_x) / x_scale_px_per_year
        packed.at[idx, "x_packed"] = x_min + (x_px / x_scale_px_per_year)
        y_value = from_y_px(y_px)
        packed.at[idx, "y_packed"] = y_value
        packed.at[idx, "y_plot"] = plot_y_value(y_value, y_scale)

    return packed


# =============================================================================
# PLOTTING
# =============================================================================

def add_marker_traces(fig: go.Figure, plot_df: pd.DataFrame) -> None:
    """Add packed marker traces using capability fill and persona outline."""
    for site in DEFAULT_SITES:
        for persona in PERSONA_ORDER:
            for capability in CAPABILITY_ORDER:
                subset = plot_df[
                    (plot_df["site"] == site)
                    & (plot_df["persona"] == persona)
                    & (plot_df["capability"] == capability)
                    & (plot_df["plottable"])
                ].copy()
                if subset.empty:
                    continue

                customdata = np.array(
                    list(
                        zip(
                            subset["site_label"],
                            subset["persona"],
                            subset["capability_label"],
                            subset["measure_label"],
                            subset["comparison_label"],
                            subset["positive_pct"].round(1),
                            subset["trending_pct"].round(1),
                            subset["positive_count"].fillna(0).round(0),
                            subset["trending_count"].fillna(0).round(0),
                            subset["year"],
                        )
                    ),
                    dtype=object,
                )

                fig.add_trace(
                    go.Scatter(
                        x=subset["x_packed"],
                        y=subset["y_plot"],
                        mode="markers",
                        marker=dict(
                            size=subset["marker_size_px"],
                            sizemode="diameter",
                            color=CAPABILITY_COLORS[capability],
                            symbol="circle",
                            opacity=0.8,
                            line=dict(color=PERSONA_COLORS[persona], width=2),
                        ),
                        customdata=customdata,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Persona: %{customdata[1]}<br>"
                            "Capability: %{customdata[2]}<br>"
                            "Measure: %{customdata[3]}<br>"
                            "Year: %{customdata[9]:.0f}<br>"
                            "Performance: %{customdata[4]}<br>"
                            "Positive: %{customdata[5]:.1f}% of baseline (%{customdata[7]:,.0f})<br>"
                            "Human-led: %{customdata[6]:.1f}% of baseline (%{customdata[8]:,.0f})"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )


def add_legends(fig: go.Figure) -> None:
    """Add separate legend entries for capability fills and persona outlines."""
    for index, capability in enumerate(CAPABILITY_ORDER):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=14,
                    color=CAPABILITY_COLORS[capability],
                    line=dict(color="rgba(0, 0, 0, 0.25)", width=1),
                ),
                name=CAPABILITY_LABELS[capability],
                legendgroup="capability",
                legendgrouptitle_text="Fill = Capability" if index == 0 else None,
                showlegend=True,
            )
        )

    for index, persona in enumerate(PERSONA_ORDER):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=14,
                    color="white",
                    line=dict(color=PERSONA_COLORS[persona], width=3),
                ),
                name=persona,
                legendgroup="persona",
                legendgrouptitle_text="Outline = Persona" if index == 0 else None,
                showlegend=True,
            )
        )


def filter_plot_data(
    packed_df: pd.DataFrame,
    sites: list[str],
    ratio_min: float | None = None,
    ratio_max: float | None = None,
) -> pd.DataFrame:
    """Filter a packed dataset to a ratio band while keeping zero-denominator rows available for notes."""
    plot_df = packed_df[packed_df["site"].isin(sites)].copy()
    finite_mask = plot_df["plottable"]
    if ratio_min is not None:
        finite_mask &= plot_df["ratio_display"] >= ratio_min
    if ratio_max is not None:
        finite_mask &= plot_df["ratio_display"] <= ratio_max
    plot_df["plottable_view"] = finite_mask
    return plot_df


def get_axis_bounds(
    finite_df: pd.DataFrame,
    y_scale: str,
    ratio_min: float | None = None,
    ratio_max: float | None = None,
) -> tuple[float, float]:
    """Return y-axis bounds for a filtered view."""
    if finite_df.empty:
        raise ValueError("No finite ratios available for the requested range.")

    if y_scale == "log":
        y_min = max(float(ratio_min) if ratio_min is not None else float(finite_df["ratio_display"].min()), 1.0)
        y_max = float(ratio_max) if ratio_max is not None else max(float(finite_df["ratio_display"].max()) * 1.08, y_min * 1.05)
        return y_min, y_max

    y_min = max(1.0, float(ratio_min) if ratio_min is not None else 1.0)
    y_max = float(ratio_max) if ratio_max is not None else max(float(finite_df["ratio_display"].max()) * 1.05, 1.0)
    return y_min, y_max


def get_custom_tick_values(y_scale: str, y_min: float, y_max: float) -> tuple[list[float], list[str]]:
    """Return transformed tick positions and readable labels for custom y scales."""
    tick_source = [value for value in AXIS_TICK_CANDIDATES if value >= y_min and value <= y_max * 1.001]
    if not tick_source:
        tick_source = [y_min, y_max]
    tickvals = [plot_y_value(value, y_scale) for value in tick_source]
    ticktext = [f"{value:g}x" for value in tick_source]
    return tickvals, ticktext

def create_performance_figure(
    packed_df: pd.DataFrame,
    sites: list[str],
    level: str,
    graph_width: int = GRAPH_WIDTH,
    graph_height: int = GRAPH_HEIGHT,
    y_scale: str = DEFAULT_Y_SCALE,
    ratio_min: float | None = None,
    ratio_max: float | None = None,
    title_suffix: str = "",
) -> go.Figure:
    """Create a single-panel bubble chart for all selected sites."""
    fig = go.Figure()
    plot_df = filter_plot_data(
        packed_df=packed_df,
        sites=sites,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
    )
    marker_df = plot_df[plot_df["plottable_view"]].copy()
    add_marker_traces(fig, marker_df)
    add_legends(fig)

    finite_df = marker_df.copy()
    if finite_df.empty:
        raise ValueError("No finite ratios available for the requested range.")

    y_min, y_max = get_axis_bounds(
        finite_df=finite_df,
        y_scale=y_scale,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
    )
    scale_label = {
        "linear": "linear",
        "log": "log",
        "sqrt": "square-root",
        "pseudo_log": "pseudo-log",
    }.get(y_scale, y_scale)

    fig.update_layout(
        title=dict(
            text=(
                f"Performance Bubble Comparison ({level.capitalize()} level)"
                f"{title_suffix}"
                f"<br><sup>Y = positive / human-led ({scale_label} scale), size = positive % of baseline</sup>"
            ),
            x=0.5,
        ),
        width=graph_width,
        height=graph_height,
        autosize=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        margin=PLOT_MARGIN,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_xaxes(
        title_text="Year",
        tickmode="array",
        tickvals=sorted(marker_df["year"].dropna().astype(int).unique().tolist()),
        ticktext=[str(year) for year in sorted(marker_df["year"].dropna().astype(int).unique().tolist())],
        range=[marker_df["year"].min() - 6, marker_df["year"].max() + 6],
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        zeroline=False,
    )

    if y_scale == "log":
        tickvals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        tickvals = [value for value in tickvals if value >= max(y_min, 1.0) and value <= y_max * 1.4]
        fig.update_yaxes(
            title_text="Relative performance (x)",
            type="log",
            range=[math.log10(max(y_min, 1.0)), math.log10(y_max * 1.08)],
            tickmode="array",
            tickvals=tickvals,
            ticktext=[f"{value:g}x" for value in tickvals],
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.08)",
            zeroline=False,
        )
    else:
        if y_scale in {"sqrt", "pseudo_log"}:
            tickvals, ticktext = get_custom_tick_values(y_scale, y_min, y_max)
            upper = plot_y_value(max(y_max * 1.05, y_max + 0.1), y_scale)
            lower = plot_y_value(y_min, y_scale)
            fig.update_yaxes(
                title_text="Relative performance (x)",
                range=[lower, upper],
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                showgrid=True,
                gridcolor="rgba(0, 0, 0, 0.08)",
                zeroline=False,
            )
        else:
            fig.update_yaxes(
                title_text="Relative performance (x)",
                range=[
                    1,
                    max(y_max * 1.05, ratio_max if ratio_max is not None else y_max * 1.05),
                ],
                showgrid=True,
                gridcolor="rgba(0, 0, 0, 0.08)",
                zeroline=False,
            )

    omitted = plot_df[plot_df["zero_denominator"]].copy()
    note_parts = []
    if ratio_max is not None:
        high_count = int(plot_df[(plot_df["plottable"]) & (plot_df["ratio_display"] > ratio_max)].shape[0])
        if high_count:
            note_parts.append(f"{high_count} finite point(s) above {ratio_max:.0f}x omitted from this view")
    if ratio_min is not None:
        low_count = int(plot_df[(plot_df["plottable"]) & (plot_df["ratio_display"] < ratio_min)].shape[0])
        if low_count:
            note_parts.append(f"{low_count} finite point(s) below {ratio_min:.0f}x omitted from this view")
    if not omitted.empty:
        note_parts.append(
            f"{len(omitted)} undefined zero-denominator point(s) omitted because human-led = 0.0%"
        )

    if note_parts:
        note_parts.append("Bubble positions are collision-adjusted around their true coordinates; hover shows the exact underlying values")
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.0,
            y=-0.09,
            xanchor="right",
            showarrow=False,
            font=dict(size=11, color="rgba(60, 60, 60, 0.85)"),
            text=". ".join(note_parts) + ".",
        )

    return fig


def save_outputs(
    fig: go.Figure,
    output_df: pd.DataFrame,
    sites: list[str],
    voxel_size: int,
    level: str,
    graph_width: int,
    graph_height: int,
    suffix: str,
) -> tuple[Path, Path]:
    """Save plot HTML and plotting CSV."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    sites_str = "-".join(sites)
    base_name = f"performance_bubbles_{level}_{suffix}_{voxel_size}_{sites_str}_{timestamp}"

    html_path = PLOT_DIR / f"{base_name}.html"
    csv_path = DATA_DIR / f"{base_name}.csv"

    html_content = fig.to_html(
        config={"responsive": False},
        include_plotlyjs=True,
        full_html=True,
        default_width=f"{graph_width}px",
        default_height=f"{graph_height}px",
    )
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(html_content)

    output_df.to_csv(csv_path, index=False)

    try:
        png_path = PLOT_DIR / f"{base_name}.png"
        fig.write_image(
            str(png_path),
            width=graph_width,
            height=graph_height,
            scale=2,
        )
        print(f"Saved PNG:  {png_path}")
    except Exception as exc:
        print(f"Could not save PNG (kaleido may not be installed): {exc}")

    return html_path, csv_path


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate performance bubble charts from indicator counts."
    )
    parser.add_argument(
        "--sites",
        type=str,
        default="trimmed-parade,city,uni",
        help="Comma-separated list of sites to include.",
    )
    parser.add_argument(
        "--voxel-size",
        type=int,
        default=1,
        help="Voxel size for the indicator count CSV.",
    )
    parser.add_argument(
        "--level",
        choices=["capability", "indicator"],
        default="capability",
        help="Bubble grouping level.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=GRAPH_WIDTH,
        help=f"Figure width in pixels (default: {GRAPH_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=GRAPH_HEIGHT,
        help=f"Figure height in pixels (default: {GRAPH_HEIGHT}).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CUTOFF,
        help=f"Split between the linear and high-range views in x-units (default: {DEFAULT_CUTOFF}).",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years to include. Defaults to all non-baseline years present in the source CSV.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the figure in a browser window.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write HTML/CSV outputs.",
    )
    parser.add_argument(
        "--extra-full-range-scales",
        type=str,
        default=None,
        help="Comma-separated extra full-range y scales to render. Options: sqrt, pseudo_log.",
    )

    args = parser.parse_args()
    sites = [site.strip() for site in args.sites.split(",") if site.strip()]
    years = (
        [int(year.strip()) for year in args.years.split(",") if year.strip()]
        if args.years
        else None
    )
    extra_full_range_scales = (
        [scale.strip() for scale in args.extra_full_range_scales.split(",") if scale.strip()]
        if args.extra_full_range_scales
        else []
    )
    invalid_scales = [scale for scale in extra_full_range_scales if scale not in {"sqrt", "pseudo_log"}]
    if invalid_scales:
        raise ValueError(f"Unsupported extra full-range scales: {', '.join(invalid_scales)}")

    print(f"Loading indicator data for: {sites}")
    raw_df = load_indicator_data(sites=sites, voxel_size=args.voxel_size)

    print(f"Preparing {args.level}-level performance series...")
    performance_df = prepare_performance_data(
        raw_df=raw_df,
        level=args.level,
        years=years,
    )
    years_for_plot = sorted(performance_df["year"].dropna().astype(int).unique().tolist())

    print("Building plot...")
    linear_df = filter_plot_data(
        packed_df=performance_df,
        sites=sites,
        ratio_max=args.cutoff,
    )
    print("Packing linear view...")
    linear_df = apply_horizontal_packing(
        linear_df,
        years=years_for_plot,
        graph_width=args.width,
        graph_height=args.height,
        y_scale="linear",
        plottable_col="plottable_view",
        ratio_max=args.cutoff,
    )
    high_df = filter_plot_data(
        packed_df=performance_df,
        sites=sites,
        ratio_min=args.cutoff,
    )
    print("Packing high-range view...")
    high_df = apply_horizontal_packing(
        high_df,
        years=years_for_plot,
        graph_width=args.width,
        graph_height=args.height,
        y_scale=DEFAULT_Y_SCALE,
        plottable_col="plottable_view",
        ratio_min=args.cutoff,
    )

    fig_linear = create_performance_figure(
        packed_df=linear_df,
        sites=sites,
        level=args.level,
        graph_width=args.width,
        graph_height=args.height,
        y_scale="linear",
        ratio_max=args.cutoff,
        title_suffix=f", 1-{args.cutoff:.0f}x",
    )
    fig_high = create_performance_figure(
        packed_df=high_df,
        sites=sites,
        level=args.level,
        graph_width=args.width,
        graph_height=args.height,
        y_scale=DEFAULT_Y_SCALE,
        ratio_min=args.cutoff,
        title_suffix=f", >{args.cutoff:.0f}x",
    )

    extra_outputs: list[tuple[go.Figure, pd.DataFrame, str]] = []
    for scale in extra_full_range_scales:
        print(f"Packing full-range {scale} view...")
        extra_df = apply_horizontal_packing(
            performance_df,
            years=years_for_plot,
            graph_width=args.width,
            graph_height=args.height,
            y_scale=scale,
            plottable_col="plottable",
        )
        extra_fig = create_performance_figure(
            packed_df=extra_df,
            sites=sites,
            level=args.level,
            graph_width=args.width,
            graph_height=args.height,
            y_scale=scale,
            title_suffix=f", full range {scale.replace('_', '-')}",
        )
        extra_outputs.append((extra_fig, extra_df, scale))

    if not args.no_save:
        html_linear, csv_linear = save_outputs(
            fig=fig_linear,
            output_df=linear_df,
            sites=sites,
            voxel_size=args.voxel_size,
            level=args.level,
            graph_width=args.width,
            graph_height=args.height,
            suffix=f"linear_upto_{int(args.cutoff)}x",
        )
        html_high, csv_high = save_outputs(
            fig=fig_high,
            output_df=high_df,
            sites=sites,
            voxel_size=args.voxel_size,
            level=args.level,
            graph_width=args.width,
            graph_height=args.height,
            suffix=f"log_above_{int(args.cutoff)}x",
        )
        print(f"Saved HTML: {html_linear}")
        print(f"Saved HTML: {html_high}")
        print(f"Saved CSV:  {csv_linear}")
        print(f"Saved CSV:  {csv_high}")
        for extra_fig, extra_df, scale in extra_outputs:
            html_extra, csv_extra = save_outputs(
                fig=extra_fig,
                output_df=extra_df,
                sites=sites,
                voxel_size=args.voxel_size,
                level=args.level,
                graph_width=args.width,
                graph_height=args.height,
                suffix=f"{scale}_all_range",
            )
            print(f"Saved HTML: {html_extra}")
            print(f"Saved CSV:  {csv_extra}")

    if not args.no_show:
        fig_linear.show()
        fig_high.show()
        for extra_fig, _, _ in extra_outputs:
            extra_fig.show()


if __name__ == "__main__":
    main()
