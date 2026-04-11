"""
Extract V4 indicator voxel counts from yr-180 VTKs.
Reads baseline + positive + trending for each site, computes all V4 indicators.

Respects REFACTOR_RUN_OUTPUT_ROOT env var for the data root.
Writes comparison table to {ROOT}/comparison/v4_indicator_comparison.md
"""
import os
import sys
import numpy as np
import pyvista as pv
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_code-refactored")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.sim.setup.constants import (
    RECRUIT_FULL,
    RECRUIT_PARTIAL,
    RELEASECONTROL_FULL,
    RELEASECONTROL_PARTIAL,
)

_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "_data-refactored" / "model-outputs" / "generated-states" / "v4updatedterms"
ROOT = Path(os.environ.get("REFACTOR_RUN_OUTPUT_ROOT", str(_DEFAULT_ROOT)))
VTK_DIR = ROOT / "output" / "vtks"

SITES = ["trimmed-parade", "city", "uni"]
SCENARIOS = ["positive", "trending"]

def load_vtk(path):
    return pv.read(str(path))

def arr(vtk, name):
    if name in vtk.point_data:
        return vtk.point_data[name]
    return None

def str_arr(vtk, name):
    a = arr(vtk, name)
    if a is None:
        return np.full(vtk.n_points, "none", dtype="<U64")
    return np.char.lower(np.asarray(a).astype(str))

def num_arr(vtk, name):
    a = arr(vtk, name)
    if a is None:
        return np.zeros(vtk.n_points, dtype=float)
    return np.nan_to_num(np.asarray(a).astype(float), nan=0.0)

def ground_not_paved(vtk):
    bio = str_arr(vtk, "search_bioavailable")
    urban = str_arr(vtk, "search_urban_elements")
    ground = (bio == "low-vegetation") | (bio == "open space")
    paved = (urban == "roadway") | (urban == "busy roadway") | (urban == "parking")
    return ground & ~paved

def compute_indicators(vtk, is_baseline=False):
    fs = str_arr(vtk, "forest_size")
    release = str_arr(vtk, "proposal_release_control_intervention")
    recruit = str_arr(vtk, "proposal_recruit_intervention")

    results = {}

    # Bird
    results["Bird.acquire.peeling-bark"] = int(np.sum(num_arr(vtk, "stat_peeling bark") > 0))
    perch = num_arr(vtk, "stat_perch branch") > 0
    bird_comm_fs = np.isin(fs, ["senescing", "snag", "artificial"])
    results["Bird.communicate.perch-branch"] = int(np.sum(perch & bird_comm_fs))
    results["Bird.reproduce.hollow"] = int(np.sum(num_arr(vtk, "stat_hollow") > 0))

    # Lizard
    bio = str_arr(vtk, "search_bioavailable")
    grass = bio == "low-vegetation"
    dead = num_arr(vtk, "stat_dead branch") > 0
    epiphyte = num_arr(vtk, "stat_epiphyte") > 0
    results["Lizard.acquire.grass"] = int(np.sum(grass))
    results["Lizard.acquire.dead-branch"] = int(np.sum(dead))
    results["Lizard.acquire.epiphyte"] = int(np.sum(epiphyte))
    results["Lizard.acquire"] = int(np.sum(grass | dead | epiphyte))
    results["Lizard.communicate.not-paved"] = int(np.sum(ground_not_paved(vtk)))
    nurse = num_arr(vtk, "stat_fallen log") > 0
    fallen_tree = np.isin(fs, ["fallen", "decayed"])
    results["Lizard.reproduce.nurse-log"] = int(np.sum(nurse))
    results["Lizard.reproduce.fallen-tree"] = int(np.sum(fallen_tree))
    results["Lizard.reproduce"] = int(np.sum(nurse | fallen_tree))

    # Tree.acquire
    moderated = release == RELEASECONTROL_PARTIAL
    autonomous = release == RELEASECONTROL_FULL
    results["Tree.acquire.moderated"] = int(np.sum(moderated))
    results["Tree.acquire.autonomous"] = int(np.sum(autonomous))
    results["Tree.acquire"] = int(np.sum(moderated | autonomous))

    # Tree.communicate
    snag = fs == "snag"
    fallen = fs == "fallen"
    decayed = fs == "decayed"
    results["Tree.communicate.snag"] = int(np.sum(snag))
    results["Tree.communicate.fallen"] = int(np.sum(fallen))
    results["Tree.communicate.decayed"] = int(np.sum(decayed))
    results["Tree.communicate"] = int(np.sum(snag | fallen | decayed))

    # Tree.reproduce
    smaller = recruit == RECRUIT_PARTIAL
    larger = recruit == RECRUIT_FULL
    results["Tree.reproduce.smaller-patches-rewild"] = int(np.sum(smaller))
    results["Tree.reproduce.larger-patches-rewild"] = int(np.sum(larger))
    results["Tree.reproduce"] = int(np.sum(smaller | larger))

    # For baseline Tree.reproduce, also grab indicator_Tree_generations_grassland
    if is_baseline:
        grassland_ind = arr(vtk, "indicator_Tree_generations_grassland")
        if grassland_ind is not None:
            results["Tree.reproduce._baseline_grassland"] = int(np.sum(np.asarray(grassland_ind).astype(bool)))

    return results


def write_v4_indicator_csv(
    polydata_or_path,
    site: str,
    scenario: str,
    year: int,
    *,
    output_dir: str | Path | None = None,
    is_baseline: bool = False,
) -> Path:
    """Write per-state V4 indicator CSV.

    Args:
        polydata_or_path: pyvista.PolyData or path to a VTK file.
        site: Site name.
        scenario: Scenario name (or 'baseline').
        year: Simulation year.
        output_dir: Directory to write CSV. If None, uses
            {REFACTOR_RUN_OUTPUT_ROOT}/output/stats/per-state/{site}.
        is_baseline: Whether this is a baseline state.

    Returns:
        Path to the written CSV.
    """
    import pandas as pd

    if isinstance(polydata_or_path, (str, Path)):
        polydata = pv.read(str(polydata_or_path))
    else:
        polydata = polydata_or_path

    counts = compute_indicators(polydata, is_baseline=is_baseline)
    rows = [
        {"site": site, "scenario": scenario, "year": year, "indicator": ind, "count": counts.get(ind, 0)}
        for ind in INDICATOR_ORDER
    ]

    if output_dir is None:
        output_dir = ROOT / "output" / "stats" / "per-state" / site
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_baseline:
        path = output_dir / f"{site}_baseline_1_v4_indicators.csv"
    else:
        path = output_dir / f"{site}_{scenario}_1_yr{year}_v4_indicators.csv"

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def fmt(count):
    return f"{count:,}"

def pct_baseline(count, baseline):
    if baseline == 0:
        return "n/a" if count > 0 else "—"
    return f"{100*count/baseline:.0f}%"

def ratio(pos, trend):
    if trend == 0 and pos == 0:
        return "—"
    if trend == 0:
        return "INF"
    return f"{pos/trend:.1f}x"

def trend_pct_pos(pos, trend):
    if pos == 0:
        return "—"
    return f"{100*trend/pos:.0f}%"


INDICATOR_ORDER = [
    "Bird.acquire.peeling-bark",
    "Bird.communicate.perch-branch",
    "Bird.reproduce.hollow",
    "Lizard.acquire.grass",
    "Lizard.acquire.dead-branch",
    "Lizard.acquire.epiphyte",
    "Lizard.acquire",
    "Lizard.communicate.not-paved",
    "Lizard.reproduce.nurse-log",
    "Lizard.reproduce.fallen-tree",
    "Lizard.reproduce",
    "Tree.acquire.moderated",
    "Tree.acquire.autonomous",
    "Tree.acquire",
    "Tree.communicate.snag",
    "Tree.communicate.fallen",
    "Tree.communicate.decayed",
    "Tree.communicate",
    "Tree.reproduce.smaller-patches-rewild",
    "Tree.reproduce.larger-patches-rewild",
    "Tree.reproduce",
]

AGGREGATES = {
    "Lizard.acquire", "Lizard.reproduce",
    "Tree.acquire", "Tree.communicate", "Tree.reproduce",
}


def format_site_table(site, baseline_counts, pos_counts, trend_counts) -> list[str]:
    """Return markdown table lines for one site."""
    lines = []
    lines.append(f"\n### {site}\n")
    lines.append("| indicator | measure | baseline | positive yr180 | trending yr180 | positive / trending | trending % of positive |")
    lines.append("|---|---|---|---|---|---|---|")

    for ind in INDICATOR_ORDER:
        b = baseline_counts.get(ind, 0)
        p = pos_counts.get(ind, 0)
        t = trend_counts.get(ind, 0)

        # Use grassland baseline for Tree.reproduce aggregate
        if ind == "Tree.reproduce" and "Tree.reproduce._baseline_grassland" in baseline_counts:
            b = baseline_counts["Tree.reproduce._baseline_grassland"]

        # Use full arboreal baseline for Tree.acquire aggregate and autonomous
        # (baseline woodland has all trees as eliminate-pruning equivalent)
        # Actually we just use the computed values directly

        is_agg = ind in AGGREGATES
        bold = "**" if is_agg else ""

        # If trending is 0, substitute ~1% of baseline for comparison columns
        t_display = t
        t_substituted = False
        if t == 0 and b > 0:
            t_display = max(1, round(b * 0.01))
            t_substituted = True
        star = "*" if t_substituted else ""

        pct_b_pos = pct_baseline(p, b)
        pct_b_trend = pct_baseline(t, b)

        if b > 0:
            pos_str = f"{pct_b_pos} of baseline ({fmt(p)})"
            trend_str = f"~1% of baseline (~{fmt(t_display)}){star}" if t_substituted else f"{pct_b_trend} of baseline ({fmt(t)})"
        else:
            pos_str = f"n/a ({fmt(p)})"
            trend_str = f"n/a ({fmt(t)})"

        r = ratio(p, t_display)
        tp = trend_pct_pos(p, t_display)
        r_str = f"{r}{star}"
        tp_str = f"{tp}{star}"

        # measure column - use vtk query syntax
        measure = get_measure(ind)

        lines.append(f"| {bold}{ind}{bold} | {bold}{measure}{bold} | {bold}{fmt(b)}{bold} | {bold}{pos_str}{bold} | {bold}{trend_str}{bold} | {bold}{r_str}{bold} | {bold}{tp_str}{bold} |")

    return lines


def print_site_table(site, baseline_counts, pos_counts, trend_counts):
    for line in format_site_table(site, baseline_counts, pos_counts, trend_counts):
        print(line)


MEASURES = {
    "Bird.acquire.peeling-bark": '`vtk["stat_peeling bark"] > 0`',
    "Bird.communicate.perch-branch": '`vtk["stat_perch branch"] > 0` AND `vtk["forest_size"] in senescing|snag|artificial`',
    "Bird.reproduce.hollow": '`vtk["stat_hollow"] > 0`',
    "Lizard.acquire.grass": '`vtk["search_bioavailable"] == "low-vegetation"`',
    "Lizard.acquire.dead-branch": '`vtk["stat_dead branch"] > 0`',
    "Lizard.acquire.epiphyte": '`vtk["stat_epiphyte"] > 0`',
    "Lizard.acquire": "union",
    "Lizard.communicate.not-paved": '`vtk["search_bioavailable"] in low-vegetation|open space` AND NOT `vtk["search_urban_elements"] in roadway|busy roadway|parking`',
    "Lizard.reproduce.nurse-log": '`vtk["stat_fallen log"] > 0`',
    "Lizard.reproduce.fallen-tree": '`vtk["forest_size"] in fallen|decayed`',
    "Lizard.reproduce": "union",
    "Tree.acquire.moderated": f'`vtk["proposal_release_control_intervention"] == "{RELEASECONTROL_PARTIAL}"`',
    "Tree.acquire.autonomous": f'`vtk["proposal_release_control_intervention"] == "{RELEASECONTROL_FULL}"`',
    "Tree.acquire": "union",
    "Tree.communicate.snag": '`vtk["forest_size"] == "snag"`',
    "Tree.communicate.fallen": '`vtk["forest_size"] == "fallen"`',
    "Tree.communicate.decayed": '`vtk["forest_size"] == "decayed"`',
    "Tree.communicate": '`vtk["forest_size"] in snag|fallen|decayed`',
    "Tree.reproduce.smaller-patches-rewild": f'`vtk["proposal_recruit_intervention"] == "{RECRUIT_PARTIAL}"`',
    "Tree.reproduce.larger-patches-rewild": f'`vtk["proposal_recruit_intervention"] == "{RECRUIT_FULL}"`',
    "Tree.reproduce": "union",
}

def get_measure(ind):
    raw = MEASURES.get(ind, ind)
    # Escape pipe characters so they don't break markdown table columns
    return raw.replace("|", "\\|")


if __name__ == "__main__":
    import sys

    all_lines = ["# V4 Indicator Comparisons (voxel counts, yr 180)", ""]
    all_lines.append(f"Data root: `{ROOT}`")
    all_lines.append("")
    all_lines.append("\\* trending was 0; substituted ~1% of baseline for comparison columns")

    for site in SITES:
        baseline_path = VTK_DIR / site / f"{site}_baseline_1_state_with_indicators.vtk"
        pos_path = VTK_DIR / site / f"{site}_positive_1_yr180_state_with_indicators.vtk"
        trend_path = VTK_DIR / site / f"{site}_trending_1_yr180_state_with_indicators.vtk"

        if not pos_path.exists() or not trend_path.exists():
            print(f"SKIP {site}: VTK not found", file=sys.stderr)
            continue

        print(f"Loading {site}...", file=sys.stderr)
        baseline = load_vtk(baseline_path)
        pos = load_vtk(pos_path)
        trend = load_vtk(trend_path)

        bc = compute_indicators(baseline, is_baseline=True)
        pc = compute_indicators(pos)
        tc = compute_indicators(trend)

        table_lines = format_site_table(site, bc, pc, tc)
        all_lines.extend(table_lines)

        # Also print to stdout
        for line in table_lines:
            print(line)

    # Write to {ROOT}/comparison/v4_indicator_comparison.md
    comparison_dir = ROOT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    output_path = comparison_dir / "v4_indicator_comparison.md"
    output_path.write_text("\n".join(all_lines) + "\n")
    print(f"\nWrote: {output_path}", file=sys.stderr)
