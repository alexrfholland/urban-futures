"""
Comparisons to external targets and how we compute them from the model.

Headings:
- 1.1 Target for tree canopy cover and shrubs
- 1.2 Target for hollow bearing trees and artificial hollows
- 1.3 Target for removing large, old trees
- 1.4 Target for improving permeability
- 1.5 Target for improving connectivity (Lizards and Hollow-nesting Birds)
- 1.6 Linking Melbourne targets

Notes:
- This module focuses first on 1.3 (removing large, old trees).
- No files are written by default; functions return DataFrames.
"""

from typing import List, Dict, Any, Set, Tuple, Optional
import os
import pandas as pd
import math


# ----------------------------------------------------------------------------
# Defaults and constants for clarity and reuse
# ----------------------------------------------------------------------------
DEFAULT_SITES: List[str] = ["trimmed-parade", "city", "uni"]
DEFAULT_SCENARIOS: List[str] = ["trending", "positive"]
DEFAULT_TIMESTEPS: List[int] = [0, 10, 30, 60, 180]
DEFAULT_VOXEL_SIZE: int = 1

# Tree lifecycle cohort used for denominator in replacement stats
COHORT_SIZES: List[str] = ["large", "senescing", "snag", "fallen"]

# Common data paths
RESOURCE_DIC_PATH: str = "data/revised/trees/resource_dicDF.csv"
OUTPUT_COMPARISON_DIR: str = "data/revised/final/comparison"
OUTPUT_ART_HOLLOWS_FILE: str = "comparison_artificial-hollows.csv"


# Small utility to avoid repeating directory creation
def _ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _compute_annual_rate_from_steps(sorted_pairs: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
    """
    Given [(year, value), ...] sorted by year, compute per-step deltas and annualized rates.
    Returns list of dicts with year, value, years_since_prev, delta, annual_rate.
    """
    results: List[Dict[str, Any]] = []
    prev_year: Optional[int] = None
    prev_value: Optional[float] = None
    for year, val in sorted_pairs:
        if prev_year is None:
            results.append(
                {
                    "timestep": year,
                    "value": val,
                    "years_since_prev": 0,
                    "delta": 0.0,
                    "annual_rate": 0.0,
                }
            )
        else:
            years = max(0, year - prev_year)
            delta = val - (prev_value if prev_value is not None else 0.0)
            rate = (delta / years) if years > 0 else 0.0
            results.append(
                {
                    "timestep": year,
                    "value": val,
                    "years_since_prev": years,
                    "delta": delta,
                    "annual_rate": rate,
                }
            )
        prev_year = year
        prev_value = val
    return results

# ---------------------------------------------------------------------------
# 1.1 Target for tree canopy cover and shrubs (skeleton)
# ---------------------------------------------------------------------------

def compute_canopy_shrub_targets() -> None:
    """Placeholder for canopy + shrubs target computations (1.1)."""
    return None
#
# 1.1b Canopy+Shrubs target tables and unified comparison
#

def save_canopy_targets(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "comparison_canopy-shrubs_targets.csv",
    site_area_m2: float = 157_641.0,
) -> str:
    """
    Persist Living Melbourne canopy/shrub targets for Inner and Eastern contexts.
    Includes full calculation columns:
    - coverage_start_pct, coverage_end_pct, pct_change
    - years (end-start)
    - site_area_m2
    - total_change_m2 = site_area_m2 * pct_change/100
    - annual_rate_m2_per_year = total_change_m2 / years
    """
    _ensure_directory(out_dir)
    rows: List[Dict[str, Any]] = []

    def add_row(context: str, start_year: int, end_year: int, pct_start: float, pct_end: float):
        years = end_year - start_year
        pct_change = pct_end - pct_start
        total_change_m2 = site_area_m2 * (pct_change / 100.0)
        annual_m2 = (total_change_m2 / years) if years > 0 else 0.0
        rows.append(
            {
                "context": context,
                "start_year": start_year,
                "end_year": end_year,
                "years": years,
                "coverage_start_pct": pct_start,
                "coverage_end_pct": pct_end,
                "pct_change": pct_change,
                "site_area_m2": site_area_m2,
                "total_change_m2": total_change_m2,
                "annual_rate_m2_per_year": annual_m2,
            }
        )

    # Inner
    add_row("Inner", 2015, 2030, 18.0, 23.0)
    add_row("Inner", 2030, 2040, 23.0, 28.0)
    add_row("Inner", 2040, 2050, 28.0, 33.0)

    # Eastern
    add_row("Eastern", 2015, 2030, 44.0, 49.0)
    add_row("Eastern", 2030, 2040, 49.0, 50.0)
    add_row("Eastern", 2040, 2050, 50.0, 50.0)

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


def save_canopy_unified(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "comparison_canopy-shrubs_unified.csv",
    sites: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
) -> str:
    """
    Compute our per-step annualized change for canopy+shrubs and save CSV.
    Columns include:
    - arboreal_2d_m2 (m2)
    - low_veg_3d_m2 (m2)
    - canopy cover + shrubs (total m2)
    - canopy cover + shrubs (increase since last year)
    - annual canopy cover + shrubs added (m2/year)
    """
    if sites is None:
        sites = DEFAULT_SITES
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS

    # Build prediction dataframe (includes arboreal_2d_m2 and low_veg_3d_m2)
    out_df = build_canopy_shrubs_prediction(
        sites=sites, scenarios=scenarios, timesteps=timesteps, voxel_size=voxel_size
    )
    _ensure_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    out_df.to_csv(out_path, index=False)
    return out_path


def build_canopy_shrubs_prediction(
    sites: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
) -> pd.DataFrame:
    """
    Build "canopy-shrubs_prediction" as a DataFrame using explicit, labeled steps.
    Returns a DataFrame; saving is performed by save_canopy_unified().
    Adds both arboreal 2D and low-vegetation 3D columns for transparency,
    plus a total_canopy_shrub_m2 = arboreal_2d_m2 + low_veg_3d_m2 column.
    """
    if sites is None:
        sites = DEFAULT_SITES
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS

    # STEP 1: calculate canopy_shrubs from VTK files (arboreal 2D projection)
    base_df = compute_canopy_shrub_coverage(sites, scenarios, timesteps, voxel_size, project_only_arboreal=True)
    # STEP 1b: compute 2D/3D breakdown for arboreal and low-vegetation
    breakdown_df = compute_bioavailability_2d3d(sites, scenarios, timesteps, voxel_size)

    # STEP 2: start with just identifiers and years_since_prev scaffold
    # We will compute deltas/rates after deriving the total metric.
    scaffold_rows: List[Dict[str, Any]] = []
    for (site, scen), g in base_df.groupby(["site", "scenario"]):
        for year in sorted(g["timestep"].astype(int).tolist()):
            scaffold_rows.append({"site": site, "scenario": scen, "timestep": year})
    df_pred = pd.DataFrame.from_records(scaffold_rows)

    # STEP 3: merge breakdown columns
    if not breakdown_df.empty:
        df_pred = df_pred.merge(
            breakdown_df[["site", "scenario", "timestep", "arboreal_2d_m2", "arboreal_3d_m2", "low_veg_3d_m2"]],
            on=["site", "scenario", "timestep"],
            how="left",
        )
        # Derived total canopy+shrub metric (m2): arboreal 2D + low-vegetation 3D
        df_pred["total_canopy_shrub_m2"] = df_pred["arboreal_2d_m2"].fillna(0.0) + df_pred["low_veg_3d_m2"].fillna(0.0)

    # STEP 4: compute per-step increase and annual rate based on total_canopy_shrub_m2
    df_pred["canopy cover + shrubs (increase since last year)"] = 0.0
    df_pred["years_since_prev"] = 0
    df_pred["annual canopy cover + shrubs added (m2/year)"] = 0.0
    for (site, scen), g in df_pred.groupby(["site", "scenario"]):
        series = sorted(zip(g["timestep"].astype(int).tolist(), g["total_canopy_shrub_m2"].fillna(0.0).astype(float).tolist()))
        computed = _compute_annual_rate_from_steps(series)
        by_year = {d["timestep"]: d for d in computed}
        idx = (df_pred["site"] == site) & (df_pred["scenario"] == scen)
        for year in g["timestep"].astype(int).tolist():
            r = by_year.get(year, {"delta": 0.0, "years_since_prev": 0, "annual_rate": 0.0})
            df_pred.loc[idx & (df_pred["timestep"] == year), "canopy cover + shrubs (increase since last year)"] = float(r["delta"])
            df_pred.loc[idx & (df_pred["timestep"] == year), "years_since_prev"] = int(r["years_since_prev"])
            df_pred.loc[idx & (df_pred["timestep"] == year), "annual canopy cover + shrubs added (m2/year)"] = float(r["annual_rate"])

    # STEP 5: drop intermediate columns we no longer need from base_df
    # (area_selected_m2, delta, annual_rate are not included)
    # Ensure column order: identifiers, breakdowns, totals, increases, annual rate
    cols_order = [
        "site", "scenario", "timestep",
        "arboreal_2d_m2", "arboreal_3d_m2", "low_veg_3d_m2",
        "total_canopy_shrub_m2",
        "canopy cover + shrubs (increase since last year)",
        "years_since_prev",
        "annual canopy cover + shrubs added (m2/year)",
    ]
    existing = [c for c in cols_order if c in df_pred.columns]
    df_pred = df_pred[existing]
    return df_pred


# --------------------------------
# Manager: Canopy + Shrubs (1.1)
# --------------------------------
def canopy_shrub_manager(voxel_size: int = DEFAULT_VOXEL_SIZE) -> Dict[str, str]:
    """
    Build canopy/shrubs prediction, external and comparison CSVs.
    Returns dict with file paths: {'prediction', 'external', 'comparison'}.
    """
    _ensure_directory(OUTPUT_COMPARISON_DIR)
    pred_path = save_canopy_unified(out_name="canopy-shrubs_prediction.csv", voxel_size=voxel_size)
    ext_path = save_canopy_targets(out_name="canopy-shrubs_external.csv")

    # Load
    ours = pd.read_csv(pred_path) if os.path.exists(pred_path) else pd.DataFrame()
    ext = pd.read_csv(ext_path) if os.path.exists(ext_path) else pd.DataFrame()

    # Build comparison
    if not ours.empty and not ext.empty:
        ours2 = ours[["site", "scenario", "timestep", "annual canopy cover + shrubs added (m2/year)"]].copy()
        ours2 = ours2.rename(columns={"annual canopy cover + shrubs added (m2/year)": "annual_rate (m2/year)"})

        # External rows: site='living Melbourne', scenario=context, timestep=end_year; include baseline at first start_year
        rows = []
        for ctx, grp in ext.groupby("context"):
            first_start = int(grp["start_year"].min())
            rows.append({"site": "living Melbourne", "scenario": ctx, "timestep": first_start, "annual_rate (m2/year)": 0.0})
            for _, r in grp.iterrows():
                rows.append(
                    {
                        "site": "living Melbourne",
                        "scenario": r["context"],
                        "timestep": int(r["end_year"]),
                        "annual_rate (m2/year)": float(r["annual_rate_m2_per_year"]),
                    }
                )
        ext2 = pd.DataFrame(rows, columns=["site", "scenario", "timestep", "annual_rate (m2/year)"])
        cmp_df = pd.concat([ours2, ext2], ignore_index=True, sort=False)
    else:
        cmp_df = pd.DataFrame()

    cmp_path = os.path.join(OUTPUT_COMPARISON_DIR, "canopy-shrubs_comparison.csv")
    cmp_df.to_csv(cmp_path, index=False)
    return {"prediction": pred_path, "external": ext_path, "comparison": cmp_path}

# ---------------------------------------------------------------------------
# 1.2 Target for hollow bearing trees and artificial hollows (implemented)
# ---------------------------------------------------------------------------

def _load_resource_dic(resource_dic_path: str = RESOURCE_DIC_PATH) -> pd.DataFrame:
    """Load the resource dictionary used to infer hollow-bearing categories."""
    return pd.read_csv(resource_dic_path)


def _get_hollow_positive_combos(
    resource_dic_path: str = RESOURCE_DIC_PATH,
) -> Set[Tuple[bool, str, str]]:
    """
    Build the set of (precolonial, size, control) combinations considered hollow-bearing
    based on resource_dicDF.csv where hollow >= 1.
    """
    df = _load_resource_dic(resource_dic_path)
    df = df[df["hollow"] >= 1]
    combos: Set[Tuple[bool, str, str]] = set(
        zip(df["precolonial"].astype(bool), df["size"].astype(str), df["control"].astype(str))
    )
    return combos


def _load_pole_csv(site: str, scenario: str, voxel_size: int, year: int) -> pd.DataFrame:
    """
    Load the pole dataframe for a given site/scenario/year.
    Returns empty DataFrame if file doesn't exist (simplifies calling code).
    """
    base_path = f"data/revised/final/{site}"
    fp = f"{base_path}/{site}_{scenario}_{voxel_size}_poleDF_{year}.csv"
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp)

def _load_baseline_tree_csv(site: str, base_dir: str = "data/revised/final/baselines") -> pd.DataFrame:
    """
    Load baseline trees CSV for a site. Returns empty DataFrame if not present.
    Expected path: data/revised/final/baselines/{site}_baseline_trees.csv
    """
    fp = os.path.join(base_dir, f"{site}_baseline_trees.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp)


def _count_hollow_trees(tree_df: pd.DataFrame, combos: Set[Tuple[bool, str, str]]) -> int:
    """
    Count rows in the tree dataframe that match any hollow-bearing combination
    of (precolonial, size, control).
    """
    if tree_df.empty:
        return 0
    # Robust boolean parsing: strings like 'False' should be False (astype(bool) would incorrectly make them True)
    precolonial_bool = (
        tree_df["precolonial"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )
    keys = list(zip(precolonial_bool, tree_df["size"].astype(str), tree_df["control"].astype(str)))
    return sum((k in combos) for k in keys)


def compute_artificial_hollows(
    sites: List[str],
    scenarios: List[str],
    timesteps: List[int],
    voxel_size: int = DEFAULT_VOXEL_SIZE,
    resource_dic_path: str = RESOURCE_DIC_PATH,
) -> pd.DataFrame:
    """
    Compute hollow-bearing availability per timestep and context.
    - hollow_trees: count of trees matching hollow-bearing combos (from resource_dicDF.csv)
    - artificial_poles: count of rows in poleDF (proxy for artificial hollows)
    - total_hollow_bearing: hollow_trees + artificial_poles
    Returns a DataFrame spanning all (site, scenario, timestep).
    """
    combos = _get_hollow_positive_combos(resource_dic_path)
    records: List[Dict[str, Any]] = []

    for site in sites:
        for scenario in scenarios:
            for year in sorted(timesteps):
                tree_df = _load_tree_csv(site, scenario, voxel_size, year)
                pole_df = _load_pole_csv(site, scenario, voxel_size, year)

                hollow_trees = _count_hollow_trees(tree_df, combos)
                artificial_poles = len(pole_df) if not pole_df.empty else 0
                total = hollow_trees + artificial_poles

                records.append(
                    {
                        "site": site,
                        "scenario": scenario,
                        "timestep": year,
                        "voxel_size": voxel_size,
                        "hollow_trees": hollow_trees,
                        "artificial_poles": artificial_poles,
                        "total_hollow_bearing": total,
                    }
                )

    return pd.DataFrame.from_records(records)


def save_artificial_hollows_comparison(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = OUTPUT_ART_HOLLOWS_FILE,
    sites: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
    resource_dic_path: str = RESOURCE_DIC_PATH,
) -> str:
    """
    Compute and save the artificial hollows comparison CSV.
    - Ensures output directory exists.
    - Uses defaults for sites/scenarios/timesteps if not provided.
    Returns the output path written.
    """
    if sites is None:
        sites = DEFAULT_SITES
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS

    df = compute_artificial_hollows(
        sites=sites,
        scenarios=scenarios,
        timesteps=timesteps,
        voxel_size=voxel_size,
        resource_dic_path=resource_dic_path,
    )
    _ensure_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# 1.2b Hollow-bearing unified comparison (targets vs ours)
# ---------------------------------------------------------------------------

def save_hollows_targets(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "comparison_hollows_targets.csv",
    site_area_m2: float = 157_641.0,
) -> str:
    """
    Persist target table for hollow-bearing trees (Le Roux) with site conversion.
    Scenarios:
      - Le Roux predicted trend
      - Le Roux baseline woodland
      - Le Roux management plan  (legacy accelerate/increase scenario)
    Columns:
      scenario, year, hollow_bearing_per_ha, years_since_prev, annual_change_per_ha_per_year,
      site_area_m2, site_hectares, site_hollow_bearing_trees,
      site_change_from_prev, site_annual_rate_per_year, site_rate_per_year (alias)
    """
    _ensure_directory(out_dir)
    site_hectares = site_area_m2 / 10_000.0

    # Per-ha series
    series_pred_trend = [(0, 5.0), (10, 4.5), (20, 4.2), (30, 4.0), (60, 3.5), (180, 1.5), (300, 0.5)]
    series_baseline_woodland = [(0, 12.0), (10, 14.0), (20, 13.5), (30, 13.0), (60, 13.0), (180, 16.0), (300, 12.5)]
    # Management plan series from user-provided table
    series_management_plan = [(0, 5.5), (10, 6.0), (20, 6.5), (30, 7.0), (60, 7.5), (180, 5.0), (300, 5.5)]

    def bake(scenario: str, series: list[tuple[int, float]]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        prev_year: int | None = None
        prev_per_ha: float | None = None
        prev_site: float | None = None
        for year, per_ha in series:
            site_val = float(per_ha * site_hectares)
            if prev_year is None:
                years = 0
                delta_per_ha = 0.0
                delta_site = 0.0
            else:
                years = max(0, year - prev_year)
                delta_per_ha = per_ha - (prev_per_ha if prev_per_ha is not None else 0.0)
                delta_site = site_val - (prev_site if prev_site is not None else 0.0)
            rate_per_ha = (delta_per_ha / years) if years > 0 else 0.0
            site_rate = (delta_site / years) if years > 0 else 0.0
            out.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "hollow-bearing trees (structures/hectare)": per_ha,
                    "years since previous (years)": years,
                    "annual change (structures/hectare/year)": rate_per_ha,
                    "site area (m2)": site_area_m2,
                    "site area (hectares)": site_hectares,
                    "hollow-bearing trees on site (structures)": site_val,
                    "change from previous step (structures)": delta_site,
                    "annual rate (structures/year)": site_rate,
                }
            )
            prev_year = year
            prev_per_ha = per_ha
            prev_site = site_val
        return out

    rows: list[dict[str, object]] = []
    rows += bake("Le Roux predicted trend", series_pred_trend)
    rows += bake("Le Roux baseline woodland", series_baseline_woodland)
    rows += bake("Le Roux management plan", series_management_plan)

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


def save_hollows_unified(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "comparison_hollows_unified.csv",
    sites: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
    resource_dic_path: str = RESOURCE_DIC_PATH,
) -> str:
    """
    Compute detailed hollow-bearing metrics per timestep:
    - total_artificial_tree: count of artificial poles (rows in poleDF)
    - total_hollow_bearing_tree: count of trees matching hollow-bearing combos
    - artificial_added_this_timestep: delta vs previous timestep (artificial)
    - hollow_bearing_tree_added_this_timestep: delta vs previous timestep (trees)
    - total_hollow_bearing_structures_added_this_timestep: sum of the above two deltas
    - hollow_bearing_structure_annual_rate: total_added_this_timestep / years_since_prev

    Also includes aggregated fields for convenience:
    - total_hollow_bearing = total_artificial_tree + total_hollow_bearing_tree
    - years_since_prev
    - annual_rate (alias of hollow_bearing_structure_annual_rate for downstream join compatibility)
    """
    if sites is None:
        sites = DEFAULT_SITES
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS

    ours = compute_artificial_hollows(sites, scenarios, timesteps, voxel_size, resource_dic_path)
    combos = _get_hollow_positive_combos(resource_dic_path)
    records: List[Dict[str, Any]] = []
    # Add per-site baseline rows first (scenario='baseline', timestep=-1)
    for site in sites:
        bdf = _load_baseline_tree_csv(site)
        if not bdf.empty:
            # Use same combo mapping as for scenarios
            base_keys = list(
                zip(
                    bdf.get("precolonial", pd.Series([], dtype=bool)).astype(bool),
                    bdf.get("size", pd.Series([], dtype=str)).astype(str),
                    bdf.get("control", pd.Series([], dtype=str)).astype(str),
                )
            )
            base_hollowtrees = sum((k in combos) for k in base_keys)
        else:
            base_hollowtrees = 0
        records.append(
            {
                "site": site,
                "scenario": "baseline",
                "timestep": -1,
                "total_artificial_tree": 0,
                "total_hollow_bearing_tree": int(base_hollowtrees),
                "total_hollow_bearing": int(base_hollowtrees),
                "artificial_added_this_timestep": 0,
                "hollow_bearing_tree_added_this_timestep": 0,
                "total_hollow_bearing_structures_added_this_timestep": 0,
                "hollow_bearing_structure_annual_rate": 0.0,
                "years_since_prev": 0,
                "annual_rate": 0.0,
            }
        )
    for (site, scen), g in ours.groupby(["site", "scenario"]):
        # Sort by timestep for stable deltas
        g_sorted = g.sort_values(by="timestep").reset_index(drop=True)
        prev_artificial = None
        prev_hollowtrees = None
        prev_year = None
        for _, row in g_sorted.iterrows():
            year = int(row["timestep"])
            total_artificial = int(row["artificial_poles"])
            total_hollowtrees = int(row["hollow_trees"])

            if prev_year is None:
                # Manual adjustment: set total_hollow_bearing_tree to 0 at timestep 0
                # Rationale: normalize baseline so stepwise additions fully reflect gains post-baseline
                total_hollowtrees = 0
                years_since_prev = 0
                added_artificial = 0
                added_hollowtrees = 0
            else:
                years_since_prev = max(0, year - prev_year)
                added_artificial = total_artificial - (prev_artificial if prev_artificial is not None else 0)
                added_hollowtrees = total_hollowtrees - (prev_hollowtrees if prev_hollowtrees is not None else 0)

            total_added = added_artificial + added_hollowtrees
            annual_rate = (total_added / years_since_prev) if years_since_prev > 0 else 0.0

            total_hollow = total_artificial + total_hollowtrees
            records.append(
                {
                    "site": site,
                    "scenario": scen,
                    "timestep": year,
                    "total_artificial_tree": total_artificial,
                    "total_hollow_bearing_tree": total_hollowtrees,
                    "total_hollow_bearing": total_hollow,
                    "artificial_added_this_timestep": added_artificial,
                    "hollow_bearing_tree_added_this_timestep": added_hollowtrees,
                    "total_hollow_bearing_structures_added_this_timestep": total_added,
                    "hollow_bearing_structure_annual_rate": annual_rate,
                    "years_since_prev": years_since_prev,
                    # alias to keep downstream comparison code simple
                    "annual_rate": annual_rate,
                }
            )
            prev_artificial = total_artificial
            prev_hollowtrees = total_hollowtrees
            prev_year = year

    df = pd.DataFrame.from_records(records)
    _ensure_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path

# ---------------------------------------------------------------------------
# 1.4 Target for improving permeability (implemented)
# ---------------------------------------------------------------------------
#
# Our equivalent metric: 2D coverage for categories improving permeability:
# - search_bioavailable in {'open space', 'low-vegetation', 'arboreal'}
#

def compute_permeability_coverage(
    sites: List[str],
    scenarios: List[str],
    timesteps: List[int],
    voxel_size: int = DEFAULT_VOXEL_SIZE,
) -> pd.DataFrame:
    """Compute 2D coverage for permeability-related categories."""
    return compute_canopy_shrub_coverage(
        sites=sites,
        scenarios=scenarios,
        timesteps=timesteps,
        voxel_size=voxel_size,
        include_categories=("open space", "low-vegetation", "arboreal"),
        auto_generate_urban_features=True,
    )


def compute_permeability_3d(
    sites: List[str],
    scenarios: List[str],
    timesteps: List[int],
    voxel_size: int = DEFAULT_VOXEL_SIZE,
) -> pd.DataFrame:
    """
    Compute 3D stacked-area metrics (count * voxel_size^2) for permeability categories:
    - 'open space'
    - 'low-vegetation'
    Excludes 'arboreal' per updated requirement.
    Returns: site, scenario, timestep, open_space_3d_m2, low_veg_3d_m2, total_permeability_m2
    """
    import numpy as np
    rows: List[Dict[str, Any]] = []
    for site in sites:
        for scenario in scenarios:
            for year in sorted(timesteps):
                _ensure_urban_features(site, scenario, voxel_size, year)
                poly = _load_polydata(site, scenario, voxel_size, year)
                if poly is None or poly.n_points == 0 or "search_bioavailable" not in poly.point_data:
                    rows.append(
                        {
                            "site": site,
                            "scenario": scenario,
                            "timestep": year,
                            "open_space_3d_m2": 0.0,
                            "low_veg_3d_m2": 0.0,
                            "total_permeability_m2": 0.0,
                        }
                    )
                    continue
                vals = poly.point_data["search_bioavailable"]
                mask_open = vals == "open space"
                mask_low = vals == "low-vegetation"
                open_3d = float(np.count_nonzero(mask_open) * (voxel_size ** 2))
                low_3d = float(np.count_nonzero(mask_low) * (voxel_size ** 2))
                rows.append(
                    {
                        "site": site,
                        "scenario": scenario,
                        "timestep": year,
                        "open_space_3d_m2": open_3d,
                        "low_veg_3d_m2": low_3d,
                        "total_permeability_m2": open_3d + low_3d,
                    }
                )
    return pd.DataFrame.from_records(rows)

def save_permeability_comparison(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "comparison_permeability.csv",
    sites: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
) -> str:
    """Compute and save permeability comparison CSV."""
    if sites is None:
        sites = DEFAULT_SITES
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS

    df = compute_permeability_coverage(
        sites=sites,
        scenarios=scenarios,
        timesteps=timesteps,
        voxel_size=voxel_size,
    )
    _ensure_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# 1.4b Permeability target tables and unified comparison
# ---------------------------------------------------------------------------

def save_permeability_targets(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "comparison_permeability_targets.csv",
    site_area_m2: float = 157_641.0,
) -> str:
    """
    Persist permeability targets:
    - Parking conversion (given totals and site equivalents)
    - Elizabeth St unsealed soil (given totals and site equivalents)
    """
    _ensure_directory(out_dir)
    rows = [
        {
            "context": "parking_conversion",
            "scenario": "convert 47% redundant on-street parking",
            "target_year": 2035,
            "total_depaved_m2": 245_000.0,
            "timeframe_years": 10,
            "annual_rate_m2_per_year": 24_500.0,
            "site_equivalent_total_m2": 1_025.0,
            "site_equivalent_annual_m2_per_year": 103.0,
        },
        {
            "context": "elizabeth_catchment",
            "scenario": "unsealed soil target",
            "baseline_year": 2014,
            "baseline_pct": 17.0,
            "target_year": 2030,
            "target_pct": 40.0,
            "change_pct_points": 23.0,
            "timeframe_years": 16,
            "site_equivalent_total_m2": 36_257.0,
            "site_equivalent_annual_m2_per_year": 2_266.0,
        },
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


def save_permeability_unified(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "comparison_permeability_unified.csv",
    sites: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
) -> str:
    """
    Compute our per-step annualized change in permeability (3D) and save unified CSV.
    Includes:
    - open_space_3d_m2
    - low_veg_3d_m2
    - total_permeability_m2
    - permeability (increase since last year)
    - years_since_prev
    - annual permeability added (m2/year)
    """
    if sites is None:
        sites = DEFAULT_SITES
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS

    base = compute_permeability_3d(sites, scenarios, timesteps, voxel_size)
    # Compute deltas and annual rates for the combined 3D metric
    base["permeability (increase since last year)"] = 0.0
    base["years_since_prev"] = 0
    base["annual permeability added (m2/year)"] = 0.0
    for (site, scen), g in base.groupby(["site", "scenario"]):
        series = sorted(zip(g["timestep"].astype(int).tolist(), g["total_permeability_m2"].astype(float).tolist()))
        computed = _compute_annual_rate_from_steps(series)
        by_year = {d["timestep"]: d for d in computed}
        idx = (base["site"] == site) & (base["scenario"] == scen)
        for year in g["timestep"].astype(int).tolist():
            r = by_year.get(year, {"delta": 0.0, "years_since_prev": 0, "annual_rate": 0.0})
            base.loc[idx & (base["timestep"] == year), "permeability (increase since last year)"] = float(r["delta"])
            base.loc[idx & (base["timestep"] == year), "years_since_prev"] = int(r["years_since_prev"])
            base.loc[idx & (base["timestep"] == year), "annual permeability added (m2/year)"] = float(r["annual_rate"])

    out_df = base[
        [
            "site",
            "scenario",
            "timestep",
            "open_space_3d_m2",
            "low_veg_3d_m2",
            "total_permeability_m2",
            "permeability (increase since last year)",
            "years_since_prev",
            "annual permeability added (m2/year)",
        ]
    ].copy()
    _ensure_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    out_df.to_csv(out_path, index=False)
    return out_path

# ---------------------------------------------------------------------------
# 1.3 Target for removing large, old trees (implemented)
# ---------------------------------------------------------------------------

def _load_tree_csv(site: str, scenario: str, voxel_size: int, year: int) -> pd.DataFrame:
    """
    Load the tree dataframe for a given site/scenario/year.
    Returns empty DataFrame if file doesn't exist (simplifies calling code).
    """
    base_path = f"data/revised/final/{site}"
    fp = f"{base_path}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv"
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp)


def _compute_replacement_row(
    site: str,
    scenario: str,
    year: int,
    df: pd.DataFrame,
    cohort_sizes: List[str],
) -> Dict[str, Any]:
    total_replaced = int((df["action"] == "REPLACE").sum()) if "action" in df.columns else 0
    cohort_mask = df["size"].isin(cohort_sizes) if "size" in df.columns else pd.Series([], dtype=bool)
    cohort_count = int(cohort_mask.sum()) if not df.empty else 0
    row = {
        "site": site,
        "scenario": scenario,
        "timestep": year,
        "totalReplaced": total_replaced,
        "cohortCount": cohort_count,
    }
    return row


def compute_removal_stats(
    sites: List[str],
    scenarios: List[str],
    timesteps: List[int],
    voxel_size: int = 1,
) -> pd.DataFrame:
    """
    Compute removal metrics across timesteps for each site/scenario.
    - totalReplaced: cumulative count where action == 'REPLACE'
    - replacedThisStep: delta of totalReplaced relative to previous timestep
    - yearsSincePrev: years passed since previous timestep
    - annualReplacementRate: replacedThisStep / yearsSincePrev
    - cohortCount: count where size in ['large','senescing','snag','fallen']
    - replacementPctOfCohort: (annualReplacementRate / cohortCount) * 100
    - age-threshold (elm proxy) removals using temp_yearspassed >= 139:
        cumulativeAgedOut, agedOutThisStep, annualAgedOutRate, agedOutPctOfCohort
      Note: 139-year threshold corresponds to elm lifespan; currently applied across all trees.
    - tolerance threshold (Croeser) per step: toleranceTreesPerYear = cohortCount * 0.01; tolerancePctOfCohort = 1.0
    - combinedTreesPerYear/combinedPctOfCohort = max(tolerance vs age-threshold) per step
    """
    cohort_sizes = COHORT_SIZES
    records: List[Dict[str, Any]] = []

    for site in sites:
        for scenario in scenarios:
            prev_total = None
            prev_year = None
            prev_agedout_cum = None
            for year in sorted(timesteps):
                df = _load_tree_csv(site, scenario, voxel_size, year)
                row = _compute_replacement_row(site, scenario, year, df, cohort_sizes)

                # First timestep: define baseline as the cumulative at that step
                if prev_total is None:
                    replaced_this_step = row["totalReplaced"]
                    years_since_prev = 0
                else:
                    # Subsequent timesteps: compute deltas vs previous cumulative
                    replaced_this_step = row["totalReplaced"] - prev_total
                    years_since_prev = max(0, year - prev_year) if prev_year is not None else 0

                # Protect against divide-by-zero at the baseline timestep
                annual_rate = (replaced_this_step / years_since_prev) if years_since_prev > 0 else 0.0
                cohort = row["cohortCount"]
                pct_of_cohort = (annual_rate / cohort) * 100 if cohort > 0 else 0.0

                # Age-threshold removals (elm proxy, threshold 139 years)
                # We use 'temp_yearspassed' as age proxy if available, else 0.
                aged_series = df.get("temp_yearspassed", pd.Series([], dtype=float))
                try:
                    aged_vals = pd.to_numeric(aged_series, errors="coerce").fillna(0.0)
                except Exception:
                    aged_vals = pd.Series([0.0] * len(df))
                cumulative_agedout = int((aged_vals >= 139.0).sum())
                if prev_agedout_cum is None:
                    aged_out_this_step = cumulative_agedout
                else:
                    aged_out_this_step = cumulative_agedout - prev_agedout_cum
                annual_aged_out_rate = (aged_out_this_step / years_since_prev) if years_since_prev > 0 else 0.0
                aged_out_pct_of_cohort = (annual_aged_out_rate / cohort) * 100 if cohort > 0 else 0.0

                # Tolerance threshold (Croeser): 1% per annum of cohort
                tolerance_trees_per_year = float(cohort) * 0.01
                tolerance_pct_of_cohort = 1.0 if cohort > 0 else 0.0

                # Combined: whichever is higher between tolerance and age-threshold rate
                combined_trees_per_year = max(tolerance_trees_per_year, annual_aged_out_rate)
                combined_pct_of_cohort = max(tolerance_pct_of_cohort, aged_out_pct_of_cohort)

                row["replacedThisStep"] = int(replaced_this_step)
                row["yearsSincePrev"] = int(years_since_prev)
                row["annualReplacementRate"] = float(annual_rate)
                row["replacementPctOfCohort"] = float(pct_of_cohort)
                row["cumulativeAgedOut"] = int(cumulative_agedout)
                row["agedOutThisStep"] = int(aged_out_this_step)
                row["annualAgedOutRate"] = float(annual_aged_out_rate)
                row["agedOutPctOfCohort"] = float(aged_out_pct_of_cohort)
                row["toleranceTreesPerYear"] = float(tolerance_trees_per_year)
                row["tolerancePctOfCohort"] = float(tolerance_pct_of_cohort)
                row["combinedTreesPerYear"] = float(combined_trees_per_year)
                row["combinedPctOfCohort"] = float(combined_pct_of_cohort)
                records.append(row)

                # Update baselines for next timestep
                prev_total = row["totalReplaced"]
                prev_year = year
                prev_agedout_cum = cumulative_agedout

    return pd.DataFrame.from_records(records)


def save_removal_unified_targets(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "removal_external.csv",
) -> str:
    """
    Save external targets for reducing removals of large, old trees (Croesser 2025):
    - tolerance_threshold: 1% per annum (percent of cohort per year)
    Outputs columns: site, scenario, rate_per_year
    """
    _ensure_directory(out_dir)
    df = pd.DataFrame([{"site": "croesser 2025", "scenario": "tolerance_threshold", "rate_per_year": 1.0}])
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


def save_removal_unified(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "removal_prediction.csv",
    sites: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
) -> str:
    """
    Save our removal stats per timestep with percent-of-cohort rate for comparison to targets.
    Columns include: site, scenario, timestep,
      - annualReplacementRate, replacementPctOfCohort
      - cumulativeAgedOut, agedOutThisStep, annualAgedOutRate, agedOutPctOfCohort
      - toleranceTreesPerYear, tolerancePctOfCohort
      - combinedTreesPerYear, combinedPctOfCohort
    """
    if sites is None:
        sites = DEFAULT_SITES
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS
    df = compute_removal_stats(sites, scenarios, timesteps, voxel_size)
    _ensure_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# 1.4 Target for improving permeability (skeleton)
# ---------------------------------------------------------------------------

def compute_permeability_targets() -> None:
    """Placeholder for permeability target computations (1.4)."""
    return None


# ---------------------------------------------------------------------------
# 1.5 Target for improving connectivity (skeleton)
# ---------------------------------------------------------------------------

def compute_connectivity_targets() -> None:
    """Deprecated placeholder; use compute_capabilities_coverage/save_capabilities_comparison instead."""
    return None


# ---------------------------------------------------------------------------
# 1.5 Target for improving connectivity (implemented)
# ---------------------------------------------------------------------------
#
# Our equivalent metric:
# - For birds: presence where point_data['capabilities_bird'] != 'none'
# - For reptiles: presence where point_data['capabilities_reptile'] != 'none'
# Use 2D collapse (unique XY columns), area in m².
#
def compute_capabilities_coverage(
    sites: List[str],
    scenarios: List[str],
    timesteps: List[int],
    voxel_size: int = DEFAULT_VOXEL_SIZE,
) -> pd.DataFrame:
    """Compute 2D coverage for bird and reptile aggregate capabilities."""
    import numpy as np

    records: List[Dict[str, Any]] = []
    for site in sites:
        for scenario in scenarios:
            for year in sorted(timesteps):
                # Capabilities are added in info_generate_capabilitiesVTKs step; prefer *_with_capabilities.vtk
                poly = _load_capabilities_polydata(site, scenario, voxel_size, year)
                if poly is None or poly.n_points == 0:
                    records.append(
                        {
                            "site": site,
                            "scenario": scenario,
                            "timestep": year,
                            "voxel_size": voxel_size,
                            "area_total_m2": 0.0,
                            "bird_area_m2": 0.0,
                            "reptile_area_m2": 0.0,
                            "bird_coverage_pct": 0.0,
                            "reptile_coverage_pct": 0.0,
                        }
                    )
                    continue

                # Build base XY columns universe
                all_xy = _unique_xy_keys(poly.points, voxel_size)
                all_xy_unique = np.unique(all_xy, axis=0)
                area_total_m2 = float(len(all_xy_unique) * (voxel_size ** 2))

                # Birds
                if "capabilities_bird" in poly.point_data:
                    bird_vals = poly.point_data["capabilities_bird"]
                    bird_mask = (bird_vals != 'none')
                    bird_xy_unique = np.unique(all_xy[bird_mask], axis=0) if np.any(bird_mask) else np.empty((0, 2), dtype=int)
                    bird_area_m2 = float(len(bird_xy_unique) * (voxel_size ** 2))
                else:
                    bird_area_m2 = 0.0

                # Reptiles
                if "capabilities_reptile" in poly.point_data:
                    rept_vals = poly.point_data["capabilities_reptile"]
                    rept_mask = (rept_vals != 'none')
                    rept_xy_unique = np.unique(all_xy[rept_mask], axis=0) if np.any(rept_mask) else np.empty((0, 2), dtype=int)
                    reptile_area_m2 = float(len(rept_xy_unique) * (voxel_size ** 2))
                else:
                    reptile_area_m2 = 0.0

                records.append(
                    {
                        "site": site,
                        "scenario": scenario,
                        "timestep": year,
                        "voxel_size": voxel_size,
                        "area_total_m2": area_total_m2,
                        "bird_area_m2": bird_area_m2,
                        "reptile_area_m2": reptile_area_m2,
                        "bird_coverage_pct": (bird_area_m2 / area_total_m2 * 100.0) if area_total_m2 > 0 else 0.0,
                        "reptile_coverage_pct": (reptile_area_m2 / area_total_m2 * 100.0) if area_total_m2 > 0 else 0.0,
                    }
                )

    return pd.DataFrame.from_records(records)

def compute_connectivity_advanced(
    sites: List[str],
    scenarios: List[str],
    timesteps: List[int],
    voxel_size: int = DEFAULT_VOXEL_SIZE,
    bird_radius_m: float = 500.0,
    bird_baseline_radius_m: float = 1000.0,
) -> pd.DataFrame:
    """
    Advanced connectivity:
    - Reptiles: mask where capabilities_reptile indicates presence; split by search_bioavailable:
        * reptile_capabilities_arboreal2d_m2: where search_bioavailable == 'arboreal' (2D unique XY)
        * reptile_capabilities_3d_m2: where search_bioavailable != 'arboreal' (3D stacked)
    - Birds: 2D unique XY where capabilities_bird present AND within `bird_radius_m` of any voxel
      with capabilities_bird_raise-young_hollow == 1; distance computed in XY using cKDTree.
      Also compute extra_from_Year_0_m2 = (coverage at baseline with `bird_baseline_radius_m`) - (coverage at baseline with `bird_radius_m`).
    Returns a DataFrame with per (site, scenario, timestep):
      reptile_capabilities_arboreal2d_m2, reptile_capabilities_3d_m2,
      bird_capabilities_within_500m_hollows_m2, extra_from_Year_0_m2,
      plus combined bird/reptile totals and per-step deltas/rates.
    """
    import numpy as np
    from scipy.spatial import cKDTree  # type: ignore

    rows: List[Dict[str, Any]] = []
    # Precompute baseline extras per site/scenario
    baseline_extra: Dict[Tuple[str, str], float] = {}

    for site in sites:
        for scenario in scenarios:
            # Determine baseline t0 extra using timestep 0 if present, else min timestep
            tsteps = sorted(timesteps)
            t0 = 0 if 0 in tsteps else (tsteps[0] if tsteps else 0)
            # Compute once per site/scenario
            poly0 = _load_capabilities_polydata(site, scenario, voxel_size, t0)
            if poly0 is None or poly0.n_points == 0:
                baseline_extra[(site, scenario)] = 0.0
            else:
                # Bird present XY
                all_xy0 = _unique_xy_keys(poly0.points, voxel_size)
                if "capabilities_bird" in poly0.point_data and "capabilities_bird_raise-young_hollow" in poly0.point_data:
                    bird_vals0 = poly0.point_data["capabilities_bird"]
                    bird_mask0 = (bird_vals0 != 'none')
                    bird_xy0 = np.unique(all_xy0[bird_mask0], axis=0) if np.any(bird_mask0) else np.empty((0, 2), dtype=int)
                    hollows0 = poly0.point_data["capabilities_bird_raise-young_hollow"]
                    holl_xy0 = np.unique(all_xy0[hollows0 == 1], axis=0) if np.any(hollows0 == 1) else np.empty((0, 2), dtype=int)
                    if bird_xy0.size == 0 or holl_xy0.size == 0:
                        baseline_extra[(site, scenario)] = 0.0
                    else:
                        tree0 = cKDTree(holl_xy0.astype(float))
                        dists500, _ = tree0.query(bird_xy0.astype(float), k=1, distance_upper_bound=float(bird_radius_m / float(voxel_size)))
                        dists1000, _ = tree0.query(bird_xy0.astype(float), k=1, distance_upper_bound=float(bird_baseline_radius_m / float(voxel_size)))
                        sel500 = np.isfinite(dists500)
                        sel1000 = np.isfinite(dists1000)
                        bird500_m2 = float(sel500.sum() * (voxel_size ** 2))
                        bird1000_m2 = float(sel1000.sum() * (voxel_size ** 2))
                        baseline_extra[(site, scenario)] = max(0.0, bird1000_m2 - bird500_m2)
                else:
                    baseline_extra[(site, scenario)] = 0.0

            for year in tsteps:
                poly = _load_capabilities_polydata(site, scenario, voxel_size, year)
                if poly is None or poly.n_points == 0:
                    rows.append(
                        {
                            "site": site,
                            "scenario": scenario,
                            "timestep": year,
                            "reptile_capabilities_arboreal2d_m2": 0.0,
                            "reptile_capabilities_3d_m2": 0.0,
                            "bird_capabilities_within_500m_hollows_m2": 0.0,
                            "extra_from_Year_0_m2": baseline_extra.get((site, scenario), 0.0),
                        }
                    )
                    continue

                vals_rept = poly.point_data.get("capabilities_reptile", None)
                vals_bird = poly.point_data.get("capabilities_bird", None)
                holl_bird = poly.point_data.get("capabilities_bird_raise-young_hollow", None)
                search = poly.point_data.get("search_bioavailable", None)

                all_xy = _unique_xy_keys(poly.points, voxel_size)

                # Reptiles
                rept_arb_2d = 0.0
                rept_nonarb_3d = 0.0
                if vals_rept is not None and search is not None:
                    # Presence mask: treat non-'none' or >0 as present
                    try:
                        present_mask = vals_rept != 'none'
                    except Exception:
                        present_mask = vals_rept.astype(float) > 0
                    arb_mask = (search == "arboreal") & present_mask
                    nonarb_mask = (search != "arboreal") & present_mask
                    if np.any(arb_mask):
                        arb_xy = np.unique(all_xy[arb_mask], axis=0)
                        rept_arb_2d = float(len(arb_xy) * (voxel_size ** 2))
                    if np.any(nonarb_mask):
                        rept_nonarb_3d = float(np.count_nonzero(nonarb_mask) * (voxel_size ** 2))

                # Birds within radius of hollows (2D)
                bird500 = 0.0
                if vals_bird is not None and holl_bird is not None:
                    try:
                        bird_present = vals_bird != 'none'
                    except Exception:
                        bird_present = vals_bird.astype(float) > 0
                    bird_xy = np.unique(all_xy[bird_present], axis=0) if np.any(bird_present) else np.empty((0, 2), dtype=int)
                    holl_xy = np.unique(all_xy[holl_bird == 1], axis=0) if np.any(holl_bird == 1) else np.empty((0, 2), dtype=int)
                    if bird_xy.size > 0 and holl_xy.size > 0:
                        tree = cKDTree(holl_xy.astype(float))
                        dists, _ = tree.query(bird_xy.astype(float), k=1, distance_upper_bound=float(bird_radius_m / float(voxel_size)))
                        sel = np.isfinite(dists)
                        bird500 = float(sel.sum() * (voxel_size ** 2))

                rows.append(
                    {
                        "site": site,
                        "scenario": scenario,
                        "timestep": year,
                        "reptile_capabilities_arboreal2d_m2": rept_arb_2d,
                        "reptile_capabilities_3d_m2": rept_nonarb_3d,
                        "bird_capabilities_within_500m_hollows_m2": bird500,
                        "extra_from_Year_0_m2": baseline_extra.get((site, scenario), 0.0),
                    }
                )

    df = pd.DataFrame.from_records(rows)
    # Derived totals and annualized metrics
    if not df.empty:
        df["reptile_capabilities_m2"] = df["reptile_capabilities_arboreal2d_m2"].fillna(0.0) + df["reptile_capabilities_3d_m2"].fillna(0.0)
        df["bird_capabilities_m2"] = df["bird_capabilities_within_500m_hollows_m2"].fillna(0.0)
        df["combined_capabilities_m2"] = df["reptile_capabilities_m2"] + df["bird_capabilities_m2"]
        # Initialize deltas/rates
        df["reptile_increase_since_last_year"] = 0.0
        df["bird_increase_since_last_year"] = 0.0
        df["combined_increase_since_last_year"] = 0.0
        df["years_since_prev"] = 0
        df["reptile_added_per_year_m2"] = 0.0
        df["bird_added_per_year_m2"] = 0.0
        df["combined_added_per_year_m2"] = 0.0
        for (site, scen), g in df.groupby(["site", "scenario"]):
            # Reptile
            seq_r = sorted(zip(g["timestep"].astype(int).tolist(), g["reptile_capabilities_m2"].astype(float).tolist()))
            seq_b = sorted(zip(g["timestep"].astype(int).tolist(), g["bird_capabilities_m2"].astype(float).tolist()))
            seq_c = sorted(zip(g["timestep"].astype(int).tolist(), g["combined_capabilities_m2"].astype(float).tolist()))
            rR = {d["timestep"]: d for d in _compute_annual_rate_from_steps(seq_r)}
            rB = {d["timestep"]: d for d in _compute_annual_rate_from_steps(seq_b)}
            rC = {d["timestep"]: d for d in _compute_annual_rate_from_steps(seq_c)}
            mask = (df["site"] == site) & (df["scenario"] == scen)
            for year in g["timestep"].astype(int).tolist():
                df.loc[mask & (df["timestep"] == year), "reptile_increase_since_last_year"] = float(rR[year]["delta"])
                df.loc[mask & (df["timestep"] == year), "bird_increase_since_last_year"] = float(rB[year]["delta"])
                df.loc[mask & (df["timestep"] == year), "combined_increase_since_last_year"] = float(rC[year]["delta"])
                df.loc[mask & (df["timestep"] == year), "years_since_prev"] = int(rC[year]["years_since_prev"])
                df.loc[mask & (df["timestep"] == year), "reptile_added_per_year_m2"] = float(rR[year]["annual_rate"])
                df.loc[mask & (df["timestep"] == year), "bird_added_per_year_m2"] = float(rB[year]["annual_rate"])
                df.loc[mask & (df["timestep"] == year), "combined_added_per_year_m2"] = float(rC[year]["annual_rate"])
    return df

def save_capabilities_comparison(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "comparison_capabilities.csv",
    sites: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
) -> str:
    """Compute and save bird/reptile connectivity coverage CSV."""
    if sites is None:
        sites = DEFAULT_SITES
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS

    df = compute_capabilities_coverage(
        sites=sites,
        scenarios=scenarios,
        timesteps=timesteps,
        voxel_size=voxel_size,
    )
    _ensure_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# 1.6 Linking Melbourne targets (skeleton)
# ---------------------------------------------------------------------------

def compute_linking_melbourne_targets() -> None:
    """Deprecated placeholder; use save_linking_melbourne_comparison instead."""
    return None


# ---------------------------------------------------------------------------
# 1.6 Linking Melbourne targets (implemented)
# ---------------------------------------------------------------------------
#
# We convert region-wide area changes to site-equivalents using a proportional
# scaling factor based on areas:
# - Region area (Kirk study): 3,770 ha = 37,700,000 m²
# - Our site: 157,641 m²
# - Scaling factor = 157,641 / 37,700,000 ≈ 0.00418
# Time period considered: 20 years
#
# Inputs (from spec):
# - Birds: +154.9 ha over 20 years
# - Reptiles: +11.3 ha over 20 years
#

def save_linking_melbourne_comparison(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "comparison_linking-melbourne.csv",
    region_area_m2: float = 37_700_000.0,
    site_area_m2: float = 157_641.0,
    period_years: int = 20,
    birds_increase_ha: float = 154.9,
    reptiles_increase_ha: float = 11.3,
) -> str:
    """
    Compute site-equivalent area increases and annual rates for Linking Melbourne targets.
    - Returns and saves a small CSV with target class, site-equivalent total change (m²),
      and annual rate (m²/year).
    """
    _ensure_directory(out_dir)
    scaling = site_area_m2 / region_area_m2

    # Convert ha to m²
    birds_region_m2 = birds_increase_ha * 10_000.0
    reptiles_region_m2 = reptiles_increase_ha * 10_000.0

    birds_site_total_m2 = birds_region_m2 * scaling
    reptiles_site_total_m2 = reptiles_region_m2 * scaling

    birds_site_annual_m2 = birds_site_total_m2 / float(period_years)
    reptiles_site_annual_m2 = reptiles_site_total_m2 / float(period_years)

    df = pd.DataFrame(
        [
            {
                "target": "tree-hollow birds",
                "site_total_change_m2": birds_site_total_m2,
                "annual_rate_m2_per_year": birds_site_annual_m2,
                "period_years": period_years,
                "scaling_factor": scaling,
            },
            {
                "target": "reptiles",
                "site_total_change_m2": reptiles_site_total_m2,
                "annual_rate_m2_per_year": reptiles_site_annual_m2,
                "period_years": period_years,
                "scaling_factor": scaling,
            },
        ]
    )
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Connectivity (1.5) prediction/external/comparison helpers
# ---------------------------------------------------------------------------

def save_connectivity_prediction(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "connectivity_prediction.csv",
    sites: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
) -> str:
    """
    Advanced connectivity metrics and annualized changes:
    - reptile_capabilities_arboreal2d_m2
    - reptile_capabilities_3d_m2
    - reptile_capabilities_m2, reptile_increase_since_last_year, reptile_added_per_year_m2
    - bird_capabilities_within_500m_hollows_m2
    - extra_from_Year_0_m2
    - bird_capabilities_m2, bird_increase_since_last_year, bird_added_per_year_m2
    - combined_capabilities_m2, combined_increase_since_last_year, combined_added_per_year_m2
    """
    if sites is None:
        sites = DEFAULT_SITES
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS

    df = compute_connectivity_advanced(sites, scenarios, timesteps, voxel_size)
    _ensure_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


def save_connectivity_external(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "connectivity_external.csv",
    lm_path: Optional[str] = None,
) -> str:
    """
    External connectivity targets (from Linking Melbourne), normalized for our site.
    Columns: persona, site_total_change_m2, annual_rate_m2_per_year, period_years
    """
    _ensure_directory(out_dir)
    if lm_path is None:
        lm_path = save_linking_melbourne_comparison()
    lm = pd.read_csv(lm_path)
    # Map target to persona
    persona_map = {"tree-hollow birds": "bird", "reptiles": "reptile"}
    lm["persona"] = lm["target"].map(persona_map)
    df = lm[["persona", "site_total_change_m2", "annual_rate_m2_per_year", "period_years"]].copy()
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


def save_connectivity_comparison(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "connectivity_comparison.csv",
    prediction_path: Optional[str] = None,
    external_path: Optional[str] = None,
) -> str:
    """
    Merge ours (per-year rates per timestep) with external (per-year over period) in a single table.
    Columns include: source, persona, rate_per_year, plus identifying columns.
    """
    _ensure_directory(out_dir)
    if prediction_path is None:
        prediction_path = save_connectivity_prediction()
    if external_path is None:
        external_path = save_connectivity_external()

    ours = pd.read_csv(prediction_path)
    ours["source"] = "ours"
    ours = ours[["source", "persona", "site", "scenario", "timestep", "rate_per_year"]]

    ext = pd.read_csv(external_path)
    ext["source"] = "external"
    # Align column names
    ext = ext.rename(columns={"annual_rate_m2_per_year": "rate_per_year"})
    ext["site"] = ""
    ext["scenario"] = ""
    ext["timestep"] = ""
    ext = ext[["source", "persona", "site", "scenario", "timestep", "rate_per_year"]]

    df = pd.concat([ours, ext], ignore_index=True)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Example usage (no saving)
# ---------------------------------------------------------------------------

def example_removal_stats() -> pd.DataFrame:
    """Convenience wrapper to compute removal stats over default contexts."""
    sites = DEFAULT_SITES
    scenarios = DEFAULT_SCENARIOS
    timesteps = DEFAULT_TIMESTEPS
    return compute_removal_stats(sites, scenarios, timesteps, voxel_size=1)

#
# ---------------------------------------------------------------------------
# 1.1 Target for tree canopy cover and shrubs (implemented)
# ---------------------------------------------------------------------------
#
# Our equivalent metric:
# - Treat coverage as any voxel with search_bioavailable in {'low-vegetation','arboreal'}
# - Collapse 3D to 2D by grouping voxels that share the same (x, y) column (any z)
# - Area in m² = number of unique (x,y) columns in group * (voxel_size ** 2)
#
try:
    import pyvista as pv
except Exception:
    pv = None


def _scenario_vtk_path(site: str, scenario: str, voxel_size: int, year: int) -> str:
    """Build the path to the scenario polydata VTK."""
    base_path = f"data/revised/final/{site}"
    return f"{base_path}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk"


def _load_polydata(site: str, scenario: str, voxel_size: int, year: int) -> pd.DataFrame | None:
    """
    Load scenario polydata with PyVista. Returns None if not available or pyvista missing.
    This polydata should include 'search_bioavailable' in point_data.
    """
    if pv is None:
        return None
    raw_fp = _scenario_vtk_path(site, scenario, voxel_size, year)
    # Prefer the urban-features-augmented file if present (contains search_bioavailable)
    urban_fp = raw_fp.replace(".vtk", "_urban_features.vtk")
    fp = urban_fp if os.path.exists(urban_fp) else raw_fp
    if not os.path.exists(fp):
        return None
    try:
        return pv.read(fp)
    except Exception:
        return None


def _ensure_urban_features(
    site: str,
    scenario: str,
    voxel_size: int,
    year: int,
) -> None:
    """
    Ensure the *_urban_features.vtk exists by invoking the urban elements processor if available.
    Safe no-op if the processor cannot be imported or files are already present.
    """
    try:
        import final.a_scenario_urban_elements_count as a_scenario_urban_elements_count  # type: ignore
    except Exception:
        return  # Processor not available; leave as is

    raw_fp = _scenario_vtk_path(site, scenario, voxel_size, year)
    if not os.path.exists(raw_fp):
        return

    urban_fp = raw_fp.replace(".vtk", "_urban_features.vtk")
    if os.path.exists(urban_fp):
        return

    try:
        # Run only on this specific file to minimize work
        a_scenario_urban_elements_count.run_from_manager(
            site=site,
            voxel_size=voxel_size,
            specific_files=[raw_fp],
            should_process_baseline=False,
            enable_visualization=False
        )
    except Exception:
        # If processing fails, leave it; the caller will handle missing data
        return


def _load_capabilities_polydata(site: str, scenario: str, voxel_size: int, year: int):
    """
    Load the polydata prioritizing the *_with_capabilities.vtk output, then urban_features, then raw.
    Returns None if not found or load fails.
    """
    if pv is None:
        return None
    base = _scenario_vtk_path(site, scenario, voxel_size, year)
    caps = base.replace(".vtk", "_with_capabilities.vtk")
    if os.path.exists(caps):
        try:
            return pv.read(caps)
        except Exception:
            pass
    # fallback to urban_features/raw
    return _load_polydata(site, scenario, voxel_size, year)


def _unique_xy_keys(points: "np.ndarray", voxel_size: int) -> "np.ndarray":
    """
    Discretize XY to voxel grid keys. We round to nearest grid step to group columns.
    Returns an array of shape (N, 2) of integer keys.
    """
    import numpy as np
    xy = points[:, :2]
    # Convert to grid indices by dividing by voxel size and rounding to nearest int
    grid_idx = np.rint(xy / float(voxel_size)).astype(int)
    return grid_idx


def compute_canopy_shrub_coverage(
    sites: List[str],
    scenarios: List[str],
    timesteps: List[int],
    voxel_size: int = DEFAULT_VOXEL_SIZE,
    include_categories: Tuple[str, str] = ("low-vegetation", "arboreal"),
    auto_generate_urban_features: bool = True,
    project_only_arboreal: bool = False,
) -> pd.DataFrame:
    """
    Compute 2D coverage by collapsing along Z.
    - If project_only_arboreal is True, only 'arboreal' is projected into 2D.
    - Otherwise, all values in include_categories are projected together.
    Returns a DataFrame with area and percent coverage per (site, scenario, timestep).
    """
    import numpy as np

    records: List[Dict[str, Any]] = []
    for site in sites:
        for scenario in scenarios:
            for year in sorted(timesteps):
                # Attempt to create *_urban_features.vtk on-the-fly if missing
                if auto_generate_urban_features:
                    _ensure_urban_features(site, scenario, voxel_size, year)

                poly = _load_polydata(site, scenario, voxel_size, year)
                if poly is None or poly.n_points == 0 or "search_bioavailable" not in poly.point_data:
                    # No data: record zeros to keep a consistent table
                    records.append(
                        {
                            "site": site,
                            "scenario": scenario,
                            "timestep": year,
                            "voxel_size": voxel_size,
                            "area_total_m2": 0.0,
                            "area_selected_m2": 0.0,
                            "coverage_pct": 0.0,
                            "selected_categories": ",".join(include_categories),
                        }
                    )
                    continue

                # Build XY keys for all points (denominator)
                all_xy = _unique_xy_keys(poly.points, voxel_size)
                # Unique columns in the dataset
                all_xy_unique = np.unique(all_xy, axis=0)
                area_total_m2 = float(len(all_xy_unique) * (voxel_size ** 2))

                # Filter by categories (numerator)
                bioavail = poly.point_data["search_bioavailable"]
                if project_only_arboreal:
                    mask_arb = bioavail == "arboreal"
                    if np.any(mask_arb):
                        sel_xy_unique = np.unique(all_xy[mask_arb], axis=0)
                        area_selected_m2 = float(len(sel_xy_unique) * (voxel_size ** 2))
                    else:
                        area_selected_m2 = 0.0
                else:
                    mask = np.isin(bioavail, list(include_categories))
                    if np.any(mask):
                        sel_xy_unique = np.unique(all_xy[mask], axis=0)
                        area_selected_m2 = float(len(sel_xy_unique) * (voxel_size ** 2))
                    else:
                        area_selected_m2 = 0.0

                coverage_pct = (area_selected_m2 / area_total_m2 * 100.0) if area_total_m2 > 0 else 0.0
                records.append(
                    {
                        "site": site,
                        "scenario": scenario,
                        "timestep": year,
                        "voxel_size": voxel_size,
                        "area_total_m2": area_total_m2,
                        "area_selected_m2": area_selected_m2,
                        "coverage_pct": coverage_pct,
                        "selected_categories": ",".join(include_categories),
                    }
                )

    return pd.DataFrame.from_records(records)


def compute_bioavailability_2d3d(
    sites: List[str],
    scenarios: List[str],
    timesteps: List[int],
    voxel_size: int = DEFAULT_VOXEL_SIZE,
    auto_generate_urban_features: bool = True,
) -> pd.DataFrame:
    """
    Compute both 2D and 3D metrics for bioavailability:
    - Arboreal: 2D (unique XY columns) and 3D "stacked area" (count * voxel_size^2)
    - Low-vegetation: 3D "stacked area" (count * voxel_size^2)
    Returns a DataFrame with columns:
      site, scenario, timestep, arboreal_2d_m2, arboreal_3d_m2, low_veg_3d_m2
    """
    import numpy as np
    rows: List[Dict[str, Any]] = []

    for site in sites:
        for scenario in scenarios:
            for year in sorted(timesteps):
                if auto_generate_urban_features:
                    _ensure_urban_features(site, scenario, voxel_size, year)
                poly = _load_polydata(site, scenario, voxel_size, year)
                if poly is None or poly.n_points == 0 or "search_bioavailable" not in poly.point_data:
                    rows.append(
                        {
                            "site": site,
                            "scenario": scenario,
                            "timestep": year,
                            "arboreal_2d_m2": 0.0,
                            "arboreal_3d_m2": 0.0,
                            "low_veg_3d_m2": 0.0,
                        }
                    )
                    continue

                bio = poly.point_data["search_bioavailable"]
                all_xy = _unique_xy_keys(poly.points, voxel_size)

                # Arboreal 2D
                mask_arb = bio == "arboreal"
                if np.any(mask_arb):
                    arb_xy_unique = np.unique(all_xy[mask_arb], axis=0)
                    arb_2d = float(len(arb_xy_unique) * (voxel_size ** 2))
                    arb_3d = float(np.count_nonzero(mask_arb) * (voxel_size ** 2))
                else:
                    arb_2d = 0.0
                    arb_3d = 0.0

                # Low-vegetation 3D
                mask_low = bio == "low-vegetation"
                low_3d = float(np.count_nonzero(mask_low) * (voxel_size ** 2)) if np.any(mask_low) else 0.0

                rows.append(
                    {
                        "site": site,
                        "scenario": scenario,
                        "timestep": year,
                        "arboreal_2d_m2": arb_2d,
                        "arboreal_3d_m2": arb_3d,
                        "low_veg_3d_m2": low_3d,
                    }
                )
    return pd.DataFrame.from_records(rows)


def save_canopy_shrub_comparison(
    out_dir: str = OUTPUT_COMPARISON_DIR,
    out_name: str = "comparison_canopy-shrubs.csv",
    sites: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    timesteps: Optional[List[int]] = None,
    voxel_size: int = DEFAULT_VOXEL_SIZE,
    include_categories: Tuple[str, str] = ("low-vegetation", "arboreal"),
) -> str:
    """
    Compute canopy+shrubs 2D coverage and save a CSV in out_dir/out_name.
    """
    if sites is None:
        sites = DEFAULT_SITES
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if timesteps is None:
        timesteps = DEFAULT_TIMESTEPS

    df = compute_canopy_shrub_coverage(
        sites=sites,
        scenarios=scenarios,
        timesteps=timesteps,
        voxel_size=voxel_size,
        include_categories=include_categories,
    )
    _ensure_directory(out_dir)
    out_path = os.path.join(out_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path

if __name__ == "__main__":
    # Minimal demo: print first few rows if available
    df = example_removal_stats()
    with pd.option_context("display.max_columns", None):
        print(df.head(20))

    # Hollows trio
    print(save_hollows_unified(out_name="hollows_prediction.csv"))
    print(save_hollows_targets(out_name="hollows_external.csv"))
    # Build hollows_comparison.csv
    _ensure_directory(OUTPUT_COMPARISON_DIR)
    _h_pred = os.path.join(OUTPUT_COMPARISON_DIR, "hollows_prediction.csv")
    _h_ext = os.path.join(OUTPUT_COMPARISON_DIR, "hollows_external.csv")
    ours_h = pd.read_csv(_h_pred) if os.path.exists(_h_pred) else pd.DataFrame()
    ext_h = pd.read_csv(_h_ext) if os.path.exists(_h_ext) else pd.DataFrame()
    if not ours_h.empty and not ext_h.empty:
        # Compute per-step rates for trees and artificial from our prediction
        oh = ours_h.copy()
        oh["annual_rate_trees (structures/year)"] = oh.apply(
            lambda r: (r["hollow_bearing_tree_added_this_timestep"] / r["years_since_prev"]) if r["years_since_prev"] > 0 else 0.0,
            axis=1
        )
        oh["annual_rate_artificial (structures/year)"] = oh.apply(
            lambda r: (r["artificial_added_this_timestep"] / r["years_since_prev"]) if r["years_since_prev"] > 0 else 0.0,
            axis=1
        )
        oh["annual_rate_combined (structures/year)"] = oh.apply(
            lambda r: ((r["hollow_bearing_tree_added_this_timestep"] + r["artificial_added_this_timestep"]) / r["years_since_prev"]) if r["years_since_prev"] > 0 else 0.0,
            axis=1
        )
        ours_h2 = oh[[
            "site", "scenario", "timestep",
            "annual_rate_trees (structures/year)",
            "annual_rate_artificial (structures/year)",
            "annual_rate_combined (structures/year)",
        ]].copy()
        # External rows: map to comparative schema, set artificial to 0, use provided scenarios
        ext = ext_h.copy()
        ext["site"] = "le roux 2014"
        ext = ext.rename(columns={"year": "timestep"})
        # Prefer new column with units; fall back to legacy aliases if present
        if "annual rate (structures/year)" in ext.columns:
            ext["annual_rate_trees (structures/year)"] = ext["annual rate (structures/year)"].fillna(0.0)
        elif "site_annual_rate_per_year" in ext.columns:
            ext["annual_rate_trees (structures/year)"] = ext["site_annual_rate_per_year"].fillna(0.0)
        else:
            ext["annual_rate_trees (structures/year)"] = ext.get("site_rate_per_year", 0.0)
        ext["annual_rate_artificial (structures/year)"] = 0.0
        ext["annual_rate_combined (structures/year)"] = ext["annual_rate_trees (structures/year)"]
        ext_h2 = ext[[
            "site", "scenario", "timestep",
            "annual_rate_trees (structures/year)",
            "annual_rate_artificial (structures/year)",
            "annual_rate_combined (structures/year)",
        ]].copy()
        hollows_cmp = pd.concat([ours_h2, ext_h2], ignore_index=True, sort=False)
        hollows_cmp.to_csv(os.path.join(OUTPUT_COMPARISON_DIR, "hollows_comparison.csv"), index=False)
        print(os.path.join(OUTPUT_COMPARISON_DIR, "hollows_comparison.csv"))

    # Canopy/shrubs trio
    print(save_canopy_unified(out_name="canopy-shrubs_prediction.csv"))
    print(save_canopy_targets(out_name="canopy-shrubs_external.csv"))
    _c_pred = os.path.join(OUTPUT_COMPARISON_DIR, "canopy-shrubs_prediction.csv")
    _c_ext = os.path.join(OUTPUT_COMPARISON_DIR, "canopy-shrubs_external.csv")
    ours_c = pd.read_csv(_c_pred) if os.path.exists(_c_pred) else pd.DataFrame()
    ext_c = pd.read_csv(_c_ext) if os.path.exists(_c_ext) else pd.DataFrame()
    if not ours_c.empty and not ext_c.empty:
        # Our rows, rename with units for clarity
        ours_c2 = ours_c[["site", "scenario", "timestep", "annual_rate"]].copy()
        ours_c2 = ours_c2.rename(columns={"annual_rate": "annual_rate (m2/year)"})
        # External rows → site='living Melbourne', scenario=context, timestep=end_year
        ext = ext_c.rename(columns={"annual_rate_m2_per_year": "annual_rate (m2/year)"}).copy()
        ext_rows = []
        for ctx, grp in ext.groupby("context"):
            # Baseline row at first start_year with 0 rate
            first_start = int(grp["start_year"].min())
            ext_rows.append(
                {"site": "living Melbourne", "scenario": ctx, "timestep": first_start, "annual_rate (m2/year)": 0.0}
            )
            # Window rows at end_year with computed annual rate
            for _, r in grp.iterrows():
                ext_rows.append(
                    {
                        "site": "living Melbourne",
                        "scenario": r["context"],
                        "timestep": int(r["end_year"]),
                        "annual_rate (m2/year)": float(r["annual_rate (m2/year)"]),
                    }
                )
        ext_c2 = pd.DataFrame(ext_rows, columns=["site", "scenario", "timestep", "annual_rate (m2/year)"])
        canopy_cmp = pd.concat([ours_c2, ext_c2], ignore_index=True, sort=False)
        canopy_cmp.to_csv(os.path.join(OUTPUT_COMPARISON_DIR, "canopy-shrubs_comparison.csv"), index=False)
        print(os.path.join(OUTPUT_COMPARISON_DIR, "canopy-shrubs_comparison.csv"))

    # Legacy single-file
    canopy_out = save_canopy_shrub_comparison()
    print(f"Saved canopy+shrubs legacy comparison to: {canopy_out}")

    # Permeability trio
    print(save_permeability_unified(out_name="permeability_prediction.csv"))
    print(save_permeability_targets(out_name="permeability_external.csv"))
    _p_pred = os.path.join(OUTPUT_COMPARISON_DIR, "permeability_prediction.csv")
    _p_ext = os.path.join(OUTPUT_COMPARISON_DIR, "permeability_external.csv")
    ours_p = pd.read_csv(_p_pred) if os.path.exists(_p_pred) else pd.DataFrame()
    ext_p = pd.read_csv(_p_ext) if os.path.exists(_p_ext) else pd.DataFrame()
    if not ours_p.empty and not ext_p.empty:
        ours_p2 = ours_p[["site", "scenario", "timestep", "annual permeability added (m2/year)"]].copy()
        ours_p2["source"] = "ours"
        if "site_equivalent_annual_m2_per_year" in ext_p.columns:
            ext_p["annual_rate"] = ext_p["site_equivalent_annual_m2_per_year"]
        elif "annual_rate_m2_per_year" in ext_p.columns:
            ext_p["annual_rate"] = ext_p["annual_rate_m2_per_year"]
        ext_p2 = ext_p[["context", "scenario", "annual_rate"]].copy()
        ext_p2["source"] = "external"
        # Align column name to compare
        ours_p2 = ours_p2.rename(columns={"annual permeability added (m2/year)": "annual_rate"})
        perm_cmp = pd.concat(
            [
                ours_p2[["site", "scenario", "timestep", "annual_rate"]].assign(source="ours"),
                ext_p2.rename(columns={"context": "site"}).assign(timestep=""),
            ],
            ignore_index=True,
            sort=False,
        )
        perm_cmp.to_csv(os.path.join(OUTPUT_COMPARISON_DIR, "permeability_comparison.csv"), index=False)
        print(os.path.join(OUTPUT_COMPARISON_DIR, "permeability_comparison.csv"))

    # Connectivity trio
    print(save_connectivity_prediction(out_name="connectivity_prediction.csv"))
    print(save_connectivity_external(out_name="connectivity_external.csv"))
    print(save_connectivity_comparison(out_name="connectivity_comparison.csv"))

    # Removals trio (targets + ours unified)
    print(save_removal_unified(out_name="removal_prediction.csv"))
    print(save_removal_unified_targets(out_name="removal_external.csv"))
    # Build simple comparison: annualReplacementRate vs target rates
    _r_pred = os.path.join(OUTPUT_COMPARISON_DIR, "removal_prediction.csv")
    _r_ext = os.path.join(OUTPUT_COMPARISON_DIR, "removal_external.csv")
    ours_r = pd.read_csv(_r_pred) if os.path.exists(_r_pred) else pd.DataFrame()
    ext_r = pd.read_csv(_r_ext) if os.path.exists(_r_ext) else pd.DataFrame()
    if not ours_r.empty and not ext_r.empty:
        # Build our metrics: replacement, age-threshold, tolerance, combined
        ours_r2 = ours_r[[
            "site", "scenario", "timestep",
            "annualReplacementRate", "replacementPctOfCohort",
            "annualAgedOutRate", "agedOutPctOfCohort",
            "toleranceTreesPerYear", "tolerancePctOfCohort",
            "combinedTreesPerYear", "combinedPctOfCohort",
        ]].copy()
        ours_r2 = ours_r2.rename(columns={
            "annualReplacementRate": "replacement_rate_per_year (trees/year)",
            "replacementPctOfCohort": "replacement_rate_per_year (% of cohort/year)",
            "annualAgedOutRate": "age_threshold_rate_per_year (trees/year)",
            "agedOutPctOfCohort": "age_threshold_rate_per_year (% of cohort/year)",
            "toleranceTreesPerYear": "tolerance_rate_per_year (trees/year)",
            "tolerancePctOfCohort": "tolerance_rate_per_year (% of cohort/year)",
            "combinedTreesPerYear": "combined_rate_per_year (trees/year)",
            "combinedPctOfCohort": "combined_rate_per_year (% of cohort/year)",
        })
        # External rows provide percentage-of-cohort per year only
        ext_r2 = ext_r[["site", "scenario", "rate_per_year"]].copy()
        ext_r2["timestep"] = ""
        ext_r2 = ext_r2.rename(columns={"rate_per_year": "tolerance_rate_per_year (% of cohort/year)"})
        ext_r2["tolerance_rate_per_year (trees/year)"] = ""
        # Align column order
        ext_r2 = ext_r2[[
            "site", "scenario", "timestep",
            "tolerance_rate_per_year (trees/year)",
            "tolerance_rate_per_year (% of cohort/year)",
        ]]
        # For comparison CSV, include all our columns and the external tolerance row
        cols_keep = [
            "site", "scenario", "timestep",
            "replacement_rate_per_year (trees/year)",
            "replacement_rate_per_year (% of cohort/year)",
            "age_threshold_rate_per_year (trees/year)",
            "age_threshold_rate_per_year (% of cohort/year)",
            "tolerance_rate_per_year (trees/year)",
            "tolerance_rate_per_year (% of cohort/year)",
            "combined_rate_per_year (trees/year)",
            "combined_rate_per_year (% of cohort/year)",
        ]
        ours_r2 = ours_r2[cols_keep]
        removal_cmp = pd.concat([ours_r2, ext_r2], ignore_index=True, sort=False)
        removal_cmp.to_csv(os.path.join(OUTPUT_COMPARISON_DIR, "removal_comparison.csv"), index=False)
        print(os.path.join(OUTPUT_COMPARISON_DIR, "removal_comparison.csv"))


