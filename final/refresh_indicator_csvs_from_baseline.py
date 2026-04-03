#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "final"))
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

import a_info_gather_capabilities as gather_capabilities
from refactor_code.paths import refactor_statistics_root


DEFAULT_SITES = ("trimmed-parade", "city", "uni")
SCENARIO_ORDER = {"baseline": 0, "positive": 1, "trending": 2}


def parse_sites(raw: str | None) -> list[str]:
    if not raw or raw.strip().lower() == "all":
        return list(DEFAULT_SITES)
    return [site.strip() for site in raw.split(",") if site.strip()]


def indicator_csv_path(site: str, voxel_size: int) -> Path:
    return refactor_statistics_root("validation") / "csv" / f"{site}_{voxel_size}_indicator_counts.csv"


def action_csv_path(site: str, voxel_size: int) -> Path:
    return refactor_statistics_root("validation") / "csv" / f"{site}_{voxel_size}_action_counts.csv"


def sort_indicator_df(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["_scenario_order"] = working["scenario"].map(SCENARIO_ORDER).fillna(99)
    working["_year_num"] = pd.to_numeric(working["year"], errors="coerce")
    working["_indicator_sort"] = working["indicator_id"].astype(str)
    return (
        working
        .sort_values(["_scenario_order", "_year_num", "_indicator_sort"])
        .drop(columns=["_scenario_order", "_year_num", "_indicator_sort"])
        .reset_index(drop=True)
    )


def sort_action_df(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["_scenario_order"] = working["scenario"].map(SCENARIO_ORDER).fillna(99)
    working["_year_num"] = pd.to_numeric(working["year"], errors="coerce")
    sort_cols = ["_scenario_order", "_year_num"]
    for optional in ("proposal", "intervention", "support_action", "category"):
        if optional in working.columns:
            sort_cols.append(optional)
    return (
        working
        .sort_values(sort_cols)
        .drop(columns=["_scenario_order", "_year_num"])
        .reset_index(drop=True)
    )


def refresh_site(site: str, run_root: Path, voxel_size: int) -> None:
    os.environ["REFACTOR_RUN_OUTPUT_ROOT"] = str(run_root.resolve())

    vtk_path = gather_capabilities.get_vtk_path(site, "baseline", -180, voxel_size, "validation")
    baseline_indicator_counts, baseline_action_counts, _ = gather_capabilities.process_vtk(
        vtk_path=vtk_path,
        site=site,
        scenario="baseline",
        year=-180,
        voxel_size=voxel_size,
        save_vtk=True,
        output_mode="validation",
    )

    indicator_path = indicator_csv_path(site, voxel_size)
    indicator_df = pd.read_csv(indicator_path)
    baseline_indicator_df = pd.DataFrame(baseline_indicator_counts)

    indicator_df = indicator_df[~((indicator_df["scenario"] == "baseline") & (indicator_df["year"] == -180))]
    indicator_df = pd.concat([indicator_df, baseline_indicator_df], ignore_index=True)

    baseline_map = {
        row["indicator_id"]: row["count"]
        for _, row in baseline_indicator_df.iterrows()
    }

    def calc_pct(row: pd.Series):
        baseline_value = baseline_map.get(row["indicator_id"], 0)
        if baseline_value and baseline_value > 0:
            return round((float(row["count"]) / float(baseline_value)) * 100, 1)
        return None

    indicator_df["pct_of_baseline"] = indicator_df.apply(calc_pct, axis=1)
    indicator_df = sort_indicator_df(indicator_df)
    indicator_path.parent.mkdir(parents=True, exist_ok=True)
    indicator_df.to_csv(indicator_path, index=False)

    action_path = action_csv_path(site, voxel_size)
    if action_path.exists():
        action_df = pd.read_csv(action_path)
        baseline_action_df = pd.DataFrame(baseline_action_counts)
        action_df = action_df[~((action_df["scenario"] == "baseline") & (action_df["year"] == -180))]
        action_df = pd.concat([action_df, baseline_action_df], ignore_index=True)
        action_df = sort_action_df(action_df)
        action_df.to_csv(action_path, index=False)

    print(f"Refreshed baseline-relative CSVs for {site}")
    for _, row in baseline_indicator_df.iterrows():
        print(f"  {row['indicator_id']}: baseline={row['count']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh per-site indicator/action CSVs using a regenerated baseline, without rerunning all scenario states."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--sites", default="all")
    parser.add_argument("--voxel-size", type=int, default=1)
    args = parser.parse_args()

    for site in parse_sites(args.sites):
        refresh_site(site, args.run_root, args.voxel_size)


if __name__ == "__main__":
    main()
