from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyvista as pv

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT.parent))

from _futureSim_refactored.paths import (
    engine_output_validation_dir,
    scenario_log_df_path,
    scenario_node_df_path,
    scenario_pole_df_path,
    scenario_state_vtk_path,
    scenario_tree_df_path,
)


TREE_COLUMNS = [
    "control",
    "control_reached",
    "under-node-treatment",
    "action",
    "replacement_reason",
    "proposal-decay_decision",
    "proposal-decay_intervention",
    "proposal-release-control_decision",
    "proposal-release-control_intervention",
    "proposal-release-control_target_years",
    "proposal-release-control_years",
    "proposal-recruit_decision",
    "proposal-recruit_intervention",
    "proposal-colonise_decision",
    "proposal-colonise_intervention",
    "lifecycle_state",
    "size",
]
NODE_COLUMNS = [
    "control",
    "under-node-treatment",
    "action",
    "forest_control",
]
VTK_ARRAYS = [
    "scenario_under-node-treatment",
    "scenario_bioEnvelope",
    "forest_control",
    "proposal_decayV3",
    "proposal_release_controlV3",
    "proposal_coloniseV3",
    "proposal_recruitV3",
    "proposal_deploy_structureV3",
    "proposal_decayV3_intervention",
    "proposal_release_controlV3_intervention",
    "proposal_coloniseV3_intervention",
    "proposal_recruitV3_intervention",
    "proposal_deploy_structureV3_intervention",
]


def _value_counts(series: pd.Series) -> dict[str, int]:
    cleaned = series.fillna("NaN").astype(str)
    counts = cleaned.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def summarize_csv(path: Path, columns: list[str]) -> dict:
    if not path.exists():
        return {"exists": False}

    df = pd.read_csv(path)
    summary = {"exists": True, "row_count": int(len(df)), "columns": df.columns.tolist(), "counts": {}}
    for column in columns:
        if column in df.columns:
            summary["counts"][column] = _value_counts(df[column])
    return summary


def summarize_vtk(path: Path, arrays: list[str]) -> dict:
    if not path.exists():
        return {"exists": False}

    mesh = pv.read(path)
    summary = {
        "exists": True,
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "arrays": {},
    }
    for array_name in arrays:
        if array_name in mesh.array_names:
            summary["arrays"][array_name] = _value_counts(pd.Series(mesh[array_name]))
    return summary


def build_summary(
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: int = 1,
    output_mode: str = "canonical",
) -> dict:
    summary = {
        "output_mode": output_mode,
        "sites": sites,
        "scenarios": scenarios,
        "years": years,
        "voxel_size": voxel_size,
        "results": {},
    }

    for site in sites:
        summary["results"][site] = {}
        for scenario in scenarios:
            summary["results"][site][scenario] = {}
            for year in years:
                summary["results"][site][scenario][str(year)] = {
                    "tree": summarize_csv(scenario_tree_df_path(site, scenario, year, voxel_size, output_mode), TREE_COLUMNS),
                    "log": summarize_csv(scenario_log_df_path(site, scenario, year, voxel_size, output_mode), []),
                    "pole": summarize_csv(scenario_pole_df_path(site, scenario, year, voxel_size, output_mode), []),
                    "node": summarize_csv(scenario_node_df_path(site, scenario, year, voxel_size, output_mode), NODE_COLUMNS),
                    "vtk": summarize_vtk(scenario_state_vtk_path(site, scenario, year, voxel_size, output_mode), VTK_ARRAYS),
                }
    return summary


def compare_summaries(left: dict, right: dict) -> dict:
    diff = {}
    for site, site_data in left["results"].items():
        for scenario, scenario_data in site_data.items():
            for year, year_data in scenario_data.items():
                right_year = right["results"].get(site, {}).get(scenario, {}).get(year, {})
                year_diff = {}
                for artifact_name, artifact_summary in year_data.items():
                    other_summary = right_year.get(artifact_name, {})
                    if artifact_summary != other_summary:
                        year_diff[artifact_name] = {
                            "left": artifact_summary,
                            "right": other_summary,
                        }
                if year_diff:
                    diff.setdefault(site, {}).setdefault(scenario, {})[year] = year_diff
    return diff


def write_summary(summary: dict, filename: str, output_mode: str = "validation") -> Path:
    output_path = engine_output_validation_dir(output_mode) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return output_path
