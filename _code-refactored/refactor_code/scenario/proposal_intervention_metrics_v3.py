from __future__ import annotations

"""
Export direct v3 proposal/intervention quantity tables from assessed VTKs.

This module uses the five `proposal_*V3` + `*_intervention` point-data arrays
from the assessed `state_with_indicators.vtk` files rather than the older
semantic proxy logic in `final/a_info_proposal_interventions.py`.

Outputs are written in the same long-format raw intervention shape used by the
proposal-intervention streamgraph pipeline:

- per-site raw tables:
  - {statistics_root}/raw/{site}/interventions.csv
- aggregate raw table:
  - {statistics_root}/raw/interventions.csv
- comparison/highlight tables:
  - {statistics_root}/comparison/interventions.csv
  - {statistics_root}/highlights/interventions.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.blender.proposal_framebuffers import DEFAULT_OUTPUT_COLUMNS
from refactor_code.blender.proposal_framebuffers_vtk import (
    VTK_PROPOSAL_FAMILIES,
    build_blender_proposal_framebuffer_pointdata,
)
from refactor_code.paths import engine_output_state_vtk_path, refactor_statistics_root


SITES = ["trimmed-parade", "city", "uni"]
SCENARIOS = ["positive", "trending"]
YEARS = [0, 10, 30, 60, 90, 120, 150, 180]

PROPOSAL_LABELS = {
    "proposal-decay": "Decay",
    "proposal-release-control": "Release-Control",
    "proposal-recruit": "Recruit",
    "proposal-colonise": "Colonise",
    "proposal-deploy-structure": "Deploy-Structure",
}

SUPPORT_LEVELS = {
    ("proposal-deploy-structure", "Adapt-Utility-Pole"): "full",
    ("proposal-deploy-structure", "Translocated-Log"): "full",
    ("proposal-deploy-structure", "Upgrade-Feature"): "full",
    ("proposal-decay", "Buffer-Feature"): "full",
    ("proposal-decay", "Brace-Feature"): "partial",
    ("proposal-recruit", "Buffer-Feature"): "partial",
    ("proposal-recruit", "Rewild-Ground"): "full",
    ("proposal-colonise", "Rewild-Ground"): "mixed",
    ("proposal-colonise", "Enrich-Envelope"): "full",
    ("proposal-colonise", "Roughen-Envelope"): "partial",
    ("proposal-release-control", "Reduce-Pruning"): "partial",
    ("proposal-release-control", "Eliminate-Pruning"): "full",
}

RAW_COLUMNS = [
    "site",
    "scenario",
    "year",
    "proposal_id",
    "proposal_label",
    "intervention",
    "support",
    "measure",
    "metric_column",
    "value",
    "notes",
    "last_updated",
]


def normalize_output_mode(output_mode: str | None) -> str | None:
    return output_mode


def output_root(output_mode: str | None = None) -> Path:
    return refactor_statistics_root(output_mode)


def raw_site_path(site: str, output_mode: str | None = None) -> Path:
    path = output_root(output_mode) / "raw" / site / "interventions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def raw_all_path(output_mode: str | None = None) -> Path:
    path = output_root(output_mode) / "raw" / "interventions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def comparison_path(output_mode: str | None = None) -> Path:
    path = output_root(output_mode) / "comparison" / "interventions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def highlights_path(output_mode: str | None = None) -> Path:
    path = output_root(output_mode) / "highlights" / "interventions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def update_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _title_intervention(label: str) -> str:
    return label.replace("-", " ").title().replace(" ", "-")


def _family_measure(intervention: str) -> str:
    return f"{intervention.lower()} voxels"


def _read_framebuffers(mesh: pv.PolyData) -> dict[str, np.ndarray]:
    point_data = mesh.point_data
    expected = {DEFAULT_OUTPUT_COLUMNS[family] for family, _, _ in VTK_PROPOSAL_FAMILIES}
    if expected.issubset(set(point_data.keys())):
        return {name: np.asarray(point_data[name]).astype(np.uint8) for name in expected}
    return build_blender_proposal_framebuffer_pointdata(mesh)


def build_raw_interventions(
    *,
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: float = 1,
    output_mode: str | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    timestamp = update_timestamp()

    for site in sites:
        for scenario in scenarios:
            for year in years:
                vtk_path = engine_output_state_vtk_path(site, scenario, year, voxel_size, output_mode)
                if not vtk_path.exists():
                    continue

                mesh = pv.read(vtk_path)
                framebuffers = _read_framebuffers(mesh)

                # Use the direct VTK family tuples so the mapping is explicit per family.
                for family, _, _ in VTK_PROPOSAL_FAMILIES:
                    output_name = DEFAULT_OUTPUT_COLUMNS[family]
                    values = framebuffers[output_name]

                    intervention_labels: dict[int, str]
                    if family == "proposal-deploy-structure":
                        intervention_labels = {
                            2: "Adapt-Utility-Pole",
                            3: "Translocated-Log",
                            4: "Upgrade-Feature",
                        }
                    elif family == "proposal-decay":
                        intervention_labels = {
                            2: "Buffer-Feature",
                            3: "Brace-Feature",
                        }
                    elif family == "proposal-recruit":
                        intervention_labels = {
                            2: "Buffer-Feature",
                            3: "Rewild-Ground",
                        }
                    elif family == "proposal-colonise":
                        intervention_labels = {
                            2: "Rewild-Ground",
                            3: "Enrich-Envelope",
                            4: "Roughen-Envelope",
                        }
                    elif family == "proposal-release-control":
                        intervention_labels = {
                            2: "Reduce-Pruning",
                            3: "Eliminate-Pruning",
                        }
                    else:
                        raise ValueError(f"Unexpected family: {family}")

                    for code, intervention_label in intervention_labels.items():
                        count = int(np.sum(values == code))
                        rows.append(
                            {
                                "site": site,
                                "scenario": scenario,
                                "year": year,
                                "proposal_id": family.replace("proposal-", "").replace("-", "_"),
                                "proposal_label": PROPOSAL_LABELS[family],
                                "intervention": intervention_label,
                                "support": SUPPORT_LEVELS[(family, intervention_label)],
                                "measure": _family_measure(intervention_label),
                                "metric_column": "supported_voxel_count",
                                "value": count,
                                "notes": "Direct count from proposal_*V3_intervention assessed VTK arrays via blender_proposal-* framebuffers.",
                                "last_updated": timestamp,
                            }
                        )

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def build_comparison_table(raw_df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=key_columns
            + [
                "positive_value",
                "trending_value",
                "delta_trending_minus_positive",
                "trending_pct_of_positive",
                "positive_multiple_of_trending",
            ]
        )

    comparison = raw_df[key_columns].drop_duplicates().reset_index(drop=True)
    positive_values = (
        raw_df[raw_df["scenario"] == "positive"][key_columns + ["value"]]
        .drop_duplicates(subset=key_columns, keep="first")
        .rename(columns={"value": "positive_value"})
    )
    trending_values = (
        raw_df[raw_df["scenario"] == "trending"][key_columns + ["value"]]
        .drop_duplicates(subset=key_columns, keep="first")
        .rename(columns={"value": "trending_value"})
    )
    comparison = comparison.merge(positive_values, on=key_columns, how="left")
    comparison = comparison.merge(trending_values, on=key_columns, how="left")

    def pct_of_positive(row):
        positive = row["positive_value"]
        trending = row["trending_value"]
        if pd.isna(positive) or pd.isna(trending) or float(positive) == 0:
            return pd.NA
        return round(float(trending) / float(positive) * 100.0, 6)

    def positive_multiple(row):
        positive = row["positive_value"]
        trending = row["trending_value"]
        if pd.isna(positive) or pd.isna(trending) or float(trending) == 0:
            return pd.NA
        return round(float(positive) / float(trending), 6)

    comparison["delta_trending_minus_positive"] = comparison["trending_value"] - comparison["positive_value"]
    comparison["trending_pct_of_positive"] = comparison.apply(pct_of_positive, axis=1)
    comparison["positive_multiple_of_trending"] = comparison.apply(positive_multiple, axis=1)
    return comparison


def build_highlights_table(comparison_df: pd.DataFrame, key_columns: list[str], top_n: int = 15) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame()

    positive_dominates = comparison_df[
        comparison_df["positive_value"].fillna(0) > comparison_df["trending_value"].fillna(0)
    ].copy()
    positive_dominates = positive_dominates.sort_values(
        ["trending_pct_of_positive", "positive_multiple_of_trending"],
        ascending=[True, False],
        na_position="last",
    ).head(top_n)
    positive_dominates["highlight_type"] = "positive_dominates"

    trending_dominates = comparison_df[
        comparison_df["trending_value"].fillna(0) > comparison_df["positive_value"].fillna(0)
    ].copy()
    trending_dominates = trending_dominates.sort_values(
        ["delta_trending_minus_positive"],
        ascending=[False],
        na_position="last",
    ).head(top_n)
    trending_dominates["highlight_type"] = "trending_dominates"

    ordered_columns = ["highlight_type"] + key_columns + [
        "positive_value",
        "trending_value",
        "delta_trending_minus_positive",
        "trending_pct_of_positive",
        "positive_multiple_of_trending",
    ]
    return pd.concat([positive_dominates, trending_dominates], ignore_index=True)[ordered_columns]


def save_outputs(raw_df: pd.DataFrame, output_mode: str | None = None) -> None:
    for site in sorted(raw_df["site"].dropna().unique().tolist()):
        site_df = raw_df[raw_df["site"] == site].copy()
        site_path = raw_site_path(site, output_mode)
        site_df.to_csv(site_path, index=False)

    raw_all = raw_all_path(output_mode)
    raw_df.to_csv(raw_all, index=False)

    key_columns = [
        "site",
        "year",
        "proposal_id",
        "proposal_label",
        "intervention",
        "support",
        "measure",
        "metric_column",
    ]
    comparison_df = build_comparison_table(raw_df, key_columns)
    highlights_df = build_highlights_table(comparison_df, key_columns)
    comparison_df.to_csv(comparison_path(output_mode), index=False)
    highlights_df.to_csv(highlights_path(output_mode), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export direct v3 proposal intervention quantities from proposal_*V3_intervention arrays."
    )
    parser.add_argument("--sites", default="trimmed-parade,city,uni")
    parser.add_argument("--scenarios", default="positive,trending")
    parser.add_argument("--years", default="0,10,30,60,90,120,150,180")
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--output-mode", default=None)
    return parser.parse_args()


def _parse_csv_list(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    raw_df = build_raw_interventions(
        sites=_parse_csv_list(args.sites, str),
        scenarios=_parse_csv_list(args.scenarios, str),
        years=_parse_csv_list(args.years, int),
        voxel_size=args.voxel_size,
        output_mode=normalize_output_mode(args.output_mode),
    )
    save_outputs(raw_df, normalize_output_mode(args.output_mode))


if __name__ == "__main__":
    main()
