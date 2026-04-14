"""
Schema: v4_decisions

Extracts per-state proposal decision counts (rejected / accepted_full /
accepted_partial) from an in-memory mesh + nodeDF.

Two output CSVs per state:
  - v4_decisions.csv      — top-level counts per proposal family
  - v4_decision_subgroups.csv — finer breakdowns (colonise by bioenvelope,
    deploy by type, recruit by urban element)

The extraction logic is lifted from proposal_decisions_metrics_v4.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

from _futureSim_refactored.sim.setup.constants import INTERVENTION_SUPPORT
from _futureSim_refactored.outputs.stats.vtk_to_stat_counts import Schema, register


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# NodeDF-based families (tree-node level).  Bool = include rejected.
NODEDF_FAMILIES: dict[str, bool] = {
    "decay": True,
    "recruit": False,
}

# VTK-based families (voxel level).
VTK_FAMILIES = {
    "release-control":  ("proposal_release_control",  "proposal_release_control_intervention",  True),
    "colonise":         ("proposal_colonise",         "proposal_colonise_intervention",         False),
    "deploy-structure": ("proposal_deploy_structure", "proposal_deploy_structure_intervention", False),
}

CSV_COLUMNS = ["site", "scenario", "year", "proposal", "unit", "rejected", "accepted_full", "accepted_partial"]
SUBGROUP_CSV_COLUMNS = ["site", "scenario", "year", "proposal", "unit", "support_bucket", "subgroup", "count"]

# --- Colonise subgroups ---
COLONISE_SUBGROUP_ORDER = ["ground", "green roof", "brown roof", "biofacade"]
COLONISE_BIOENVELOPE_TO_SUBGROUP = {
    "node-rewilded": "ground",
    "footprint-depaved": "ground",
    "footprint-depaved-connected": "ground",
    "rewilded": "ground",
    "otherground": "ground",
    "scenario-rewilded": "ground",
    "greenroof": "green roof",
    "brownroof": "brown roof",
    "livingfacade": "biofacade",
}

# --- Deploy subgroups ---
DEPLOY_SUBGROUP_ORDER = ["pole", "log", "upgrade"]
DEPLOY_INTERVENTION_TO_SUBGROUP = {
    "adapt-utility-pole": "pole",
    "translocate-log": "log",
    "translocate-deadwood": "log",
    "upgrade-feature": "upgrade",
}

# --- Recruit subgroups ---
RECRUIT_SUBGROUP_ORDER = ["open", "roads", "street potential", "other"]
RECRUIT_URBAN_TO_SUBGROUP = {
    "open space": "open",
    "existing conversion": "open",
    "roadway": "roads",
    "busy roadway": "roads",
    "parking": "street potential",
    "other street potential": "street potential",
    "other street potenti": "street potential",  # <U20 truncation
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _node_df_path(root: Path, site: str, scenario: str, year: int) -> Path:
    return root / "temp" / "interim-data" / site / f"{site}_{scenario}_1_nodeDF_{year}.csv"


def _normalise_bio(val: str) -> str:
    return str(val).strip().lower().replace(" ", "").replace("_", "")


# ---------------------------------------------------------------------------
# NodeDF extraction (decay, recruit)
# ---------------------------------------------------------------------------

def _extract_nodedf_rows(root: Path, site: str, scenario: str, year: int) -> list[dict]:
    path = _node_df_path(root, site, scenario, year)
    if not path.exists():
        return []

    df = pd.read_csv(path, low_memory=False)
    rows: list[dict] = []

    for family, include_rej in NODEDF_FAMILIES.items():
        decision_col = f"proposal-{family}_decision"
        intervention_col = f"proposal-{family}_intervention"
        if decision_col not in df.columns or intervention_col not in df.columns:
            continue

        accepted_label = f"proposal-{family}_accepted"
        rejected_label = f"proposal-{family}_rejected"

        rejected = int((df[decision_col] == rejected_label).sum()) if include_rej else 0
        accepted_mask = df[decision_col] == accepted_label
        supports = df.loc[accepted_mask, intervention_col].map(INTERVENTION_SUPPORT)
        accepted_full = int((supports == "full").sum())
        accepted_partial = int((supports == "partial").sum())

        rows.append({
            "site": site,
            "scenario": scenario,
            "year": year,
            "proposal": family,
            "unit": "trees",
            "rejected": rejected,
            "accepted_full": accepted_full,
            "accepted_partial": accepted_partial,
        })
    return rows


# ---------------------------------------------------------------------------
# VTK extraction (release-control, colonise, deploy-structure)
# ---------------------------------------------------------------------------

def _extract_vtk_rows(mesh: pv.PolyData, site: str, scenario: str, year: int) -> list[dict]:
    rows: list[dict] = []
    for family, (decision_array, intervention_array, include_rej) in VTK_FAMILIES.items():
        if decision_array not in mesh.point_data.keys():
            continue
        if intervention_array not in mesh.point_data.keys():
            continue

        decisions = np.array([str(v) for v in mesh.point_data[decision_array]])
        interventions = np.array([str(v) for v in mesh.point_data[intervention_array]])

        accepted_label = f"proposal-{family}_accepted"
        rejected_label = f"proposal-{family}_rejected"

        rejected = int((decisions == rejected_label).sum()) if include_rej else 0
        accepted_mask = decisions == accepted_label
        accepted_supports = np.array([INTERVENTION_SUPPORT.get(iv, "") for iv in interventions[accepted_mask]])
        accepted_full = int((accepted_supports == "full").sum())
        accepted_partial = int((accepted_supports == "partial").sum())

        rows.append({
            "site": site,
            "scenario": scenario,
            "year": year,
            "proposal": family,
            "unit": "voxels",
            "rejected": rejected,
            "accepted_full": accepted_full,
            "accepted_partial": accepted_partial,
        })
    return rows


# ---------------------------------------------------------------------------
# Subgroup extraction
# ---------------------------------------------------------------------------

def _extract_colonise_subgroups(mesh: pv.PolyData, site: str, scenario: str, year: int) -> list[dict]:
    needed = {"proposal_colonise", "scenario_bioEnvelope", "proposal_colonise_intervention"}
    if not needed.issubset(mesh.point_data.keys()):
        return []

    decisions = np.array([str(v) for v in mesh.point_data["proposal_colonise"]])
    bio = np.array([str(v) for v in mesh.point_data["scenario_bioEnvelope"]])
    interventions = np.array([str(v) for v in mesh.point_data["proposal_colonise_intervention"]])
    accepted = decisions == "proposal-colonise_accepted"

    counts: dict[tuple[str, str], int] = {}
    for idx in np.where(accepted)[0]:
        sub = COLONISE_BIOENVELOPE_TO_SUBGROUP.get(_normalise_bio(bio[idx]))
        if sub is None:
            continue
        bucket = INTERVENTION_SUPPORT.get(interventions[idx])
        if bucket not in ("full", "partial"):
            continue
        counts[(bucket, sub)] = counts.get((bucket, sub), 0) + 1

    rows: list[dict] = []
    for bucket in ("full", "partial"):
        for sub in COLONISE_SUBGROUP_ORDER:
            rows.append({
                "site": site, "scenario": scenario, "year": year,
                "proposal": "colonise", "unit": "voxels",
                "support_bucket": bucket, "subgroup": sub,
                "count": counts.get((bucket, sub), 0),
            })
    return rows


def _extract_deploy_subgroups(mesh: pv.PolyData, site: str, scenario: str, year: int) -> list[dict]:
    needed = {"proposal_deploy_structure", "proposal_deploy_structure_intervention"}
    if not needed.issubset(mesh.point_data.keys()):
        return []

    decisions = np.array([str(v) for v in mesh.point_data["proposal_deploy_structure"]])
    interventions = np.array([str(v) for v in mesh.point_data["proposal_deploy_structure_intervention"]])
    accepted = decisions == "proposal-deploy-structure_accepted"

    counts: dict[str, int] = {}
    for idx in np.where(accepted)[0]:
        sub = DEPLOY_INTERVENTION_TO_SUBGROUP.get(interventions[idx])
        if sub is not None:
            counts[sub] = counts.get(sub, 0) + 1

    rows: list[dict] = []
    for sub in DEPLOY_SUBGROUP_ORDER:
        rows.append({
            "site": site, "scenario": scenario, "year": year,
            "proposal": "deploy-structure", "unit": "voxels",
            "support_bucket": "full", "subgroup": sub,
            "count": counts.get(sub, 0),
        })
    return rows


def _extract_recruit_subgroups(root: Path, mesh: pv.PolyData, site: str, scenario: str, year: int) -> list[dict]:
    df_path = _node_df_path(root, site, scenario, year)
    if not df_path.exists():
        return []
    if "search_urban_elements" not in mesh.point_data.keys():
        return []

    df = pd.read_csv(df_path, low_memory=False)
    if "proposal-recruit_decision" not in df.columns or "voxel_index" not in df.columns:
        return []

    accepted = df[df["proposal-recruit_decision"] == "proposal-recruit_accepted"].copy()
    if accepted.empty:
        return [
            {"site": site, "scenario": scenario, "year": year, "proposal": "recruit",
             "unit": "trees", "support_bucket": bucket, "subgroup": sub, "count": 0}
            for bucket in ("full", "partial")
            for sub in RECRUIT_SUBGROUP_ORDER
        ]

    urban = np.array([str(v) for v in mesh.point_data["search_urban_elements"]])
    voxel_idx = accepted["voxel_index"].astype(int).to_numpy()
    accepted["_urban"] = urban[voxel_idx]
    accepted["_subgroup"] = accepted["_urban"].map(RECRUIT_URBAN_TO_SUBGROUP).fillna("other")
    accepted["_bucket"] = accepted["proposal-recruit_intervention"].map(INTERVENTION_SUPPORT).fillna("")

    group_counts = accepted.groupby(["_bucket", "_subgroup"]).size().to_dict()

    rows: list[dict] = []
    for bucket in ("full", "partial"):
        for sub in RECRUIT_SUBGROUP_ORDER:
            rows.append({
                "site": site, "scenario": scenario, "year": year,
                "proposal": "recruit", "unit": "trees",
                "support_bucket": bucket, "subgroup": sub,
                "count": int(group_counts.get((bucket, sub), 0)),
            })
    return rows


# ---------------------------------------------------------------------------
# Top-level extract function (schema contract)
# ---------------------------------------------------------------------------

def extract_v4_decisions(mesh: pv.PolyData, site: str, scenario: str, year: int, **ctx) -> list[dict]:
    root = Path(ctx["root"])
    rows: list[dict] = []
    rows.extend(_extract_nodedf_rows(root, site, scenario, year))
    rows.extend(_extract_vtk_rows(mesh, site, scenario, year))
    return rows


def extract_v4_decision_subgroups(mesh: pv.PolyData, site: str, scenario: str, year: int, **ctx) -> list[dict]:
    root = Path(ctx["root"])
    rows: list[dict] = []
    rows.extend(_extract_colonise_subgroups(mesh, site, scenario, year))
    rows.extend(_extract_deploy_subgroups(mesh, site, scenario, year))
    rows.extend(_extract_recruit_subgroups(root, mesh, site, scenario, year))
    return rows


# ---------------------------------------------------------------------------
# Register both schemas
# ---------------------------------------------------------------------------

register(Schema(
    name="v4_decisions",
    extract=extract_v4_decisions,
    columns=CSV_COLUMNS,
))

register(Schema(
    name="v4_decision_subgroups",
    extract=extract_v4_decision_subgroups,
    columns=SUBGROUP_CSV_COLUMNS,
))
