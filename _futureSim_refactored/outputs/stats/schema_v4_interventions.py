"""
Schema: v4_interventions

Extracts per-state V4 proposal intervention voxel counts from an in-memory mesh.
Produces one row per (proposal, intervention) with columns:
    site, scenario, year, proposal, intervention, support, count

The extraction logic is lifted from proposal_intervention_metrics_v4.py.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from _futureSim_refactored.sim.setup.constants import (
    COLONISE_FULL_ENVELOPE,
    COLONISE_FULL_GROUND,
    COLONISE_PARTIAL_ENVELOPE,
    DECAY_FULL,
    DECAY_PARTIAL,
    DEPLOY_FULL_LOG,
    DEPLOY_FULL_POLE,
    DEPLOY_FULL_UPGRADE,
    INTERVENTION_SUPPORT,
    RECRUIT_FULL,
    RECRUIT_PARTIAL,
    RELEASECONTROL_FULL,
    RELEASECONTROL_PARTIAL,
)
from _futureSim_refactored.outputs.stats.vtk_to_stat_counts import Schema, register


FAMILY_VTK_ARRAY = {
    "decay": "proposal_decay_intervention",
    "release-control": "proposal_release_control_intervention",
    "recruit": "proposal_recruit_intervention",
    "colonise": "proposal_colonise_intervention",
    "deploy-structure": "proposal_deploy_structure_intervention",
}

FAMILY_INTERVENTIONS = {
    "decay": [DECAY_FULL, DECAY_PARTIAL],
    "release-control": [RELEASECONTROL_FULL, RELEASECONTROL_PARTIAL],
    "recruit": [RECRUIT_FULL, RECRUIT_PARTIAL],
    "colonise": [COLONISE_FULL_GROUND, COLONISE_FULL_ENVELOPE, COLONISE_PARTIAL_ENVELOPE],
    "deploy-structure": [DEPLOY_FULL_POLE, DEPLOY_FULL_LOG, DEPLOY_FULL_UPGRADE],
}

CSV_COLUMNS = ["site", "scenario", "year", "proposal", "intervention", "support", "count"]


def extract_v4_interventions(mesh: pv.PolyData, site: str, scenario: str, year: int, **ctx) -> list[dict]:
    rows: list[dict] = []
    for family, array_name in FAMILY_VTK_ARRAY.items():
        if array_name not in mesh.point_data.keys():
            continue
        values = np.array([str(v) for v in mesh.point_data[array_name]])
        for intervention in FAMILY_INTERVENTIONS[family]:
            count = int(np.sum(values == intervention))
            rows.append({
                "site": site,
                "scenario": scenario,
                "year": year,
                "proposal": family,
                "intervention": intervention,
                "support": INTERVENTION_SUPPORT[intervention],
                "count": count,
            })
    return rows


register(Schema(
    name="v4_interventions",
    extract=extract_v4_interventions,
    columns=CSV_COLUMNS,
))
