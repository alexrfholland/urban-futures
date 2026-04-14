"""
Schema: v4_indicators

Extracts per-state V4 capability indicator voxel counts from an in-memory mesh.
Produces one row per indicator with columns: site, scenario, year, indicator, count.

The extraction logic is lifted from v4_indicator_extract.py.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from _futureSim_refactored.sim.setup.constants import (
    RECRUIT_FULL,
    RECRUIT_PARTIAL,
    RELEASECONTROL_FULL,
    RELEASECONTROL_PARTIAL,
)
from _futureSim_refactored.outputs.stats.vtk_to_stat_counts import Schema, register


# ---------------------------------------------------------------------------
# Indicator order (canonical)
# ---------------------------------------------------------------------------

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

CSV_COLUMNS = ["site", "scenario", "year", "indicator", "count"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str_arr(mesh: pv.PolyData, name: str) -> np.ndarray:
    if name in mesh.point_data:
        return np.char.lower(np.asarray(mesh.point_data[name]).astype(str))
    return np.full(mesh.n_points, "none", dtype="<U64")


def _num_arr(mesh: pv.PolyData, name: str) -> np.ndarray:
    if name in mesh.point_data:
        return np.nan_to_num(np.asarray(mesh.point_data[name]).astype(float), nan=0.0)
    return np.zeros(mesh.n_points, dtype=float)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def compute_indicators(mesh: pv.PolyData, *, is_baseline: bool = False) -> dict[str, int]:
    fs = _str_arr(mesh, "forest_size")
    release = _str_arr(mesh, "proposal_release_control_intervention")
    recruit = _str_arr(mesh, "proposal_recruit_intervention")

    r: dict[str, int] = {}

    # Bird
    r["Bird.acquire.peeling-bark"] = int(np.sum(_num_arr(mesh, "stat_peeling bark") > 0))
    perch = _num_arr(mesh, "stat_perch branch") > 0
    bird_comm_fs = np.isin(fs, ["senescing", "snag", "artificial"])
    r["Bird.communicate.perch-branch"] = int(np.sum(perch & bird_comm_fs))
    r["Bird.reproduce.hollow"] = int(np.sum(_num_arr(mesh, "stat_hollow") > 0))

    # Lizard
    bio = _str_arr(mesh, "search_bioavailable")
    grass = bio == "low-vegetation"
    dead = _num_arr(mesh, "stat_dead branch") > 0
    epiphyte = _num_arr(mesh, "stat_epiphyte") > 0
    r["Lizard.acquire.grass"] = int(np.sum(grass))
    r["Lizard.acquire.dead-branch"] = int(np.sum(dead))
    r["Lizard.acquire.epiphyte"] = int(np.sum(epiphyte))
    r["Lizard.acquire"] = int(np.sum(grass | dead | epiphyte))
    urban = _str_arr(mesh, "search_urban_elements")
    ground = (bio == "low-vegetation") | (bio == "open space")
    paved = (urban == "roadway") | (urban == "busy roadway") | (urban == "parking")
    r["Lizard.communicate.not-paved"] = int(np.sum(ground & ~paved))
    nurse = _num_arr(mesh, "stat_fallen log") > 0
    fallen_tree = np.isin(fs, ["fallen", "decayed"])
    r["Lizard.reproduce.nurse-log"] = int(np.sum(nurse))
    r["Lizard.reproduce.fallen-tree"] = int(np.sum(fallen_tree))
    r["Lizard.reproduce"] = int(np.sum(nurse | fallen_tree))

    # Tree.acquire
    moderated = release == RELEASECONTROL_PARTIAL.lower()
    autonomous = release == RELEASECONTROL_FULL.lower()
    r["Tree.acquire.moderated"] = int(np.sum(moderated))
    r["Tree.acquire.autonomous"] = int(np.sum(autonomous))
    r["Tree.acquire"] = int(np.sum(moderated | autonomous))

    # Tree.communicate
    snag = fs == "snag"
    fallen = fs == "fallen"
    decayed = fs == "decayed"
    r["Tree.communicate.snag"] = int(np.sum(snag))
    r["Tree.communicate.fallen"] = int(np.sum(fallen))
    r["Tree.communicate.decayed"] = int(np.sum(decayed))
    r["Tree.communicate"] = int(np.sum(snag | fallen | decayed))

    # Tree.reproduce
    smaller = recruit == RECRUIT_PARTIAL.lower()
    larger = recruit == RECRUIT_FULL.lower()
    r["Tree.reproduce.smaller-patches-rewild"] = int(np.sum(smaller))
    r["Tree.reproduce.larger-patches-rewild"] = int(np.sum(larger))
    r["Tree.reproduce"] = int(np.sum(smaller | larger))

    return r


def extract_v4_indicators(mesh: pv.PolyData, site: str, scenario: str, year: int, **ctx) -> list[dict]:
    is_baseline = ctx.get("is_baseline", False)
    counts = compute_indicators(mesh, is_baseline=is_baseline)
    return [
        {"site": site, "scenario": scenario, "year": year, "indicator": ind, "count": counts.get(ind, 0)}
        for ind in INDICATOR_ORDER
    ]


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

register(Schema(
    name="v4_indicators",
    extract=extract_v4_indicators,
    columns=CSV_COLUMNS,
))
