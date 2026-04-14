"""
V4 proposal decision extractor (accepted-full / accepted-partial / rejected).

Per-family data source contract:

    family            source            unit       include rejected?
    --------------------------------------------------------------
    decay             nodeDF            trees      yes
    recruit           nodeDF            trees      no
    release-control   VTK point data    voxels     yes
    colonise          VTK point data    voxels     no
    deploy-structure  VTK point data    voxels     no

Decay and recruit both operate at the tree-node level, so counts come from
the nodeDF via `proposal-{family}_decision` + `proposal-{family}_intervention`.
The `recruit_mechanism` column is telemetry (and conflates deploy-structure
deadwood placement, which also sets `isNewTree=True`), so we only consult the
proposal columns for the decision/intervention breakdown.

For release-control, rejection is a meaningful counterfactual (suppressed
canopy growth) so rejected voxels are shown alongside accepted ones. For
colonise/deploy-structure the "rejected" channel is a setup artefact
(candidate-site screening) rather than a simulated decision, so only accepted
voxels (full vs partial) are meaningful. Recruit similarly does not track
rejection at the node level — nodes are either accepted or not-assessed.

Output: one CSV `proposal_decisions.csv` under
`{root}/output/stats/proposal-decisions/` with unified schema:

    site,scenario,year,proposal,unit,rejected,accepted_full,accepted_partial

USAGE:
    uv run python _futureSim_refactored/outputs/stats/proposal_decisions_metrics_v4.py \
        --root _data-refactored/model-outputs/generated-states/4.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _futureSim_refactored.sim.setup.constants import INTERVENTION_SUPPORT


SITES = ["trimmed-parade", "city", "uni"]
SCENARIOS = ["positive", "trending"]
YEARS = [0, 1, 10, 30, 60, 90, 120, 150, 180]

# Families counted from the nodeDF (tree-node level). Bool flag = include rejected.
NODEDF_FAMILIES: dict[str, bool] = {
    "decay":   True,
    "recruit": False,
}

# Families counted from the VTK point-data arrays (voxel level). Bool flag = include rejected.
VTK_FAMILIES = {
    "release-control":  ("proposal_release_control",  "proposal_release_control_intervention",  True),
    "colonise":         ("proposal_colonise",         "proposal_colonise_intervention",         False),
    "deploy-structure": ("proposal_deploy_structure", "proposal_deploy_structure_intervention", False),
}

CSV_COLUMNS = ["site", "scenario", "year", "proposal", "unit", "rejected", "accepted_full", "accepted_partial"]
SUBGROUP_CSV_COLUMNS = ["site", "scenario", "year", "proposal", "unit", "support_bucket", "subgroup", "count"]

# --- Sub-grouping rules --------------------------------------------------
# Colonise: split accepted voxels by scenario_bioEnvelope value.
# Group labels are display order left-to-right within each support bucket.
COLONISE_SUBGROUP_ORDER = ["ground", "green roof", "brown roof", "biofacade"]
COLONISE_BIOENVELOPE_TO_SUBGROUP = {
    # ground variants — all rolled together
    "node-rewilded": "ground",
    "footprint-depaved": "ground",
    "footprint-depaved-connected": "ground",
    "rewilded": "ground",
    "otherground": "ground",
    "scenario-rewilded": "ground",
    # envelopes
    "greenroof": "green roof",
    "brownroof": "brown roof",
    "livingfacade": "biofacade",
}

# Deploy-structure: split accepted voxels by intervention type.
# Deadwood is grouped with logs. Every intervention is 'full' support.
DEPLOY_SUBGROUP_ORDER = ["pole", "log", "upgrade"]
DEPLOY_INTERVENTION_TO_SUBGROUP = {
    "adapt-utility-pole": "pole",
    "translocate-log": "log",
    "translocate-deadwood": "log",
    "upgrade-feature": "upgrade",
}

# Recruit: split accepted tree nodes by the urban_element of their voxel_index.
# search_urban_elements array is dtype <U20, so "other street potential" is
# truncated to "other street potenti" — match both forms.
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


def _normalise_bio(val: str) -> str:
    """Normalise bioenvelope value for subgroup lookup: lowercase, no spaces/hyphens."""
    return str(val).strip().lower().replace(" ", "").replace("_", "")


def state_vtk_path(root: Path, site: str, scenario: str, year: int) -> Path:
    return root / "output" / "vtks" / site / f"{site}_{scenario}_1_yr{year}_state_with_indicators.vtk"


def node_df_path(root: Path, site: str, scenario: str, year: int) -> Path:
    return (
        root
        / "temp"
        / "interim-data"
        / site
        / f"{site}_{scenario}_1_nodeDF_{year}.csv"
    )


def output_csv_path(root: Path) -> Path:
    return root / "output" / "stats" / "proposal-decisions" / "proposal_decisions.csv"


def output_subgroups_csv_path(root: Path) -> Path:
    return root / "output" / "stats" / "proposal-decisions" / "proposal_decision_subgroups.csv"


def extract_nodedf_rows(root: Path, site: str, scenario: str, year: int) -> list[dict]:
    """Count nodeDF-based family decisions at tree-node granularity.

    Uses only `proposal-{family}_decision` and `proposal-{family}_intervention`.
    Ignores telemetry columns like `recruit_mechanism` (which conflates the
    deploy-structure deadwood placement into the recruit namespace).
    """
    path = node_df_path(root, site, scenario, year)
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

        rows.append(
            {
                "site": site,
                "scenario": scenario,
                "year": year,
                "proposal": family,
                "unit": "trees",
                "rejected": rejected,
                "accepted_full": accepted_full,
                "accepted_partial": accepted_partial,
            }
        )
    return rows


def extract_vtk_rows(
    root: Path,
    site: str,
    scenario: str,
    year: int,
    mesh: pv.PolyData | None = None,
) -> list[dict]:
    """Count VTK-based families' decisions at voxel granularity."""
    if mesh is None:
        path = state_vtk_path(root, site, scenario, year)
        if not path.exists():
            return []
        mesh = pv.read(path)
    rows: list[dict] = []

    for family, (decision_array, intervention_array, include_rej) in VTK_FAMILIES.items():
        if decision_array not in mesh.point_data.keys():
            raise KeyError(f"Missing point-data array {decision_array!r}")
        if intervention_array not in mesh.point_data.keys():
            raise KeyError(f"Missing point-data array {intervention_array!r}")

        decisions = np.asarray([str(v) for v in mesh.point_data[decision_array]])
        interventions = np.asarray([str(v) for v in mesh.point_data[intervention_array]])

        accepted_label = f"proposal-{family}_accepted"
        rejected_label = f"proposal-{family}_rejected"

        rejected = int((decisions == rejected_label).sum()) if include_rej else 0
        accepted_mask = decisions == accepted_label
        accepted_supports = np.array([INTERVENTION_SUPPORT.get(iv, "") for iv in interventions[accepted_mask]])
        accepted_full = int((accepted_supports == "full").sum())
        accepted_partial = int((accepted_supports == "partial").sum())

        rows.append(
            {
                "site": site,
                "scenario": scenario,
                "year": year,
                "proposal": family,
                "unit": "voxels",
                "rejected": rejected,
                "accepted_full": accepted_full,
                "accepted_partial": accepted_partial,
            }
        )
    return rows


def extract_colonise_subgroups(
    mesh: pv.PolyData, site: str, scenario: str, year: int
) -> list[dict]:
    """Break down colonise accepted voxels by bioenvelope subgroup.

    Each subgroup's support bucket is determined by its intervention
    (COLONISE_FULL_GROUND / COLONISE_FULL_ENVELOPE / COLONISE_PARTIAL_ENVELOPE),
    so we report the support bucket alongside the subgroup.
    """
    if "proposal_colonise" not in mesh.point_data.keys():
        return []
    if "scenario_bioEnvelope" not in mesh.point_data.keys():
        return []
    if "proposal_colonise_intervention" not in mesh.point_data.keys():
        return []

    decisions = np.asarray([str(v) for v in mesh.point_data["proposal_colonise"]])
    bio = np.asarray([str(v) for v in mesh.point_data["scenario_bioEnvelope"]])
    interventions = np.asarray([str(v) for v in mesh.point_data["proposal_colonise_intervention"]])

    accepted = decisions == "proposal-colonise_accepted"

    # Counts per subgroup, split by the support bucket the voxel's intervention lands in.
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
            rows.append(
                {
                    "site": site,
                    "scenario": scenario,
                    "year": year,
                    "proposal": "colonise",
                    "unit": "voxels",
                    "support_bucket": bucket,
                    "subgroup": sub,
                    "count": counts.get((bucket, sub), 0),
                }
            )
    return rows


def extract_deploy_subgroups(
    mesh: pv.PolyData, site: str, scenario: str, year: int
) -> list[dict]:
    """Break down deploy-structure accepted voxels by intervention type."""
    if "proposal_deploy_structure" not in mesh.point_data.keys():
        return []
    if "proposal_deploy_structure_intervention" not in mesh.point_data.keys():
        return []

    decisions = np.asarray([str(v) for v in mesh.point_data["proposal_deploy_structure"]])
    interventions = np.asarray([str(v) for v in mesh.point_data["proposal_deploy_structure_intervention"]])

    accepted = decisions == "proposal-deploy-structure_accepted"

    counts: dict[str, int] = {}
    for idx in np.where(accepted)[0]:
        sub = DEPLOY_INTERVENTION_TO_SUBGROUP.get(interventions[idx])
        if sub is None:
            continue
        counts[sub] = counts.get(sub, 0) + 1

    rows: list[dict] = []
    for sub in DEPLOY_SUBGROUP_ORDER:
        rows.append(
            {
                "site": site,
                "scenario": scenario,
                "year": year,
                "proposal": "deploy-structure",
                "unit": "voxels",
                "support_bucket": "full",
                "subgroup": sub,
                "count": counts.get(sub, 0),
            }
        )
    return rows


def extract_recruit_subgroups(
    root: Path,
    site: str,
    scenario: str,
    year: int,
    mesh: pv.PolyData | None,
) -> list[dict]:
    """Break down recruit accepted tree nodes by urban_element subgroup.

    Uses nodeDF for tree-level counts (matching `extract_nodedf_rows`), then
    looks up each accepted node's voxel_index against the VTK's
    `search_urban_elements` array to classify.
    """
    df_path = node_df_path(root, site, scenario, year)
    if not df_path.exists():
        return []
    if mesh is None:
        vtk_path = state_vtk_path(root, site, scenario, year)
        if not vtk_path.exists():
            return []
        mesh = pv.read(vtk_path)
    if "search_urban_elements" not in mesh.point_data.keys():
        return []

    df = pd.read_csv(df_path, low_memory=False)
    if "proposal-recruit_decision" not in df.columns:
        return []
    if "voxel_index" not in df.columns:
        return []

    accepted = df[df["proposal-recruit_decision"] == "proposal-recruit_accepted"].copy()
    if accepted.empty:
        return [
            {
                "site": site,
                "scenario": scenario,
                "year": year,
                "proposal": "recruit",
                "unit": "trees",
                "support_bucket": bucket,
                "subgroup": sub,
                "count": 0,
            }
            for bucket in ("full", "partial")
            for sub in RECRUIT_SUBGROUP_ORDER
        ]

    urban = np.asarray([str(v) for v in mesh.point_data["search_urban_elements"]])
    voxel_idx = accepted["voxel_index"].astype(int).to_numpy()
    accepted["_urban"] = urban[voxel_idx]
    accepted["_subgroup"] = accepted["_urban"].map(RECRUIT_URBAN_TO_SUBGROUP).fillna("other")
    accepted["_bucket"] = (
        accepted["proposal-recruit_intervention"].map(INTERVENTION_SUPPORT).fillna("")
    )

    counts = accepted.groupby(["_bucket", "_subgroup"]).size().to_dict()

    rows: list[dict] = []
    for bucket in ("full", "partial"):
        for sub in RECRUIT_SUBGROUP_ORDER:
            rows.append(
                {
                    "site": site,
                    "scenario": scenario,
                    "year": year,
                    "proposal": "recruit",
                    "unit": "trees",
                    "support_bucket": bucket,
                    "subgroup": sub,
                    "count": int(counts.get((bucket, sub), 0)),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract V4 proposal decisions (full/partial/rejected) per state.")
    parser.add_argument("--root", required=True, help="Run output root, e.g. _data-refactored/model-outputs/generated-states/4.9")
    parser.add_argument("--sites", type=str, default=",".join(SITES))
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIOS))
    parser.add_argument("--years", type=str, default=",".join(str(y) for y in YEARS))
    args = parser.parse_args()

    root = Path(args.root)
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    all_rows: list[dict] = []
    sub_rows: list[dict] = []
    for site in sites:
        for scenario in scenarios:
            for year in years:
                nodedf_rows = extract_nodedf_rows(root, site, scenario, year)
                if not nodedf_rows:
                    print(f"  skip {site}/{scenario}/yr{year}: missing nodeDF")
                all_rows.extend(nodedf_rows)

                vtk_path = state_vtk_path(root, site, scenario, year)
                mesh = pv.read(vtk_path) if vtk_path.exists() else None
                if mesh is None:
                    print(f"  skip {site}/{scenario}/yr{year}: missing VTK")
                    continue

                vtk_rows = extract_vtk_rows(root, site, scenario, year, mesh=mesh)
                all_rows.extend(vtk_rows)

                sub_rows.extend(extract_colonise_subgroups(mesh, site, scenario, year))
                sub_rows.extend(extract_deploy_subgroups(mesh, site, scenario, year))
                sub_rows.extend(
                    extract_recruit_subgroups(root, site, scenario, year, mesh=mesh)
                )

    out_path = output_csv_path(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows, columns=CSV_COLUMNS).to_csv(out_path, index=False)
    print(f"Wrote {len(all_rows)} rows -> {out_path}")

    sub_path = output_subgroups_csv_path(root)
    pd.DataFrame(sub_rows, columns=SUBGROUP_CSV_COLUMNS).to_csv(sub_path, index=False)
    print(f"Wrote {len(sub_rows)} subgroup rows -> {sub_path}")


if __name__ == "__main__":
    main()
