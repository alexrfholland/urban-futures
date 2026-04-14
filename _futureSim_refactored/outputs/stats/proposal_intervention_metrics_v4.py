"""
V4 proposal intervention voxel count extractor.

Reads the five `proposal_*_intervention` point-data arrays from each V4 state
VTK, counts voxels per intervention, and writes one CSV per state under
`{root}/output/stats/per-state/{site}/`.

USAGE:
    uv run python _futureSim_refactored/outputs/stats/proposal_intervention_metrics_v4.py \
        --root _data-refactored/model-outputs/generated-states/4.9

Output schema (one row per (site, scenario, year, proposal, intervention)):
    site,scenario,year,proposal,intervention,support,count

Zero-count rows are emitted for every known intervention in each family so the
plotter stacks stably across years.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT.parent))

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


SITES = ["trimmed-parade", "city", "uni"]
SCENARIOS = ["positive", "trending"]
YEARS = [0, 1, 10, 30, 60, 90, 120, 150, 180]

# Proposal family → VTK array name
FAMILY_VTK_ARRAY = {
    "decay": "proposal_decay_intervention",
    "release-control": "proposal_release_control_intervention",
    "recruit": "proposal_recruit_intervention",
    "colonise": "proposal_colonise_intervention",
    "deploy-structure": "proposal_deploy_structure_intervention",
}

# Proposal family → ordered list of known interventions (drives row emission)
FAMILY_INTERVENTIONS = {
    "decay": [DECAY_FULL, DECAY_PARTIAL],
    "release-control": [RELEASECONTROL_FULL, RELEASECONTROL_PARTIAL],
    "recruit": [RECRUIT_FULL, RECRUIT_PARTIAL],
    "colonise": [COLONISE_FULL_GROUND, COLONISE_FULL_ENVELOPE, COLONISE_PARTIAL_ENVELOPE],
    "deploy-structure": [DEPLOY_FULL_POLE, DEPLOY_FULL_LOG, DEPLOY_FULL_UPGRADE],
}

CSV_COLUMNS = ["site", "scenario", "year", "proposal", "intervention", "support", "count"]


def state_vtk_path(root: Path, site: str, scenario: str, year: int) -> Path:
    return root / "output" / "vtks" / site / f"{site}_{scenario}_1_yr{year}_state_with_indicators.vtk"


def per_state_csv_path(root: Path, site: str, scenario: str, year: int) -> Path:
    return (
        root
        / "output"
        / "stats"
        / "per-state"
        / site
        / f"{site}_{scenario}_1_yr{year}_v4_interventions.csv"
    )


def count_interventions(mesh: pv.PolyData, site: str, scenario: str, year: int) -> list[dict]:
    rows: list[dict] = []
    for family, array_name in FAMILY_VTK_ARRAY.items():
        if array_name not in mesh.point_data.keys():
            raise KeyError(f"Missing point-data array {array_name!r} in VTK")
        values = np.asarray(mesh.point_data[array_name])
        # String arrays from VTK may load as bytes or objects — normalise to str
        values = np.array([str(v) for v in values])

        for intervention in FAMILY_INTERVENTIONS[family]:
            count = int(np.sum(values == intervention))
            rows.append(
                {
                    "site": site,
                    "scenario": scenario,
                    "year": year,
                    "proposal": family,
                    "intervention": intervention,
                    "support": INTERVENTION_SUPPORT[intervention],
                    "count": count,
                }
            )
    return rows


def process_state(root: Path, site: str, scenario: str, year: int) -> Path | None:
    vtk_path = state_vtk_path(root, site, scenario, year)
    if not vtk_path.exists():
        print(f"  skip {site}/{scenario}/yr{year}: missing VTK")
        return None

    mesh = pv.read(vtk_path)
    rows = count_interventions(mesh, site, scenario, year)

    csv_path = per_state_csv_path(root, site, scenario, year)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract V4 proposal intervention voxel counts per state.")
    parser.add_argument("--root", required=True, help="Run output root, e.g. _data-refactored/model-outputs/generated-states/4.9")
    parser.add_argument("--sites", type=str, default=",".join(SITES))
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIOS))
    parser.add_argument("--years", type=str, default=",".join(str(y) for y in YEARS))
    args = parser.parse_args()

    root = Path(args.root)
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    written: list[Path] = []
    for site in sites:
        for scenario in scenarios:
            for year in years:
                path = process_state(root, site, scenario, year)
                if path is not None:
                    written.append(path)

    print(f"Wrote {len(written)} intervention CSVs under {root}/output/stats/per-state/")


if __name__ == "__main__":
    main()
