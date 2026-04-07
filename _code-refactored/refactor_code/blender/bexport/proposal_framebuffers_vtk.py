from __future__ import annotations

"""
Build Blender-ready proposal framebuffer point-data arrays from a final v3 VTK.

Purpose:
- read the five v3 proposal decision/intervention point-data pairs
- collapse each proposal family into one integer-coded framebuffer array
- write arrays that use the same mapping as the dataframe helper

Output point-data arrays:
- blender_proposal-decay
- blender_proposal-release-control
- blender_proposal-recruit
- blender_proposal-colonise
- blender_proposal-deploy-structure

General encoding pattern:
- -1 = accepted with no intervention allocated yet
- 0 = not-assessed
- 1 = rejected
- higher values = accepted intervention variants for that proposal family

Important note:
- this script keeps one framebuffer per proposal family
- it does not merge all proposal families into one channel
- overlaps between different proposal families can still exist on the same voxel
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from refactor_code.blender.bexport.proposal_framebuffers import DEFAULT_OUTPUT_COLUMNS, FRAMEBUFFER_STATE_MAPPINGS


VTK_PROPOSAL_FAMILIES = [
    ("proposal-decay", "proposal_decayV3", "proposal_decayV3_intervention"),
    ("proposal-release-control", "proposal_release_controlV3", "proposal_release_controlV3_intervention"),
    ("proposal-recruit", "proposal_recruitV3", "proposal_recruitV3_intervention"),
    ("proposal-colonise", "proposal_coloniseV3", "proposal_coloniseV3_intervention"),
    ("proposal-deploy-structure", "proposal_deploy_structureV3", "proposal_deploy_structureV3_intervention"),
]


def _normalized_array(values_by_name, name: str, fallback: str) -> np.ndarray:
    if name not in values_by_name:
        raise KeyError(f"Missing required point-data array: {name}")
    values = np.asarray(values_by_name[name]).astype(str)
    values[values == "nan"] = fallback
    return values


def build_blender_proposal_framebuffer_arrays(values_by_name) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}

    first_key = VTK_PROPOSAL_FAMILIES[0][1]
    if first_key not in values_by_name:
        raise KeyError(f"Missing required point-data array: {first_key}")
    point_count = len(np.asarray(values_by_name[first_key]))

    for family, decision_name, intervention_name in VTK_PROPOSAL_FAMILIES:
        decisions = _normalized_array(values_by_name, decision_name, "not-assessed")
        interventions = _normalized_array(values_by_name, intervention_name, "none")
        combined = np.full(point_count, "not-assessed", dtype="<U32")

        accepted_mask = np.char.find(np.char.lower(decisions), "_accepted") >= 0
        rejected_mask = np.char.find(np.char.lower(decisions), "_rejected") >= 0
        not_assessed_mask = decisions == "not-assessed"
        known_mask = accepted_mask | rejected_mask | not_assessed_mask
        if (~known_mask).any():
            unknown = sorted(np.unique(decisions[~known_mask]).tolist())
            raise ValueError(f"Unexpected {family} decision values: {unknown}")

        combined[rejected_mask] = "rejected"

        active_intervention_mask = accepted_mask & (interventions != "none")
        combined[active_intervention_mask] = interventions[active_intervention_mask]
        accepted_none_mask = accepted_mask & (interventions == "none")
        combined[accepted_none_mask] = "accepted-no-intervention"

        rejected_or_unset_with_intervention = (rejected_mask | not_assessed_mask) & (interventions != "none")
        if rejected_or_unset_with_intervention.any():
            bad_rows = int(rejected_or_unset_with_intervention.sum())
            raise ValueError(
                f"{family} has {bad_rows} points with intervention data on rejected/not-assessed decisions"
            )

        mapping = FRAMEBUFFER_STATE_MAPPINGS[family]
        missing_states = sorted(set(np.unique(combined).tolist()) - set(mapping.keys()))
        if missing_states:
            raise ValueError(f"Unexpected unmapped {family} framebuffer states: {missing_states}")
        output[DEFAULT_OUTPUT_COLUMNS[family]] = np.fromiter(
            (mapping[state] for state in combined),
            dtype=np.int16,
            count=point_count,
        )

    return output


def build_blender_proposal_framebuffer_pointdata(mesh: pv.PolyData) -> dict[str, np.ndarray]:
    return build_blender_proposal_framebuffer_arrays(mesh.point_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append Blender proposal framebuffer int arrays to a VTK using the v3 proposal point-data arrays."
    )
    parser.add_argument("--input", required=True, help="Input VTK path.")
    parser.add_argument("--output", required=True, help="Output VTK path.")
    parser.add_argument("--mapping-output", help="Optional JSON path for the per-framebuffer integer mappings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    mesh = pv.read(input_path)
    framebuffer_arrays = build_blender_proposal_framebuffer_pointdata(mesh)
    for name, values in framebuffer_arrays.items():
        mesh.point_data[name] = values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(output_path)

    if args.mapping_output:
        mapping_output_path = Path(args.mapping_output)
        mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
        mappings = {
            DEFAULT_OUTPUT_COLUMNS[family]: mapping
            for family, mapping in FRAMEBUFFER_STATE_MAPPINGS.items()
        }
        mapping_output_path.write_text(json.dumps(mappings, indent=2) + "\n")


if __name__ == "__main__":
    main()
