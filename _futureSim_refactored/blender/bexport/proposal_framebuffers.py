from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_futureSim_refactored")
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
    RECRUIT_FULL,
    RECRUIT_PARTIAL,
    RELEASECONTROL_FULL,
    RELEASECONTROL_PARTIAL,
)

"""
Build Blender-ready proposal framebuffer columns from a scenario node dataframe.

Purpose:
- collapse each proposal family's `*_decision` + `*_intervention` pair into one
  integer-coded framebuffer column
- keep one framebuffer per proposal family
- use a stable shared mapping so dataframe and VTK exports can be interpreted
  the same way in Blender

Output columns:
- blender_proposal-decay
- blender_proposal-release-control
- blender_proposal-recruit
- blender_proposal-colonise
- blender_proposal-deploy-structure

General encoding pattern:
- 10 = accepted with no intervention allocated yet
- 0 = not-assessed
- 1 = rejected
- higher values = accepted intervention variants for that proposal family

Important note:
- old decay rows created before the brace-collapse cleanup can carry
  `proposal-decay_accepted + none + replacement_reason=brace-collapse`
- when `clean_legacy_brace_collapse=True`, those rows are normalized to
  `rejected`
- when cleanup is disabled, that condition is ignored and no extra legacy-only
  framebuffer code is emitted
"""

import argparse
import json

import pandas as pd


PROPOSAL_FAMILIES = [
    ("proposal-decay", "proposal-decay_decision", "proposal-decay_intervention"),
    ("proposal-release-control", "proposal-release-control_decision", "proposal-release-control_intervention"),
    ("proposal-recruit", "proposal-recruit_decision", "proposal-recruit_intervention"),
    ("proposal-colonise", "proposal-colonise_decision", "proposal-colonise_intervention"),
    ("proposal-deploy-structure", "proposal-deploy-structure_decision", "proposal-deploy-structure_intervention"),
]

DEFAULT_OUTPUT_COLUMNS = {
    family: f"blender_{family}"
    for family, _, _ in PROPOSAL_FAMILIES
}

FRAMEBUFFER_STATE_MAPPINGS = {
    "proposal-decay": {
        "accepted-no-intervention": 10,
        "not-assessed": 0,
        "rejected": 1,
        DECAY_FULL: 2,
        DECAY_PARTIAL: 3,
    },
    "proposal-release-control": {
        "accepted-no-intervention": 10,
        "not-assessed": 0,
        "rejected": 1,
        RELEASECONTROL_PARTIAL: 2,
        RELEASECONTROL_FULL: 3,
    },
    "proposal-recruit": {
        "accepted-no-intervention": 10,
        "not-assessed": 0,
        "rejected": 1,
        RECRUIT_PARTIAL: 2,
        RECRUIT_FULL: 3,
    },
    "proposal-colonise": {
        "accepted-no-intervention": 10,
        "not-assessed": 0,
        "rejected": 1,
        COLONISE_FULL_GROUND: 2,
        COLONISE_FULL_ENVELOPE: 3,
        COLONISE_PARTIAL_ENVELOPE: 4,
    },
    "proposal-deploy-structure": {
        "accepted-no-intervention": 10,
        "not-assessed": 0,
        "rejected": 1,
        DEPLOY_FULL_POLE: 2,
        DEPLOY_FULL_LOG: 3,
        DEPLOY_FULL_UPGRADE: 4,
    },
}


def _normalized_series(df: pd.DataFrame, column: str, fallback: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Missing required column: {column}")
    values = df[column].astype("object")
    values = values.where(values.notna(), fallback)
    return values.astype(str)


def _combine_family_states(
    df: pd.DataFrame,
    *,
    family: str,
    decision_column: str,
    intervention_column: str,
    clean_legacy_brace_collapse: bool,
) -> pd.Series:
    decisions = _normalized_series(df, decision_column, "not-assessed")
    interventions = _normalized_series(df, intervention_column, "none")
    combined = pd.Series("not-assessed", index=df.index, dtype="object")

    accepted_mask = decisions.str.endswith("_accepted")
    rejected_mask = decisions.str.endswith("_rejected")
    not_assessed_mask = decisions.eq("not-assessed")
    known_mask = accepted_mask | rejected_mask | not_assessed_mask
    if (~known_mask).any():
        unknown = sorted(decisions.loc[~known_mask].unique().tolist())
        raise ValueError(f"Unexpected {family} decision values: {unknown}")

    combined.loc[rejected_mask] = "rejected"

    active_intervention_mask = accepted_mask & interventions.ne("none")
    combined.loc[active_intervention_mask] = interventions.loc[active_intervention_mask]

    accepted_none_mask = accepted_mask & interventions.eq("none")
    if accepted_none_mask.any() and family == "proposal-decay" and clean_legacy_brace_collapse:
        replacement_reasons = _normalized_series(df, "replacement_reason", "none")
        legacy_brace_collapse_mask = accepted_none_mask & replacement_reasons.eq("brace-collapse")
        combined.loc[legacy_brace_collapse_mask] = "rejected"
        accepted_none_mask = accepted_none_mask & ~legacy_brace_collapse_mask

    combined.loc[accepted_none_mask] = "accepted-no-intervention"

    rejected_or_unset_with_intervention = (rejected_mask | not_assessed_mask) & interventions.ne("none")
    if rejected_or_unset_with_intervention.any():
        bad_rows = int(rejected_or_unset_with_intervention.sum())
        raise ValueError(
            f"{family} has {bad_rows} rows with intervention data on rejected/not-assessed decisions"
        )

    return combined


def proposal_framebuffer_mappings() -> dict[str, dict[str, int]]:
    return {
        DEFAULT_OUTPUT_COLUMNS[family]: mapping.copy()
        for family, mapping in FRAMEBUFFER_STATE_MAPPINGS.items()
    }


def build_blender_proposal_framebuffer_state_columns(
    df: pd.DataFrame,
    *,
    clean_legacy_brace_collapse: bool = True,
) -> pd.DataFrame:
    """
    Collapse proposal decision + intervention pairs into one categorical column
    per proposal family for Blender framebuffer transfer.

    Output columns:
    - blender_proposal-decay
    - blender_proposal-release-control
    - blender_proposal-recruit
    - blender_proposal-colonise
    - blender_proposal-deploy-structure

    Notes:
    - Missing / NaN decision values are treated as `not-assessed`.
    - Missing / NaN intervention values are treated as `none`.
    - Older decay rows with `proposal-decay_accepted + none + replacement_reason=brace-collapse`
      can be normalized to `rejected` by leaving `clean_legacy_brace_collapse=True`.
    - No extra legacy-only framebuffer code is emitted.
    """

    output: dict[str, pd.Series] = {}
    for family, decision_column, intervention_column in PROPOSAL_FAMILIES:
        output_column = DEFAULT_OUTPUT_COLUMNS[family]
        output[output_column] = _combine_family_states(
            df,
            family=family,
            decision_column=decision_column,
            intervention_column=intervention_column,
            clean_legacy_brace_collapse=clean_legacy_brace_collapse,
        )
    return pd.DataFrame(output, index=df.index)


def build_blender_proposal_framebuffer_columns(
    df: pd.DataFrame,
    *,
    clean_legacy_brace_collapse: bool = True,
) -> pd.DataFrame:
    """
    Return one integer framebuffer column per proposal family.
    """
    state_df = build_blender_proposal_framebuffer_state_columns(
        df,
        clean_legacy_brace_collapse=clean_legacy_brace_collapse,
    )
    output: dict[str, pd.Series] = {}
    for family, _, _ in PROPOSAL_FAMILIES:
        output_column = DEFAULT_OUTPUT_COLUMNS[family]
        mapping = FRAMEBUFFER_STATE_MAPPINGS[family]
        encoded = state_df[output_column].map(mapping)
        if encoded.isna().any():
            unknown = sorted(state_df.loc[encoded.isna(), output_column].astype(str).unique().tolist())
            raise ValueError(f"Unexpected unmapped {family} framebuffer states: {unknown}")
        output[output_column] = encoded.astype("int16")
    return pd.DataFrame(output, index=df.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one Blender framebuffer-state column per proposal family from a nodeDF CSV."
    )
    parser.add_argument("--input", required=True, help="Input nodeDF CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append the Blender proposal columns to the original dataframe instead of writing only the five new columns.",
    )
    parser.add_argument(
        "--mapping-output",
        help="Optional JSON path for the per-framebuffer integer mappings.",
    )
    parser.add_argument(
        "--state-output",
        help="Optional CSV path for the string-state framebuffer columns before integer encoding.",
    )
    parser.add_argument(
        "--no-clean-legacy-brace-collapse",
        action="store_true",
        help="Skip the older decay brace-collapse cleanup instead of normalizing those rows to `rejected`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    blender_state_columns = build_blender_proposal_framebuffer_state_columns(
        df,
        clean_legacy_brace_collapse=not args.no_clean_legacy_brace_collapse,
    )
    blender_columns = build_blender_proposal_framebuffer_columns(
        df,
        clean_legacy_brace_collapse=not args.no_clean_legacy_brace_collapse,
    )

    if args.state_output:
        state_output_path = Path(args.state_output)
        state_output_path.parent.mkdir(parents=True, exist_ok=True)
        state_output_df = pd.concat([df, blender_state_columns], axis=1) if args.append else blender_state_columns
        state_output_df.to_csv(state_output_path, index=False)

    output_df = pd.concat([df, blender_columns], axis=1) if args.append else blender_columns
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    if args.mapping_output:
        mapping_output_path = Path(args.mapping_output)
        mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_output_path.write_text(json.dumps(proposal_framebuffer_mappings(), indent=2) + "\n")


if __name__ == "__main__":
    main()
