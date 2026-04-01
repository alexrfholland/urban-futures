# Scenario Engine V3 Status

## Candidate Roots

- scenario-output root: [data/revised/final-v3](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v3)
- engine-output root: [_data-refactored/v3engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs)
- statistics root: [_statistics-refactored-v3](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v3)

Current canonical references remain unchanged:

- scenario-output root: [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
- engine-output root: [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)

## Required Template Root

V3 candidate runs must explicitly use the approved canonical deadwood template bundle:

- [_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

Required settings:

- `fallens_use = nonpre-direct`
- `snags_use = elm-snags-old`
- include `decayed-small-fallen`
- `voxel_size = 1`

The default loader root `data/revised/trees` is not an acceptable V3 validation root.

## Required Run Metadata

Every v3 candidate run should persist the full run-config block, not just the template root.

Required env/config record:

- `TREE_TEMPLATE_ROOT`
- `REFACTOR_SCENARIO_OUTPUT_ROOT`
- `REFACTOR_ENGINE_OUTPUT_ROOT`
- `REFACTOR_STATISTICS_ROOT`
- `EXPORT_ALL_POINTDATA_VARIABLES`

## Runtime Split

- v3 engine runtime: [_code-refactored/refactor_code/scenario/engine_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v3.py)
- scenario runner now imports v3: [a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py)
- v3 proposal arrays are computed during scenario VTK generation and carried forward into `urban_features` / `state_with_indicators`: [a_scenario_generateVTKs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_generateVTKs.py)
- v3 proposal arrays are recomputed on the enriched augmented VTKs, which are the main validation artifact: [a_info_gather_capabilities.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_gather_capabilities.py)
- v3 proposal render view: [render_forest_size_views.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/render_forest_size_views.py)

## Schema Additions

Tree-level v3 fields are present in parallel with legacy v2 fields:

- `under-node-treatment`
- `control_reached`
- `proposal-decay_*`
- `proposal-release-control_*`
- `proposal-recruit_*`
- `proposal-colonise_*`
- `proposal-deploy-structure_*`

Augmented VTKs now carry:

- `proposal_decayV3`
- `proposal_release_controlV3`
- `proposal_coloniseV3`
- `proposal_recruitV3`
- `proposal_deploy_structureV3`
- and their `*_intervention` companions

Redundant support alias families have now been removed from the live v3 schema.

In the current v3 path, the proposal families are:

- `proposal-decay`
- `proposal-release-control`
- `proposal-recruit`
- `proposal-colonise`
- `proposal-deploy-structure`

In the dataframe, each of those exists as a `*_decision` and `*_intervention` pair in [engine_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v3.py):

- `proposal-decay_decision`
- `proposal-decay_intervention`
- `proposal-release-control_decision`
- `proposal-release-control_intervention`
- `proposal-recruit_decision`
- `proposal-recruit_intervention`
- `proposal-colonise_decision`
- `proposal-colonise_intervention`
- `proposal-deploy-structure_decision`
- `proposal-deploy-structure_intervention`

Release-control also carries its timing fields:

- `proposal-release-control_target_years`
- `proposal-release-control_years`

On the VTK side, the v3 point-data outputs are the same five proposal families:

- `proposal_decayV3`
- `proposal_release_controlV3`
- `proposal_recruitV3`
- `proposal_coloniseV3`
- `proposal_deploy_structureV3`

with matching intervention arrays:

- `proposal_decayV3_intervention`
- `proposal_release_controlV3_intervention`
- `proposal_recruitV3_intervention`
- `proposal_coloniseV3_intervention`
- `proposal_deploy_structureV3_intervention`

The older `proposal_{proposal name}` point-data fields have been left in place for now.

- these are the v2-engine proposal arrays
- they are being retained while v2 vs v3 comparison is still in progress

## Quick Verification Status

Quick verification has been completed for the planned subset:

- sites: `trimmed-parade`, `city`
- scenarios: `positive`, `trending`
- years: `0`, `30`, `180`

Saved quick verification artifacts:

- summary: [_data-refactored/v3engine_outputs/validation/verification_summary.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/verification_summary.md)
- counts: [_data-refactored/v3engine_outputs/validation/verification_counts.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/verification_counts.csv)
- repeatability: [_data-refactored/v3engine_outputs/validation/quick_repeatability.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/quick_repeatability.md)

Quick verification result is split:

- repeatability checks passed
- V3 vocabulary checks passed
- quick file-presence checks passed
- quick render counts passed
- quick pathway sanity checks passed for the sensitive cells
- initial template-root validity failed for the first quick run
- targeted reruns have corrected specific cases, but the full quick subset has not yet been re-derived end to end under the approved template root

## Current Checkpoint

Current local checkpoint after the latest v3 cleanup and targeted reruns:

- `proposal-release-control_support` has been removed from the v3 schema.
- The remaining legacy support aliases have also been removed from the live v3 output schema:
  - `decay_support`
  - `release_control_support`
  - `recruit_support`
  - `colonise_support`
  - `deploy_structure_support`
- V3 now writes directly to the `proposal-*decision` and `proposal-*intervention` fields.
- `_refresh_schema(...)` still reads those legacy columns if they appear in older files, backfills the current intervention fields, and then drops them.
- Decayed rows are no longer dropped from the simulation.
- When the old removal threshold is reached, they now transition to `size = gone`.
- `gone` rows stay in the dataframe but are filtered out of voxel/node export.
- `gone` is a new v3 candidate-state behavior. It is not an inherited canonical v2 behavior and should be treated as a semantic divergence that still needs verification.
- `trimmed-parade / positive / yr180` has been re-run through the canonical manager path using the approved edited template root.
- The refreshed validation renders for that case have also been generated.

Current `trimmed-parade / positive / yr180` `treeDF` counts:

- v2: `small 43`, `fallen 311`, `decayed 64`
- current v3: `small 84`, `fallen 313`, `decayed 7`

Current interpretation:

- `fallen` is now effectively back at canonical v2 level for this case.
- `small` remains materially higher than v2.
- `decayed` is now much lower than v2 because long-lived decayed rows now transition to `gone` instead of remaining in the visible/exported deadwood states.

## Open State

V3 is still a candidate engine.

It has not been promoted.

## Known Issues / To Do

- the initial quick V3 export loaded templates from the default loader root instead of the approved edited variant bundle
- that means the first quick deadwood renders are not valid as a canonical v2 visual parity check
- this should be treated as a verification validity failure on the first quick run, not as the current live template configuration
- targeted reruns have corrected `trimmed-parade / positive / yr180`, but the full quick subset still needs to be re-derived under the approved root before the quick bundle can be treated as fully valid
- `trimmed-parade / yr180 / Tree / Reproduce` currently shows a compressed `positive` vs `trending` gap compared with canonical v2.
- Current quick-test comparison:
  - v2: `positive = 188,539` voxels (`88.4%` baseline), `trending = 37` voxels (`0.0%` baseline)
  - v3 candidate: `positive = 80,486` voxels (`37.7%` baseline), `trending = 16,063` voxels (`7.5%` baseline)
- Observed cause in the current candidate: `trending` is no longer near-zero because many `park-tree` rows are receiving `footprint-depaved` ground treatment, which creates `Tree.generations.grassland` support and compresses the `positive` vs `trending` divergence from both sides.
- Follow-up:
  - keep the ground-clearing behavior where it is needed
  - add a condition so `trending` does not assign `footprint-depaved` to `park-tree` rows in this pathway
  - re-run the Parade quick subset and compare `Tree / Reproduce`, `Tree / Communicate`, and the linked ground indicators against canonical v2
- V3 proposal point-data is currently being expressed on the assessed site voxels, not transferred back onto the tree/node rows as part of the dataframe attribute handoff.
- Follow-up:
  - specify that the proposal/intervention columns are transferred with the other dataframe attributes during the tree/node-to-voxel export path
  - do not treat the current VTK proposal arrays as tree/node-row parity until that transfer is implemented
- `proposal_deploy_structureV3` is still using voxel-side fallback for `upgrade-feature` and `adapt-utility-pole`.
- Follow-up is to standardise the path for the proxy logic.
- `proposal-deploy-structure_accepted` + `upgrade-feature` should be tracked as a proxy through `(~forest_precolonial) & indicator_Bird_self_peeling`, using peeling bark in elms as a proxy for artificial bark and thus upgraded structures.
- Fallback for artificial trees might be duplicated; investigate.

Full verification is still required for:

- all sites: `trimmed-parade`, `city`, `uni`
- all years: `0, 10, 30, 60, 90, 120, 150, 180`
- full pathway tables
- direct `v2 vs v3` delta outputs
