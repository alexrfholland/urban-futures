# Scenario Engine V3 Validation

## Candidate Roots Checked

- scenario-output root: [data/revised/final-v3](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v3)
- engine-output root: [_data-refactored/v3engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs)
- statistics root: [_statistics-refactored-v3](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v3)

Current canonical comparison target remains:

- scenario-output root: [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
- engine-output root: [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)

## Template Root Requirement

V3 validation is not valid unless the run explicitly records the approved template root.

Approved template root:

- [_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

Required settings:

- `fallens_use = nonpre-direct`
- `snags_use = elm-snags-old`
- `decayed-small-fallen`
- `voxel_size = 1`

Validation must fail if:

- no template-root metadata is present
- the run used the loader default `data/revised/trees`
- the run used a different tree-variant root without explicit approval

Before comparing V3 renders or deadwood voxel counts against canonical v2, confirm the candidate run used this template root.

## Required Run Metadata

Validation should record the full candidate run-config block, not just the template root.

Required env/config record:

- `TREE_TEMPLATE_ROOT`
- `REFACTOR_SCENARIO_OUTPUT_ROOT`
- `REFACTOR_ENGINE_OUTPUT_ROOT`
- `REFACTOR_STATISTICS_ROOT`
- `EXPORT_ALL_POINTDATA_VARIABLES`

## Quick Verification

Quick verification used:

- sites: `trimmed-parade`, `city`
- scenarios: `positive`, `trending`
- years: `0`, `30`, `180`

Saved quick verification artifacts:

- summary: [_data-refactored/v3engine_outputs/validation/verification_summary.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/verification_summary.md)
- counts CSV: [_data-refactored/v3engine_outputs/validation/verification_counts.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/verification_counts.csv)
- repeatability note: [_data-refactored/v3engine_outputs/validation/quick_repeatability.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/quick_repeatability.md)

Quick verification result is split:

- repeatability passed
- all expected quick-subset `treeDF`, `urban_features`, and augmented `state_with_indicators` files exist
- all quick-subset V3 proposal arrays exist on augmented VTKs
- no unexpected V3 proposal labels were found
- quick render counts passed for `classic`, `merged`, `proposal-hybrid`, and `proposal-hybrid-v3`
- quick year-180 sensitive-cell checks kept `positive > trending`
- initial template-root validity failed for the first quick run
- targeted reruns have corrected specific cases, but the full quick subset has not yet been re-derived end to end under the approved template root

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

Release-control also carries:

- `proposal-release-control_target_years`
- `proposal-release-control_years`

On the VTK side, the v3 point-data outputs are:

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

## Current Checkpoint

Current local checkpoint after the latest v3 cleanup and targeted reruns:

- The live v3 schema no longer writes:
  - `decay_support`
  - `release_control_support`
  - `recruit_support`
  - `colonise_support`
  - `deploy_structure_support`
- The engine now writes directly to the `proposal-*intervention` fields.
- `_refresh_schema(...)` still accepts those legacy columns from older files, transfers their values into the current proposal intervention columns, and drops them.
- Decayed rows now end at `size = gone` instead of being dropped from the dataframe.
- `gone` rows stay in `treeDF` but do not export into node/voxel outputs.
- `gone` is a new v3 candidate-state behavior. It is not inherited canonical v2 behavior and should be treated as a semantic divergence to verify.
- `trimmed-parade / positive / yr180` was re-run through the canonical manager path with the approved edited template root.
- Refreshed validation renders were generated for that case.

Current `trimmed-parade / positive / yr180` `treeDF` size counts:

- v2: `small 43`, `fallen 311`, `decayed 64`
- current v3: `small 84`, `fallen 313`, `decayed 7`

Current interpretation:

- `fallen` is back near canonical v2 level for this case.
- `small` remains materially elevated.
- `decayed` is now much lower because rows that would previously have remained decayed now move to `gone` and disappear from exported outputs.

## Repeatability

The saved-state export was re-derived twice for:

- `trimmed-parade / positive / yr30`
- `city / positive / yr30`

Both checks passed with:

- matching point counts
- `0` mismatches in `scenario_outputs`
- `0` mismatches in `search_bioavailable`
- `0` mismatches in `search_design_action`
- `0` mismatches in `search_urban_elements`
- `0` mismatches in all V3 proposal arrays and intervention arrays

## Full Verification

Full verification has not been completed yet.

Still required:

- all sites: `trimmed-parade`, `city`, `uni`
- all years: `0, 10, 30, 60, 90, 120, 150, 180`
- full augmented VTK count checks
- full render count checks
- full V3 pathway table
- direct `v2 vs v3` delta outputs

Current status for the required minimum record:

- candidate roots checked: yes
- quick verification subset recorded: yes
- repeatability passed: yes
- full file counts passed: not yet checked
- `v2 vs v3` delta location: not yet generated

## Known Issues / To Do

- the initial quick V3 export was run against the loader default template root `data/revised/trees`, not the approved edited deadwood variant root
- that means the first quick deadwood renders are not valid as a visual parity check against canonical v2
- this should be treated as a verification validity failure on the first quick run, not as the current live template configuration
- targeted reruns have corrected specific cases, but the full quick subset still needs to be re-derived under the approved root before the quick bundle can be treated as fully valid
- `trimmed-parade / yr180 / Tree / Reproduce` requires follow-up before promotion.
- Current quick-test comparison:
  - v2: `positive = 188,539` voxels (`88.4%` baseline), `trending = 37` voxels (`0.0%` baseline)
  - v3 candidate: `positive = 80,486` voxels (`37.7%` baseline), `trending = 16,063` voxels (`7.5%` baseline)
- Current interpretation: the v3 candidate is allowing `trending` `park-tree` rows to contribute `footprint-depaved` ground treatment, so `Tree.generations.grassland` is no longer near-zero in `trending`.
- Follow-up requirement: keep the required ground-clearing logic, but add a condition so `trending` does not give `park-tree` rows `footprint-depaved` in this path, then re-check Parade `yr180` ground-linked indicators.
- Current proposal-field limitation: the V3 `proposal_*V3` and `*_intervention` arrays are currently being applied at the assessed site-voxel level only, not transferred back onto the tree/node dataframe rows during attribute handoff.
- Follow-up requirement: specify that these proposal/intervention columns are transferred with the other dataframe attributes in the tree/node-to-voxel export path before using VTK proposal counts as a direct proxy for tree/node proposal counts.
- `proposal_deploy_structureV3` is still using voxel-side fallback for `upgrade-feature` and `adapt-utility-pole`.
- Follow-up is to standardise the path for the proxy logic.
- `proposal-deploy-structure_accepted` + `upgrade-feature` should be tracked as a proxy through `(~forest_precolonial) & indicator_Bird_self_peeling`, using peeling bark in elms as a proxy for artificial bark and thus upgraded structures.
- Fallback for artificial trees might be duplicated; investigate.
