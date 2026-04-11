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

- [_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

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

- `REFACTOR_RUN_OUTPUT_ROOT`
- `TREE_TEMPLATE_ROOT`
- `EXPORT_ALL_POINTDATA_VARIABLES`

Custom proposal render schema:

- [_documentation-refactored/scenario_engine_v3_render_schema.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v3_render_schema.md)

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
- all expected quick-subset `treeDF` and final `state_with_indicators` files exist
- all quick-subset V3 proposal arrays exist on augmented VTKs
- no unexpected V3 proposal labels were found
- quick render counts passed for `classic`, `merged`, `proposal-hybrid`, and `proposal-hybrid-v3`
- quick year-180 sensitive-cell checks kept `positive > trending`
- initial template-root validity failed for the first quick run
- targeted reruns and the full approved-root batch have now been completed

Redundant support alias families have now been removed from the live v3 schema.

In the current v3 path, the proposal families are:

- `proposal-decay`
- `proposal-release-control`
- `proposal-recruit`
- `proposal-colonise`
- `proposal-deploy-structure`

In the dataframe, each of those exists as a `*_decision` and `*_intervention` pair in [engine_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/scenario/engine_v3.py):

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

Full verification has been completed for the current approved-root v3 batch.

Current status for the required minimum record:

- candidate roots checked: yes
- quick verification subset recorded: yes
- repeatability passed: yes
- full file counts passed: yes
- `v2 vs v3` delta location: generated

## Current Refactored Validation Flow

The normal candidate flow is now split into three explicit phases:

1. `--node-only`
   - writes interim `treeDF`
   - writes `logDF` and `poleDF` when those resources exist for the site
2. `--vtk-only`
   - rebuilds `possibility_space_ds`
   - loads saved interim CSVs
   - builds one in-memory polydata per state
   - mutates that same object through search-layer and indicator/proposal enrichment
   - writes only the final `state_with_indicators.vtk`
   - writes the final integrated `nodeDF`
3. `--compile-stats-only`
   - reads final `state_with_indicators.vtk`
   - writes per-state stats
   - merges site-level stats CSVs

Normal candidate runs no longer require saved intermediate `urban_features.vtk` files.

Current full candidate root:

- [_data-refactored/model-outputs/generated-states/simv3-5](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5)

Verified counts for that root:

- `48` pathway `state_with_indicators.vtk`
- `3` baseline `state_with_indicators.vtk`
- `48` integrated `nodeDF` CSVs
- `102` per-state stats CSVs
- `6` merged site stats CSVs

## Known Issues / To Do

- the initial quick V3 export was run against the loader default template root `data/revised/trees`, not the approved edited deadwood variant root
- that means the first quick deadwood renders are not valid as a visual parity check against canonical v2
- this should be treated as a verification validity failure on the first quick run, not as the current live template configuration
- the wrong-root quick run is now a closed historical validation failure, not a current blocker
- Current proposal-field limitation: the V3 `proposal_*V3` and `*_intervention` arrays are currently being applied at the assessed site-voxel level only, not transferred back onto the tree/node dataframe rows during attribute handoff.
- Follow-up requirement: specify that these proposal/intervention columns are transferred with the other dataframe attributes in the tree/node-to-voxel export path before using VTK proposal counts as a direct proxy for tree/node proposal counts.
- `proposal_deploy_structureV3` is still using voxel-side fallback for `upgrade-feature` and `adapt-utility-pole`.
- Follow-up is to standardise the path for the proxy logic.
- `proposal-deploy-structure_accepted` + `upgrade-feature` should be tracked as a proxy through `(~forest_precolonial) & indicator_Bird_self_peeling`, using peeling bark in elms as a proxy for artificial bark and thus upgraded structures.
- Fallback for artificial trees might be duplicated; investigate.
- Need to decide whether `fallen` and `decayed` should count primarily as `fallen log` or `dead branch`.
- Current issue:
  - `fallen` is still dominated by `stat_dead branch`, while `stat_fallen log` remains sparse
  - `resource_dead branch` and `resource_fallen log` are both relatively low compared with the visible deadwood footprint
  - `decayed` has minimal resource presence across both `resource_*` and `stat_*`
- Current reference stats for `trimmed-parade / positive / yr180` assessed VTK:
  - Fallen voxel count: `88,053`
  - Fallen `resource_dead branch`: `2,316`
  - Fallen `resource_epiphyte`: `50`
  - Fallen `resource_fallen log`: `276`
  - Fallen `resource_hollow`: `49`
  - Fallen `resource_other`: `107,568`
  - Fallen `resource_peeling bark`: `843`
  - Fallen `resource_perch branch`: `2,768`
  - Fallen `stat_dead branch`: `108,172`
  - Fallen `stat_epiphyte`: `26`
  - Fallen `stat_fallen log`: `276`
  - Fallen `stat_hollow`: `24`
  - Fallen `stat_other`: `106,316`
  - Fallen `stat_peeling bark`: `775`
  - Fallen `stat_perch branch`: `2,734`
  - Decayed voxel count: `11,512`
  - Decayed `resource_dead branch`: `98`
  - Decayed `resource_epiphyte`: `0`
  - Decayed `resource_fallen log`: `79`
  - Decayed `resource_hollow`: `1`
  - Decayed `resource_other`: `13,126`
  - Decayed `resource_peeling bark`: `46`
  - Decayed `resource_perch branch`: `741`
  - Decayed `stat_dead branch`: `1,353`
  - Decayed `stat_epiphyte`: `1`
  - Decayed `stat_fallen log`: `79`
  - Decayed `stat_hollow`: `0`
  - Decayed `stat_other`: `1,287`
  - Decayed `stat_peeling bark`: `45`
  - Decayed `stat_perch branch`: `98`
