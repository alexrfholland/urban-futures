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

- [_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

Required settings:

- `fallens_use = nonpre-direct`
- `snags_use = elm-snags-old`
- include `decayed-small-fallen`
- `voxel_size = 1`

The default loader root `data/revised/trees` is not an acceptable V3 validation root.

Current optional template-root env vars:

- `TREE_TEMPLATE_ROOT`
  - explicit variant root for runtime loading
- `TREE_TEMPLATE_BASE_ROOT`
  - optional canonical base-library override
- `TREE_TEMPLATE_VARIANTS_ROOT`
  - optional variant-builder root override
- `BASE_TREE_TEMPLATES_ROOT`
  - legacy alias for `TREE_TEMPLATE_BASE_ROOT`

## Template Library Layout

Current rule:

- all tree-template artifacts live under `_data-refactored/model-inputs`
- `model-outputs` is reserved for simulation-state outputs

Template libraries now use three clearer filenames:

- `template-library.base.pkl`
  - full base library
- `template-library.selected-overrides.pkl`
  - only the override rows chosen for this variant
- `template-library.overrides-applied.pkl`
  - full resolved library after those overrides are applied

Current canonical base-library root:

- [_data-refactored/model-inputs/tree_libraries/base/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_libraries/base/trees)

Current canonical v3 variant root:

- [_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

Here, `variant` means one selected template bundle under `_data-refactored/model-inputs/tree_variants`.

To decide the variant, we currently select between:

- fallen modes:
  - `canonical`
  - `nonpre-direct`
  - `nonpre-geometry-pre-attrs`
- snag modes:
  - `elm-models-new`
  - `elm-snags-old`
- decayed strategy:
  - currently the script includes the `decayed-small-fallen` bundle logic

The main scripts involved are:

- builder:
  - [_code-refactored/refactor_code/tree_processing/build_tree_variants.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/tree_processing/build_tree_variants.py)
- runtime template loader:
  - [final/a_resource_distributor_dataframes.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_resource_distributor_dataframes.py)
- v3 baseline helper:
  - [_code-refactored/refactor_code/scenario/baseline_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/baseline_v3.py)

## Required Run Metadata

Every v3 candidate run should persist the full run-config block, not just the template root.

Required env/config record:

- `REFACTOR_RUN_OUTPUT_ROOT`
- `TREE_TEMPLATE_ROOT`
- `EXPORT_ALL_POINTDATA_VARIABLES`

## Runtime Split

- v3 engine runtime: [_code-refactored/refactor_code/scenario/engine_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v3.py)
- scenario runner now imports v3: [a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py)
- the normal per-state chain is now:
  - interim `treeDF` / `logDF` / `poleDF`
  - in-memory polydata enrichment through search layers and indicator/proposal arrays
  - final `state_with_indicators.vtk`
  - final integrated `nodeDF`
- normal v3 candidate runs do not write intermediate `urban_features.vtk` files
- v3 proposal arrays are derived on the final enriched state artifact in [a_info_gather_capabilities.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_gather_capabilities.py)
- v3 proposal render view: [render_forest_size_views.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/render_forest_size_views.py)
- custom v3 proposal render schema: [_documentation-refactored/scenario_engine_v3_render_schema.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v3_render_schema.md)

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
- targeted reruns and the full approved-root batch have now been completed

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

## Scratch Experiment: simv3recruitanddecaytweaks

A full scratch candidate run has now been generated under:

- scenario-output root: [data/revised/simv3recruitanddecaytweaks](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/simv3recruitanddecaytweaks)
- engine-output root: [_data-refactored/simv3recruitanddecaytweaks](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/simv3recruitanddecaytweaks)
- statistics root: [_statistics-refactored-simv3recruitanddecaytweaks](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-simv3recruitanddecaytweaks)

This scratch run used the approved canonical v3 template root explicitly:

- [_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

The scratch run covered all six `{site, scenario}` pathways:

- `trimmed-parade / positive`
- `trimmed-parade / trending`
- `city / positive`
- `city / trending`
- `uni / positive`
- `uni / trending`

All assessed years were generated:

- `yr0`
- `yr10`
- `yr30`
- `yr60`
- `yr90`
- `yr120`
- `yr150`
- `yr180`

Output completeness for the scratch run:

- all expected scenario `treeDF` CSVs were generated
- all expected `nodeDF` CSVs were generated under [feature-locations](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/simv3recruitanddecaytweaks/feature-locations)
- all expected scenario `urban_features` VTKs were generated
- all expected assessed `state_with_indicators` VTKs were generated under [vtks](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/simv3recruitanddecaytweaks/vtks)
- all expected validation renders were generated: `288 / 288`
- indicator/action CSVs were exported for `trimmed-parade`, `city`, and `uni` under [output/csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/simv3recruitanddecaytweaks/output/csv)

Current refactor direction for candidate runs:

- prefer a unified refactored run root under `_data-refactored/{run-name}/`
- use:
  - `temp/interim-data`
  - `temp/validation`
  - `output`
- keep Blender-facing durable products under `output/`, especially:
  - `vtks/`
  - `feature-locations/`
  - `bioenvelopes/`

This scratch experiment includes the following engine-behaviour changes relative to the current canonical v3 candidate:

- recruit spacing threshold reduced to `2.5m`
- `buffer-feature` recruit parent offset reduced to `1m`
- recruit spacing only checks living trees through `senescing`
- `buffer-feature` parent self-spacing is excluded
- future-stock `useful_life_expectancy` raised to `140` for replacements and recruits
- blank `useful_life_expectancy` rows are normalised by size default:
  - `small -> 80`
  - `medium -> 50`
  - `large -> 10`
- deadwood windows changed to:
  - `fallen -> decayed = 40-100 years`
  - `decayed -> gone = 30-75 years`
- deadwood transition timestamps are now sampled reproducibly within the active pulse window instead of always using the pulse-end year
- temporary scratch restriction: `buffer-feature` recruit parents are limited to `under-node-treatment == node-rewilded`
  - `footprint-depaved` parents are excluded in this experiment

This scratch experiment is not canonical v3.

It is a candidate branch-state experiment recorded for recruit/deadwood testing only.

## Current Candidate Run: simv3-5

The current streamlined candidate run root is:

- [_data-refactored/model-outputs/generated-states/simv3-5](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5)

Its output layout is:

- [temp/interim-data](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/temp/interim-data)
- [temp/validation](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/temp/validation)
- [output/vtks](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/vtks)
- [output/feature-locations](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/feature-locations)
- [output/stats](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-outputs/generated-states/simv3-5/output/stats)

Verified output counts:

- `48` pathway `state_with_indicators.vtk`
- `3` baseline `state_with_indicators.vtk`
- `48` integrated `nodeDF` CSVs
- `102` per-state stats CSVs
- `6` merged site stats CSVs

Verified non-outputs:

- no intermediate `urban_features.vtk` files under `temp/interim-data`
- no raw `scenarioYR*.vtk` files under `temp/interim-data`

## Open State

V3 is still a candidate engine.

It has not been promoted.

## Known Issues / To Do

- the initial quick V3 export loaded templates from the default loader root instead of the approved edited variant bundle
- that means the first quick deadwood renders are not valid as a canonical v2 visual parity check
- this should be treated as a verification validity failure on the first quick run, not as the current live template configuration
- the wrong-root quick run is now a closed historical validation failure, not a current blocker
- V3 proposal point-data is currently being expressed on the assessed site voxels, not transferred back onto the tree/node rows as part of the dataframe attribute handoff.
- Follow-up:
  - specify that the proposal/intervention columns are transferred with the other dataframe attributes during the tree/node-to-voxel export path
  - do not treat the current VTK proposal arrays as tree/node-row parity until that transfer is implemented
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
