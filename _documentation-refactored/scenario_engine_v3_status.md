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

## Runtime Split

- v3 engine runtime: [_code-refactored/refactor_code/scenario/engine_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v3.py)
- scenario runner now imports v3: [a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py)
- v3 proposal arrays are written in raw scenario VTK generation: [a_scenario_generateVTKs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_generateVTKs.py)
- v3 proposal arrays are recomputed on enriched augmented VTKs: [a_info_gather_capabilities.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_gather_capabilities.py)
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

## Quick Verification Status

Quick verification has been completed for the planned subset:

- sites: `trimmed-parade`, `city`
- scenarios: `positive`, `trending`
- years: `0`, `30`, `180`

Saved quick verification artifacts:

- summary: [_data-refactored/v3engine_outputs/validation/verification_summary.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/verification_summary.md)
- counts: [_data-refactored/v3engine_outputs/validation/verification_counts.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/verification_counts.csv)
- repeatability: [_data-refactored/v3engine_outputs/validation/quick_repeatability.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/quick_repeatability.md)

Quick verification result:

- repeatability checks passed
- V3 vocabulary checks passed
- quick file-presence checks passed
- quick render counts passed
- quick pathway sanity checks passed for the sensitive cells

## Open State

V3 is still a candidate engine.

It has not been promoted.

## Known Issues / To Do

- the first quick V3 export loaded templates from `data/revised/trees/1_combined_voxel_templateDF.pkl` instead of the approved edited variant bundle
- this means the earlier quick V3 deadwood renders are not valid as a visual parity check against canonical v2
- before further visual comparison, re-run the quick subset with the approved template root and persist that root into run metadata
- `trimmed-parade / yr180 / Tree / Reproduce` currently shows a compressed `positive` vs `trending` gap compared with canonical v2.
- Current quick-test comparison:
  - v2: `positive = 188,539` voxels (`88.4%` baseline), `trending = 37` voxels (`0.0%` baseline)
  - v3 candidate: `positive = 80,486` voxels (`37.7%` baseline), `trending = 16,063` voxels (`7.5%` baseline)
- Observed cause in the current candidate: `trending` is no longer near-zero because many `park-tree` rows are receiving `footprint-depaved` ground treatment, which creates `Tree.generations.grassland` support and compresses the `positive` vs `trending` divergence from both sides.
- Follow-up:
  - keep the ground-clearing behavior where it is needed
  - add a condition so `trending` does not assign `footprint-depaved` to `park-tree` rows in this pathway
  - re-run the Parade quick subset and compare `Tree / Reproduce`, `Tree / Communicate`, and the linked ground indicators against canonical v2

Full verification is still required for:

- all sites: `trimmed-parade`, `city`, `uni`
- all years: `0, 10, 30, 60, 90, 120, 150, 180`
- full pathway tables
- direct `v2 vs v3` delta outputs
