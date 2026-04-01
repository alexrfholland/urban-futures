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

## Quick Verification

Quick verification used:

- sites: `trimmed-parade`, `city`
- scenarios: `positive`, `trending`
- years: `0`, `30`, `180`

Saved quick verification artifacts:

- summary: [_data-refactored/v3engine_outputs/validation/verification_summary.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/verification_summary.md)
- counts CSV: [_data-refactored/v3engine_outputs/validation/verification_counts.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/verification_counts.csv)
- repeatability note: [_data-refactored/v3engine_outputs/validation/quick_repeatability.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs/validation/quick_repeatability.md)

Quick verification result:

- repeatability passed
- all expected quick-subset `treeDF`, raw `urban_features`, and augmented `state_with_indicators` files exist
- all quick-subset V3 proposal arrays exist on augmented VTKs
- no unexpected V3 proposal labels were found
- quick render counts passed for `classic`, `merged`, `proposal-hybrid`, and `proposal-hybrid-v3`
- quick year-180 sensitive-cell checks kept `positive > trending`

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

- the first quick V3 export was run against the loader default template root `data/revised/trees`, not the approved edited deadwood variant root
- that means the earlier quick deadwood renders are not valid as a visual parity check against canonical v2
- before further visual comparison, re-run the quick subset with the approved template root and persist that root in run metadata
- `trimmed-parade / yr180 / Tree / Reproduce` requires follow-up before promotion.
- Current quick-test comparison:
  - v2: `positive = 188,539` voxels (`88.4%` baseline), `trending = 37` voxels (`0.0%` baseline)
  - v3 candidate: `positive = 80,486` voxels (`37.7%` baseline), `trending = 16,063` voxels (`7.5%` baseline)
- Current interpretation: the v3 candidate is allowing `trending` `park-tree` rows to contribute `footprint-depaved` ground treatment, so `Tree.generations.grassland` is no longer near-zero in `trending`.
- Follow-up requirement: keep the required ground-clearing logic, but add a condition so `trending` does not give `park-tree` rows `footprint-depaved` in this path, then re-check Parade `yr180` ground-linked indicators.
