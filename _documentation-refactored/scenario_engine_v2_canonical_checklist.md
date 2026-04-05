# Scenario Engine V2 Canonical Checklist

Use this note when checking whether a run, script, or refactor path is truly aligned with the current canonical v2 flow, rather than an older v1 or ad hoc debug path.

## Canonical Reference

- Canonical branch: `master`
- Pre-v2 baseline branch: `engine-v1`
- Canonical scenario outputs: [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
- Canonical refactored outputs: [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)
- Canonical statistics: [_statistics-refactored-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v2)

If a script is writing into the old v1 roots by default, it is not following canonical v2.

## Runtime / Entry Point

- Canonical engine core is [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py).
- [final/a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py) is only a compatibility wrapper.
- Canonical orchestration goes through [final/a_scenario_manager.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_manager.py).

If a validation or regeneration script bypasses the manager and directly saves raw `scenarioYR*.vtk` files, that is an older or debug path, not the canonical v2 flow.

## 1. Year 0 Behavior

Canonical v2:

- year `0` is a seeded baseline write
- year `0` is not a simulated scenario pulse

Why it matters:

- `positive` and `trending` should not diverge at year `0` because of applied scenario logic

If you see:

- year `0` already changing control distributions
- `positive yr0` and `trending yr0` diverging from the base treeDF

then you are following the old v1-style behavior.

## 2. Path Dependence

Canonical v2:

- years `10, 30, 60, 90, 120, 150, 180` consume the previous saved timestep state
- later years are not re-run from year `0` each time

If you see:

- each year being computed independently from the same starting state

then you are on the old path.

## 3. Release-Control Model

Canonical v2:

- release-control is split into:
  - `pruning_target`
  - `pruning_target_years`
  - `autonomy_years`
  - `control_realized`
- downstream wording is:
  - `Eliminate-Pruning`
  - `Reduce-Pruning`

If you see only the older buffer/brace naming in outputs or graphs, that path has not been brought forward to canonical v2 vocabulary.

## 4. Recruit Model

Canonical v2:

- recruit is pulse-based
- recruit origin occupancy is tracked with:
  - `recruit_intervention_type`
  - `recruit_source_id`
  - `recruit_year`

If those fields are absent from the scenario state tables, it is not canonical v2 behavior.

## 5. Stable `structureID`

Canonical v2:

- `structureID` is the stable persistent structure identity across years
- this applies to:
  - baseline trees
  - recruited trees
  - replacement trees
  - logs
  - poles

If you see:

- `structureID` reassigned from scratch every saved year
- fallen persistence checked by row order or temporary per-file IDs

then you are on the old path.

## 6. Deadwood Lifecycle

Canonical v2:

- `fallen` persists for `50-150` years
- then becomes `decayed`
- `decayed` persists for `50-100` years
- then is removed

Canonical v2 therefore has a real `decayed` size class in:

- sim core
- template library
- baselines
- VTK outputs
- capability indicators
- renders

If you see no `decayed` size at all, that path is missing a canonical v2 deadwood change.

## 7. Canonical Template Configuration

Canonical v2 requires:

- `fallens_use = nonpre-direct`
- `snags_use = elm-snags-old`
- include `decayed-small-fallen`
- `voxel_size = 1`

Approved template root:

- [_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

Required environment variable:

- `TREE_TEMPLATE_ROOT`

Unsafe fallback:

- `data/revised/trees`

If a run does not explicitly set `TREE_TEMPLATE_ROOT`, it can silently fall back to the wrong template library and will not be a valid canonical-v2 visual/deadwood comparison.

## 8. Template Resolution

Canonical v2 export uses:

- `templateResolution = 1`

If anyone assumes canonical v2 was generated from `0.5 m` template tables, that is incorrect.

## 9. Deterministic Template / Log Selection

Canonical v2 now expects deterministic donor-template selection in the resource distributor.

That means:

- no shared-RNG-dependent fallback sampling across delayed tasks
- no random log donor choice tied to task execution order

If repeated exports from the same saved scenario state produce different `urban_features` VTKs, the exporter is not following the corrected canonical-v2 behavior.

## 10. Colonise Logs

Canonical v2:

- translocated log donor selection can still use original `logSize`
- but exported translocated logs carry `size == 'fallen'`
- `log_template_size` preserves the donor size for debugging

If logs still export under their donor size instead of `fallen`, that path is not canonical v2.

## 11. `NodeID = -1` Ground-Mapping Fix

Canonical v2:

- invalid negative node ids are skipped when writing:
  - `node-rewilded`
  - `footprint-depaved`
  - `exoskeleton`

Why it matters:

- without this fix, background voxels with `node_CanopyID == -1` or `sim_Nodes == -1` get flooded into rewilded masks
- that massively inflates:
  - `Lizard.self.grass`
  - `Lizard.others.notpaved`
  - `Tree.others.notpaved`
  - `Tree.generations.grassland`

If those indicators suddenly explode in trending/city/street, check this first.

## 12. Baseline Handling

Canonical v2:

- uses template-aware baseline regeneration when template/lifecycle rules change
- keeps baseline variants on file under:
  - [data/revised/baseline-variants](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/baseline-variants)
- can refresh indicator CSVs against new baseline counts with:
  - [final/refresh_indicator_csvs_from_baseline.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/refresh_indicator_csvs_from_baseline.py)

If a candidate run changes deadwood templates or lifecycle rules but reuses older canonical baselines, the `% baseline` statistics are not aligned.

## 13. VTK Flow

Canonical v2 manager flow:

- scenario CSVs
- `urban_features.vtk`
- `state_with_indicators.vtk`

Canonical v2 manager settings:

- `return_polydata=True`
- `save_raw_vtk=False`

Important nuance:

- the exporter can still save a raw `scenarioYR*.vtk` if explicitly asked
- but that is not the canonical default path anymore

If a run writes:

- `scenarioYR*.vtk`
- then reloads that raw file only to produce `urban_features`

that is an older or debug path, not the canonical v2 manager flow.

## 14. In-Memory `urban_features` Handoff

Canonical v2:

- uses the in-memory `PolyData` returned by `generate_vtk(...)`
- immediately builds `*_urban_features.vtk` from that in-memory object
- does not require a raw-save/raw-reload cycle just to add:
  - `search_bioavailable`
  - `search_design_action`
  - `search_urban_elements`

If a script is reprocessing saved raw VTKs as its normal route, it is not using the canonical v2 handoff.

## 15. Proposal Arrays

Canonical v2:

- proposal arrays are derived on the augmented `state_with_indicators` VTKs
- not on the raw scenario-state VTKs

Current canonical array names:

- `proposal_decay`
- `proposal_recruit`
- `proposal_release_control`
- `proposal_colonise`
- `proposal_deploy_structure`

If someone expects those arrays on raw `scenarioYR*.vtk`, that is the wrong layer.

## 16. Render Source

Canonical v2 preview renders are built from:

- augmented `state_with_indicators` VTKs

Not from:

- raw scenario-state VTKs
- unaugmented `urban_features` VTKs

If render comparisons are being done from the wrong VTK layer, the result is not a canonical-v2 visual comparison.

## 17. Output Naming

Canonical v2 active assessed state naming is:

- `{site}_{scenario}_{voxel}_yr{year}_state_with_indicators.vtk`

Example:

- [city_positive_1_yr180_state_with_indicators.vtk](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs/vtks/city/city_positive_1_yr180_state_with_indicators.vtk)

If a tool is treating raw `scenarioYR*.vtk` as the main state artifact, it is following older expectations.

## 18. Verification Minimums

For a run to count as aligned with canonical v2, check all of these:

- it writes into `final-v2` / `v2engine_outputs` style roots, not v1 roots
- it uses `engine_v2.py`
- year `0` is neutral
- later years are path-dependent
- `TREE_TEMPLATE_ROOT` is explicitly set to the approved variant root
- template resolution is `1`
- `structureID` is stable across years
- `decayed` exists
- colonise logs export as `fallen`
- invalid `NodeID = -1` rows do not flood rewilded masks
- manager path skips raw VTK by default
- `urban_features` is built from the in-memory handoff
- final review and renders use `state_with_indicators`

## 19. Fast “Old Path” Smells

If you see any of the following, stop and check the route before trusting the run:

- output root is `data/revised/final` instead of `data/revised/final-v2`
- output root is `_data-refactored/final-hooks` instead of `_data-refactored/v2engine_outputs`
- template root is missing, or defaults to `data/revised/trees`
- raw `scenarioYR*.vtk` is being treated as the main product
- no `decayed` size class
- `structureID` changes across years
- year `0` already diverges by scenario
- trend/city ground indicators suddenly explode
- renders are built from the wrong VTK layer

