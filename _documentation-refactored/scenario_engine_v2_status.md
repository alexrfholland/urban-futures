# Scenario Engine V2 Status

## Quick Description

This work introduces a new path-dependent scenario engine in [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py) and turns [final/a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py) into a compatibility wrapper that delegates to it.

The main functional changes are:

- year `0` is now a seeded baseline write, not a simulated scenario step
- later years are path-dependent and consume the previous timestep state
- release-control is decoupled into management target vs realized autonomy/control state
- release-control support names are now `Reduce-Pruning` / `Eliminate-Pruning`
- recruit now has persistent source occupancy memory using `recruit_intervention_type`, `recruit_source_id`, and `recruit_year`
- validation outputs and validation statistics are isolated from canonical outputs

## Old Engine vs New Engine

### Git Path

- Old engine baseline is preserved on branch `master`
- V2 work is currently in the dirty worktree on branch `engine-v2`
- Both branches were at the same base commit when the worktree moved to `engine-v2`

Important detail:

- old engine code is preserved by branch separation, not by a parallel live `*_legacy.py` engine module inside this branch
- canonical generated outputs remain in [data/revised/final](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final)
- canonical refactor statistics remain in [_statistics-refactored](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored)

### Runtime Path Split

- New engine source of truth: [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py)
- Compatibility entrypoint: [final/a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py)
- Scenario orchestration: [final/a_scenario_manager.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_manager.py)
- VTK generation: [final/a_scenario_generateVTKs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_generateVTKs.py)

### Output Roots

- Legacy v1 scenario outputs: [data/revised/final](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final)
- Current canonical v2 scenario outputs: [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
- Legacy v1 refactored outputs: [_data-refactored/final-hooks](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/final-hooks)
- Current canonical v2 refactored outputs: [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)
- Legacy v1 statistics: [_statistics-refactored](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored)
- Current canonical v2 statistics: [_statistics-refactored-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v2)

The temporary `nodeidfix` and `inmemorycheck` roots have now been consolidated into the canonical v2 validation roots:

- `city` and `trimmed-parade` came forward from the verified `nodeidfix` run
- `uni` came forward from the verified `inmemorycheck` run
- the temporary roots were then removed

The path resolver for this split lives in [_code-refactored/refactor_code/paths.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/paths.py).

## Mechanism Changes In The Code

### 1. Year 0 Baseline Fix

The year-0 issue described in [AGENTS.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/AGENTS.md) is handled by `_year_zero_state(...)` in [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py#L839).

Effect:

- year `0` writes a neutral seeded state
- `positive` and `trending` no longer diverge immediately because no scenario logic is applied at year `0`

### 2. Path-Dependent Timesteps

`run_timestep(...)` in [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py#L882) now chains the previous `treeDF` forward through each requested timestep.

Effect:

- later years are derived from the previous scenario state, not from year-0 conditions
- recruit, senescence, pruning progression, and fallen-tree persistence can accumulate over time

### 3. Decoupled Release Control / Autonomy

The release-control/autonomy model is now split into:

- `pruning_target`
- `pruning_target_years`
- `autonomy_years`
- `control_realized`

These are initialized and normalized in [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py#L101) and applied in `apply_release_control(...)` at [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py#L421).

Current behavior:

- `reduce-pruning` can reach `park-tree` and caps there
- `eliminate-pruning` can progress beyond `park-tree` to low-control
- exported `control` still stays in the legacy asset vocabulary

### 4. Release-Control Rename

Release-control labels are renamed in downstream reporting from buffer/brace to:

- `Eliminate-Pruning`
- `Reduce-Pruning`

This is handled in [final/a_info_proposal_interventions.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_proposal_interventions.py) and [final/a_info_pathway_tracking_graphs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_pathway_tracking_graphs.py).

### 5. Recruit Pulse Model + Occupancy Memory

Recruit is now pulse-based and tracks recruit origins using:

- `recruit_intervention_type`
- `recruit_source_id`
- `recruit_year`

These fields are added in [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py#L124).

Occupancy is counted in `_recruit_occupancy_counts(...)` at [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py#L553).

Source IDs are picked in `_pick_source_id(...)` at [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py#L546).

Current source logic:

- `buffer-feature` recruit uses parent-tree node ids
- `rewild-ground` recruit prefers `sim_Nodes`, then `analysis_nodeID`, then `node_CanopyID`

This is a carrying-capacity style cap aligned to the existing code shape rather than a new ecological patch model.

### 6. Validation Stats / Graph Separation

Validation statistics are now routed through `refactor_statistics_root(...)` in [_code-refactored/refactor_code/paths.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/paths.py#L49).

This now applies to:

- [final/a_info_proposal_interventions.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_proposal_interventions.py#L1236)
- [final/a_info_pathway_tracking_graphs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_pathway_tracking_graphs.py#L153)

Effect:

- validation CSV stats go to `_statistics-refactored-v2`
- validation pathway-tracking plots also go to `_statistics-refactored-v2`
- canonical `_statistics-refactored` is preserved

### 7. Proposal Point-Data Arrays On State-With-Indicators VTKs

Proposal/intervention point-data arrays are now written onto the augmented `state_with_indicators` VTKs in [final/a_info_gather_capabilities.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_gather_capabilities.py).

Point-data names:

- `proposal_decay`
- `proposal_recruit`
- `proposal_release_control`
- `proposal_colonise`
- `proposal_deploy_structure`

Value convention:

- `{proposal-name}_{intervention-name}`
- `{proposal-name}-other`
- `none`

String storage:

- these arrays use fixed-width unicode dtype `<U64`
- current longest defined value is `deploy-structure_adapt-utility-pole` at `35` characters
- current naming therefore fits comfortably inside `64` characters
- there is no project-imposed small VTK/PyVista string limit here, but fixed-width unicode is safer than relying on variable-length/object strings during save/load

Defined values by array:

`proposal_decay`

- `none` (`4`)
- `decay-other` (`11`)
- `decay_buffer-feature` (`20`)
- `decay_brace-feature` (`19`)

`proposal_recruit`

- `none` (`4`)
- `recruit-other` (`13`)
- `recruit_buffer-feature` (`22`)
- `recruit_rewild-ground` (`21`)

`proposal_release_control`

- `none` (`4`)
- `release-control-other` (`21`)
- `release-control_reduce-pruning` (`30`)
- `release-control_eliminate-pruning` (`33`)

`proposal_colonise`

- `none` (`4`)
- `colonise-other` (`14`)
- `colonise_rewild-ground` (`22`)
- `colonise_enrich-envelope` (`24`)
- `colonise_roughen-envelope` (`25`)

`proposal_deploy_structure`

- `none` (`4`)
- `deploy-structure-other` (`22`)
- `deploy-structure_adapt-utility-pole` (`35`)
- `deploy-structure_upgrade-feature` (`32`)

Important scope detail:

- these arrays are derived on the augmented `state_with_indicators` VTKs, not on the raw scenario-state VTKs
- the raw scenario-state VTK export does not yet have the needed `search_*` and `indicator_*` arrays
- baseline `state_with_indicators` VTKs currently receive `none` for all `proposal_*` arrays because they do not carry the scenario physical-state arrays those labels are derived from

### 8. Render Test Views

Render test views are now generated from the augmented `state_with_indicators` VTKs using [_code-refactored/refactor_code/scenario/render_forest_size_views.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/render_forest_size_views.py).

Current behavior:

- renders as spheres
- enables eye dome lighting
- uses the fixed site camera presets for `trimmed-parade`, `city`, and `uni`
- writes PNG sequences under `_data-refactored/v2engine_outputs/validation/renders/`
- naming format is `{site}_{scenario}_yr{year}_{view}.png`

Views:

- `classic`: grey rest + trees by `forest_size`
- `merged`: `scenario_bioEnvelope` colors for non-tree voxels + trees by `forest_size`
- `proposal-hybrid`: a single merged proposal/intervention view derived from the `proposal_*` arrays

`classic` / `merged` tree colors:

- `small`: `#AADB5E`
- `medium`: `#9AB9DE`
- `large`: `#F99F76`
- `senescing`: `#EB9BC5`
- `snag`: `#FCE358`
- `fallen`: `#82CBB9`
- `artificial`: `#FF0000`
- rest / unmatched: `#F2F2F2`

`merged` bio-envelope colors:

- `exoskeleton`: `#D9638C`
- `brownRoof`: `#B87A38`
- `otherGround`: `#75A3C4`
- `rewilded`: `#5CB857`
- `footprintDepaved`: `#EBBF54`
- `livingFacade`: `#4CA899`
- `greenRoof`: `#8CCC4F`
- unmatched: `#F2F2F2`

`proposal-hybrid` view colors:

- `senescing`: `#EB9BC5`
- `snag`: `#FCE358`
- `fallen`: `#82CBB9`
- `decay_buffer-feature`: `#D25880`
- `decay_brace-feature`: `#ECB3C7`
- `recruit_rewild-ground`: `#3F82BF`
- `recruit_buffer-feature`: `#9EC7E6`
- `release-control_eliminate-pruning`: `#D48822`
- `release-control_reduce-pruning`: `#F1C67A`
- `colonise_rewild-ground`: `#43A85C`
- `colonise_enrich-envelope`: `#76C66B`
- `colonise_roughen-envelope`: `#B5DD9B`
- `deploy-structure_adapt-utility-pole`: `#CC5353`
- `deploy-structure_upgrade-feature`: `#E59178`
- unmatched: `#F2F2F2`

Proposal-view precedence for overlapping proposal masks:

- `deploy_structure`
- `decay`
- `release_control`
- `recruit`
- `colonise`

Implementation note:

- `senescing`, `snag`, and `fallen` override the proposal palette and render in their lifecycle colors for legibility
- `proposal-hybrid` suppresses all `*-other` categories by rendering them as unmatched grey
- partial support uses the lighter tint within a proposal family where that distinction exists
- `proposal-hybrid` is a render-only merged view for now; it does not yet write a single persisted `proposal_primary` array back into the VTK

### 9. In-Memory `urban_features` Handoff

The scenario and baseline workflows no longer need a second read/write cycle just to add `urban_features`.

Current flow:

- [final/a_scenario_generateVTKs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_generateVTKs.py) can now return the saved raw state VTK path and the in-memory `PolyData`
- [final/a_scenario_urban_elements_count.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_urban_elements_count.py) now exposes:
  - `process_scenario_polydata(...)`
  - `process_baseline_polydata(...)`
- [final/a_scenario_manager.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_manager.py) and [final/run_all_simulations.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/run_all_simulations.py) generate raw scenario VTKs and immediately build `*_urban_features.vtk` from the in-memory `PolyData`
- [final/a_scenario_get_baselines.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_get_baselines.py) now builds baseline `*_urban_features.vtk` directly from the in-memory baseline `PolyData`

Effect:

- raw state VTKs are still saved
- `urban_features` VTKs are still saved
- but the scenario and baseline paths no longer save raw VTKs and then reload them just to add:
  - `search_bioavailable`
  - `search_design_action`
  - `search_urban_elements`

## Checks And Verification Strategy

### Engine / Schema

- `py_compile` on changed Python modules
- direct inspection of new schema columns in validation CSVs
- direct inspection of year-0 control distributions

### Regression / Diff

- compare canonical vs validation outputs with [_code-refactored/refactor_code/scenario/compare_outputs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/compare_outputs.py)
- use CSV row counts and categorical counts for:
  - `control`
  - `rewilded`
  - proposal/support fields
  - lifecycle fields
  - `isNewTree`

### Manual Spatial Review

- inspect scenario VTKs with [_code-refactored/refactor_code/scenario/inspect_vtk.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/inspect_vtk.py)
- compare `scenario_rewilded`, `scenario_bioEnvelope`, and `forest_control`

### Downstream Preservation

- run proposal/intervention exports in validation mode
- run pathway-tracking graph generation in validation mode
- ensure those write into `_statistics-refactored-v2`, not canonical paths

## Verification Done So Far

### Confirmed

- Changed modules compile successfully:
  - [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py)
  - [_code-refactored/refactor_code/paths.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/paths.py)
  - [final/a_info_proposal_interventions.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_proposal_interventions.py)
  - [final/a_info_pathway_tracking_graphs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_pathway_tracking_graphs.py)

- Year-0 parity fix was previously verified for all three sites:
  - `trimmed-parade` base and v2 `positive yr0` both match `park-tree 315 / street-tree 141`
  - `city` base and v2 `positive yr0` both match `street-tree 243 / park-tree 31`
  - `uni` base and v2 `positive yr0` both match `street-tree 187`

- Validation stats roots are isolated:
  - canonical raw/plots stay under [_statistics-refactored](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored)
  - validation raw/plots route to [_statistics-refactored-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v2)

- Focused recruit-capacity regression was run on `uni positive`:
  - earlier v2 overgrowth: `yr180 = 881 total / 711 new`
  - first coarse cap: `383 / 200`
  - revised source-aware cap: `386 / 207`
  - canonical reference remains `482 / 295`

- The revised `uni positive` recruit rows are now spread over multiple source ids rather than one global bucket:
  - `18` unique `recruit_source_id` values at `yr180`

- The focused post-cap comparison artifact exists at [_data-refactored/v2engine_outputs/validation/uni_positive_post_capacity_comparison.json](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs/validation/uni_positive_post_capacity_comparison.json)

### In Progress / Not Yet Refreshed

- A fresh full all-sites/all-scenarios/all-years VTK rerun was started after the recruit-cap tweak, but the full refresh had not completed when this note was written
- Because of that, the older all-sites comparison and some downstream validation artifacts are stale relative to the latest recruit-cap code

### In-Memory `urban_features` Validation

Fresh verification roots used for the in-memory handoff check:

- scenario outputs: [data/revised/final-v2-inmemorycheck](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2-inmemorycheck)
- refactored outputs: [_data-refactored/v2engine_outputs-inmemorycheck](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs-inmemorycheck)
- statistics: [_statistics-refactored-v2-inmemorycheck](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v2-inmemorycheck)

What was rebuilt:

- `uni` baseline
- `uni` positive: `0, 10, 30, 60, 90, 120, 150, 180`
- `uni` trending: `0, 10, 30, 60, 90, 120, 150, 180`

Direct workflow-equivalence check:

- the same saved raw VTK was processed twice:
  - once through the new in-memory handoff
  - once by reloading the saved raw VTK and re-running `process_scenario_polydata(...)`
- result for the `search_*` arrays used by the indicator pipeline:
  - `positive yr180`: `0` mismatches across all `3` `search_*` arrays
  - `trending yr180`: `0` mismatches across all `3` `search_*` arrays
- saved report:
  - [uni_inmemory_vs_reprocess_search_arrays.txt](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs-inmemorycheck/validation/uni_inmemory_vs_reprocess_search_arrays.txt)

Important nuance:

- a broader comparison against the existing `final-v2-nodeidfix/output/csv` exports is not exact
- overlap-only diff reports currently show differences at `year -180` and `180`:
  - [uni_inmemory_vs_nodeidfix_overlap_diff_summary.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs-inmemorycheck/validation/uni_inmemory_vs_nodeidfix_overlap_diff_summary.csv)
- that comparison is not a clean test of the in-memory handoff by itself, because the current `nodeidfix` CSV exports are older than the present VTK / indicator code path
- the narrow same-raw-VTK check above is the decisive validation for this refactor

Preview renders built from the refreshed `uni` `state_with_indicators` VTKs:

- [classic](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs-inmemorycheck/validation/renders/classic)
- [merged](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs-inmemorycheck/validation/renders/merged)
- [proposal](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs-inmemorycheck/validation/renders/proposal)

## Tree Processing Investigation

### Fallen Pipeline Finding

The stronger non-precolonial fallen templates were not lost at the raw-source level. They still exist in:

- [updated_tree_dict.pkl](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/trees/updated_tree_dict.pkl)
- [final/treeMeshes](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/treeMeshes)

They are lost in the active combined-template pipeline in [combined_tree_manager.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/tree_processing/combined_tree_manager.py).

Cause:

- `update_template_files(...)` only appends eucalyptus rows when `precolonial` is `True`
- because elm has no fallen templates, the active [template-library.base.pkl](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_libraries/base/trees/template-library.base.pkl) ends up with only `precolonial=True` fallen rows

Related note:

- [tree_VTKpts](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/tree_VTKpts) is not useful for debugging fallen templates because that export filters to `large` and `snag` only

### Snag Pipeline Finding

The current active non-precolonial snag source is the regenerated elm-derived snag in [combined_redoSnags.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/tree_processing/combined_redoSnags.py).

Important distinction:

- `regenerated_snags`: new geometry, new data
- `updated_snags`: original eucalypt snag geometry with regenerated snag attributes transferred back onto it

That means the project currently has both non-precolonial snag variants in code, but only the regenerated version is active in [combined_tree_manager.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/tree_processing/combined_tree_manager.py).

## Pathway Comparison Refresh

The v2 pathway-comparison refresh is now built without touching the canonical comparison files.

New files:

- [build_comparison_pathways_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/build_comparison_pathways_v2.py)
- [comparison_pathways_indicators_v2.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_v2.csv)
- [comparison_pathways_indicators_v2.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_v2.md)
- [comparison_pathways_v2_deltas.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_v2_deltas.md)

Current read:

- positive still outranks trending in all `27` cells
- the change is magnitude, not sign
- the biggest compressions are in Street and City lizard/tree communication and reproduction
- the biggest expansions are in Parade tree reproduction and City tree acquisition

## Non-Destructive Tree Variant Builder

A new refactor-side variant builder now exists at [build_tree_variants.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/tree_processing/build_tree_variants.py).

Purpose:

- build fallen/snag source variants without touching canonical template tables
- write variant outputs into [_data-refactored/model-inputs/tree_variants](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants)
- make the fallen and snag source switches explicit and repeatable

Current exposed switches:

- `fallens_use`
  - `canonical`
  - `nonpre-direct`
  - `nonpre-geometry-pre-attrs`
- `snags_use`
  - `elm-models-new`
  - `elm-snags-old`

Meaning:

- `nonpre-direct`
  - use the non-precolonial fallen template directly for the precolonial row
- `nonpre-geometry-pre-attrs`
  - keeps the stronger non-precolonial fallen geometry
  - transfers non-coordinate attributes from the original precolonial fallen template onto that donor geometry with nearest-neighbour matching
  - this is the current best fit for “use the better false geometry, but preserve the true row’s attribute logic”
- `elm-models-new`
  - use the newer elm-derived snag models
  - resources are mapped better, but the models are too detailed for `1 m` voxels
- `elm-snags-old`
  - use the older snag models
  - resources are not mapped as well, but the models look better

Current working choice:

- `fallens_use`: `nonpre-direct`
- follow-up still needed: expose fallen attribute-source choice separately rather than baking that decision into one mode name

Important implementation detail:

- full variant template pickles are disabled by default to avoid multi-gigabyte duplicate outputs
- the builder writes `template-library.selected-overrides.pkl`, compact summaries, metadata, VTK meshes, and sample renders by default
- full template pickles and voxel tables remain optional via:
  - `--save-template-pickle`
  - `--build-voxel-tables`

### What `template-library.selected-overrides.pkl` Is

Primary artifact:

- `trees/template-library.selected-overrides.pkl`

Logic:

- [template-library.base.pkl](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_libraries/base/trees/template-library.base.pkl) is the full base template table
- `template-library.selected-overrides.pkl` contains only the rows that should replace canonical rows
- in the current fallen/snag investigation, that means the edited `fallen` and `snag` rows only

Practical use:

1. load canonical `template-library.base.pkl`
2. remove rows whose keys appear in `template-library.selected-overrides.pkl`
3. append the rows from `template-library.selected-overrides.pkl`
4. continue to voxelisation or mesh generation from that edited template state

So:

- `template-library.base.pkl` = the whole base table
- `template-library.selected-overrides.pkl` = the patch to apply to that base table

### Resource Adjustment Point For Fallen Variants

If the project wants the donor false fallen geometry but the target true fallen resource quantities at the `1 m` voxel stage, the current adjustment point is:

- [combined_voxelise_dfs.adjust_resource_quantities(...)](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/tree_processing/combined_voxelise_dfs.py#L309)

That function adjusts voxel resource quantities from the row metadata:

- `precolonial`
- `size`
- `control`

So the clean path is:

- swap geometry in the variant builder
- then run voxelisation/resource adjustment if a voxel-table refresh is required

## Variant Outputs Built So Far

Two non-destructive variant roots have been built:

- [fallen-nonpre-preattrs__snag-regenerated](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/fallen-nonpre-preattrs__snag-regenerated)
- [fallen-nonpre-preattrs__snag-updated-original-geometry](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/fallen-nonpre-preattrs__snag-updated-original-geometry)
- [template-edits__fallens-nonpre-direct__snags-elm-snags-old](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old)

Each variant contains:

- `trees/template-library.selected-overrides.pkl`
- `trees/template-edits_summary.csv`
- `trees/variant_metadata.json`
- `trees/fallen_rows_summary.csv`
- `trees/snag_rows_summary.csv`
- `trees/affected_rows_summary.csv`
- `final/treeMeshes/*.vtk` for affected `fallen` and `snag` templates
- `renders/*.png` sample outputs

Sample renders:

- `elm-models-new` variant fallen sample:
  - [precolonial.True_size.fallen_control.improved-tree_id.15.png](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/fallen-nonpre-preattrs__snag-regenerated/renders/precolonial.True_size.fallen_control.improved-tree_id.15.png)
- `elm-models-new` snag sample:
  - [precolonial.False_size.snag_control.improved-tree_id.11.png](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/fallen-nonpre-preattrs__snag-regenerated/renders/precolonial.False_size.snag_control.improved-tree_id.11.png)
- `elm-snags-old` snag sample:
  - [precolonial.False_size.snag_control.improved-tree_id.11.png](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/fallen-nonpre-preattrs__snag-updated-original-geometry/renders/precolonial.False_size.snag_control.improved-tree_id.11.png)

## Template-Edits Validation Run

The current comparison run used this tree-template configuration:

- `fallens_use = nonpre-direct`
- `snags_use = elm-snags-old`

Variant root:

- [_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old)

Non-overwriting validation output roots for this run:

- scenario CSVs and VTKs: [data/revised/final-v2-template-edits](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2-template-edits)
- augmented state VTKs and render outputs: [_data-refactored/v2engine_outputs-template-edits](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs-template-edits)
- proposal/intervention statistics: [_statistics-refactored-v2-template-edits](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v2-template-edits)

Files produced for the pathway comparison:

- template-edits indicator table: [comparison_pathways_indicators_template_edits_fallens-nonpre-direct_snags-elm-snags-old.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_template_edits_fallens-nonpre-direct_snags-elm-snags-old.csv)
- template-edits indicator markdown: [comparison_pathways_indicators_template_edits_fallens-nonpre-direct_snags-elm-snags-old.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_template_edits_fallens-nonpre-direct_snags-elm-snags-old.md)
- direct delta against the prior v2 engine: [comparison_pathways_v2_vs_template_edits_fallens-nonpre-direct_snags-elm-snags-old.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_v2_vs_template_edits_fallens-nonpre-direct_snags-elm-snags-old.md)
- direct delta CSV against the prior v2 engine: [comparison_pathways_v2_vs_template_edits_fallens-nonpre-direct_snags-elm-snags-old.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_v2_vs_template_edits_fallens-nonpre-direct_snags-elm-snags-old.csv)
- template-edits state renders: [_data-refactored/v2engine_outputs-template-edits/validation/renders](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs-template-edits/validation/renders)

Headline result from this run:

- the fallen/snag template swap does not flip the `positive` versus `trending` direction in any pathway cell
- the main effect is magnitude, not sign
- the largest upward shifts are in Parade deadwood and ground-linked indicators, especially:
  - `Parade / Tree / Reproduce`
  - `Parade / Tree / Communicate`
  - `Parade / Lizard / Acquire Resources`
  - `Parade / Lizard / Communicate`
- the largest downward shift is:
  - `Street / Lizard / Reproduce`

Interpretation:

- swapping in the non-precolonial fallen set broadly increases the deadwood contribution available to `positive`
- that propagates into nurse-log / fallen-tree and some ground-linked tree indicators
- the old snag geometry choice is comparatively less important to the pathway table than the fallen swap

Refreshed status:

- the refreshed `template-edits` run is now complete on top of the fixed v2 engine
- raw scenario VTKs and `urban_features` VTKs are complete for all sites, both scenarios, and years `0, 10, 30, 60, 90, 120, 150, 180`
- augmented `state_with_indicators` VTKs are complete for all sites:
  - `17` per site = baseline + `positive`/`trending` across the `8` target years
- refreshed render sequences are complete under:
  - [_data-refactored/v2engine_outputs-template-edits/validation/renders](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs-template-edits/validation/renders)
  - `classic`
  - `merged`
  - `proposal-hybrid`
  - `51` PNGs per view
- refreshed indicator CSVs are complete under:
  - [data/revised/final-v2-template-edits/output/csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2-template-edits/output/csv)

## Outstanding Tasks

### 1. Complete Full Validation Refresh

Re-run all sites, both scenarios, and full year set:

- `0, 10, 30, 60, 90, 120, 150, 180`

Then regenerate:

- comparison JSONs under [_data-refactored/v2engine_outputs/validation](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs/validation)
- validation nodeDFs and scenario VTKs

### 2. Refresh Downstream Validation Outputs

After any major scenario or template-edits refresh, rerun:

- [final/a_info_proposal_interventions.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_proposal_interventions.py)
- [final/a_info_pathway_tracking_graphs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_pathway_tracking_graphs.py)
- optionally [final/a_info_gather_capabilities.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_gather_capabilities.py)

Use validation mode so outputs continue to land in the appropriate non-canonical roots:

- `-v2`
- or `-template-edits`

### 3. Decide Whether Recruit Capacity Is Calibrated Correctly

The new occupancy memory fixes the runaway `uni` growth, but it may now be conservative:

- canonical `uni positive yr180`: `482 / 295 new`
- current source-aware cap: `386 / 207 new`

That needs a project decision:

- keep the stricter cap
- loosen quotas
- change source granularity
- or add a different carrying-capacity rule

### 4. Improve Batch Validation Ergonomics

The scenario/VTK batch run is currently dominated by verbose voxeliser logging.

Useful follow-up:

- add a quieter batch runner or logging flag for validation sweeps
- avoid surfacing Dask metadata warnings in routine validation runs

### 4b. Cache Repeated Indicator / Proposal Distance Work

The augmented `state_with_indicators` pass in [final/a_info_gather_capabilities.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_gather_capabilities.py) is slower than it needs to be because it repeats large whole-array and KDTree work on the same VTK.

Current hot path:

- load large `urban_features` VTK with `pv.read(...)`
- build/query canopy-distance relationships multiple times:
  - `Tree.others.notpaved` (`within 50m`)
  - `Tree.generations.grassland` (`within 20m`)
  - proposal `recruit_opportunity` (`within 20m`) in `add_proposal_point_data(...)`
- then save the full augmented VTK again

Optimization TODO:

- cache the canopy-feature reference mask once per VTK
- cache `within_20m` once per VTK
- cache `within_50m` once per VTK
- reuse those cached masks in:
  - `apply_indicators(...)`
  - `add_proposal_point_data(...)`
- avoid rebuilding `cKDTree` more than necessary for the same point cloud

Expected benefit:

- less repeated KDTree/query work
- fewer repeated full-array passes
- faster generation of `state_with_indicators` VTKs without changing results

### 5. Decide Which Fallen Variant Should Become Active

Status:

- completed for the current canonical v2
- canonical now uses:
  - `fallens_use = nonpre-direct`

Historic options kept on file:

- keep canonical fallen rows
- use `nonpre-direct`
- or use `nonpre-geometry-pre-attrs`

Related project question:

- whether precolonial fallen rows should inherit donor false resource structure directly
- or whether they should always be re-adjusted to precolonial targets at voxel stage

Current working preference:

- `fallens_use = nonpre-direct`

Follow-up design cleanup:

- expose fallen geometry source and fallen attribute/resource source as separate switches
- TODO:
  - `fallens_use = elm-fallens-everywhere`
  - `eucs-use-elms-with-new-stats`

### 6. Decide Which Non-Precolonial Snag Variant Should Become Active

Status:

- completed for the current canonical v2
- canonical now uses:
  - `snags_use = elm-snags-old`

Historic options remain buildable:

- `snags_use = elm-models-new`
  - resources are mapped better
  - the models are too detailed for `1 m` voxels
- `snags_use = elm-snags-old`
  - resources are not mapped as well
  - the models look better

The likely next decision is whether the canonical combined-template build should keep:

- the current regenerated snag geometry
- the old-geometry updated snag
- or an explicit runtime switch

### 7. Baseline Variants Are Now First-Class Outputs

Baseline regeneration no longer needs manual env juggling.

Use:

- [final/a_scenario_baseline_variants.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_baseline_variants.py)

Purpose:

- generate baseline-only iterations against a chosen template library
- keep each iteration on disk under a named folder
- make it easy to compare baseline effects before promoting a template variant

Output pattern:

- scenario root:
  - `data/revised/baseline-variants/<variant-name>/baselines/...`
- engine root:
  - `_data-refactored/baseline-variants/<variant-name>/...`

Each baseline iteration writes:

- baseline trees CSV
- baseline resources VTK
- baseline terrain VTK
- baseline combined VTK
- baseline `urban_features` VTK
- `variant_metadata.json`

Current live use:

- the template-edits candidate now also has regenerated template-edits baselines on file under:
  - [data/revised/baseline-variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/baseline-variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old)
- those baseline assets were then copied into:
  - [data/revised/final-v2-template-edits/baselines](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2-template-edits/baselines)
- the older reused canonical baselines were preserved in:
  - [data/revised/final-v2-template-edits/baselines/SS_20260331_135725_pre_template_variant_baselines](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2-template-edits/baselines/SS_20260331_135725_pre_template_variant_baselines)
- per-site indicator CSVs were then refreshed against those new baseline counts with:
  - [final/refresh_indicator_csvs_from_baseline.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/refresh_indicator_csvs_from_baseline.py)
- refreshed comparison outputs now live at:
  - [comparison_pathways_indicators_template_edits_fallens-nonpre-direct_snags-elm-snags-old.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_template_edits_fallens-nonpre-direct_snags-elm-snags-old.md)
  - [comparison_pathways_v2_vs_template_edits_fallens-nonpre-direct_snags-elm-snags-old.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_v2_vs_template_edits_fallens-nonpre-direct_snags-elm-snags-old.md)

Example:

```bash
.venv/bin/python final/a_scenario_baseline_variants.py \
  --variant-name template-edits__fallens-nonpre-direct__snags-elm-snags-old \
  --template-root _data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old/trees \
  --sites all
```

### 8. Completed: Reclassify Colonise Translocated Logs

Implemented in [a_resource_distributor_dataframes.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_resource_distributor_dataframes.py):

- donor log-model selection still uses the original `logSize`
- exported translocated log structures now carry `size == 'fallen'`
- original donor class is preserved in `log_template_size` for debugging

Still open:

- issue: `Lizard / Communicate` and `Tree / Communicate` are very high relative to baseline because in [a_scenario_urban_elements_count.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_urban_elements_count.py), `fallen`, `decayed`, and some deadwood/log-related voxels are added into `low-vegetation`, and then in [a_info_gather_capabilities.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_gather_capabilities.py), `Lizard.others.notpaved` and `Tree.others.notpaved` are built from that ground-like mask
- fix: add more deadwood to baselines

### 9. Investigate Ground / Envelope Divergence

This issue is now narrowed to a specific VTK writeback bug, and the first fix pass is implemented.

Root cause that was confirmed:

- rewild-ground recruits can carry `NodeID = -1`
- later, some of those recruit rows are reassigned to `node-rewilded` or `footprint-depaved`
- in [create_rewilded_variable(...)](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_generateVTKs.py#L212), `NodeID = -1` was being treated as a real node match against background `node_CanopyID == -1` or `sim_Nodes == -1` voxels
- that flooded large background voxel fields into `scenario_rewilded = node-rewilded` or `footprint-depaved`
- [create_bioavailablity_layer(...)](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_urban_elements_count.py#L180) then counted those voxels as `low-vegetation`
- that inflated:
  - `Lizard.self.grass`
  - `Lizard.others.notpaved`
  - `Tree.others.notpaved`
  - `Tree.generations.grassland`

Implemented fix:

- [final/a_scenario_generateVTKs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_generateVTKs.py) now skips invalid negative node ids when writing:
  - `node-rewilded`
  - `footprint-depaved`
  - `exoskeleton`
- the log now reports `Skipped <n> rewilded rows with invalid NodeID values`

Pre-fix evidence:

- `uni` trending:
  - v1 `search_bioavailable == low-vegetation`: `6,325`

### 10. Completed: Stable Persistent Tree Identity Across Years

This item is complete and is retained here as a completed record, with the implemented state summarized again in Section 11 below.

Original issue:

- `structureID` is not a persistent tree identity in the exported v2 CSVs
- it is reassigned from scratch at every saved year in:
  - [engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py#L880)
  - [engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py#L928)
- new recruits are also created with temporary:
  - `tree_number = -1`
  - `NodeID = -1`
  - `debugNodeID = -1`
  - in [engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py#L609)

What this breaks:

- simple year-to-year tracking of whether a specific fallen tree really disappeared
- persistence auditing for:
  - `fallen`
  - `snag`
  - replacement trees
  - recruited trees

Observed consequence:

- some IDs that drop out of the fallen set appear in later years under different sizes
- so `structureID` currently cannot be used as proof of object persistence or deletion

Implemented:

- stable persistent `structureID`
- preserved through:
  - aging
  - senescence transitions
  - snag/fallen transitions
  - replacement
  - recruitment
- fallen disappearance validation now uses stable `structureID`
  - v2 `search_bioavailable == low-vegetation`: `480,872`
  - v2 recruit rows with `NodeID = -1` and `rewilded in {node-rewilded, footprint-depaved}`: `3`
- `city` trending:
  - v1 `search_bioavailable == low-vegetation`: `37,265`
  - v2 `search_bioavailable == low-vegetation`: `758,074`
  - v2 recruit rows with `NodeID = -1` and `rewilded in {node-rewilded, footprint-depaved}`: `4`
- background voxel counts in the source datasets are large enough for this to matter:
  - `uni`: `node_CanopyID == -1` on `475,706` voxels; `sim_Nodes == -1` on `222,907`
  - `city`: `node_CanopyID == -1` on `748,100` voxels; `sim_Nodes == -1` on `465,568`

Year-180 verification after the fix, using the `final-v2-nodeidfix` roots:

- `uni` trending:
  - `Lizard.self.grass`: `285.3% -> 3.4%` of baseline
  - `Lizard.others.notpaved`: `275.7% -> 41.8%`
  - `Tree.others.notpaved`: `67.9% -> 16.5%`
  - `Tree.generations.grassland`: `52.3% -> 3.3%`
- `city` trending:
  - `Lizard.self.grass`: `438.5% -> 20.0%`
  - `Lizard.others.notpaved`: `426.8% -> 23.2%`
  - `Tree.others.notpaved`: `98.6% -> 9.8%`
  - `Tree.generations.grassland`: `80.1% -> 6.5%`

These corrected values now sit very close to the old v1 human-led values:

- `uni`:
  - v1 `3.8%`, `42.1%`, `17.3%`, `3.5%`
- `city`:
  - v1 `21.6%`, `24.5%`, `11.0%`, `8.0%`

Pathway-table effect at year `180`:

- `Street / Lizard / Acquire Resources`:
  - v2 `1.26x` -> nodeidfix `21.41x`
- `Street / Lizard / Communicate`:
  - v2 `1.02x` -> nodeidfix `3.60x`
- `Street / Tree / Communicate`:
  - v2 `1.52x` -> nodeidfix `3.65x`
- `Street / Tree / Reproduce`:
  - v2 `1.99x` -> nodeidfix `12.51x`
- `City / Lizard / Acquire Resources`:
  - v2 `1.11x` -> nodeidfix `8.16x`
- `City / Lizard / Communicate`:
  - v2 `1.03x` -> nodeidfix `6.86x`
- `City / Tree / Communicate`:
  - v2 `1.15x` -> nodeidfix `4.08x`
- `City / Tree / Reproduce`:
  - v2 `1.48x` -> nodeidfix `7.13x`

Saved comparison artifacts for this fix pass:

- [comparison_pathways_indicators_nodeidfix.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_nodeidfix.csv)
- [comparison_pathways_indicators_nodeidfix.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_nodeidfix.md)
- [comparison_pathways_nodeidfix_deltas.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_nodeidfix_deltas.md)

Current status:

- the `NodeID = -1` ground-mapping fix is verified and is now part of the canonical v2 validation roots
- the in-memory `urban_features` handoff is verified against reprocessing of the same raw VTKs for `uni`, with `0` mismatches across the `search_*` arrays that feed the indicator pipeline
- canonical v2 validation scenario outputs now live in [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
- canonical v2 validation refactored outputs now live in [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)
- all three sites now have `17` `state_with_indicators` VTKs each in the canonical v2 validation root: baseline plus `positive` and `trending` for `0, 10, 30, 60, 90, 120, 150, 180`
- proposal preview sequences now exist for all three sites, both scenarios, and all year states under [_data-refactored/v2engine_outputs/validation/renders/proposal](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs/validation/renders/proposal)

### 11. Stable Structure IDs and Decayed Phase

Implemented:

- `structureID` is now the stable persistent structure identity across years for:
  - trees
  - replacement trees
  - recruited trees
  - baseline trees
  - logs
  - poles
- helper module:
  - [_code-refactored/refactor_code/scenario/structure_ids.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/structure_ids.py)
- v2 engine integration:
  - [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py)

Verification:

- duplicate `structureID` count within a saved year: `0`
- fallen removal can now be checked cleanly by stable `structureID`
- trimmed-parade positive check:
  - `90 -> 120`: `4` fallen IDs disappear entirely
  - `120 -> 150`: `14` fallen IDs disappear entirely
  - `150 -> 180`: `22` fallen IDs disappear entirely
- `dropped_but_still_present_other_size = 0`

Implemented new deadwood end phase:

- `fallen` persists for `50-150` years, then becomes `decayed`
- `decayed` persists for `50-100` years, then is removed
- the new size is now wired through:
  - sim core
  - template variants
  - baselines
  - VTK outputs
  - capability indicators
  - preview colors
  - Blender import size mapping

Template source for `decayed`:

- based on the old small fallen geometry
- generated by the tree-variant builder as:
  - `template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen`

Canonical v2 now includes:

- the template edits:
  - `fallens_use = nonpre-direct`
    - both elm and euc fallens use elm fallen models, because those models are much larger
  - `snags_use = elm-snags-old`
    - elm snags use their old versions, because they look better as simplified voxels, even though their branch mapping is less accurate
- the `NodeID = -1` ground-mapping fix
- the in-memory `urban_features` handoff
- stable persistent `structureID`
- the `decayed` lifecycle phase

Canonical definition:

- when this note says `canonical`, it means the current accepted `final-v2` model setup and outputs
- use the settings above as the canonical tree-template definition
- canonical roots are:
  - scenario outputs in [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
  - refactored outputs in [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)

Canonical roots after promotion:

- scenario outputs:
  - [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
- refactored outputs:
  - [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)

Comparison artifacts for the promoted candidate:

- [comparison_pathways_indicators_template_edits_decayed.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_template_edits_decayed.csv)
- [comparison_pathways_indicators_template_edits_decayed.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_template_edits_decayed.md)
- [comparison_pathways_v2_vs_template_edits_decayed.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_v2_vs_template_edits_decayed.csv)
- [comparison_pathways_v2_vs_template_edits_decayed.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_v2_vs_template_edits_decayed.md)

Current next step:

- safe optimization pass implemented:
  - add `save_raw_vtk=False`
  - cache the template pickle once per run
  - reduce xarray debug / validation in production mode
- optimization caveat:
  - exact rerun equality from the saved year-state CSVs is currently blocked by pre-existing nondeterministic tree-template fallback sampling in the Dask resource distributor
  - the more aggressive lookup/iteration rewrite was not kept because it changed exported VTK contents
- next optimization prerequisite:
  - make fallback template selection deterministic before changing the tree-distribution hot path further

### 12. TODO Review

Completed:

- stable persistent `structureID`
- fallen removal verification
- `decayed` lifecycle phase
- baseline regeneration with template-aware variants
- all-sites canonical promotion to `final-v2`
- safe export-path optimization pass

Still open:

- review whether `ground_not_paved` should keep counting deadwood as ground-like habitat
- make fallback template selection deterministic before deeper tree-distributor optimization
- then revisit:
  - cached lookup / hot-path tree distribution optimization
  - repeated KDTree/query caching in `a_info_gather_capabilities.py`
