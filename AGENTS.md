# AGENTS

## Current Simulation Stack

There are multiple generations of the simulation core and multiple tree-template libraries in this repo.

Do not guess which one is current.

### Branch Meaning

- `engine-v1`
  - practical pre-v2 baseline branch
  - use this only as the old pre-refactor reference
- `master`
  - old canonical v2 branch
  - use this as the accepted pre-v3 reference
- `engine-v3`
  - current canonical branch
  - this is the latest accepted simulation core

### Current Canonical V3

Canonical v3 currently means:

- branch: `engine-v3`
- scenario outputs: [data/revised/final-v3](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v3)
- engine outputs: [_data-refactored/v3engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs)
- statistics: [_statistics-refactored-v3](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v3)

Canonical v3 template configuration means:

- `fallens_use = nonpre-direct`
- `snags_use = elm-snags-old`
- include the `decayed-small-fallen` variant bundle
- `voxel_size = 1`

Approved canonical template root:

- [_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

### Required Rule For Simulation Runs

If you are running or validating candidate simulation outputs, explicitly set the template root.

Use:

- `TREE_TEMPLATE_ROOT`

Do not rely on the loader default.

The default fallback path:

- [_data-refactored/model-inputs/tree_libraries/base/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_libraries/base/trees)

is not sufficient to guarantee the canonical accepted deadwood template bundle.

### Environment Variables

`TREE_TEMPLATE_ROOT`

- controls which tree-template library is used by resource distribution / VTK export
- must be set explicitly for candidate simulation runs and validation runs
- current approved canonical value:
  - [_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)
- default fallback:
  - [_data-refactored/model-inputs/tree_libraries/base/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_libraries/base/trees)

Rule:

- do not rely on the fallback for canonical or candidate validation work

`TREE_TEMPLATE_BASE_ROOT`

- optional override for the canonical base template-library root
- current default value:
  - [_data-refactored/model-inputs/tree_libraries/base/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_libraries/base/trees)
- legacy alias still accepted by the builder/loader:
  - `BASE_TREE_TEMPLATES_ROOT`

`TREE_TEMPLATE_VARIANTS_ROOT`

- optional override for the tree-variant root used by the variant builder
- current default value:
  - [_data-refactored/model-inputs/tree_variants](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants)

`REFACTOR_RUN_OUTPUT_ROOT`

- the only supported override for candidate run outputs
- used by [_code-refactored/refactor_code/paths.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/paths.py)
- a run written to this root is split as:
  - `temp/interim-data`
  - `temp/validation`
  - `output`
- durable postprocessed outputs for rendering/validation live under:
  - `output/vtks`
  - `output/feature-locations`
  - `output/bioenvelopes`
- old split vars are removed:
  - `REFACTOR_SCENARIO_OUTPUT_ROOT`
  - `REFACTOR_ENGINE_OUTPUT_ROOT`
  - `REFACTOR_STATISTICS_ROOT`

### How To Get The Latest Accepted Setup

If you need the latest accepted simulation core and accepted templates:

1. use branch `engine-v3`
2. use the canonical v3 roots:
   - [data/revised/final-v3](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v3)
   - [_data-refactored/v3engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v3engine_outputs)
   - [_statistics-refactored-v3](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v3)
3. set `TREE_TEMPLATE_ROOT` to:
   - [_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)
4. read these docs before changing anything:
   - [_documentation-refactored/scenario_engine_v3_status.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v3_status.md)
   - [_documentation-refactored/scenario_engine_v3_validation.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v3_validation.md)

### V3 Rule

If you are working on v3:

- treat v3 outputs as canonical outputs
- use the `engine-v3` branch
- use the canonical v3 roots unless you are intentionally creating a scratch candidate root
- still inherit the approved canonical template root unless an explicitly approved new template bundle is being tested
- if a run does not record its template root in metadata, treat that as a validation failure

### V3 Batch Modes

The main non-interactive batch runner is:

- [final/run_full_v3_batch.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/run_full_v3_batch.py)

Default behavior:

- runs scenario generation
- writes interim scenario CSVs
- builds the final integrated `nodeDF`
- builds the final `state_with_indicators.vtk`
- does **not** regenerate baselines unless explicitly asked
- does **not** run the stats pass unless explicitly asked
- does **not** write intermediate `urban_features.vtk` files in the normal path

Important flags:

- `--node-only`
  - write scenario CSV/dataframe outputs only
  - skip VTK generation
  - skip baseline regeneration
  - skip capability / indicator generation
  - writes:
    - `treeDF`
    - `logDF` if present
    - `poleDF` if present

- `--vtk-only`
  - skip scenario CSV generation
  - load saved `treeDF` / `logDF` / `poleDF`
  - rebuild `possibility_space_ds`
  - build final enriched `state_with_indicators.vtk` and final integrated `nodeDF` from saved CSVs

- `--baselines-only`
  - regenerate baseline outputs only
  - write final baseline `state_with_indicators.vtk`
  - do not run pathway generation or stats

- `--regenerate-baselines`
  - opt-in baseline regeneration
  - default is off

- `--compile-stats-only`
  - read final `state_with_indicators.vtk` files
  - write per-state stats
  - merge site-level stats CSVs
  - do no scenario or VTK generation

- `--multiple-agent`
  - intended for split parallel work
  - runs only the requested site / scenario / year slice
  - respects `--vtk-only`
  - keeps stats for a later explicit `--compile-stats-only` pass

- `--save-raw-vtk`
  - writes raw scenario state VTKs in addition to the normal downstream outputs

Useful selector flags:

- `--sites`
- `--scenarios`
- `--years`
- `--voxel-size`

### Saved-CSV VTK Workflow

If scenario CSVs already exist and you only want to regenerate VTK-side outputs:

1. set the explicit roots and `TREE_TEMPLATE_ROOT`
2. use `run_full_v3_batch.py --vtk-only`
3. add `--multiple-agent` if you are intentionally splitting the years across agents
4. add `--save-raw-vtk` if you want the raw scenario VTKs written too

This workflow still requires:

- saved `treeDF` / `logDF` / `poleDF`
- rebuilt `possibility_space_ds`
- the approved template root

It does **not** rerun the scenario engine.

Normal outputs under the unified run root are:

- `temp/interim-data`
  - interim `treeDF` / `logDF` / `poleDF`
- `temp/validation`
  - metadata, timing logs, validation renders
- `output/vtks`
  - final `state_with_indicators.vtk`
- `output/feature-locations`
  - final integrated `nodeDF`
- `output/stats/per-state`
  - per-state indicator and action counts
- `output/stats/csv`
  - merged site-level indicator and action counts

### Never Assume

Never assume that:

- `engine-v1` is the latest branch
- `data/revised/trees` contains the approved canonical deadwood templates
- a run is valid just because the VTKs and renders were generated

Always verify:

- branch
- output roots
- template root
- verification note
- current canonical status document

## TODO

### SIMULATION

#### Ticket 1. Year 0 Scenario Behaviour

Completed.

High amount of release control in trending because `reduce_control_of_trees(...)` in [a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py#L204) assigns trees in positive to `street-tree`.

Specifics:

Because year `0` is already a scenario run, not a shared untouched baseline.

Starting `trimmed-parade` tree controls in the base [trimmed-parade_1_treeDF.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/trimmed-parade/trimmed-parade_1_treeDF.csv):

- `park-tree`: `315`
- `street-tree`: `141`

After the year-0 run:

- `positive`: `454 street-tree`, `399 reserve-tree`, `2 park-tree`
- `trending`: `298 park-tree`, `158 street-tree`

Why this happens:

- both scenarios start from the same base treeDF via [a_scenario_initialiseDS.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_initialiseDS.py#L325)
- then year `0` still runs `run_scenario(...)` via [a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py#L648)
- in `reduce_control_of_trees(...)` via [a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py#L204), any non-senescent tree that falls under the rewild/depave mask gets its control reassigned from `unmanagedCount`
- at year `0`, `unmanagedCount` is `0`, so that reassignment lands in `street-tree`

Why positive gets hit much harder:

- `positive` thresholds are broad in the v3 parameters:
- `maximum-tree-support-threshold = 10`
- `moderate-tree-support-threshold = 50`
- `trending` thresholds are much tighter:
- `maximum-tree-support-threshold = 0`
- `moderate-tree-support-threshold = 1`

So in `positive`, many more trees enter that mask and get reassigned to `street-tree` at year `0`.
In `trending`, far fewer do, so most original `park-tree` canopy stays `park-tree`.

That is why:

- `trending` has much more `Release-Control -> Brace-Feature`
- `positive` has much more `street-tree`
- year `0` is not a neutral shared baseline for this metric

#### Ticket 2. Rename Release-Control Buffer / Brace

Completed.

Rename `Release-Control -> Buffer-Feature` and `Release-Control -> Brace-Feature` to:

- `Release-Control -> Eliminate-Pruning`
- `Release-Control -> Reduce-Pruning`

#### Ticket 3. Parade Tree / Reproduce Follow-Up

Completed.

The Parade `yr180 / Tree / Reproduce` divergence has been investigated, re-run, and incorporated into the current v3 comparison outputs.

#### Ticket 4. Fallen / Decayed Resource Classification

Open.

Need to decide whether `fallen` and `decayed` are counted primarily as `fallen log` or `dead branch`.

Current issue:

- `fallen` is still dominated by `stat_dead branch`, while `stat_fallen log` remains sparse
- `resource_dead branch` and `resource_fallen log` are both relatively low compared with the visible deadwood footprint
- `decayed` has minimal resource presence across both `resource_*` and `stat_*`

Current reference stats for `trimmed-parade / positive / yr180`:

Fallen

- voxel count: `88,053`
- `resource_dead branch`: `2,316`
- `resource_epiphyte`: `50`
- `resource_fallen log`: `276`
- `resource_hollow`: `49`
- `resource_other`: `107,568`
- `resource_peeling bark`: `843`
- `resource_perch branch`: `2,768`
- `stat_dead branch`: `108,172`
- `stat_epiphyte`: `26`
- `stat_fallen log`: `276`
- `stat_hollow`: `24`
- `stat_other`: `106,316`
- `stat_peeling bark`: `775`
- `stat_perch branch`: `2,734`

Decayed

- voxel count: `11,512`
- `resource_dead branch`: `98`
- `resource_epiphyte`: `0`
- `resource_fallen log`: `79`
- `resource_hollow`: `1`
- `resource_other`: `13,126`
- `resource_peeling bark`: `46`
- `resource_perch branch`: `741`
- `stat_dead branch`: `1,353`
- `stat_epiphyte`: `1`
- `stat_fallen log`: `79`
- `stat_hollow`: `0`
- `stat_other`: `1,287`
- `stat_peeling bark`: `45`
- `stat_perch branch`: `98`

#### Ticket 5. Simplify Tree Variant Data Structure

Open.

Current tree-variant storage is too confusing, too bloated, and still carries unnecessary test/debug artifacts.

To do:

- simplify the tree variant data structure
- remove unnecessary duplicated derivative files where possible
- reduce legacy test/debug artifacts stored alongside live inputs
