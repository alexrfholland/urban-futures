# AGENTS

## Current Simulation Stack

There are multiple generations of the simulation core and multiple tree-template libraries in this repo.

Do not guess which one is current.

### Branch Meaning

- `engine-v1`
  - practical pre-v2 baseline branch
  - use this only as the old pre-refactor reference
- `master`
  - current canonical v2 branch
  - this is the latest accepted simulation core
- `engine-v3`
  - v3 candidate branch
  - use this only for the new refactor work

### Current Canonical V2

Canonical v2 currently means:

- branch: `master`
- scenario outputs: [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
- engine outputs: [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)
- statistics: [_statistics-refactored-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v2)

Canonical v2 template configuration means:

- `fallens_use = nonpre-direct`
- `snags_use = elm-snags-old`
- include the `decayed-small-fallen` variant bundle
- `voxel_size = 1`

Approved canonical template root:

- [_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

### Required Rule For Simulation Runs

If you are running or validating candidate simulation outputs, explicitly set the template root.

Use:

- `TREE_TEMPLATE_ROOT`

Do not rely on the loader default.

The default fallback path:

- `data/revised/trees`

is not sufficient to guarantee the canonical accepted deadwood template bundle.

### Environment Variables

`TREE_TEMPLATE_ROOT`

- controls which tree-template library is used by resource distribution / VTK export
- must be set explicitly for candidate simulation runs and validation runs
- current approved canonical value:
  - [_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)
- unsafe fallback:
  - [data/revised/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/trees)

Rule:

- do not rely on the fallback for canonical or candidate validation work

`REFACTOR_SCENARIO_OUTPUT_ROOT`

- overrides the scenario-output root used by [_code-refactored/refactor_code/paths.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/paths.py)
- use this when a run should write somewhere other than the default mode root
- canonical v2 root:
  - [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
- typical v3 candidate root:
  - `data/revised/final-v3`

`REFACTOR_ENGINE_OUTPUT_ROOT`

- overrides the refactored engine-output root used by [_code-refactored/refactor_code/paths.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/paths.py)
- use this when a run should write augmented VTKs, renders, and related engine outputs somewhere other than the default mode root
- canonical v2 root:
  - [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)
- typical v3 candidate root:
  - `_data-refactored/v3engine_outputs`

`REFACTOR_STATISTICS_ROOT`

- overrides the statistics root used by [_code-refactored/refactor_code/paths.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/paths.py)
- use this when a run should write stats/graphs somewhere other than the default mode root
- canonical v2 root:
  - [_statistics-refactored-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v2)
- typical v3 candidate root:
  - `_statistics-refactored-v3`

### How To Get The Latest Accepted Setup

If you need the latest accepted simulation core and accepted templates:

1. use branch `master`
2. use the canonical v2 roots:
   - [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
   - [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)
3. set `TREE_TEMPLATE_ROOT` to:
   - [_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)
4. read these docs before changing anything:
   - [_documentation-refactored/scenario_engine_v2_model.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_model.md)
   - [_documentation-refactored/scenario_engine_v2_status.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_status.md)
   - [_documentation-refactored/validation.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/validation.md)

### V3 Rule

If you are working on v3:

- do not overwrite canonical v2 outputs
- use the `engine-v3` branch
- use the dedicated v3 roots
- still inherit the approved canonical template root unless an explicitly approved new template bundle is being tested
- if a run does not record its template root in metadata, treat that as a validation failure

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

- `positive` thresholds are broad in [a_scenario_params.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_params.py#L148):
- `rewildThreshold = 10`
- `plantThreshold = 50`
- `trending` thresholds are much tighter:
- `rewildThreshold = 0`
- `plantThreshold = 1`

So in `positive`, many more trees enter that mask and get reassigned to `street-tree` at year `0`.
In `trending`, far fewer do, so most original `park-tree` canopy stays `park-tree`.

That is why:

- `trending` has much more `Release-Control -> Brace-Feature`
- `positive` has much more `street-tree`
- year `0` is not a neutral shared baseline for this metric

#### Ticket 2. Rename Release-Control Buffer / Brace

Rename `Release-Control -> Buffer-Feature` and `Release-Control -> Brace-Feature` to:

- `Release-Control -> Eliminate-Pruning`
- `Release-Control -> Reduce-Pruning`
