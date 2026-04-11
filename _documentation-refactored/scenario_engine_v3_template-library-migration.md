## Scenario Engine V3 Template Library Migration

This note records the repository state immediately before the planned template-library filename and path migration.

### Pre-change checkpoint

- repository root:
  - `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia`
- branch:
  - `engine-v3`
- git commit:
  - `b6864f24bf0e0845aa81a61a1782906780e0c0e5`
- recorded at:
  - `2026-04-04 20:29:19 AEDT`

### Working tree at checkpoint

The working tree was not clean when this note was written.

Modified files already present at this checkpoint:

- `_futureSim_refactored/sim/generate_interim_state_data/engine_v3.py`
- `_futureSim_refactored/input_processing/tree_processing/build_tree_variants.py`
- `_documentation-refactored/appendix/appendix-g/recruit and mortality notes.md`

### Current template-library naming

Current filenames in use across the repo:

- `combined_templateDF.pkl`
- `edited_combined_templateDF.pkl`
- `template-edits.pkl`

These names are semantically confusing because:

- `combined_templateDF.pkl` sometimes means the canonical base table
- `combined_templateDF.pkl` in a variant root can also mean a fully resolved edited table
- `edited_combined_templateDF.pkl` is another full-table form
- `template-edits.pkl` is only the selected override rows, not the full library

### Planned rename

Proposed clearer names:

- `template-library.base.pkl`
- `template-library.selected-overrides.pkl`
- `template-library.overrides-applied.pkl`

### Current storage locations

Canonical base library currently referenced by many scripts:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/trees`

Canonical v3 variant root currently used for simulation runs:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees`

### Intended migration direction

The intended direction is:

- move canonical template-library storage into `_data-refactored/model-inputs`
- keep variant roots in `_data-refactored/model-inputs/tree_variants`
- rename files so the base table, selected overrides, and overrides-applied full table are unambiguous

### Post-change outcome

The migration is now applied in the current working tree:

- canonical base-library root:
  - `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_libraries/base/trees`
- canonical variant root:
  - `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/model-inputs/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees`
- optional env overrides now supported:
  - `TREE_TEMPLATE_ROOT`
  - `TREE_TEMPLATE_BASE_ROOT`
  - `TREE_TEMPLATE_VARIANTS_ROOT`
  - `BASE_TREE_TEMPLATES_ROOT` as a legacy alias

### Main runtime loader to update

Primary runtime loader:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/input_processing/tree_processing/a_resource_distributor_dataframes.py`

Current load order there is:

1. `edited_combined_templateDF.pkl`
2. `combined_templateDF.pkl`
3. `template-edits.pkl`

### Main builder to update

Primary variant builder:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/input_processing/tree_processing/build_tree_variants.py`

### Direct references found for `combined_templateDF.pkl`

Runtime / builder references:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/input_processing/tree_processing/a_resource_distributor_dataframes.py`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/input_processing/tree_processing/build_tree_variants.py`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/input_processing/tree_processing/combined_tree_manager.py`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/input_processing/tree_processing/combine_edit_individual_trees.py`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/input_processing/tree_processing/aa_tree_helper_functions.py`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/input_processing/tree_processing/combined_voxelise_dfs.py`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/input_processing/tree_processing/combine_resource_treeMeshGenerator.py`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/bexport/vtk_to_ply.py`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/stanislav/predictions.py`

Documentation references also exist and must be updated after the code/path migration.
