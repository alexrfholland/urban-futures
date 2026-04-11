# `final/` pipeline is broken

As of commit **`1b477b5` "Restructure v4 simulation layout"**, the entry points in this folder no longer run from scratch. The v4 restructure moved three helper modules out of `final/` and into `_code-refactored/refactor_code/sim/voxel/`, but the original orchestrators still try to import them by their old flat names.

## What broke

| Moved from `final/` | Now at |
|---|---|
| `a_voxeliser.py` | `_code-refactored/refactor_code/sim/voxel/voxel_a_voxeliser.py` |
| `a_helper_functions.py` | `_code-refactored/refactor_code/sim/voxel/voxel_a_helper_functions.py` |
| `f_SiteCoordinates.py` | `_code-refactored/refactor_code/sim/voxel/voxel_f_SiteCoordinates.py` |

Both entry points crash on the first import:

- `final/f_manager.py` — world preparation + Melbourne Open Data fetch (API + photomesh + terrain/contours/buildings)
- `final/a_manager.py` — simulation build (voxelise → trees/poles → logs → resistance → rewilding nodes)

## Why things still appear to work

The refactor consumes the **checkpoint data** produced by `final/` *before* the restructure — `data/revised/final/{site}/*.nc`, `*.csv`, and the various site VTKs. That checkpoint is on disk, so `_code-refactored/` runs happily without ever needing to re-invoke `final/`.

## To re-enable the old flow (e.g. to re-fetch Melbourne Open Data or re-process photomeshes)

The simplest fix — copy the three files back under their original flat names:

```bash
cp _code-refactored/refactor_code/sim/voxel/voxel_a_voxeliser.py       final/a_voxeliser.py
cp _code-refactored/refactor_code/sim/voxel/voxel_a_helper_functions.py final/a_helper_functions.py
cp _code-refactored/refactor_code/sim/voxel/voxel_f_SiteCoordinates.py  final/f_SiteCoordinates.py
```

(May still need minor import tweaks inside those files if they were refactored after the move.)

## Full old pipeline documentation

See [`_documentation-refactored/DATA_PIPELINE_SCRIPT_SUMMARIES.md`](../_documentation-refactored/DATA_PIPELINE_SCRIPT_SUMMARIES.md) for the complete step-by-step flow (inputs, processing, outputs) that `final/` used to run end-to-end.
