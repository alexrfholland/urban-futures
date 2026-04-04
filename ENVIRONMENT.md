# Python Environment

This repo now uses `uv` with a repo-root `.venv`.

## Setup

Install the default `core` environment:

```bash
uv sync
```

Install everything used in this repo:

```bash
uv sync --extra world-construction --extra visuals --extra blender --extra rhino-legacy
```

Run commands through the repo environment:

```bash
uv run python final/run_full_v3_batch.py --help
uv run python final/run_saved_v3_vtks.py --help
```

In VS Code, select the interpreter from the repo `.venv`.

The current import layout is unchanged:

- repo packages come from the repo `.venv`
- `refactor_code.*` still resolves from `_code-refactored`

## Deferred legacy native packages

These are intentionally not included in the main shared extras yet:

- `hnswlib`
  - needed by `final/f_resource_scenarios.py`
  - needed by `final/f_resource_scenarios2.py`
  - used for approximate nearest-neighbor colour transfer in the site prep / resource voxel pipeline
- `infomap`
  - needed by `final/tree_processing/adTree_AssignLargerClusters.py`
  - needed by `final/tree_processing/adTree_viewGraph`
  - used for graph community detection in the tree data prep / cluster-assignment pipeline

If you need to rerun those deferred workflows or prep stages, add and verify those packages separately.
