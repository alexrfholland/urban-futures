# Python Environment

This repo now uses `uv` with a repo-root `.venv`.

## First Use On A New Machine

Start with [FIRST_MACHINE_SETUP.md](FIRST_MACHINE_SETUP.md).

Use the repo-local `uv` wrapper during bootstrap when `uv` is not already on `PATH`:

- Windows: `uv.cmd` forwards to `.tools\uv\uv.exe`
- macOS / Linux: `./uv` uses machine `uv` unless `UV_BIN` is set

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
uv run python _futureSim_refactored/scenario/runtime/run_full_v3_batch.py --help
uv run python _futureSim_refactored/scenario/runtime/run_saved_v3_vtks.py --help
```

If `uv` is not on `PATH`, use `.\uv.cmd run python ...` on Windows or `./uv run python ...` on macOS / Linux.

In VS Code, select the interpreter from the repo `.venv`.

The active import layout is now:

- repo packages come from the repo `.venv`
- active simulation, tree-library, and Blender-export code lives under `_futureSim_refactored`
- old `final/` locations are no longer the live entrypoint surface

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
