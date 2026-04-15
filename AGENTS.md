# AGENTS

## NEVER quick-fix, hack, or add a failsafe without explicit instruction

When something fails, investigate the root cause and report back — do not patch symptoms (timeouts, try/excepts, fallbacks, bypass flags, workarounds) on your own initiative.

## 0. Runtime Setup

- Branch: `engine-v4`
- For Python repo work, use the standard `uv` project model: `uv sync`, `uv run python ...`, `uv add ...`
- For Blender work in bV2 and compositor, run Blender directly per the relevant pipeline contract
- Do not use system `python` / `python3` or direct `.venv` interpreter paths as the normal workflow
- All commands run from repo root (the directory containing `_futureSim_refactored/`)

### Stale documentation warning

All current code lives in `_futureSim_refactored/`. Anything in `final/`, `data/`, `_code-refactored/` is legacy/stale. Documentation under `_documentation-refactored/` is mixed — some files reference old paths and are outdated. The authoritative docs for each pipeline stage are the ones linked from this file. If you find a doc that references `final/` or `data/revised/` paths, treat it as historical context only.

## 0a. Contracts and Inputs

### Simulation Inputs

All simulation inputs live under `_data-refactored/model-inputs/`:

- `sites/{site}/` — per-site voxel arrays, node CSVs, world reference VTKs
- `shared/` — cross-site reference data (baseline density, log library, site coords)
- `tree_libraries/` — base tree template library
- `tree_variants/` — variant template sets and approved template root

Full specification: [v4-input-contract.md](_futureSim_refactored/sim/run/v4-input-contract.md)

### Simulation Outputs

Outputs go to `_data-refactored/model-outputs/generated-states/<root>/` via `REFACTOR_RUN_OUTPUT_ROOT`.

Output lineage (simulation -> EXR families -> compositor runs) and Mediaflux sync rules: [MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md)

Mediaflux:

- Use `mediafluxsync` for actual upload/download work. Do not use raw `unimelb-mf-*` commands.
- Use [MEDIAFLUX.md](MEDIAFLUX.md) for operational guidance and [_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md) for the sync/path contract.

### Environment Variables

`REFACTOR_RUN_OUTPUT_ROOT`

- The only supported output root override. Set this every run.
- Splits output as: `temp/interim-data`, `temp/validation`, `output/`
- Old split vars (`REFACTOR_SCENARIO_OUTPUT_ROOT`, `REFACTOR_ENGINE_OUTPUT_ROOT`, `REFACTOR_STATISTICS_ROOT`) are rejected at startup.

`TREE_TEMPLATE_ROOT` — defaults to the approved canonical template set. No need to set unless explicitly asked. Details in [v4-input-contract.md](_futureSim_refactored/sim/run/v4-input-contract.md).

---

## Pipeline Overview

The full pipeline is: **Simulation → Blender EXRs → Compositor PNGs → Mediaflux sync**. Each stage is independent — you will usually be asked to run just one. The simulation stage itself has sub-steps (1a–1h below) that must run in order.

---

## 1. Simulation Pipeline

**Entry point**: `_futureSim_refactored/sim/run/run_full_v3_batch.py`
**Full reference**: [v4-run-instructions.md](_futureSim_refactored/sim/run/v4-run-instructions.md) — read this before running any simulation step.

**A full simulation run** = Steps 1a → 1b → 1c → 1d → 1g → 1i (mandatory, in order). Steps 1e, 1f, 1h are optional re-renders/regeneration — only run if asked. Steps 2–4 (Blender, Compositor, Mediaflux) are separate pipelines that consume simulation outputs.

**Defaults** (no need to pass explicitly unless subsetting):

- Sites: `trimmed-parade`, `city`, `uni`
- Scenarios: `positive`, `trending`
- Years: `0, 1, 10, 30, 60, 90, 120, 150, 180`

**Before any step**: Ask the user for the output root name. Set it as:

```bash
export REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name>
```

Do not invent a root name. Do not set `TREE_TEMPLATE_ROOT` or `SIM_OUTPUT_ROOT`.

### 1a. Node-only (treeDFs + nodeDFs)

Runs the scenario engine to produce tree/log/pole DataFrames. Must complete before all other steps.

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root> \
  uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py --node-only
```

**Outputs** (in `{root}/temp/interim-data/{site}/`): `{site}_{scenario}_1_treeDF_{year}.csv` (+ logDF, poleDF), recruit telemetry/stats CSVs, size stats CSV.

**Verify**: 18 treeDF files per site (9 years × 2 scenarios).

### 1b. VTK generation

Reads saved treeDFs and builds VTK point clouds with indicators, proposals, bioenvelopes, and renders. **Slowest step.**

**Always use `--multiple-agent`** — this skips the cross-state capability pass so slices can run in parallel. The capability pass runs separately in Step 1g.

Single slice:

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root> \
  uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py \
  --vtk-only --multiple-agent --sites trimmed-parade --scenarios positive --years 180
```

All sites (parallel, batch by site — 18 processes each):

```bash
export REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root>
for site in trimmed-parade city uni; do
  for scenario in positive trending; do
    for year in 0 1 10 30 60 90 120 150 180; do
      uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py \
        --vtk-only --multiple-agent --sites "$site" --scenarios "$scenario" --years "$year" \
        > /tmp/vtk_${site}_${scenario}_${year}.log 2>&1 &
    done
  done
  wait  # finish site before starting next
done
```

**Outputs per slice**: state VTK, nodeDF CSV, per-state stat CSVs (indicators, interventions, decisions, decision subgroups), bioenvelope PLY (no yr0), proposal render PNG, debug recruit PNGs (yrs 10/60/180 only).

**Verify (full run)**: 54 VTKs, 54 nodeDFs, 54× stat CSVs (4 schemas each), 48 PLYs, 54 proposal renders.

### 1c. Baselines

Generates woodland baseline VTKs and nodeDFs. Requires Step 1a (uses same cached `.nc` subset).

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root> \
  uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py --baselines-only
```

**Verify**: 3 baseline VTKs, 3 baseline V4 indicator CSVs.

### 1d. V4 indicator comparisons

Extracts V4 indicators from yr 180 VTKs and produces a markdown comparison table. Requires Steps 1b + 1c.

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root> \
  uv run python _futureSim_refactored/sim/v4_indicator_extract.py
```

**Output**: `{root}/comparison/v4_indicator_comparison.md`

### 1e. Proposal renders (optional re-render)

Re-renders proposal visualisations from existing VTKs. Only needed if re-rendering — Step 1b already produces these inline.

```bash
export REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root>
for site in trimmed-parade city uni; do
  for scenario in positive trending; do
    for year in 0 1 10 30 60 90 120 150 180; do
      uv run python _futureSim_refactored/outputs/report/render_proposal_v4.py \
        --site "$site" --scenario "$scenario" --year "$year" \
        > /tmp/render_proposal_${site}_${scenario}_${year}.log 2>&1 &
    done
  done
done
wait
```

### 1f. Debug recruit renders (optional re-render)

Re-renders diagnostic recruit images. Only needed if re-rendering — Step 1b already produces these inline for years 10, 60, 180.

```bash
export REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root>
for site in trimmed-parade city uni; do
  for scenario in positive trending; do
    for year in 10 60 180; do
      uv run python _futureSim_refactored/outputs/report/render_debug_recruit.py \
        --site "$site" --scenario "$scenario" --years "$year" \
        > /tmp/render_debug_${site}_${scenario}_${year}.log 2>&1 &
    done
  done
done
wait
```

### 1g. Compile stats (merge site-level CSVs)

Merges per-state indicator and action CSVs into site-level CSVs with `pct_of_baseline`. Fast CSV-only pass.

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root> \
  uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py --compile-stats-only
```

**Verify**: 6 merged CSVs (3 sites × 2 types: indicator + action) in `{root}/output/stats/csv/`.

### 1h. Bioenvelope regeneration (optional)

Regenerates bioenvelope PLYs from existing VTKs without rerunning the full pipeline.

```bash
export REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root>
for site in trimmed-parade city uni; do
  for scenario in positive trending; do
    for year in 0 1 10 30 60 90 120 150 180; do
      uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py \
        --bioenvelope-only --sites "$site" --scenarios "$scenario" --years "$year" \
        > /tmp/bioenv_${site}_${scenario}_${year}.log 2>&1 &
    done
  done
  wait
done
```

### 1i. Graphs (capability stream, proposal stream, decision pies)

Reads per-state stat CSVs (written during Step 1b or regenerated via `vtk_to_stat_counts`). No VTK loading. To regenerate stats standalone, see Step 4b in [v4-run-instructions.md](_futureSim_refactored/sim/run/v4-run-instructions.md).

```bash
# Capability stream graphs
uv run python -m _futureSim_refactored.outputs.graphs.stream_graph_v4 \
    --root _data-refactored/model-outputs/generated-states/<root>

# Proposal intervention stream graphs
uv run python -m _futureSim_refactored.outputs.graphs.proposal_stream_graph_v4 \
    --root _data-refactored/model-outputs/generated-states/<root>

# Proposal decision pie charts (per site)
for site in trimmed-parade city uni; do
  uv run python -m _futureSim_refactored.outputs.graphs.proposal_decision_pies_v4 \
    --root _data-refactored/model-outputs/generated-states/<root> --site "$site" &
done
wait
```

**Outputs**: `{root}/output/plots/` (stream graphs) and `{root}/output/graphs/proposal-decision-pies/` (pie charts).

---

## 2. Generate EXRs (bV2 pipeline)

Headless Blender pipeline that imports simulation VTKs/PLYs and renders multi-layer EXR image sets per camera/view-layer.

Authoritative contract and run instructions:
[_documentation-refactored/blenderv2/bV2_run-instructions.md](/_documentation-refactored/blenderv2/bV2_run-instructions.md).


## 3. Generate PNGs via Compositor

Blender compositor pipeline: consumes EXR families, produces PNG outputs per compositor family.

**Authoritative doc**: [COMPOSITOR_RUN-INSTRUCTIONS.md](_futureSim_refactored/blender/compositor/COMPOSITOR_RUN-INSTRUCTIONS.md) — family→runner→blend table, batch driver usage (`batch_parade_timeline_4_10.py --parallel N`), known Blender-4.2 workarounds (`animation=True`, `rebuild_file_output`), living-doc policy (canonical vs `_archive/` vs `temp_blends/`).

**Family registry**: [compositor_families.json](_futureSim_refactored/blender/compositor/compositor_families.json) — per-family schema (EXR inputs, required passes, per-branch, arboreal mark). Update when you add a compositor.

**Other contracts**:
- [COMPOSITOR_TEMPLATE_CONTRACT.md](_futureSim_refactored/blender/compositor/COMPOSITOR_TEMPLATE_CONTRACT.md) — what blends own vs what runners own.
- [COMPOSITOR_SYNC_CONTRACT.md](_futureSim_refactored/blender/compositor/COMPOSITOR_SYNC_CONTRACT.md) — `sim_root`/`exr_family`/`compositor_run` path lineage.

Canonical runners are `render_current_<family>.py` under `_futureSim_refactored/blender/compositor/scripts/`. Anything under `scripts/_archive/` or `canonical_templates/_archive/` is legacy — do not use. `COMPOSITOR_RUN.md` and `scripts/README.md` at that path are also superseded by the run-instructions doc above.

---

## 4. Sync to Mediaflux

Upload/download simulation outputs, EXR families, and compositor runs to the University of Melbourne Mediaflux research archive.

**Operational guide**: [MEDIAFLUX.md](MEDIAFLUX.md)
**Contract**: [_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md)

- Use `uv run python -m mediafluxsync ...` from the repo root for sync commands.
- Use `uv run python -m _futureSim_refactored.sim.run.mediaflux_browse ...` for mounted discovery when needed.

## Utilities

### vtk_extract — fast partial VTK reader

`_futureSim_refactored.sim.vtk_extract` extracts individual point-data arrays from legacy binary VTK files without loading the full mesh (~10x faster than `pv.read`). Refer to the module docstring and `KNOWN_COLUMNS` registry in [vtk_extract.py](_futureSim_refactored/sim/vtk_extract.py) for API details and supported dtypes.


#FURTHER CLAUDE INSTRUCTIONS

## Project

`urban-futures` is the simulation engine and data pipeline for the 2026 Arboreal Futures project. Source lives in `_code-refactored/`. The canonical local pipeline data now lives repo-locally under `_data-refactored/`. Legacy source/reference material still exists under `data/`.

