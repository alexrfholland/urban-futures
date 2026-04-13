# AGENTS

## 0. Runtime Setup

- Branch: `engine-v4`
- Always use `uv` and the repo-local `.venv`
  - `uv run python ...` for project Python commands
  - `./.venv/bin/python` if invoking the interpreter directly
  - Do not use system `python` / `python3`
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

**A full simulation run** = Steps 1a → 1b → 1c → 1d → 1g (mandatory, in order). Steps 1e, 1f, 1h are optional re-renders/regeneration — only run if asked. Steps 2–4 (Blender, Compositor, Mediaflux) are separate pipelines that consume simulation outputs.

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

**Outputs per slice**: state VTK, nodeDF CSV, V4 indicator CSV, bioenvelope PLY (no yr0), proposal render PNG, debug recruit PNGs (yrs 10/60/180 only).

**Verify (full run)**: 54 VTKs, 54 nodeDFs, 54 indicator CSVs, 48 PLYs, 54 proposal renders.

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

---

## 2. Generate EXRs via Blender v2

Headless Blender pipeline that imports simulation VTKs/PLYs and renders multi-layer EXR image sets per camera/view-layer.

**Current code**: `_futureSim_refactored/blender/blenderv2/bV2_*.py` — these are the only current scripts. Anything prefixed `b2026_unified_*`, `b2026_timeline_*`, or under `timeline/` is **stale**. The `blenderv2/AGENTS.md` is also partially outdated.

**This code works.** If you get wrong paths, missing files, or Blender errors, you are doing something wrong — do not assume the code is broken.

### Contract

`bV2_scene_contract.py` is the single source of truth for naming conventions, valid values, and scene structure. Read it before making any changes. It defines:

- **Sites**: `city`, `trimmed-parade`, `uni` (never use `street`)
- **Modes**: `single_state`, `timeline`, `baseline`
- **View layers**: `existing_condition_positive`, `positive_state`, `positive_priority_state`, `existing_condition_trending`, `trending_state`, `bioenvelope_positive`, `bioenvelope_trending` (baseline uses only the first three)
- **Collection tree**: `cameras`, `world`, `instancers`, `bioenvelopes`, `build`
- All canonical naming functions: `make_scene_name()`, `make_position_object_name()`, `make_bioenvelope_object_name()`, etc.

### Entry point

There is **one entry point**: `bV2_build_scene.py`. It orchestrates everything. Do not call the sub-modules directly.

```bash
BV2_SITE=city BV2_MODE=single_state BV2_YEAR=180 BV2_SIM_ROOT=<sim_root> BV2_RENDER_EXRS=1 \
  blender --background _data-refactored/blenderv2/bV2_template.blend \
  --python _futureSim_refactored/blender/blenderv2/bV2_build_scene.py
```

### What it does (in order)

1. Opens the template blend
2. `init_scene` — creates scene, collection shells, view layers, AOVs from contract
3. `build_instancers` — builds tree/log/pole point clouds with framebuffer positioning
4. `build_bioenvelopes` — imports bioenvelope PLYs (skipped for baseline)
5. `build_world_attributes` — rebuilds world geometry with source-year attributes
6. `validate_scene` — structural validation (collections, AOVs, cameras)
7. Saves .blend (if `BV2_SAVE_BLEND=1`, default)
8. `setup_render_outputs` + render EXRs (if `BV2_RENDER_EXRS=1`)
9. Upload to Mediaflux (if `BV2_UPLOAD_TO_MEDIAFLUX=1`)

### Required env vars

```bash
BV2_SITE=city                    # city | trimmed-parade | uni
BV2_MODE=single_state            # single_state | timeline | baseline
BV2_YEAR=180                     # required for single_state mode
BV2_SIM_ROOT=<sim_root>          # simulation output root name
```

### Optional env vars

```bash
BV2_RENDER_EXRS=1                # trigger EXR render after build (default: off)
BV2_SAVE_BLEND=1                 # save the built .blend file (default: on)
BV2_UPLOAD_TO_MEDIAFLUX=1        # upload EXRs after render
BV2_RES_X=7680                   # render resolution (default: 7680 = 8K)
BV2_RES_Y=4320                   # (default: 4320)
BV2_RES_PERCENT=100              # resolution percentage
BV2_SAMPLES=64                   # Cycles samples (default: 64)
BV2_RENDER_TAG=8k                # output subfolder tag (default: "8k")
BV2_CAMERA_NAME=<name>           # override camera (defaults per contract)
BV2_LOG_PATH=<path>              # write build log to file
```

### Inputs

- Template blend: `_data-refactored/blenderv2/bV2_template.blend`
- State VTKs from Step 1b: `_data-refactored/model-outputs/generated-states/<root>/output/vtks/{site}/`
- Bioenvelope PLYs from Step 1b: `_data-refactored/model-outputs/generated-states/<root>/output/bioenvelopes/{site}/`
- NodeDF CSVs from Step 1b: `_data-refactored/model-outputs/generated-states/<root>/output/feature-locations/{site}/`

### Outputs

- Scene blend: `_data-refactored/blenderv2/blends/` (or sim-root-derived path when `BV2_SIM_ROOT` is set)
- Multi-layer EXRs: `_data-refactored/blenderv2/output/<sim_root>/<exr_family>/`

---

## 3. Generate PNGs via Compositor

Blender compositor pipeline that takes EXR families and produces final PNG outputs per compositor template.

**This code works.** If you get wrong paths, missing files, or Blender errors, you are doing something wrong — do not assume the code is broken.

**Start here**: [COMPOSITOR_RUN.md](_futureSim_refactored/blender/compositor/COMPOSITOR_RUN.md) — quick-start with family→script→blend mapping.
**Directory structure + workflow**: [README.md](_futureSim_refactored/blender/compositor/README.md)
**Template contract**: [COMPOSITOR_TEMPLATE_CONTRACT.md](_futureSim_refactored/blender/compositor/COMPOSITOR_TEMPLATE_CONTRACT.md)

### Family → script → blend

| Family | Runner | Canonical blend |
|---|---|---|
| `ao` | `render_edge_lab_current_core_outputs.py` | `compositor_ao.blend` |
| `normals` | `render_edge_lab_current_core_outputs.py` | `compositor_normals.blend` |
| `resources` | `render_edge_lab_current_core_outputs.py` | `compositor_resources.blend` |
| `base` | `render_edge_lab_current_base.py` | `compositor_base.blend` |
| `shading` | `render_edge_lab_current_shading.py` | `compositor_shading.blend` |
| `bioenvelope` | `render_edge_lab_current_bioenvelopes.py` | `compositor_bioenvelope.blend` |
| `sizes` | `render_edge_lab_current_sizes.py` | `compositor_sizes.blend` |
| `mist` | `render_edge_lab_current_mist.py` | `compositor_mist.blend` |
| `depth_outliner` | `render_edge_lab_current_depth_outliner.py` | `compositor_depth_outliner.blend` |
| `proposals` | `render_edge_lab_current_proposals.py` | `compositor_proposal_masks.blend` |

Scripts live in: `_futureSim_refactored/blender/compositor/scripts/`
Canonical blends live in: `_futureSim_refactored/blender/compositor/canonical_templates/`

### Inputs and outputs

- **Input EXRs**: `_data-refactored/blenderv2/output/<sim_root>/<exr_family>/`
- **Output PNGs**: `_data-refactored/compositor/outputs/<sim_root>/<exr_family>/<compositor_run>/`
- `<compositor_run>` = `<family>__<timestamp>` (optional `__<note>`)

### Env var pattern

Runners read env vars — set these, don't edit paths in code:

```bash
COMPOSITOR_BLEND_PATH=<path to working copy of canonical blend>
COMPOSITOR_OUTPUT_DIR=<full output dir for this run>
COMPOSITOR_SCENE_NAME=Current
COMPOSITOR_PATHWAY_EXR=<path>       # positive_state
COMPOSITOR_PRIORITY_EXR=<path>      # positive_priority_state
COMPOSITOR_EXISTING_EXR=<path>      # existing_condition_positive
COMPOSITOR_TRENDING_EXR=<path>      # trending_state (omit for baseline)
COMPOSITOR_BIOENVELOPE_EXR=<path>   # bioenvelope_positive (omit for baseline)
```

### Typical flow

1. Copy canonical blend → working copy in `_data-refactored/compositor/temp_blends/template_instantiations/`
2. Set env vars (above)
3. Run: `blender --background --factory-startup --python scripts/render_edge_lab_current_<family>.py`
4. Verify PNGs landed in `COMPOSITOR_OUTPUT_DIR`

### Key rules

- **Canonical blend owns the graph.** Runners only repath inputs and render — never rebuild graph logic in a runner.
- **No hidden fallbacks.** If a resolution, EXR, or node is missing, raise — don't hardcode.
- **Positive and trending are separate branches.** Never combine them in one run.
- **Working copies go in `temp_blends/`** — never save over the canonical blend.
- **Read EXR dimensions from the header** (`_exr_header.py`) — `image.size` returns `(0, 0)` in Blender 4.x.

---

## 4. Sync to Mediaflux

Upload/download simulation outputs, EXR families, and compositor runs to the University of Melbourne Mediaflux research archive.

**Contract**: [MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md)
**Full rules**: see `CLAUDE.md` Mediaflux section — this is the authoritative reference for all sync commands.

### Remote path mapping

| Local path under `_data-refactored/` | Remote subpath |
|---|---|
| `model-outputs/generated-states/<sim_root>/output/` | `pipeline/<sim_root>/simulation_outputs/output` |
| `blenderv2/output/<sim_root>/<exr_family>/` | `pipeline/<sim_root>/blender_exrs/<exr_family>` |
| `compositor/outputs/<sim_root>/<exr_family>/<compositor_run>/` | `pipeline/<sim_root>/compositor_pngs/<exr_family>/<compositor_run>` |

### Simulation-specific helper

```bash
# Upload a full sim run
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload <sim_root>

# Download a full sim run
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync download <sim_root>

# Include temp/ and comparison/ debug artifacts
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload <sim_root> --include-debug
```

### Key rules

- **Always use `mediafluxsync`** — never `rsync`, `cp`, or raw `unimelb-mf-download`, even when Mediaflux is mounted locally.
- **`--out` is the parent directory** — the client creates a subfolder inside it.
- **Remote subpath always starts with `pipeline/`** — never upload to a bare path.
- **Append `--dry-run`** to preview any transfer without executing it.
- Browse remote contents via the mounted-volume browser: `uv run python -m _futureSim_refactored.sim.run.mediaflux_browse --last 5`

## Utilities

### vtk_extract — fast partial VTK reader

`_futureSim_refactored.sim.vtk_extract` extracts individual point-data arrays from legacy binary VTK files without loading the full mesh (~10x faster than `pv.read`). Refer to the module docstring and `KNOWN_COLUMNS` registry in [vtk_extract.py](_futureSim_refactored/sim/vtk_extract.py) for API details and supported dtypes.
