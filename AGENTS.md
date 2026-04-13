# AGENTS

## 0. Runtime Setup

- Branch: `engine-v4`
- Always use `uv` and the repo-local `.venv`
  - `.\uv.cmd run python ...` on Windows or `./uv run python ...` on macOS/Linux for project Python commands
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

Mediaflux overview:

- Use the shared `mediafluxsync` package from this repo's `.venv` for all transfers.
- Do not rely on a Codex-only skill or on raw `unimelb-mf-*` commands.
- Use [MEDIAFLUX.md](MEDIAFLUX.md) for machine setup, bootstrap, and command examples.
- Use [MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md) for the path contract under `pipeline/<sim_root>/...`.
- Use `mediaflux_browse` for fast mounted discovery and `mediafluxsync` for actual upload/download work.

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
- For bV2 EXR families, default to EXR-only sync. Do not transfer `__full_pipeline.blend`, `__manifest.txt`, or other metadata sidecars unless the user explicitly asks for them.
- For default bV2 family sync, use `uv run python -m _futureSim_refactored.blender.blenderv2.bV2_mediaflux_sync <upload|download> <sim_root> <exr_family>`. Add `--include-metadata` only when you intentionally want the `.blend` / manifest sidecars too.

## Utilities

### vtk_extract — fast partial VTK reader

`_futureSim_refactored.sim.vtk_extract` extracts individual point-data arrays from legacy binary VTK files without loading the full mesh (~10x faster than `pv.read`). Refer to the module docstring and `KNOWN_COLUMNS` registry in [vtk_extract.py](_futureSim_refactored/sim/vtk_extract.py) for API details and supported dtypes.


#FURTHER CLAUDE INSTRUCTIONS

## Project

`urban-futures` is the simulation engine and data pipeline for the 2026 Arboreal Futures project. Source lives in `_code-refactored/`. The canonical local pipeline data now lives repo-locally under `_data-refactored/`. Legacy source/reference material still exists under `data/`.


## Mediaflux

This project uses the shared `mediafluxsync` package (sibling repo at `../mediafluxsync/`) to upload/download data to the University of Melbourne Mediaflux research archive.

### Hard rules

1. **Always transfer data via `mediafluxsync`.** Never use `rsync`, `cp`, or raw `unimelb-mf-download` — even when the Mediaflux project is mounted locally. The mount is for discovery only.
2. **Discovery uses the mounted-volume browser** (see [_futureSim_refactored/sim/run/mediaflux_browse.py](_futureSim_refactored/sim/run/mediaflux_browse.py)), not `check-project`:

   ```bash
   uv run python -m _futureSim_refactored.sim.run.mediaflux_browse --last 5
   uv run python -m _futureSim_refactored.sim.run.mediaflux_browse <sim_root> --map
   uv run python -m _futureSim_refactored.sim.run.mediaflux_browse <sim_root> --section compositor_pngs --map
   ```

3. **Use `uv run python -m mediafluxsync ...`** — this works on both macOS and Windows. Do not hard-code `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS).
4. **Run from the repo root with `--project-dir .`**. Mediaflux paths are always relative to the project root (e.g. `pipeline/<sim_root>/...`).
5. **`download-project --out` is the PARENT directory**, not the final target. `unimelb-mf-download` creates a folder named after the remote path's basename inside `--out`. So to download `pipeline/4.10/compositor_pngs/city_single-state_yr180` into `_data-refactored/compositor/outputs/4.10/city_single-state_yr180/`, pass `--out ./_data-refactored/compositor/outputs/4.10` (one level up). Passing the full target path double-nests.
6. **Canonical sync layouts** are defined in [_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md) and [_documentation-refactored/MACHINE_SETUP_MEDIAFLUX.md](_documentation-refactored/MACHINE_SETUP_MEDIAFLUX.md). Mirror them locally — only the base root differs between local and remote.

### Project paths

- Allocation: `/projects/proj-7020_research_archive-1128.4.442`
- Project root: `MF 2026 Arboreal Futures`
- Remote tree: `pipeline/<sim_root>/{simulation_outputs,blender_exrs,compositor_pngs}/...`
- Local mirror: `_data-refactored/{model-outputs/generated-states,blenderv2,compositor/outputs}/<sim_root>/...`

Verify project config any time:

```bash
uv run python -m mediafluxsync project-config --project-dir .
uv run python -m mediafluxsync project-path pipeline --project-dir .
```

### Client binary (per-platform)

- **macOS**: `/opt/homebrew/bin/unimelb-mf-download` (installed via homebrew). No `MEDIAFLUX_CLIENT_BIN_DIR` needed.
- **Windows**: bundled under `.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/bin/windows/`. Set `MEDIAFLUX_CLIENT_BIN_DIR` to that path before running mediafluxsync.

### Config files

- Credentials: `~/.Arcitecta/mflux.cfg` (macOS/Linux) or `%USERPROFILE%\.Arcitecta\mflux.cfg` (Windows)
- Shared defaults: `../mediafluxsync/.env.mediaflux`
- Project paths: `.env.mediaflux` (this repo)
- Secrets: `../mediafluxsync/.env` (gitignored)

### Allowed subcommands

Only these `mediafluxsync` subcommands exist:

- `upload-project`
- `download-project`
- `check-project`
- `project-config`
- `project-path`

`ls-project` and `exists-project` were removed. Do not use them.

### Environment setup (Windows only)

Both env vars must be set before any mediafluxsync command on Windows:

```bash
export PATH="/d/2026 Arboreal Futures/urban-futures/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/jre/bin:$PATH"
export MEDIAFLUX_CLIENT_BIN_DIR="D:/2026 Arboreal Futures/urban-futures/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/bin/windows"
```

Without the JRE on PATH, every client command fails with `cannot find java`.

### Remote path contract

The Mediaflux remote tree lives under `pipeline/`. The local-to-remote mapping is:

| Local path under `_data-refactored/` | Remote subpath |
|---|---|
| `model-outputs/generated-states/<sim_root>/output/` | `pipeline/<sim_root>/simulation_outputs/output` |
| `blenderv2/output/<sim_root>/<exr_family>/` | `pipeline/<sim_root>/blender_exrs/<exr_family>` |
| `compositor/outputs/<sim_root>/<exr_family>/<compositor_run>/` | `pipeline/<sim_root>/compositor_pngs/<exr_family>/<compositor_run>` |

Non-simulation assets go under `pipeline/_library/` or `pipeline/_site/` (underscore prefix).

**Critical rule:** the remote subpath always starts with `pipeline/`. Never upload to a bare `compositor/...` or `blenderv2/...` path.

### Common commands

Always run from this repo root. Append `--dry-run` to any transfer to preview the underlying `unimelb-mf-*` invocation without executing it.

```bash
# Upload from a local dir into pipeline/<subpath> (exclude the local parent folder name)
uv run python -m mediafluxsync upload-project --create-parents --exclude-parent --project-dir . <local-source> <subpath>

# Upload into project data/ (includes the local parent folder)
uv run python -m mediafluxsync upload-project --create-parents --project-dir . <local-source> <subpath>

# Download a remote subpath into a local PARENT dir
# (the binary creates <basename-of-subpath>/ inside --out)
uv run python -m mediafluxsync download-project --project-dir . --out <local-parent-dir> <subpath>
```

Simulation-specific helper (wraps the run-root contract):

```bash
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload <sim_root>
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync download <sim_root>
# add --include-debug to additionally sync temp/ and comparison/
```

See [MEDIAFLUX.md](MEDIAFLUX.md) for full details and [MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md) for the authoritative path lineage contract.
