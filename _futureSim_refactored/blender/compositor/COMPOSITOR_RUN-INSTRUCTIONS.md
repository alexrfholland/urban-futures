# Compositor — Run Instructions

Practical how-to for running the compositor. For deeper rules see:

- [COMPOSITOR_TEMPLATE_CONTRACT.md](COMPOSITOR_TEMPLATE_CONTRACT.md) — what blends own vs what scripts own.
- [COMPOSITOR_SYNC_CONTRACT.md](COMPOSITOR_SYNC_CONTRACT.md) — sim_root / exr_family / compositor_run path lineage.
- [MEDIAFLUX_SYNC_CONTRACT.md](/_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md) — remote upload/download layout.

`COMPOSITOR_RUN.md` is the older reflex-order note and uses the legacy `render_edge_lab_current_*` names. Prefer this file.

---

## 1. How the pieces fit

```
_futureSim_refactored/blender/compositor/
├── canonical_templates/              # authoritative .blend files — own the graph
│   ├── compositor_ao.blend, compositor_normals.blend, ... (one per family)
│   ├── proposal_only_layers.blend, size_outline_layers.blend, ...
│   └── _archive/                     # superseded / abandoned — DO NOT USE (see README there)
├── scripts/
│   ├── render_current_<family>.py    # thin per-family runner (env-var driven)
│   ├── _fast_runner_core.py          # shared helper: open blend, repath EXRs, render
│   ├── _exr_header.py                # reads EXR displayWindow (avoids img.size == 0,0)
│   ├── _session_runner.py            # loops N runs inside ONE Blender process
│   ├── batch_parade_timeline_4_10.py # batch driver (serial or parallel)
│   └── _archive/                     # legacy edge_lab runners — DO NOT USE
└── COMPOSITOR_*.md                   # docs

_data-refactored/compositor/
├── temp_blends/                      # working copies + dev variants; never overwrite canonical
│   ├── template_development/         # unpromoted experiments
│   └── template_instantiations/      # per-batch working copies with pre-set EXR paths
└── outputs/<sim_root>/<exr_family>/<compositor_run>/*.png
```

**Data flow (one run):**

```
EXR (bV2 output)  →  canonical .blend  →  runner (env-driven)  →  PNG slots in outputs/
```

---

## 2. Family → runner → canonical blend

| Family                            | Runner                                          | Canonical blend                            | Inputs              |
|-----------------------------------|-------------------------------------------------|--------------------------------------------|---------------------|
| `ao`                              | `render_current_ao.py`                          | `compositor_ao.blend`                      | 3 EXRs              |
| `normals`                         | `render_current_normals.py`                     | `compositor_normals.blend`                 | 3 EXRs              |
| `resources`                       | `render_current_resources.py`                   | `compositor_resources.blend`               | 3 EXRs              |
| `bioenvelope`                     | `render_current_bioenvelope.py`                 | `compositor_bioenvelope.blend`             | 3 EXRs              |
| `shading`                         | `render_current_shading.py`                     | `compositor_shading.blend`                 | 6 EXRs              |
| `base`                            | `render_current_base.py`                        | `compositor_base.blend`                    | 1 EXR               |
| `mist`                            | `render_current_mist.py`                        | `compositor_mist.blend`                    | 1 EXR (per branch)  |
| `depth_outliner`                  | `render_current_depth_outliner.py`              | `compositor_depth_outliner.blend`          | 1 EXR (per branch)  |
| `intervention_int`                | `render_current_intervention_int.py`            | `compositor_intervention_int.blend`        | 1 EXR (bio)         |
| `proposal_only`                   | `render_current_proposal_only.py`               | `proposal_only_layers.blend`               | 1 EXR (per branch)  |
| `proposal_outline`                | `render_current_proposal_outline.py`            | `proposal_outline_layers.blend`            | 1 EXR (per branch)  |
| `proposal_colored_depth_outlines` | `render_current_proposal_colored_depth_outlines.py` | `proposal_colored_depth_outlines.blend`    | 1 EXR (per branch)  |
| `proposal_and_interventions`      | `render_current_proposal_and_interventions.py`  | `proposal_and_interventions.blend`         | 1 EXR (per branch)  |
| `size_outline`                    | `render_current_size_outline.py`                | `size_outline_layers.blend`                | 1 EXR (per branch)  |
| `sizes_single_input`              | `render_current_sizes_single_input.py`          | `compositor_sizes_single_input.blend`      | 1 EXR (per branch)  |

"Per branch" means run once for each of `positive_state`, `trending_state`, and (where relevant) `positive_priority_state`.

---

## 3. Paths (contract summary)

```
_data-refactored/blenderv2/output/<sim_root>/<exr_family>/*.exr         ← input EXRs
_data-refactored/compositor/outputs/<sim_root>/<exr_family>/<run>/      ← output PNGs

pipeline/<sim_root>/blender_exrs/<exr_family>/                          ← remote EXRs
pipeline/<sim_root>/compositor_pngs/<exr_family>/<family>/<run>/        ← remote PNGs
```

`<run>` = `<family>__<timestamp>[__<branch>]`, e.g. `mist__20260413_114049__positive`.

---

## 4. Running one family

Every runner reads env vars. The generic single-input contract is:

```bash
COMPOSITOR_EXR=<abs path to one state EXR>
COMPOSITOR_OUTPUT_DIR=<abs path to output dir>
```

Multi-input runners (ao, normals, resources, bioenvelope, shading) use family-specific names — see the docstring at the top of each `render_current_<family>.py`.

Then:

```bash
".venv/Scripts/python.exe" -m ...    # no — compositor runs IN Blender, not venv
"C:/Program Files/Blender Foundation/Blender 4.2/blender.exe" \
  --background --factory-startup \
  --python _futureSim_refactored/blender/compositor/scripts/render_current_mist.py
```

Windows gotcha: spaces in the project path are fine, but wrap in double quotes.

---

## 5. Batching — the multitasker

The batch driver lives at [scripts/batch_parade_timeline_4_10.py](scripts/batch_parade_timeline_4_10.py) (name is historical — it generalises via flags).

### 5a. Dry-run the plan

```bash
.venv/Scripts/python.exe _futureSim_refactored/blender/compositor/scripts/batch_parade_timeline_4_10.py \
  --sim-root 4.10 --exr-family city_timeline --dry-run
```

Prints the 27-run plan without spawning Blender.

### 5b. Serial (one Blender subprocess per run)

```bash
.venv/Scripts/python.exe _futureSim_refactored/blender/compositor/scripts/batch_parade_timeline_4_10.py \
  --sim-root 4.10 --exr-family city_timeline
```

Slow — each run pays ~20s Blender cold start. Useful when debugging a single run.

### 5c. Parallel + session-wrapped (the normal mode)

```bash
.venv/Scripts/python.exe _futureSim_refactored/blender/compositor/scripts/batch_parade_timeline_4_10.py \
  --sim-root 4.10 --exr-family city_timeline --parallel 4
```

How it works:

1. The driver builds the 27-run plan and shards it round-robin into 4 worker chunks.
2. For each chunk it writes a JSON manifest to a temp dir.
3. It spawns 4 parallel `blender --background --python _session_runner.py -- <manifest.json>`.
4. Each Blender session loops: `open_mainfile → repath EXR → render → audit` per entry in its manifest. One cold start per worker, not per run.
5. The driver waits on all workers, reads their `.result.json`s, and prints a single summary.

Observed timings on this machine (Xeon 8358, 248 GB, L40):

| Mode              | 27-run wall clock | Speedup |
|-------------------|-------------------|---------|
| Serial            | ~15 min           | 1×      |
| `--parallel 2`    | ~8 min            | ~1.9×   |
| `--parallel 4`    | ~7 min            | ~2.2×   |

Past 4 workers the compositor's own threading + memory bandwidth bottleneck kicks in — diminishing returns. 3–4 workers is the sweet spot on an 8K batch.

### 5d. Subsetting — a couple at a time

```bash
# only specific families (any mix)
--only mist,depth_outliner,shading

# all except some
--skip ao,normals

# combined with parallel
--parallel 2 --only proposal_only,proposal_outline
```

`--only` / `--skip` are comma-separated family names. They match the "Family" column in §2.

### 5e. Per-worker logs

Each worker streams stdout/stderr to a file in a temp dir printed at spawn time, e.g.:

```
C:\Users\<you>\AppData\Local\Temp\parade_batch_<ts>_<hash>\worker_<i>.log
```

Open those when diagnosing a failed run.

---

## 6. Where new work goes (living-doc policy)

**The short version:** `canonical_templates/` is sacred. Everything in-flight lives in `_data-refactored/compositor/temp_blends/`.

```
_futureSim_refactored/blender/compositor/
├── canonical_templates/*.blend          # CANONICAL — referenced by a runner, in §2 table
├── scripts/
│   ├── render_current_<family>.py       # CANONICAL — thin env-driven runner
│   ├── _fast_runner_core.py, _session_runner.py, _exr_header.py   # CANONICAL helpers
│   └── batch_parade_timeline_4_10.py    # CANONICAL batch driver
└── COMPOSITOR_*.md                      # CANONICAL docs — this file is the entry point

_data-refactored/compositor/temp_blends/
├── template_development/<topic>_<date>/ # experimental blends while designing a new graph
├── template_instantiations/<batch>/     # per-batch working copies (EXR paths pre-baked)
└── checkpoints/                         # snapshots of canonicals before risky edits
```

### When you're doing new compositor work

1. **Experimental blend work** → `temp_blends/template_development/<topic>_<YYYYMMDD>/`.
   Never save a working copy into `canonical_templates/` with a `__working` suffix.
2. **Ad-hoc scripts** (repair a single blend, inspect a node tree, one-off fix) → put them in `scripts/` with a leading underscore and a date suffix: `_fix_<thing>_<YYYYMMDD>.py`. The underscore signals "not a runner". Delete after the fix lands or when its purpose expires.
3. **Documenting a finding** → edit this file or the relevant `COMPOSITOR_*.md`. Do not drop a new `NOTES_<topic>.md` at the root. If the finding is a known-issue/workaround, it belongs in §7 below.
4. **Promoting an experiment to canonical** requires *all three*:
   - move the blend into `canonical_templates/` with its final name
   - wire it to a `render_current_<family>.py` runner (thin — see §7a)
   - add/update a row in the §2 family table in this file
   Without all three, it isn't canonical.

### What doesn't belong in the repo

- Temp scripts that nobody else will ever run again — delete, don't commit.
- Notes-to-self markdown files — put them in a commit message or this doc.
- Working-copy .blends with per-date suffixes — those go under `temp_blends/`.

---

## 7. Known issues and common mistakes to avoid

### 7a. Runners are thin — do not rebuild graph logic in Python

A runner's job is: read env vars → call `_fast_runner_core.run_fast_render()` → done. Everything about the graph (slots, links, formats, node groups) belongs in the canonical .blend. If you find yourself adding `bpy.ops` node manipulation to a runner, you are doing the wrong thing — fix the blend instead.

The only exceptions are the workarounds in §7c (rebuild File Output node) and §7e (dummy camera), and both live in `_fast_runner_core.py`, not in per-family runners.

### 7b. One EXR per compositor, all passes in one render

Each canonical blend takes **one EXR** (or for multi-input families, one fixed set of EXRs) and renders **every pass it owns in a single render call**. Do not:

- Loop over EXRs inside a runner (one EXR per compositor run — batch at the driver level).
- Render slot-by-slot in a loop (fire all File Output slots in one `bpy.ops.render.render` call).
- Iterate passes (if a blend owns N slots, one render produces N PNGs — if it doesn't, the blend is broken, fix the blend).

Iterating inside a runner multiplies Blender cold-start cost by N and reintroduces the slot-miss bugs this whole system was built to avoid.

### 7c. Render Execution Rule — use `animation=True`, not `write_still=True`

Blender 4.2's `write_still=True` path intermittently skips File Output nodes in compositor-only scenes, producing 0 or partial slot output.

**Rule:** render as a single-frame animation.

```python
scene.frame_start = 1
scene.frame_end   = 1
scene.frame_current = 1
bpy.ops.render.render(animation=True, scene=scene.name)
```

Then strip the `_0001` suffix the animation path appends. The shared helper [`_fast_runner_core.run_fast_render()`](scripts/_fast_runner_core.py) already does this — every runner goes through it.

### 7d. Group → File Output slots don't fire (Blender 4.2)

When a saved `CompositorNodeOutputFile` slot is fed **directly** from a `CompositorNodeGroup` output (no intermediate node), Blender 4.2 sometimes does not evaluate that slot — render returns 0 slot PNGs with no error.

**Workaround:** `FastRunnerConfig(rebuild_file_output=True)`.

This rebuilds the File Output node fresh in-memory (same slot paths, same links, same format) before rendering. `_fast_runner_core._rebuild_file_output()` handles it. Turn it on per-runner when needed — currently used by:

- `render_current_mist.py`
- `render_current_depth_outliner.py`
- `render_current_proposal_and_interventions.py`

If you add a new single-input runner and it returns 0/N slots despite a clean log, add `rebuild_file_output=True` to its config.

**Do not** swap to `write_still`, per-slot loops, or mute/unmute dances — those are all explicitly forbidden by [COMPOSITOR_TEMPLATE_CONTRACT.md](COMPOSITOR_TEMPLATE_CONTRACT.md#render-execution-rule).

### 7e. EXR resolution: read the header, never default

`bpy.data.images.load(path).size` returns `(0, 0)` in Blender 4.x until something forces a decode. Runners must read the EXR `displayWindow` via [`_exr_header.read_exr_dimensions()`](scripts/_exr_header.py) and raise if it can't. The helper does this already.

Silent fallbacks to 4K have ruined entire 8K batches historically — see Hidden Fallback Rule in the template contract.

### 7f. Dummy camera for compositor-only scenes

`animation=True` requires `scene.camera`. Compositor-only canonicals don't always have one. `_fast_runner_core._ensure_camera()` attaches an existing camera or a transient dummy when needed — no action required by runners.

### 7g. Slot-miss = silent Blender exception

Blender returns rc=0 even when the Python runner raised (e.g. a missing env var). The batch driver now also checks that `>0 PNGs` land in the output dir — a 0-PNG dir is treated as FAIL regardless of exit code.

---

## 8. Upload to Mediaflux

Follow [CLAUDE.md](/CLAUDE.md#mediaflux) — short version:

```bash
# Windows env prerequisites (both required)
export PATH="/d/2026 Arboreal Futures/urban-futures/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/jre/bin:$PATH"
export MEDIAFLUX_CLIENT_BIN_DIR="D:/2026 Arboreal Futures/urban-futures/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/bin/windows"

# one run
.venv/Scripts/python.exe -m mediafluxsync upload-project \
  --create-parents --exclude-parent --project-dir . \
  "_data-refactored/compositor/outputs/<sim_root>/<exr_family>/<run>" \
  "pipeline/<sim_root>/compositor_pngs/<exr_family>/<family>/<run>"
```

Full sim-level helper: `python -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload <sim_root>`.

---

## 9. Typical flow recap

1. Verify EXRs exist under `_data-refactored/blenderv2/output/<sim_root>/<exr_family>/`.
2. Dry-run the batch to confirm the plan: `--dry-run`.
3. Run parallel: `--parallel 4 --sim-root <x> --exr-family <y>`.
4. Check the summary — every row should be `OK`.
5. Spot-check one output dir (`ls -la <out>/*.png`) to confirm PNG counts match the blend's slot count.
6. Upload to Mediaflux when the user asks.
