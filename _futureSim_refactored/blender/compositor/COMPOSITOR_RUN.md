# Compositor Run — Quick Direction

Execution rule: run compositor renders from the repo root in Blender directly. Do not use `uv run python` for the Blender step.

Short note for the agent. **Don't overthink it.** When the user says "render X for Y EXRs",
this file is the first thing to read.

## 1. Reflex order

1. **Look for an existing `render_current_*.py`** in [scripts/](scripts/) that matches
   the family. Reuse it. Do not write a new runner.
2. If the blend's EXR hooks don't line up 1:1 with the available EXRs,
   **ask the user** how to wire them (see [COMPOSITOR_SYNC_CONTRACT.md](COMPOSITOR_SYNC_CONTRACT.md) → *Input Wiring On Mismatch*).
3. Run it with env vars — do not hack the script to change paths.

## 2. Family → script → blend

| Family        | Runner                                  | Canonical blend                          |
|---------------|-----------------------------------------|------------------------------------------|
| `ao`          | `render_current_ao.py`                    | `compositor_ao.blend`                    |
| `normals`     | `render_current_normals.py`               | `compositor_normals.blend`               |
| `resources`   | `render_current_resources.py`             | `compositor_resources.blend`             |
| `base`        | `render_current_base.py`                  | `compositor_base.blend`                  |
| `shading`     | `render_current_shading.py`               | `compositor_shading.blend`               |
| `bioenvelope` | `render_current_bioenvelope.py`           | `compositor_bioenvelope.blend`           |
| `sizes`       | `render_current_sizes_single_input.py`    | `compositor_sizes_single_input.blend`    |
| `mist`        | `render_current_mist.py`                  | `compositor_mist.blend`                  |
| `mist_complex_outlines` | `render_current_mist_complex_outlines.py` | `compositor_mist_complex_outlines.blend` |
| `depth_outliner` | `render_current_depth_outliner.py`    | `compositor_depth_outliner.blend`        |
| `proposals`   | `render_current_proposal_only.py` / friends | proposal blends                       |

The canonical `.blend` owns the graph. The runner is a thin wrapper that repaths inputs and
renders. **Never rebuild graph logic in the runner.**

## 3. Where things go

Lineage: `sim_root → exr_family → compositor_run`. Full rules in [COMPOSITOR_SYNC_CONTRACT.md](COMPOSITOR_SYNC_CONTRACT.md).

- **Input EXRs (local):** `_data-refactored/blenderv2/output/<sim_root>/<exr_family>/`
- **Input EXRs (remote):** `pipeline/<sim_root>/blender_exrs/<exr_family>/`
- **Output PNGs (local):** `_data-refactored/compositor/outputs/<sim_root>/<exr_family>/<compositor_run>/`
- **Output PNGs (remote):** `pipeline/<sim_root>/compositor_pngs/<exr_family>/<compositor_run>/`

`<compositor_run>` = `<family>__<timestamp>` (optional `__<note>`), e.g. `sizes__20260411_1621__2input-positive-priority`.

## 4. Env var pattern

The runners read these. Set them; don't edit paths in code.

```bash
COMPOSITOR_BLEND_PATH=<path to working copy of canonical blend>
COMPOSITOR_OUTPUT_DIR=<full output dir for this run>
COMPOSITOR_SCENE_NAME=Current
COMPOSITOR_PATHWAY_EXR=<path>       # positive_state
COMPOSITOR_PRIORITY_EXR=<path>      # positive_priority_state
COMPOSITOR_EXISTING_EXR=<path>      # existing_condition_positive
COMPOSITOR_TRENDING_EXR=<path>      # trending_state (omit / dummy for baseline)
COMPOSITOR_BIOENVELOPE_EXR=<path>   # bioenvelope_positive (omit / dummy for baseline)
# ...trending / bioenvelope_trending as required by the blend
COMPOSITOR_OUTPUT_FILTER=slotA,slotB  # optional — only render these slots
```

Baseline has **no trending, no bioenvelope** — only `existing_condition_positive`,
`positive_state`, `positive_priority_state`. If the blend wants more hooks, stop and ask.

## 5. Hard rules (don't break these)

- **No hidden fallbacks.** If a resolution, EXR, or node is missing, raise — don't hardcode.
- **Read EXR dimensions from the header** ([_exr_header.py](scripts/_exr_header.py)) — `image.size` returns `(0, 0)` in Blender 4.x.
- **Positive and trending are separate branches.** Never combine them in one run.
- **Working copies go in `_data-refactored/compositor/temp_blends/`** — never save over the canonical blend.

## 6. Typical flow

1. Copy canonical blend → working copy under `_data-refactored/compositor/temp_blends/template_instantiations/`.
2. Set env vars (section 4).
3. Run: `blender --background --factory-startup --python scripts/render_current_<family>.py`.
4. Verify PNGs landed in `COMPOSITOR_OUTPUT_DIR`.
5. If the user says upload: `mediafluxsync upload-project --create-parents --exclude-parent --project-dir . <local> <remote>`.
