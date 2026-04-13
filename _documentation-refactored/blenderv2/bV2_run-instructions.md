# bV2 Run Instructions

Entry point for Step 2 (generate EXRs):
[bV2_build_scene.py](../../_futureSim_refactored/blender/blenderv2/bV2_build_scene.py).

The script runs end-to-end: opens the template, initializes the scene,
builds instancers/bioenvelopes/world attributes from the sim bundle,
validates against [bV2_scene_contract.py](../../_futureSim_refactored/blender/blenderv2/bV2_scene_contract.py),
saves the pipeline `.blend`, renders one EXR per view layer, optionally
uploads to Mediaflux.

## The correct invocation

Use the checked-in launcher from repo root.

Windows PowerShell:

```powershell
.\scripts\run-bv2.ps1 -Site trimmed-parade -Mode timeline -SimRoot 4.10 -DataBundleRoot .\_data-refactored\model-outputs\generated-states\4.10\output
```

macOS / Linux:

```bash
./scripts/run-bv2.sh --site trimmed-parade --mode timeline --sim-root 4.10 --data-bundle-root ./_data-refactored/model-outputs/generated-states/4.10/output
```

For `single_state` add `-Year 180` on Windows or `--year 180` on macOS / Linux.
For `baseline`, year is optional.

## Required inputs

| Env var | Source of the value |
|---|---|
| `BV2_SITE` | One of `city`, `trimmed-parade`, `uni` (from `SUPPORTED_SITES` in the contract) |
| `BV2_MODE` | One of `single_state`, `timeline`, `baseline` (from `SUPPORTED_MODES`) |
| `BV2_SIM_ROOT` | The sim version folder name (e.g. `4.10`) — determines all output paths |
| `BV2_DATA_BUNDLE_ROOT` | Absolute path to the sim's `output/` folder containing VTKs, PLYs, nodeDFs |

---

## Mistakes to avoid

### DO NOT use `uv run python`

Blender's `bpy` module only works inside a Blender process. The script must
be executed via `blender.exe --background --python <script>`. `uv run` has
no way to load `bpy`. The project `.venv` is *only* exposed via
`PYTHONPATH` so that project imports resolve. The checked-in launcher sets
`PYTHONPATH` for you.

### DO NOT pass `--factory-startup` to Blender

`--factory-startup` strips USER_SITE from `sys.path`, and pandas/vtk live
in Blender's USER_SITE (`%APPDATA%/Python/Python311/site-packages/`).
Passing `--factory-startup` breaks the build with an ImportError.

### DO NOT install pandas/vtk into the project `.venv` and expect Blender to pick them up

Blender 4.2 ships its own Python 3.11 interpreter. Its compiled-extension
ABI is incompatible with whatever Python the repo's `.venv` was created
against. Install pandas/vtk **using Blender's own bundled `pip`** so they
land in Blender's USER_SITE. The `.venv` entry in `PYTHONPATH` is a safety
net for pure-Python packages only.

### DO NOT pass the template `.blend` on the command line

The script opens `_data-refactored/blenderv2/bV2_template.blend` itself
via `open_template_blend()`. Passing a blend to `blender.exe` as a
positional argument loads the wrong scene and the script's
`wm.open_mainfile` call then overwrites it silently.

### DO NOT skip `BV2_DATA_BUNDLE_ROOT`

Without it, `resolve_bioenvelope_ply_path` can't find the generated-state
PLYs and the build silently produces an empty-bioenvelope blend that
renders normally but has no bioenvelope geometry. This failure is
**silent** — validation passes, EXRs render — so always set this env var
explicitly when kicking off a run. Prior incident logged in
[_logs/rebuild_city_yr180_20260411.sh](../../_logs/rebuild_city_yr180_20260411.sh).

### DO NOT write a one-off wrapper shell script for every run

Existing `_logs/rebuild_*.sh` files are historical; don't add to them.
Use the checked-in launcher in `scripts/run-bv2.ps1` or `scripts/run-bv2.sh`.
Do not create a new ad-hoc script for each render.

### DO NOT use `parade` as `BV2_SITE`

The canonical site name is `trimmed-parade`. `parade` alone is not in
`SUPPORTED_SITES` and will raise `Unsupported site`.

### DO NOT guess the `<exr_family>` slug

`get_runtime_exr_family(scene)` computes it. Observed values:

| Site / mode / year | `<exr_family>` |
|---|---|
| `trimmed-parade` / `timeline` | **`parade_timeline`** (`trimmed-` is stripped) |
| `city` / `single_state` / `180` | `city_single-state_yr180` |
| `city` / `baseline` | `city_baseline` |

Read the function if you need a new site's slug — do not infer.

### DO NOT expect a `<timestamp>_<tag>/` subfolder under the render root

The render folder is flat:
`_data-refactored/blenderv2/output/<sim_root>/<exr_family>/`.
Each EXR is named `<exr_family>__<view_layer>__<tag>.exr`. A re-run with
the same tag **overwrites** the existing EXRs. Change `BV2_RENDER_TAG` or
move the previous folder aside if you need to preserve an earlier run.

### DO NOT assume the manifest is `manifest.json`

It is written as `<exr_family>__manifest.txt` alongside the EXRs.

### DO NOT forget `PYTHONPATH` at the repo root

`bV2_build_scene.py` imports `_futureSim_refactored.paths` and the sibling
`bV2_*` modules. Without the repo root on `PYTHONPATH`, the import fails
before any scene work starts. The checked-in launcher sets this for you.

---

## Re-rendering the same (site, mode, sim_root)

Outputs for a given sim run land in a **flat folder** keyed by
`BV2_SIM_ROOT` + `<exr_family>`:

```
_data-refactored/blenderv2/output/<sim_root>/<exr_family>/
    <exr_family>__<view_layer>__<tag>.exr
```

The **only** run-versioning knob inside that folder is `BV2_RENDER_TAG`
(appears as the `__<tag>` suffix in each EXR filename). Use it like this:

| Intent | What to do |
|---|---|
| **Replace** the existing EXRs (e.g. fixed a bug, re-rendering the same run) | Re-run with the **same** `BV2_RENDER_TAG`. Files overwrite in place. |
| **Keep the old render and add a new one side-by-side** (e.g. comparing samples, lighting variants) | Re-run with a **new** `BV2_RENDER_TAG` (e.g. `8k64s` → `8k64s-v2`, `8k128s`, `8k64s-lightfix`). Both sets coexist in the same folder; filenames differ by tag. |
| **Archive the old render** before a clean rebuild | Manually rename or move the `<exr_family>/` folder aside first, then run with whatever tag you want. |

`BV2_SIM_ROOT` should always be set for production — keeping the layout
flat-under-sim-root is what mirrors the Mediaflux tree and lets the
compositor find the EXRs. Omitting `BV2_SIM_ROOT` switches the pipeline
into an ad-hoc `<timestamp>_<case_tag>_<tag>/` layout meant for throwaway
test renders.

---

## Outputs (observed, trimmed-parade timeline 4.10)

- **Saved blend:**
  `_data-refactored/blenderv2/blends/4.10/parade_timeline__full_pipeline.blend`
- **Render folder:**
  `_data-refactored/blenderv2/output/4.10/parade_timeline/`
  - `parade_timeline__existing_condition_positive__8k64s.exr`
  - `parade_timeline__existing_condition_trending__8k64s.exr`
  - `parade_timeline__positive_state__8k64s.exr`
  - `parade_timeline__positive_priority_state__8k64s.exr`
  - `parade_timeline__trending_state__8k64s.exr`
  - `parade_timeline__bioenvelope_positive__8k64s.exr`
  - `parade_timeline__bioenvelope_trending__8k64s.exr`
  - `parade_timeline__full_pipeline.blend` (scene copy at render time)
  - `parade_timeline__manifest.txt`

7 EXRs for `single_state` / `timeline` (from `VIEW_LAYER_NAMES`).
3 EXRs for `baseline` (from `BASELINE_VIEW_LAYER_NAMES`).

Wall-clock on this machine: ~5 min for the 7-EXR set at 8K/64 samples.
