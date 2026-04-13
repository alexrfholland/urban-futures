# Mediaflux Setup

Note:

- the historical examples in this file include earlier `blenderV2/output/...`
  upload targets
- the current canonical pipeline sync layout is documented in:
  - [_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md)
  - [_documentation-refactored/MACHINE_SETUP_MEDIAFLUX.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/MACHINE_SETUP_MEDIAFLUX.md)
  - [_futureSim_refactored/blender/compositor/COMPOSITOR_SYNC_CONTRACT.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/COMPOSITOR_SYNC_CONTRACT.md)

`urban-futures` now consumes the shared [mediafluxsync](/d:/2026%20Arboreal%20Futures/mediafluxsync) repo.

The old in-project module under `_futureSim_refactored/modules/mediafluxsync` is transitional and is intended to be deleted.

The canonical Codex skill source now lives in:

- [codex-skill](/d:/2026%20Arboreal%20Futures/mediafluxsync/skills/codex-skill)

## Shared Repo

This project consumes the shared `mediafluxsync` repo.
It is a sibling repo, not a folder inside `urban-futures`.

Local sibling checkout:

- [mediafluxsync](/d:/2026%20Arboreal%20Futures/mediafluxsync)

GitHub:

- `https://github.com/alexrfholland/mediafluxsync`

Install it into this repo's `.venv` with:

```powershell
.\.tools\uv\uv.exe pip install --python .\.venv\Scripts\python.exe -e ..\mediafluxsync
```

Do not recreate a local in-project `mediafluxsync` module.

## Split

Shared `mediafluxsync` repo:

- tracked `.env.mediaflux`
  - generic Mediaflux defaults
- optional local gitignored `.env`
  - `MFLUX_PASSWORD`
- shared package code

This project:

- tracked [.env.mediaflux](/d:/2026%20Arboreal%20Futures/urban-futures/.env.mediaflux)
  - `MEDIAFLUX_ALLOCATION_PATH`
  - `MEDIAFLUX_PROJECT_PATH`

User profile:

- `%USERPROFILE%\.Arcitecta\mflux.cfg`

## Initialize This Machine

On a fresh machine, do this from this repo root:

```powershell
uv sync --extra visuals --extra blender
.\.tools\uv\uv.exe pip install --python .\.venv\Scripts\python.exe -e ..\mediafluxsync
.\.venv\Scripts\python.exe -m mediafluxsync bootstrap --project-root .
```

Then verify the setup before attempting real work:

```powershell
.\.venv\Scripts\python.exe -m mediafluxsync project-config --project-dir .
.\.venv\Scripts\python.exe -m mediafluxsync which --project-dir .
.\.venv\Scripts\python.exe -m mediafluxsync project-path pipeline --project-dir .
```

Then do one small real transfer in each direction:

```powershell
.\.venv\Scripts\python.exe -m mediafluxsync download-project `
  "pipeline/v4.9/blender_exrs/city_timeline__hero-test" `
  --project-dir . `
  --out ".\_tmp_unified_validation\mf-download-test\city_timeline__hero-test"
```

```powershell
.\.venv\Scripts\python.exe -m mediafluxsync upload-project `
  ".\_tmp_unified_validation\mf-upload-test\tiny-folder" `
  "pipeline/scratch-upload" `
  --project-dir . `
  --create-parents `
  --exclude-parent
```

If this machine is already configured, you do not need to bootstrap again.
In that case, just run the verification commands above.

## Auth Model

Runtime auth can come from more than one place:

- optional shared `mediafluxsync/.env`
  - for example `MFLUX_PASSWORD`
- this repo's local `.env`
- the current process environment
- `%USERPROFILE%\.Arcitecta\mflux.cfg`

So the shared repo `.env` is optional, not mandatory.
If it is missing, commands can still work if the effective environment already has
`MFLUX_PASSWORD` or if the official clients can authenticate from `mflux.cfg`.

The important practical rule is:

- on a fresh machine, run `bootstrap`
- on an existing machine, verify with `project-config`, `which`, and a small real transfer

## CLI Note

The shared package currently uses:

- `--project-root` for `bootstrap` and `bootstrap-paths`
- `--project-dir` for most other commands

That is expected for now, even though it is slightly inconsistent.

## Install In This Project

From this repo root:

```powershell
uv sync --extra visuals --extra blender
.\.tools\uv\uv.exe pip install --python .\.venv\Scripts\python.exe -e ..\mediafluxsync
```

Then verify:

```powershell
.\.venv\Scripts\python.exe -m mediafluxsync project-config --project-dir .
.\.venv\Scripts\python.exe -m mediafluxsync project-path pipeline --project-dir .
```

## Current Contract

The current canonical remote layout for this repo is:

```text
pipeline/
  <sim_root>/                        # e.g. 4.9, 4.10, 4.9test
    simulation_outputs/
    blender_exrs/
    compositor_pngs/
  _library/                          # non-sim, e.g. per-tree library renders
    blender_exrs/
      <asset_family>/
        exr/
        previews/
    compositor_pngs/                 # planned
      <asset_family>/
        <compositor_run>/
  _site/                             # non-sim site reference material
    ...
```

Examples:

- `pipeline/v4.9/simulation_outputs/`
- `pipeline/v4.9/blender_exrs/city_timeline__hero-test/`
- `pipeline/v4.9/compositor_pngs/city_timeline__hero-test/mist__20260411_1730/`
- `pipeline/_library/blender_exrs/20260407_232744_ply-library-exr-4sides-large-senescing-snag_el20_4k64s/exr/`

Anything that is **not** a simulation run — per-tree library renders,
site-level reference material, etc. — lives under a `_`-prefixed entry
inside `pipeline/` (alongside `<sim_root>` entries, not as a peer of
`pipeline/`). These `_`-prefixed roots are outside the
`sim_root → exr_family → compositor_run` lineage. See
`_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md` §"Non-Simulation-State
Roots" for the rules.

Use `upload-project` / `download-project` against those project-root-relative
paths.

## Fast Discovery On Mounted Mediaflux

If the Mediaflux project is mounted locally, do not use `check-project` as a
remote lister for broad discovery. It is much slower because it diffs every
asset under the queried path.

Use the mounted-volume browser instead:

```bash
uv run python -m _futureSim_refactored.sim.run.mediaflux_browse --last 5
uv run python -m _futureSim_refactored.sim.run.mediaflux_browse --last 2 --map
uv run python -m _futureSim_refactored.sim.run.mediaflux_browse --pattern baseline --last 1
uv run python -m _futureSim_refactored.sim.run.mediaflux_browse --pattern city --last 1
uv run python -m _futureSim_refactored.sim.run.mediaflux_browse 4.9 --section blender_exrs --map
uv run python -m _futureSim_refactored.sim.run.mediaflux_browse 4.9 --section blender_exrs --pattern city_baseline
uv run python -m _futureSim_refactored.sim.run.mediaflux_browse 4.9 --section compositor_pngs --pattern city_baseline --files
```

Default mounted root:

- `/Volumes/proj-7020_research_archive-1128.4.442/MF 2026 Arboreal Futures`

Override it with `MEDIAFLUX_MOUNT_ROOT` or `--mount-root` if needed.

For simulation run roots specifically, prefer the repo helper:

```powershell
.\.venv\Scripts\python.exe -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload 4.9test
```

Add `--include-debug` only when you intentionally want `temp/` and
`comparison/` uploaded as well.

For bV2 EXR families specifically, prefer the repo helper so the default
transfer is EXR-only:

```powershell
uv run python -m _futureSim_refactored.blender.blenderv2.bV2_mediaflux_sync download 4.10 parade_single-state_yr180
uv run python -m _futureSim_refactored.blender.blenderv2.bV2_mediaflux_sync upload 4.10 parade_single-state_yr180
```

Add `--include-metadata` only when you intentionally want sidecars such as
`__full_pipeline.blend` and `__manifest.txt`.

## Current Upload Example

```powershell
.\.venv\Scripts\python.exe -m mediafluxsync upload-project `
  ".\_data-refactored\blenderv2\output\v4.9\city_timeline__hero-test" `
  "pipeline/v4.9/blender_exrs/city_timeline__hero-test" `
  --project-dir . `
  --create-parents `
  --exclude-parent
```

## Current Download Example

```powershell
.\.venv\Scripts\python.exe -m mediafluxsync download-project `
  "pipeline/v4.9/compositor_pngs/city_timeline__hero-test/mist__20260411_1730" `
  --project-dir . `
  --out ".\_data-refactored\compositor\outputs\v4.9\city_timeline__hero-test\mist__20260411_1730"
```

## Working Rule

Do not add new work to the old in-project `mediafluxsync` folder.
Use the shared package repo instead.
