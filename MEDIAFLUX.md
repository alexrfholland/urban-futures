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

## Split

Shared `mediafluxsync` repo:

- tracked `.env.mediaflux`
  - generic Mediaflux defaults
- local gitignored `.env`
  - `MFLUX_PASSWORD`
- shared package code

This project:

- tracked [.env.mediaflux](/d:/2026%20Arboreal%20Futures/urban-futures/.env.mediaflux)
  - `MEDIAFLUX_ALLOCATION_PATH`
  - `MEDIAFLUX_PROJECT_PATH`

User profile:

- `%USERPROFILE%\.Arcitecta\mflux.cfg`

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
  <sim_root>/
    simulation_outputs/
    blender_exrs/
    compositor_pngs/
```

Examples:

- `pipeline/v4.9/simulation_outputs/`
- `pipeline/v4.9/blender_exrs/city_timeline__hero-test/`
- `pipeline/v4.9/compositor_pngs/city_timeline__hero-test/mist__20260411_1730/`

Use `upload-project` / `download-project` against those project-root-relative
paths.

For simulation run roots specifically, prefer the repo helper:

```powershell
.\.venv\Scripts\python.exe -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload 4.9test
```

Add `--include-debug` only when you intentionally want `temp/` and
`comparison/` uploaded as well.

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
