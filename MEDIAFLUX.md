# Mediaflux Setup

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
  - `MEDIAFLUX_BLENDERV2_PATH`

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
.\.venv\Scripts\python.exe -m mediafluxsync blenderv2-path v3-5 --project-dir .
```

## Download Example

```powershell
.\.venv\Scripts\python.exe -m mediafluxsync download-blenderv2 `
  v3-5/feature-locations/uni `
  --project-dir . `
  --out "D:\2026 Arboreal Futures\urban-futures\_tmp_unified_validation\mf-package-test\uni"
```

## Upload Example

Confirm the destination first:

```powershell
.\.venv\Scripts\python.exe -m mediafluxsync blenderv2-path upload-test --project-dir .
```

Then dry-run the upload command:

```powershell
.\.venv\Scripts\python.exe -m mediafluxsync upload-blenderv2 `
  "D:\2026 Arboreal Futures\urban-futures\_tmp_unified_validation\mf upload source\tiny-folder" `
  upload-test `
  --project-dir . `
  --create-parents `
  --dry-run
```

When the destination is confirmed, remove `--dry-run` to perform the upload.

## Confirmed Real Upload

A real Blender v2 test-output upload succeeded with `upload-blenderv2` after switching to `--exclude-parent`.

Local folder:

- `E:\2026 Arboreal Futures\blender\tests\20260404_214908_bV2_city_timeline_existing-instancers_bioenvelope_viewport_tests_1080p_flat`

Clean remote destination:

- `/projects/proj-7020_research_archive-1128.4.442/MF 2026 Arboreal Futures/blenderV2/output/tests/20260404_214908_bV2_city_timeline_existing-instancers_bioenvelope_viewport_tests_1080p_flat_direct2`

Representative uploaded assets landed directly under that collection:

- `city_timeline__bioenvelope_positive.png`
- `city_timeline__instancers_bioenv_debug.blend`
- `city_timeline__manifest.txt`

Working command pattern:

```powershell
.\.venv\Scripts\python.exe -m mediafluxsync upload-blenderv2 `
  "E:\2026 Arboreal Futures\blender\tests\20260404_214908_bV2_city_timeline_existing-instancers_bioenvelope_viewport_tests_1080p_flat" `
  "output/tests/20260404_214908_bV2_city_timeline_existing-instancers_bioenvelope_viewport_tests_1080p_flat_direct2" `
  --project-dir . `
  --create-parents `
  --exclude-parent
```

Notes:

- Without `--exclude-parent`, Mediaflux created one extra nested folder with the same source directory name.
- `check-blenderv2` is currently a little flaky for some fresh uploads and may report the collection as missing even when the upload log clearly shows files were written there.
- Treat that as a checker-side issue to tidy separately, not automatically as an upload failure.

## Working Rule

Do not add new work to the old in-project `mediafluxsync` folder.
Use the shared package repo instead.
