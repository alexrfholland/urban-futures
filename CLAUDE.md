# CLAUDE.md

## Project

`urban-futures` is the simulation engine and data pipeline for the 2026 Arboreal Futures project. Source lives in `_code-refactored/`. The canonical local pipeline data now lives repo-locally under `_data-refactored/`. Legacy source/reference material still exists under `data/`.

## Mediaflux

This project uses the shared `mediafluxsync` package (sibling repo at `../mediafluxsync/`) to upload/download data to University of Melbourne Mediaflux research archive.

### Client binaries

The official `unimelb-mf-clients` are extracted locally at:

```
.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/bin/windows/
```

Set `MEDIAFLUX_CLIENT_BIN_DIR` when invoking mediafluxsync commands:

```bash
export MEDIAFLUX_CLIENT_BIN_DIR="/d/2026 Arboreal Futures/urban-futures/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/bin/windows"
```

### Config files

- Credentials: `%USERPROFILE%\.Arcitecta\mflux.cfg`
- Shared defaults: `../mediafluxsync/.env.mediaflux`
- Project paths: `.env.mediaflux` (this repo)
- Secrets: `../mediafluxsync/.env` (gitignored)

### Project paths

- Allocation: `/projects/proj-7020_research_archive-1128.4.442`
- Project: `MF 2026 Arboreal Futures`

### Allowed subcommands

Only these `mediafluxsync` subcommands exist:

- `upload-project`
- `download-project`
- `check-project`
- `project-config`
- `project-path`

`ls-project` and `exists-project` were removed. Do not use them.

### Environment setup

Both env vars must be set before any mediafluxsync command:

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

### Upload commands

Always run from repo root with `--project-dir .`:

```bash
# Upload a folder to pipeline/ (--exclude-parent keeps the folder name as the remote leaf)
.venv/Scripts/python.exe -m mediafluxsync upload-project \
  --create-parents --exclude-parent --project-dir . \
  <local-source> <remote-subpath>
```

**Compositor PNG example** — uploading a compositor run folder:

```bash
# Local: _data-refactored/compositor/outputs/4.10/city_single-state_yr180/mist__20260411_1530/
# Remote: pipeline/4.10/compositor_pngs/city_single-state_yr180/mist__20260411_1530/

.venv/Scripts/python.exe -m mediafluxsync upload-project \
  --create-parents --exclude-parent --project-dir . \
  "_data-refactored/compositor/outputs/4.10/city_single-state_yr180/mist__20260411_1530" \
  "pipeline/4.10/compositor_pngs/city_single-state_yr180/mist__20260411_1530"
```

**EXR family example:**

```bash
.venv/Scripts/python.exe -m mediafluxsync upload-project \
  --create-parents --exclude-parent --project-dir . \
  "_data-refactored/blenderv2/output/4.10/city_timeline" \
  "pipeline/4.10/blender_exrs/city_timeline"
```

### Download and verify

```bash
# Download
.venv/Scripts/python.exe -m mediafluxsync download-project \
  --project-dir . --out <local-dest> <remote-subpath>

# Verify upload (compare local dir against remote)
.venv/Scripts/python.exe -m mediafluxsync check-project \
  --project-dir . --direction up \
  <local-dir> <remote-subpath>

# Dry-run (add --dry-run to any upload/download)
```

See [MEDIAFLUX.md](MEDIAFLUX.md) for full details and [MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md) for the authoritative path lineage contract.
