# CLAUDE.md

## Project

`urban-futures` is the simulation engine and data pipeline for the 2026 Arboreal Futures project. Source lives in `_code-refactored/`. Data lives in the sibling `../data/` directory.

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
- BlenderV2: `blenderV2`

### Common commands

Always run from this repo root with `--project-dir .`:

```bash
# Upload a file to project data/
.venv/Scripts/python.exe -m mediafluxsync upload-project --create-parents --project-dir . <local-source> <subpath>

# Upload to pipeline/
.venv/Scripts/python.exe -m mediafluxsync upload-project --create-parents --exclude-parent --project-dir . <local-source> <subpath>

# Download
.venv/Scripts/python.exe -m mediafluxsync download-project --project-dir . --out <local-dest> <subpath>

# Dry-run (add --dry-run to any upload/download)
```

See [MEDIAFLUX.md](MEDIAFLUX.md) for full details and confirmed working examples.
