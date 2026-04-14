# Mediaflux

Execution rule: run Mediaflux commands from the repo root with standard `uv` — `uv run python -m ...`.

This repo uses the shared `mediafluxsync` package to upload/download data to the University of Melbourne Mediaflux research archive.

## Authoritative Docs

- Operational guide: `MEDIAFLUX.md` (this file)
- Sync/path contract: [_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md)
- Repo-local path expectations and selectors: [_documentation-refactored/MACHINE_SETUP_MEDIAFLUX.md](_documentation-refactored/MACHINE_SETUP_MEDIAFLUX.md)
- Compositor sync lineage: [_futureSim_refactored/blender/compositor/COMPOSITOR_SYNC_CONTRACT.md](_futureSim_refactored/blender/compositor/COMPOSITOR_SYNC_CONTRACT.md)

## Shared Repo

This project consumes the shared `mediafluxsync` repo as a sibling checkout:

- local sibling checkout: `../mediafluxsync`
- GitHub: `https://github.com/alexrfholland/mediafluxsync`
- shared package docs: [../mediafluxsync/README.md](</d:/2026 Arboreal Futures/mediafluxsync/README.md>)

Do not recreate a local in-project `mediafluxsync` module.

Use the shared repo README for package-level install/bootstrap behavior.
Use this file for repo-specific operational guidance and use
[_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md)
for the sync/path contract.

## Setup

From the repo root:

```bash
uv sync --extra visuals --extra blender
uv pip install -e ../mediafluxsync
```

Verify the setup:

```bash
uv run python -m mediafluxsync project-config --project-dir .
uv run python -m mediafluxsync project-path pipeline --project-dir .
```

If the shared package is already installed in this environment, you do not need to reinstall it.

## Credentials And Config

- credentials:
  - macOS/Linux: `~/.Arcitecta/mflux.cfg`
  - Windows: `%USERPROFILE%\.Arcitecta\mflux.cfg`
- shared defaults: `../mediafluxsync/.env.mediaflux`
- project paths: `.env.mediaflux` in this repo
- secrets: `../mediafluxsync/.env` (gitignored)

## Windows Setup

Both env vars must be set before any `mediafluxsync` command on Windows:

```powershell
$env:PATH = "D:\2026 Arboreal Futures\urban-futures\.tools\mediaflux-bin\unimelb-mf-clients-0.8.5\jre\bin;$env:PATH"
$env:MEDIAFLUX_CLIENT_BIN_DIR = "D:\2026 Arboreal Futures\urban-futures\.tools\mediaflux-bin\unimelb-mf-clients-0.8.5\bin\windows"
```

Without the JRE on `PATH`, every client command fails with `cannot find java`.

## Allowed `mediafluxsync` Subcommands

- `upload-project`
- `download-project`
- `check-project`
- `project-config`
- `project-path`

`ls-project` and `exists-project` were removed. Do not use them.

## Working Rules

- Always transfer data via `mediafluxsync`. Do not use `rsync`, `cp`, or raw `unimelb-mf-*` commands.
- Always run from the repo root with `--project-dir .`.
- Remote subpaths always start with `pipeline/`.
- `download-project --out` is the parent directory, not the final leaf directory.
- Append `--dry-run` to preview a transfer without executing it.

For the canonical local/remote layout, see [_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md](_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md).

## Discovery On Mounted Mediaflux

Use the mounted-volume browser for discovery instead of using `check-project` as a broad remote lister:

```bash
uv run python -m _futureSim_refactored.sim.run.mediaflux_browse --last 5
uv run python -m _futureSim_refactored.sim.run.mediaflux_browse <sim_root> --map
uv run python -m _futureSim_refactored.sim.run.mediaflux_browse <sim_root> --section compositor_pngs --map
```

This helper is for discovery only. It does not replace `upload-project` / `download-project`.

## Common Commands

```bash
# Verify project config
uv run python -m mediafluxsync project-config --project-dir .
uv run python -m mediafluxsync project-path pipeline --project-dir .

# Upload from a local dir into pipeline/<subpath>
uv run python -m mediafluxsync upload-project --create-parents --exclude-parent --project-dir . <local-source> <subpath>

# Download a remote subpath into a local parent dir
uv run python -m mediafluxsync download-project --project-dir . --out <local-parent-dir> <subpath>
```

## Repo Helpers

Simulation run helper:

```bash
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload <sim_root>
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync download <sim_root>
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload <sim_root> --include-debug
```

bV2 EXR-family helper:

```bash
uv run python -m _futureSim_refactored.blender.blenderv2.bV2_mediaflux_sync download <sim_root> <exr_family>
uv run python -m _futureSim_refactored.blender.blenderv2.bV2_mediaflux_sync upload <sim_root> <exr_family>
```

Add `--include-metadata` only when you intentionally want sidecars such as `__full_pipeline.blend` and `__manifest.txt`.
