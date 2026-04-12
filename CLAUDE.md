# CLAUDE.md

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
