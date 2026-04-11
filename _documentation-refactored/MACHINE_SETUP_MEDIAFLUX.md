# Machine Setup For Mediaflux And Local Paths

This note is repo-specific.

Do not use it as a replacement for the shared `mediafluxsync` setup workflow.

For generic Mediaflux bootstrap and verification, use:

- [MEDIAFLUX.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/MEDIAFLUX.md)
- the shared `mediafluxsync` skill/workflow

This file only records what this repo expects locally once the generic
`mediafluxsync` setup is already working.

## Repo-Local Canonical Roots

This repo now treats these as the canonical local roots:

- simulation runs:
  - `_data-refactored/model-outputs/generated-states/<sim_root>/`
- Blender v2 EXR inputs cache:
  - `_data-refactored/blenderv2/inputs/`
- Blender v2 EXR outputs:
  - `_data-refactored/blenderv2/output/<sim_root>/<exr_family>/`
- Blender v2 saved full-pipeline blends:
  - `_data-refactored/blenderv2/blends/<sim_root>/`
- compositor outputs:
  - `_data-refactored/compositor/outputs/<sim_root>/<exr_family>/<compositor_run>/`
- compositor temp/working blends:
  - `_data-refactored/compositor/temp_blends/`

Rule:

- treat these repo-local roots as canonical
- do not treat machine-specific `E:` roots as the contract root anymore

## Required Project Config

This repo expects the repo-local [.env.mediaflux](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/.env.mediaflux)
to resolve to:

- project root:
  - `MF 2026 Arboreal Futures`

The canonical sync contract now uses the project root only.

Verify from repo root with:

```bash
./.venv/bin/python -m mediafluxsync project-config --project-dir .
./.venv/bin/python -m mediafluxsync project-path pipeline --project-dir .
```

Expected shape:

- project:
  - `.../MF 2026 Arboreal Futures/pipeline`

## Blender v2 Selectors

The intended bV2 selectors are:

- `BV2_SIM_ROOT`
- optional `BV2_EXR_FAMILY_NOTE`
- optional `BV2_RENDER_TAG`

When `BV2_SIM_ROOT` is set, the default local EXR output root becomes:

- `_data-refactored/blenderv2/output/<sim_root>/<exr_family>/`

and the default Mediaflux upload target becomes:

- `pipeline/<sim_root>/blender_exrs/<exr_family>/`

Example:

```bash
export BV2_SIM_ROOT=4.9
export BV2_EXR_FAMILY_NOTE=hero-test
```

Default local EXR root:

- `_data-refactored/blenderv2/output/4.9/city_timeline__hero-test/`

Default remote EXR target:

- `pipeline/4.9/blender_exrs/city_timeline__hero-test/`

## Simulation Sync

Use the simulation-only helper when you want the run-root contract without
having to remember child-folder rules:

```bash
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload 4.9test
```

Default behavior:

- uploads only:
  - `output/`

Opt-in debug upload:

```bash
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload 4.9test --include-debug
```

That additionally includes:

- `temp/`
- `comparison/`

The same switch works for download and check:

```bash
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync download 4.9test
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync check 4.9test --output ./_tmp_mediaflux_check/4.9test.csv
```

## Compositor Selectors

The intended compositor selectors are:

- `COMPOSITOR_SIM_ROOT`
- `COMPOSITOR_EXR_FAMILY`
- `COMPOSITOR_FAMILY`
- optional `COMPOSITOR_RUN_TIMESTAMP`
- optional `COMPOSITOR_RUN_NOTE`

When those selectors are set, the main compositor runners default to:

- EXR input root:
  - `_data-refactored/blenderv2/output/<sim_root>/<exr_family>/`
- compositor output root:
  - `_data-refactored/compositor/outputs/<sim_root>/<exr_family>/<compositor_run>/`

Example:

```bash
export COMPOSITOR_SIM_ROOT=4.9
export COMPOSITOR_EXR_FAMILY=city_timeline__hero-test
export COMPOSITOR_FAMILY=mist
```

Default local compositor output root:

- `_data-refactored/compositor/outputs/4.9/city_timeline__hero-test/mist__<timestamp>/`

Default remote compositor target:

- `pipeline/4.9/compositor_pngs/city_timeline__hero-test/mist__<timestamp>/`

## Cross-Machine Rule

On another machine, the repo can live at a different absolute filesystem path.

The contract should still be the same relative to the repo root:

- `_data-refactored/model-outputs/generated-states/...`
- `_data-refactored/blenderv2/...`
- `_data-refactored/compositor/...`

The absolute parent path does not matter.

The repo-local structure does.

## Other Machine Quickstart

On the other machine:

1. Open this repo and run:

```bash
uv sync
```

2. Verify the project-scoped Mediaflux root:

```bash
uv run python -m mediafluxsync project-config --project-dir .
uv run python -m mediafluxsync project-path pipeline --project-dir .
```

3. Pull or push simulation outputs with the repo helper:

```bash
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync download 4.9test
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync upload 4.9test
```

Add debug material only when intended:

```bash
uv run python -m _futureSim_refactored.sim.run.sim_mediaflux_sync download 4.9test --include-debug
```

4. For bV2 work, set the simulation selector:

```bash
export BV2_SIM_ROOT=4.9test
export BV2_EXR_FAMILY_NOTE=hero-test
```

5. For compositor work, set the lineage selectors:

```bash
export COMPOSITOR_SIM_ROOT=4.9test
export COMPOSITOR_EXR_FAMILY=city_timeline__hero-test
export COMPOSITOR_FAMILY=mist
```

Rule:

- do not rebuild machine-specific `E:` or `Z:` path assumptions
- keep everything relative to this repo and the `pipeline/<sim_root>/...` Mediaflux contract

## Current Limitation

Some older scripts still contain hardcoded absolute repo paths for this machine.

That is not part of the intended contract.

If those scripts are needed on another machine, they should be cleaned up to
derive paths from the repo root instead of assuming:

- `/Users/alexholland/...`

The main sync/path contract work is now in place, but not every historical
script has been normalized yet.
