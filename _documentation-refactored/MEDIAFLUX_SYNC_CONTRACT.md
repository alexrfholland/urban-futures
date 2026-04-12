# Mediaflux Sync Contract

## Temporary Download Workaround

Until the shared `mediafluxsync` package supports `--exclude-parent` for
`download-project`, simulation downloads in this repo use a temporary
workaround.

Current temporary behavior:

- remote child paths such as `pipeline/<sim_root>/simulation_outputs/output`
  are downloaded into the parent local run root
- the download binary recreates the `output/` tail locally

This is temporary.

When `mediafluxsync` supports download-side `--exclude-parent`, the intended
steady-state behavior is:

- download remote `.../output`
- target local `<sim_root>/output`
- pass `--exclude-parent`

This note is the working contract for how simulation outputs, Blender v2 EXR
families, and compositor runs should relate to each other locally and on
Mediaflux.

This is a design contract first.

It is not an implementation note for wrappers or scripts.

## Mounted Discovery

When the Mediaflux project is mounted locally, discovery should use the mounted
pipeline tree directly rather than `check-project` CSV diffs.

The repo-local discovery helper is:

- `uv run python -m _futureSim_refactored.sim.run.mediaflux_browse`

Supported discovery modes:

- recent roots:
  - `--last 5`
- recent root map:
  - `--last 2 --map`
- most recent matching root:
  - `--pattern baseline --last 1`
- specific root browsing:
  - `4.9 --section blender_exrs --map`
  - `4.9 --section blender_exrs --pattern city_baseline --files`

This helper is for discovery only.

It does not replace `upload-project` / `download-project`.

## Goal

Use one clear lineage:

- `sim_root`
- `exr_family`
- `compositor_run`

Each downstream layer should remain a child of the layer above it.

That allows selective sync:

- sync a whole simulation run
- sync one EXR family from that simulation run
- sync one compositor run from that EXR family

## Parent / Child Model

### 1. Simulation Root

The parent unit is the simulation root:

- `<sim_root>`

Examples:

- `4.9`
- `4.9test`
- `v4.10-deadwood`

The simulation root is the authority for simulation outputs.

### 2. EXR Family

The child unit beneath a simulation root is an EXR family:

- `<exr_family>`

An EXR family is not one EXR file.

It is one grouped Blender v2 case, containing the set of EXR files produced for
that case.

The identity of an EXR family should align with the canonical bV2 `case`
identity, not with a separate Mediaflux-only naming system.

### 3. Compositor Run

The child unit beneath an EXR family is a compositor run:

- `<compositor_run>`

A compositor run is one family/version of PNG outputs made from one EXR family.

Multiple compositor runs may exist for the same EXR family as the compositor
work develops.

## Canonical Local Tree

The local canonical tree should be:

```text
_data-refactored/
  model-outputs/
    generated-states/
      <sim_root>/
        output/
          vtks/
          feature-locations/
          bioenvelopes/
          baselines/
        temp/
        comparison/

  blenderv2/
    blends/
      <sim_root>/
        <exr_family>__full_pipeline.blend
    output/
      <sim_root>/
        <exr_family>/
          *.exr
          manifest.txt
          provenance.json
          optional_saved_blend.blend

  compositor/
    outputs/
      <sim_root>/
        <exr_family>/
          <compositor_run>/
            *.png
            manifest.txt
            provenance.json
```

Rule:

- local canonical structure should mirror remote canonical structure
- only the base root changes between local and Mediaflux
- for bV2, canonical local roots should live under `_data-refactored/blenderv2`
  rather than machine-specific `E:` folders

## Simulation Outputs

Simulation remains the source of truth for:

- `output/vtks`
- `output/feature-locations`
- `output/bioenvelopes`
- `output/baselines` when needed

For Blender v2 purposes, we do not create a separate conceptual “import source”.

Instead:

- the authority remains the simulation root
- only selected simulation output families are synced or consumed downstream

Current simulation sync rule:

- default simulation sync should include only `output/`
- `temp/` and `comparison/` should be opt-in debug material
- EXR and compositor sync do not need an equivalent debug switch

## EXR Family Contract

An EXR family should be the grouped output for one bV2 case.

It should contain the EXR set for the relevant view layers, not a single EXR.

Current bV2 EXR filename pattern is:

- `<case>__<view_layer>__<tag>.exr`

Examples:

- `city_timeline__positive_state__8k64s.exr`
- `city_single-state_yr180__trending_state__8k64s.exr`

That means:

- the family identity should be based on `<case>`
- the individual EXR files then vary by `<view_layer>` and `<tag>`

Current bV2 view-layer set includes:

- `existing_condition_positive`
- `existing_condition_trending`
- `positive_state`
- `positive_priority_state`
- `trending_state`
- `bioenvelope_positive`
- `bioenvelope_trending`

Not every workflow must use every EXR, but the EXR family is the grouped case
identity above those files.

## EXR Family Naming

The EXR family name is the grouped bV2 case identifier.

It is not the filename of an individual EXR.

Base pattern:

- `<case>`

Optional extended pattern:

- `<case>__<note>`

Current canonical bV2 `case` forms are:

- `<site_label>_timeline`
- `<site_label>_single-state_yr<year>`
- `<site_label>_baseline`
- `<site_label>_baseline_yr<year>` (optional — only when a specific baseline year is meaningful)

The baseline `yr<year>` suffix is optional because the simulation baseline
source data is year-agnostic for most sites (one baseline state per site).
A year should only be included when a workflow genuinely needs to distinguish
multiple baselines for the same site.

Important current rule:

- `camera` is not currently part of the canonical bV2 case name
- `tag` belongs in the EXR filename, not in the family folder name

`<note>` is optional and should only be used when the family needs an explicit
human-readable distinction beyond the canonical bV2 case.

Formatting rule for `<note>`:

- lowercase
- hyphen-separated
- no spaces

Examples:

- `city_timeline`
- `city_timeline__hero-view-test`
- `city_single-state_yr180`
- `city_single-state_yr180__hero-camera`
- `city_baseline`
- `city_baseline__hero-view-test`
- `parade_timeline__cuda-rerender`

Implementation note:

- when `BV2_SIM_ROOT` is set, the default local EXR family root should resolve
  to `_data-refactored/blenderv2/output/<sim_root>/<exr_family>/`
- optional EXR-family note can be passed via `BV2_EXR_FAMILY_NOTE`

## Compositor Run Contract

A compositor run is a child of one EXR family.

It should not live beside the EXR family as a peer.

It should live under the EXR family it consumed.

Required identity fields:

- `compositor_family`
- `timestamp`

`compositor_family` should remain flexible.

It should be a short descriptive workflow label, not a tightly controlled token
set.

Formatting rule:

- lowercase
- hyphen-separated
- no spaces

Suggested pattern:

- `<compositor_family>__<timestamp>`

Optional extended pattern:

- `<compositor_family>__<timestamp>__<note>`

Examples:

- `mist__20260411_1530`
- `depth-outliner__20260411_1605`
- `final-template__20260411_1632`
- `proposal-outline__20260411_1701__mask-fix`

## Remote Tree

The intended Mediaflux tree should mirror the local lineage:

```text
<project-root>/
  pipeline/
    <sim_root>/
      simulation_outputs/
        ...
      blender_exrs/
        <exr_family>/
          ...
      compositor_pngs/
        <exr_family>/
          <compositor_run>/
            ...
    _library/
      blender_exrs/
        <asset_family>/
          exr/
            <asset_token>_<view>.exr
          previews/
            <asset_token>_<view>.png
      compositor_pngs/
        <asset_family>/
          <compositor_run>/
            *.png
    _site/
      ...
```

Current code-side selectors now align with this shape:

- bV2:
  - `BV2_SIM_ROOT`
  - optional `BV2_EXR_FAMILY_NOTE`
- compositor:
  - `COMPOSITOR_SIM_ROOT`
  - `COMPOSITOR_EXR_FAMILY`
  - `COMPOSITOR_FAMILY`
  - optional `COMPOSITOR_RUN_TIMESTAMP`
  - optional `COMPOSITOR_RUN_NOTE`

## Non-Simulation-State Roots

Not every artefact in this project is tied to a simulation run. The
`pipeline/` namespace carries both simulation runs (under plain `<sim_root>`
names like `4.9`, `4.10`) and non-simulation-state areas whose names start
with an underscore.

Naming rule:

- non-simulation-state roots live **under `pipeline/`** alongside
  `<sim_root>` entries — they are peers of sim roots, not peers of `pipeline/`
- their names always start with `_`
- the underscore marks them as outside the sim lineage contract

Examples:

- `pipeline/_library/` — reusable asset renders and asset-library outputs
  that are not tied to a specific simulation run (e.g. per-tree library
  renders keyed by the tree PLY library identity such as
  `precolonial.True_size.large_control.park-tree_id.11`)
- `pipeline/_site/` — site-level reference material that is not produced by
  a sim run

Rules for `_`-prefixed roots under `pipeline/`:

- they are **not** governed by the sim → exr_family → compositor_run lineage
- each `pipeline/_<category>/` subtree defines its own internal layout
- nothing under a `_`-prefixed root should be addressed via `BV2_SIM_ROOT`,
  `COMPOSITOR_SIM_ROOT`, `COMPOSITOR_EXR_FAMILY`, or any other selector whose
  semantics assume a simulation root
- new non-lineage artefacts should go into a `_`-prefixed root rather than be
  forced into `pipeline/<sim_root>/`

### `_library/` internal layout

The current `pipeline/_library/` layout (observed on Mediaflux 2026-04-11):

```text
pipeline/_library/
  blender_exrs/
    <asset_family>/
      exr/
        <asset_token>_<view>.exr
      previews/
        <asset_token>_<view>.png
  compositor_pngs/              # planned — not yet populated on remote
    <asset_family>/
      <compositor_run>/
        *.png
```

Naming notes for `_library/`:

- `<asset_family>` is a timestamped descriptive run label, e.g.
  `20260407_232744_ply-library-exr-4sides-large-senescing-snag_el20_4k64s`
- Each asset family is one batch of library renders with a consistent
  render setting (resolution/samples/workflow)
- `<asset_token>` for per-tree renders is the PLY library identity:
  `precolonial.<bool>_size.<category>_control.<control>-tree_id.<id>`
- `<view>` for the current 4-sides library renders is one of
  `north`, `south`, `east`, `west`
- Inside each asset family, `exr/` and `previews/` are parallel mirrors
  (one PNG preview per EXR). This is a `_library/`-specific shape — the
  sim-side `pipeline/<sim_root>/blender_exrs/<exr_family>/` does not have
  this `exr/` + `previews/` split.
- The compositor sibling (`pipeline/_library/compositor_pngs/`) mirrors the
  sim-side `pipeline/<sim_root>/compositor_pngs/` shape: one
  `<compositor_run>` folder per run, child of `<asset_family>`.

Local mirror for `_library/`:

```text
_data-refactored/blenderv2/output/_library/
  <asset_family>/
    exr/
      *.exr
    previews/
      *.png
_data-refactored/compositor/outputs/_library/
  <asset_family>/
    <compositor_run>/
      *.png
```

The local mirror drops the `blender_exrs/` / `compositor_pngs/` category
level the same way sim-side local paths drop them — local `blenderv2/output`
and `compositor/outputs` already imply the category.

## Provenance

Each EXR family should record:

- `sim_root`
- `site`
- `mode`
- `year` if applicable
- `camera` if applicable
- render tag/settings
- the EXR files present

Each compositor run should record:

- `sim_root`
- `exr_family`
- `compositor_family`
- `timestamp`
- source template/blend/script when relevant

## Current Open Questions

These still need to be tightened before implementation:

1. Controlled vocabulary for `camera` names.
2. Whether baseline families should be part of the default shared simulation
   contract or only included when a specific downstream workflow needs them.
