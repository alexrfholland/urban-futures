# bV2 EXR And AOV Contract

This note is the canonical reference for the current `blenderv2` EXR output
contract.

Use it alongside:

- `D:\2026 Arboreal Futures\urban-futures\_futureSim_refactored\blenderv2\bV2_scene_contract.py`
- `D:\2026 Arboreal Futures\urban-futures\_futureSim_refactored\blenderv2\bV2_setup_render_outputs.py`

The old `v1.5` reference contract is:

- `D:\2026 Arboreal Futures\urban-futures\final\_futureSim_refactored\blender\timeline\TIMELINE_SCENE_TEMPLATE.md`

## Current bV2 output pattern

`bV2` writes one multilayer EXR per view layer.

Filename pattern:

- `<case>__<view_layer>__<tag>.exr`

Examples:

- `city_timeline__positive_state__8k64s.exr`
- `city_single-state_yr180__trending_state__8k64s.exr`

Current target view layers:

- `existing_condition_positive`
- `existing_condition_trending`
- `positive_state`
- `positive_priority_state`
- `trending_state`
- `bioenvelope_positive`
- `bioenvelope_trending`

## Standard render-layer passes

These are the standard passes currently enabled in `bV2`:

| EXR layer | Type | Notes |
| --- | --- | --- |
| `Image` | `RGBA` | combined / beauty |
| `Alpha` | `VALUE` | alpha |
| `Depth` | `VALUE` | Z depth |
| `Mist` | `VALUE` | mist pass |
| `Normal` | `VECTOR` | world-space normal |
| `IndexOB` | `VALUE` | object index |
| `IndexMA` | `VALUE` | material index |
| `AO` | `RGBA` | ambient occlusion |

`bV2` does not currently register `Noisy Image` as part of the explicit render
contract.

## Render backend caveat

- `trimmed-parade` timeline showed intermittent zero-alpha seam columns in the
  production `existing_condition_*` EXRs when rendered with Cycles `OPTIX`
  on this machine
- the same scene rerendered cleanly with Cycles `CUDA`
- treat this as a render-backend artifact, not a stale-cache or scene-geometry
  problem
- practical workaround:
  - prefer `CUDA` for `trimmed-parade` production rerenders if the seam
    artifact reappears under `OPTIX`

## Current custom bV2 AOVs

These AOV names are currently registered on every `bV2` view layer.

| AOV name in EXR | Type | Primary source in bV2 |
| --- | --- | --- |
| `proposal-decay` | `VALUE` | instancers, world, bioenvelopes |
| `proposal-release-control` | `VALUE` | instancers, world, bioenvelopes |
| `proposal-recruit` | `VALUE` | instancers, world, bioenvelopes |
| `proposal-colonise` | `VALUE` | instancers, world, bioenvelopes |
| `proposal-deploy-structure` | `VALUE` | instancers, world, bioenvelopes |
| `resource_none_mask` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `resource_dead_branch_mask` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `resource_peeling_bark_mask` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `resource_perch_branch_mask` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `resource_epiphyte_mask` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `resource_fallen_log_mask` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `resource_hollow_mask` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `size` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `control` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `precolonial` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `improvement` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `canopy_resistance` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `bioEnvelopeType` | `VALUE` | instancers and bioenvelopes |
| `sim_Turns` | `VALUE` | instancers and bioenvelopes |
| `world_sim_turns` | `VALUE` | world via `v2WorldAOV` |
| `world_sim_nodes` | `VALUE` | world via `v2WorldAOV` |
| `world_design_bioenvelope` | `VALUE` | world via `v2WorldAOV` |
| `world_design_bioenvelope_simple` | `VALUE` | world via `v2WorldAOV` |
| `node_id` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `instanceID` | `VALUE` | instancers via `MINIMAL_RESOURCES` |
| `source-year` | `VALUE` | instancers, world, bioenvelopes |
| `world_sim_matched` | `VALUE` | world via `v2WorldAOV` |

## Material paths

Current non-debug material ownership:

- world objects use `v2WorldAOV`
- instancer point clouds and imported models use `MINIMAL_RESOURCES`
- bioenvelopes use `Envelope`

Debug-only material path:

- `debug-source-years`

## Source mapping notes

### Instancers

`bV2_build_instancers.py` currently writes point attributes including:

- `structure_id`
- `precolonial`
- `size`
- `instanceID`
- `improvement`
- `canopy_resistance`
- `node_id`
- `bioEnvelopeType`
- `sim_Turns`
- `source-year`
- `control`
- `resource_*`
- `proposal-*`

Not every stored point attribute is currently registered as a production AOV.

### Bioenvelopes

`bV2_build_bioenvelopes.py` currently patches the live `Envelope` material so
it writes these AOVs:

- `source-year`
- `proposal-decay`
- `proposal-release-control`
- `proposal-recruit`
- `proposal-colonise`
- `proposal-deploy-structure`

### World

`bV2_build_world_attributes.py` currently transfers these key attrs onto
buildings, hires roads, and lores roads:

**Current world-source architecture:** Each site now uses two point-cloud source
objects:

- `<site>_buildings_source`
- `<site>_roads_source`

Both use the `v2WorldAOV` material and `v2WorldPoints` geometry nodes so the
world AOV contract transfers consistently through the rebuilt world objects.

Cube mode is currently disabled because it carries a large render cost. If it is
ever revived, it must be reintroduced as an explicit template refresh path
rather than a runtime toggle.

Source PLYs live at `_data-refactored/model_inputs/world/`. Use
`bV2_setup_template.py --reset <sites>` to reimport from PLY into the
template world-source payload.

Transferred attributes:

- `sim_Turns`
- `sim_Nodes`
- `scenario_bioEnvelope`
- `scenario_bioEnvelopeSimple`
- `sim_Matched`
- `source-year`
- all 5 `blender_proposal-*` framebuffers

The `v2WorldAOV` material exposes those through the world-facing EXR names:

- `world_sim_turns`
- `world_sim_nodes`
- `world_design_bioenvelope`
- `world_design_bioenvelope_simple`
- `world_sim_matched`
- the 5 proposal AOV names listed above

## v1.5 comparison

### AOV names kept unchanged in bV2

- `proposal-decay`
- `proposal-release-control`
- `proposal-recruit`
- `proposal-colonise`
- `proposal-deploy-structure`
- `size`
- `control`
- `precolonial`
- `improvement`
- `canopy_resistance`
- `bioEnvelopeType`
- `sim_Turns`
- `world_sim_turns`
- `world_sim_nodes`
- `world_design_bioenvelope`
- `world_design_bioenvelope_simple`
- `node_id`
- `instanceID`
- `world_sim_matched`
- `resource_none_mask`
- `resource_dead_branch_mask`
- `resource_peeling_bark_mask`
- `resource_perch_branch_mask`
- `resource_epiphyte_mask`
- `resource_fallen_log_mask`
- `resource_hollow_mask`

### New in bV2

- `source-year`

This is now the canonical provenance AOV and should be preserved everywhere.

### Present in v1.5 but not in the current explicit bV2 AOV contract

- `structure_id`
- `resource`
- `instance_id`
- `isSenescent`
- `bioSimple`
- `node_type`
- `tree_interventions`
- `tree_proposals`
- `resource_colour`
- `isTerminal`
- `resource_tree_mask`

These may still exist as geometry attributes or legacy material outputs in old
files, but they are not currently part of the explicit `bV2_scene_contract.py`
AOV registration list.

## View-layer rename comparison

The main `v1.5` to `bV2` view-layer renames are:

| v1.5 | bV2 |
| --- | --- |
| `existing_condition` | `existing_condition_positive` |
| `pathway_state` | `positive_state` |
| `priority_state` | `positive_priority_state` |

These renames affect EXR filenames because `bV2` uses the view-layer name in
the output filename.

## Rule

If an AOV should be considered production-canonical for `bV2`, it should be
represented in all three places:

- `bV2_scene_contract.py`
- the live material / GN path that actually writes it
- this document

## Library asset EXR variant

For isolated tree / log library renders, use:

- `_code-refactored/refactor_code/blenderv2/bV2_render_ply_library_assets.py`

This is a reduced contract for one-asset-per-EXR library export, not the full
state-scene contract.

Current intended outputs for that library path:

- one Workbench PNG preview per asset
- one multilayer EXR per asset
- shared orthographic isometric camera fit derived from the asset with the
  largest required spatial isometric footprint in the selected library set,
  not from file size

The library-face EXR path intentionally excludes instancer-level AOVs such as:

- `size`
- `control`
- `node_id`
- `instanceID`
- `improvement`
- `canopy_resistance`
- proposal AOVs

The reduced face/material AOV set for that path is:

- `resource`
- `resource_tree_mask`
- `resource_colour`
- `resource_none_mask`
- `resource_dead_branch_mask`
- `resource_peeling_bark_mask`
- `resource_perch_branch_mask`
- `resource_epiphyte_mask`
- `resource_fallen_log_mask`
- `resource_hollow_mask`

Optional geometry flags can also be enabled for that path when needed:

- `isSenescent`
- `isTerminal`

## Compositor consumption

See
[EXR_INPUT_GUIDE.md](../compositor/EXR_INPUT_GUIDE.md)
for the compositor-side lookup of which view layers feed which blends.

### intervention_int compositor family

The `intervention_bioenvelope_ply-int` AOV encodes intervention categories as
integers 0–8. The `compositor_intervention_int.blend` template reads this AOV
from a single bioenvelope EXR (`bioenvelope_positive` or `bioenvelope_trending`)
and produces:

- 1 combined RGBA PNG (`interventions_bioenvelope.png`) — all categories
  coloured, int 0 transparent
- 8 per-category RGBA PNGs (`interventions_bioenvelope_<category>.png`) — each
  showing only that category's colour with alpha mask

Integer-to-colour mapping (sRGB hex):

| Int | Category | Hex |
| --- | --- | --- |
| 0 | none | transparent |
| 1 | deploy-any | #DCC090 |
| 2 | decay | #F0DC90 |
| 3 | colonise-ground | #8ED8C8 |
| 4 | colonise-partial | #D0A040 |
| 5 | colonise-full | #B8E86C |
| 6 | recruit-partial | #F0DC90 |
| 7 | recruit-full | #8ED8C8 |
| 8 | depaved | #DC78A0 |

Runner: `render_edge_lab_current_intervention_int.py`

Rebuild: `rebuild_intervention_int_template_20260412.py`
