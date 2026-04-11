# Timeline Scene Template

This document is the scene-contract reference for the refactored timeline pipeline.
It describes what a valid site blend must contain, how the two supported build modes are structured, and what the EXR contract is.

Use this together with:

- [TIMELINE_RUNBOOK.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_futureSim_refactored/blender/timeline/TIMELINE_RUNBOOK.md) for operator steps
- [b2026_unified_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_futureSim_refactored/blender/timeline/b2026_unified_scene_contract.py) for the public naming/source-of-truth helpers
- [b2026_timeline_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_futureSim_refactored/blender/timeline/b2026_timeline_scene_contract.py) for the implementation-layer contract helpers

## Supported Modes

- `timeline`
  - builds the multi-year strip layout
  - uses the site top-level collection names from `SITE_CONTRACTS`
- `single_state`
  - builds one full-site year such as `yr180`
  - uses generated top-level names from `get_single_state_top_level_name(site, role)`

## Standard View Layers

These are the standard render layers shared by both `timeline` and `single_state`:

- `pathway_state`
- `existing_condition`
- `priority_state`
- `existing_condition_trending`
- `bioenvelope_positive`
- `bioenvelope_trending`
- `trending_state`

These come from `STANDARD_VIEW_LAYERS` in [b2026_unified_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_futureSim_refactored/blender/timeline/b2026_unified_scene_contract.py).

`existing_condition_trending` is:

- trending buildings and roads only
- no instancers
- no envelope

## Site Reference Blends

| Site | Reference blend | Scene | Base world objects |
| --- | --- | --- | --- |
| `city` | `D:\2026 Arboreal Futures\data\2026 futures city lightweight cleaned.blend` | `city` | `city_buildings.001`, `city_highResRoad.001` |
| `trimmed-parade` | `D:\2026 Arboreal Futures\data\2026 futures parade lightweight cleaned.blend` | `parade` | `trimmed-parade_base`, `trimmed-parade_highResRoad` |
| `street` | `D:\2026 Arboreal Futures\data\2026 futures street lightweight cleaned.blend` | `street` | `uni_base`, `uni_highResRoad` |

## Timeline-Mode Top-Level Collections

These are the site-specific top-level collection names used by `timeline` mode.

| Site | Base | Base cubes | Manager | Cameras | Positive bio | Trending bio | Positive | Priority | Trending |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `city` | `City_Base` | `City_Base-cubes` | `City_Manager` | `City_Cameras` | `City_Bioenvelopes-positive` | `City_Bioenvelopes-trending` | `city_positive` | `city_priority` | `city_trending` |
| `trimmed-parade` | `Parade_Base` | `Parade_Base-cubes` | `Parade_Manager` | `Parade_Cameras` | `Parade_Bioenvelopes-positive` | `Parade_Bioenvelopes-trending` | `trimmed-parade_positive` | `trimmed-parade_priority` | `trimmed-parade_trending` |
| `street` | `Street_Base` | `Street_Base-cubes` | `Street_Manager` | `Street_Cameras` | `Street_Bioenvelopes-positive` | `Street_Bioenvelopes-trending` | `street_positive` | `street_priority` | `street_trending` |

## Single-State Top-Level Collections

Single-state mode uses generated top-level names:

- `{site}_manager`
- `{site}_setup`
- `{site}_cameras`
- `{site}_positive`
- `{site}_priority`
- `{site}_trending`

The current single-state city file follows that structure:

- [2026 futures city single-state yr180.blend](/e:/2026%20Arboreal%20Futures/data/2026%20futures%20city%20single-state%20yr180.blend)

## Required Materials And Geometry Nodes

Required materials:

- `MINIMAL_RESOURCES`
- `Envelope`
- `Envelope Parade` for parade
- `WORLD_AOV`

Optional preview/debug material:

- `PROPOSALS`
  - proposal-colour preview material for world points and instancers
  - not part of the production EXR contract

Required Geometry Nodes groups:

- `instance_template`
- site world groups such as `Background` and `Background - Large pts`
- `Timeline Clip Translate` for timeline-mode strip assets

## Generic Single-State Hierarchy Matrix

Use this as the generic target hierarchy for single-state renders.
The `Name` column explicitly shows the collection path using `>`.
`Object Render Number` is the object pass index used by the render setup.

| Name | Object Render Number | `existing_condition` | `pathway_state` | `priority_state` | `trending_state` | `bioenvelope_positive` | `bioenvelope_trending` | `existing_condition_trending` |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `{site}_manager` | `-` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_manager > helper/control objects only` | `0` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_setup` | `-` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_setup > {site}_highResRoad.001` | `1` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_setup > {site}_buildings.001` | `2` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_setup > approved render camera` | `0` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_setup > any required camera helpers / clip proxies` | `0` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_setup > any required setup-only helper objects` | `0` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_cameras` | `-` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_cameras > approved cameras` | `0` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_cameras > optional archived cameras / camera proxy objects` | `0` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_positive` | `-` | Disabled | Enabled | Disabled | Disabled | Enabled | Disabled | Disabled |
| `{site}_positive > {site}_highResRoad.001__yr{year}_positive_state` | `1` | Enabled | Enabled | Disabled | Disabled | Enabled | Disabled | Disabled |
| `{site}_positive > {site}_buildings.001__yr{year}_positive_state` | `2` | Enabled | Enabled | Disabled | Disabled | Enabled | Disabled | Disabled |
| `{site}_positive > tree_{site}_yr{year}_positive_positions` | `3` | Disabled | Enabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_positive > tree_{site}_yr{year}_positive_plyModels` | `3` | Disabled | Enabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_positive > log_{site}_yr{year}_positive_positions` | `3` | Disabled | Enabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_positive > log_{site}_yr{year}_positive_plyModels` | `3` | Disabled | Enabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_positive > {site}_positive_envelope__yr{year}` | `5` | Disabled | Disabled | Disabled | Disabled | Enabled | Disabled | Disabled |
| `{site}_priority` | `-` | Disabled | Disabled | Enabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_priority > {site}_highResRoad.001__yr{year}_positive_state` | `1` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_priority > {site}_buildings.001__yr{year}_positive_state` | `2` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_priority > tree_{site}_yr{year}_positive_priority_positions` | `3` | Disabled | Disabled | Enabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_priority > tree_{site}_yr{year}_positive_priority_plyModels` | `3` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_priority > log_{site}_yr{year}_positive_priority_positions` | `3` | Disabled | Disabled | Enabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_priority > log_{site}_yr{year}_positive_priority_plyModels` | `3` | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled | Disabled |
| `{site}_trending` | `-` | Disabled | Disabled | Disabled | Enabled | Disabled | Enabled | Enabled |
| `{site}_trending > {site}_highResRoad.001__yr{year}_trending_state` | `1` | Disabled | Disabled | Disabled | Enabled | Disabled | Enabled | Enabled |
| `{site}_trending > {site}_buildings.001__yr{year}_trending_state` | `2` | Disabled | Disabled | Disabled | Enabled | Disabled | Enabled | Enabled |
| `{site}_trending > tree_{site}_yr{year}_trending_positions` | `3` | Disabled | Disabled | Disabled | Enabled | Disabled | Disabled | Disabled |
| `{site}_trending > tree_{site}_yr{year}_trending_plyModels` | `3` | Disabled | Disabled | Disabled | Enabled | Disabled | Disabled | Disabled |
| `{site}_trending > log_{site}_yr{year}_trending_positions` | `3` | Disabled | Disabled | Disabled | Enabled | Disabled | Disabled | Disabled |
| `{site}_trending > log_{site}_yr{year}_trending_plyModels` | `3` | Disabled | Disabled | Disabled | Enabled | Disabled | Disabled | Disabled |
| `{site}_trending > {site}_trending_envelope__yr{year}` | `5` | Disabled | Disabled | Disabled | Disabled | Disabled | Enabled | Disabled |

## EXR Framebuffer Contract

The EXR export path writes every enabled output socket from the `Render Layers` node for each target view layer.

Standard render-layer outputs:

| EXR layer | Type | Notes |
| --- | --- | --- |
| `Image` | `RGBA` | beauty / combined color |
| `Alpha` | `VALUE` | alpha |
| `Depth` | `VALUE` | Z depth |
| `Mist` | `VALUE` | mist pass |
| `Normal` | `VECTOR` | world-space normal pass |
| `IndexOB` | `VALUE` | object index |
| `IndexMA` | `VALUE` | material index |
| `AO` | `RGBA` | ambient occlusion |
| `Noisy Image` | `RGBA` | noisy combined pass |

Custom AOV outputs:

| EXR layer | Type | Source |
| --- | --- | --- |
| `structure_id` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `resource` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `size` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `instance_id` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `isSenescent` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `bioEnvelopeType` | `VALUE` | bioenvelope/world point attribute |
| `sim_Turns` | `VALUE` | bioenvelope/world point attribute |
| `bioSimple` | `VALUE` | bioenvelope/world point attribute |
| `control` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `node_type` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `tree_interventions` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `tree_proposals` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `improvement` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `canopy_resistance` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `node_id` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `resource_colour` | `COLOR` | instancer point attribute via `MINIMAL_RESOURCES` |
| `isTerminal` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `world_sim_turns` | `VALUE` | world point attribute via `WORLD_AOV` |
| `world_sim_nodes` | `VALUE` | world point attribute via `WORLD_AOV` |
| `world_design_bioenvelope` | `VALUE` | world point attribute via `WORLD_AOV` |
| `world_design_bioenvelope_simple` | `VALUE` | world point attribute via `WORLD_AOV` |
| `world_sim_matched` | `VALUE` | world point attribute via `WORLD_AOV` |
| `instanceID` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `precolonial` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `resource_tree_mask` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `resource_none_mask` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `resource_dead_branch_mask` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `resource_peeling_bark_mask` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `resource_perch_branch_mask` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `resource_epiphyte_mask` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `resource_fallen_log_mask` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `resource_hollow_mask` | `VALUE` | instancer point attribute via `MINIMAL_RESOURCES` |
| `proposal-decay` | `VALUE` | instancer and world point attributes |
| `proposal-release-control` | `VALUE` | instancer and world point attributes |
| `proposal-recruit` | `VALUE` | instancer and world point attributes |
| `proposal-colonise` | `VALUE` | instancer and world point attributes |
| `proposal-deploy-structure` | `VALUE` | instancer and world point attributes |
