# Lightweight Scene Blend

This note describes the current lightweight timeline scene contract for parade, city, and street.

## Parade Reference Blend

- `D:\2026 Arboreal Futures\data\2026 futures parade lightweight cleaned.blend`

## Scene

- scene name: `parade`
- active camera: `paraview_camera_parade`
- current approved framing: `VIEWS.md` parade `v3`

## Top-Level Collections

- `Parade_Manager`
- `Parade_Base`
- `Parade_Base-cubes`
- `Parade_Cameras`
- `Parade_Bioenvelopes-positive`
- `Parade_Bioenvelopes-trending`
- `trimmed-parade_positive`
- `trimmed-parade_priority`
- `trimmed-parade_trending`

## View Layers

- `pathway_state`
- `existing_condition`
- `priority_state`
- `bioenvelope_positive`
- `bioenvelope_trending`
- `trending_state`

## Materials

- `MINIMAL_RESOURCES`
  - required on all active tree/log instancer Geometry Nodes groups
- `Envelope`
  - required on all `trimmed-parade_*_envelope__yr*` objects
- `WORLD_AOV`
  - required on the active world/base Geometry Nodes groups

## Active Geometry Nodes Groups

Tree instancers:
- `tree_trimmed-parade_timeline_positive`
- `tree_trimmed-parade_timeline_positive_priority`
- `tree_trimmed-parade_timeline_trending`

World:
- `Background`
- `Background - Large pts`

Bioenvelopes:
- `Timeline Clip Translate`
- `Envelope Parade`

## Bioenvelope Objects

Positive:
- `trimmed-parade_positive_envelope__yr0`
- `trimmed-parade_positive_envelope__yr10`
- `trimmed-parade_positive_envelope__yr30`
- `trimmed-parade_positive_envelope__yr60`
- `trimmed-parade_positive_envelope__yr180`

Trending:
- `trimmed-parade_trending_envelope__yr0`
- `trimmed-parade_trending_envelope__yr10`
- `trimmed-parade_trending_envelope__yr30`
- `trimmed-parade_trending_envelope__yr60`
- `trimmed-parade_trending_envelope__yr180`

## AOV Contract

Tree/resource AOV family expected on the production EXRs:
- `structure_id`
- `resource`
- `size`
- `instance_id`
- `isSenescent`
- `resource_colour`
- `isTerminal`
- `control`
- `node_type`
- `tree_interventions`
- `tree_proposals`
- `improvement`
- `canopy_resistance`
- `node_id`
- `instanceID`
- `precolonial`
- `bioEnvelopeType`
- `bioSimple`
- `sim_Turns`
- `resource_tree_mask`
- `resource_none_mask`
- `resource_dead_branch_mask`
- `resource_peeling_bark_mask`
- `resource_perch_branch_mask`
- `resource_epiphyte_mask`
- `resource_fallen_log_mask`
- `resource_hollow_mask`

World AOV family expected on the production EXRs:
- `world_sim_turns`
- `world_sim_nodes`
- `world_design_bioenvelope`
- `world_design_bioenvelope_simple`
- `world_sim_matched`

Standard render passes expected where relevant:
- `Combined`
- `AO`
- `Depth`
- `Normal`
- `IndexOB`
- `Mist`

## Current Parade 8K Outputs

Local:
- `D:\2026 Arboreal Futures\data\renders\timeslices\parade\8k\parade_existing_condition_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\parade\8k\parade_pathway_state_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\parade\8k\parade_priority_state_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\parade\8k\parade_trending_state_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\parade\8k\parade_bioenvelope_positive_8k.exr`

Network:
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\parade\8k\parade_existing_condition_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\parade\8k\parade_pathway_state_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\parade\8k\parade_priority_state_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\parade\8k\parade_trending_state_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\parade\8k\parade_bioenvelope_positive_8k.exr`

## City Reference Blend

- `D:\2026 Arboreal Futures\data\2026 futures city lightweight cleaned.blend`

Scene:
- `city`

Top-level collections:
- `City_Manager`
- `City_Base`
- `City_Base-cubes`
- `City_Cameras`
- `City_Bioenvelopes-positive`
- `City_Bioenvelopes-trending`
- `city_positive`
- `city_priority`
- `city_trending`

View layers:
- `pathway_state`
- `existing_condition`
- `city_priority`
- `city_bioenvelope`
- `trending_state`
- `priority_state`
- `bioenvelope_positive`
- `bioenvelope_trending`

Active camera:
- `paraview_camera_city`

Required materials:
- `MINIMAL_RESOURCES`
- `Envelope`
- `WORLD_AOV`

8K outputs:
- `D:\2026 Arboreal Futures\data\renders\timeslices\city\8k\city_existing_condition_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\city\8k\city_pathway_state_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\city\8k\city_priority_state_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\city\8k\city_trending_state_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\city\8k\city_bioenvelope_positive_8k.exr`

Network:
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\city\8k\city_existing_condition_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\city\8k\city_pathway_state_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\city\8k\city_priority_state_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\city\8k\city_trending_state_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\city\8k\city_bioenvelope_positive_8k.exr`

## Street Reference Blend

- `D:\2026 Arboreal Futures\data\2026 futures street lightweight cleaned.blend`

Source note:
- street is currently built from the `uni` source scene/data, with the cleaned file exposing it as `street`

Scene:
- `street`

Top-level collections:
- `Street_Base`
- `Street_Base-cubes`
- `Street_Manager`
- `Street_Cameras`
- `Street_Bioenvelopes-positive`
- `Street_Bioenvelopes-trending`
- `street_positive`
- `street_priority`
- `street_trending`

View layers:
- `pathway_state`
- `existing_condition`
- `priority_state`
- `bioenvelope_positive`
- `bioenvelope_trending`
- `trending_state`

Active camera:
- `paraview_camera_street`

Required materials:
- `MINIMAL_RESOURCES`
- `Envelope`
- `WORLD_AOV`

Current source-object prefixes retained in the cleaned street file:
- world: `uni_*`
- bioenvelopes: `uni_*`
- tree instancers: `tree_uni_*`

8K outputs:
- `D:\2026 Arboreal Futures\data\renders\timeslices\street\8k\street_existing_condition_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\street\8k\street_pathway_state_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\street\8k\street_priority_state_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\street\8k\street_trending_state_8k.exr`
- `D:\2026 Arboreal Futures\data\renders\timeslices\street\8k\street_bioenvelope_positive_8k.exr`

Network:
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\street\8k\street_existing_condition_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\street\8k\street_pathway_state_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\street\8k\street_priority_state_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\street\8k\street_trending_state_8k.exr`
- `Z:\MF 2026 Arboreal Futures\blender\outputs\time slices\street\8k\street_bioenvelope_positive_8k.exr`
