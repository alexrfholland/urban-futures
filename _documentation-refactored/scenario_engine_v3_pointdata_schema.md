# Scenario Engine V3 Point-Data Schema Review

Reference file inspected:

- `_data-refactored/v3engine_outputs/vtks/trimmed-parade/trimmed-parade_positive_1_yr180_state_with_indicators.vtk`

Point-data count in that file:

- `101`

## Status Meanings

- `keep`: keep in the lean final export
- `drop`: remove at preflight `possibility_space_ds`
- `drop-before-save`: keep during processing, strip before final VTK save

## Notes

- This classification is for the current V3 candidate path.
- `FEATURES-*` is `drop-before-save`, not `drop`, because those arrays are still used to derive `search_*`.
- `proposal_*V3` and `*_intervention` arrays are the active V3 proposal outputs and should be kept.
- Legacy `proposal_*` point-data arrays are being kept for now because V2/V3 comparison is still in progress.
- `sim_Turns`, `sim_averageResistance`, `sim_Nodes`, and `maskForRewilding` remain `keep` because current Blender tooling still depends on them.

## Grid

```text
voxel_I | drop-before-save
voxel_J | drop-before-save
voxel_K | drop-before-save
```

## Raw Site / Preflight Analysis

```text
site_building_element | drop-before-save
isTerrainUnderBuilding | drop-before-save
envelope_roofType | drop-before-save
analysis_busyRoadway | drop
analysis_Roadway | drop
analysis_potentialNOCANOPY | drop
analysis_combined_resistance | drop-before-save
analysis_combined_resistanceNOCANOPY | drop-before-save
analysis_nodeType | drop
analysis_nodeID | drop-before-save
node_CanopyID | drop-before-save
node_CanopyResistance | drop
envelopeIsBrownRoof | drop
sim_nodeType | drop
```

## Core Sim

```text
sim_Nodes | keep
sim_Turns | keep
sim_averageResistance | keep
```

## Scenario State

```text
scenario_rewildingEnabled | keep
scenario_rewildingPlantings | keep
scenario_under-node-treatment | keep
scenario_bioEnvelope | keep
scenario_outputs | keep
```

## Resources

```text
resource_hollow | keep
resource_epiphyte | keep
resource_dead branch | keep
resource_perch branch | keep
resource_peeling bark | keep
resource_fallen log | keep
resource_other | keep
```

## Resource Stats

```text
stat_hollow | keep
stat_epiphyte | keep
stat_dead branch | keep
stat_perch branch | keep
stat_peeling bark | keep
stat_fallen log | keep
stat_other | keep
```

## Forest Payload

```text
nodeType | drop-before-save
forest_resource | drop-before-save
forest_precolonial | keep
forest_size | keep
forest_control | keep
forest_tree_id | drop-before-save
forest_diameter_breast_height | drop-before-save
forest_tree_number | drop-before-save
forest_NodeID | drop-before-save
forest_structureID | drop-before-save
forest_useful_life_expectancy | drop-before-save
forest_isNewTree | drop-before-save
forest_rotateZ | drop-before-save
nodeTypeInt | drop-before-save
```

## Derived / Debug

```text
updatedResource_elevatedDeadBranches | drop-before-save
updatedResource_groundDeadBranches | drop-before-save
maskforTrees | drop-before-save
maskForRewilding | keep
```

## V3 Proposals

```text
proposal_decayV3 | keep
proposal_release_controlV3 | keep
proposal_coloniseV3 | keep
proposal_recruitV3 | keep
proposal_deploy_structureV3 | keep
```

## V3 Proposal Interventions

```text
proposal_decayV3_intervention | keep
proposal_release_controlV3_intervention | keep
proposal_coloniseV3_intervention | keep
proposal_recruitV3_intervention | keep
proposal_deploy_structureV3_intervention | keep
```

## Transferred Site FEATURES

```text
FEATURES-site_building_element | drop-before-save
FEATURES-site_canopy_isCanopy | drop-before-save
FEATURES-road_terrainInfo_roadCorridors_str_type | drop-before-save
FEATURES-road_roadInfo_type | drop-before-save
FEATURES-road_terrainInfo_forest | drop-before-save
FEATURES-road_terrainInfo_isOpenSpace | drop-before-save
FEATURES-road_terrainInfo_isParkingMedian3mBuffer | drop-before-save
FEATURES-road_terrainInfo_isLittleStreet | drop-before-save
FEATURES-road_terrainInfo_isParking | drop-before-save
FEATURES-road_canopy_isCanopy | drop-before-save
FEATURES-envelope_roofType | drop-before-save
FEATURES-analysis_busyRoadway | drop-before-save
FEATURES-analysis_Roadway | drop-before-save
FEATURES-analysis_Canopies | drop-before-save
```

## Search

```text
search_bioavailable | keep
search_design_action | keep
search_urban_elements | keep
```

## Indicators

```text
indicator_Bird_self_peeling | keep
indicator_Bird_others_perch | keep
indicator_Bird_generations_hollow | keep
indicator_Lizard_self_grass | keep
indicator_Lizard_self_dead | keep
indicator_Lizard_self_epiphyte | keep
indicator_Lizard_others_notpaved | keep
indicator_Lizard_generations_nurse-log | keep
indicator_Lizard_generations_fallen-tree | keep
indicator_Tree_self_senescent | keep
indicator_Tree_others_notpaved | keep
indicator_Tree_generations_grassland | keep
```

## Legacy V2 Proposal Arrays

```text
proposal_decay | keep
proposal_recruit | keep
proposal_release_control | keep
proposal_colonise | keep
proposal_deploy_structure | keep
```
