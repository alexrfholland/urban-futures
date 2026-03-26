# Tree Render Attribute Map

These are the point attributes and compositor-facing AOV names used by the 2026 Blender tree pipeline.

The instancer writes these values onto the tree point cloud. The `RESOURCES` material family then reads them as instancer attributes and writes them to render AOVs.

Primary source:

- `final/_blender/2026/b2026_instancer.py`

## Blender-imported resource mask attribute names

When the tree/log PLYs are imported into Blender, some `resource_*` attribute names are sanitized.
Use these Blender mesh attribute names in materials and debug tools:

| Source PLY / VTK name | Blender imported name |
| --- | --- |
| `resource_hollow` | `resource_hollow` |
| `resource_epiphyte` | `resource_epiphyte` |
| `resource_dead branch` | `resource_dead` |
| `resource_perch branch` | `resource_perch` |
| `resource_peeling bark` | `resource_peeling` |
| `resource_fallen log` | `resource_fallen` |
| `resource_other` | `resource_other` |

## size

- AOV name: `size`
- Original column: `size`
- Type: integer enum

| Value | Label |
| --- | --- |
| `1` | `small` |
| `2` | `medium` |
| `3` | `large` |
| `4` | `senescing` |
| `5` | `snag` |
| `6` | `fallen` |
| `-1` | unknown / unmapped |

## control

- AOV name: `control`
- Original column: `control`
- Type: integer enum

| Value | Label |
| --- | --- |
| `1` | `street-tree` |
| `2` | `park-tree` |
| `3` | `reserve-tree` |
| `4` | `improved-tree` |
| `-1` | unknown / unmapped |

## node_type

- AOV name: `node_type`
- Original column: `nodeType`
- Type: integer enum

| Value | Label |
| --- | --- |
| `0` | `tree` |
| `1` | `pole` |
| `2` | `log` |
| `-1` | unknown / unmapped |

## tree_interventions

- AOV name: `tree_interventions`
- Original column: `rewilded`
- Type: integer enum

| Value | Label |
| --- | --- |
| `3` | `Rewilded` (`node-rewilded`, `rewilded`) |
| `2` | `Habitat island` (blank / missing, `footprint-depaved`) |
| `1` | `Brace` (`exoskeleton`) |
| `0` | `None` (`none`, `paved`) |
| `-1` | unknown / unmapped |

## tree_proposals

- AOV name: `tree_proposals`
- Original column: `action`
- Type: integer enum

| Value | Label |
| --- | --- |
| `2` | `AGE-IN-PLACE` |
| `1` | `SENESCENT` |
| `0` | `REPLACE` |
| `-1` | blank / missing / unknown |

## improvement

- AOV name: `improvement`
- Original column: `Improvement`
- Type: integer enum

| Value | Label |
| --- | --- |
| `1` | yes / true / `1` |
| `0` | no / false / blank / `0` |
| `-1` | unknown / unmapped |

## canopy_resistance

- AOV name: `canopy_resistance`
- Original column: `CanopyResistance`
- Type: float passthrough

Notes:

- This is written directly as a numeric value.
- Missing values are stored as `-1.0`.

## node_id

- AOV name: `node_id`
- Original column: `nodeID`
- Type: integer passthrough

Notes:

- This is written directly as a numeric value.
- Missing values are stored as `-1`.

## Existing passthrough attrs still used by the tree renderer

These are already part of the current tree system and remain available:

| AOV / attr | Source |
| --- | --- |
| `structure_id` | `structureID` |
| `instanceID` / `instance_id` | derived model index |
| `tree_type` | `tree_id` |
| `rotation` | `rotateZ` |
| `life_expectancy` | `useful_life_expectancy` |

## Current trimmed-parade notes

- In `trimmed-parade_positive_1_nodeDF_180.csv`, `rewilded` currently includes `node-rewilded`, `footprint-depaved`, blank, `exoskeleton`, and `paved`.
- In `trimmed-parade_trending_1_nodeDF_180.csv`, `rewilded` currently includes `paved` and `none`.
- In the current trimmed-parade data, `Improvement` is effectively `False` or blank only.
- In the current trimmed-parade data, `nodeID` is mostly missing / `-1`.

# Rewilded Envelope PLY Attribute Map

These are the integer fields baked into the rewilded envelope PLYs before Blender import.

Primary sources:

- `final/_blender/b_generate_rewilded_envelopes.py`
- `final/_blender/2026/b2026_city_envelope_aov_setup.py`

The envelope generator writes integer point-data arrays onto the output polydata. Blender then exposes them as value AOVs:

| PLY point attribute | Blender AOV | Type | Notes |
| --- | --- | --- | --- |
| `scenario_bioEnvelope_int` | `bioEnvelopeType` | integer enum | Full category mapping from `scenario_bioEnvelope`. |
| `scenario_bioEnvelope_simple_int` | `bioSimple` | integer enum | Simplified grouping from `scenario_bioEnvelope`. |
| `sim_Turns` | `sim_Turns` | integer passthrough | Written directly from the source sim data. |

## bioEnvelopeType

- PLY point attribute: `scenario_bioEnvelope_int`
- Blender AOV name: `bioEnvelopeType`
- Source string attribute: `scenario_bioEnvelope`
- Type: integer enum

| Value | Label |
| --- | --- |
| `0` | fallback / unmapped |
| `1` | `exoskeleton` |
| `2` | `brownRoof` |
| `3` | `otherGround` |
| `4` | `node-rewilded`, `rewilded` |
| `5` | `footprint-depaved` |
| `6` | `livingFacade` |
| `7` | `greenRoof` |

## bioSimple

- PLY point attribute: `scenario_bioEnvelope_simple_int`
- Blender AOV name: `bioSimple`
- Source string attribute: `scenario_bioEnvelope`
- Type: integer enum

| Value | Label |
| --- | --- |
| `1` | default / other |
| `2` | `brownRoof` |
| `3` | `livingFacade` |
| `4` | `greenRoof` |

## Notes

- `node-rewilded` and `rewilded` are intentionally collapsed to the same full-category value: `4`.
- Any `scenario_bioEnvelope` category not listed in the simplified map falls back to `1` in `bioSimple`.
- Any `scenario_bioEnvelope` category not listed in the full map falls back to `0` in `bioEnvelopeType`.
