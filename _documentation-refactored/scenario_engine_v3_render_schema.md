# Scenario Engine V3 Render Schema

Source of truth:

- [_code-refactored/refactor_code/scenario/render_proposal_schema_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/render_proposal_schema_v3.py)
- [_code-refactored/refactor_code/scenario/pyvista_render_settings/README.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/pyvista_render_settings/README.md)

Layer priority, top to bottom:

1. `proposal-deploy-structure`
2. `proposal-decay`
3. `proposal-recruit`
4. `proposal-colonise`
5. `proposal-release-control`
6. deadwood base
7. white

## `proposal-deploy-structure`

| intervention | number | colour |
| --- | ---: | --- |
| `not-assessed` | `0` | `not visible` |
| `rejected` | `1` | `not visible` |
| `adapt-utility-pole` | `2` | `#FF0000` |
| `translocated-log` | `3` | `#8F89BF` |
| `upgrade-feature` | `4` | `#CE6DD9` |

## `proposal-decay`

| intervention | number | colour |
| --- | ---: | --- |
| `not-assessed` | `0` | `not visible` |
| `rejected` | `1` | `not visible` |
| `buffer-feature` | `2` | `#B83B6B` |
| `brace-feature` | `3` | `#D9638C` |

## `proposal-recruit`

| intervention | number | colour |
| --- | ---: | --- |
| `not-assessed` | `0` | `not visible` |
| `rejected` | `1` | `not visible` |
| `buffer-feature` | `2` | `#C5E28E` |
| `rewild-ground` | `3` | `#5CB85C` |

## `proposal-colonise`

| intervention | number | colour |
| --- | ---: | --- |
| `not-assessed` | `0` | `not visible` |
| `rejected` | `1` | `not visible` |
| `rewild-ground` | `2` | `#5CB85C` |
| `enrich-envelope` | `3` | `#8CCC4F` |
| `roughen-envelope` | `4` | `#B87A38` |

## `proposal-release-control`

| intervention | number | colour |
| --- | ---: | --- |
| `not-assessed` | `0` | `not visible` |
| `rejected` | `1` | use lifecycle colours with saturation reduced to `20%` |
| `reduce-pruning` | `2` | use lifecycle colours with saturation reduced to `50%` |
| `eliminate-pruning` | `3` | use lifecycle colours at `100%` saturation |

Lifecycle colours:

- `small` -> `#AADB5E`
- `medium` -> `#9AB9DE`
- `large` -> `#F99F76`
- `senescing` -> `#EB9BC5`
- `snag` -> `#FCE358`
- `fallen` -> `#82CBB9`
- `decayed` -> `#5F867E`
- `early-tree-death` -> `#6E6E6E`
- `artificial` -> `#FF0000`

`early-tree-death` is a terminal absent state used for young-tree mortality.

- it is defined in the render schema for debug safety
- it should normally be filtered out before voxel / render export, like `gone`
- it is not part of the intended visible lifecycle palette

Deadwood base:

- `fallen` -> `#8F89BF`
- `decayed` -> `#5F867E`

White fallback:

- `#FFFFFF`

Render settings:

- `2200 x 1600`
- `point_size = 4.0`
- `render_points_as_spheres = False`
- `lighting = False`
- EDL on
- white background

Named PyVista settings schema:

- `engine3-proposals`
