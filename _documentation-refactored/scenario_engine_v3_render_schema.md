# Scenario Engine V3 Render Schema

Source of truth:

- [_code-refactored/refactor_code/scenario/render_proposal_schema_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/render_proposal_schema_v3.py)
- [_code-refactored/refactor_code/scenario/render_custom_proposal_schema_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/render_custom_proposal_schema_v3.py)
- [_code-refactored/refactor_code/scenario/pyvista_render_settings/README.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/pyvista_render_settings/README.md)

This renderer now writes two related products:

- default outputs:
  - `engine3-proposals_interventions_with-legend`
  - `engine3-proposals`
- opt-in extra variants:
  - `engine3-proposals_interventions`
  - `engine3-proposals_with-legend`

- `engine3-proposals_interventions`
  - intervention-focused view
  - only accepted intervention states are visible
- `engine3-proposals`
  - proposal-presence view
  - a proposal counts as visible when its framebuffer value is anything except:
    - `0` = `not-assessed`
    - `1` = `rejected`
  - so this includes:
    - `-1` = accepted with no intervention allocated yet
    - any accepted intervention state `> 1`

## `engine3-proposals_interventions`

This is not part of the default visualisation set.

The default intervention-facing image is:

- `engine3-proposals_interventions_with-legend`

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

## `engine3-proposals`

This is the proposal-only companion image.

This is part of the default visualisation set.

It uses the same layer priority:

1. `proposal-deploy-structure`
2. `proposal-decay`
3. `proposal-recruit`
4. `proposal-colonise`
5. `proposal-release-control`
6. deadwood base
7. white

But it ignores intervention identity within a family.

Visibility rule:

- visible if framebuffer value is not `0` and not `1`
- hidden if framebuffer value is:
  - `0` = `not-assessed`
  - `1` = `rejected`

So this view shows both:

- `-1` accepted with no intervention allocated
- `2+` accepted with a specific intervention

Family colours:

- `proposal-deploy-structure` -> `#C05E5E`
- `proposal-decay` -> `#B83B6B`
- `proposal-recruit` -> `#5CB85C`
- `proposal-colonise` -> `#8CCC4F`
- `proposal-release-control` -> `#D4882B`

Deadwood base:

- `fallen` -> `#8F89BF`
- `decayed` -> `#5F867E`

White fallback:

- `#FFFFFF`
