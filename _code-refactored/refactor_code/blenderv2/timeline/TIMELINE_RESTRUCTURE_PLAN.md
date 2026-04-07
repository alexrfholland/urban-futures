# Timeline / Single-State Shared Pipeline Plan

## Goal

We want one Blender pipeline with two data-assembly modes:

- `single_state`
- `timeline`

The goal is not to maintain two different Blender systems.
The goal is to maintain one scene contract, one render contract, and one operator workflow, with only the data assembly differing by mode.

## Why This Exists

Right now, `single_state` is the cleaner path.
`timeline` has historically drifted because it inherited extra city-specific behavior, alias layers, and older world-assembly logic.

We want `timeline` to behave like `single_state`, except that timeline mode assembles five years into strip positions.

## Acceptance Criteria

This restructure is successful when both modes share:

- the same 7 view-layer names
- the same EXR / AOV contract
- the same role-based parent hierarchy
- the same render-setup path
- the same meaning for `positive`, `priority`, and `trending`

And when timeline mode differs only by:

- assembling multiple years instead of one year
- applying strip placement / translation
- keeping timeline-specific bioenvelope strip handling where needed

The following legacy aliases should not be required in generated files:

- `city_priority`
- `city_bioenvelope`

This restructure should also produce a new unified script family, rather than continuing to expand the older `b2026_timeline_*` names.

Target naming direction:

- `b2026_unified_scene_contract.py`
- `b2026_unified_scene_setup.py`
- `b2026_unified_build_template.py`
- `b2026_unified_build_instancers.py`
- `b2026_unified_build_bioenvelopes.py`
- `b2026_unified_build_world.py`
- `b2026_unified_setup_render.py`
- `b2026_unified_render_exrs.py`
- `b2026_unified_render_workbench_view_layers.py`
- optionally `b2026_unified_build_scene.py` as the one wrapper entrypoint

The current `b2026_timeline_*` scripts should be treated as implementation references to migrate from, not as the final naming pattern.

## Scene Model

The Blender scene represents:

- `{site}`
- `{scenario}`
- `{year}`

It combines:

- tree positions
- log positions
- tree model libraries
- log model libraries
- bioenvelopes per state
- fixed road and building point clouds

The road/building point clouds are stable source geometry, but their point attributes are scenario/year-dependent and come from VTK-derived transfers.

## Modes

## `single_state`

This is one year only.

It contains:

- `positive` world and instancers
- `trending` world and instancers
- `priority`, which is a subset of `positive`

## `timeline`

This is the same logical scene, but across five years:

- `0`
- `10`
- `30`
- `60`
- `180`

Timeline mode should assemble those into strip positions, while preserving the same scenario semantics and render layers as `single_state`.

## Source Of Truth Files

Contract and naming:

- [b2026_timeline_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_scene_contract.py)

Shared scene setup:

- [b2026_timeline_scene_setup.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_scene_setup.py)

Timeline site specs, years, strip positions, and merged dataframe assembly:

- [b2026_timeline_layout.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_layout.py)

Template build entrypoint:

- [b2026_timeline_build_template_from_single_state.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_build_template_from_single_state.py)

Instancer build:

- [b2026_timeline_instancer.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_instancer.py)

Bioenvelope build:

- [b2026_timeline_bioenvelopes.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_bioenvelopes.py)

World rebuild:

- [b2026_timeline_rebuild_world_year_attrs.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_rebuild_world_year_attrs.py)

Render setup and EXR output:

- [b2026_timeline_render_lightweight_isolated_exrs_generic.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_render_lightweight_isolated_exrs_generic.py)

Workbench QA previews:

- [b2026_timeline_render_workbench_view_layers.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_render_workbench_view_layers.py)

Current contract docs:

- [TIMELINE_SCENE_TEMPLATE.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/TIMELINE_SCENE_TEMPLATE.md)
- [TIMELINE_RUNBOOK.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/TIMELINE_RUNBOOK.md)

Legacy Blender pipeline reference, out of scope for this refactor unless explicitly needed:

- [KEY_SCRIPTS.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/KEY_SCRIPTS.md)

## Scripts In Scope

This plan is about these scripts:

- [b2026_timeline_build_template_from_single_state.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_build_template_from_single_state.py)
- [b2026_timeline_instancer.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_instancer.py)
- [b2026_timeline_bioenvelopes.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_bioenvelopes.py)
- [b2026_timeline_rebuild_world_year_attrs.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_rebuild_world_year_attrs.py)
- [b2026_timeline_render_lightweight_isolated_exrs_generic.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_render_lightweight_isolated_exrs_generic.py)

Shared helpers introduced to reduce drift:

- [b2026_timeline_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_scene_contract.py)
- [b2026_timeline_scene_setup.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_scene_setup.py)

## Target Script Deliverables

The refactor should build new unified scripts with clearer ownership, instead of continuing to pile behavior into the older timeline-specific files.

Target deliverables:

- `b2026_unified_scene_contract.py`
  - one source of truth for naming, view layers, hierarchy roles, and scenario semantics
- `b2026_unified_scene_setup.py`
  - one source of truth for scene shell creation, collection setup, and view-layer setup
- `b2026_unified_build_template.py`
  - start from the single-state template and save out either a single-state-ready shell or a timeline-ready shell
- `b2026_unified_build_instancers.py`
  - build tree/log position objects, model libraries, and GN instancers from a mode-specific assembled node dataframe
- `b2026_unified_build_bioenvelopes.py`
  - build envelopes for both modes, with timeline-specific strip assembly only where needed
- `b2026_unified_build_world.py`
  - build the road/building point clouds for both modes from a shared world-assembly path
- `b2026_unified_setup_render.py`
  - ensure cameras, world, passes, AOVs, materials, and view-layer exclusions are correct
- `b2026_unified_render_exrs.py`
  - output the production multilayer EXRs
- `b2026_unified_render_workbench_view_layers.py`
  - output quick QA previews per view layer
- `b2026_unified_build_scene.py`
  - optional wrapper that runs the correct build order for `single_state` or `timeline`

Migration rule:

- existing `b2026_timeline_*` files are the migration source
- new `b2026_unified_*` files are the target public pipeline
- once the unified scripts are stable, the older files can become legacy/internal

## Shared View-Layer Contract

Both modes should use these exact view layers:

- `pathway_state`
- `existing_condition`
- `priority_state`
- `existing_condition_trending`
- `bioenvelope_positive`
- `bioenvelope_trending`
- `trending_state`

### Layer Visibility Semantics

This is the intended visibility contract at the scenario level.

| View layer | Visible world | Visible instancers | Visible envelope |
| --- | --- | --- | --- |
| `existing_condition` | positive world only | none | none |
| `pathway_state` | positive world | positive instancers | none |
| `priority_state` | positive world | priority instancers only | none |
| `existing_condition_trending` | trending world only | none | none |
| `trending_state` | trending world | trending instancers | none |
| `bioenvelope_positive` | positive world | none | positive envelope |
| `bioenvelope_trending` | trending world | none | trending envelope |

### Current Code For Layer Visibility

Single-state layer exclusions:

- [b2026_timeline_generate_single_state.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_generate_single_state.py)

Timeline layer exclusions:

- [b2026_timeline_rebuild_world_year_attrs.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_rebuild_world_year_attrs.py)

Detailed single-state hierarchy matrix:

- [TIMELINE_SCENE_TEMPLATE.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/TIMELINE_SCENE_TEMPLATE.md)

## Shared Hierarchy Contract

The intent is to share the same hierarchy roles, even if literal collection names remain mode-specific for now.

Shared roles:

- manager
- base / setup
- cameras
- positive
- priority
- trending
- bioenvelope positive
- bioenvelope trending

Short-term clarification:

- `single_state` currently uses generated roots such as `{site}_setup`
- `timeline` currently uses site-specific roots such as `City_Base`

So "shared hierarchy" currently means:

- same role set
- same parent semantics
- same layer semantics

It does **not** yet mean:

- identical literal collection names across both modes

## What Already Exists

These parts are already in place and should be reused, not reinvented:

### Timeline dataframe precompute for instancers

Already exists in:

- [b2026_timeline_layout.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_layout.py)
- [b2026_timeline_instancer.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_instancer.py)

This already:

- builds merged multi-year node dataframes
- applies strip translations during dataframe assembly
- passes one merged translated dataframe into the instancer builder

### Single-state build wrapper

Already exists in:

- [b2026_timeline_generate_single_state.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_generate_single_state.py)

### Shared view-layer naming contract

Already exists in:

- [b2026_timeline_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_scene_contract.py)

## What Still Needs To Change

### 1. World/base should match the instancer approach

This is the main remaining drift point.

The target behavior is:

- build one merged translated world dataset for `positive`
- build one merged translated world dataset for `trending`
- write those into explicit timeline world branches

Instead of relying on a different special-case path for timeline world assembly.

### 2. Timeline should precompute reusable strip subsets for roads/buildings

For world/base point clouds:

- compute the five strip subsets once
- reuse those subsets for both positive and trending
- apply year/scenario-specific VTK transfers onto those subsets
- translate and concatenate into final merged timeline world objects

### 3. Timeline should keep using explicit world branches

The target timeline world structure is:

- `City_Base > City_World`
  - raw source road/building point clouds
- `City_Base > City_World_Timeline > Year_{site}_timeline_world_positive`
  - merged translated positive timeline world objects
- `City_Base > City_World_Timeline > Year_{site}_timeline_world_trending`
  - merged translated trending timeline world objects

This allows the shared view layers to work cleanly.

### 4. Bioenvelopes may remain the main mode-specific exception

Bioenvelopes are allowed to keep some timeline-specific strip assembly logic, because they are currently clipped/assembled as strip geometry.

That is acceptable if they still follow:

- the same parent roles
- the same layer names
- the same scenario semantics

## Exact Mode Delta After Refactor

After the refactor, these should be the only meaningful differences:

| Area | `single_state` | `timeline` |
| --- | --- | --- |
| years | one year | `0, 10, 30, 60, 180` |
| node dataframe assembly | one-year dataframe | merged translated multi-year dataframe |
| world/base assembly | one-year world copy per scenario | merged translated multi-year world copy per scenario |
| strip placement | none | yes |
| strip boxes / translate nodes | none | yes |
| bioenvelope assembly | one-year envelope | multi-year strip envelope assembly |

Everything else should be shared.

## Proposed Build Order

Both modes should follow the same high-level build order:

1. build scene shell
2. build instancers
3. build bioenvelopes
4. build world/base point clouds
5. apply render setup
6. render previews or EXRs

### Target Unified Build Order

Once the unified script family exists, the desired order should be:

1. `b2026_unified_build_template.py`
2. `b2026_unified_build_instancers.py`
3. `b2026_unified_build_bioenvelopes.py`
4. `b2026_unified_build_world.py`
5. `b2026_unified_setup_render.py`
6. `b2026_unified_render_workbench_view_layers.py` or `b2026_unified_render_exrs.py`

Or, if using the wrapper:

1. `b2026_unified_build_scene.py`
2. `b2026_unified_setup_render.py`
3. `b2026_unified_render_workbench_view_layers.py` or `b2026_unified_render_exrs.py`

### Current Script Order

Timeline:

1. [b2026_timeline_build_template_from_single_state.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_build_template_from_single_state.py)
2. [b2026_timeline_instancer.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_instancer.py)
3. [b2026_timeline_bioenvelopes.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_bioenvelopes.py)
4. [b2026_timeline_rebuild_world_year_attrs.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_rebuild_world_year_attrs.py)
5. [b2026_timeline_render_lightweight_isolated_exrs_generic.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_render_lightweight_isolated_exrs_generic.py) with `B2026_SETUP_ONLY=1`
6. previews or EXRs

Single-state:

1. open single-state template
2. [b2026_timeline_instancer.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_instancer.py)
3. [b2026_timeline_bioenvelopes.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_bioenvelopes.py)
4. [b2026_timeline_rebuild_world_year_attrs.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_rebuild_world_year_attrs.py)
5. [b2026_timeline_render_lightweight_isolated_exrs_generic.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_render_lightweight_isolated_exrs_generic.py) with `B2026_SETUP_ONLY=1`
6. previews or EXRs

## Starting Files

This plan assumes:

- the single-state template is the cleanest source template
- timeline templates can be saved out as convenience artifacts
- timeline templates do not need to be conceptually separate templates

Current city source template:

- [2026 futures city single-state template.blend](/d:/2026%20Arboreal%20Futures/data/2026%20futures%20city%20single-state%20template.blend)

Current timeline builder output examples:

- [2026 futures city timeline template v4.blend](/e:/2026%20Arboreal%20Futures/blender/2026%20futures%20city%20timeline%20template%20v4.blend)

Open question for future rollout:

- city is covered now
- parade / street / uni still need the same source-template story formalized if this shared approach is rolled out across all sites

## Immediate Implementation Priorities

1. keep the shared contract strict
2. keep the shared layer semantics strict
3. move timeline world assembly to the same conceptual model as instancer assembly
4. allow bioenvelopes to remain the one controlled exception
5. reduce operator-facing scripts to the minimal core path

## Summary

The refactor is behavior-preserving at the contract level if:

- both modes expose the same 7 layers
- those layers mean the same thing
- the EXR/AOV outputs stay the same
- the parent roles stay the same

The only legitimate mode difference should be:

- one-year assembly in `single_state`
- merged strip assembly in `timeline`

## Validation Gate

Do not treat this refactor as complete until the generated Blender scenes pass a full view-layer validation test.

Validation requirements:

- every required view layer exists
- every view layer shows the correct world branch
- every view layer shows the correct instancer branch, or no instancers where none should be visible
- every view layer shows the correct bioenvelope branch, or no envelope where none should be visible
- no legacy alias layers are required
- no extra collections are visibly leaking into the wrong layer

Required layers to validate:

- `pathway_state`
- `existing_condition`
- `priority_state`
- `existing_condition_trending`
- `bioenvelope_positive`
- `bioenvelope_trending`
- `trending_state`

Minimum validation workflow:

1. build the scene
2. run [b2026_timeline_render_workbench_view_layers.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_render_workbench_view_layers.py) or the future `b2026_unified_render_workbench_view_layers.py`
3. inspect every generated preview and confirm the visibility contract is correct
4. fix any incorrect view-layer exclusions, hierarchy leaks, or missing collections
5. repeat until all layers are correct
6. only then run the EXR setup and production EXR render

Final acceptance rule:

- if the view layers are not all correct, the refactor is not done
- once the view layers are correct, render the EXRs as the final proof that the pipeline is working end to end

