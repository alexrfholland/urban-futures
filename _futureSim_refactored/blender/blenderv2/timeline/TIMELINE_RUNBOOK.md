# Timeline Runbook

This is the operator-facing run order for the refactored timeline pipeline.
Use [TIMELINE_SCENE_TEMPLATE.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_futureSim_refactored/blender/timeline/TIMELINE_SCENE_TEMPLATE.md) as the scene contract.

## Start Here

Open or copy a cleaned site blend:

- city: `D:\2026 Arboreal Futures\data\2026 futures city lightweight cleaned.blend`
- parade: `D:\2026 Arboreal Futures\data\2026 futures parade lightweight cleaned.blend`
- street: `D:\2026 Arboreal Futures\data\2026 futures street lightweight cleaned.blend`

For experimental work, make a copy first.

## Preconditions

Before generation, the file should already contain:

- the site scene
- the required view layers
- the required base world objects
- `instance_template`
- `MINIMAL_RESOURCES`
- `Envelope` and `Envelope Parade` where needed
- `WORLD_AOV`
- the approved render camera, or a known source blend to import it from

For single-state work, do not keep stale timeline cubes or old `__timeline` world objects in the working file.

## Data Inputs

Simulation data:

- `D:\2026 Arboreal Futures\urban-futures\_data-refactored\v3engine_outputs\feature-locations\{site}`
- `D:\2026 Arboreal Futures\urban-futures\_data-refactored\v3engine_outputs\bioenvelopes\{site}`
- `D:\2026 Arboreal Futures\urban-futures\_data-refactored\v3engine_outputs\vtks\{site}`

Model libraries (canonical, under `_data-refactored/model-inputs/tree_library_exports/`):

- trees: `_data-refactored/model-inputs/tree_library_exports/treeMeshesPly`
- logs: `_data-refactored/model-inputs/tree_library_exports/logMeshesPly`

Resolved at runtime by `resolve_tree_ply_folder()` / `resolve_log_ply_folder()` in
`bV2_build_instancers.py`.

## Build Mode

Set one of:

- `B2026_TIMELINE_BUILD_MODE=timeline`
- `B2026_TIMELINE_BUILD_MODE=single_state`

For single-state mode, also set:

- `B2026_SINGLE_STATE_YEAR`

## Generate Or Regenerate A Scene

Public unified pipeline:

1. `b2026_unified_build_template.py`
   - opens the single-state template source
   - prepares either a `timeline` shell or a `single_state` shell
   - removes legacy alias view layers from the saved output
2. `b2026_unified_build_scene.py`
   - runs instancers, bioenvelopes, and world rebuild
   - applies the single-state layer contract when `B2026_TIMELINE_BUILD_MODE=single_state`
   - runs `b2026_unified_validate_scene.py` by default
3. `b2026_unified_setup_render.py`
   - restores materials, passes, AOVs, camera, and mist without rendering
4. `b2026_unified_render_workbench_view_layers.py` or `b2026_unified_render_exrs.py`

If you want one wrapper for build generation, use:

- `b2026_unified_build_scene.py`
  - set `B2026_UNIFIED_BUILD_TEMPLATE=1` when the run should start from the single-state template
  - leave `B2026_UNIFIED_BUILD_TEMPLATE=0` when the target blend is already open and should just be regenerated

Implementation order behind the unified wrapper:

1. `b2026_unified_build_instancers.py`
   - builds tree/log point clouds
   - attaches the generated Geometry Nodes instancers
2. `b2026_unified_build_bioenvelopes.py`
   - imports and wires the envelope meshes
   - ensures timeline strip boxes exist so bio builds do not depend on the later world pass
3. `b2026_unified_build_world.py`
   - rebuilds the world-state geometry attributes

Use generation scripts when:

- source data changed
- scripts changed and the file must be refreshed
- you are building from a template shell

Do not rerun generation scripts just to render an already-finished saved single-state file.

## Prepare A Saved File For Rendering

Run:

- `b2026_unified_setup_render.py`

This step:

- restores required materials
- ensures passes and AOVs exist
- assigns the target camera
- applies mist settings
- saves the blend

Useful camera and mist overrides:

- `B2026_CAMERA_SOURCE_BLEND`
- `B2026_CAMERA_SOURCE_NAME`
- `B2026_MIST_SOURCE_BLEND`
- `B2026_MIST_SOURCE_SCENE`

## Preview Renders

Optional preview helpers:

- `b2026_unified_render_workbench_view_layers.py`
  - simple fast PNG previews for each view layer using `BLENDER_WORKBENCH`
  - intended for framing and isolation checks, not AOV or EXR validation
- `b2026_timeline_render_lightweight_previews_generic.py`
- `b2026_timeline_render_previews.py`
- `b2026_timeline_apply_proposals_material.py`
  - builds `PROPOSALS`
  - swaps world point and instancer materials to a proposal-colour preview material
  - useful for checking proposal state in the viewport before render
  - production EXR setup restores `WORLD_AOV` and `MINIMAL_RESOURCES`

Use previews to confirm:

- view-layer isolation
- camera framing
- instancer distribution
- envelope visibility
- mist

### Fast Workbench View-Layer Previews

Use `b2026_unified_render_workbench_view_layers.py` when the goal is:

- confirm framing
- confirm crop
- confirm view-layer isolation
- get fast previews without the cost of Cycles

This script:

- takes a scene
- optionally takes a camera
- optionally filters to target view layers
- renders one PNG per view layer using `BLENDER_WORKBENCH`
- restores the original scene render settings afterward

This is appropriate for quick QA only.
Do not use it to validate the production EXR/AOV contract.

Useful env vars:

- `B2026_SCENE_NAME`
- `B2026_CAMERA_NAME`
- `B2026_OUTPUT_DIR`
- `B2026_OUTPUT_PREFIX`
- `B2026_TARGET_VIEW_LAYERS`
- `B2026_RES_X`
- `B2026_RES_Y`
- `B2026_RES_PERCENT`

## Production EXRs

Run:

- `b2026_unified_render_exrs.py`

Expected output view layers:

- `existing_condition`
- `pathway_state`
- `priority_state`
- `existing_condition_trending`
- `trending_state`
- `bioenvelope_positive`
- `bioenvelope_trending`

The EXR framebuffer schema is documented in [TIMELINE_SCENE_TEMPLATE.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_futureSim_refactored/blender/timeline/TIMELINE_SCENE_TEMPLATE.md).

## Preflight

Check:

- the scene camera is correct
- mist is correct
- the target collections exist
- the target view layers exist
- the output directory exists
- instancers are present on the point-position objects
- world objects and envelopes follow the scene contract
- `b2026_unified_validate_scene.py` passes before EXR render

## If The File Is Wrong

Use this order:

1. compare the blend to [TIMELINE_SCENE_TEMPLATE.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_futureSim_refactored/blender/timeline/TIMELINE_SCENE_TEMPLATE.md)
2. rerun `b2026_unified_build_scene.py`
3. rerun `b2026_unified_validate_scene.py`
4. rerun `b2026_unified_setup_render.py`

## Related Files

- [AGENTS.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_futureSim_refactored/blender/timeline/AGENTS.md)
- [TIMELINE_SCENE_TEMPLATE.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_futureSim_refactored/blender/timeline/TIMELINE_SCENE_TEMPLATE.md)
- [VIEWS.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_futureSim_refactored/blender/timeline/VIEWS.md)
