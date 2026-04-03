# Timeline Blender Scripts

## Generation Note

This is the `v1.5` timeline rewrite/reference note.

- created: `2026-04-02 20:01:42`
- last updated: `2026-04-03 13:56:56`
- scope: refactored `b2026_unified_*` / `b2026_timeline_*` timeline pipeline
- status: current `v1.5` reference, but not the active `blenderv2` note

It still has relevant operational guidance for the timeline pipeline, but new `blenderv2` work should use the dedicated `blenderv2` docs and scripts instead of extending this file as the primary source of truth.

This folder is the refactored timeline pipeline for the 2026 Blender workflow.
Prefer these docs before reading individual scripts:

- [TIMELINE_RUNBOOK.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/TIMELINE_RUNBOOK.md)
- [TIMELINE_SCENE_TEMPLATE.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/TIMELINE_SCENE_TEMPLATE.md)

## Core Files

Use this public unified surface first:

- `b2026_unified_scene_contract.py`
  - public source of truth for naming, layer semantics, and validation expectations
- `b2026_unified_scene_setup.py`
  - shared collection/view-layer helpers for both modes
- `b2026_unified_build_template.py`
  - prepares either a `timeline` or `single_state` shell from the single-state template
- `b2026_unified_build_scene.py`
  - runs instancers, bioenvelopes, world rebuild, and scene validation
- `b2026_unified_setup_render.py`
  - setup-only render preparation
- `b2026_unified_render_workbench_view_layers.py`
  - fast QA previews for all standard layers
- `b2026_unified_render_exrs.py`
  - production EXR render entrypoint
- `b2026_unified_validate_scene.py`
  - structural validation gate for the 7-layer contract

Implementation references behind the unified surface:

- `b2026_timeline_scene_contract.py`
- `b2026_timeline_scene_setup.py`
- `b2026_timeline_layout.py`
- `b2026_timeline_instancer.py`
- `b2026_timeline_bioenvelopes.py`
- `b2026_timeline_rebuild_world_year_attrs.py`
- `b2026_timeline_render_lightweight_isolated_exrs_generic.py`
- `b2026_timeline_generate_single_state.py`

## Optional Helpers

- `b2026_timeline_render_lightweight_previews_generic.py`
- `b2026_timeline_render_previews.py`
- `b2026_timeline_render_workbench_view_layers.py`
- `b2026_timeline_apply_proposals_material.py`
- `b2026_timeline_import_paraview_cameras.py`
- `b2026_timeline_clipbox_setup.py`
- `b2026_timeline_camera_clipboxes.py`
- `b2026_timeline_runtime_flags.py`

Clip-box helpers are legacy/optional and are off by default in the current workflow.

## Legacy / One-Offs

These are not the primary pipeline path and should not be the first scripts you reach for:

- site-specific cleaned-blend makers and refreshers
- camera fitting / zoom tests / debug camera rebuilds
- old parade-specific render/export scripts
- debug material and compositor one-offs
- older preview/export variants that predate the generic scripts

## Notes

- the refactored pipeline supports `timeline` and `single_state`
- the `b2026_unified_*` files are the public workflow; the older `b2026_timeline_*` files are implementation references
- the older non-timeline workflow remains in [b2026_instancer.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/b2026_instancer.py)
- the scene contract should be updated from verified saved blends, not from memory

## GUI Blend Opening

- when opening a `.blend` for the user in Blender GUI, launch Blender with the target blend path as an explicit quoted argument
- prefer `Start-Process ... -PassThru`, wait a few seconds, and verify the Blender process is actually running
- after launch, confirm the process command line includes the target `.blend` path before telling the user it is open
- if Blender is running but the window title is ambiguous, trust the verified command line over the title text
- do not report success based only on `Start-Process` returning without checking that Blender stayed open
