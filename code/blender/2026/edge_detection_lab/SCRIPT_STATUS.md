# Script Status

This file records the current status of the main Edge Lab scripts.

It is intentionally blunt.

## Current Main Template

Primary compositor in active use:

- `edge_lab_final_template_safe_rebuild_20260405.blend`

## Script Roles

### Thin Or Mostly Thin Template Runners

- `render_edge_lab_current_core_outputs.py`
  - renders `ao`, `normals`, `resources` from the current template

- `render_edge_lab_current_bioenvelopes.py`
  - renders `bioenvelope` outputs from the current template

- `render_edge_lab_current_sizes.py`
  - renders `sizes` from the current template

- `render_edge_lab_current_base.py`
  - renders `base` outputs from the current template

### Current Shading

- `render_edge_lab_current_shading.py`
  - thin current-only shading renderer
  - repaths the saved current shading source EXRs
  - renders the saved `Current Shading ::Outputs` node
  - does not create helper nodes or own shading graph logic

### Template Runner With Bad Name

- `render_edge_lab_legacy_shading.py`
  - now should be treated as the legacy-scene renderer only
  - name is still bad
  - should eventually be renamed or replaced

### Wrapper Around Legacy Logic

- `render_edge_lab_current_mist.py`
  - opens the template
  - but actual mist generation is delegated to legacy scripted mist logic on a scratch scene
  - not yet template-native

- `render_edge_lab_current_depth_outliner.py`
  - opens the template
  - but actual depth outliner generation is delegated to legacy scripted depth logic on a scratch scene
  - not yet template-native

### Suite Orchestrator

- `run_edge_lab_final_template.py`
  - runs multiple family renderers to produce a suite
  - currently useful operationally
  - not the best debugging entrypoint

### Canonical Helper Renderers

- `render_proposal_only_layers.py`
  - renders filled proposal-only masks from the canonical helper blend
  - helper workflow, not part of the main template

- `render_proposal_outline_layers.py`
  - renders proposal outlines from the canonical helper blend
  - helper workflow, not part of the main template

## Current Refactor Priority

The main architectural cleanup target is:

1. mist
2. depth outliner
3. shading naming cleanup

because those are the places where runtime script ownership and template
ownership are still blurred.

## Validation Snapshot

Validated on `2026-04-07` using:

- `instantiate_template_and_render.py`
- latest-remote `city_timeline` EXRs
- a derived working copy of the canonical template

Confirmed working through the instantiation path:

- `ao`
- `normals`
- `resources`
- `bioenvelope`
- `base`

Not yet clean through the instantiation path:

- `shading`
  - moved to a dedicated current-only renderer
  - should now be revalidated through the instantiation path
