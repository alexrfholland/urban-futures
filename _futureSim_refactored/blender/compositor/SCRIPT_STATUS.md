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

### Current Depth Outliner

- `render_edge_lab_current_depth_outliner.py`
  - thin current-only depth renderer
  - repaths the saved depth EXRs
  - renders the saved `DepthOutliner::Outputs` node
  - does not create scratch scenes or delegate depth graph logic

### Current Mist

- `render_edge_lab_current_mist.py`
  - thin current-only mist renderer
  - repaths the saved mist EXRs
  - renders the saved `MistOutlines::Outputs` node
  - falls back to direct socket renders from the saved template contract if Blender skips file-output slots
  - does not create scratch scenes or delegate mist graph logic

### Template Runner With Bad Name

- `render_edge_lab_legacy_shading.py`
  - now should be treated as the legacy-scene renderer only
  - name is still bad
  - should eventually be renamed or replaced

### Suite Orchestrator

- `run_edge_lab_final_template.py`
  - runs multiple family renderers to produce a suite
  - currently useful operationally
  - not the best debugging entrypoint

### Canonical Helper Renderers

- `render_proposal_only_layers.py`
  - should be treated as a thin runner for:
    - `canonical_templates/proposal_only_layers.blend`
  - helper workflow, not part of the main template
  - should open the canonical helper blend, repath the single `EXR` node, render, and exit

- `render_proposal_outline_layers.py`
  - should be treated as a thin runner for:
    - `canonical_templates/proposal_outline_layers.blend`
  - helper workflow, not part of the main template
  - should open the canonical helper blend, repath the single `EXR` node, render, and exit

### Agent Rule

For agents working in this folder:

- edit canonical blends when workflow logic needs to change
- use thin runners when only the EXR set or output folder changes
- use one-off headless Blender commands for inspection or probing
- do not rebuild existing canonical helper blends from scratch as the normal render path

## Current Refactor Priority

The main architectural cleanup target is:

1. continue normalizing helper renderers so they are clearly thin runners
2. any cleanup needed to remove mist file-output fallback if Blender behavior becomes reliable

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

- none of the current template-owned families above are blocked on legacy scratch-scene logic

Validated separately as thin current-only renderers:

- `shading`
- `depth_outliner`
- `mist`
