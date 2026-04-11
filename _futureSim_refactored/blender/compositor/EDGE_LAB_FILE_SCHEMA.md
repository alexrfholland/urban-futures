# Edge Lab File Schema

This document defines the intended relationship between:

- saved `.blend` files
- generated `.blend` files
- render/build scripts
- inputs
- outputs

The immediate goal is to stop treating every `.blend` in `data/blender/2026/edge_detection_lab/` as equivalent.

## Core Principle

Scripts are the execution logic.

Blends are one of three things:

1. canonical templates
2. generated working copies
3. experiments / historical artifacts

Outputs are render artifacts, not templates.

## The Three Blend Roles

### 1. Canonical Template Blends

These are the human-facing, durable compositor sources that define the approved node graphs.

They are the files we intentionally edit and preserve.

Examples:

- `edge_lab_final_template.blend`
- `edge_lab_final_template_safe_rebuild_20260405.blend`
- `compositor_proposal_masks.blend`
- `mist_pathway_kirsch_simple.blend`
- `proposal_only_layers.blend`
- `proposal_outline_layers.blend`
- `base_sim_debug_city_timeline.blend`

Rules:

- these are the authoritative graph files
- they may be edited manually in Blender
- they should not be overwritten casually by broad builder scripts
- if a script depends on one, that dependency should be explicit

### 2. Generated Working Blends

These are reproducible derivatives of a canonical template.

They exist to test, checkpoint, repath, or adapt a canonical graph without risking the main file.

Examples:

- `edge_lab_final_template_safe_rebuild_20260405__checkpoint_*.blend`
- `edge_lab_final_template_pre_layout_refactor_20260405_1457.blend`
- `edge_lab_final_template_manual.blend`
- `edge_lab_final_template_city_timeline_positive_state_20260405.blend`
- `city_exr_compositor_lightweight_baseline_repathed.blend`

Rules:

- these are not new canonical workflows
- they are generated or saved-off variants of a canonical template
- they should be named to show lineage
- preferred naming pattern:
  - `[canonical-name]__checkpoint_[nn]_[short-purpose].blend`
  - `[canonical-name]__[dataset-or-purpose].blend`

### 3. Experimental / Historical Blends

These are one-off or legacy compositor experiments.

Examples:

- `exr_city_blender_*`
- `city_exr_compositor_lightweight*`
- `edge_lab_output_suite_combined.blend`
- `edge_lab_output_suite_refined.blend`
- `proposal_release_control_pathway_debug.blend`

Rules:

- do not confuse these with current canonical templates
- treat them as references, prototypes, or legacy milestones
- if a workflow is still important, promote it explicitly into a canonical template or a documented helper blend

## Script Roles

Scripts are the real workflow definitions.

### Build Scripts

These create or restructure `.blend` files.

Examples:

- `build_edge_lab_final_template_blend.py`
- `build_edge_lab_combined_compositor.py`
- `build_city_exr_compositor_lightweight.py`

Role:

- construct a blend graph
- add nodes, groups, frames, outputs
- save a blend

These should not be mistaken for renderers.

### Render Scripts

These execute a specific family or helper workflow.

Examples:

- `render_edge_lab_current_core_outputs.py`
- `render_edge_lab_current_mist.py`
- `render_edge_lab_current_bioenvelopes.py`
- `render_edge_lab_current_sizes.py`
- `render_edge_lab_legacy_shading.py`
- `render_proposal_only_layers.py`
- `render_proposal_outline_layers.py`

Role:

- open a blend or create a scratch scene
- repath inputs
- render PNGs
- write outputs

These are the operational scripts for production output.

### Run / Orchestration Scripts

These coordinate several render scripts into one suite run.

Examples:

- `run_edge_lab_final_template.py`
- `run_edge_lab_combined_compositor.py`
- `run_edge_lab_output_suite.py`

Role:

- select a dataset
- dispatch render families
- build a full output set

### Utility / Adapter Scripts

These adjust an existing workflow without defining a new canonical one.

Examples:

- `repath_city_exr_compositor_inputs.py`
- `add_trending_bioenvelope_exports.py`
- `enable_baseline_lightweight_resource_exports.py`

Role:

- repath
- patch
- add exports
- adapt a template for a nearby use case

## Relationship Between Templates, Generated Blends, And Scripts

The intended relationship is:

1. a canonical template defines the graph
2. build or utility scripts may derive a working copy from it
3. render scripts execute that graph against a specific EXR set
4. outputs are written into a run folder

So:

- template blend = durable graph source
- generated blend = temporary or checkpointed derivative
- script = executable workflow logic
- output folder = rendered artifact set

## Inputs

Inputs should live under:

- `data/blender/2026/edge_detection_lab/inputs/`

There are two input classes:

### Stable Local Named Sets

Examples:

- `city_8k_network_20260402`
- `parade_8k_network_20260402`

These are copied or frozen local sets.

### Latest Remote Mirrors

Examples:

- `inputs/LATEST_REMOTE_EXRS/...`

These are the current preferred source when the user says to use the latest remote EXRs.

Rule:

- if the user asks for latest data, use `LATEST_REMOTE_EXRS`
- do not silently fall back to an older local copied set

## Outputs

Outputs should live under:

- `data/blender/2026/edge_detection_lab/outputs/`

Each run should have a single run root, then workflow subfolders.

Preferred pattern:

- `outputs/[run-name]/current/ao`
- `outputs/[run-name]/current/normals`
- `outputs/[run-name]/current/resources`
- `outputs/[run-name]/current/depth_outliner`
- `outputs/[run-name]/current/outlines_mist`
- `outputs/[run-name]/current/shading`
- `outputs/[run-name]/current/base`
- `outputs/[run-name]/current/sizes`
- `outputs/[run-name]/current/bioenvelope`

Proposal-specific helpers may use their own run roots:

- `outputs/[run-name]/pathway`
- `outputs/[run-name]/priority`
- `outputs/[run-name]/trending`

Rules:

- do not dump flat PNG sets into the top of `outputs/` unless the workflow is explicitly single-family
- remove `_discard_render.png`
- normalize `_0001.png` names to final names where appropriate

## Naming Rules

### For Canonical Templates

Use short, durable names.

Examples:

- `edge_lab_final_template.blend`
- `compositor_proposal_masks.blend`

Avoid embedding run dates in the canonical filename unless the date is part of the version identity.

### For Working Copies

Use lineage-oriented names.

Examples:

- `edge_lab_final_template__checkpoint_03_sizes_stack_group.blend`
- `edge_lab_final_template__manual_layout_pass.blend`

### For Run Roots

Use:

- workflow or template name
- dataset
- sim version or input version
- date

Examples:

- `edge_lab_final_template_safe_rebuild_parade_timeline_simv3-7_20260406`
- `proposal_outline_layers_parade_timeline_simv3-7_20260406`

## Practical Current Mapping

Today, the current practical file roles are:

- main current template:
  - `data/blender/2026/edge_detection_lab/edge_lab_final_template_safe_rebuild_20260405.blend`
- legacy human-facing template:
  - `data/blender/2026/edge_detection_lab/edge_lab_final_template.blend`
- canonical proposal extractor:
  - `data/blender/2026/edge_detection_lab/compositor_proposal_masks.blend`
- simple mist debug template:
  - `data/blender/2026/edge_detection_lab/mist_pathway_kirsch_simple.blend`

## Restructure Target

The medium-term target should be:

### `data/.../edge_detection_lab/templates/`

Canonical human-facing blends only.

### `data/.../edge_detection_lab/checkpoints/`

Generated working copies and checkpoint blends.

### `data/.../edge_detection_lab/experiments/`

Legacy and one-off `.blend` experiments.

### `data/.../edge_detection_lab/inputs/`

All EXR inputs.

### `data/.../edge_detection_lab/outputs/`

All rendered PNG roots.

### `code/.../edge_detection_lab/`

All build, render, orchestration, and utility scripts plus schema docs.

## Decision Rule

If a file is being used as:

- a stable graph source edited by humans:
  - it is a template
- a saved-off intermediate or checkpoint:
  - it is a working blend
- a historical one-off test:
  - it is an experiment
- executable logic:
  - it is a script
- rendered artifact:
  - it belongs in outputs

That distinction should be explicit in both filename and directory placement.
