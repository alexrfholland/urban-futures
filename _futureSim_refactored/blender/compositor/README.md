# Compositor

## IMPORTANT PLEASE READ

- use a thin runner when the workflow already exists and only the EXR set or output path changes
- runners do not rebuild an existing canonical workflow
- runners open the canonical blend, repath inputs, set outputs, render, and exit
- if the workflow logic needs to change, edit the canonical blend instead

This folder is the current refactored home for the compositor-only EXR workflow.

## Structure

Code and docs live here:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor`

Current active scripts live here:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/scripts`

Canonical compositor blends live here:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates`

Temporary and working blends live here:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends`

Current output root is:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/outputs`

Current EXR inputs are still being read from the existing Edge Lab input root:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/inputs`

## Canonical Templates

Main merged compositor:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/edge_lab_final_template_safe_rebuild_20260405.blend`

Standalone workflow canonicals:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/compositor_ao.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/compositor_base.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/compositor_bioenvelope.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/compositor_normals.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/compositor_resources.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/compositor_shading.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/compositor_depth_outliner.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/compositor_mist.blend`

Helper canonicals:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/compositor_proposal_masks.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/proposal_only_layers.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/proposal_outline_layers.blend`

## Temp Blends

Development blends:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends/template_development`

Dataset or run-specific instantiations:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends/template_instantiations`

Checkpoints and `.blend1` backups:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends/checkpoints`

Scratch/debug blends:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends/scratch`

## Contract

The working contract is:

- canonical `.blend` files own compositor graph logic
- scripts own runtime actions around those blends

In practice:

- if the user asks for a visual or workflow change, edit the canonical blend
- if the user asks to run an existing workflow on different EXRs, use a thin runner
- if the task is a one-off inspection or probe, a one-off Blender command is acceptable
- do not rebuild an existing canonical workflow from factory startup just to render it

Scripts should:

- repath EXR inputs
- choose output folders
- trigger renders
- normalize `_0001` filenames

Scripts should not:

- become alternate owners of graph logic
- redefine workflows that already exist in the canonical blend

See:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/COMPOSITOR_TEMPLATE_CONTRACT.md`

## Agent Workflow

Use this decision rule when working in the compositor:

1. If the requested output already exists in a canonical blend:
   - open that canonical blend
   - repath the EXR inputs
   - set the output folder
   - render
   - do not save runtime changes back into the canonical blend

2. If the requested output does not exist yet, or the graph logic is wrong:
   - edit the canonical blend first
   - save a checkpoint if the change is substantial
   - then render from the updated canonical blend

3. If the task is exploratory:
   - use a working copy in `temp_blends/template_development` or `temp_blends/scratch`
   - only promote back into `canonical_templates` once the workflow is accepted

4. If a workflow is repeatable:
   - prefer a thin runner script in `scripts/`
   - the runner should only open the blend, repath EXRs, set outputs, render, and exit

5. Do not treat runner scripts as template builders:
   - if a canonical blend already exists, do not recreate it from scratch in the normal path

## Standard Masks

Current semantic mask names:

- `arboreal_positive_mask`
- `arboreal_priority_mask`
- `arboreal_trending_mask`
- `bioenvelope_positive_mask`
- `bioenvelope_trending_mask`

These should be exposed as named reroutes rather than hidden behind implementation-specific labels.

See:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/SCHEMA_FOR_ORGANISING_COMPOSITOR_GRAPHS.md`

## Current Workflow Split

Whole-suite work:

- use the merged main template
- repath the EXRs
- render only the requested workflow families

Workflow-specific work:

- use the standalone workflow blend
- repath only the EXRs that workflow needs
- render that workflow in isolation

Current notable cases:

- `depth_outliner`
  - standalone blend is single-input
  - one EXR input
  - one internal arboreal mask
  - one output `depth_outliner.png`

- `mist`
  - standalone blend is single-input
  - one EXR input
  - one semantic positive arboreal mask:
    - `arboreal_positive_mask`
  - three outputs:
    - `mist_kirsch_thin.png`
    - `mist_kirsch_fine.png`
    - `mist_kirsch_extra_thin.png`

- `shading`
  - standalone blend exists
  - remains multi-source because it genuinely needs multiple EXRs

## Current Mist Status

The standalone mist split is now structurally correct.

Important note:

- the earlier bad standalone mist result was caused by the wrong mask contract after splitting
- the standalone blend must use `arboreal_positive_mask`
- it must not fall back to a vague generic mask or the wrong `IndexOB` selection

Current fixed canonical standalone:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/canonical_templates/compositor_mist.blend`

Recent validated positive-only output:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/outputs/mist_single_positive_20260408_arboreal_mask/mist_kirsch_thin.png`
