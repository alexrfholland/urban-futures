# Compositor

This folder is the current refactored home for the compositor-only EXR workflow.

## Structure

Code and docs live here:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor`

Current active scripts live here:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/scripts`

Canonical compositor blends live here:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates`

Temporary and working blends live here:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends`

Current output root is:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/outputs`

Current EXR inputs are still being read from the existing Edge Lab input root:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/inputs`

## Canonical Templates

Main merged compositor:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/edge_lab_final_template_safe_rebuild_20260405.blend`

Standalone workflow canonicals:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/compositor_ao.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/compositor_base.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/compositor_bioenvelope.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/compositor_normals.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/compositor_resources.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/compositor_shading.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/compositor_depth_outliner.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/compositor_mist.blend`

Helper canonicals:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/compositor_proposal_masks.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/proposal_only_layers.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/canonical_templates/proposal_outline_layers.blend`

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

Scripts should:

- repath EXR inputs
- choose output folders
- trigger renders
- normalize `_0001` filenames

Scripts should not:

- become alternate owners of graph logic
- redefine workflows that already exist in the canonical blend

See:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/COMPOSITOR_TEMPLATE_CONTRACT.md`

## Standard Masks

Current semantic mask names:

- `arboreal_positive_mask`
- `arboreal_priority_mask`
- `arboreal_trending_mask`
- `bioenvelope_positive_mask`
- `bioenvelope_trending_mask`

These should be exposed as named reroutes rather than hidden behind implementation-specific labels.

See:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/blender/compositor/SCHEMA_FOR_ORGANISING_COMPOSITOR_GRAPHS.md`

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
  - one internal arboreal mask
  - three outputs:
    - `mist_kirsch_thin.png`
    - `mist_kirsch_fine.png`
    - `mist_kirsch_extra_thin.png`

- `shading`
  - standalone blend exists
  - remains multi-source because it genuinely needs multiple EXRs

## Current Mist Issue

The mist workflow is structurally in the right place now, but the standalone result is still not good enough.

Current issue:

- the standalone mist result is weaker and sparser than the earlier acceptable main-template output

Likely causes:

- the quantize step
- overly aggressive Kirsch thresholds in the simplified standalone blend

Recent raw-pass reference:

- a plain black-and-white render of the positive `Mist` channel was written to:
  - `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/outputs/mist_channel_bw_city_timeline_positive_20260408/city_timeline_positive_mist_bw.png`
