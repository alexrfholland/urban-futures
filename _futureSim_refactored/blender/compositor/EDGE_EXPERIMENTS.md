# Edge Experiments

This note records where the stepped mist/depth edge experiments live and how
they should be used under the compositor contract.

## Status

These scripts are:

- experimental
- repeatable
- intentionally preserved

These scripts are not:

- canonical workflow runners
- the source of truth for accepted compositor graph logic

If an experiment is accepted, promote the logic into a canonical blend and
render from that blend with a thin runner.

## Script Home

Repeatable edge experiments now live here:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/scripts/experimental_edges`

Current scripts:

- `experiment_boost_kirsch_v3.py`
- `experiment_boost_threshold_v4.py`
- `experiment_depth_outliner_v1.py`
- `experiment_mask_before_kirsch_v2.py`
- `experiment_mist_bands_v9.py`
- `experiment_mist_highthreshold_v7.py`
- `experiment_mist_highthreshold_v7b.py`
- `experiment_mist_kirschsizes_v5.py`
- `experiment_mist_outliner_v1.py`
- `experiment_mist_preblur_v8.py`
- `experiment_mist_push_v6.py`

## Blend Home

Development blends for this exploration live here:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends/template_development`

Useful current examples:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends/template_development/compositor_mist_tuning_20260408.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends/template_development/compositor_mist_tuning_maskfix_20260408.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends/template_development/mist_pathway_kirsch_simple.blend`
- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/temp_blends/template_development/mist_pathway_kirsch_variants.blend`

## Output Home

Experiment outputs live under:

- `/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/compositor/outputs`

Important existing output families include:

- `experiment_mist_push_*`
- `experiment_mist_bands_*`
- `experiment_mist_threshold_high_*`
- `experiment_mist_threshold_only_*`
- `experiment_mist_kirschsizes_*`
- `experiment_mask_before_kirsch_*`
- `experiment_boost_kirsch_*`
- `experiment_boost_threshold_*`
- `experiment_depth_canopy_*`

## Operational Rule

Use these when the task is:

- compare edge styles
- sweep thresholds
- sweep blur or preblur
- test banding or posterization before edge detection
- inspect how mist and depth behave on a new EXR family

Do not use these as the normal rendering path when:

- a canonical blend already exists for the accepted workflow

In that case:

- edit the canonical blend if the accepted logic needs to change
- otherwise run the canonical blend with a thin runner
