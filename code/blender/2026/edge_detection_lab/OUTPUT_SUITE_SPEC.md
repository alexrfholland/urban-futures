# Edge-Lab Output Suite Spec

This file defines the canonical edge-lab output suite. It is narrower than the
full workflow registry. The goal is to say which outputs matter, which script
is canonical for each one, and which existing scripts are only legacy or
special-case adapters.

## 1. Output Contract

The edge-lab suite produces post-render PNG products from multilayer EXRs. The
heavy Blender scene renders the EXRs. The edge-lab workflows consume them.

Canonical EXR inputs:

- `pathway_state.exr`
- `priority.exr`
- `existing_condition.exr`
- optional `bioenvelope.exr`
- optional `trending_state.exr`

The canonical output families are:

1. resource fills
2. AO
3. normals
4. shading
5. mist-based arboreal outlines
6. depth-based arboreal outliners

## 2. Canonical Output Families

### 2.1. Resource Fills

Required products:

- per-resource PNGs for `pathway`
- per-resource PNGs for `priority`
- combined coloured resource PNG for `pathway`
- combined coloured resource PNG for `priority`

Canonical workflow:

- `blender_exr_arboreal_resource_fills_v1`

Canonical script:

- [render_exr_arboreal_resource_fills_v1_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_arboreal_resource_fills_v1_blender.py)

Current status:

- active
- now present in `edge_lab_final_template.blend`, scene `Current`
- now rendered directly from `Current`
- current-vs-combined validation on the latest city EXRs is pixel-identical

Legacy / special-case adapters:

- [render_exr_arboreal_resource_fills_baseline_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_arboreal_resource_fills_baseline_blender.py)

### 2.2. AO

Required products:

- `pathway_ao_masked.png`
- `priority_ao_masked.png`
- `existing_condition_ao_full.png`

Canonical workflow:

- `blender_exr_ao_v2`

Canonical script:

- [render_exr_ao_v2_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_ao_v2_blender.py)

Current status:

- active
- now present in `edge_lab_final_template.blend`, scene `Current`
- now rendered directly from `Current`
- current-vs-combined validation on the latest city EXRs is pixel-identical

Legacy / special-case adapters:

- [render_exr_ao_baseline_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_ao_baseline_blender.py)

### 2.3. Normals

Required products:

- `pathway_tree_normal.png`
- `priority_tree_normal.png`
- `existing_condition_normal_full.png`

Canonical workflow:

- `blender_exr_normals_v2`

Canonical script:

- [render_exr_normals_v2_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_normals_v2_blender.py)

Current status:

- active
- now present in `edge_lab_final_template.blend`, scene `Current`
- now rendered directly from `Current`
- current-vs-combined validation on the latest city EXRs is pixel-identical

Secondary but still useful:

- [render_exr_normals_xyz_all_layers_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_normals_xyz_all_layers_blender.py)
  - channel diagnostics, not the main output contract

Legacy / special-case adapters:

- [render_exr_normals_baseline_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_normals_baseline_blender.py)

### 2.4. Shading

Required products:

- `pathway_shading.png`
- `priority_shading.png`
- `existing_condition_shading.png`

Canonical workflow:

- `blender_exr_lightweight_shading_v1`

Canonical script:

- [add_named_masks_and_render_baseline_lightweight_ao.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/add_named_masks_and_render_baseline_lightweight_ao.py)

Dependencies:

- lightweight compositor template
- lightweight compositor repath adapter
- inherited compositor node group `_AO SHADING.001`

Current status:

- active
- now present in `edge_lab_final_template.blend`, scene `Current`
- now rendered directly from `Current`
- current-vs-combined validation on the latest city EXRs is pixel-identical

Legacy / special-case adapters:

- [render_zoom3x_lightweight_shading.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_zoom3x_lightweight_shading.py)

### 2.5. Mist-Based Arboreal Outlines

Required products:

- `pathway_mist_kirsch_thin.png`
- `pathway_mist_kirsch_fine.png`
- `pathway_mist_kirsch_extra_thin.png`
- `priority_mist_kirsch_thin.png`
- `priority_mist_kirsch_fine.png`
- `priority_mist_kirsch_extra_thin.png`

Canonical workflow:

- `blender_exr_arboreal_mist_kirschsizes_v1`

Canonical script:

- [render_exr_arboreal_mist_kirsch_sizes_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_kirsch_sizes_blender.py)

Current status:

- active
- now rendered in the final-template workflow from EXRs
- current-vs-combined validation on the latest city EXRs is pixel-identical
- the trusted path is [render_edge_lab_current_mist.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_edge_lab_current_mist.py), which runs the validated `kirschsizes` mist workflow on a temporary scratch scene during the final-template run
- the saved mist branch inside `edge_lab_final_template.blend`, scene `Current`, is not yet the trusted source of truth
- pause further mist tuning until regenerated EXRs with updated mist/world settings are available
- when that retest happens, use these as the visual references:
  - `pathway_mist_kirsch_extra_thin.png`
  - `priority_mist_kirsch_fine.png`
  - reference root: `data/blender/2026/edge_detection_lab/outputs/edge_lab_final_template_city_20260329/current/outlines_mist/`

Legacy / exploratory variants:

- [render_exr_arboreal_mist_variants_v1_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_variants_v1_blender.py)
- [render_exr_arboreal_mist_variants_v2_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_variants_v2_blender.py)
- [render_exr_arboreal_mist_variants_v2_extrathin_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_variants_v2_extrathin_blender.py)
- [render_exr_arboreal_mist_variants_v3_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_variants_v3_blender.py)
- [render_exr_arboreal_mist_variants_v4_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_variants_v4_blender.py)
- [render_exr_arboreal_mist_kirsch_sizes_baseline_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_kirsch_sizes_baseline_blender.py)

### 2.6. Depth-Based Arboreal Outliners

Required products:

- `pathway_depth_outliner.png`
- `priority_depth_outliner.png`

Canonical workflow:

- `blender_exr_arboreal_depth_outliner_v1`

Canonical script:

- [render_exr_arboreal_depth_outliner_baseline_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_arboreal_depth_outliner_baseline_blender.py)

Current status:

- active

Legacy / related depth-edge workflows:

- [render_exr_depth_edges_pipeline_v4_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_depth_edges_pipeline_v4_blender.py)
- [render_exr_depth_edges_three_v3_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_depth_edges_three_v3_blender.py)
- [render_exr_trending_depth_edges_v2_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_trending_depth_edges_v2_blender.py)

## 3. Current Consolidation Status

Consolidated now:

- compositor template builder
- compositor EXR repathing
- canonical EXR filename contract
- canonical output-family map
- combined output-suite blend builder
- final template blend with `Current` and `Legacy`

Current combined blend:

- [edge_lab_output_suite_combined.blend](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/edge_lab_output_suite_combined.blend)

Current combined builder:

- [build_edge_lab_combined_compositor.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/build_edge_lab_combined_compositor.py)

The combined blend keeps one scene per output family:

- `AO`
- `Normals`
- `Resources`
- `MistOutlines`
- `DepthOutliner`

Current final-template blend:

- [edge_lab_final_template.blend](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/edge_lab_final_template.blend)

Current final-template scenes:

- `Current`
- `Legacy`

What `Current` already contains:

- AO
- normals
- resources
- depth outliner
- mist outlines
- shading
- base outputs
- bioenvelopes

What is already rendered from `Current`:

- AO
- normals
- resources
- depth outliner
- mist outlines
- shading
- base outputs
- bioenvelopes

What has been verified:

- `edge_lab_final_template.blend` plus the final-template runners writes working AO, normals, resources, depth outliner, mist, shading, base, and bioenvelope outputs
- the `Current` AO, normals, resources, and depth-outliner PNGs are pixel-identical to the older combined-suite outputs on the latest city EXRs
- the final-template mist PNGs are pixel-identical to the older combined-suite mist outputs on the latest city EXRs
- the `Current` bioenvelope full-image outputs now match the old lightweight compositor outputs exactly

Not yet consolidated:

- one trusted saved-scene mist branch inside `Current`

Legacy lightweight compositor content still kept for reference:

- final composite image outputs
- classic outline branches
- world/base layer palette branches
- trending resource exports

Already moved into `Current`:

- shading branches
- base image outputs
- bioenvelope exports

Base-depth variant note:

- older standalone base/world depth-window variants are still available for reference in [render_exr_base_lines_v4_tuned_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_base_lines_v4_tuned_blender.py)
- useful historical outputs include `base_depth_windowed_balanced_refined`, `base_depth_windowed_internal_refined`, `base_depth_windowed_internal_dense`, and `base_depth_windowed_balanced_dense`
- these four variants are now revived inside `Current` base outputs
- they have been verified against the standalone tuned workflow for both:
  - `data/blender/2026/2026 futures heroes6-city/city-existing_condition.exr`
  - `data/blender/2026/edge_detection_lab/inputs/city_8k_network_20260330/city_existing_condition_8k.exr`
- active render contract is `Standard` / `sRGB`; older `AgX`-authored reference PNGs should not be treated as the target colour contract

Latest working files:

- human-facing compositor:
  - [edge_lab_final_template.blend](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/edge_lab_final_template.blend)
- older validated execution blend:
  - [edge_lab_output_suite_combined.blend](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/edge_lab_output_suite_combined.blend)
- final-template driver:
  - [run_edge_lab_final_template.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/run_edge_lab_final_template.py)
- final-template mist adapter:
  - [render_edge_lab_current_mist.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_edge_lab_current_mist.py)
- current render-only cutover script for the non-mist families:
  - [render_edge_lab_current_core_outputs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_edge_lab_current_core_outputs.py)
- older validated runner:
  - [run_edge_lab_combined_compositor.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/run_edge_lab_combined_compositor.py)
- latest validated final-template output root:
  - [edge_lab_final_template_city_20260329](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/outputs/edge_lab_final_template_city_20260329)
- latest validated combined-suite output root:
  - [edge_lab_output_suite_city_20260329](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/outputs/edge_lab_output_suite_city_20260329)

## 4. Near-Term Implementation Target

The next canonical orchestration layer should do this:

1. attach a chosen EXR set to `edge_lab_final_template.blend`
2. render AO from `Current`
3. render normals from `Current`
4. render resource fills from `Current`
5. render shading from `Current`
6. render mist outlines in the final-template workflow from EXRs
7. render depth outliners from `Current`
8. render base outputs from `Current`
9. render bioenvelopes from `Current`
10. write one dated output bundle with stable subfolders

That orchestration layer should treat:

- camera choice as part of the heavy-scene EXR set
- baseline as an EXR input variant
- zoom level as an EXR input variant

It should not create separate compositor workflows for those.
