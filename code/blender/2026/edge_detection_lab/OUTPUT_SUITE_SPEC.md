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

Secondary but still useful:

- [render_exr_normals_xyz_all_layers_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_normals_xyz_all_layers_blender.py)
  - channel diagnostics, not the main output contract

Legacy / special-case adapters:

- [render_exr_normals_baseline_blender.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/render_exr_normals_baseline_blender.py)

### 2.4. Shading

Required products:

- `pathway_shading.png`
- `priority_shading.png`

Canonical workflow:

- `blender_exr_lightweight_shading_v1`

Canonical script:

- [add_named_masks_and_render_baseline_lightweight_ao.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/add_named_masks_and_render_baseline_lightweight_ao.py)

Dependencies:

- lightweight compositor template
- lightweight compositor repath adapter
- inherited compositor node group `_AO SHADING.001`

Current status:

- active, but still coupled to the lightweight compositor graph rather than a
  self-contained edge-lab builder

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

Not yet consolidated:

- the classic lightweight compositor branches that still sit outside the output suite
- one single-scene compositor graph for all output families

Classic lightweight compositor branches not yet moved into the combined suite:

- final composite image outputs
- base image outputs
- shading branches
- classic outline branches
- bioenvelope exports
- world/base layer palette branches
- trending resource exports

## 4. Near-Term Implementation Target

The next canonical orchestration layer should do this:

1. attach a chosen EXR set to the lightweight compositor template
2. run resource fills
3. run AO
4. run normals
5. run shading
6. run mist outlines
7. run depth outliners
8. write one dated output bundle with stable subfolders

That orchestration layer should treat:

- camera choice as part of the heavy-scene EXR set
- baseline as an EXR input variant
- zoom level as an EXR input variant

It should not create separate compositor workflows for those.
