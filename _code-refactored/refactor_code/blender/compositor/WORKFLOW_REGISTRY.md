# Workflow Registry

This lab has multiple experimental scripts. Reuse should reference a named workflow first, then the adapter.

For the canonical output contract, use:
- [OUTPUT_SUITE_SPEC.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/OUTPUT_SUITE_SPEC.md)

## `city_exr_compositor_template_v1`

- Canonical script: `code/blender/2026/edge_detection_lab/build_city_exr_compositor_lightweight.py`
- Purpose: build the canonical city lightweight compositor from the heavy Blender scene, replacing live Render Layers nodes with multilayer EXR image nodes while keeping the compositor graph intact.

## `city_exr_compositor_repath_v1`

- Canonical script: `code/blender/2026/edge_detection_lab/repath_city_exr_compositor_inputs.py`
- Reuses workflow: `city_exr_compositor_template_v1`
- Additional adaptation: repaths the existing `EXR :: <layer>` image nodes to another EXR input set, such as baseline or another heavy-scene camera render, without creating a separate compositor workflow.
- Canonical EXR filenames are `pathway_state.exr`, `priority.exr`, `existing_condition.exr`, and optional `bioenvelope.exr`, `trending_state.exr`.
- Note: zoom3x vs worldcam is not a compositor distinction. It is just a different EXR set from the heavy blend.

## `png_depth_tuned_v1`

- Canonical script: `code/blender/2026/edge_detection_lab/render_depth_edge_variants.py`
- Purpose: fine arboreal depth edges using the tuned detector set and `thin` / `regular` width shaping.
- Inputs:
  - `base.png`
  - `world.png`
  - `depth.png` where RGB stores normalized depth and alpha stores the arboreal mask
- Current adapter for updated EXRs:
  - `code/blender/2026/edge_detection_lab/run_exr_0000_depth_workflow.py`

## `blender_exr_depth_v4`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_depth_edges_pipeline_v4_blender.py`
- Purpose: historical Blender-based EXR depth-edge comparison.
- Status: comparison / legacy, not the canonical fine-edge workflow.

## `blender_exr_arboreal_mist_v1`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_variants_v1_blender.py`
- Reuses pattern from: `code/blender/2026/edge_detection_lab/render_exr_base_mist_bestpractice_v5_blender.py`
- Purpose: headless Blender compositor mist-edge variants for arboreal pathway / priority / trending layers using the existing arboreal mask logic.

## `blender_exr_arboreal_mist_v2`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_variants_v2_blender.py`
- Reuses workflow: `blender_exr_arboreal_mist_v1`
- Additional adaptation: a bottom-weighted screen-space gain mask before thresholding to reduce vertical screen-space fade in mist-derived edges.
- Purpose: headless Blender compositor mist-edge variants for arboreal pathway / priority / trending layers with bottom-of-frame edge compensation.

## `blender_exr_arboreal_mist_v2_extrathin`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_variants_v2_extrathin_blender.py`
- Reuses workflow: `blender_exr_arboreal_mist_v2`
- Additional adaptation: higher thresholds and narrower width shaping for a finer extra-thin arboreal line.
- Purpose: headless Blender compositor mist-edge outputs for pathway / priority / trending when the current thin screen-lift result is still too heavy.

## `blender_exr_arboreal_mist_kirschsizes_v1`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_kirsch_sizes_blender.py`
- Reuses workflow: `blender_exr_arboreal_mist_v2`
- Additional adaptation: edge-only export of three Kirsch widths, `thin`, `fine`, and `extra_thin`, with flat filenames like `pathway_mist_kirsch_fine.png`.
- Purpose: generate just the arboreal Kirsch line PNGs without composites or retained prep outputs.

## `blender_exr_arboreal_mist_v3`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_variants_v3_blender.py`
- Reuses workflow: `blender_exr_arboreal_mist_v2`
- Additional adaptation: local mist normalization in the compositor before edge detection, plus one mild hybrid screen-lift variant for comparison.
- Purpose: headless Blender compositor mist-edge variants that try to remove the low-frequency mist gradient before thresholding.

## `blender_exr_arboreal_mist_v4`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_arboreal_mist_variants_v4_blender.py`
- Reuses workflow: `blender_exr_arboreal_mist_v2`
- Additional adaptation: local normalization of the filtered edge-response field before width shaping, plus one mild hybrid screen-lift variant for comparison.
- Purpose: headless Blender compositor mist-edge variants that target uneven lineweight directly in the detected edge signal instead of the raw mist pass.

## `city_mist_tune_and_regen_v1`

- Source script: `code/blender/2026/edge_detection_lab/tune_city_mist_and_render_exrs_v1_blender.py`
- Reuses setup script: `final/_blender/2026/b2026_setup_view_layer_exr_outputs.py`
- Purpose: save a separate tuned city blend, apply revised world mist settings, and regenerate per-view-layer EXRs headlessly without overwriting the current source blend.

## `blender_exr_ao_v2`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_ao_v2_blender.py`
- Purpose: masked AO extraction from EXRs.

## `blender_exr_lightweight_shading_v1`

- Source script: `code/blender/2026/edge_detection_lab/add_named_masks_and_render_baseline_lightweight_ao.py`
- Reuses existing compositor group: `_AO SHADING.001` from the lightweight compositor blend
- Additional adaptation: renames the group output to `shading`, builds named arboreal masks, and exports masked `pathway_shading` and `priority_shading` PNGs plus the full base AO pass.
- Purpose: reuse the inherited city compositor shading treatment for arboreal pathway / priority layers instead of rebuilding a separate shading look in the lab.

## `blender_exr_lightweight_shading_zoom3x_v1`

- Source script: `code/blender/2026/edge_detection_lab/render_zoom3x_lightweight_shading.py`
- Reuses workflow: `blender_exr_lightweight_shading_v1`
- Additional adaptation: fixed-target zoom3x exporter that repaths the lightweight baseline compositor to the `city_zoom3x_8k` EXRs, applies the visible arboreal masks, and writes only the two masked shading PNGs at `7680 x 4320`.
- Purpose: one-off lightweight compositor shading export for the baseline zoom3x 8K deliverable set.
- Status: special-case adapter, not a separate canonical compositor workflow.

## `blender_exr_normals_v2`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_normals_v2_blender.py`
- Purpose: masked normals extraction from EXRs.

## `blender_exr_normals_xyz_all_layers_v1`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_normals_xyz_all_layers_blender.py`
- Reuses workflow: `blender_exr_normals_v2`
- Additional adaptation: exports separate remapped `Normal.X`, `Normal.Y`, and `Normal.Z` grayscale PNGs for every current view-layer EXR instead of masked arboreal RGB normals.
- Purpose: headless Blender compositor normals-channel export for pathway, priority, trending, existing condition, and bioenvelope layers.

## `blender_exr_arboreal_resource_fills_v1`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_arboreal_resource_fills_v1_blender.py`
- Reuses workflow: `blender_exr_normals_v2`
- Additional adaptation: applies the established pathway / priority / trending arboreal visibility masks to the explicit EXR resource mask channels, colors each resource fill using the adjusted palette, and exports both per-resource PNGs and a combined stack with `hollow` on top.
- Palette note: `code/blender/2026/edge_detection_lab/ARBOREAL_RESOURCE_COLOURS.md`
- Purpose: headless Blender compositor export of arboreal resource fill layers for pathway, priority, and trending.

## `blender_exr_arboreal_depth_outliner_v1`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_arboreal_depth_outliner_baseline_blender.py`
- Reuses existing compositor logic from: `_OUTLINER.001` in the baseline city compositor
- Additional adaptation: keeps only the core depth-outline branch, `Depth -> Normalize -> Kirsch -> hard threshold -> flat purple colour`, and reapplies the visible arboreal mask to the output alpha.
- Purpose: reproduce the baseline compositor's internal branch-detail linework from normalized depth without the thicker `TRIM_THICK_FOCUS_OBJECT_OUTLINE.001` overlay.

## `blender_priority_resource_outline_normals_v1`

- Source script: `code/blender/2026/edge_detection_lab/render_priority_resource_outline_normals_v1_blender.py`
- Reuses workflows: `blender_exr_arboreal_resource_fills_v1`, `blender_exr_arboreal_mist_kirschsizes_v1`, `blender_exr_normals_xyz_all_layers_v1`
- Additional adaptation: uses the existing `priority_resource_combined.png`, the mid-strength `priority_mist_kirsch_fine.png` outline, and the existing `priority_normal_x/y/z.png` channels to build stronger per-axis `x`, `y`, and `z` shading branches plus outlined comparison composites.
- Purpose: headless Blender compositor presentation pass for the priority resource layer with mid-strength outline and separate normal-axis depth variants.

## `blender_exr_base_lines_v1`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_base_lines_v1_blender.py`
- Reuses workflow ideas from: `blender_exr_base_mist_bestpractice_v5`, `blender_exr_normals_xyz_all_layers_v1`
- Additional adaptation: uses the base-only EXR, splits `ground` and `buildings` by `IndexOB`, drives the main linework from `Depth`, and adds a restrained `Normal` assist for building creases only in the hybrid variants.
- Purpose: headless Blender compositor export of transparent purple base-line PNGs for Photoshop compositing.

## `blender_exr_base_lines_v2_depthpost`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_base_lines_v2_depthpost_blender.py`
- Reuses workflow: `blender_exr_base_lines_v1`
- Additional adaptation: stays fully post-compositing, removes the normal-assist branch, reshapes the base `Depth` field harder before edge detection with tighter depth-contrast ramps, and exports only stronger depth-driven Kirsch/Sobel base-line variants.
- Purpose: headless Blender compositor export of more assertive base-only linework for Photoshop compositing when the first base-line pass is too flat or too noisy.

## `blender_exr_base_lines_v3_depthwindows`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_base_lines_v3_depthwindows_blender.py`
- Reuses workflow: `blender_exr_base_lines_v2_depthpost`
- Additional adaptation: applies post-only fixed depth windows instead of full-frame normalization, runs multiple building depth windows in parallel, and combines them with `Max` so inter-building separation survives across near, mid, and farther depth bands.
- Purpose: headless Blender compositor export of stronger internal base scene linework when a single depth-contrast pass is not giving enough lines between buildings at different depths.

## `blender_exr_base_lines_v4_tuned`

- Source script: `code/blender/2026/edge_detection_lab/render_exr_base_lines_v4_tuned_blender.py`
- Reuses workflow: `blender_exr_base_lines_v3_depthwindows`
- Additional adaptation: tunes only around the useful `balanced` and `internal` multi-window depth setups with slightly lower building thresholds, tighter near/mid windows, and restrained ground thresholds so internal separation comes up without reverting to the noisy full-scene hybrid look.
- Purpose: headless Blender compositor export of tuned base-only line variants for Photoshop compositing after reviewing the first multi-window results.

## `proposal_colored_depth_outlines_v1`

- Canonical template: `_code-refactored/refactor_code/blender/compositor/canonical_templates/proposal_colored_depth_outlines.blend`
- Canonical runner: `_code-refactored/refactor_code/blender/compositor/scripts/render_current_proposal_colored_depth_outlines.py`
- Purpose: headless Blender compositor export of proposal-only colored depth-outline PNGs from the proposal EXR channels.
- Runtime note: on Blender 4.2 the runner rebuilds the saved `ProposalColoredDepthOutput` node in memory and may add a transient `Composite` sink so the saved canonical graph executes reliably without saving runtime scaffolding back into the template.
