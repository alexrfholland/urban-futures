# Edge Detection Lab

This folder is an isolated compositor-only Blender experiment for testing edge detection on urban-scale renders.

Reference docs:
- [WORKFLOW_REGISTRY.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/WORKFLOW_REGISTRY.md)
- [OUTPUT_SUITE_SPEC.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/OUTPUT_SUITE_SPEC.md)
- [ARBOREAL_RESOURCE_COLOURS.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/code/blender/2026/edge_detection_lab/ARBOREAL_RESOURCE_COLOURS.md)

Layout:
- `code/blender/2026/edge_detection_lab/` contains the runnable Blender Python script.
- `code/blender/2026/edge_detection_lab/render_depth_edge_variants.py` contains the practical local renderer for the current PNG workflow.
- `code/blender/2026/edge_detection_lab/render_exr_edge_variants.py` contains the EXR-based renderer for the `2026 futures heroes6-city` outputs.
- `data/blender/2026/edge_detection_lab/` contains the saved `.blend`, inputs, and outputs.

Canonical lightweight compositor pattern:
- `code/blender/2026/edge_detection_lab/build_city_exr_compositor_lightweight.py` builds the template compositor blend from the heavy scene.
- `code/blender/2026/edge_detection_lab/repath_city_exr_compositor_inputs.py` repaths that template to another EXR set.
- `data/blender/2026/edge_detection_lab/outputs/edge_lab_output_suite_baseline_20260329/blends/lightweight_classic_inputs.blend` is the prepared classic lightweight compositor for the current baseline EXRs.
- Camera choice stays in the heavy blend. `worldcam` vs `zoom3x` is not a separate compositor workflow.
- Use the repath script for baseline or alternate camera EXRs instead of creating another compositor-only blend builder.
- Canonical EXR filenames are:
  - `pathway_state.exr`
  - `priority.exr`
  - `existing_condition.exr`
  - optional: `bioenvelope.exr`
  - optional: `trending_state.exr`

Combined output-suite compositor:
- `code/blender/2026/edge_detection_lab/build_edge_lab_combined_compositor.py` builds one combined blend from the separate output-family blends.
- `data/blender/2026/edge_detection_lab/edge_lab_output_suite_combined.blend` is the current combined file.
- The combined file keeps one scene per output family:
  - `AO`
  - `Normals`
  - `Resources`
  - `MistOutlines`
  - `DepthOutliner`

Final template status:
- `data/blender/2026/edge_detection_lab/edge_lab_final_template.blend` is the current human-facing compositor file.
- It has two scenes:
  - `Current`
  - `Legacy`
- `Current` now contains one merged compositor scene with framed branches for:
  - AO
  - normals
  - resources
  - depth outliner
  - mist outlines
  - shading
  - base outputs
  - bioenvelopes
- `Legacy` keeps the classic lightweight compositor as reference.
- The current execution split is now:
  - `Current` for AO, normals, resources, depth outliner, shading, base outputs, and bioenvelopes
  - the final-template mist runner now also writes the canonical mist PNGs from EXRs, using the validated `kirschsizes` mist workflow on a temporary scratch scene during the final-template run
- The remaining cleanup is to decide whether the saved mist branch inside `Current` should be made canonical, or kept as a reference while the scratch-scene mist adapter remains the trusted path.

Verified now:
- `edge_lab_final_template.blend`, scene `Current`, works for:
  - AO
  - normals
  - resources
  - depth outliner
  - shading
  - base outputs
  - bioenvelopes
- the `Current` AO, normals, resources, and depth-outliner PNGs now match the older combined-suite outputs pixel-for-pixel on the latest city EXRs
- the final-template mist runner now writes mist outlines from EXRs and those PNGs match the older combined-suite outputs pixel-for-pixel on the latest city EXRs
- the current bioenvelope colours in `Current` now match the legacy palette outputs exactly
- the older base/world depth-window family is now wired into `Current` base outputs and follows the standalone tuned node flow for both:
  - `data/blender/2026/2026 futures heroes6-city/city-existing_condition.exr`
  - `data/blender/2026/edge_detection_lab/inputs/city_8k_network_20260330/city_existing_condition_8k.exr`
  - current output contract is `Standard` / `sRGB`; do not treat older `AgX`-authored PNGs as the target colour contract

Still to do:
- decide whether to rebuild the saved mist branch inside `Current` so it matches the validated mist runner exactly
- revisit mist once regenerated EXRs with updated mist/world settings are ready
  - target references: `pathway_mist_kirsch_extra_thin.png` and `priority_mist_kirsch_fine.png`
  - correct reference root: `data/blender/2026/edge_detection_lab/outputs/edge_lab_final_template_city_20260329/current/outlines_mist/`
- tidy `Current` once that mist-branch decision is settled
- decide whether the four `base_depth_windowed_*` variants should become part of the default PSD stack once the next composite round is approved

Latest files:
- human-facing compositor:
  - `data/blender/2026/edge_detection_lab/edge_lab_final_template.blend`
- old validated multi-scene runner blend:
  - `data/blender/2026/edge_detection_lab/edge_lab_output_suite_combined.blend`
- final-template driver:
  - `code/blender/2026/edge_detection_lab/run_edge_lab_final_template.py`
  - now renders `AO`, `normals`, `resources`, `depth outliner`, `mist outlines`, `shading`, `base`, and `bioenvelopes` in the final-template workflow
- current-suite driver:
  - `code/blender/2026/edge_detection_lab/run_edge_lab_combined_compositor.py`
  - still useful for legacy comparison
- current render-only cutover script for `Current` core families:
  - `code/blender/2026/edge_detection_lab/render_edge_lab_current_core_outputs.py`
- current mist adapter:
  - `code/blender/2026/edge_detection_lab/render_edge_lab_current_mist.py`
  - runs the validated `kirschsizes` mist workflow from EXRs on a temporary scratch scene
- latest validated current-template outputs:
  - `data/blender/2026/edge_detection_lab/outputs/edge_lab_final_template_city_20260329`
- latest validated combined-suite outputs:
  - `data/blender/2026/edge_detection_lab/outputs/edge_lab_output_suite_city_20260329`

What the scaffold does:
- creates a dedicated scene named `edge_detection_lab`
- sets the render size to `3840 x 2160`
- builds several edge variants for comparison:
  - `beauty_sobel_soft`
  - `depth_laplace_sparse`
  - `normal_kirsch_fine`
  - `mask_silhouette_band`
  - `hybrid_max_consensus`
- saves a reusable `.blend` file under `data/blender/2026/edge_detection_lab/`
- tolerates missing inputs so you can build the project first and drop PNGs in later

Current explicit PNG workflow:
- `base.png` = white render of the city base
- `test-world.png` = rough full-colour render
- `test-depth.png` = normalized depth pass with transparency for priority trees only
- the local renderer extracts tree shading from `test-world.png` using the alpha from `test-depth.png`
- it runs several depth-first edge variants, then composites both shading and edges over `base.png`
- outline colour is fixed to `#22123B`
- each detector now writes both `thin` and `regular` variable-width outline versions

EXR workflow:
- source EXRs live under `data/blender/2026/`
- main EXR channels used in the lab are `Image.*`, `Alpha.V`, `AO.*`, `Depth.V`, `IndexOB.V`, `Normal.*`, and, where available, `Mist.V`
- arboreal visibility masks are built from `IndexOB == 3`
- the positive visible-arboreal mask usually comes from `pathway_state`
- the visible priority mask is `priority(IndexOB == 3) * pathway visible-arboreal`
- the base layer uses `IndexOB == 1` and `IndexOB == 2`
- the bioenvelope layer uses `IndexOB == 5`
- outline colour is fixed to `#22123B`
- flat arboreal resource exports use `sRGB / Standard` so the palette lands on the intended hex values

Key passes we generate now:
- arboreal resource fills for `pathway` and `priority`
  - per-resource PNGs
  - combined coloured resource stack
- AO
  - `pathway_ao_masked.png`
  - `priority_ao_masked.png`
  - `existing_condition_ao_full.png`
- normals
  - `pathway_tree_normal.png`
  - `priority_tree_normal.png`
  - `existing_condition_normal_full.png`
- shading
  - `pathway_shading.png`
  - `priority_shading.png`
  - `existing_condition_shading.png`
  - these come from the existing compositor group `_AO SHADING.001`
  - `_AO SHADING.001` takes `AO`, `Normal`, and `Alpha`
  - inside the group: `AO -> Denoise`, `Normal -> same Denoise`, then `Color Ramp -> Overlay -> Set Alpha`
  - the final shading PNGs have the pathway / priority visible-arboreal masks applied
- base
  - `base_rgb.png`
  - `base_outlines.png`
  - `base_sim-turns.png`
  - `base_sim-nodes.png`
  - `base_sim-turns_ripple-effect.png`
- bioenvelopes
  - `base_bioenvelope_full-image.png`
  - `base_bioenvelope_exoskeleton.png`
  - `base_bioenvelope_brownroof.png`
  - `base_bioenvelope_otherground.png`
  - `base_bioenvelope_rewilded.png`
  - `base_bioenvelope_footprintdepaved.png`
  - `base_bioenvelope_livingfacade.png`
  - `base_bioenvelope_greenroof.png`
  - `bioenvelope_full-image.png`
  - `bioenvelope_exoskeleton.png`
  - `bioenvelope_brownroof.png`
  - `bioenvelope_otherground.png`
  - `bioenvelope_rewilded.png`
  - `bioenvelope_footprintdepaved.png`
  - `bioenvelope_livingfacade.png`
  - `bioenvelope_greenroof.png`
  - `trending_bioenvelope_full-image.png`
  - `trending_bioenvelope_exoskeleton.png`
  - `trending_bioenvelope_brownroof.png`
  - `trending_bioenvelope_otherground.png`
  - `trending_bioenvelope_rewilded.png`
  - `trending_bioenvelope_footprintdepaved.png`
  - `trending_bioenvelope_livingfacade.png`
  - `trending_bioenvelope_greenroof.png`
- mist-based arboreal outlines
  - `pathway_mist_kirsch_thin.png`
  - `pathway_mist_kirsch_fine.png`
  - `pathway_mist_kirsch_extra_thin.png`
  - `priority_mist_kirsch_thin.png`
  - `priority_mist_kirsch_fine.png`
  - `priority_mist_kirsch_extra_thin.png`
- depth-based arboreal outliner
  - `pathway_depth_outliner.png`
  - `priority_depth_outliner.png`
  - this matches the baseline compositor's `_OUTLINER.001` logic without the thicker focus-object overlay
  - core path is `Depth -> Normalize -> Kirsch -> hard threshold -> flat purple colour`, then masked back to visible arboreal alpha
- base/world depth-window variants
  - `base_depth_windowed_balanced_refined.png`
  - `base_depth_windowed_internal_refined.png`
  - `base_depth_windowed_internal_dense.png`
  - `base_depth_windowed_balanced_dense.png`

Current arboreal resource palette:
- `hollow` = `#ce6dd9`
- `epiphyte` = `#c5e28e`
- `dead` / `dead_branch` = `#ffcc01`
- `peeling` / `peeling_bark` = `#ff85be`
- `fallen` / `fallen_log` = `#8f89bf`
- `perch` / `perch_branch` = `#ffcb00`
- `other` / `none` = `#cecece`

Expected inputs:
- put PNGs in `inputs/`
- the script looks for files whose names match or contain:
  - `beauty`, `color`, `albedo`, `rgb`
  - `depth`, `z`, `distance`
  - `normal`, `normals`
  - `mask`, `alpha`, `matte`, `silhouette`

Headless run:

```bash
blender --background --factory-startup --python code/blender/2026/edge_detection_lab/headless_edge_detection_lab.py -- \
  --inputs data/blender/2026/edge_detection_lab/inputs \
  --outputs data/blender/2026/edge_detection_lab/outputs \
  --blend data/blender/2026/edge_detection_lab/edge_detection_lab.blend
```

Notes:
- The scaffold is compositing only. No 3D geometry is required.
- The current script saves the blend and wires the output nodes, but it does not force a render until you are ready to test with real PNG passes.
- If you want to compare another pass naming scheme, edit `INPUT_SPECS` in the Python script.

Practical render command for the current files:

```bash
uv run --with pillow --with numpy --with scipy \
  python code/blender/2026/edge_detection_lab/render_depth_edge_variants.py
```

Practical EXR render command:

```bash
uv run --with pillow --with numpy --with scipy --with OpenEXR --with Imath \
  python code/blender/2026/edge_detection_lab/render_exr_edge_variants.py
```

Outputs from the practical renderer:
- `tree_shading_extracted.png`
- one `*_edges.png` image per variant
- one `*_composite.png` image per variant
- `variant_contact_sheet.png`
- `variant_summary.json`

Outputs from the EXR renderer:
- `outputs/exr_city_heroes6/pathway_visible_tree_mask.png`
- one folder per scene preset:
  - `priority_positive/`
  - `trending_existing/`
- each scene folder contains:
  - a punched-out base stack PNG
  - a tree extract PNG
  - one `*_edges.png` image per detector/width combination
  - one `*_composite.png` image per detector/width combination
  - a contact sheet PNG
- `outputs/exr_city_heroes6/summary.json`

Typical current output bundles:
- `resources/`
  - per-resource flat-colour PNGs
  - one combined resource PNG for pathway
  - one combined resource PNG for priority
- `ao/`
  - masked pathway AO
  - masked priority AO
  - full existing-condition AO
- `normals/`
  - masked pathway normals
  - masked priority normals
  - full existing-condition normals
- `shading/`
  - masked pathway shading
  - masked priority shading
  - masked existing-condition shading
- `base/`
  - base beauty
  - base outlines
  - normalized `sim_Turns`
  - normalized `sim_nodes`
  - ripple-effect `sim_Turns`
- `bioenvelope/`
  - coloured base-world bioenvelope layers
  - coloured direct envelope EXR layers
  - coloured trending bioenvelope layers
- `outlines_mist/`
  - mist-derived Kirsch outline sizes for arboreals
- `depth_outliner/`
  - normalized visible-arboreal depth prep PNGs
  - final depth-derived outliner PNGs
