# Edge Detection Lab

This folder is an isolated compositor-only Blender experiment for testing edge detection on urban-scale renders.

Layout:
- `code/blender/2026/edge_detection_lab/` contains the runnable Blender Python script.
- `code/blender/2026/edge_detection_lab/render_depth_edge_variants.py` contains the practical local renderer for the current PNG workflow.
- `code/blender/2026/edge_detection_lab/render_exr_edge_variants.py` contains the EXR-based renderer for the `2026 futures heroes6-city` outputs.
- `data/blender/2026/edge_detection_lab/` contains the saved `.blend`, inputs, and outputs.

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
  - these come from the existing compositor group `_AO SHADING.001`
  - `_AO SHADING.001` takes `AO`, `Normal`, and `Alpha`
  - inside the group: `AO -> Denoise`, `Normal -> same Denoise`, then `Color Ramp -> Overlay -> Set Alpha`
  - the final shading PNGs have the pathway / priority visible-arboreal masks applied
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
- `outlines_mist/`
  - mist-derived Kirsch outline sizes for arboreals
- `depth_outliner/`
  - normalized visible-arboreal depth prep PNGs
  - final depth-derived outliner PNGs
