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
- source EXRs live in `data/blender/2026/2026 futures heroes6-city`
- the EXR renderer reads `Image.*`, `Alpha.V`, `Depth.V`, `IndexOB.V`, and `Normal.*`
- `Mist.V` is only present in `city-pathway_state.exr`, so the EXR pipeline does not depend on Mist for the main path
- the positive visible-tree mask is built from `city-pathway_state.exr` using `IndexOB == 3`
- that visible-tree mask is used to punch trees out of the non-tree layers before stacking, so building and envelope pixels do not show through tree antialiasing
- `city-city_bioenvelope.exr` contains base-city pixels as well as envelope pixels, so the EXR renderer isolates `IndexOB == 5` from that file instead of trusting the file name alone
- `city-trending_state.exr` also contains base-city `IndexOB` values, so the EXR renderer isolates `IndexOB == 3` for the tree contribution there
- outline colour is fixed to `#22123B`
- each detector writes both `thin` and `regular` variable-width outline versions

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
