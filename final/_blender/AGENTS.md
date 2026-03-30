# Blender Agent Guide

Refactor copy:
- [`_documentation-refactored/blender/AGENTS.md`](../../_documentation-refactored/blender/AGENTS.md)

## 0. Refactor Status

We are currently doing a limited compatibility-first refactor of the Blender-facing data endpoints.

- Current scope: the last-mile Blender bundle only, not the full upstream simulation pipeline.
- Current target: move the canonical Blender bundle to `_data-refactored/...` while preserving the legacy `data/...` tree during transition.
- Current helper names: `refactored_data_read_path(...)` and `refactored_data_write_path(...)`.
- Planned refactor areas: `_documentation-refactored/`, `_data-refactored/`, `_code-refactored/`.
- Current hook map: [`_documentation-refactored/blender/blender_output_hooks.md`](../../_documentation-refactored/blender/blender_output_hooks.md)

## 1. Project Context

This folder covers Blender scene building, rendering, and EXR/compositor handoff for the broader PyVista simulation project. It supports the Futures journal project. The manuscript is here:

- [`final/assesment/manuscript.md`](../assesment/manuscript.md)

Pathway naming follows the manuscript:

- `positive` = the nonhuman-led pathway
- `trending` = the human-led pathway
- the Blender render contract exposes `positive` as the view layer `pathway_state`

A state is a unique combination of:

| Component | Contract |
| --- | --- |
| `site` | `city`, `uni`, `trimmed-parade` |
| `scenario` | `positive`, `trending` |
| `year` | `0`, `10`, `30`, `60`, `180` |

Each site also has a separate baseline state.

Pipeline:

1. Upstream scripts produce the state inputs: per-state VTKs, one `nodeDF` CSV per state, shared site buildings/road data, habitat-feature PLY libraries, and baseline variants; see section 2.
2. For each site, the Blender scene imports the buildings and road PLYs as point clouds and runs [`final/_blender/2026/b2026_world_cubes.py`](./2026/b2026_world_cubes.py) to build renderable voxel cubes; see sections 2.3, 4, and 5.
3. For each state, [`final/_blender/2026/b2026_instancer.py`](./2026/b2026_instancer.py) reads the `nodeDF` CSV, imports the required habitat-feature PLYs, and builds the state collections at the CSV locations. In the city scene these are `Year_city_180_positive`, `Year_city_180_positive_priority`, and `Year_city_180_trending`; see sections 2.1, 2.2, 3.3, and 5.
4. After instancing, run [`final/_blender/2026/b2026_clipbox_setup.py`](./2026/b2026_clipbox_setup.py), then [`final/_blender/2026/b2026_camera_clipboxes.py`](./2026/b2026_camera_clipboxes.py). Do not rely on the instancer auto-run flags; see sections 4 and 5.
5. Heavy Blender scenes render multilayer EXRs by view layer, and lightweight compositor scenes consume those EXRs to produce final images; see sections 4 and 5.

TODO: baseline scene generation does not follow the standard city instancer contract exactly. [`final/_blender/2026/b2026_build_city_baseline.py`](./2026/b2026_build_city_baseline.py) overrides the instancer state and builds a reduced collection set. Document that baseline-specific path separately.

## 2. Inputs

### 2.1. State And Baseline Products

State generator:

- [`final/a_scenario_generateVTKs.py`](../a_scenario_generateVTKs.py)

Per state:

- raw state VTK: `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`
- instancer CSV: `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_nodeDF_{year}.csv`

Analysis-rich state VTK:

- `data/revised/final/output/{site}_{scenario}_1_scenarioYR{year}_urban_features_with_indicators.vtk`

The `nodeDF` file is the Blender instancing CSV. It contains all enabled `nodeType` rows for that state.

Current site-level `nodeType` patterns are:

| Site | Enabled `nodeType` values |
| --- | --- |
| `city` | `tree`, `log` |
| `trimmed-parade` | `tree` |
| `uni` | `tree`, `pole` |

`nodeDF` columns used by the Blender instancer include:

| CSV column | Meaning in Blender | Values / notes |
| --- | --- | --- |
| `x`, `y`, `z` | point location | world-space placement |
| `nodeType` | feature class | `tree`, `pole`, `log` |
| `size` | lifecycle / morphology class | `small`, `medium`, `large`, `senescing`, `snag`, `fallen` |
| `control` | management class | `street-tree`, `park-tree`, `reserve-tree`, `improved-tree` |
| `precolonial` | origin class | boolean-like, mapped to Blender int |
| `tree_id` | template selector | used to match the PLY library filename |
| `structureID` | source structure identifier | transferred as `structure_id` |
| `rotateZ` | rotation value | transferred as `rotation` |
| `useful_life_expectancy` | remaining lifetime | transferred as `life_expectancy` |
| `rewilded` | intervention state | mapped to `tree_interventions` |
| `action` | proposal state | mapped to `tree_proposals` |
| `Improvement` | improvement flag | mapped to `improvement` |
| `CanopyResistance` | canopy resistance value | mapped to `canopy_resistance` |
| `nodeID` | simulation node identifier | mapped to `node_id` |

Baseline generator:

- [`final/a_scenario_get_baselines.py`](../a_scenario_get_baselines.py)

Outputs include:

- `data/revised/final/baselines/{site}_baseline_trees.csv`
- `data/revised/final/baselines/{site}_baseline_resources_{voxel_size}.vtk`
- `data/revised/final/baselines/{site}_baseline_terrain_{voxel_size}.vtk`
- `data/revised/final/baselines/{site}_baseline_combined_{voxel_size}.vtk`

### 2.2. Tree / Log / Pole PLY Libraries

PLY export:

- [`final/_blender/a_vtk_to_ply.py`](./a_vtk_to_ply.py)

For tree and pole templates, the filename identity is:

- `precolonial`
- `size`
- `control`
- `tree_id`

Log templates use a separate log filename convention.

Arboreal PLYs preserve numeric point data. Key resource properties:

- `resource_hollow`
- `resource_epiphyte`
- `resource_dead branch`
- `resource_perch branch`
- `resource_peeling bark`
- `resource_fallen log`
- `resource_other`

These `resource_*` fields are binary per-vertex flags. `int_resource` is the older combined category. The binary fields are richer because a vertex can carry multiple resource flags.

If the source VTK already has the binary resource columns, the exporter preserves them. It only synthesises missing `resource_*` columns from `int_resource`.

Primary folders are:

- `data/revised/final/treeMeshesPly`
- `data/revised/final/logMeshesPly`

In the 2026 instancer, `treeMeshesPly` also contains the pole / artificial-support templates used by `nodeType == pole`.

### 2.3. Base World PLYs

Base world point clouds:

- buildings
- road / ground surface

Extractor:

- [`final/_blender/b_extract_scene.py`](./b_extract_scene.py)

It reads:

- `data/revised/{site}-siteVoxels-masked.vtk`
- `data/revised/{site}-roadVoxels-coloured.vtk`

And writes:

- `{site}_buildings.ply`
- `{site}_highResRoad.ply`

### 2.4. Envelope PLYs

Envelope models are the per-state design modifications: green roofs, living facades, depaved footprints, and related bioenvelope actions.

Envelope generator:

- [`final/_blender/b_generate_rewilded_envelopes.py`](./b_generate_rewilded_envelopes.py)

It reads:

- `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`

And writes:

- `data/revised/final/{site}/ply/{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}.ply`

Envelope PLY attributes:

- `scenario_bioEnvelope_int`
- `scenario_bioEnvelope_simple_int`
- `sim_Turns`
- `sim_averageResistance` when available

Also:

- [`final/_blender/b_generate_rewilded_ground.py`](./b_generate_rewilded_ground.py)

It writes:

- `{site}_{scenario}_{voxel_size}_ground_scenarioYR{year}.ply`

## 3. Scene Contract

### 3.1. View Layers

City view-layer contract:

- `pathway_state`
  - main `positive` pathway
- `existing_condition`
  - base world only
- `city_priority`
  - priority subset of the `positive` pathway
- `city_bioenvelope`
  - bioenvelope modifications
- `trending_state`
  - `trending` pathway

### 3.2. Object Pass Indices

City compositor contract:

| Pass index | Meaning | Current city objects |
| --- | --- | --- |
| `1` | world / road / ground receivers and cube derivatives | `city_highResRoad`, `city_highResRoad.001`, and `_cubes` variants; baseline ground derivatives also belong here |
| `2` | buildings / base built form | `city_buildings`, `city_buildings.001` |
| `3` | arboreal point clouds and instanced tree/log/pole systems | `TreePositions_*`, `LogPositions_*`, `PolePositions_*`, plus imported instance collections |
| `5` | envelope geometry | `city_envelope`, `Envelope_trending`, and related envelope imports |

### 3.3. Collection Hierarchy

Use this city hierarchy:

- `City_World`
  - base world point-cloud objects such as `city_buildings(.001)` and `city_highResRoad(.001)`
- `City_WorldCubes`
  - cube-derived versions of the world point clouds
- `City_Manager`
  - clip boxes and scene helpers
- `City_Camera`
  - active cameras
- `City_Camera_Archive`
  - archived / alternate cameras
- `Year_city_180_positive`
  - generated by [`final/_blender/2026/b2026_instancer.py`](./2026/b2026_instancer.py); main positive arboreal distribution
- `Year_city_180_positive_priority`
  - generated by [`final/_blender/2026/b2026_instancer.py`](./2026/b2026_instancer.py); positive priority subset
- `Year_city_180_trending`
  - generated by [`final/_blender/2026/b2026_instancer.py`](./2026/b2026_instancer.py); trending arboreal distribution
- `city_envelope`
  - positive envelope geometry
- `Envelope_trending`
  - trending envelope geometry

## 4. Workflow

Scene update order:

1. choose the target state: `site + scenario + year`
2. open the working heavy-scene blend
3. if the base world changed, refresh the base world PLYs and rerun [`final/_blender/2026/b2026_world_cubes.py`](./2026/b2026_world_cubes.py)
4. if the state VTK changed and the world objects need state-linked attributes such as `sim_Turns`, rerun [`final/_blender/2026/b2026_transfer_vtk_sim_layers.py`](./2026/b2026_transfer_vtk_sim_layers.py)
5. rerun [`final/_blender/2026/b2026_instancer.py`](./2026/b2026_instancer.py) on the state `nodeDF`
6. after every instancer run, rerun [`final/_blender/2026/b2026_clipbox_setup.py`](./2026/b2026_clipbox_setup.py); do not rely on the instancer auto-run flags
7. then rerun [`final/_blender/2026/b2026_camera_clipboxes.py`](./2026/b2026_camera_clipboxes.py)
8. if the material or AOV setup is missing in that blend, rerun the relevant setup scripts from section 5
9. if the EXR output branches are missing or stale, rerun [`final/_blender/2026/b2026_setup_view_layer_exr_outputs.py`](./2026/b2026_setup_view_layer_exr_outputs.py)
10. render multilayer EXRs from the heavy scene
11. open or build the lightweight EXR compositor scene and produce the final images

Baseline work follows the same pattern, but uses the baseline inputs and helpers from section 5.

## 5. Scripts, Files, And References

### 5.1. Working Blend Files

The main 2026 working blends are:

- `data/blender/2026/2026 futures heroes6.blend`
- `data/blender/2026/2026 futures heroes6_baseline.blend`

These are the active production blends. They are not yet the final reusable template.

### 5.2. Core Scripts By Role

Scene generation:

- [`final/_blender/2026/b2026_instancer.py`](./2026/b2026_instancer.py)
- [`final/_blender/2026/b2026_world_cubes.py`](./2026/b2026_world_cubes.py)
- [`final/_blender/2026/b2026_build_city_baseline.py`](./2026/b2026_build_city_baseline.py)

Clipping and camera sync:

- [`final/_blender/2026/b2026_clipbox_setup.py`](./2026/b2026_clipbox_setup.py)
- [`final/_blender/2026/b2026_camera_clipboxes.py`](./2026/b2026_camera_clipboxes.py)

Materials and AOV support:

- [`final/_blender/2026/MINIMAL_RESOURCES.py`](./2026/MINIMAL_RESOURCES.py)
- [`final/_blender/2026/b2026_patch_resource_binary_aovs.py`](./2026/b2026_patch_resource_binary_aovs.py)
- [`final/_blender/2026/b2026_city_envelope_aov_setup.py`](./2026/b2026_city_envelope_aov_setup.py)
- [`final/_blender/2026/b2026_transfer_vtk_sim_layers.py`](./2026/b2026_transfer_vtk_sim_layers.py)

`b2026_transfer_vtk_sim_layers.py`:

- reads the per-state VTK at `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk`
- transfers state-linked point data onto the base world point clouds
- current world targets are the base / buildings point cloud and the high-res road point cloud for the active site
- writes Blender point attributes:
  - `sim_Turns`
  - `sim_Nodes`
  - `scenario_bioEnvelope`
  - `scenario_bioEnvelopeSimple`
  - `sim_Matched`

EXR outputs and compositor builders:

- [`final/_blender/2026/b2026_setup_view_layer_exr_outputs.py`](./2026/b2026_setup_view_layer_exr_outputs.py)
- [`final/_blender/2026/b2026_build_city_exr_compositor.py`](./2026/b2026_build_city_exr_compositor.py)
- [`code/blender/2026/edge_detection_lab/build_city_exr_compositor_lightweight.py`](../../code/blender/2026/edge_detection_lab/build_city_exr_compositor_lightweight.py)

Baseline terrain conversion:

- [`final/_blender/2026/b2026_export_vtk_points_to_ascii_ply.py`](./2026/b2026_export_vtk_points_to_ascii_ply.py)

### 5.3. Embedded Text Blocks

Disk scripts are canonical. Update the disk script first, then sync the embedded copy in the blend you render from.

Embedded names that matter operationally:

- disk: `b2026_clipbox_setup.py`
  - embedded: `b2026_clipbox_setup`
- disk: `b2026_camera_clipboxes.py`
  - embedded: `camera_clipboxes_SS`

### 5.4. Further Info

- [`_documentation-refactored/DATA_PIPELINE_SCRIPT_SUMMARIES.md`](../../_documentation-refactored/DATA_PIPELINE_SCRIPT_SUMMARIES.md)
  - maps the Blender-facing inputs back through the upstream data pipeline, step by step, so you can see where each file comes from and where the branches diverge.

### 5.5. Reference Docs

For detailed enums, attribute inventories, and current notes:

- [`final/_blender/TEMPLATE_BLEND.md`](./TEMPLATE_BLEND.md)
- [`final/_blender/paraview-to-blender-info.md`](./paraview-to-blender-info.md)
- [`final/_blender/2026/KEY_SCRIPTS.md`](./2026/KEY_SCRIPTS.md)
- [`final/_blender/2026/pyvista-attributes-to-blender-info.md`](./2026/pyvista-attributes-to-blender-info.md)

### 5.6. TODO

- define what stays in the model Blender compositor and what moves to the lightweight compositor
- create an `AGENTS.md` for the edge-detection lab work and the compositor workflow
- move the edge-detection scripts out of `data` so they can be tracked in git
- consider renaming `Release-Control -> Brace-Feature` to `Reduce-Pruning` in stats/graph outputs, because the current measure is `park-tree` arboreal voxels rather than structural bracing
