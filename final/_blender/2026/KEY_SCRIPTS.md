# 2026 Blender Key Scripts

## Instancer
- Script: [b2026_instancer.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/_blender/2026/b2026_instancer.py)
- Purpose:
  - reads the city/parade node CSV
  - filters to the active area
  - imports tree/log/pole source meshes
  - creates the point-cloud objects and geometry-node instance systems
  - builds the available scenario collections for the target scene
  - configures the city scenario view-layer visibility after instancing
  - writes the run log
- Important notes:
  - infers `SITE` from the target scene when `AUTO_SITE_FROM_SCENE = True`
  - when both CSVs exist, it builds both `positive` and `trending`
  - for city scenes, the three scenario view layers are:
    - `pathway_state`: shows the main `positive` collection
    - `city_priority`: shows the `positive` priority collection
    - `trending_state`: shows the main `trending` collection
  - the priority branch is city-only and only built for the `positive` scenario when the `city_priority` view layer exists
  - imports `resource_*` binary attributes from tree PLYs, with fallback from `int_resource` when a PLY is missing them
  - it has optional follow-up hooks for `b2026_clipbox_setup.py` and `b2026_camera_clipboxes.py`, but those only run when `AUTO_RUN_CLIPBOX_SETUP` and/or `AUTO_RUN_CAMERA_CLIPBOXES` are enabled

## Clipbox Setup
- Script: [b2026_clipbox_setup.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/_blender/2026/b2026_clipbox_setup.py)
- Purpose:
  - creates or updates the live scene clip boxes:
    - `City_ClipBox`
    - `ClipBox`
  - rebuilds the clip geometry-node groups
  - patches generated world/tree node groups so they use the live clip box
- Important notes:
  - this is the script that must run after instancing, because new `tree_city_*` / `log_city_*` groups are recreated on each instancer run
  - it is not view-layer specific; it updates the live clip objects and node groups used by the scene

## Camera Clipboxes
- Script: [b2026_camera_clipboxes.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/_blender/2026/b2026_camera_clipboxes.py)
- Purpose:
  - ensures each camera has a proxy clip box
  - stores the proxy link on the camera
  - syncs the active camera's proxy transform onto the live clip box
  - registers handlers so camera changes keep the live clip box updated
- Important notes:
  - there are many per-camera proxy boxes, but only one live clip box per scene
  - Geometry Nodes use the live clip box, not the proxies directly
  - like clipbox setup, this script is view-layer agnostic; it manages camera-to-live-clip syncing for the whole scene

## Scenario View Layers
- `pathway_state`
  - main positive scenario collection
- `city_priority`
  - positive priority collection only
- `trending_state`
  - main trending scenario collection

## Order Of Operations
1. Run the instancer.
2. The instancer builds the available scenarios and configures the city view layers: `pathway_state`, `city_priority`, `trending_state`.
3. If the auto-run flags are enabled, the instancer then runs clipbox setup and camera clipboxes automatically.
4. If the auto-run flags are disabled, run clipbox setup manually after instancing, then run camera clipboxes.

If you are using embedded Blender text blocks instead of the external files, keep the embedded copies in sync with these scripts.

## PLY Audit
- Script: [check_ply.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/_blender/2026/check_ply.py)
- Purpose:
  - scans the tree PLY library
  - finds legacy files missing `resource_*` vertex properties
  - rewrites those files to add only the resource-binary columns actually used by that tree model

## TODO
- Current outstanding issue: we are still manually adjusting the `resource_*` binary properties on the following tree PLY models.
- To do: trace the upstream export step that is dropping or miswriting those binary attributes.
- `precolonial.False_size.fallen_control.improved-tree_id.10.ply`
- `precolonial.False_size.fallen_control.improved-tree_id.11.ply`
- `precolonial.False_size.fallen_control.improved-tree_id.12.ply`
- `precolonial.False_size.fallen_control.improved-tree_id.13.ply`
- `precolonial.False_size.fallen_control.improved-tree_id.14.ply`
- `precolonial.False_size.fallen_control.improved-tree_id.7.ply`
- `precolonial.False_size.fallen_control.improved-tree_id.8.ply`
- `precolonial.False_size.fallen_control.improved-tree_id.9.ply`
- `precolonial.False_size.large_control.park-tree_id.8.ply`
- `precolonial.False_size.large_control.reserve-tree_id.8.ply`
- `precolonial.False_size.large_control.street-tree_id.8.ply`
- `precolonial.False_size.propped_control.improved-tree_id.10.ply`
- `precolonial.False_size.propped_control.improved-tree_id.11.ply`
- `precolonial.False_size.propped_control.improved-tree_id.12.ply`
- `precolonial.False_size.propped_control.improved-tree_id.13.ply`
- `precolonial.False_size.propped_control.improved-tree_id.14.ply`
- `precolonial.False_size.propped_control.improved-tree_id.7.ply`
- `precolonial.False_size.propped_control.improved-tree_id.8.ply`
- `precolonial.False_size.propped_control.improved-tree_id.9.ply`
- `precolonial.False_size.senescing_control.reserve-tree_id.10.ply`
- `precolonial.False_size.senescing_control.reserve-tree_id.11.ply`
- `precolonial.False_size.senescing_control.reserve-tree_id.12.ply`
- `precolonial.False_size.senescing_control.reserve-tree_id.13.ply`
- `precolonial.False_size.senescing_control.reserve-tree_id.14.ply`
- `precolonial.False_size.senescing_control.reserve-tree_id.7.ply`
- `precolonial.False_size.senescing_control.reserve-tree_id.8.ply`
- `precolonial.False_size.senescing_control.reserve-tree_id.9.ply`
- `precolonial.False_size.snag_control.improved-tree_id.8.ply`
- `precolonial.True_size.propped_control.improved-tree_id.11.ply`
- `precolonial.True_size.propped_control.improved-tree_id.12.ply`
- `precolonial.True_size.propped_control.improved-tree_id.13.ply`
- `precolonial.True_size.propped_control.improved-tree_id.14.ply`
- `precolonial.True_size.propped_control.improved-tree_id.15.ply`
- `precolonial.True_size.propped_control.improved-tree_id.16.ply`
