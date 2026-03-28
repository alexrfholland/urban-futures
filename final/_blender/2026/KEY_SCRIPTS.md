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
  - do not rely on `AUTO_RUN_CLIPBOX_SETUP` or `AUTO_RUN_CAMERA_CLIPBOXES`; run the two follow-up scripts manually after instancing

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

## When Changing State
1. Run the instancer.
2. The instancer builds the available scenarios and configures the city view layers: `pathway_state`, `city_priority`, `trending_state`.
3. Run clipbox setup manually after instancing.
4. Then run camera clipboxes.

If you are using embedded Blender text blocks instead of the external files, keep the embedded copies in sync with these scripts.

## PLY Audit
- Script: [check_ply.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/_blender/2026/check_ply.py)
- Purpose:
  - scans the tree PLY library
  - finds legacy files missing `resource_*` vertex properties
  - rewrites those files to add only the resource-binary columns actually used by that tree model
