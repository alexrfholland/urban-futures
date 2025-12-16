## Habitat volume workflow

Keep steps minimal. Use as a quick reference.

### Frame the site

- Get site centre and dims from `final/f_SiteCoordinates.get_site_coordinates`
- Load photo meshes inside bounds via `final/f_photoMesh.get_meshes_in_bounds`
- Update centre and dims using mesh bounds with `final/f_SiteCoordinates.get_center_and_dims`

### Terrain and fabric

- Terrain from contours with `final/f_GeospatialModule.handle_contours`
- Buildings to mesh with `final/f_GetSiteMeshBuildings.process_buildings`
- Segment site points against buildings and LAS via `final/f_segmentation_manager.segmentFunction`

### Urban features

- Road segments from API via `final/f_GeospatialModule.handle_road_segments`
- Under‑canopy voxels from canopies via `final/f_GeospatialModule.handle_tree_canopies`
- Urban forest records via `final/f_GeospatialModule.handle_urban_forest`
- Optional poles, parking, laneways, open space, green roofs via `final/f_GeospatialModule.handle_*`

### Scenario setup

- Initialise dataset and load node tables via `final/a_scenario_initialiseDS.*`
- Either run per‑year scenario or load cached via `final/a_scenario_runscenario`
- Export per‑year VTK via `final/a_scenario_generateVTKs`

### Resistance field

- Assign resistance 0–100, aggregate canopy footprint and patches

### Optional utilities

- Roof loads and log allocation via `final/a_logDistributor.py`
- Patch aggregation for connectivity via `final/a_rewilding.py`






