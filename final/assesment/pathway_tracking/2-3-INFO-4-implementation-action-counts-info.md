# 2.3-INFO-4 Implementation Action Counts Info

Script:
- `final/a_info_gather_capabilities.py` - defines the implementation logic, counts support actions for each indicator outcome, and writes the per-site `action_counts.csv`.
- `final/a_info_output_capabilities.py` - reads the per-site `action_counts.csv` files and writes the combined all-sites action table.
Inputs:
- `urban features state [vtk]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk`
Description of processing:
- `a_info_gather_capabilities.py` reads the indicator-supporting voxels, counts which implementation conditions support each one, and writes the per-site support-action table.
- `a_info_output_capabilities.py` combines those per-site tables into a single all-sites summary; it does not define or recompute the logic.
Outputs:
- `a_info_gather_capabilities.py` outputs:
- `action counts [dataframe csv]` - `data/revised/final/output/csv/{site}_{voxel_size}_action_counts.csv`
- `a_info_output_capabilities.py` outputs:
- `all sites action counts [dataframe csv]` - `data/revised/final/output/csv/all_sites_{voxel_size}_action_counts.csv`
Chain note:
- This is the support-action layer used to describe implementation.
- The action-count logic is code-defined in `a_info_gather_capabilities.py` through `SUPPORT_ACTIONS`, `CONTROL_LEVELS`, `URBAN_ELEMENTS`, and `REWILDING_TYPES`.

Implementation logic types:

| Logic type | Source field | What it measures | Values written into `action_counts.csv` |
| --- | --- | --- | --- |
| `control_level` | `forest_control` | how much indicator-supporting canopy sits under each management level | `high`, `medium`, `low` |
| `urban_element` | `search_urban_elements` | how much indicator-supporting space sits on each urban surface or conversion type | `open space`, `green roof`, `brown roof`, `facade`, `roadway`, `busy roadway`, `existing conversion`, `other street potential`, `parking`, `none` |
| `rewilding_status` | `scenario_rewilded` | how much indicator-supporting tree space sits in each rewilding state | `footprint-depaved`, `exoskeleton`, `node-rewilded`, `none` |
| `artificial_structures_deployed` | `forest_size` | how much bird-supporting outcome is provided by artificial structures | `artificial` |
| `artificial` | `forest_precolonial` | how much indicator-supporting outcome is provided by installed non-precolonial elements | `installed` |

Capability-to-implementation mapping:

| Indicator outcome | Main implementation logic | Extra implementation counts |
| --- | --- | --- |
| `Bird.self.peeling` | `control_level` | `artificial_structures_deployed`, `artificial` |
| `Bird.others.perch` | `control_level` | `artificial_structures_deployed` |
| `Bird.generations.hollow` | `control_level` | `artificial_structures_deployed`, `artificial` |
| `Lizard.self.grass` | `urban_element` | none |
| `Lizard.self.dead` | `control_level` | none |
| `Lizard.self.epiphyte` | `control_level` | `artificial` |
| `Lizard.others.notpaved` | `urban_element` | none |
| `Lizard.generations.nurse-log` | `urban_element` | none |
| `Lizard.generations.fallen-tree` | `urban_element` | none |
| `Tree.self.senescent` | `rewilding_status` | none |
| `Tree.others.notpaved` | `urban_element` | none |
| `Tree.generations.grassland` | `urban_element` | none |
