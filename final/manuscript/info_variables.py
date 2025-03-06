Key Variables in Scenario Generator
=================================

scenario_rewildingPlantings
--------------------------
Description: Tracks where new plantings are enabled in the scenario
Values:
- -1: Areas not eligible for planting
- years_passed: Areas enabled for planting in current timestep (0, 10, 30, 60, 180)
Thresholds: Determined by sim_TurnsThreshold in params dictionary for each site/scenario combination
Used in: plot_scenario_details()

scenario_bioEnvelope
-------------------
Description: Enhanced version of rewilded that includes green envelope classifications
Values:
- 'none': Default/unchanged areas
- 'otherGround': Ground areas that meet bio criteria
- 'livingFacade': Building facades that are bio-enabled
- 'greenRoof': Roof areas designated as green roofs
- 'brownRoof': Roof areas designated as brown roofs
- 'exoskeleton': Areas with structural support for aging trees
- 'footprint-depaved': Areas where pavement has been removed
- 'node-rewilded': Areas that have been rewilded based on node logic
Conditions:
- bioMask = (sim_Turns <= turnThreshold) & (sim_averageResistance <= resistanceThreshold) & (sim_Turns >= 0)
- 'otherGround': bioMask True & scenario_bioEnvelope 'none'
- 'livingFacade': site_building_element == 'facade' & bioMask True
- 'greenRoof': envelope_roofType == 'green roof' & bioMask True
- 'brownRoof': envelope_roofType == 'brown roof' & bioMask True
Used in: update_bioEnvelope_voxel_catagories(), process_scenarios()

maskforTrees
-----------
Description: Boolean mask indicating presence of tree resources
Values:
- True: Voxels containing tree resources (any resource except leaf litter)
- False: Voxels without tree resources
Conditions:
- True when any resource_* variable > 0 (excluding resource_leaf_litter)
Used in: plot_scenario_rewilded(), plot_scenario_details()

maskForRewilding
---------------
Description: Boolean mask indicating areas being rewilded
Values:
- True: Voxels that are part of rewilding areas
- False: Voxels not part of rewilding areas
Conditions:
- True when scenario_rewilded != 'none'
- True when scenario_bioEnvelope != 'none' (if logDF or poleDF present)
Used in: plot_scenario_rewilded(), process_scenarios()

Note: Thresholds for bioMask vary by site, scenario, and year:
Example thresholds for city/positive:
- Year 0:  turns ≤ 0,    resistance ≤ 0
- Year 10: turns ≤ 300,  resistance ≤ 50
- Year 30: turns ≤ 1250, resistance ≤ 50
- Year 60: turns ≤ 5000, resistance ≤ 68
- Year 180: turns ≤ 5000, resistance ≤ 96
"""