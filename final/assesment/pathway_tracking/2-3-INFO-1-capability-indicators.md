# 2.3-INFO-1 Capability Indicators

Scripts:

- `final/a_info_gather_capabilities.py`
- `final/a_info_output_capabilities.py`

Inputs:

- `urban features state [vtk]` - `data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk`
- `baseline urban features state [optional vtk]` - `data/revised/final/baselines/{site}_baseline_combined_{voxel_size}_urban_features.vtk`

Description of processing:

- `a_info_gather_capabilities.py` reads the urban-features VTK, applies the capability queries, writes one boolean indicator layer per outcome into a new VTK, and writes the per-site indicator and support-action CSVs.
- `a_info_output_capabilities.py` reads those per-site CSVs and writes the combined all-sites and pathway-summary CSVs; it does not write a VTK.

Outputs:

- `indicator outcomes state [vtk]` - `data/revised/final/output/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features_with_indicators.vtk`
- `indicator counts [dataframe csv]` - `data/revised/final/output/csv/{site}_{voxel_size}_indicator_counts.csv`
- `action counts [dataframe csv]` - `data/revised/final/output/csv/{site}_{voxel_size}_action_counts.csv`
- `all sites indicator counts [dataframe csv]` - `data/revised/final/output/csv/all_sites_{voxel_size}_indicator_counts.csv`
- `all sites action counts [dataframe csv]` - `data/revised/final/output/csv/all_sites_{voxel_size}_action_counts.csv`
- `all sites proposal opportunities [dataframe csv]` - `data/revised/final/output/csv/all_sites_{voxel_size}_proposal_opportunities.csv`
- `all sites proposal interventions [dataframe csv]` - `data/revised/final/output/csv/all_sites_{voxel_size}_proposal_interventions.csv`
- `totals by site, persona, pathway [dataframe csv]` - `data/revised/final/output/csv/totals_by_site_persona_pathway_{voxel_size}.csv`
- `totals by pathway and persona [dataframe csv]` - `data/revised/final/output/csv/totals_by_pathway_persona_{voxel_size}.csv`
- `totals by pathway [dataframe csv]` - `data/revised/final/output/csv/totals_by_pathway_{voxel_size}.csv`

Chain note:

- The source of truth for the capability definitions lives in `a_info_gather_capabilities.py`.
- `a_info_gather_capabilities.py` is the only script in this step that writes a VTK.
- The VTK layer names use `indicator_{Persona}_{capability}_{indicator}` rather than the dotted ids in code.

Related docs:

- [comparison_pathways_indicators.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators.md)

Capability layers written into `*_urban_features_with_indicators.vtk`:

| Persona | Capability | Indicator label | VTK array name |
| --- | --- | --- | --- |
| Bird | `self` | Peeling bark volume | `indicator_Bird_self_peeling` |
| Bird | `others` | Perchable canopy volume | `indicator_Bird_others_perch` |
| Bird | `generations` | Hollow count | `indicator_Bird_generations_hollow` |
| Lizard | `self` | Ground cover area | `indicator_Lizard_self_grass` |
| Lizard | `self` | Dead branch volume | `indicator_Lizard_self_dead` |
| Lizard | `self` | Epiphyte count | `indicator_Lizard_self_epiphyte` |
| Lizard | `others` | Non-paved surface area | `indicator_Lizard_others_notpaved` |
| Lizard | `generations` | Nurse log volume | `indicator_Lizard_generations_nurse-log` |
| Lizard | `generations` | Fallen tree volume | `indicator_Lizard_generations_fallen-tree` |
| Tree | `self` | Senescing tree volume | `indicator_Tree_self_senescent` |
| Tree | `others` | Soil near canopy features | `indicator_Tree_others_notpaved` |
| Tree | `generations` | Grassland for recruitment | `indicator_Tree_generations_grassland` |
