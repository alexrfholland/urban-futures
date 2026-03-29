# 2.3-INFO-2 Proposal And Intervention Technical Specifications

This note freezes the current technical mapping for:

- `2.3-INFO-2. Proposal Opportunities`
- `2.3-INFO-3. Community Response`
- `2.3-INFO-3.1. Refactor Statistics Exports`

For conceptual definitions and manuscript-facing descriptions, use [2-3-INFO-2-proposal-and-intervention-descriptions.md](./2-3-INFO-2-proposal-and-intervention-descriptions.md).

It separates:

- how each proposal or intervention is measured for assessment
- how each proposal or intervention is visualised in Blender

## Proposal Set

Use this five-proposal manuscript set:

- `Deploy-Structure`
- `Decay`
- `Recruit`
- `Colonise`
- `Release-Control`

Do not emit any extra stub proposals outside the five manuscript proposals.

## Intervention Set

Use this manuscript intervention set:

- `Buffer-Feature`
- `Brace-Feature`
- `Rewild-Ground`
- `Adapt-Utility-Pole`
- `Upgrade-Feature`
- `Enrich-Envelope`
- `Roughen-Envelope`

## Detailed Source Mapping

These tables keep the earlier pipeline-style detail:

- proposal or intervention name
- what gets counted
- dataset used
- node-table source fields
- VTK source fields

## 2.3-INFO-2. Proposal Opportunities Detailed

| Proposal | What gets counted as an opportunity to make this proposal | Current tracking | Node DF name | Node DF column | Node DF field/value used | VTK name | VTK point data attribute name | VTK point data attribute field/value used | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Deploy-Structure` | utility poles designed for artificial canopies this turn | direct | `poleDF_{year}.csv` | `isEnabled`, `size`, `precolonial` | `isEnabled == True`; `size == artificial`; `precolonial == False` | `*_urban_features_with_indicators.vtk` | `forest_size`, `forest_precolonial` | `artificial`; `False` | Canonical proposal measurement is pole count from `poleDF`. |
| `Decay` | trees reaching senescence this turn | direct | `treeDF_{year}.csv` | `isNewTree`, `action` | `isNewTree == False`; `action == AGE-IN-PLACE` or `SENESCENT` | `*_urban_features_with_indicators.vtk` | `scenario_rewilded` | `exoskeleton`, `footprint-depaved`, `node-rewilded`, `rewilded` | Canonical proposal measurement is senescing tree count from `treeDF`. |
| `Recruit` | all `ground_only` voxels within `20m` of a canopy-feature | direct | none | none | none | `*_urban_features_with_indicators.vtk` | derived from capability search logic | `ground_only`; within `20m` of `canopy-feature` | `ground_only` excludes facade, green roof, and brown roof. |
| `Colonise` | voxels designated as `brownRoof`, `greenRoof`, `livingFacade`, `footprint-depaved`, `node-rewilded`, `otherGround`, `rewilded` | direct | none | none | none | `*_urban_features_with_indicators.vtk` | `scenario_outputs` | `brownRoof`, `greenRoof`, `livingFacade`, `footprint-depaved`, `node-rewilded`, `otherGround`, `rewilded` | Canonical proposal measurement is from VTK surface-state categories. |
| `Release-Control` | all arboreal voxels | direct | none | none | none | `*_urban_features_with_indicators.vtk` | `search_bioavailable` | `arboreal` | Canonical proposal measurement is all arboreal voxel space. |

## 2.3-INFO-3. Community Response Detailed

| Proposal | Intervention | What gets counted as this intervention | Current tracking | Node DF name | Node DF column | Node DF field/value used | VTK name | VTK point data attribute name | VTK point data attribute field/value used | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Deploy-Structure` | `Adapt-Utility-Pole` | utility poles designed for artificial canopies this turn | direct | `poleDF_{year}.csv` | `isEnabled`, `size`, `precolonial` | `isEnabled == True`; `size == artificial`; `precolonial == False` | `*_urban_features_with_indicators.vtk` | `forest_size`, `forest_precolonial` | `artificial`; `False` | Canonical intervention measurement stays on `poleDF`. |
| `Deploy-Structure` | `Upgrade-Feature` | non-precolonial peeling-bark voxels | assumed | none | none | none | `*_urban_features_with_indicators.vtk` | `forest_precolonial`, `indicator_Bird_self_peeling` | `False`; `True` | Assumes all peeling bark in elms is artificial bark installation; currently does not track artificial hollows in upgraded canopies. |
| `Decay` | `Buffer-Feature` | trees reaching senescence this turn and allocated to `node-rewilded` or `footprint-depaved` | direct | `treeDF_{year}.csv` | `rewilded` | `node-rewilded`, `footprint-depaved` | `*_urban_features_with_indicators.vtk` | `scenario_bioEnvelope` | `node-rewilded`, `footprint-depaved` | Blender state comes from `bioEnvelope`; measurement comes from `treeDF`. |
| `Decay` | `Brace-Feature` | trees reaching senescence this turn and allocated to `exoskeleton` | direct | `treeDF_{year}.csv` | `rewilded` | `exoskeleton` | `*_urban_features_with_indicators.vtk` | `scenario_bioEnvelope` | `exoskeleton` | Blender state comes from `bioEnvelope`; measurement comes from `treeDF`. |
| `Recruit` | `Buffer-Feature` | voxels where `indicator_Tree_generations_grassland == True` and Blender state is `node-rewilded` or `footprint-depaved` | proxy | none | none | none | `*_urban_features_with_indicators.vtk` | `indicator_Tree_generations_grassland`, `scenario_bioEnvelope` | `True`; `node-rewilded`, `footprint-depaved` | Intervention is measured from achieved recruit grassland and split by `bioEnvelope` state. |
| `Recruit` | `Rewild-Ground` | voxels where `indicator_Tree_generations_grassland == True` and Blender state is `otherGround` or `rewilded` | proxy | none | none | none | `*_urban_features_with_indicators.vtk` | `indicator_Tree_generations_grassland`, `scenario_bioEnvelope` | `True`; `otherGround`, `rewilded` | Intervention is measured from achieved recruit grassland and split by `bioEnvelope` state. |
| `Colonise` | `Rewild-Ground` | voxels designated `node-rewilded`, `rewilded`, `footprint-depaved` | direct | none | none | none | `*_urban_features_with_indicators.vtk` | `scenario_outputs`, `scenario_bioEnvelope` | `node-rewilded`, `rewilded`, `footprint-depaved` | `footprint-depaved` remains grouped with rewilded ground. |
| `Colonise` | `Enrich-Envelope` | `greenRoof` voxels, plus enabled rooftop log rows where `roofID` is present | direct | `logDF_{year}.csv` | `isEnabled`, `roofID` | `isEnabled == True`; rooftop log rows with `roofID` present | `*_urban_features_with_indicators.vtk` | `scenario_outputs`, `scenario_bioEnvelope` | `greenRoof` | Blender still needs a dedicated transported-log field in `logDF` to highlight logs explicitly. |
| `Colonise` | `Roughen-Envelope` | voxels designated `brownRoof`, `livingFacade` | direct | none | none | none | `*_urban_features_with_indicators.vtk` | `scenario_outputs`, `scenario_bioEnvelope` | `brownRoof`, `livingFacade` | Brown roof and facade roughening counted together. |
| `Release-Control` | `Buffer-Feature` | arboreal voxels where pruning is fully withdrawn | direct | none | none | none | `*_urban_features_with_indicators.vtk` | `search_bioavailable`, `forest_control` | `arboreal`; `reserve-tree`, `improved-tree` | Assumes pruning withdrawn from canopy. |
| `Release-Control` | `Brace-Feature` | arboreal voxels where pruning is partially withdrawn | direct | none | none | none | `*_urban_features_with_indicators.vtk` | `search_bioavailable`, `forest_control` | `arboreal`; `park-tree` | Assumes intermediate pruning withdrawn from canopy. |

## 2.3 Visualisation Strategy In Blender

These simplified tables are the Blender-facing version of the mapping. They describe:

- the canonical measurement used for assessment
- the dataset used for that measurement
- the state or source Blender should use for visualisation

## 2.3-INFO-2. Proposal Opportunities Blender Strategy

| Proposal | Canonical measurement | Measurement dataset | Blender |
| --- | --- | --- | --- |
| `Deploy-Structure` | number of utility poles designed for artificial canopies this turn | `poleDF` | from `poleDF` |
| `Decay` | number of trees reaching senescence this turn | `treeDF` | from `treeDF` / `nodeDF` |
| `Recruit` | all `ground_only` voxels within `20m` of a canopy-feature | `vtk` | `ground_only` rewilded ground from `scenario_bioEnvelope`, constrained by the recruit-capability mask |
| `Colonise` | voxels designated as `brownRoof`, `greenRoof`, `livingFacade`, `footprint-depaved`, `node-rewilded`, `otherGround`, `rewilded` | `vtk` | from `scenario_bioEnvelope` |
| `Release-Control` | all arboreal voxels | `vtk` | from arboreal canopy and tree control states |

## 2.3-INFO-3. Community Response Blender Strategy

| Proposal | Intervention | Measurement | Measurement dataset | Blender | Note |
| --- | --- | --- | --- | --- | --- |
| `Deploy-Structure` | `Adapt-Utility-Pole` | number of utility poles designed for artificial canopies this turn | `poleDF` | from `poleDF` | Current direct measure for deployed artificial canopy structures. |
| `Deploy-Structure` | `Upgrade-Feature` | non-precolonial peeling-bark voxels | `vtk` | from `vtk` | Assumes all peeling bark in elms is artificial bark installation; currently does not track artificial hollows in upgraded canopies. |
| `Decay` | `Buffer-Feature` | number of trees reaching senescence this turn and allocated to full-support states (`node-rewilded`, `footprint-depaved`) | `treeDF` | `node-rewilded`, `footprint-depaved` from `bioEnvelope` | Ageing feature can senesce and collapse in place. |
| `Decay` | `Brace-Feature` | number of trees reaching senescence this turn and allocated to `exoskeleton` | `treeDF` | `exoskeleton` from `bioEnvelope` | Ageing feature retained in place without collapse zone. |
| `Recruit` | `Buffer-Feature` | `indicator_Tree_generations_grassland == True` | `vtk` | `node-rewilded`, `footprint-depaved` in `bioEnvelope` where capability tree recruit is on | Tree recruitment supported through tree-led local rewilding states. |
| `Recruit` | `Rewild-Ground` | `indicator_Tree_generations_grassland == True` | `vtk` | `otherGround`, `rewilded` in `bioEnvelope` where capability tree recruit is on | Tree recruitment supported through broader ground release and rewilding. |
| `Colonise` | `Rewild-Ground` | voxels designated `node-rewilded`, `rewilded`, `footprint-depaved` | `vtk` | `node-rewilded`, `rewilded`, `footprint-depaved` from `bioEnvelope` | `footprint-depaved` remains grouped with rewilded ground. |
| `Colonise` | `Enrich-Envelope` | `greenRoof` voxels, plus enabled rooftop log rows where `roofID` is present | `vtk` + `logDF` | `greenRoof` from `bioEnvelope` | The renderer will need to add a dedicated field to transported logs in `logDF` so these can be highlighted explicitly under this proposal. |
| `Colonise` | `Roughen-Envelope` | voxels designated `brownRoof`, `livingFacade` | `vtk` | `brownRoof`, `livingFacade` from `bioEnvelope` | Brown roof and facade roughening counted together. |
| `Release-Control` | `Buffer-Feature` | arboreal voxels where pruning is fully withdrawn (`forest_control == reserve-tree` or `improved-tree`) | `vtk` | `reserve-tree`, `improved-tree` control states | Assumes pruning withdrawn from canopy. |
| `Release-Control` | `Brace-Feature` | arboreal voxels where pruning is partially withdrawn (`forest_control == park-tree`) | `vtk` | `park-tree` control state | Assumes intermediate pruning withdrawn from canopy. |

## `scenario_bioEnvelope` Values Used In Blender

`scenario_bioEnvelope` is initialized as a copy of `scenario_rewilded`, then extended with envelope categories in [a_scenario_generateVTKs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_generateVTKs.py).

The current values are:

- `node-rewilded`
- `footprint-depaved`
- `exoskeleton`
- `rewilded`
- `otherGround`
- `livingFacade`
- `greenRoof`
- `brownRoof`
- `none`

## Refactor Statistics Export

`a_info_proposal_interventions.py --export-refactor-statistics` now writes:

- per-site raw long tables:
  - `_statistics-refactored/raw/{site}/proposals.csv`
  - `_statistics-refactored/raw/{site}/interventions.csv`
- aggregate comparison longs:
  - `_statistics-refactored/comparison/proposals.csv`
  - `_statistics-refactored/comparison/interventions.csv`
- aggregate highlights:
  - `_statistics-refactored/highlights/proposals.csv`
  - `_statistics-refactored/highlights/interventions.csv`

Each raw row carries `last_updated`. Partial reruns replace only the requested `site + scenario + year` scope in the site raw files, then rebuild aggregate comparison and highlight outputs.

## Open Temporary Assumptions

- `Upgrade-Feature` is currently a proxy based on `forest_precolonial == False` and `indicator_Bird_self_peeling == True`.
- `Upgrade-Feature` currently assumes all peeling bark in elms is artificial bark installation and does not separately track artificial hollows in upgraded canopies.
- `Recruit` proposal opportunity is broader than the `Recruit` intervention measurement. The proposal is measured as recruitable ground near canopy; the intervention is measured as achieved `Tree.generations.grassland`.
- `Colonise -> Enrich-Envelope` includes rooftop logs in measurement logic, but Blender still needs a dedicated transported-log field in `logDF` to highlight them explicitly.

## Indicator Coverage Check

This table checks whether the current proposed proposal/intervention set captures each indicator outcome, and where coverage remains partial or proxy-based.

| Indicator | Main proposal / intervention coverage | Coverage status | What is not fully captured |
| --- | --- | --- | --- |
| `Bird.self.peeling` | `Decay`, `Release-Control`, `Deploy-Structure`, `Colonise` at City through `Enrich-Envelope` | mostly covered | `Upgrade-Feature` remains a proxy assumption; inherited peeling bark in retained canopy is not separated from proposal-driven gains. |
| `Bird.others.perch` | `Release-Control`, `Decay`, `Deploy-Structure` | partial | Large portions of perchable canopy persist in ordinary younger canopy and are not clearly attributable to proposal support. |
| `Bird.generations.hollow` | `Decay`, `Deploy-Structure` | partial | Artificial hollows in upgraded canopies are not separately tracked; current `Upgrade-Feature` logic does not yet measure hollow support directly. |
| `Lizard.self.grass` | `Colonise`, `Recruit`, and indirectly `Decay` / `Release-Control` through lower-management ground | partial | Existing open ground and inherited low-vegetation conditions are not clearly separated from proposal-driven conversion. |
| `Lizard.self.dead` | `Decay`, `Release-Control`, `Colonise` | partial | Dead branch support is not decomposed cleanly by proposal family, especially where retained canopy already carries deadwood. |
| `Lizard.self.epiphyte` | `Decay`, `Release-Control`, `Colonise` | partial | Existing epiphyte-bearing structures and artificial installations are not cleanly separated from proposal-driven gains. |
| `Lizard.others.notpaved` | `Colonise`, `Recruit`, `Decay`, `Release-Control` | partial | Existing open space and inherited unpaved movement surfaces remain mixed into the indicator and are not fully attributable to proposals. |
| `Lizard.generations.nurse-log` | `Decay`, `Colonise` through `Enrich-Envelope` | mostly covered | Existing retained logs and debris are not always separated from proposal-driven additions. |
| `Lizard.generations.fallen-tree` | `Decay`, `Colonise` through `Enrich-Envelope` | mostly covered | Same issue as nurse logs: inherited or retained fallen structure is not always separated from proposal-driven additions. |
| `Tree.self.senescent` | `Decay` | mostly covered | Some senescent volume may persist without explicit support, but the main logic is well captured. |
| `Tree.others.notpaved` | `Colonise`, `Recruit`, `Decay`, `Release-Control` | partial | Existing open soil near canopy is a major contributor and is not fully separated from proposal-generated conditions. |
| `Tree.generations.grassland` | `Recruit`, `Colonise`, and indirectly `Decay` / `Release-Control` through ground conversion near canopy | mostly covered with proxy | `Recruit` intervention is measured directly as achieved grassland, but the broader `Recruit` proposal opportunity remains an assessment-layer construct. |
