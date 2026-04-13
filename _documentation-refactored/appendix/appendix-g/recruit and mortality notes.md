# Recruit Proposal notes

# Mortality difference between full and partial recruit support

In addition to being differentiated by the size of the groundspace being made appropriate, support for the `recruit` proposal also changes the mortality of saplings, depending on whether it has full support (`rewild ground` intervention that opens larger swathes of ground to recruit) or partial (`buffer tree` only allowing recruitment under a selection of trees that have undergone `buffer`).

For mortality, we use two Le Roux data values that are described as derived from raw vegetation data or, where applicable, published estimates.[^leroux]

- `annual_tree_death_urban = 0.06`. Roughly 61% of trees survive over 8 years.[^leroux]
- `annual_tree_death_nature-reserves = 0.03`. Roughly 78% of trees survive over 8 years.[^leroux]

We use `annual_tree_death_urban` by default and `annual_tree_death_nature-reserves` for tree mortality in larger rewilded areas to reflect these different conditions of proposals under recruit that are partially supported (smaller nodes, higher mortality) and fully supported (rewilded nodes, lower mortality). We also increase the density of plantings in the larger recruitment zones to reflect the increased support available in those areas.

In the current v4 implementation, we follow Le Roux's cohort-thinning logic, keep their standard `0.06` urban and `0.03` nature-reserve annual mortality values as anchors, and apply DBH-cohort survival curves based on their assessments of tree survival across successive cohorts.

### Three recruitment types and their mortality mapping

The v4 engine distinguishes three recruitment mechanisms via the `recruit_mechanism` column on the treeDF:

| `recruit_mechanism` | Zone mask | Mortality anchor | Rationale |
|---|---|---|---|
| `node-rewild` | `scenario_nodeRewildRecruitZone` | Reserve (0.03) | Large rewilded ground patches around node-rewilded trees — conditions approximate nature reserves |
| `under-canopy` | `scenario_underCanopyRecruitZone` | Urban (0.06) | Smaller patches under footprint-depaved/exoskeleton tree canopies — urban conditions persist |
| `ground` | `scenario_rewildGroundRecruitZone` | Reserve (0.03) | Remaining depaved ground filtered by node-exclusion — larger open areas similar to node-rewild |

Previously (v3), these were called `buffer-feature` (urban mortality) and `rewild-ground` (reserve mortality). The v4 split into three types provides finer spatial control while preserving the same mortality logic.

Other `small`, `medium`, and `large` trees use the urban mortality rate by default.

### Zone masks

All three zone masks are computed together in `calculate_under_node_treatment_status()` (`engine_v3.py`). Each mask uses the convention `>= 0` = active (value is the year enabled), `-1` = inactive.

- `scenario_nodeRewildRecruitZone` — ground voxels mapped via `sim_Nodes` to node-rewilded tree NodeIDs
- `scenario_underCanopyRecruitZone` — canopy voxels mapped via `node_CanopyID` to footprint-depaved/exoskeleton tree NodeIDs
- `scenario_rewildGroundRecruitZone` — remaining depaved ground, filtered by `ground_filter_mode` (default: `node-exclusion`, which excludes voxels already covered by the other two zones)

### Summary

- adds annual_tree_death_urban = 0.06
- adds annual_tree_death_nature-reserves = 0.03
- applies this to small, medium, and large trees (v4: extended to large; v3 applied only to small and medium)
- uses node-rewild and ground recruits as the lower-mortality case (reserve rate)
- uses under-canopy recruits as the urban-rate case
- uses the urban rate for other small / medium / large trees by default

Definitions for the current implementation:

- `raw mortality` = the literal annual mortality implied by a direct Le Roux-style cohort calculation from the raw cohort totals, using `1 - s^(1/8)`. This is useful as a shape reference, but it is noisy and can produce invalid negative values.
- `shaped mortality` = the smoothed cohort factor we apply to the anchor rate in the current engine.
- `s over 8 years` = the implied proportion of trees surviving an 8-year period for that cohort after shaping, calculated as `(1 - annual mortality)^8`.
- `annual mortality` = the final annual cohort mortality rate used in the current engine.

Urban cohorts, anchored to `0.06`

| DBH cohort | Raw mortality | Shaped mortality | `s` over 8 years | Annual mortality |
| --- | ---: | ---: | ---: | ---: |
| `0-10` | `0.1067` | `1.0000` | `0.6096` | `0.060` |
| `10-20` | `-0.0407` | `0.7500` | `0.6919` | `0.045` |
| `20-30` | `0.0503` | `0.5833` | `0.7520` | `0.035` |
| `30-40` | `0.0540` | `0.5000` | `0.7837` | `0.030` |
| `40-50` | `0.0632` | `0.4333` | `0.8100` | `0.026` |
| `50-60` | `0.0511` | `0.3500` | `0.8438` | `0.021` |
| `60-70` | `0.0173` | `0.2500` | `0.8861` | `0.015` |
| `70-80` | `0.0950` | `0.1667` | `0.9227` | `0.010` |

Nature-reserve cohorts, anchored to `0.03`

| DBH cohort | Raw mortality | Shaped mortality | `s` over 8 years | Annual mortality |
| --- | ---: | ---: | ---: | ---: |
| `0-10` | `0.2186` | `1.0000` | `0.7837` | `0.030` |
| `10-20` | `0.0974` | `0.8000` | `0.8234` | `0.024` |
| `20-30` | `0.1085` | `0.6333` | `0.8577` | `0.019` |
| `30-40` | `0.1919` | `0.5000` | `0.8861` | `0.015` |
| `40-50` | `-0.1366` | `0.4000` | `0.9079` | `0.012` |
| `50-60` | `0.0745` | `0.3000` | `0.9302` | `0.009` |
| `60-70` | `0.0261` | `0.2333` | `0.9454` | `0.007` |
| `70-80` | `0.0000` | `0.1667` | `0.9607` | `0.005` |

# Recruit density values

Our current recruit density is `50 trees/ha`, but this is a pulse-scaled establishment value, not a final mature woodland density. In practice, that means:

- `10-year pulse = 16.7 trees/ha`
- `20-year pulse = 33.3 trees/ha`
- `30-year pulse = 50 trees/ha`

This places the model on the assertive side, but still within the same broad range as several Australian woodland references.

- `Planting for Wildlife` gives an open-woodland benchmark of about `30 mature trees/ha` on pre-cleared hilltops and plains, with higher densities along watercourses.[^planting]
- `A Guide to Managing Box Gum Grassy Woodlands` illustrates box-gum woodland at `35 trees/ha`.[^boxgum]
- `The Future of Scattered Trees in Agricultural Landscapes` uses `10-25 trees/ha` as the initial yellow-box scattered-tree density. The same paper then models a stronger regeneration response of `2 recruits per initial tree every 30 years`, so it supports low mature densities but also shows that recruit pulses can be substantially higher than the final standing structure.[^gibbons]
- `Reversing a Tree Regeneration Crisis in an Endangered Ecoregion` is less useful as a direct density benchmark, but it is important context: the problem in these systems is not just low mature tree density but failed regeneration over time.[^fischer]
- `The Future of Large Old Trees in Urban Landscapes` reports `11-13 seedlings/ha` in the `0-10 cm DBH` cohort over an approximately `8-year` step in urban greenspace.[^leroux_density] That is not the same thing as our planting quota, but it is the same order of magnitude. If our `50 trees/ha` setting is treated as a 30-year establishment pulse, it scales to `13.3 trees/ha` over 8 years, which is very close to Le Roux's urban cohort figure.


[^leroux]: Le Roux et al. describe annual mortality as density-dependent across successive `10 cm DBH` cohorts, using `1 - s^(1/y)`, where `s` is survival from one cohort to the next and `y` is the years required to grow through that `10 cm` DBH increment. They also report annual mortality values of `0.06` for urban greenspace and `0.03` for nature reserves in their parameter table. Source: [Le Roux et al. - 2014 - The Future of Large Old Trees in Urban Landscapes](/Users/alexholland/Zotero/storage/AZDXFYZR/Le%20Roux%20et%20al.%20-%202014%20-%20The%20Future%20of%20Large%20Old%20Trees%20in%20Urban%20Landscapes.pdf).
[^planting]: Munro and Lindenmayer note that in pre-cleared landscapes, tree density on hilltops and plains was often low, "about 30 mature trees per hectare", with higher density along watercourses. Source: [Munro and Lindenmayer - Planting for Wildlife](/Users/alexholland/Zotero/storage/L4JWA59N/Munro%20and%20Lindenmayer%20-%20Planting%20for%20Wildlife.pdf).
[^boxgum]: Rawlings et al. illustrate box-gum woodland at `35 trees/ha` in Figure 2.1. Source: [Rawlings et al. - 2010 - A Guide to Managing Box Gum Grassy Woodlands](/Users/alexholland/Zotero/storage/EIPBQHHE/Rawlings%20et%20al.%20-%202010%20-%20A%20Guide%20to%20Managing%20Box%20Gum%20Grassy%20Woodlands.pdf).
[^gibbons]: Gibbons et al. report `10-25 trees/ha` as the initial yellow-box scattered-tree density in their case study table, and model regeneration in yellow-box stands by increasing recruits to `2.0` for each initial tree every `30 years`. Source: [Gibbons et al. - 2008 - The future of scattered trees in agricultural landscapes](/Users/alexholland/Zotero/storage/4PKDU46W/Gibbons-2008-The%20future%20of%20scattered%20trees%20in.pdf).
[^fischer]: Fischer et al. show that many paddock systems are regeneration-limited and ageing, with young trees often missing from low-density sites. This is a regeneration-crisis argument, not a direct planting-density prescription. Source: [Fischer et al. - 2009 - Reversing a Tree Regeneration Crisis in an Endangered Ecoregion](/Users/alexholland/Zotero/storage/H73BB8AC/Fischer%20et%20al.%20-%202009%20-%20Reversing%20a%20Tree%20Regeneration%20Crisis%20in%20an%20Endange.pdf).
[^leroux_density]: Le Roux et al. report a mean of `11` and `13 seedlings/ha` in the `0-10 cm DBH` cohort for two urban species groups, with an approximately `8-year` model time step. Source: [Le Roux et al. - 2014 - The Future of Large Old Trees in Urban Landscapes](/Users/alexholland/Zotero/storage/AZDXFYZR/Le%20Roux%20et%20al.%20-%202014%20-%20The%20Future%20of%20Large%20Old%20Trees%20in%20Urban%20Landscapes.pdf).
