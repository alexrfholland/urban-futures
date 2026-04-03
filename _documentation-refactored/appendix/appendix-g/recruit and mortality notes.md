# Recruit Proposal notes

# Mortality difference between full and partial recruit support

In addition to being differentiated by the size of the groundspace being made appropriate, support for the `recruit` proposal also changes the mortality of saplings, depending on whether it has full support (`rewild ground` intervention that opens larger swathes of ground to recruit) or partial (`buffer tree` only allowing recruitment under a selection of trees that have undergone `buffer`).

For mortality, we use two Le Roux data values that are described as derived from raw vegetation data or, where applicable, published estimates.[^leroux]

- `annual_tree_death_urban = 0.06`. Roughly 61% of trees survive over 8 years.[^leroux]
- `annual_tree_death_nature-reserves = 0.03`. Roughly 78% of trees survive over 8 years.[^leroux]

We use `annual_tree_death_urban` by default and `annual_tree_death_nature-reserves` for tree mortality in larger rewilded areas to reflect these different conditions of proposals under recruit that are partially supported (smaller nodes, higher mortality) and fully supported (rewilded nodes, lower mortality).

In the current v3 implementation, `buffer-feature` recruits use the urban mortality rate and `rewild-ground` recruits use the nature-reserve mortality rate. Other `small` and `medium` trees also use the urban mortality rate by default.

- adds annual_tree_death_urban = 0.06
- adds annual_tree_death_nature-reserves = 0.03
- applies this only to small and medium trees
- uses rewild-ground recruits as the lower-mortality case
- uses buffer-feature recruits as the urban-rate case
- uses the urban rate for other small / medium trees by default

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
