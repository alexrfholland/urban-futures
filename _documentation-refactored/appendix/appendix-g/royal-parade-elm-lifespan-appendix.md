# Royal Parade Elm Lifespan Appendix

We estimate elm lifespans under nonhuman-led conditions by starting from the documented Melbourne maximum and adding back a fraction of the European biological ceiling. We express this through the formula:

\[
A = D + g(E-D)
\]

In the project model, we also apply an additional sudden-death term for elms:

\[
p_{\mathrm{sudden\_death}} = 0.3
\]

**A**  
**Melbourne age-in-place**  
The lifespan distribution if trees are irrigated and pest-managed but not removed when they become hazardous, so they can persist while still alive beyond normal removal thresholds.

**D**  
**Melbourne documented maximums**  
The observed or published upper-tail ages for managed urban elms in Melbourne and Victoria: **150 / 170 / 200**.

**E**  
**Europe baseline**  
The biological ceiling under much better native or near-native conditions in European woodlands.

**E − D**  
**The gap**  
The gap between the European biological ceiling and the Melbourne documented maximum.

**g**  
**Gap-recovery factor**  
The fraction of that gap we assume is recovered when trees are left in place instead of being removed, taking into account the extra persistence of decaying in place, but also the continuing reduction imposed by urban settings, Melbourne climate, and future climate change.[^future-range-note]

**p\_sudden\_death**  
**Sudden-death proportion**  
An additional death chance used in the project model to mimic sudden collapse from heat stress and related factors in non-precolonial elms.[^project-sudden-death]

The distributions used in this appendix are:

- **E distribution:** **300 / 350 / 400**
- **D distribution:** **150 / 170 / 200**
- **A distribution, current or historical Melbourne:** **180 / 205 / 240**
- **A + climate distribution:** **165 / 190 / 220**
- **Melbourne typical distribution:** **105 / 115 / 125**

Melbourne documented maximums refers to observed or published upper-tail ages for managed urban elms in Melbourne and Victoria. Melbourne age-in-place assumes irrigation and pest control continue, but trees are not pruned or removed for public safety.[^age-in-place-note]

## 1. Biological Lifespan in Unmodified Conditions in Europe (E distribution)

- Royal Parade is dominated by English elm, *Ulmus procera*, with a small number of *Ulmus × hollandica*.[^rp-species]
- A Europe-wide elm reference treats English elm as *Ulmus minor* var. *vulgaris*, synonym *Ulmus procera*.[^atlas-taxonomy]
- That makes the closest wild comparator for the Royal Parade English elms the *U. minor* lineage, not a separate forest species with its own independent longevity table.[^atlas-taxonomy]
- In Europe, native elms are trees of cool mixed broadleaved forests, usually on fertile, moist soils, especially along rivers, streams, and floodplains.[^atlas-habitat]
- *U. glabra* is a moist, rich-soil forest and grove species of hemiboreal and temperate climates.[^atlas-habitat]
- *U. minor* is more riparian and edge-associated: floodplain woods, stream banks, wooded steppe margins, and forest edges rather than pure stands.[^atlas-habitat]
- The best forest-based European longevity table found gives maximum ages under optimum conditions of 300 years for *U. minor*, 250 years for *U. laevis*, and 400 years for *U. glabra*.[^efi]
- *U. × hollandica* is a hybrid between *U. minor* and *U. glabra*.[^atlas-taxonomy]

### Adopted European biological lifespan distribution

| Case | Years | Basis |
|---|---:|---|
| Min | 300 | Directly supported by the forest maximum for *U. minor*, the closest comparator for *U. procera*. |
| Mode | 350 | Inference: midpoint for a population dominated by *U. procera* / *U. minor* lineage, with a small admixture of *U. × hollandica*. |
| Max | 400 | Directly supported for *U. glabra*; used here as the upper ceiling for hybrid-adjacent elm potential in ideal European conditions. |

## 2. Documented Maximums in Melbourne and Victorian Conditions (D distribution)

- Victorian elm-management documents consistently state that elms in Australian urban conditions can be expected to reach about 100 to 150 years if well maintained and grown in favourable conditions.[^bunbury][^kingston]
- The strongest hard local evidence for the upper tail is Arthur's Elms at the Royal Botanic Gardens Melbourne. A Victorian management strategy notes that two of Arthur's Elms, thought to be the oldest elms in Australia, are about 170 years old.[^bacchus]
- A Victorian secondary source adds that a healthy Australian elm may possibly reach 200 years in ideal conditions.[^bendigo]
- These are managed urban specimens and managed-condition estimates. They are not evidence for unmanaged persistence after a tree has passed its safe retention age.

### Adopted Melbourne/Victoria documented-maximum distribution

| Case | Years | Basis |
|---|---:|---|
| Min | 150 | Published Australian benchmark for well-maintained elms in favourable conditions. |
| Mode | 170 | Anchored to Arthur's Elms, the clearest documented local exemplar of extreme age under managed Melbourne conditions. |
| Max | 200 | Secondary Victorian upper-tail estimate for ideal managed conditions; plausible but less directly evidenced than the 170-year observation. |

## 3. g Factor

- Melbourne typical refers to useful life expectancy under normal boulevard care.[^parkville-ule][^kingston-ule]
- For Royal Parade, a reasonable Melbourne typical distribution is **105 / 115 / 125**, based on the 2015-2025 replacement horizon for a cohort generally dated to 1913.[^parkville-55][^mmra][^rp-date]
- The age-in-place assumption is that irrigation and pest control continue, but trees are not pruned or removed for public safety, and can therefore persist while still alive beyond ordinary removal thresholds.[^age-in-place-note]
- Future climate adjustment follows the direction indicated by the City of Melbourne vulnerability study and the 2024 Greater Melbourne climate projections.[^future-urban-forest][^climate-2024][^future-range-note]
- The project also applies `proportion_sudden_death = 0.3` for non-precolonial trees, with a shorter sudden-death senescing duration, to mimic heat stress and related collapse pathways.[^project-sudden-death]

| Quantity | Value |
|---|---|
| Melbourne typical distribution | 105 / 115 / 125 |
| D distribution | 150 / 170 / 200 |
| g, current or historical Melbourne | 0.20 |
| g, future climate-adjusted Melbourne | 0.10 |
| proportion_sudden_death | 0.3 |

Using these values:

- **A distribution, current or historical Melbourne:** **180 / 205 / 240**
- **A + climate distribution:** **165 / 190 / 220**

## References

[^rp-species]: National Trust of Australia, "English Elm (*Ulmus procera*) - Royal Parade", notes that the avenue is English elm and also contains a small number of *Ulmus × hollandica*. <https://trusttrees.org.au/tree/VIC/Parkville/Royal_Parade>

[^rp-date]: City of Melbourne, *Parkville Urban Forest Precinct Plan 2015-2025*, which states that the elms present today are thought to have been planted in 1913; the National Trust listing gives the avenue age as approximately 120 years. <https://s3.ap-southeast-2.amazonaws.com/hdp.au.prod.app.com-participate.files/7614/3640/8190/903330_Parkville_final_small.pdf>; <https://trusttrees.org.au/tree/VIC/Parkville/Royal_Parade>

[^atlas-taxonomy]: Caudullo, G., and de Rigo, D. (2016), "Ulmus - elms in Europe: distribution, habitat, usage and threats", *European Atlas of Forest Tree Species*. The atlas treats English elm as *Ulmus minor* var. *vulgaris*, synonym *Ulmus procera*, and states that *U. glabra* and *U. minor* can hybridise to form *U. × hollandica*. <https://forest.jrc.ec.europa.eu/media/atlas/Ulmus_spp.pdf>

[^atlas-habitat]: Caudullo, G., and de Rigo, D. (2016), *European Atlas of Forest Tree Species*. The atlas describes European elms as components of cool mixed broadleaved forests on water- and nutrient-rich soils, mainly near rivers, streams and floodplains, and distinguishes the more forest-grove ecology of *U. glabra* from the more riparian and edge-associated ecology of *U. minor*. <https://forest.jrc.ec.europa.eu/media/atlas/Ulmus_spp.pdf>

[^efi]: Sabatini, F. M. et al. (2021), *Old-growth forests in Europe*. Table 4 gives maximum ages under optimum conditions, measured inside forests, of 300 years for *Ulmus minor*, 250 years for *Ulmus laevis*, and 400 years for *Ulmus glabra*. The same report notes that trees outside forests can sometimes become older than forest maxima. <https://efi.int/sites/default/files/images/resilience/OLD-GROWTH%20FORESTS_28.06.21.pdf>

[^bunbury]: John Patrick Landscape Architects (2022), *Landscape Heritage Assessment - Heritage Impact Statement: Bunbury Street, Footscray*. This report states that Australian conditions reduce elm lifespan to about 100-150 years for well-maintained specimens in favourable conditions, compared with 250-350 years in Europe. <https://www.maribyrnong.vic.gov.au/files/assets/public/planning-services-documents/city-design/projects/bunbury-street/landscape-heritage-assessment-bunbury-st-footscray-r2-nov-2022.pdf>

[^kingston]: Tree Logic (2022), *Tree Management Plan: Kingston Avenue of Honour*. This report states that elms in Europe live around 250 years, rarely exceeding 350 years, and that well-managed Australian elms could be expected to live up to about 150 years. <https://kingstonavenueofhonour.org.au/wp-content/uploads/22-03-01_KingstonFoAKingstonAvenueofHonour_TRA.pdf>

[^bacchus]: Heritage Victoria / Moorabool Shire Council (2023), *The Bacchus Marsh Avenue of Honour Management Strategy*. This states that two of Arthur's Elms at the Royal Botanic Gardens Melbourne, thought to be the oldest elms in Australia, are 170 years old. <https://www.heritage.vic.gov.au/__data/assets/pdf_file/0026/741671/Bacchus-Marsh-Avenue-of-Honour-Management-Strategy.pdf>

[^bendigo]: Edwards, M. (2012), "History lives in old survivor". This Victorian secondary source states that a healthy Australian elm could achieve 100-150 years, and possibly 200 years in ideal conditions. <https://communityhistoryoz.wordpress.com/wp-content/uploads/2015/10/ltu-cha-2012-tree-edwards.pdf>

[^parkville-ule]: City of Melbourne, *Parkville Urban Forest Precinct Plan 2015-2025*. The plan defines useful life expectancy as an estimate of how long a tree is likely to remain in the landscape based on health, amenity, environmental service contribution, and risk to the community. <https://s3.ap-southeast-2.amazonaws.com/hdp.au.prod.app.com-participate.files/7614/3640/8190/903330_Parkville_final_small.pdf>

[^kingston-ule]: Tree Logic (2022), *Tree Management Plan: Kingston Avenue of Honour*. This plan states explicitly that biological lifespan exceeds ULE, and that ULE includes health, structure, maintenance cost, and public safety risk. <https://kingstonavenueofhonour.org.au/wp-content/uploads/22-03-01_KingstonFoAKingstonAvenueofHonour_TRA.pdf>

[^parkville-55]: City of Melbourne, *Parkville Urban Forest Precinct Plan 2015-2025*. The plan states that 55% of Melbourne's elms were in severe decline and likely to require removal within 10 years. <https://s3.ap-southeast-2.amazonaws.com/hdp.au.prod.app.com-participate.files/7614/3640/8190/903330_Parkville_final_small.pdf>

[^mmra]: Melbourne Metro Rail Authority (2016), Technical Note 072, which cites City of Melbourne urban forest precinct plans and notes that Royal Parade was identified as a corridor where trees would require replacement within 10 years. <https://bigbuild.vic.gov.au/__data/assets/pdf_file/0007/75229/MT-EES-TN072-Clarification-of-impacts-on-trees.PDF>

[^future-urban-forest]: Kendal, D., and Baumann, J. (2016), *The City of Melbourne's Future Urban Forest: Identifying the Vulnerability of Trees to the City's Future Temperature*. In the species tables, *Ulmus procera* and *Ulmus × hollandica* are rated in the highest risk category under future temperature scenarios. <https://futurenature.au/wp-content/uploads/2023/01/2016-CoM-Future-Urban-Forest-Final-Report-sml.pdf>

[^climate-2024]: Victorian Government (2024), *Greater Melbourne Climate Projections 2024*. The report projects more frequent hot days, hotter very hot days, longer and more intense heatwaves, declining cool-season rainfall, and lower future water availability across Greater Melbourne. <https://www.climatechange.vic.gov.au/__data/assets/pdf_file/0023/732380/greater-melbourne-victorias-climate-science-report-2024-collateral.pdf>

[^age-in-place-note]: The age-in-place values are not taken from a published table. They are derived from \(A = D + g(E-D)\), where \(D\) is the Melbourne documented maximum, \(E\) is the European biological ceiling, and \(g\) is the share of the remaining gap recovered when trees are left in place and persist while still alive beyond normal removal thresholds.

[^future-range-note]: No published source gives a direct future climate-adjusted lifespan table for the Royal Parade elms. In this project, the climate-adjusted distribution is estimated from the same \(A = D + g(E-D)\) relationship, with **\(g = 0.10\)**, in the direction indicated by the City of Melbourne vulnerability study and the 2024 Greater Melbourne climate projections.

[^project-sudden-death]: In the project model, `proportion_sudden_death = 0.3` for non-precolonial trees, with `senescing_duration_years_sudden_death = triangular_duration(0, 25, 40)`. The engine applies this as an additional sudden-death pathway during senescence. See [params_v3.py](../../../_futureSim_refactored/sim/setup/params_v3.py) and [engine_v3.py](../../../_futureSim_refactored/sim/generate_interim_state_data/engine_v3.py).
