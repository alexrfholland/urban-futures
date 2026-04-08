# V4 Comparisons

Data root: `_data-refactored/model-outputs/generated-states/v4-proposal-broadcast`

Baselines: `{root}/output/feature-locations/{site}/{site}_baseline_1_nodeDF_yr-180.csv`

Scenario nodeDFs: `{root}/output/feature-locations/{site}/{site}_{scenario}_1_nodeDF_yr{year}.csv`

Sites: trimmed-parade, city, uni

Scenarios: positive, trending

Years: 0, 10, 30, 60, 90, 120, 150, 180

---

## Node DF Comparisons

Per site/scenario, count nodes by `size` column at each year. Delta is vs the baseline woodland (not yr 0).

Format: `count (+/-delta from baseline)`

### trimmed-parade / positive

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2014 | 160 (-1854) | 540 (-1474) | 654 (-1360) | 702 (-1312) | 682 (-1332) | 673 (-1341) | 674 (-1340) | 681 (-1333) |
| medium | 255 | 235 (-20) | 195 (-60) | 106 (-149) | 32 (-223) | 31 (-224) | 46 (-209) | 50 (-205) | 47 (-208) |
| large | 56 | 61 (+5) | 88 (+32) | 58 (+2) | 50 (-6) | 0 (-56) | 0 (-56) | 0 (-56) | 0 (-56) |
| senescing | 104 | 0 (-104) | 7 (-97) | 77 (-27) | 125 (+21) | 182 (+78) | 147 (+43) | 99 (-5) | 73 (-31) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 2 (-46) | 19 (-29) | 50 (+2) | 88 (+40) | 81 (+33) |
| fallen | 33 | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 1 (-32) | 4 (-29) | 10 (-23) | 40 (+7) |
| decayed | 38 | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 2 (-36) | 6 (-32) |
| **total** | **2548** | **456 (-2092)** | **830 (-1718)** | **895 (-1653)** | **911 (-1637)** | **915 (-1633)** | **920 (-1628)** | **923 (-1625)** | **928 (-1620)** |

### trimmed-parade / trending

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2014 | 160 (-1854) | 199 (-1815) | 318 (-1696) | 395 (-1619) | 456 (-1558) | 456 (-1558) | 456 (-1558) | 456 (-1558) |
| medium | 255 | 235 (-20) | 178 (-77) | 94 (-161) | 12 (-243) | 0 (-255) | 0 (-255) | 0 (-255) | 0 (-255) |
| large | 56 | 61 (+5) | 79 (+23) | 44 (-12) | 49 (-7) | 0 (-56) | 0 (-56) | 0 (-56) | 0 (-56) |
| senescing | 104 | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) |
| fallen | 33 | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) |
| decayed | 38 | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) |
| **total** | **2548** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** |

### city / positive

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2015 | 117 (-1898) | 354 (-1661) | 419 (-1596) | 482 (-1533) | 470 (-1545) | 454 (-1561) | 455 (-1560) | 452 (-1563) |
| medium | 255 | 154 (-101) | 160 (-95) | 88 (-167) | 23 (-232) | 19 (-236) | 35 (-220) | 34 (-221) | 31 (-224) |
| large | 56 | 3 (-53) | 4 (-52) | 4 (-52) | 3 (-53) | 0 (-56) | 0 (-56) | 0 (-56) | 0 (-56) |
| senescing | 104 | 0 (-104) | 6 (-98) | 30 (-74) | 65 (-39) | 81 (-23) | 66 (-38) | 41 (-63) | 35 (-69) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 7 (-41) | 19 (-29) | 41 (-7) | 41 (-7) |
| fallen | 33 | 0 (-33) | 241 (+208) | 475 (+442) | 759 (+726) | 764 (+731) | 778 (+745) | 787 (+754) | 838 (+805) |
| decayed | 34 | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 1 (-33) | 7 (-27) |
| **total** | **2545** | **274 (-2271)** | **765 (-1780)** | **1016 (-1529)** | **1332 (-1213)** | **1341 (-1204)** | **1352 (-1193)** | **1359 (-1186)** | **1404 (-1141)** |

### city / trending

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2015 | 117 (-1898) | 118 (-1897) | 203 (-1812) | 272 (-1743) | 279 (-1736) | 280 (-1735) | 279 (-1736) | 278 (-1737) |
| medium | 255 | 154 (-101) | 155 (-100) | 72 (-183) | 3 (-252) | 1 (-254) | 1 (-254) | 3 (-252) | 5 (-250) |
| large | 56 | 3 (-53) | 4 (-52) | 2 (-54) | 3 (-53) | 0 (-56) | 0 (-56) | 0 (-56) | 0 (-56) |
| senescing | 104 | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) |
| fallen | 33 | 0 (-33) | 0 (-33) | 19 (-14) | 64 (+31) | 94 (+61) | 120 (+87) | 144 (+111) | 193 (+160) |
| decayed | 34 | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) |
| **total** | **2545** | **274 (-2271)** | **277 (-2268)** | **296 (-2249)** | **342 (-2203)** | **374 (-2171)** | **401 (-2144)** | **426 (-2119)** | **476 (-2069)** |

### uni / positive

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2015 | 102 (-1913) | 204 (-1811) | 243 (-1772) | 272 (-1743) | 270 (-1745) | 265 (-1750) | 268 (-1747) | 296 (-1719) |
| medium | 255 | 83 (-172) | 90 (-165) | 40 (-215) | 17 (-238) | 9 (-246) | 18 (-237) | 19 (-236) | 18 (-237) |
| large | 56 | 2 (-54) | 4 (-52) | 2 (-54) | 1 (-55) | 0 (-56) | 0 (-56) | 0 (-56) | 0 (-56) |
| senescing | 104 | 0 (-104) | 3 (-101) | 27 (-77) | 42 (-62) | 52 (-52) | 42 (-62) | 24 (-80) | 17 (-87) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 4 (-44) | 13 (-35) | 31 (-17) | 29 (-19) |
| fallen | 33 | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 2 (-31) | 5 (-28) | 13 (-20) |
| decayed | 34 | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 1 (-33) |
| **total** | **2545** | **187 (-2358)** | **301 (-2244)** | **312 (-2233)** | **332 (-2213)** | **335 (-2210)** | **340 (-2205)** | **347 (-2198)** | **374 (-2171)** |

### uni / trending

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2015 | 102 (-1913) | 103 (-1912) | 150 (-1865) | 178 (-1837) | 184 (-1831) | 185 (-1830) | 187 (-1828) | 188 (-1827) |
| medium | 255 | 83 (-172) | 80 (-175) | 36 (-219) | 5 (-250) | 0 (-255) | 0 (-255) | 0 (-255) | 1 (-254) |
| large | 56 | 2 (-54) | 4 (-52) | 1 (-55) | 1 (-55) | 0 (-56) | 0 (-56) | 0 (-56) | 0 (-56) |
| senescing | 104 | 0 (-104) | 0 (-104) | 0 (-104) | 3 (-101) | 3 (-101) | 3 (-101) | 2 (-102) | 2 (-102) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 1 (-47) | 1 (-47) |
| fallen | 33 | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) |
| decayed | 34 | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) |
| **total** | **2545** | **187 (-2358)** | **187 (-2358)** | **187 (-2358)** | **187 (-2358)** | **187 (-2358)** | **188 (-2357)** | **190 (-2355)** | **192 (-2353)** |

---

## V4 Indicator Comparisons (voxel counts, yr 180)

Indicator definitions from `v4_indicator_definitions.md`. Baseline uses `indicator_Tree_generations_grassland` for `Tree.reproduce` aggregate (no scenario_rewilded on baselines). Tree.acquire.autonomous baseline reflects the full woodland arboreal volume under no management (all trees are "eliminate-pruning"). Tree.acquire.moderated baseline is 0 (no park trees in a woodland).

\* trending was 0; substituted ~1% of baseline for comparison columns

### trimmed-parade

| indicator | measure | baseline | positive yr180 | trending yr180 | positive / trending | trending % of positive |
|---|---|---|---|---|---|---|
| Bird.acquire.peeling-bark | `vtk["stat_peeling bark"] > 0` | 82,374 | 40% of baseline (32,673) | 1% of baseline (953) | 34.3x | 3% |
| Bird.communicate.perch-branch | `vtk["stat_perch branch"] > 0` AND `vtk["forest_size"] in senescing\|snag\|artificial` | 122,088 | 61% of baseline (74,398) | 0% of baseline (0) | INF | 0% |
| Bird.reproduce.hollow | `vtk["stat_hollow"] > 0` | 1,910 | 53% of baseline (1,019) | 0% of baseline (0) | INF | 0% |
| Lizard.acquire.grass | `vtk["search_bioavailable"] == "low-vegetation"` | 186,215 | 56% of baseline (105,200) | 0% of baseline (39) | 2697.4x | 0% |
| Lizard.acquire.dead-branch | `vtk["stat_dead branch"] > 0` | 201,376 | 48% of baseline (97,547) | 0% of baseline (946) | 103.1x | 1% |
| Lizard.acquire.epiphyte | `vtk["stat_epiphyte"] > 0` | 984 | 70% of baseline (692) | 0% of baseline (0) | INF | 0% |
| **Lizard.acquire** | **union** | **371,021** | **50% of baseline (184,523)** | **0% of baseline (985)** | **187.3x** | **1%** |
| Lizard.communicate.not-paved | `vtk["search_bioavailable"] in low-vegetation\|open space` AND NOT `vtk["search_urban_elements"] in roadway\|busy roadway\|parking` | 186,215 | 76% of baseline (140,640) | 45% of baseline (83,259) | 1.7x | 59% |
| Lizard.reproduce.nurse-log | `vtk["stat_fallen log"] > 0` | 140,242 | 22% of baseline (30,525) | 0% of baseline (0) | INF | 0% |
| Lizard.reproduce.fallen-tree | `vtk["forest_size"] in fallen\|decayed` | 24,144 | 71% of baseline (17,215) | 0% of baseline (0) | INF | 0% |
| **Lizard.reproduce** | **union** | **164,386** | **29% of baseline (47,250)** | **0% of baseline (0)** | **INF** | **0%** |
| Tree.acquire.moderated | `vtk["proposal_release_controlV4_intervention"] == "reduce-pruning"` | 0 | n/a (4,225) | n/a (0) | INF | 0% |
| Tree.acquire.autonomous | `vtk["proposal_release_controlV4_intervention"] == "eliminate-pruning"` | 663,120 | 17% of baseline (112,680) | 0% of baseline (0) | INF | 0% |
| **Tree.acquire** | **union** | **663,120** | **18% of baseline (116,905)** | **0% of baseline (0)** | **INF** | **0%** |
| Tree.communicate.snag | `vtk["forest_size"] == "snag"` | 19,747 | 137% of baseline (27,115) | 0% of baseline (0) | INF | 0% |
| Tree.communicate.fallen | `vtk["forest_size"] == "fallen"` | 16,915 | 97% of baseline (16,343) | 0% of baseline (0) | INF | 0% |
| Tree.communicate.decayed | `vtk["forest_size"] == "decayed"` | 7,229 | 12% of baseline (872) | 0% of baseline (0) | INF | 0% |
| **Tree.communicate** | **`vtk["forest_size"] in snag\|fallen\|decayed`** | **43,891** | **101% of baseline (44,330)** | **0% of baseline (0)** | **INF** | **0%** |
| Tree.reproduce.smaller-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "buffer-feature"` | 0 | n/a (13,309) | n/a (0) | INF | 0% |
| Tree.reproduce.larger-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "rewild-ground"` | 14,462 | 303% of baseline (43,765) | 0% of baseline (39) | 1122.2x | 0% |
| **Tree.reproduce** | **union** | **186,215** | **31% of baseline (57,074)** | **0% of baseline (39)** | **1463.4x** | **0%** |

### city

| indicator | measure | baseline | positive yr180 | trending yr180 | positive / trending | trending % of positive |
|---|---|---|---|---|---|---|
| Bird.acquire.peeling-bark | `vtk["stat_peeling bark"] > 0` | 80,911 | 19% of baseline (15,548) | 1% of baseline (940) | 16.5x | 6% |
| Bird.communicate.perch-branch | `vtk["stat_perch branch"] > 0` AND `vtk["forest_size"] in senescing\|snag\|artificial` | 122,061 | 26% of baseline (32,205) | 0% of baseline (0) | INF | 0% |
| Bird.reproduce.hollow | `vtk["stat_hollow"] > 0` | 1,910 | 26% of baseline (498) | 1% of baseline (10) | 49.8x | 2% |
| Lizard.acquire.grass | `vtk["search_bioavailable"] == "low-vegetation"` | 197,061 | 153% of baseline (301,464) | 17% of baseline (32,936) | 9.2x | 11% |
| Lizard.acquire.dead-branch | `vtk["stat_dead branch"] > 0` | 197,455 | 23% of baseline (46,142) | 1% of baseline (1,526) | 30.2x | 3% |
| Lizard.acquire.epiphyte | `vtk["stat_epiphyte"] > 0` | 984 | 34% of baseline (332) | 0% of baseline (0) | INF | 0% |
| **Lizard.acquire** | **union** | **377,939** | **89% of baseline (337,988)** | **9% of baseline (34,448)** | **9.8x** | **10%** |
| Lizard.communicate.not-paved | `vtk["search_bioavailable"] in low-vegetation\|open space` AND NOT `vtk["search_urban_elements"] in roadway\|busy roadway\|parking` | 197,061 | 147% of baseline (290,284) | 20% of baseline (39,651) | 7.3x | 14% |
| Lizard.reproduce.nurse-log | `vtk["stat_fallen log"] > 0` | 139,281 | 28% of baseline (38,618) | 4% of baseline (5,509) | 7.0x | 14% |
| Lizard.reproduce.fallen-tree | `vtk["forest_size"] in fallen\|decayed` | 24,198 | 130% of baseline (31,451) | 20% of baseline (4,957) | 6.3x | 16% |
| **Lizard.reproduce** | **union** | **163,479** | **29% of baseline (47,341)** | **3% of baseline (5,509)** | **8.6x** | **12%** |
| Tree.acquire.moderated | `vtk["proposal_release_controlV4_intervention"] == "reduce-pruning"` | 0 | n/a (9,834) | n/a (0) | INF | 0% |
| Tree.acquire.autonomous | `vtk["proposal_release_controlV4_intervention"] == "eliminate-pruning"` | 657,660 | 10% of baseline (63,083) | 1% of baseline (6,025) | 10.5x | 10% |
| **Tree.acquire** | **union** | **657,660** | **11% of baseline (72,917)** | **1% of baseline (6,025)** | **12.1x** | **8%** |
| Tree.communicate.snag | `vtk["forest_size"] == "snag"` | 20,895 | 64% of baseline (13,378) | 0% of baseline (0) | INF | 0% |
| Tree.communicate.fallen | `vtk["forest_size"] == "fallen"` | 16,915 | 179% of baseline (30,351) | 29% of baseline (4,957) | 6.1x | 16% |
| Tree.communicate.decayed | `vtk["forest_size"] == "decayed"` | 7,283 | 15% of baseline (1,100) | 0% of baseline (0) | INF | 0% |
| **Tree.communicate** | **`vtk["forest_size"] in snag\|fallen\|decayed`** | **45,093** | **99% of baseline (44,829)** | **11% of baseline (4,957)** | **9.0x** | **11%** |
| Tree.reproduce.smaller-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "buffer-feature"` | 0 | n/a (18,951) | n/a (0) | INF | 0% |
| Tree.reproduce.larger-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "rewild-ground"` | 15,968 | 145% of baseline (23,081) | 25% of baseline (3,923) | 5.9x | 17% |
| **Tree.reproduce** | **union** | **197,061** | **21% of baseline (42,032)** | **2% of baseline (3,923)** | **10.7x** | **9%** |

### uni

| indicator | measure | baseline | positive yr180 | trending yr180 | positive / trending | trending % of positive |
|---|---|---|---|---|---|---|
| Bird.acquire.peeling-bark | `vtk["stat_peeling bark"] > 0` | 80,077 | 21% of baseline (16,648) | 1% of baseline (783) | 21.3x | 5% |
| Bird.communicate.perch-branch | `vtk["stat_perch branch"] > 0` AND `vtk["forest_size"] in senescing\|snag\|artificial` | 121,527 | 34% of baseline (41,715) | 1% of baseline (962) | 43.4x | 2% |
| Bird.reproduce.hollow | `vtk["stat_hollow"] > 0` | 1,910 | 46% of baseline (880) | 1% of baseline (15) | 58.7x | 2% |
| Lizard.acquire.grass | `vtk["search_bioavailable"] == "low-vegetation"` | 192,748 | 118% of baseline (228,059) | 3% of baseline (5,195) | 43.9x | 2% |
| Lizard.acquire.dead-branch | `vtk["stat_dead branch"] > 0` | 195,305 | 32% of baseline (62,234) | 1% of baseline (1,341) | 46.4x | 2% |
| Lizard.acquire.epiphyte | `vtk["stat_epiphyte"] > 0` | 984 | 63% of baseline (622) | 1% of baseline (11) | 56.5x | 2% |
| **Lizard.acquire** | **union** | **371,493** | **76% of baseline (281,828)** | **2% of baseline (6,531)** | **43.2x** | **2%** |
| Lizard.communicate.not-paved | `vtk["search_bioavailable"] in low-vegetation\|open space` AND NOT `vtk["search_urban_elements"] in roadway\|busy roadway\|parking` | 192,748 | 134% of baseline (258,131) | 36% of baseline (70,046) | 3.7x | 27% |
| Lizard.reproduce.nurse-log | `vtk["stat_fallen log"] > 0` | 139,635 | 8% of baseline (11,161) | 0% of baseline (414) | 27.0x | 4% |
| Lizard.reproduce.fallen-tree | `vtk["forest_size"] in fallen\|decayed` | 24,198 | 22% of baseline (5,209) | 0% of baseline (0) | INF | 0% |
| **Lizard.reproduce** | **union** | **163,833** | **10% of baseline (16,317)** | **0% of baseline (414)** | **39.4x** | **3%** |
| Tree.acquire.moderated | `vtk["proposal_release_controlV4_intervention"] == "reduce-pruning"` | 0 | n/a (7,084) | n/a (0) | INF | 0% |
| Tree.acquire.autonomous | `vtk["proposal_release_controlV4_intervention"] == "eliminate-pruning"` | 651,761 | 6% of baseline (38,069) | 0% of baseline (2,156) | 17.7x | 6% |
| **Tree.acquire** | **union** | **651,761** | **7% of baseline (45,153)** | **0% of baseline (2,156)** | **20.9x** | **5%** |
| Tree.communicate.snag | `vtk["forest_size"] == "snag"` | 20,800 | 51% of baseline (10,633) | 2% of baseline (427) | 24.9x | 4% |
| Tree.communicate.fallen | `vtk["forest_size"] == "fallen"` | 16,915 | 30% of baseline (5,116) | 0% of baseline (0) | INF | 0% |
| Tree.communicate.decayed | `vtk["forest_size"] == "decayed"` | 7,283 | 1% of baseline (93) | 0% of baseline (0) | INF | 0% |
| **Tree.communicate** | **`vtk["forest_size"] in snag\|fallen\|decayed`** | **44,998** | **35% of baseline (15,842)** | **1% of baseline (427)** | **37.1x** | **3%** |
| Tree.reproduce.smaller-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "buffer-feature"` | 0 | n/a (8,172) | n/a (0) | INF | 0% |
| Tree.reproduce.larger-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "rewild-ground"` | 15,067 | 291% of baseline (43,781) | 32% of baseline (4,804) | 9.1x | 11% |
| **Tree.reproduce** | **union** | **192,748** | **27% of baseline (51,953)** | **2% of baseline (4,804)** | **10.8x** | **9%** |

---

## Other Comparison Methods

The simulation pipeline has two additional comparison systems that operate at the **voxel level** rather than the node level. Neither has been run yet against the v4-proposal-broadcast root.

### Capability Indicator Counts

Source: `sim/generate_vtk_and_nodeDFs/a_info_gather_capabilities.py`

Generated by: `run_full_v3_batch.py --compile-stats-only`

Reads the `state_with_indicators.vtk` files and counts voxels matching each of 12 capability indicators across three personas:

| persona | capability | indicator | what it counts |
|---|---|---|---|
| Bird | self | Bird.self.peeling | peeling bark voxels |
| Bird | others | Bird.others.perch | perchable canopy voxels |
| Bird | generations | Bird.generations.hollow | hollow voxels |
| Lizard | self | Lizard.self.grass | ground cover voxels |
| Lizard | self | Lizard.self.dead | dead branch voxels |
| Lizard | self | Lizard.self.epiphyte | epiphyte voxels |
| Lizard | others | Lizard.others.notpaved | non-paved surface voxels |
| Lizard | generations | Lizard.generations.nurse-log | nurse log voxels |
| Lizard | generations | Lizard.generations.fallen-tree | fallen tree voxels |
| Tree | self | Tree.self.senescent | late-life tree + deadwood voxels |
| Tree | others | Tree.others.notpaved | soil near canopy features |
| Tree | generations | Tree.generations.grassland | grassland for recruitment |

Per-state output: `{root}/output/stats/per-state/{site}/{site}_{scenario}_{voxel}_yr{year}_indicator_counts.csv`

Comparison metric: `pct_of_baseline` (count as percentage of the baseline count for the same indicator).

Action breakdown output: `{root}/output/stats/per-state/{site}/{site}_{scenario}_{voxel}_yr{year}_action_counts.csv`

Breaks each indicator count into sub-counts by `control_level` (high/medium/low), `artificial` (installed), and `urban_element` type.

### Cross-Scenario Proposal and Intervention Comparisons

Source: `outputs/report/a_info_proposal_interventions.py`

Compares **positive vs trending** side-by-side for each site/year, for:

**Proposals** (opportunity counts):
- deploy_structure: utility poles + artificial canopy voxels
- decay: trees reaching senescence + decay opportunity voxels
- recruit: recruitable ground voxels
- colonise: colonisable surface voxels
- release_control: arboreal voxels

**Interventions** (support action counts):
- Decay: Buffer-Feature (full), Brace-Feature (partial)
- Recruit: Buffer-Feature (partial), Rewild-Ground (full)
- Colonise: Rewild-Ground, Enrich-Envelope (full), Roughen-Envelope (partial)
- Release-Control: Eliminate-Pruning (full), Reduce-Pruning (partial)
- Deploy-Structure: Adapt-Utility-Pole, Upgrade-Feature

Comparison columns: `positive_value`, `trending_value`, `delta_trending_minus_positive`, `trending_pct_of_positive`, `positive_multiple_of_trending`

Output: `_statistics-refactored-v3/comparison/proposals.csv`, `_statistics-refactored-v3/comparison/interventions.csv`

Note: these outputs are currently from a prior v3 run, not the v4-proposal-broadcast root.
