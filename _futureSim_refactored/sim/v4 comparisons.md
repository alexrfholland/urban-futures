# V4 Comparisons

Data root: `_data-refactored/model-outputs/generated-states/v4updatedterms`

Baselines: `{root}/temp/interim-data/{site}/{site}_baseline_1_nodeDF_-180.csv`

Scenario nodeDFs: `{root}/temp/interim-data/{site}/{site}_{scenario}_1_nodeDF_{year}.csv`

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
| small | 2014 | 160 (-1854) | 572 (-1442) | 550 (-1464) | 514 (-1500) | 511 (-1503) | 511 (-1503) | 512 (-1502) | 516 (-1498) |
| medium | 255 | 235 (-20) | 143 (-112) | 185 (-70) | 227 (-28) | 234 (-21) | 238 (-17) | 236 (-19) | 236 (-19) |
| large | 56 | 61 (+5) | 112 (+56) | 94 (+38) | 67 (+11) | 2 (-54) | 2 (-54) | 11 (-45) | 10 (-46) |
| senescing | 104 | 0 (-104) | 3 (-101) | 66 (-38) | 102 (-2) | 151 (+47) | 125 (+21) | 81 (-23) | 58 (-46) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 1 (-47) | 16 (-32) | 38 (-10) | 70 (+22) | 70 (+22) |
| fallen | 33 | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 1 (-32) | 5 (-28) | 12 (-21) | 28 (-5) |
| decayed | 38 | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 1 (-37) | 2 (-36) | 8 (-30) |
| **total** | **2548** | **456 (-2092)** | **830 (-1718)** | **895 (-1653)** | **911 (-1637)** | **915 (-1633)** | **920 (-1628)** | **924 (-1624)** | **926 (-1622)** |

### trimmed-parade / trending

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2014 | 160 (-1854) | 220 (-1794) | 302 (-1712) | 292 (-1722) | 325 (-1689) | 322 (-1692) | 324 (-1690) | 322 (-1692) |
| medium | 255 | 235 (-20) | 139 (-116) | 78 (-177) | 107 (-148) | 131 (-124) | 131 (-124) | 131 (-124) | 131 (-124) |
| large | 56 | 61 (+5) | 97 (+41) | 76 (+20) | 57 (+1) | 0 (-56) | 3 (-53) | 1 (-55) | 3 (-53) |
| senescing | 104 | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) |
| fallen | 33 | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) |
| decayed | 38 | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) | 0 (-38) |
| **total** | **2548** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** | **456 (-2092)** |

### city / positive

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2015 | 117 (-1898) | 390 (-1625) | 355 (-1660) | 352 (-1663) | 345 (-1670) | 343 (-1672) | 342 (-1673) | 342 (-1673) |
| medium | 255 | 154 (-101) | 123 (-132) | 120 (-135) | 149 (-106) | 162 (-93) | 161 (-94) | 162 (-93) | 160 (-95) |
| large | 56 | 3 (-53) | 8 (-48) | 47 (-9) | 26 (-30) | 0 (-56) | 4 (-52) | 5 (-51) | 8 (-48) |
| senescing | 104 | 0 (-104) | 3 (-101) | 19 (-85) | 45 (-59) | 68 (-36) | 57 (-47) | 35 (-69) | 23 (-81) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 2 (-46) | 13 (-35) | 35 (-13) | 36 (-12) |
| fallen | 33 | 0 (-33) | 241 (+208) | 475 (+442) | 759 (+726) | 764 (+731) | 774 (+741) | 780 (+747) | 834 (+801) |
| decayed | 34 | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) |
| **total** | **2545** | **274 (-2271)** | **765 (-1780)** | **1016 (-1529)** | **1331 (-1214)** | **1341 (-1204)** | **1352 (-1193)** | **1359 (-1186)** | **1403 (-1142)** |

### city / trending

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2015 | 117 (-1898) | 153 (-1862) | 191 (-1824) | 206 (-1809) | 198 (-1817) | 198 (-1817) | 197 (-1818) | 198 (-1817) |
| medium | 255 | 154 (-101) | 116 (-139) | 44 (-211) | 67 (-188) | 82 (-173) | 82 (-173) | 83 (-172) | 83 (-172) |
| large | 56 | 3 (-53) | 8 (-48) | 42 (-14) | 5 (-51) | 0 (-56) | 1 (-55) | 2 (-54) | 2 (-54) |
| senescing | 104 | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) | 0 (-104) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) |
| fallen | 33 | 0 (-33) | 0 (-33) | 19 (-14) | 64 (+31) | 94 (+61) | 120 (+87) | 144 (+111) | 193 (+160) |
| decayed | 34 | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) |
| **total** | **2545** | **274 (-2271)** | **277 (-2268)** | **296 (-2249)** | **342 (-2203)** | **374 (-2171)** | **401 (-2144)** | **426 (-2119)** | **476 (-2069)** |

### uni / positive

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2015 | 102 (-1913) | 223 (-1792) | 202 (-1813) | 203 (-1812) | 198 (-1817) | 200 (-1815) | 201 (-1814) | 227 (-1788) |
| medium | 255 | 83 (-172) | 67 (-188) | 72 (-183) | 84 (-171) | 95 (-160) | 96 (-159) | 99 (-156) | 100 (-155) |
| large | 56 | 2 (-54) | 9 (-47) | 22 (-34) | 18 (-38) | 0 (-56) | 2 (-54) | 3 (-53) | 5 (-51) |
| senescing | 104 | 0 (-104) | 2 (-102) | 16 (-88) | 27 (-77) | 40 (-64) | 32 (-72) | 20 (-84) | 13 (-91) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 2 (-46) | 9 (-39) | 22 (-26) | 23 (-25) |
| fallen | 33 | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 1 (-32) | 2 (-31) | 6 (-27) |
| decayed | 34 | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) |
| **total** | **2545** | **187 (-2358)** | **362 (-2183)** | **373 (-2172)** | **422 (-2123)** | **425 (-2120)** | **430 (-2115)** | **437 (-2108)** | **464 (-2081)** |

### uni / trending

| size | baseline | yr 0 | yr 10 | yr 30 | yr 60 | yr 90 | yr 120 | yr 150 | yr 180 |
|---|---|---|---|---|---|---|---|---|---|
| small | 2015 | 102 (-1913) | 120 (-1895) | 130 (-1885) | 135 (-1880) | 132 (-1883) | 132 (-1883) | 133 (-1882) | 133 (-1882) |
| medium | 255 | 83 (-172) | 60 (-195) | 36 (-219) | 46 (-209) | 54 (-201) | 54 (-201) | 54 (-201) | 57 (-198) |
| large | 56 | 2 (-54) | 7 (-49) | 21 (-35) | 5 (-51) | 0 (-56) | 1 (-55) | 2 (-54) | 1 (-55) |
| senescing | 104 | 0 (-104) | 0 (-104) | 0 (-104) | 1 (-103) | 1 (-103) | 1 (-103) | 0 (-104) | 0 (-104) |
| snag | 48 | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 0 (-48) | 1 (-47) | 1 (-47) |
| fallen | 33 | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) | 0 (-33) |
| decayed | 34 | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) | 0 (-34) |
| **total** | **2545** | **187 (-2358)** | **187 (-2358)** | **187 (-2358)** | **187 (-2358)** | **187 (-2358)** | **188 (-2357)** | **190 (-2355)** | **192 (-2353)** |

---

## V4 Indicator Comparisons (voxel counts, yr 180)

Indicator definitions from `v4_indicator_definitions.md`. Baseline uses `indicator_Tree_generations_grassland` for `Tree.reproduce` aggregate (no scenario_rewilded on baselines). Tree.acquire.autonomous baseline reflects the full woodland arboreal volume under no management (all trees are "eliminate-canopy-pruning" equivalent). Tree.acquire.moderated baseline is 0 (no park trees in a woodland).

\* trending was 0; substituted ~1% of baseline for comparison columns

### trimmed-parade

| indicator | measure | baseline | positive yr180 | trending yr180 | positive / trending | trending % of positive |
|---|---|---|---|---|---|---|
| Bird.acquire.peeling-bark | `vtk["stat_peeling bark"] > 0` | 82,374 | 42% of baseline (34,710) | 9% of baseline (7,071) | 4.9x | 20% |
| Bird.communicate.perch-branch | `vtk["stat_perch branch"] > 0` AND `vtk["forest_size"] in senescing\|snag\|artificial` | 122,088 | 48% of baseline (58,001) | ~1% of baseline (~1,221)* | 47.5x* | 2%* |
| Bird.reproduce.hollow | `vtk["stat_hollow"] > 0` | 1,910 | 63% of baseline (1,207) | 7% of baseline (131) | 9.2x | 11% |
| Lizard.acquire.grass | `vtk["search_bioavailable"] == "low-vegetation"` | 186,215 | 52% of baseline (97,236) | 0% of baseline (39) | 2493.2x | 0% |
| Lizard.acquire.dead-branch | `vtk["stat_dead branch"] > 0` | 201,376 | 51% of baseline (102,831) | 8% of baseline (15,623) | 6.6x | 15% |
| Lizard.acquire.epiphyte | `vtk["stat_epiphyte"] > 0` | 984 | 61% of baseline (602) | ~1% of baseline (~10)* | 60.2x* | 2%* |
| **Lizard.acquire** | **union** | **371,021** | **50% of baseline (186,737)** | **4% of baseline (15,662)** | **11.9x** | **8%** |
| Lizard.communicate.not-paved | `vtk["search_bioavailable"] in low-vegetation\|open space` AND NOT `vtk["search_urban_elements"] in roadway\|busy roadway\|parking` | 186,215 | 72% of baseline (133,474) | 45% of baseline (83,259) | 1.6x | 62% |
| Lizard.reproduce.nurse-log | `vtk["stat_fallen log"] > 0` | 140,242 | 18% of baseline (25,553) | ~1% of baseline (~1,402)* | 18.2x* | 5%* |
| Lizard.reproduce.fallen-tree | `vtk["forest_size"] in fallen\|decayed` | 24,144 | 51% of baseline (12,304) | ~1% of baseline (~241)* | 51.1x* | 2%* |
| **Lizard.reproduce** | **union** | **164,386** | **23% of baseline (37,487)** | **~1% of baseline (~1,644)*** | **22.8x*** | **4%*** |
| Tree.acquire.moderated | `vtk["proposal_release_controlV4_intervention"] == "reduce-canopy-pruning"` | 0 | n/a (11,554) | n/a (0) | INF | 0% |
| Tree.acquire.autonomous | `vtk["proposal_release_controlV4_intervention"] == "eliminate-canopy-pruning"` | 663,120 | 31% of baseline (208,214) | ~1% of baseline (~6,631)* | 31.4x* | 3%* |
| **Tree.acquire** | **union** | **663,120** | **33% of baseline (219,768)** | **~1% of baseline (~6,631)*** | **33.1x*** | **3%*** |
| Tree.communicate.snag | `vtk["forest_size"] == "snag"` | 19,747 | 120% of baseline (23,712) | ~1% of baseline (~197)* | 120.4x* | 1%* |
| Tree.communicate.fallen | `vtk["forest_size"] == "fallen"` | 16,915 | 66% of baseline (11,190) | ~1% of baseline (~169)* | 66.2x* | 2%* |
| Tree.communicate.decayed | `vtk["forest_size"] == "decayed"` | 7,229 | 15% of baseline (1,114) | ~1% of baseline (~72)* | 15.5x* | 6%* |
| **Tree.communicate** | **`vtk["forest_size"] in snag\|fallen\|decayed`** | **43,891** | **82% of baseline (36,016)** | **~1% of baseline (~439)*** | **82.0x*** | **1%*** |
| Tree.reproduce.smaller-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "rewild-smaller-patch"` | 0 | n/a (10,627) | n/a (0) | INF | 0% |
| Tree.reproduce.larger-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "rewild-larger-patch"` | 14,462 | 259% of baseline (37,478) | 0% of baseline (35) | 1070.8x | 0% |
| **Tree.reproduce** | **union** | **186,215** | **26% of baseline (48,105)** | **0% of baseline (35)** | **1374.4x** | **0%** |

### city

| indicator | measure | baseline | positive yr180 | trending yr180 | positive / trending | trending % of positive |
|---|---|---|---|---|---|---|
| Bird.acquire.peeling-bark | `vtk["stat_peeling bark"] > 0` | 80,911 | 22% of baseline (17,741) | 5% of baseline (4,133) | 4.3x | 23% |
| Bird.communicate.perch-branch | `vtk["stat_perch branch"] > 0` AND `vtk["forest_size"] in senescing\|snag\|artificial` | 122,061 | 18% of baseline (22,025) | ~1% of baseline (~1,221)* | 18.0x* | 6%* |
| Bird.reproduce.hollow | `vtk["stat_hollow"] > 0` | 1,910 | 32% of baseline (611) | 5% of baseline (91) | 6.7x | 15% |
| Lizard.acquire.grass | `vtk["search_bioavailable"] == "low-vegetation"` | 197,061 | 152% of baseline (298,674) | 17% of baseline (32,837) | 9.1x | 11% |
| Lizard.acquire.dead-branch | `vtk["stat_dead branch"] > 0` | 197,455 | 26% of baseline (51,587) | 5% of baseline (9,683) | 5.3x | 19% |
| Lizard.acquire.epiphyte | `vtk["stat_epiphyte"] > 0` | 984 | 29% of baseline (282) | 0% of baseline (4) | 70.5x | 1% |
| **Lizard.acquire** | **union** | **377,939** | **91% of baseline (342,373)** | **11% of baseline (42,465)** | **8.1x** | **12%** |
| Lizard.communicate.not-paved | `vtk["search_bioavailable"] in low-vegetation\|open space` AND NOT `vtk["search_urban_elements"] in roadway\|busy roadway\|parking` | 197,061 | 146% of baseline (287,527) | 20% of baseline (39,566) | 7.3x | 14% |
| Lizard.reproduce.nurse-log | `vtk["stat_fallen log"] > 0` | 139,281 | 28% of baseline (38,436) | 4% of baseline (5,484) | 7.0x | 14% |
| Lizard.reproduce.fallen-tree | `vtk["forest_size"] in fallen\|decayed` | 24,198 | 117% of baseline (28,291) | 20% of baseline (4,952) | 5.7x | 18% |
| **Lizard.reproduce** | **union** | **163,479** | **27% of baseline (44,027)** | **3% of baseline (5,484)** | **8.0x** | **12%** |
| Tree.acquire.moderated | `vtk["proposal_release_controlV4_intervention"] == "reduce-canopy-pruning"` | 0 | n/a (25,212) | n/a (0) | INF | 0% |
| Tree.acquire.autonomous | `vtk["proposal_release_controlV4_intervention"] == "eliminate-canopy-pruning"` | 657,660 | 18% of baseline (116,028) | 1% of baseline (4,906) | 23.7x | 4% |
| **Tree.acquire** | **union** | **657,660** | **21% of baseline (141,240)** | **1% of baseline (4,906)** | **28.8x** | **3%** |
| Tree.communicate.snag | `vtk["forest_size"] == "snag"` | 20,895 | 59% of baseline (12,403) | ~1% of baseline (~209)* | 59.3x* | 2%* |
| Tree.communicate.fallen | `vtk["forest_size"] == "fallen"` | 16,915 | 167% of baseline (28,291) | 29% of baseline (4,952) | 5.7x | 18% |
| Tree.communicate.decayed | `vtk["forest_size"] == "decayed"` | 7,283 | 0% of baseline (0) | ~1% of baseline (~73)* | 0.0x* | —* |
| **Tree.communicate** | **`vtk["forest_size"] in snag\|fallen\|decayed`** | **45,093** | **90% of baseline (40,694)** | **11% of baseline (4,952)** | **8.2x** | **12%** |
| Tree.reproduce.smaller-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "rewild-smaller-patch"` | 0 | n/a (15,493) | n/a (8) | 1936.6x | 0% |
| Tree.reproduce.larger-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "rewild-larger-patch"` | 15,968 | 132% of baseline (21,150) | 22% of baseline (3,590) | 5.9x | 17% |
| **Tree.reproduce** | **union** | **197,061** | **19% of baseline (36,643)** | **2% of baseline (3,598)** | **10.2x** | **10%** |

### uni

| indicator | measure | baseline | positive yr180 | trending yr180 | positive / trending | trending % of positive |
|---|---|---|---|---|---|---|
| Bird.acquire.peeling-bark | `vtk["stat_peeling bark"] > 0` | 80,077 | 23% of baseline (18,623) | 3% of baseline (2,615) | 7.1x | 14% |
| Bird.communicate.perch-branch | `vtk["stat_perch branch"] > 0` AND `vtk["forest_size"] in senescing\|snag\|artificial` | 121,527 | 29% of baseline (34,982) | 0% of baseline (154) | 227.2x | 0% |
| Bird.reproduce.hollow | `vtk["stat_hollow"] > 0` | 1,910 | 49% of baseline (943) | 3% of baseline (64) | 14.7x | 7% |
| Lizard.acquire.grass | `vtk["search_bioavailable"] == "low-vegetation"` | 192,748 | 117% of baseline (225,809) | 3% of baseline (5,076) | 44.5x | 2% |
| Lizard.acquire.dead-branch | `vtk["stat_dead branch"] > 0` | 195,305 | 33% of baseline (64,815) | 3% of baseline (6,335) | 10.2x | 10% |
| Lizard.acquire.epiphyte | `vtk["stat_epiphyte"] > 0` | 984 | 61% of baseline (598) | 0% of baseline (3) | 199.3x | 1% |
| **Lizard.acquire** | **union** | **371,493** | **77% of baseline (284,800)** | **3% of baseline (11,389)** | **25.0x** | **4%** |
| Lizard.communicate.not-paved | `vtk["search_bioavailable"] in low-vegetation\|open space` AND NOT `vtk["search_urban_elements"] in roadway\|busy roadway\|parking` | 192,748 | 133% of baseline (255,936) | 36% of baseline (69,922) | 3.7x | 27% |
| Lizard.reproduce.nurse-log | `vtk["stat_fallen log"] > 0` | 139,635 | 8% of baseline (11,404) | 0% of baseline (329) | 34.7x | 3% |
| Lizard.reproduce.fallen-tree | `vtk["forest_size"] in fallen\|decayed` | 24,198 | 10% of baseline (2,356) | ~1% of baseline (~242)* | 9.7x* | 10%* |
| **Lizard.reproduce** | **union** | **163,833** | **8% of baseline (13,727)** | **0% of baseline (329)** | **41.7x** | **2%** |
| Tree.acquire.moderated | `vtk["proposal_release_controlV4_intervention"] == "reduce-canopy-pruning"` | 0 | n/a (18,737) | n/a (0) | INF | 0% |
| Tree.acquire.autonomous | `vtk["proposal_release_controlV4_intervention"] == "eliminate-canopy-pruning"` | 651,761 | 12% of baseline (76,747) | 1% of baseline (5,137) | 14.9x | 7% |
| **Tree.acquire** | **union** | **651,761** | **15% of baseline (95,484)** | **1% of baseline (5,137)** | **18.6x** | **5%** |
| Tree.communicate.snag | `vtk["forest_size"] == "snag"` | 20,800 | 39% of baseline (8,131) | 1% of baseline (245) | 33.2x | 3% |
| Tree.communicate.fallen | `vtk["forest_size"] == "fallen"` | 16,915 | 14% of baseline (2,356) | ~1% of baseline (~169)* | 13.9x* | 7%* |
| Tree.communicate.decayed | `vtk["forest_size"] == "decayed"` | 7,283 | 0% of baseline (0) | ~1% of baseline (~73)* | 0.0x* | —* |
| **Tree.communicate** | **`vtk["forest_size"] in snag\|fallen\|decayed`** | **44,998** | **23% of baseline (10,487)** | **1% of baseline (245)** | **42.8x** | **2%** |
| Tree.reproduce.smaller-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "rewild-smaller-patch"` | 0 | n/a (6,590) | n/a (0) | INF | 0% |
| Tree.reproduce.larger-patches-rewild | `vtk["proposal_recruitV4_intervention"] == "rewild-larger-patch"` | 15,067 | 276% of baseline (41,580) | 29% of baseline (4,366) | 9.5x | 11% |
| **Tree.reproduce** | **union** | **192,748** | **25% of baseline (48,170)** | **2% of baseline (4,366)** | **11.0x** | **9%** |

---

## Other Comparison Methods

The simulation pipeline has two additional comparison systems that operate at the **voxel level** rather than the node level. Neither has been run yet against the v4updatedterms root.

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

Note: these outputs are currently from a prior v3 run, not the v4updatedterms root.
