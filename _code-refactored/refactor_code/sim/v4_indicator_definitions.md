# V4 Indicator Definitions

## Current V3 Indicators

These are the indicators currently defined in `a_info_gather_capabilities.py`.

### Bird

| ID | Capability | Label | Query |
|---|---|---|---|
| Bird.self.peeling | self | Peeling bark volume | `stat_peeling bark > 0` |
| Bird.others.perch | others | Perchable canopy volume | `stat_perch branch > 0` |
| Bird.generations.hollow | generations | Hollow count | `stat_hollow > 0` |

### Lizard

| ID | Capability | Label | Query |
|---|---|---|---|
| Lizard.self.grass | self | Ground cover area | `search_bioavailable == low-vegetation` |
| Lizard.self.dead | self | Dead branch volume | `stat_dead branch > 0` |
| Lizard.self.epiphyte | self | Epiphyte count | `stat_epiphyte > 0` |
| Lizard.others.notpaved | others | Non-paved surface area | `ground_not_paved` |
| Lizard.generations.nurse-log | generations | Nurse log volume | `stat_fallen log > 0` |
| Lizard.generations.fallen-tree | generations | Fallen tree volume | `forest_size in fallen\|decayed` |

### Tree

| ID | Capability | Label | Query | Spatial filter |
|---|---|---|---|---|
| Tree.self.senescent | self | Late-life tree and deadwood volume | `forest_size in senescing\|snag\|fallen\|decayed` | — |
| Tree.others.notpaved | others | Soil near canopy features | `ground_not_paved` | within 50m of canopy-feature, ground_only |
| Tree.generations.grassland | generations | Grassland for recruitment | `search_bioavailable == low-vegetation` | within 20m of canopy-feature, ground_only |

### Support Action Breakdowns

Each indicator also has a support action breakdown type:

| Indicator | Breakdown | Also count |
|---|---|---|
| Bird.self.peeling | control_level | artificial |
| Bird.others.perch | control_level | — |
| Bird.generations.hollow | control_level | artificial |
| Lizard.self.grass | urban_element | — |
| Lizard.self.dead | control_level | — |
| Lizard.self.epiphyte | control_level | artificial |
| Lizard.others.notpaved | urban_element | — |
| Lizard.generations.nurse-log | urban_element | — |
| Lizard.generations.fallen-tree | urban_element | — |
| Tree.self.senescent | rewilding_status | — |
| Tree.others.notpaved | urban_element | — |
| Tree.generations.grassland | urban_element | — |

### Reference Constants

**Control levels:**
- high: street-tree
- medium: park-tree
- low: reserve-tree, improved-tree

**Urban elements:** open space, green roof, brown roof, facade, roadway, busy roadway, existing conversion, other street potential, parking, none

**Rewilding types:** footprint-depaved, exoskeleton, node-rewilded, none

---

## V4 Indicators

V4 indicators use the `acquire`/`communicate`/`reproduce` capability naming.

| ID | Persona | Capability | VTK query | Notes |
|---|---|---|---|---|
| Bird.acquire.peeling-bark | Bird | acquire | `stat_peeling bark > 0` | Same as V3 `Bird.self.peeling` |
| Bird.communicate.perch-branch | Bird | communicate | `stat_perch branch > 0` AND `forest_size in senescing\|snag\|artificial` | **Updated from V3** — added forest_size filter |
| Bird.reproduce.hollow | Bird | reproduce | `stat_hollow > 0` | Same as V3 `Bird.generations.hollow` |
| Lizard.acquire.grass | Lizard | acquire | `search_bioavailable == low-vegetation` | |
| Lizard.acquire.dead-branch | Lizard | acquire | `stat_dead branch > 0` | |
| Lizard.acquire.epiphyte | Lizard | acquire | `stat_epiphyte > 0` | |
| Lizard.acquire | Lizard | acquire | Union of grass + dead-branch + epiphyte | Aggregate |
| Lizard.communicate.not-paved | Lizard | communicate | `ground_not_paved` | Same as V3 `Lizard.others.notpaved` |
| Lizard.reproduce.nurse-log | Lizard | reproduce | `stat_fallen log > 0` | |
| Lizard.reproduce.fallen-tree | Lizard | reproduce | `forest_size in fallen\|decayed` | |
| Lizard.reproduce | Lizard | reproduce | Union of nurse-log + fallen-tree | Aggregate |
| Tree.acquire.moderated | Tree | acquire | `proposal_release_controlV4_intervention == "reduce-canopy-pruning"` | Park trees with reduced pruning |
| Tree.acquire.autonomous | Tree | acquire | `proposal_release_controlV4_intervention == "eliminate-canopy-pruning"` | Reserve/improved trees with pruning eliminated |
| Tree.acquire | Tree | acquire | Union of moderated + autonomous | Total autonomous growth |
| Tree.communicate.snag | Tree | communicate | `forest_size == "snag"` | |
| Tree.communicate.fallen | Tree | communicate | `forest_size == "fallen"` | |
| Tree.communicate.decayed | Tree | communicate | `forest_size == "decayed"` | |
| Tree.communicate | Tree | communicate | `forest_size in snag\|fallen\|decayed` | Drops `senescing` from V3 |
| Tree.reproduce.smaller-patches-rewild | Tree | reproduce | `proposal_recruitV4_intervention == "rewild-smaller-patch"` | node-rewilded, footprint-depaved |
| Tree.reproduce.larger-patches-rewild | Tree | reproduce | `proposal_recruitV4_intervention == "rewild-larger-patch"` | otherground, rewilded |
| Tree.reproduce | Tree | reproduce | Union of smaller + larger patches | Baseline borrowed from `indicator_Tree_generations_grassland` |

> **TODO:** Baseline recruit also includes saplings — consider whether to do this or not.

### Key Differences from V3

- **Bird.communicate**: Added `forest_size in senescing|snag|artificial` filter on top of `stat_perch branch > 0`
- **Tree.acquire**: Proposal-derived (`proposal_release_controlV4_intervention`) instead of forest_size-based
- **Tree.communicate**: Drops `senescing` from V3 `Tree.self.senescent`
- **Tree.reproduce**: Proposal-derived (`proposal_recruitV4_intervention`) instead of spatial grassland query. Baseline uses `indicator_Tree_generations_grassland` since baselines have no `scenario_rewilded`
- **Lizard.acquire / Lizard.reproduce**: Unions of existing V3 sub-indicators
