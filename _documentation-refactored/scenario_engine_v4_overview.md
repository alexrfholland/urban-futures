# Scenario Engine V4 Overview

This note is the shortest route into the current v4 stack.

Use it first, then follow the linked documents for detail.

Branch: `engine-v4` (forked from `engine-v3` at `65f1691`)

## Core Links

- run instructions: `_code-refactored/refactor_code/sim/run/v4-run-instructions.md`
- indicator definitions: `_code-refactored/refactor_code/sim/v4_indicator_definitions.md`
- indicator extraction script: `_code-refactored/refactor_code/sim/v4_indicator_extract.py`
- indicator comparisons format: `_code-refactored/refactor_code/sim/v4 comparisons.md`
- visualisation instructions: `_code-refactored/refactor_code/v4visualisation-instructions.md`
- recruit and mortality notes: `_documentation-refactored/appendix/appendix-g/recruit and mortality notes.md`
- restructure handoff: `_documentation-refactored/v4 restructure handoff.md`
- restructure detail: `_documentation-refactored/v4 restructure.md`
- v3 overview (predecessor): `_documentation-refactored/scenario_engine_v3_overview.md`
- v3 validation: `_documentation-refactored/scenario_engine_v3_validation.md`

## Entry Point

```
_code-refactored/refactor_code/sim/run/run_full_v3_batch.py
```

All pipeline steps use this single script with different flags. The script name still says v3 but it is the v4 runtime.

## Sites, Scenarios, Years

- **Sites**: `trimmed-parade`, `city`, `uni`
- **Scenarios**: `positive`, `trending`
- **Years**: `0, 1, 10, 30, 60, 90, 120, 150, 180` (yr 1 is included in v4)

## Settings

Required:

- `REFACTOR_RUN_OUTPUT_ROOT` — set to the run root path (e.g. `_data-refactored/model-outputs/generated-states/v4-myrun`)

Not required (uses defaults):

- `TREE_TEMPLATE_ROOT` — defaults to the approved template root
- `EXPORT_ALL_POINTDATA_VARIABLES` — optional

Do not set `SIM_OUTPUT_ROOT` — that variable does not exist.

All commands must be run from `_code-refactored/` (or prefixed with `cd _code-refactored &&`) so that `refactor_code` is importable.

## Flow

The pipeline is run in sequential steps using the batch runner with different flags:

1. **`--node-only`** — core simulation: writes interim `treeDF`, `logDF`, `poleDF` per site/scenario/year
2. **`--vtk-only`** — loads saved CSVs, builds polydata, enriches with indicators/proposals, writes `state_with_indicators.vtk` and integrated `nodeDF`
3. **`--baselines-only`** or **`--regenerate-baselines`** — generates baseline VTKs and nodeDFs
4. **`--compile-stats-only`** — reads final VTKs, writes per-state indicator and action stats CSVs, writes merged site-level stats
5. **Render proposals** — `render_proposal_v4.py` (hybrid views with bottom legend)
6. **Render debug recruit** — `render_debug_recruit.py` (10 diagnostic layers)
7. **V4 indicator extraction** — `v4_indicator_extract.py` (yr 180 comparison tables)

Use `--multiple-agent` for per-slice parallel VTK execution (skips cross-state capability pass so another agent can compile it later).

See `_code-refactored/refactor_code/sim/run/v4-run-instructions.md` for full step-by-step commands.

## Where Files Live

Run root: `_data-refactored/model-outputs/generated-states/<root-name>`

Set via `REFACTOR_RUN_OUTPUT_ROOT` env var. When not set, falls back to the last logged root in `_data-refactored/run_log.csv`.

```
{root}/
  temp/
    interim-data/{site}/
      {site}_{scenario}_1_treeDF_{year}.csv
      {site}_{scenario}_1_logDF_{year}.csv
      {site}_{scenario}_1_poleDF_{year}.csv
      {site}_{scenario}_recruit_telemetry.csv
      {site}_{scenario}_recruit_stats.csv
      {site}_{scenario}_size_stats.csv
    validation/
      v3_full_run_metadata_{stamp}.json
      renders/
        {site}_{scenario}_yr{year}_proposal-and-interventions_hybrid_with-legend.png
      renders/debugRecruit/
        {site}_{scenario}_yr{year}_{layer}_with-legend.png
  output/
    vtks/{site}/
      {site}_{scenario}_1_yr{year}_state_with_indicators.vtk
      {site}_baseline_1_state_with_indicators.vtk
    feature-locations/{site}/
      {site}_{scenario}_1_nodeDF_yr{year}.csv
    stats/
      per-state/{site}/
        {site}_{scenario}_1_yr{year}_indicator_counts.csv
        {site}_{scenario}_1_yr{year}_action_counts.csv
        {site}_baseline_1_indicator_counts.csv
        {site}_baseline_1_action_counts.csv
      csv/
        {site}_indicator_counts.csv
        {site}_action_counts.csv
  comparison/
    v4_indicator_comparison.md
```

## Proposals and Interventions

The core focus of the simulation is to assess changes as **proposals**. Each proposal represents a category of ecological change that could occur at a site. Whether a proposal receives support depends on **effort thresholds** set by the scenario pathway (`positive` or `trending`). For each proposal, the pathway determines whether the site provides **full support**, **partial support**, or **rejects** the proposal entirely.

Each supported proposal maps to a specific **intervention** — the concrete action taken. Full and partial support map to different intervention types:

| Proposal | Full support intervention | Partial support intervention |
|---|---|---|
| `decay` | `buffer-feature` | `brace-feature` |
| `recruit` | `rewild-larger-patch` | `rewild-smaller-patch` |
| `release_control` | `eliminate-canopy-pruning` | `reduce-canopy-pruning` |
| `deploy_structure` | `adapt-utility-pole` / `upgrade-feature` | — |
| `colonise` | `larger-patches-rewild` / `enrich-envelope` | `roughen-envelope` |

Intervention label constants are centralised in `sim/setup/constants.py`.

V4 proposal arrays are broadcast to the VTK as point-data fields (each has a proposal flag and an intervention label):

- `proposal_decayV4` / `proposal_decayV4_intervention`
- `proposal_release_controlV4` / `proposal_release_controlV4_intervention`
- `proposal_recruitV4` / `proposal_recruitV4_intervention`
- `proposal_coloniseV4` / `proposal_coloniseV4_intervention`
- `proposal_deploy_structureV4` / `proposal_deploy_structureV4_intervention`

These replace the v3 proposal arrays.

#### Recruitment Mechanism to Proposal Mapping

The three spatial recruitment mechanisms map to full or partial recruit interventions:

| `recruit_mechanism` | Proposal intervention | Support level | Zone mask |
|---|---|---|---|
| `node-rewild` | `rewild-larger-patch` | Full | `scenario_nodeRewildRecruitZone` |
| `under-canopy` | `rewild-smaller-patch` | Partial | `scenario_underCanopyRecruitZone` |
| `ground` | `rewild-larger-patch` | Full | `scenario_rewildGroundRecruitZone` |

The support level determines the mortality rate for saplings generated by each mechanism (see Mortality below). Indicator queries use `proposal_recruitV4_intervention`, not `recruit_mechanism` directly.

## Parameters

Per-site/scenario parameters are in `_code-refactored/refactor_code/sim/setup/params_v3.py`.

### Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| `growth_model` | `split` | Precolonial trees use `fischer`, colonial use `ulmus` | `split` is the default.
| `growth_factor_range` | `[0.37, 0.51]` | Applied to growth curve DBH increments |
| `mortality_model` | `flat` | Le Roux cohort-thinning with DBH-dependent shaping |
| `annual_tree_death_urban` | `0.06` | Urban mortality anchor. Used for under-canopy recruits and other trees |
| `annual_tree_death_nature-reserves` | `0.03` | Reserve mortality anchor. Used for node-rewild and ground recruits |
| `plantingDensity` | `50` | Trees per hectare (pulse-scaled) |
| `sim_TurnsThreshold` | per-year dict | Controls how much ground opens for recruitment. Higher = more ground |
| `ground_filter_mode` | `node-exclusion` | Ground zone minus any voxels already covered by node-rewild or under-canopy zones |
| `senescing_duration_years` | `[10, 90, 200]` | Triangular(min, mode, max) |
| `snag_duration_years` | `[0, 40, 100]` | Mode changed from 50 to 40 in v4 |
| `fallen_duration_years` | `[10, 40, 100]` | Triangular |
| `decayed_duration_years` | `[30, 40, 75]` | Triangular |
| `lifecycle_senescing_ramp_start` | `-25` or `-5` | Years before useful-life-end when decay risk starts |

### sim_TurnsThreshold

Year-keyed dictionary controlling which ground voxels are eligible for rewilding. Example (trimmed-parade / positive):

```
{0: 0, 10: 300, 30: 1249.75, 60: 4000, 180: 5000}
```

Higher threshold = more ground available. Interpolated between keyed years via `get_interpolated_param()`.

### Pruning Thresholds

Release-control interventions are assigned based on `CanopyResistance` values:

| Threshold | positive | trending |
|---|---|---|
| `minimal-tree-support-threshold` | 70 | 10 |
| `moderate-tree-support-threshold` | 50 | 10 |
| `maximum-tree-support-threshold` | 10 | 10 |

Trees above the maximum threshold get `eliminate-canopy-pruning`; between moderate and maximum get `reduce-canopy-pruning`.

## Simulation Model

### Two Calculation Layers

The model works through two linked calculation layers:

**Path-dependent node calculations** happen per pulse on the node dataframes (`tree_df`, `log_df`, `pole_df`). These change which habitat features exist, how they are modified by proposal and intervention decisions, and how these changes carry through time. All information carried to the next pulse or state is recorded in the dataframes.

**Possibility-space calculations** happen on the possibility-space xarray dataset (`possibility_space_ds`). This is a prebaked 3D voxel grid that classifies where proposals and interventions can spread, connect, or be read spatially. The dataset can be used as a temporary spatial layer during a pulse (e.g. for computing recruitment zones) and is reset per pulse. It is also used at VTK generation time to broadcast node-level decisions onto voxels.

The node ID systems that link these two layers:

| ID field | What it links | Where it lives |
|---|---|---|
| `NodeID` / `analysis_nodeID` | Each node's physical voxel position | Shared across dataframes and xarray |
| `node_CanopyID` | Under-canopy voxels to the tree they belong to | xarray — covers ~435 of 455 trees |
| `sim_Nodes` | Ground voxels to the nearest node via growth simulation | xarray — includes trees, logs, poles |
| `sim_Turns` | How many growth turns to reach each voxel | xarray — proxy for spatial difficulty |

### Under-Node Treatment Regions

Each tree can be assigned an under-node treatment level based on its `CanopyResistance` and the pathway's effort thresholds. These treatment levels determine the spatial extent of the tree's influence:

| Treatment | Spatial extent | ID system used |
|---|---|---|
| `exoskeleton` | Under-canopy voxels linked to the tree | `node_CanopyID` |
| `footprint-depaved` | Under-canopy voxels linked to the tree (same extent as exoskeleton) | `node_CanopyID` |
| `node-rewilded` | Larger simulated growth region linked to the node | `sim_Nodes` |

Treatment assignment is driven by effort thresholds (see Parameters) — lower `CanopyResistance` means the tree can support higher-effort interventions.

### Pulse-Based Execution

The engine splits each assessed-year gap into 10-year pulses. Within each pulse, the simulation first runs ecological processes, then evaluates each proposal in turn — determining whether the pathway supports it and which intervention applies.

**Processes:**

1. **Growth** — `age_trees()`: ages trees using species-specific allometric curves with +/-15% individual variation in effective growing time
2. **Mortality** — `apply_annual_tree_mortality()`: stochastic Le Roux cohort-based mortality applied to small, medium, and large trees

**Proposals (evaluated in order on the node dataframes):**

#### Proposal-Decay

Decides whether each tree begins its senescing pathway based on how close it is to the end of its useful life.

1. **Decision** — `determine_proposal_decay()`: calculates a probability ramp from `useful_life_expectancy` and `lifecycle_senescing_ramp_start`. As useful life approaches 0, the probability of decay acceptance increases. If the tree's `CanopyResistance` is below `minimal-tree-support-threshold`, the proposal is accepted; otherwise it is rejected and the tree is replaced with a sapling.
2. **Intervention** — `assign_decay_interventions()`: for accepted trees, places `CanopyResistance` into ordered effort bands to assign the under-node treatment and intervention:
   - `CanopyResistance <= maximum-tree-support-threshold` -> `node-rewilded` -> full support (`buffer-feature`)
   - `<= moderate-tree-support-threshold` -> `footprint-depaved` -> full support (`buffer-feature`)
   - `<= minimal-tree-support-threshold` -> `exoskeleton` -> partial support (`brace-feature`)
3. **Lifecycle changes** — `apply_proposal_decay_accepted_lifecycle_changes()`: transitions accepted trees through senescing -> snag -> fallen -> decayed, each with probability ramps and seeded persistence durations.

#### Proposal-Release-Control

Decides whether to reduce or eliminate canopy pruning for each living tree.

1. **Decision** — `apply_release_control()`: tests each tree's `CanopyResistance` against the pathway's pruning thresholds. If below `moderate-tree-support-threshold`, the proposal is accepted.
2. **Intervention**:
   - `CanopyResistance <= maximum-tree-support-threshold` -> full support (`eliminate-canopy-pruning`)
   - `<= moderate-tree-support-threshold` -> partial support (`reduce-canopy-pruning`)
3. **Effect**: over time, pruning release causes the tree's `control_reached` to shift from street-tree/park-tree toward low-control (after ~40 years under eliminate-pruning).

The trending pathway sets its thresholds so low that almost no trees qualify for release control.

#### Proposal-Recruit

Places new saplings in spatial zones. Operates in 30-year recruitment cycles.

1. **Zone computation** — `calculate_under_node_treatment_status()` computes three spatial zone masks on the possibility-space dataset (see Recruitment below for detail on each zone).
2. **Allocation** — `apply_recruit()` places saplings in each zone. Full support (`rewild-larger-patch`) for node-rewild and ground zones; partial support (`rewild-smaller-patch`) for under-canopy zones.
3. **Mortality consequence** — the support level determines the sapling's mortality rate: full support gets reserve rate (0.03), partial gets urban rate (0.06).

#### Proposals Computed at VTK Generation Time

These proposals are not evaluated per-pulse. They are assigned when the node-level results are broadcast onto the possibility-space voxels during VTK generation (`a_scenario_generateVTKs.py`):

- **Deploy structure** — poles and logs enabled based on `sim_averageResistance` thresholds. Interventions: `adapt-utility-pole`, `upgrade-feature`, `translocate-log`.
- **Colonise** — assigned from surface type. The model builds `scenario_bioEnvelope` from under-node treatments, then extends it using `bioMask` (from `sim_Turns <= sim_TurnsThreshold` and `sim_averageResistance`) to enable `otherGround`, `livingFacade`, `greenRoof`, `brownRoof`. Interventions: `larger-patches-rewild`, `enrich-envelope`, `roughen-envelope`.

### Growth Model

| Model | Applies to | Curve |
|---|---|---|
| `fischer` | Precolonial trees (pre-European species) | Fischer allometric DBH-to-height |
| `ulmus` | Colonial trees (European species) | Ulmus-calibrated allometric curve |

The `split` growth model switches between these based on whether the tree's species is classified as precolonial or colonial. Growth factor range `[0.37, 0.51]` scales the effective DBH increment per pulse.

### Engine Constants

| Constant | Value |
|---|---|
| `ENGINE_PULSE_INTERVAL` | 10 years |
| `RECRUIT_INTERVAL` | 30 years |
| `RECRUIT_SPACING_THRESHOLD_METERS` | 1.5 m (primary), 0.5 m (relaxed fallback) |
| `DISTANCE_THRESHOLD_METERS` | 5.0 m |
| `BUFFER_RECRUIT_PARENT_OFFSET_METERS` | varies |

### Tree Size Categories

- **Living**: small, medium, large
- **Senescing pathway**: senescing -> snag -> fallen -> decayed
- **Absent**: gone, early-tree-death
- **Non-occupying** (no canopy voxels): snag, fallen, decayed, plus absent

### Mortality

Mortality uses Le Roux cohort-thinning curves anchored to two annual rates:

- **Urban**: `0.06` (roughly 61% of trees survive over 8 years)
- **Nature-reserve**: `0.03` (roughly 78% of trees survive over 8 years)

All trees by default have the urban rate. The rate assignment follows the proposal support schema:

- **Full support** recruit intervention (`rewild-larger-patch`) — saplings generated in larger rewilded patches (node-rewild and ground recruit mechanisms) are assigned the **reserve** rate, reflecting increased thriving in larger patches.
- **Partial support** recruit intervention (`rewild-smaller-patch`) — saplings generated under individual tree canopies (under-canopy recruit mechanism) keep the **urban** rate, reflecting smaller patches where urban conditions persist.

This means the mortality rate is a consequence of the proposal support level, not an arbitrary per-mechanism assignment. See Proposals and Interventions below for the full schema.

The raw annual rate is shaped by DBH cohort using survival factors:

**Urban cohort factors** (anchored to 0.06):

| DBH cohort | Shaped factor | Annual mortality |
|---|---|---|
| 0-10 | 1.000 | 0.060 |
| 10-20 | 0.750 | 0.045 |
| 20-30 | 0.583 | 0.035 |
| 30-40 | 0.500 | 0.030 |
| 40-50 | 0.433 | 0.026 |
| 50-60 | 0.350 | 0.021 |
| 60-70 | 0.250 | 0.015 |
| 70-80 | 0.167 | 0.010 |

**Reserve cohort factors** (anchored to 0.03):

| DBH cohort | Shaped factor | Annual mortality |
|---|---|---|
| 0-10 | 1.000 | 0.030 |
| 10-20 | 0.800 | 0.024 |
| 20-30 | 0.633 | 0.019 |
| 30-40 | 0.500 | 0.015 |
| 40-50 | 0.400 | 0.012 |
| 50-60 | 0.300 | 0.009 |
| 60-70 | 0.233 | 0.007 |
| 70-80 | 0.167 | 0.005 |

V4 applies mortality to **small, medium, and large** trees (v3 applied to small and medium only).

Full mortality derivation: `_documentation-refactored/appendix/appendix-g/recruit and mortality notes.md`

### Recruitment

The engine supports three distinct recruitment mechanisms, each with its own zone mask, allocation logic, and mortality rate:

| `recruit_mechanism` | Zone mask | Allocation source | Mortality |
|---|---|---|---|
| `node-rewild` | `scenario_nodeRewildRecruitZone` | Ground voxels mapped via `sim_Nodes` to node-rewilded tree NodeIDs | Reserve (0.03) |
| `under-canopy` | `scenario_underCanopyRecruitZone` | Canopy voxels mapped via `node_CanopyID` to footprint-depaved/exoskeleton tree NodeIDs | Urban (0.06) |
| `ground` | `scenario_rewildGroundRecruitZone` | Voxels where `sim_Turns <= sim_TurnsThreshold`, filtered by `ground_filter_mode` | Reserve (0.03) |

All three zone masks are computed together in `calculate_under_node_treatment_status()` (in `engine_v3.py`). Convention: `>= 0` means active (value is the year enabled), `-1` means inactive.

Masks are computed twice:
1. Per pulse during simulation (on `pulse_ds`)
2. At VTK generation time (on a fresh `ds` copy) — same function handles both

#### Two ID Systems

- **`sim_Nodes`** — maps ground voxels to the nearest node via a growth simulation (see `a_rewilding.py`). Includes trees, logs, and poles. Only ~30 tree NodeIDs appear (most ground is "owned" by logs/poles). Used for node-rewild zone.
- **`node_CanopyID`** — maps canopy voxels to the tree they belong to. Covers ~435 of 455 trees. Used for under-canopy zone.

#### sim_Nodes and sim_Turns

Both come from the same upstream growth simulation (`a_rewilding.py` -> `grow_plants()`). Each node (tree/log/pole) spreads outward from its physical voxel location with an energy budget, respecting resistance barriers.

- `sim_Nodes` = which node "conquered" each voxel (used for node-rewild zone)
- `sim_Turns` = how many growth turns it took to reach each voxel (used for ground zone threshold)

**Known limitation:** `sim_Nodes` maps most ground to logs/poles (which start on the ground) rather than trees (which start from canopy height). Only ~30 of ~455 tree NodeIDs get ground voxels. This makes `scenario_nodeRewildRecruitZone` very sparse. Use the `sim_nodes_zones` debug render layer to visualise.

#### Allocation Logic

- **Node-rewild / under-canopy**: Primary method is voxel-mask placement (random sample from zone voxels with spacing check). Fallback is parent-offset placement (offset from parent tree position by `BUFFER_RECRUIT_PARENT_OFFSET_METERS`).
- **Ground**: Random shuffle of available ground zone voxels with spacing check.
- Spacing threshold: `1.5m` primary, `0.5m` relaxed fallback.

#### Ground Filter Mode

Controlled by `params["ground_filter_mode"]` (default: `node-exclusion`):

- **`node-exclusion`**: Ground zone minus any voxels already covered by node-rewild or under-canopy zones
- **`proximity`**: Legacy mode — ground filtered by distance from existing trees

#### Recruit Density

Current setting: `50 trees/ha`, pulse-scaled:

- 10-year pulse = 16.7 trees/ha
- 20-year pulse = 33.3 trees/ha
- 30-year pulse = 50 trees/ha

Recruitment only fires every `RECRUIT_INTERVAL` (30 years).

#### Recruit Telemetry

Each pulse writes per-type rows to `{site}_{scenario}_recruit_telemetry.csv` with fields:
`site, scenario, year, previous_year, pulse_years, type, quota, placed, occupancy, zone_voxel_count, density_per_pulse, filled, fallback_used, fallback_count`

After all years complete, `log_run_stats()` writes:
- `{site}_{scenario}_recruit_stats.csv` — per-year summary (quota, placed, occupancy, density/pulse per type)
- `{site}_{scenario}_size_stats.csv` — per-year size-class counts

## V4 Indicators

V4 indicators use the `acquire`/`communicate`/`reproduce` capability naming. Full definitions in `_code-refactored/refactor_code/sim/v4_indicator_definitions.md`.

### Bird

| ID | Capability | VTK query |
|---|---|---|
| Bird.acquire.peeling-bark | acquire | `stat_peeling bark > 0` |
| Bird.communicate.perch-branch | communicate | `stat_perch branch > 0` AND `forest_size in senescing\|snag\|artificial` |
| Bird.reproduce.hollow | reproduce | `stat_hollow > 0` |

### Lizard

| ID | Capability | VTK query |
|---|---|---|
| Lizard.acquire.grass | acquire | `search_bioavailable == low-vegetation` |
| Lizard.acquire.dead-branch | acquire | `stat_dead branch > 0` |
| Lizard.acquire.epiphyte | acquire | `stat_epiphyte > 0` |
| **Lizard.acquire** | **acquire** | **Union of grass + dead-branch + epiphyte** |
| Lizard.communicate.not-paved | communicate | `ground_not_paved` (not roadway/parking) |
| Lizard.reproduce.nurse-log | reproduce | `stat_fallen log > 0` |
| Lizard.reproduce.fallen-tree | reproduce | `forest_size in fallen\|decayed` |
| **Lizard.reproduce** | **reproduce** | **Union of nurse-log + fallen-tree** |

### Tree

| ID | Capability | VTK query |
|---|---|---|
| Tree.acquire.moderated | acquire | `proposal_release_controlV4_intervention == "reduce-canopy-pruning"` |
| Tree.acquire.autonomous | acquire | `proposal_release_controlV4_intervention == "eliminate-canopy-pruning"` |
| **Tree.acquire** | **acquire** | **Union of moderated + autonomous** |
| Tree.communicate.snag | communicate | `forest_size == "snag"` |
| Tree.communicate.fallen | communicate | `forest_size == "fallen"` |
| Tree.communicate.decayed | communicate | `forest_size == "decayed"` |
| **Tree.communicate** | **communicate** | **`forest_size in snag\|fallen\|decayed`** |
| Tree.reproduce.smaller-patches-rewild | reproduce | `proposal_recruitV4_intervention == "rewild-smaller-patch"` |
| Tree.reproduce.larger-patches-rewild | reproduce | `proposal_recruitV4_intervention == "rewild-larger-patch"` |
| **Tree.reproduce** | **reproduce** | **Union of smaller + larger patches** |

### Key Differences From V3 Indicators

- **Bird.communicate**: Added `forest_size in senescing|snag|artificial` filter on top of `stat_perch branch > 0`
- **Tree.acquire**: Proposal-derived (`proposal_release_controlV4_intervention`) instead of forest-size-based
- **Tree.communicate**: Drops `senescing` from v3 `Tree.self.senescent`
- **Tree.reproduce**: Proposal-derived (`proposal_recruitV4_intervention`) instead of spatial grassland query. Baseline uses `indicator_Tree_generations_grassland` as fallback
- **Lizard.acquire / Lizard.reproduce**: Aggregate unions of existing sub-indicators

### V3 Indicators (still computed)

The v3 indicator set (`Bird.self.peeling`, `Lizard.others.notpaved`, `Tree.self.senescent`, etc.) is still computed by `a_info_gather_capabilities.py` and written to VTKs. V4 indicators are extracted separately by `v4_indicator_extract.py`.

### Support Action Breakdowns

Each v3 indicator also has a support action breakdown:

| Indicator | Breakdown type | Also count |
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

Control levels: high (street-tree), medium (park-tree), low (reserve-tree, improved-tree).

Urban elements: open space, green roof, brown roof, facade, roadway, busy roadway, existing conversion, other street potential, parking, none.

Rewilding types: footprint-depaved, exoskeleton, node-rewilded, none.

## Statistics and Comparisons

### Node-Level Size Statistics

Count trees by `size` column from treeDFs at each year. Report deltas against the woodland baseline.

Size classes: `small`, `medium`, `large`, `senescing`, `snag`, `fallen`, `decayed`

Format: `count (+/-delta from baseline)` per size class per year, with a bold total row.

Sources:
- Baseline: `{root}/temp/interim-data/{site}/{site}_baseline_1_nodeDF_-180.csv`
- Scenario: `{root}/temp/interim-data/{site}/{site}_{scenario}_1_treeDF_{year}.csv`

These are generated by `log_run_stats()` after Step 1 completes, written to `{root}/temp/interim-data/{site}/`.

### Capability Indicator Counts (voxel-level)

Generated by `--compile-stats-only` (reads `state_with_indicators.vtk` files).

Counts voxels matching each of 12 v3 capability indicators across three personas (Bird, Lizard, Tree). Requires both scenario VTKs and baseline VTKs to exist.

Per-state output: `{root}/output/stats/per-state/{site}/{site}_{scenario}_1_yr{year}_indicator_counts.csv`

Action breakdown output: `{root}/output/stats/per-state/{site}/{site}_{scenario}_1_yr{year}_action_counts.csv`

Merged site-level output: `{root}/output/stats/csv/{site}_indicator_counts.csv` and `{site}_action_counts.csv`

Comparison metric: `pct_of_baseline` (count as percentage of the baseline count for the same indicator).

### V4 Indicator Comparisons

Generated by `v4_indicator_extract.py`. Reads yr 180 VTKs for baseline + positive + trending per site. Computes all V4 indicators and writes a markdown comparison table.

Output: `{root}/comparison/v4_indicator_comparison.md`

Columns: `indicator`, `measure`, `baseline`, `positive yr180`, `trending yr180`, `positive / trending`, `trending % of positive`.

The comparison format shows each indicator as a percentage of baseline and as multiples between scenarios.

### Comparison Workflow

1. Node-level size stats (Step 3 in run instructions) — can run during VTK generation, needs only treeDFs
2. Indicator + action stats (`--compile-stats-only`) — needs scenario + baseline VTKs
3. V4 indicator extraction (`v4_indicator_extract.py`) — needs scenario + baseline VTKs

## Visualisation

### Proposal Renderer

Script: `_code-refactored/refactor_code/outputs/report/render_proposal_v4.py`

Renders hybrid proposal views from `state_with_indicators.vtk` files. The hybrid view colours the decay pathway (senescing, snag, fallen, decayed) by forest_size colour and only applies proposal overlays on non-decay voxels. Output is a PNG with an integrated bottom legend.

Shared layout parameters (always pass these):
- `--model-base-y 1170` — aligns the bottom of all models to the same vertical position
- `--target-model-width 1757` — normalises apparent model size across sites

Output filename: `{site}_{scenario}_yr{year}_proposal-and-interventions_hybrid_with-legend.png`

Full render set: 3 sites x (2 scenarios x 9 years + 1 baseline) = **57 images**

### Debug Recruit Renderer

Script: `_code-refactored/refactor_code/outputs/report/render_debug_recruit.py`

Renders one image per recruit diagnostic variable. White background, point rendering with lighting.

10 layers total:

| Layer | Type | What it shows |
|---|---|---|
| `recruit_isNewTree` | categorical | New trees (green) vs original (grey) |
| `recruit_hasbeenReplanted` | categorical | Replanted (blue) vs not (grey) |
| `recruit_mechanism` | categorical | node-rewild (orange), under-canopy (purple), ground (green) |
| `recruit_year` | numeric ramp | Blue (early) -> red (late) |
| `recruit_mortality_rate` | numeric ramp | Yellow (low) -> dark red (high) |
| `recruit_mortality_cohort` | discrete | Green (small DBH) -> red (large DBH) |
| `ground_recruitment` | composite | Recruitable ground (green) + ground-recruited canopies (orange) |
| `node_rewild_recruitment` | composite | Node-rewild zone via sim_Nodes (blue) + node-rewild canopies (red) |
| `under_canopy_recruitment` | composite | Under-canopy zone via node_CanopyID (lavender) + under-canopy canopies (magenta) |
| `sim_nodes_zones` | composite | Each sim_Nodes zone coloured uniquely |

Output filename: `{site}_{scenario}_yr{year}_{layer}_with-legend.png` in `debugRecruit/`

Full instructions: `_code-refactored/refactor_code/v4visualisation-instructions.md`

## Run Log

Persistent CSV at `_data-refactored/run_log.csv` (columns: `timestamp`, `name`, `output_root`, `description`).

Appended by the batch runner on every run. When `REFACTOR_RUN_OUTPUT_ROOT` is not set, `refactor_code.paths.refactor_run_output_root()` falls back to the last logged output root.

Inspect recent runs:

```bash
uv run python _code-refactored/refactor_code/sim/run/run_log.py
```

Pass `--description "my note"` to the batch runner to add a description to the log entry.

## Core Runtime Modules

All paths are relative to `_code-refactored/refactor_code/`.

| Module | Purpose |
|---|---|
| `sim/run/run_full_v3_batch.py` | Batch entrypoint |
| `sim/run/run_log.py` | Persistent run log |
| `sim/setup/a_scenario_initialiseDS.py` | Dataset and source-data preparation |
| `sim/setup/params_v3.py` | Parameters and timesteps |
| `sim/setup/constants.py` | Intervention label constants |
| `sim/generate_interim_state_data/engine_v3.py` | Core simulation logic (growth, mortality, recruitment, proposals) |
| `sim/generate_interim_state_data/a_scenario_runscenario.py` | Scenario runner around the engine |
| `sim/generate_vtk_and_nodeDFs/a_scenario_generateVTKs.py` | nodeDF generation, base polydata build, proposal broadcast |
| `sim/generate_vtk_and_nodeDFs/a_scenario_urban_elements_count.py` | Urban-feature enrichment |
| `sim/generate_vtk_and_nodeDFs/a_info_gather_capabilities.py` | Indicator/proposal enrichment, per-state and merged stats |
| `sim/baseline/baseline_v3.py` | Baseline generation core |
| `sim/v4_indicator_extract.py` | V4 indicator extraction and comparison tables |
| `sim/v4_indicator_definitions.md` | V4 indicator definitions |
| `paths.py` | Unified run-root routing with run log fallback |
| `outputs/report/render_proposal_v4.py` | Hybrid proposal renderer |
| `outputs/report/render_debug_recruit.py` | Debug recruit diagnostic renderer |

### Code Layout

```
_code-refactored/refactor_code/
  sim/
    run/                              — batch runner, run log
    setup/                            — params, constants, dataset init
    generate_interim_state_data/      — engine, scenario runner
    generate_vtk_and_nodeDFs/         — VTK build, indicators, stats
    baseline/                         — baseline generation
    voxel/                            — voxeliser, helpers, site coordinates
  input_processing/
    tree_processing/                  — tree resource distribution
  outputs/
    report/                           — renderers (proposal, debug recruit, size views)
    stats/                            — statistics output helpers
  blender/
    bexport/                          — Blender export prep, proposal framebuffers
    blenderv2/                        — Blender v2 runtime
  paths.py                            — unified path routing
```

## Upstream Data

The simulation reads from pre-built voxel datasets:

- `data/revised/final/{site}/{site}_1_subsetForScenarios.nc` — cached subset dataset (rebuilt automatically if missing)
- `data/revised/final/{site}/{site}_1_voxelArray_Nodes.nc` — full dataset with `sim_Nodes` / `sim_Turns` from the rewilding growth simulation

These are built by the voxelization pipeline (`final/a_manager.py` -> `a_rewilding.py`) and are not regenerated during normal sim runs.

## Outputs

### Production-ready

- Final `state_with_indicators.vtk` — `{root}/output/vtks/{site}/`
- Final integrated `nodeDF` — `{root}/output/feature-locations/{site}/`

### Statistics

- Per-state indicator counts: `{root}/output/stats/per-state/{site}/`
- Per-state action counts: `{root}/output/stats/per-state/{site}/`
- Merged site-level stats: `{root}/output/stats/csv/`
- V4 indicator comparison: `{root}/comparison/v4_indicator_comparison.md`
- Recruit telemetry: `{root}/temp/interim-data/{site}/{site}_{scenario}_recruit_telemetry.csv`
- Recruit stats: `{root}/temp/interim-data/{site}/{site}_{scenario}_recruit_stats.csv`
- Size stats: `{root}/temp/interim-data/{site}/{site}_{scenario}_size_stats.csv`

### Validation / Visualisation

- Proposal renders: `{root}/temp/validation/renders/`
- Debug recruit renders: `{root}/temp/validation/renders/debugRecruit/`
- Run metadata: `{root}/temp/validation/v3_full_run_metadata_{stamp}.json`

### Expected Counts (full run)

- 54 scenario `state_with_indicators.vtk` (3 sites x 2 scenarios x 9 years)
- 3 baseline `state_with_indicators.vtk` (3 sites)
- 54 integrated `nodeDF` CSVs
- 57 proposal render PNGs (3 sites x (2 scenarios x 9 years + 1 baseline))

## Troubleshooting

### PermissionError on `.nc` file

The subset cache (`{site}_1_subsetForScenarios.nc`) can become 0 bytes if two processes write simultaneously. Fix:

```bash
rm data/revised/final/<site>/<site>_1_subsetForScenarios.nc
```

Then rerun. The next run rebuilds it from the full voxel array. Do not run VTK/baselines in parallel if the cache does not already exist.

### Legacy root env vars

The batch runner rejects `REFACTOR_SCENARIO_OUTPUT_ROOT`, `REFACTOR_ENGINE_OUTPUT_ROOT`, and `REFACTOR_STATISTICS_ROOT`. Unset them and use `REFACTOR_RUN_OUTPUT_ROOT` only.

---

## Changelog: V3 to V4

### Recruitment

- V3 had two recruit types (`buffer-feature`, `rewild-ground`). V4 has three (`node-rewild`, `under-canopy`, `ground`) with specific `recruit_mechanism` values.
- Three spatial zone masks (`scenario_nodeRewildRecruitZone`, `scenario_underCanopyRecruitZone`, `scenario_rewildGroundRecruitZone`) computed centrally in `calculate_under_node_treatment_status()`.
- Recruit telemetry CSV with per-pulse zone/density/occupancy tracking.
- `Tree.reproduce` is now proposal-derived (`proposal_recruitV4_intervention`) instead of spatial grassland query.

### Mortality

- Extended to `large` trees (v3: small and medium only).
- Snag duration mode changed from 50 to 40.

### Proposals

- V4 proposal arrays (`proposal_decayV4`, `proposal_release_controlV4`, `proposal_recruitV4`, `proposal_coloniseV4`, `proposal_deploy_structureV4`) replace V3 versions.
- Intervention label constants centralised in `sim/setup/constants.py`.

### Indicators

- V4 indicators use `acquire`/`communicate`/`reproduce` capability naming.
- `Bird.communicate` adds `forest_size in senescing|snag|artificial` filter.
- `Tree.acquire` now proposal-derived instead of forest-size-based.
- `Tree.communicate` drops `senescing` from v3.
- Aggregate indicators (`Lizard.acquire`, `Lizard.reproduce`, `Tree.acquire`, `Tree.communicate`, `Tree.reproduce`) are unions of sub-indicators.
- `v4_indicator_extract.py` standalone script for yr 180 comparison tables.

### Growth

- Growth model now `split` (precolonial -> `fischer`, colonial -> `ulmus`) with allometric DBH curves.
- Individual growth variation (+/-15%) applied to effective growing time.

### Infrastructure

- Persistent run log at `_data-refactored/run_log.csv` with fallback in `paths.py`.
- `paths.py` `refactor_run_output_root()` auto-resolves from run log when env var not set.
- Debug recruit renderer with 10 diagnostic layers (6 per-variable + 3 composite zone + 1 sim_nodes_zones).
- Proposal renderer updated to v4 hybrid views (`render_proposal_v4.py`).

### Code Layout

- Active sim code reorganised under `sim/run`, `sim/setup`, `sim/generate_interim_state_data`, `sim/generate_vtk_and_nodeDFs`, `sim/baseline`, `sim/voxel`.
- Legacy sim helpers moved out of `final/` into `sim/voxel/`.
- Tree processing moved to `input_processing/tree_processing/`.
- Blender export prep moved to `blender/bexport/`.
- Imports updated so the live sim runtime no longer depends on `final/` for active execution.

### Years

- Year 1 added to the default timestep set: `[0, 1, 10, 30, 60, 90, 120, 150, 180]`.
