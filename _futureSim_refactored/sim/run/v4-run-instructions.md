# V4 Run Instructions

These instructions are for agents and humans running the v4 simulation pipeline. Follow them in order. Do not skip steps. Do not run steps in parallel unless explicitly stated.

See also:

- `_documentation-refactored/scenario_engine_v4_overview.md` — v4 overview, architecture, what changed from v3, all linked docs

---

## TODO / Known Issues

- **`relabel_as_fallen_log` disabled in variant builder** (ticket TBD) — The 4 call sites in `_futureSim_refactored/input_processing/tree_processing/build_tree_variants.py` (`build_fallen_variant_rows` + `build_decayed_variant_rows`) are currently commented out. Fallen and decayed templates therefore retain their original resource labels (peeling bark, perch branch, epiphyte, other, etc.) rather than being flattened to `resource="fallen log"` with `stat_fallen log=1` and `resource_fallen log=1` on every point.

  **Why preserved**: distinguishes "small logs" (the `resource_fallen log` already set on standing-tree templates — small dead branches on the ground near the trunk) from "larger logs" (whole fallen/decayed trees, identified via `size=fallen` / `size=decayed`). Relabelling would merge the two and make it impossible to separate branch-scale logs from whole-tree logs via a single resource query.

  **Consequence for v4 indicators**: `Lizard.reproduce.nurse-log` (defined as `stat_fallen log > 0` in Step 4) currently matches only the pre-existing small-log points on tree templates, not the whole fallen/decayed bodies. Queries keyed on `size in fallen|decayed` (`Tree.communicate.fallen`, `Tree.communicate.decayed`, `Lizard.reproduce.fallen-tree`) still work because the `size`/`forest_size` columns are unaffected.

  **To re-enable**: uncomment the 4 call sites (search the file for `relabel_as_fallen_log`), rebuild the variant via `build_tree_variants.py`, and update the downstream indicator definitions to stop relying on the small-vs-large-log distinction.

---

## Entry Point

The batch runner script is:

```
_futureSim_refactored/sim/run/run_full_v3_batch.py
```

All pipeline steps use this single entry point with different flags (`--node-only`, `--vtk-only`, `--compile-stats-only`, `--baselines-only`, `--bioenvelope-only`, `--multiple-agent`).

---

## Before You Start

**ALWAYS ask the user for the output root name before running anything.** Do not assume or invent a root name. Every run needs a name that becomes the output directory. Example: `v4`, `v4-test`, `v4-snag-fix`. The user decides this — stop and ask if they have not provided one.

The full root path is always:

```
_data-refactored/model-outputs/generated-states/<root-name>
```

Set it with the environment variable `REFACTOR_RUN_OUTPUT_ROOT`:

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name>
```

Prefix every command with this variable, or export it once per session.

**All commands must be run from the repo root** (the directory containing `_futureSim_refactored/`). The `uv run python` commands use paths relative to that root.

**Do not set `TREE_TEMPLATE_ROOT`** — it uses defaults.

**Do not set `SIM_OUTPUT_ROOT`** — that variable does not exist and will be silently ignored.

---

## Sites, Scenarios, Years

- **Sites**: `trimmed-parade`, `city`, `uni`
- **Scenarios**: `positive`, `trending`
- **Years**: `0, 1, 10, 30, 60, 90, 120, 150, 180` (the default — yr 1 is included)

Years do not need to be passed explicitly unless you want a subset. The defaults include yr 1.

---

## Step 1: Node-only (treeDFs + nodeDFs)

Generates the simulation CSV outputs for all sites and scenarios. This must complete before anything else.

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py \
  --node-only
```

**Outputs**:

- `{root}/temp/interim-data/{site}/{site}_{scenario}_1_treeDF_{year}.csv` (and matching logDF, poleDF)
- `{root}/temp/interim-data/{site}/{site}_{scenario}_recruit_telemetry.csv`
- `{root}/temp/interim-data/{site}/{site}_{scenario}_recruit_stats.csv`
- `{root}/temp/interim-data/{site}/{site}_{scenario}_size_stats.csv`

**Verify**: Each site directory should have 18 treeDF files (9 years x 2 scenarios).

---

## Step 2: VTK generation

Reads the saved treeDFs/nodeDFs and builds VTK point clouds with indicators. This is the slowest step.

**ALWAYS use `--multiple-agent`** when generating VTKs. This skips the cross-state capability pass so slices can run in parallel. The capability pass is handled separately during `--compile-stats-only` (Step 5). Do not omit this flag.

### Parallel execution

Launch all slices for a site in parallel using bash `&` and `wait`. Each slice is one site/scenario/year combination. Batch **by site** (18 processes per site = 2 scenarios × 9 years) rather than all 54 at once — 54 simultaneous processes saturate CPU and memory, causing thrashing that makes the run slower overall. We have tested up to 18 parallel processes on a 10-core/64GB machine without issues.

```bash
export REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name>

# Run one site at a time (18 parallel processes each)
for site in trimmed-parade city uni; do
  for scenario in positive trending; do
    for year in 0 1 10 30 60 90 120 150 180; do
      uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py \
        --vtk-only --multiple-agent --sites "$site" --scenarios "$scenario" --years "$year" \
        > /tmp/vtk_${site}_${scenario}_${year}.log 2>&1 &
    done
  done
  wait  # finish this site before starting the next
done
```

These run inside a single shell — not as separate Claude Code subagents via the Agent tool.

For a single slice:

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py \
  --vtk-only --multiple-agent --sites <site> --scenarios <scenario> --years <year>
```

**Outputs per slice** (all produced inline from in-memory polydata):

- `{root}/output/vtks/{site}/{site}_{scenario}_1_yr{year}_state_with_indicators.vtk`
- `{root}/output/feature-locations/{site}/{site}_{scenario}_1_nodeDF_yr{year}.csv`
- `{root}/output/stats/per-state/{site}/{site}_{scenario}_1_yr{year}_v4_indicators.csv`
- `{root}/output/bioenvelopes/{site}/{site}_{scenario}_1_envelope_scenarioYR{year}.ply` (proposal-based; yr0 produces no output)
- `{root}/temp/validation/renders/{site}_{scenario}_yr{year}_proposal-and-interventions_with-legend.png`
- `{root}/temp/validation/renders/debugRecruit/{site}_{scenario}_yr{year}_*_with-legend.png` (years 10, 60, 180 only)

**Verify**: 54 scenario VTKs (3 sites x 2 scenarios x 9 years), 54 nodeDF CSVs, 54 V4 indicator CSVs, 48 bioenvelope PLYs (no yr0), 54 proposal renders, and 18 debug recruit sets (3 sites x 2 scenarios x 3 years).

### Bioenvelope generation

The VTK step generates bioenvelope PLY meshes inline from each in-memory state polydata. The default generator (`export_proposal_envelopes`) classifies voxels by proposal intervention priority, extracts per-category isosurfaces with single-voxel gap filling, merges them, and writes a PLY with `intervention_bioenvelope_ply-int` and per-family `blender_proposal-*` framebuffers.

To use the old `scenario_bioEnvelope`-based generator, pass `--old-envelopes`.

To regenerate bioenvelopes from existing VTKs without rerunning the full pipeline:

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py \
  --bioenvelope-only
```

This reads state VTKs from `{root}/output/vtks/` and writes PLYs to `{root}/output/bioenvelopes/`. Parallelise across sites the same way as VTK generation:

```bash
export REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name>

for site in trimmed-parade city uni; do
  for scenario in positive trending; do
    for year in 0 1 10 30 60 90 120 150 180; do
      uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py \
        --bioenvelope-only --sites "$site" --scenarios "$scenario" --years "$year" \
        > /tmp/bioenv_${site}_${scenario}_${year}.log 2>&1 &
    done
  done
  wait
done
```

Classification priority (highest to lowest):

| Priority | Label | Int code |
|---|---|---|
| 1 | buffer-feature+depaved (decay buffer on depaved ground) | 8 |
| 2 | rewild-larger-patch (recruit) | 7 |
| 3 | rewild-smaller-patch (recruit) | 6 |
| 4 | enrich-envelope (colonise) | 5 |
| 5 | roughen-envelope (colonise) | 4 |
| 6 | larger-patches-rewild (colonise) | 3 |
| 7 | buffer-feature (decay) | 2 |
| 8 | deploy-any | 1 |
| -- | none | 0 |

Only voxels with `scenario_bioEnvelope != 'none'` are eligible. Year 0 has no proposals and produces no PLY.

Colours and int codes are defined in `_futureSim_refactored/sim/setup/constants.py` (`BIOENVELOPE_PLY_COLORS`, `BIOENVELOPE_PLY_INT`).

---

## Step 3: Regenerate baselines

Generates woodland baseline VTKs and nodeDFs. Must run after Step 1 (node-only) because it uses the same cached `.nc` subset file.

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py \
  --baselines-only
```

**Outputs** (per site, all inline):

- `{root}/output/vtks/{site}/{site}_baseline_1_state_with_indicators.vtk`
- `{root}/temp/interim-data/{site}/{site}_baseline_1_nodeDF_-180.csv`
- `{root}/output/stats/per-state/{site}/{site}_baseline_1_v4_indicators.csv`

**Verify**: 3 baseline VTKs, 3 baseline V4 indicator CSVs.

---

## Step 4: V4 indicator comparisons

Extracts the V4 indicator set (using `acquire`/`communicate`/`reproduce` naming) from yr 180 VTKs and produces a markdown comparison table. Requires scenario + baseline VTKs from Steps 2 and 3.

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  uv run python _futureSim_refactored/sim/v4_indicator_extract.py
```

### V4 indicators extracted

| ID | VTK query |
|---|---|
| Bird.acquire.peeling-bark | `stat_peeling bark > 0` |
| Bird.communicate.perch-branch | `stat_perch branch > 0` AND `forest_size in senescing\|snag\|artificial` |
| Bird.reproduce.hollow | `stat_hollow > 0` |
| Lizard.acquire.grass | `search_bioavailable == low-vegetation` |
| Lizard.acquire.dead-branch | `stat_dead branch > 0` |
| Lizard.acquire.epiphyte | `stat_epiphyte > 0` |
| **Lizard.acquire** | Union of grass + dead-branch + epiphyte |
| Lizard.communicate.not-paved | `ground_not_paved` |
| Lizard.reproduce.nurse-log | `stat_fallen log > 0` |
| Lizard.reproduce.fallen-tree | `forest_size in fallen\|decayed` |
| **Lizard.reproduce** | Union of nurse-log + fallen-tree |
| Tree.acquire.moderated | `proposal_release_control_intervention == "reduce-canopy-pruning"` |
| Tree.acquire.autonomous | `proposal_release_control_intervention == "eliminate-canopy-pruning"` |
| **Tree.acquire** | Union of moderated + autonomous |
| Tree.communicate.snag | `forest_size == "snag"` |
| Tree.communicate.fallen | `forest_size == "fallen"` |
| Tree.communicate.decayed | `forest_size == "decayed"` |
| **Tree.communicate** | `forest_size in snag\|fallen\|decayed` |
| Tree.reproduce.smaller-patches-rewild | `proposal_recruit_intervention == "rewild-smaller-patch"` |
| Tree.reproduce.larger-patches-rewild | `proposal_recruit_intervention == "rewild-larger-patch"` |
| **Tree.reproduce** | Union of smaller + larger patches |

### Output format

Per site, columns: `indicator`, `measure`, `baseline`, `positive yr180`, `trending yr180`, `positive / trending`, `trending % of positive`.

Each indicator value shown as percentage of baseline and as multiples between scenarios.

### Output file

`{root}/comparison/v4_indicator_comparison.md`

Full indicator definitions: `_futureSim_refactored/sim/v4_indicator_definitions.md`

---

## Step 5: Proposal and intervention visualisations

Renders hybrid proposal views from the VTK files. Requires VTKs from Steps 2 and 4 to exist.

### Renderer

`_futureSim_refactored/outputs/report/render_proposal_v4.py`

### Shared layout parameters (always pass these)

- `--model-base-y 1170`
- `--target-model-width 1757`

### Defaults

- All years (0, 1, 10, 30, 60, 90, 120, 150, 180)
- Main variant only (the `_proposal-and-interventions_with-legend.png`)
- Pass `--also-families-only` to additionally render the flat `proposal-families-only` variant

### Run all sites in parallel

```bash
export REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name>
for site in trimmed-parade city uni; do
  for scenario in positive trending; do
    for year in 0 1 10 30 60 90 120 150 180; do
      uv run python _futureSim_refactored/outputs/report/render_proposal_v4.py \
        --site "$site" --scenario "$scenario" --year "$year" \
        > /tmp/render_proposal_${site}_${scenario}_${year}.log 2>&1 &
    done
  done
done
wait
```

### Single site + scenario + year

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  uv run python _futureSim_refactored/outputs/report/render_proposal_v4.py \
  --site trimmed-parade --scenario positive --years 180
```

### Output

`{root}/temp/validation/renders/{site}_{scenario}_yr{year}_proposal-and-interventions_with-legend.png`

Full render set: 3 sites × 2 scenarios × 9 years = **54 images**

---

## Step 6: Debug recruit renders

Renders diagnostic images for each recruit variable. Requires VTKs from Step 2.

### Renderer

`_futureSim_refactored/outputs/report/render_debug_recruit.py`

### 11 diagnostic layers

| Layer | Type | What it shows |
|---|---|---|
| `recruit_isNewTree` | categorical | New trees (green) vs original (grey) |
| `recruit_hasbeenReplanted` | categorical | Replanted (blue) vs not (grey) |
| `recruit_mechanism` | categorical | node-rewild (orange), under-canopy (purple), under-canopy-linked (teal), ground (green) |
| `recruit_year` | numeric ramp | Blue (early) -> red (late) |
| `recruit_mortality_rate` | numeric ramp | Yellow (low) -> dark red (high) |
| `recruit_mortality_cohort` | discrete | Green (small DBH) -> red (large DBH) |
| `ground_recruitment` | composite | Recruitable ground (green) + ground-recruited canopies (orange) |
| `node_rewild_recruitment` | composite | Node-rewild zone via sim_Nodes (blue) + node-rewild canopies (red) |
| `under_canopy_recruitment` | composite | Under-canopy zone via node_CanopyID (lavender) + under-canopy canopies (magenta) |
| `under_canopy_linked_recruitment` | composite | Under-canopy-linked zone via node_CanopyID (pale teal) + linked canopies (deep teal) |
| `sim_nodes_zones` | composite | Each sim_Nodes zone coloured uniquely |

### Defaults

- Years 10, 60, 180
- All 11 diagnostic layers

### Run all sites in parallel

```bash
export REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name>
for site in trimmed-parade city uni; do
  for scenario in positive trending; do
    for year in 10 60 180; do
      uv run python _futureSim_refactored/outputs/report/render_debug_recruit.py \
        --site "$site" --scenario "$scenario" --years "$year" \
        > /tmp/render_debug_${site}_${scenario}_${year}.log 2>&1 &
    done
  done
done
wait
```

### Single site + specific layers

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  uv run python _futureSim_refactored/outputs/report/render_debug_recruit.py \
  --site trimmed-parade --scenario positive --years 180 \
  --layers ground_recruitment node_rewild_recruitment under_canopy_recruitment under_canopy_linked_recruitment sim_nodes_zones
```

### Output

`{root}/temp/validation/renders/debugRecruit/{site}_{scenario}_yr{year}_{layer}_with-legend.png`

---

## Step 7: Node-level size statistics (optional, can run during Step 2)

This is a first-pass comparison. It does not need VTKs — only the treeDFs from Step 1 and a baseline nodeDF. The node-only run (Step 1) already writes `{root}/comparison/interim-size-assessment.csv` with yr180 size breakdowns and baseline deltas.

For each site and scenario, count trees by `size` column at each year. Report deltas against the **woodland baseline** (not yr 0).

### Size classes

`small`, `medium`, `large`, `senescing`, `snag`, `fallen`, `decayed`

### Baseline source

`{root}/temp/interim-data/{site}/{site}_baseline_1_nodeDF_-180.csv`

If the baseline does not exist in the current root yet (Step 3 not run), a default baseline root per site is used automatically (defined in `engine_v3.py` → `DEFAULT_BASELINE_ROOTS`). The woodland baseline does not change between runs.

### Scenario source

`{root}/temp/interim-data/{site}/{site}_{scenario}_1_treeDF_{year}.csv`

---

## Step 8: Merge site-level statistics (optional)

Merges per-state V3 indicator and action CSVs (already written during Steps 2 and 3) into site-level CSVs with `pct_of_baseline`. This is a fast CSV-only pass — it does not re-read VTKs if per-state CSVs already exist.

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  uv run python _futureSim_refactored/sim/run/run_full_v3_batch.py \
  --compile-stats-only
```

### What it produces

**Per-state indicator counts** — voxel counts for each of the 12 v3 capability indicators:

| Persona | Capability | Indicator ID | What it counts |
|---|---|---|---|
| Bird | self | Bird.self.peeling | Peeling bark voxels |
| Bird | others | Bird.others.perch | Perchable canopy voxels |
| Bird | generations | Bird.generations.hollow | Hollow voxels |
| Lizard | self | Lizard.self.grass | Ground cover voxels |
| Lizard | self | Lizard.self.dead | Dead branch voxels |
| Lizard | self | Lizard.self.epiphyte | Epiphyte voxels |
| Lizard | others | Lizard.others.notpaved | Non-paved surface voxels |
| Lizard | generations | Lizard.generations.nurse-log | Nurse log voxels |
| Lizard | generations | Lizard.generations.fallen-tree | Fallen tree voxels |
| Tree | self | Tree.self.senescent | Late-life tree + deadwood voxels |
| Tree | others | Tree.others.notpaved | Soil near canopy features |
| Tree | generations | Tree.generations.grassland | Grassland for recruitment |

Comparison metric: `pct_of_baseline` (count as percentage of the baseline count for the same indicator).

**Per-state action breakdowns** — each indicator count split by support action type:

- `control_level` breakdown: high (street-tree), medium (park-tree), low (reserve-tree, improved-tree)
- `urban_element` breakdown: open space, green roof, brown roof, facade, roadway, busy roadway, parking, etc.
- `rewilding_status` breakdown: footprint-depaved, footprint-depaved-connected, exoskeleton, node-rewilded, none
- Some indicators also count `artificial` (installed) separately

**Merged site-level stats** — all per-state data merged into one CSV per site.

### Output files

- Per-state: `{root}/output/stats/per-state/{site}/{site}_{scenario}_1_yr{year}_indicator_counts.csv`
- Per-state: `{root}/output/stats/per-state/{site}/{site}_{scenario}_1_yr{year}_action_counts.csv`
- Baseline: `{root}/output/stats/per-state/{site}/{site}_baseline_1_indicator_counts.csv`
- Baseline: `{root}/output/stats/per-state/{site}/{site}_baseline_1_action_counts.csv`
- Merged: `{root}/output/stats/csv/{site}_indicator_counts.csv`
- Merged: `{root}/output/stats/csv/{site}_action_counts.csv`

---

## Run Log

Every batch run appends to `_data-refactored/run_log.csv` (columns: timestamp, name, output_root, description).

When `REFACTOR_RUN_OUTPUT_ROOT` is not set, `_futureSim_refactored.paths.refactor_run_output_root()` falls back to the last logged output root. To inspect recent runs:

```bash
uv run python _futureSim_refactored/sim/run/run_log.py
```

Pass `--description "my note"` to the batch runner to add a description to the log entry.

---

## Verification Checklist

After a full run, verify these file counts:

| Artifact | Expected count |
|---|---|
| Scenario `state_with_indicators.vtk` | 54 (3 sites x 2 scenarios x 9 years) |
| Baseline `state_with_indicators.vtk` | 3 (one per site) |
| Integrated `nodeDF` CSVs | 54 |
| Interim `treeDF` CSVs | 54 |
| Bioenvelope PLYs | 48 (3 sites x 2 scenarios x 8 years; yr0 has no proposals) |
| Per-state indicator stats CSVs | 114 (57 scenario + 57 scenario action + 3 baseline + 3 baseline action... actually: (54 scenario + 3 baseline) x 2 files = 114) |
| Merged site stats CSVs | 6 (3 sites x 2 types: indicator + action) |
| Proposal render PNGs | 57 (3 sites x (2 scenarios x 9 years + 1 baseline)) |

---

## Troubleshooting

### PermissionError on `.nc` file

The subset cache (`{site}_1_subsetForScenarios.nc`) can become 0 bytes if two processes write simultaneously. Fix:

```bash
rm data/revised/final/<site>/<site>_1_subsetForScenarios.nc
```

Then rerun. The next run rebuilds it from the full voxel array. Do not run VTK/baselines in parallel if the cache does not already exist.

### Legacy root env vars

The batch runner rejects `REFACTOR_SCENARIO_OUTPUT_ROOT`, `REFACTOR_ENGINE_OUTPUT_ROOT`, and `REFACTOR_STATISTICS_ROOT`. Unset them and use `REFACTOR_RUN_OUTPUT_ROOT` only.
