# V4 Run Instructions

These instructions are for agents and humans running the v4 simulation pipeline. Follow them in order. Do not skip steps. Do not run steps in parallel unless explicitly stated.

---

## Before You Start

**Ask the user for the output root name.** Every run needs a name that becomes the output directory. Example: `v4`, `v4-test`, `v4-snag-fix`. The user decides this.

The full root path is always:

```
_data-refactored/model-outputs/generated-states/<root-name>
```

Set it with the environment variable `REFACTOR_RUN_OUTPUT_ROOT`:

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name>
```

Prefix every command with this variable, or export it once per session.

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
  python _code-refactored/refactor_code/sim/run/run_full_v3_batch.py \
  --node-only
```

**Outputs**: `{root}/temp/interim-data/{site}/{site}_{scenario}_1_treeDF_{year}.csv` (and matching nodeDFs)

**Verify**: check that each site directory has 18 treeDF files (9 years x 2 scenarios).

---

## Step 2: VTK generation

Reads the saved treeDFs/nodeDFs and builds VTK point clouds with indicators. This is the slowest step.

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  python _code-refactored/refactor_code/sim/run/run_full_v3_batch.py \
  --vtk-only
```

**Outputs**: `{root}/output/vtks/{site}/{site}_{scenario}_1_yr{year}_state_with_indicators.vtk`

**While VTKs generate**, you can compute node-level size statistics as a first pass (Step 3).

---

## Step 3: Node-level size statistics (can run during Step 2)

This is a first-pass comparison. It does not need VTKs — only the treeDFs from Step 1 and a baseline nodeDF.

For each site and scenario, count trees by `size` column at each year. Report deltas against the **woodland baseline** (not yr 0).

### Format

One table per site/scenario. Rows are size classes, columns are years. Each cell shows `count (+/-delta from baseline)`. Include a bold **total** row.

```
| size | baseline | yr 0 | yr 1 | yr 10 | ... | yr 180 |
|---|---|---|---|---|---|---|
| small | 2014 | 160 (-1854) | 515 (-1499) | ... | ... |
| medium | 255 | 235 (-20) | 249 (-6) | ... | ... |
| large | 56 | 61 (+5) | 61 (+5) | ... | ... |
| senescing | 104 | 0 (-104) | 0 (-104) | ... | ... |
| snag | 48 | 0 (-48) | 0 (-48) | ... | ... |
| fallen | 33 | 0 (-33) | 0 (-33) | ... | ... |
| decayed | 38 | 0 (-38) | 0 (-38) | ... | ... |
| **total** | **2548** | **456 (-2092)** | **825 (-1723)** | ... | ... |
```

### Size classes

`small`, `medium`, `large`, `senescing`, `snag`, `fallen`, `decayed`

### Baseline source

`{root}/temp/interim-data/{site}/{site}_baseline_1_nodeDF_-180.csv`

If the baseline does not exist in the current root yet (Step 4 not run), use the baseline from a previous root (e.g. `v4-allometric-flat-mortality` or `v4updatedterms`). The woodland baseline does not change between runs.

### Scenario source

`{root}/temp/interim-data/{site}/{site}_{scenario}_1_treeDF_{year}.csv`

---

## Step 4: Regenerate baselines

Generates woodland baseline VTKs and nodeDFs. Must run after Step 1 (node-only) because it uses the same cached `.nc` subset file.

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  python _code-refactored/refactor_code/sim/run/run_full_v3_batch.py \
  --regenerate-baselines
```

**Outputs**: baseline VTKs and `{site}_baseline_1_nodeDF_-180.csv`

---

## Step 5: Indicator statistics

Reads `state_with_indicators.vtk` files (from Steps 2 and 4) and computes voxel-level indicator counts. Requires both scenario VTKs and baseline VTKs to exist.

```bash
REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
  python _code-refactored/refactor_code/sim/run/run_full_v3_batch.py \
  --compile-stats-only
```

### Output format

See `_code-refactored/refactor_code/sim/v4 comparisons.md` for the full indicator comparison format. The key table compares yr 180 between scenarios as percentages of baseline and as multiples.

Indicator definitions are in `_code-refactored/refactor_code/sim/v4_indicator_definitions.md`.

---

## Step 6: Visualisations

Renders hybrid proposal views from the VTK files. Requires VTKs from Steps 2 and 4 to exist.

Full instructions are in `_code-refactored/refactor_code/v4visualisation-instructions.md`. Key points:

### Renderer

`_code-refactored/refactor_code/outputs/report/render_proposal_v4.py`

### Shared layout parameters (always pass these)

- `--model-base-y 1170`
- `--target-model-width 1757`

### Run all sites in parallel

```bash
for site in trimmed-parade city uni; do
  REFACTOR_RUN_OUTPUT_ROOT=_data-refactored/model-outputs/generated-states/<root-name> \
    uv run python \
    _code-refactored/refactor_code/outputs/report/render_proposal_v4.py \
    --site $site --scenario all \
    --years 0 1 10 30 60 90 120 150 180 \
    --output-mode validation --hybrid-only \
    --model-base-y 1170 --target-model-width 1757 &
done
wait
```

### Output

`{root}/temp/validation/renders/custom/{site}_{scenario}_yr{year}_engine3-proposal-hybrid_with-legend.png`

Full render set: 3 sites x (2 scenarios x 9 years + 1 baseline) = **57 images**

---

## Current Model Parameters

| Parameter | Value | Notes |
|---|---|---|
| Growth model | `split` | precolonial → `fischer`, colonial → `ulmus` |
| Mortality model | `flat` | Le Roux 0.06 urban, 0.03 reserve |
| Mortality applies to | small, medium, large | Extended to large in v4 |
| Senescing duration | triangular(10, 90, 200) | |
| Snag duration | triangular(0, 40, 100) | Mode changed from 50 to 40 in v4 |
| Fallen duration | triangular(10, 40, 100) | |
| Decayed duration | triangular(30, 40, 75) | |
| Recruit: node-rewilded | `RECRUIT_FULL` (rewild-larger-patch) | Reserve mortality (0.03) |
| Recruit: footprint-depaved | `RECRUIT_PARTIAL` (rewild-smaller-patch) | Urban mortality (0.06) |

---

## Troubleshooting

### PermissionError on `.nc` file

The subset cache (`{site}_1_subsetForScenarios.nc`) can become 0 bytes if two processes write simultaneously. Fix:

```bash
rm data/revised/final/<site>/<site>_1_subsetForScenarios.nc
```

Then rerun. The next run rebuilds it from the full voxel array. Do not run VTK/baselines in parallel if the cache does not already exist.
