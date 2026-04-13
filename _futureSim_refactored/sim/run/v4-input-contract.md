# V4 Input Contract

What the simulation pipeline needs before it can run, where it lives, and how paths resolve.

See also:

- `v4-run-instructions.md` — how to run the pipeline
- `_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md` — output lineage and sync rules

---

## Input Root

All simulation inputs live under:

```
_data-refactored/model-inputs/
```

This directory has four sections:

| Section | What it holds |
|---|---|
| `sites/{site}/` | Per-site voxel arrays, node CSVs, world reference VTKs |
| `shared/` | Cross-site reference data (baseline density, log library, site coords) |
| `tree_libraries/` | Base tree template library (PKL) |
| `tree_variants/` | Variant template sets and approved template root |

---

## Per-Site Inputs (`sites/{site}/`)

Sites: `trimmed-parade`, `city`, `uni`

### Minimum required files

| File | Format | What it is |
|---|---|---|
| `{site}_1_voxelArray_RewildingNodes.nc` | NetCDF | Full voxel array with ground data, node IDs, canopy resistance, terrain classification |
| `{site}_1_voxelArray_withLogs.nc` | NetCDF | Voxel array with log placements; used by urban element feature transfer (`a_scenario_urban_elements_count.py`) |
| `{site}_1_treeDF.csv` | CSV | Tree inventory: location, size, DBH, control class |
| `{site}_1_poleDF.csv` | CSV | Utility pole locations |
| `{site}_1_logDF.csv` | CSV | Fallen log locations |
| `{site}-siteVoxels-masked.vtk` | VTK | Site boundary reference geometry |
| `{site}-roadVoxels-coloured.vtk` | VTK | Road/terrain reference geometry |

### Optional files

| File | Format | Sites | What it is |
|---|---|---|---|
| `{site}-extraTreeDF.csv` | CSV | uni | Additional trees not in base inventory |
| `{site}-extraPoleDF.csv` | CSV | uni | Additional poles not in base inventory |

### Cache (regenerated automatically)

| File | Format | Regenerated from |
|---|---|---|
| `{site}_1_subsetForScenarios.nc` | NetCDF | Voxel array (filtered subset); auto-created on first run if missing |

---

## Shared Inputs (`shared/`)

| File | Format | What it is | Used by |
|---|---|---|---|
| `tree-baseline-density.csv` | CSV | Target tree counts by DBH class for woodland baseline | `baseline_v3.py`, `a_scenario_get_baselines.py` |
| `logLibrary.pkl` | Pickle | Point-cloud templates for fallen log geometry | `a_resource_distributor_dataframes.py`, `adTree_AssignResources.py` |
| `site_locations.csv` | CSV | Site geographic metadata (coordinates, bounds) | `voxel_f_SiteCoordinates.py` |

---

## Tree Templates (`tree_libraries/`, `tree_variants/`)

### Base library

`tree_libraries/base/trees/`

Contains the base template PKLs and resource dictionary. Override with env var `TREE_TEMPLATE_BASE_ROOT` (legacy alias: `BASE_TREE_TEMPLATES_ROOT`).

### Variants directory

`tree_variants/`

Contains variant template sets generated from the base library. Override with env var `TREE_TEMPLATE_VARIANTS_ROOT`.

### Approved variant (default for simulation)

`tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees/`

This is the canonical template set used by default. The directory name encodes the variant configuration:

| Parameter | Value | Meaning |
|---|---|---|
| `fallens_use` | `nonpre-direct` | Fallen tree variant bundle |
| `snags_use` | `elm-snags-old` | Snag variant bundle |
| decayed | `decayed-small-fallen` | Decayed/small-fallen variant bundle |
| `voxel_size` | `1` | Voxel resolution for pre-voxelized templates |

Override with env var `TREE_TEMPLATE_ROOT`. No need to set unless explicitly asked — the default points to this approved set.

### Key files in the approved template root

| File | Format | What it is |
|---|---|---|
| `template-library.overrides-applied.pkl` | Pickle | Full template table with variant edits applied |
| `{voxel}_combined_voxel_templateDF.pkl` | Pickle | Pre-voxelized templates at specified voxel size |
| `template-edits_base_geometry_volume_lookup.csv` | CSV | Volume metadata for deadwood placement budgets |

---

## Path Resolution

All paths resolve through `_futureSim_refactored/paths.py`. Two constants define the input roots:

```python
SITE_INPUTS_ROOT  = MODEL_INPUTS_ROOT / "sites"      # _data-refactored/model-inputs/sites/
SHARED_INPUTS_ROOT = MODEL_INPUTS_ROOT / "shared"     # _data-refactored/model-inputs/shared/
```

### Site inputs

`site_inputs_dir(site)` returns `SITE_INPUTS_ROOT / site`. All per-site path functions (`site_tree_locations_path`, `site_rewilding_voxel_array_path`, `site_subset_dataset_path`, `site_world_reference_vtk_path`, etc.) build on this.

### Shared inputs

Callers import `SHARED_INPUTS_ROOT` and use it directly:

```python
from _futureSim_refactored.paths import SHARED_INPUTS_ROOT
pd.read_csv(SHARED_INPUTS_ROOT / 'tree-baseline-density.csv')
```

### Tree templates

`tree_template_root()` returns the approved template dir. Overridable via `TREE_TEMPLATE_ROOT` env var (not normally needed).

---

## Environment Variables

| Variable | Controls | Default | Set it? |
|---|---|---|---|
| `REFACTOR_RUN_OUTPUT_ROOT` | Where outputs go | Last run from run log | **Yes, every run** |
| `TREE_TEMPLATE_ROOT` | Approved template dir | Built-in default | No (uses default) |
| `TREE_TEMPLATE_BASE_ROOT` | Base template library | Built-in default | No |
| `TREE_TEMPLATE_VARIANTS_ROOT` | Variants directory | Built-in default | No |

Do **not** set `SIM_OUTPUT_ROOT` (does not exist) or the legacy `REFACTOR_SCENARIO_OUTPUT_ROOT` / `REFACTOR_ENGINE_OUTPUT_ROOT` / `REFACTOR_STATISTICS_ROOT` (rejected at startup).

---

## Legacy Scripts

The `final/` and `modules/` directories contain legacy scripts that still reference the old `data/revised/final/` paths. These are **not** part of the v4 pipeline and are not maintained. The files they depend on have been moved; those scripts will break if run directly.

Legacy scripts that originally generated some of these inputs:

| Legacy script | What it produced |
|---|---|
| `modules/treeBake_recreateLogs.py` | `logLibrary.pkl` |
| `modules/trees.py`, `modules/sites_assignTrees.py` | Tree/pole CSVs, species data |
| `modules/getBaselines.py` | Baseline density reference |

### Input preprocessing scripts (stale paths)

The following scripts under `_futureSim_refactored/` generate the raw inputs (voxel arrays, tree templates) from LAS/point-cloud data. They still have hardcoded `data/revised/final/` paths and need updating if they are ever re-run:

- `sim/voxel/voxel_a_voxeliser.py` — voxel array generation from LAS tiles
- `sim/voxel/voxel_a_helper_functions.py` — voxeliser helpers
- `input_processing/tree_processing/combined_tree_manager.py` — tree template management
- `input_processing/tree_processing/combined_generateResourceDict.py` — resource dictionary generation (`data/csvs/lerouxdata-update.csv`)

---

## Verification

After setting up inputs (or after downloading from Mediaflux), verify:

```bash
# Check all required files exist
for site in trimmed-parade city uni; do
  echo "=== $site ==="
  ls _data-refactored/model-inputs/sites/$site/
done
echo "=== shared ==="
ls _data-refactored/model-inputs/shared/
```

Expected per site: 1 `.nc` voxel array, 1 `.nc` subset cache (or absent until first run), 3 node CSVs (tree/pole/log), 2 world VTKs (site/road). Plus extras for uni.

Expected shared: `tree-baseline-density.csv`, `logLibrary.pkl`, `site_locations.csv`.
