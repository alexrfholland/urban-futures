# Repo Cleanup Log

Record of working-tree tidy-ups. If you need anything "from before", check out the pre-cleanup commit SHA below — every moved file is preserved under `_archive/` or in git history.

---

## 2026-04-11 — Root archive sweep

**Branch:** `engine-v4`
**Pre-cleanup commit:** `d5f8f91` (`WIP: engine-v4 in-progress changes`)
**Cleanup commits:**
- `1bc8436` — Archive legacy root files/folders to _archive/
- `be42bdf` — Archive grey-area folders to _archive/

**Goal:** Remove decades of pre-refactor cruft from the working root without deleting anything tracked. Live refactor code (`_futureSim_refactored/`), current data (`_data-refactored/`, `_outputs-refactored/`, `_statistics-refactored-v3/`), current docs (`_documentation-refactored/`), and the upstream checkpoint pipeline (`data/`, `final/`, `modules/`) were left untouched.

### Archived (moved to `_archive/` via `git mv`)

Loose root scripts and configs:
- `app.py`, `main.py`, `viewer.py`, `query.py`, `saveroutine.py`
- `test.py`, `test2.py`, `test3.py`, `testCamera.py`
- `combine-blocks.py`, `csvparser.py`, `plot_kmz.py`, `walls_and_ceilings.py`
- `geoTiffGetter.py`, `final_geoSpatialConverter.py`, `process_obj_files.py`
- `a_info_treeGraphs.py`, `info_persona_capabilities.py`
- `camera_params.json`, `sscamera_views.json`, `stakeholders_camera_views.json`
- `RenderOption_2024-09-09-18-07-44.json`, `RenderOption_2024-09-11-13-47-10.json`
- `habitat-systems.json`, `habitat-systems2.json`, `habitat-systems3.json`
- `artificial-structures.json`
- `dendron_parser.log`, `process_edits.log`, `simulation_run_20251216_180844.log`, `simulation_run_20251216_180951.log`
- `single_cylinder_no_caps.stl`, `requirements.txt`

Legacy tracked folders:
- `Processing/` (1 file — empty dir otherwise)
- `csv converters/`, `current tests/`, `info/`, `indexes/`, `plots/`, `utilities/`
- `qsms/` — branch structure / tree params
- `ss/` — shapefile/raster/KDE experiments
- `mesh_utils-main/` — vendored mesh2pcd library
- `blender_compositor_cmaps/`
- `revised/` — 15 pre-refactor geospatial scripts (`rGeoStamp.py`, `rConvertToMesh.py`, `rProcessSites.py`, etc.)
- `code/` — 2026 Blender edge-detection lab scripts (~56 files). Previously referenced by `_futureSim_refactored/blender/compositor/WORKFLOW_REGISTRY.md` as canonical sources — those md path refs are now stale (point to `_archive/code/...` if needed).
- `_statistics-refactored/` — v1 legacy stats (canonical v3 stats live at `_statistics-refactored-v3/`, still in root)
- `documentation/` — superseded by `_documentation-refactored/`. Its `supplementary-scripts-archived/` subfolder (19 `a_*.py` / `f_*.py` scripts) was compressed into `_archive/documentation/archive_of_a_and_f_scripts.zip` rather than moved file-by-file.

### Deleted (untracked, gitignored — ~12 GB total)

- `outputs/` (8.6 MB) — pre-refactor output CSVs
- `_tmp_mediaflux/` (316 KB) — stray `working_code_path/` test dir with duplicate old scripts
- `_tmp_mf_downloads/` (91 MB) — accidental Mediaflux redownload
- `_tmp_unified_validation/` (3.3 GB) — Mediaflux upload/download roundtrip test output from 2026-04-04/06

### Resulting root

Live items only:

```
AGENTS.md  ENVIRONMENT.md  MEDIAFLUX.md  pyproject.toml  uv.lock
test_template_library_loader.py
_futureSim_refactored/          # active refactor code
_data-refactored/          # live data (gitignored)
_outputs-refactored/       # live outputs (gitignored, 8.6G)
_statistics-refactored-v3/ # canonical stats (gitignored)
_documentation-refactored/ # current docs
_archive/                  # everything above
data/  final/  modules/    # upstream checkpoint pipeline, untouched
```

### Recovering an archived file

```bash
# from _archive/ directly (still on disk)
cp -r _archive/revised/rGeoStamp.py .

# or from git history before the sweep
git show d5f8f91:revised/rGeoStamp.py > rGeoStamp.py
```
