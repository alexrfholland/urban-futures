# Activity Log — 3 Weeks (Mar 20 – Apr 9, 2026)

Branches: `master` → `engine-v3` → `engine-v4`
Total commits in period: **40**

---

## Week 1: Mar 20 – Mar 26 (master)

### Thu Mar 20 (~1 hr)
- Archived legacy supplementary scripts (helper functions, rewilding, geospatial modules, etc.)
- Added assessment materials, updated `.gitignore`

### Thu Mar 26 (~3 hrs)
- Built 2026 Blender edge detection lab pipeline — ~20 render scripts (mist variants v1–v4, AO, normals, base lines, depth edges, resource fills, Kirsch sizes)
- Added arboreal resource colour reference docs
- Updated assessment comparison pathways

---

## Week 2: Mar 28 – Apr 1 (master → engine-v3)

### Sat Mar 28 (~2 hrs)
- Added baseline Blender workflow scripts (city baseline build, instancer, VTK-to-PLY export, zoom renders, shadow catcher preview)
- Created key scripts doc and minimal resources reference
- Edge detection lab baseline renderers (AO, normals, mist, depth outliner, resource fills)
- Fixed archive script naming for Windows-compatible checkout

### Sun Mar 29 (~8 hrs) — big refactor day
- Snapshot before initial refactor; edge lab output suite spec, workflow registry, combined/refined compositor scripts
- Documented proposed proposals-and-interventions tables
- Added Blender ParaView view references
- **Major restructure** into `_code-refactored/`, `_data-refactored/`, `_documentation-refactored/`, `_statistics-refactored/`. Created `paths.py` config. Blender output hooks docs, comparison CSVs (interventions + proposals for city, trimmed-parade, highlights)

### Mon Mar 30 (~4 hrs)
- Added `AGENTS.md` project docs
- Built pathway tracking graphs script (`a_info_pathway_tracking_graphs.py` — 1059 lines)
- Edge lab expansion: final template blend builder, current mist/base/bioenvelopes/core outputs/legacy shading renderers
- Combined compositor runner, final template runner
- Rewilded envelope legacy surface variant
- Stats difference inspection docs

### Tue Mar 31 (~6 hrs) — v2 scenario engine
- Implemented v2 scenario engine validation workflow (~6k lines): `engine_v2.py`, `validation.py`, `build_tree_variants.py`, `compare_outputs.py`, `render_forest_size_views.py`
- Refactored `a_scenario_runscenario.py`, `a_scenario_manager.py`, `a_scenario_generateVTKs.py`, `a_scenario_get_baselines.py`, `a_scenario_urban_elements_count.py`
- Comparison pathways v2 builder with SS site variants; `run_all_simulations.py`
- Added decayed deadwood phase: `structure_ids.py`, `a_scenario_baseline_variants.py`, `refresh_indicator_csvs_from_baseline.py`
- Optimised v2 export path (voxeliser, VTK gen, resource distributor)

### Tue Apr 1 (~10 hrs) — v2 finalise + v3 engine launch (engine-v3 branch)
- Finalised canonical v2 outputs; `scenario_engine_v2_model.md` (526 lines)
- Edge lab: depth outliner, mist remapped, sizes renderers, Photoshop stack exporter
- **v3 planning**: scenario engine v3 planning notes (829 lines added)
- **v3 implementation**: scenario engine v3 candidate pipeline (2426 lines added, 12 files)
- Refined v3 exports and Blender proposal framebuffers (2036 lines, 27 files)
- Added refactored Blender timeline bundle (12,402 lines, 48 files)
- Blender data audit notes, fallen-log instancer tweak

---

## Week 3: Apr 2 – Apr 9 (engine-v3 → engine-v4)

### Wed Apr 2 (~3 hrs)
- Added proposal AOVs and timeline EXR render updates

### Thu Apr 3 (~8 hrs)
- Added timeline refactor docs and blenderv2 instancer scaffolding (7429 lines, 40 files)
- Refactored v3 pipeline, baselines, and render tooling (6290 lines, 59 files)

### Fri Apr 4 (~7 hrs)
- Saved current v3 work, synced remote
- Set up reproducible `uv` environment; implemented cohort-based tree mortality (2062 lines)
- Documented v3 tree template-library file renaming and migration checkpoint
- Pointed urban-futures at shared mediafluxsync workflow
- Added Blender v2 bioenvelope build and viewport test helpers (992 lines)
- Ignored local `.tools` workspace
- WIP local changes sync

### Sat Apr 5 (~3 hrs)
- Fixed proposal release-control reassessment for dying trees (25 files touched)

### Sun Apr 6 (~6 hrs)
- Pre v3 name changes for key sim engine variables (1036 lines, 20 files)
- Renamed v3 schema and backfilled canopy resistance at init (1402 lines, 25 files)
- Updated blenderv2 render pipeline and docs (2722 lines, 13 files)

### Mon Apr 7 (~8 hrs) — v4 restructure day (engine-v4 branch)
- Pre v4-restructure cleanup
- **v4 restructure**: massive reorganisation (35k+ lines moved across 219 files)
- Restructured v4 simulation layout (44k+ lines across 286 files)
- Renamed v4 sim and input folders (63 files)
- Documented v4 sim flow and proposal broadcast gap (427 lines)

### Tue Apr 8 (~7 hrs)
- V4 proposal broadcast into sim pipeline + edge lab compositor refactor (2347 lines, 18 files)
- V4 sim running — recruit mortality too aggressive, investigating (7765 lines, 74 files)

### Wed Apr 9 (~3 hrs)
- **V4 engine**: allometric growth, flat mortality with large half-rate, recruit metadata columns (2839 lines, 24 files)

---

## Summary by Day

| Date       | Day | Approx Hrs | Branch    | Focus |
|------------|-----|-----------|-----------|-------|
| Mar 20 Thu |  1  | ~1        | master    | Archiving legacy scripts |
| Mar 26 Thu |  2  | ~3        | master    | Blender edge detection lab pipeline |
| Mar 28 Sat |  3  | ~2        | master    | Baseline Blender workflows |
| Mar 29 Sun |  4  | ~8        | master    | Major project restructure, edge lab expansion |
| Mar 30 Mon |  5  | ~4        | master    | Pathway tracking graphs, AGENTS.md, edge lab renderers |
| Mar 31 Tue |  6  | ~6        | master    | v2 scenario engine, decayed deadwood, export optimisation |
| Apr 1  Tue |  7  | ~10       | master→v3 | Finalise v2, launch v3 engine + Blender timeline bundle |
| Apr 2  Wed |  8  | ~3        | v3        | Proposal AOVs, timeline EXR renders |
| Apr 3  Thu |  9  | ~8        | v3        | v3 pipeline refactor, baselines, render tooling |
| Apr 4  Fri | 10  | ~7        | v3        | uv environment, cohort mortality, bioenvelope helpers, mediafluxsync |
| Apr 5  Sat | 11  | ~3        | v3        | Fix proposal release-control for dying trees |
| Apr 6  Sun | 12  | ~6        | v3        | v3 variable renaming, canopy resistance, blenderv2 render pipeline |
| Apr 7  Mon | 13  | ~8        | v3→v4     | v4 restructure (massive reorg), sim layout, docs |
| Apr 8  Tue | 14  | ~7        | v4        | v4 proposal broadcast, recruit mortality investigation |
| Apr 9  Wed | 15  | ~3        | v4        | v4 allometric growth, flat mortality tuning, recruit metadata |

**Total: ~79 hours across 15 active days over 3 weeks**
