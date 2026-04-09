# Activity Log — 3 Weeks (Mar 20 – Apr 9, 2026)

Branches: `master` → `engine-v3` → `engine-v4`
Total commits in period: **40**

---

## Week 1: Mar 20 – Mar 26 (master)

### Thu Mar 20 (~4 hrs)
- Archived legacy supplementary scripts (helper functions, rewilding, geospatial modules, etc.)
- Added assessment materials, updated `.gitignore`
- 38 files, 2,553 lines added

### Thu Mar 26 (~8 hrs)
- Built 2026 Blender edge detection lab pipeline — ~20 render scripts (mist variants v1–v4, AO, normals, base lines, depth edges, resource fills, Kirsch sizes)
- Added arboreal resource colour reference docs
- Updated assessment comparison pathways
- 98 files, 26,975 lines added

---

## Week 2: Mar 28 – Apr 1 (master → engine-v3)

### Sat Mar 28 (~4 hrs)
- Added baseline Blender workflow scripts (city baseline build, instancer, VTK-to-PLY export, zoom renders, shadow catcher preview)
- Created key scripts doc and minimal resources reference
- Edge detection lab baseline renderers (AO, normals, mist, depth outliner, resource fills)
- Fixed archive script naming for Windows-compatible checkout
- 31 files, 5,057 lines added

### Sun Mar 29 (~8 hrs) — big refactor day
- 14:39–22:23 active commits
- Snapshot before initial refactor; edge lab output suite spec, workflow registry, combined/refined compositor scripts
- Documented proposed proposals-and-interventions tables
- Added Blender ParaView view references
- **Major restructure** into `_code-refactored/`, `_data-refactored/`, `_documentation-refactored/`, `_statistics-refactored/`. Created `paths.py` config. Blender output hooks docs, comparison CSVs (interventions + proposals for city, trimmed-parade, highlights)
- 72 files, ~7,062 lines added

### Mon Mar 30 (~6 hrs)
- Added `AGENTS.md` project docs
- Built pathway tracking graphs script (`a_info_pathway_tracking_graphs.py` — 1,059 lines)
- Edge lab expansion: final template blend builder, current mist/base/bioenvelopes/core outputs/legacy shading renderers
- Combined compositor runner, final template runner
- Rewilded envelope legacy surface variant
- Stats difference inspection docs
- 29 files, 4,784 lines added

### Tue Mar 31 (~8 hrs) — v2 scenario engine
- 11:41–16:00 active commits
- Implemented v2 scenario engine validation workflow (~6k lines): `engine_v2.py`, `validation.py`, `build_tree_variants.py`, `compare_outputs.py`, `render_forest_size_views.py`
- Refactored `a_scenario_runscenario.py`, `a_scenario_manager.py`, `a_scenario_generateVTKs.py`, `a_scenario_get_baselines.py`, `a_scenario_urban_elements_count.py`
- Comparison pathways v2 builder with SS site variants; `run_all_simulations.py`
- Added decayed deadwood phase: `structure_ids.py`, `a_scenario_baseline_variants.py`, `refresh_indicator_csvs_from_baseline.py`
- Optimised v2 export path (voxeliser, VTK gen, resource distributor)
- v3 planning notes (829 lines)
- 95 files, ~11,373 lines added

### Wed Apr 1 (~8 hrs) — v3 engine launch (engine-v3 branch)
- 16:26–23:27 active commits
- Finalised canonical v2 outputs; `scenario_engine_v2_model.md` (526 lines)
- Edge lab: depth outliner, mist remapped, sizes renderers, Photoshop stack exporter
- **v3 implementation**: scenario engine v3 candidate pipeline (2,426 lines, 12 files)
- Refined v3 exports and Blender proposal framebuffers (2,036 lines, 27 files)
- Added refactored Blender timeline bundle (12,402 lines, 48 files)
- Blender data audit notes, fallen-log instancer tweak
- 90 files, ~17,207 lines added

---

## Week 3: Apr 2 – Apr 9 (engine-v3 → engine-v4)

### Thu Apr 2 (~3 hrs)
- Added proposal AOVs and timeline EXR render updates
- 5 files, 277 lines added

### Fri Apr 3 (~8 hrs)
- 21:55–23:21 active commits
- Added timeline refactor docs and blenderv2 instancer scaffolding (7,429 lines, 40 files)
- Refactored v3 pipeline, baselines, and render tooling (6,290 lines, 59 files)
- 99 files, ~13,719 lines added

### Sat Apr 4 (~7 hrs)
- 17:41–22:37 active commits
- Saved current v3 work, synced remote
- Set up reproducible `uv` environment; implemented cohort-based tree mortality (2,062 lines)
- Documented v3 tree template-library file renaming and migration checkpoint
- Pointed urban-futures at shared mediafluxsync workflow
- Added Blender v2 bioenvelope build and viewport test helpers (992 lines)
- Ignored local `.tools` workspace
- WIP local changes sync
- 89 files, ~11,927 lines added

### Sun Apr 5 (~4 hrs)
- Fixed proposal release-control reassessment for dying trees (25 files touched)
- 25 files, 316 lines changed

### Mon Apr 6 (~6 hrs)
- 14:19–20:07 active commits
- Pre v3 name changes for key sim engine variables (1,036 lines, 20 files)
- Renamed v3 schema and backfilled canopy resistance at init (1,402 lines, 25 files)
- Updated blenderv2 render pipeline and docs (2,722 lines, 13 files)
- 58 files, ~5,160 lines added

### Tue Apr 7 (~9 hrs) — v4 restructure day (engine-v4 branch)
- 11:09–20:03 active commits
- Pre v4-restructure cleanup
- **v4 restructure**: massive reorganisation (35k+ lines moved across 219 files)
- Restructured v4 simulation layout (44k+ lines across 286 files)
- Renamed v4 sim and input folders (63 files)
- Documented v4 sim flow and proposal broadcast gap (427 lines)
- V4 proposal broadcast into sim pipeline + edge lab compositor refactor (2,347 lines)
- 331 files, ~4,518 lines net added

### Wed Apr 8 (~8 hrs)
- 07:10–17:08 active commits
- V4 sim running — recruit mortality too aggressive, investigating (7,765 lines, 74 files)
- V4 engine: allometric growth, flat mortality with large half-rate, recruit metadata columns (2,839 lines, 24 files)
- 78 files, ~6,533 lines added

### Thu Apr 9 (~4 hrs)
- V4 engine tuning continues
- Allometric growth, flat mortality with large half-rate, recruit metadata columns
- 24 files, 2,839 lines added

---

## Summary by Day

| Date       | Day | Est Hrs | Timestamps    | Branch    | Files | Lines +  | Focus |
|------------|-----|---------|---------------|-----------|-------|----------|-------|
| Mar 20 Thu |  1  |      ~4 | 19:46         | master    |    38 |   2,553  | Archiving legacy scripts |
| Mar 26 Thu |  2  |      ~8 | 15:12         | master    |    98 |  26,975  | Blender edge detection lab pipeline |
| Mar 28 Sat |  3  |      ~4 | 17:01–17:57   | master    |    31 |   5,057  | Baseline Blender workflows |
| Mar 29 Sun |  4  |      ~8 | 14:39–22:23   | master    |    72 |   7,062  | Major restructure, edge lab, pathway tracking |
| Mar 30 Mon |  5  |      ~6 | 19:02         | master    |    29 |   4,784  | Pathway tracking graphs, AGENTS.md, edge lab |
| Mar 31 Tue |  6  |      ~8 | 11:41–16:00   | master    |    95 |  11,373  | v2 engine, decayed deadwood, v3 planning |
| Apr 1  Wed |  7  |      ~8 | 16:26–23:27   | master→v3 |    90 |  17,207  | v3 engine launch, Blender timeline bundle |
| Apr 2  Thu |  8  |      ~3 | 11:59         | v3        |     5 |     277  | Proposal AOVs, timeline EXR renders |
| Apr 3  Fri |  9  |      ~8 | 21:55–23:21   | v3        |    99 |  13,719  | v3 pipeline refactor, instancer scaffolding |
| Apr 4  Sat | 10  |      ~7 | 17:41–22:37   | v3        |    89 |  11,927  | uv env, cohort mortality, bioenvelope helpers |
| Apr 5  Sun | 11  |      ~4 | 15:18         | v3        |    25 |     316  | Fix proposal release-control for dying trees |
| Apr 6  Mon | 12  |      ~6 | 14:19–20:07   | v3        |    58 |   5,160  | v3 schema rename, canopy resistance, blenderv2 |
| Apr 7  Tue | 13  |      ~9 | 11:09–20:03   | v3→v4     |   331 |   4,518  | v4 restructure, proposal broadcast |
| Apr 8  Wed | 14  |      ~8 | 07:10–17:08   | v4        |    78 |   6,533  | v4 sim, recruit mortality, allometric growth |
| Apr 9  Thu | 15  |      ~4 | 07:10         | v4        |    24 |   2,839  | v4 flat mortality tuning, recruit metadata |

**Total: ~95 hours across 15 active days over 3 weeks**
