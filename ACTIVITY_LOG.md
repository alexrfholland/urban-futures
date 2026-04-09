# Activity Log — Last 3 Weeks (Mar 20 – Apr 1, 2026)

All work on `master` branch (linear history).

---

## Week 1: Mar 20 – Mar 26

### Thu Mar 20 (~1 hr)
- Archived legacy supplementary scripts (helper functions, rewilding, geospatial modules, etc.)
- Added assessment materials and updated `.gitignore`

### Thu Mar 26 (~3 hrs)
- Built out 2026 Blender edge detection lab pipeline — added ~20 render scripts (mist variants v1–v4, AO, normals, base lines, depth edges, resource fills, Kirsch sizes)
- Added arboreal resource colour reference docs
- Updated assessment comparison pathways

---

## Week 2: Mar 28 – Mar 31

### Sat Mar 28 (~2 hrs)
- Added baseline Blender workflow scripts (city baseline build, instancer, VTK-to-PLY export, zoom renders, shadow catcher preview)
- Created key scripts doc and minimal resources reference
- Added edge detection lab baseline renderers (AO, normals, mist, depth outliner, resource fills)
- Fixed archive script naming for Windows-compatible checkout

### Sun Mar 29 (~8 hrs) — big refactor day
- **14:39** — Snapshot before initial refactor. Added edge lab output suite spec, workflow registry, combined compositor, refined compositor scripts
- **18:52** — Documented proposed proposals-and-interventions tables before updating
- **19:20** — Added Blender ParaView view references
- **22:23** — Major refactor of pathway tracking: restructured into `_code-refactored/`, `_data-refactored/`, `_documentation-refactored/`, `_statistics-refactored/` directories. Created `paths.py` config. Added blender output hooks docs, comparison CSVs (interventions + proposals for city, trimmed-parade, highlights)

### Mon Mar 30 (~4 hrs)
- Added `AGENTS.md` project docs
- Built pathway tracking graphs script (`a_info_pathway_tracking_graphs.py` — 1059 lines new)
- Major edge lab expansion: final template blend builder, current mist/base/bioenvelopes/core outputs/legacy shading renderers
- Added combined compositor runner and final template runner
- Rewilded envelope legacy surface variant
- Stats difference inspection docs

### Tue Mar 31 (~6 hrs) — v2 scenario engine day
- **12:41** — Implemented v2 scenario engine validation workflow. Massive commit (~6k lines added): new `engine_v2.py`, `validation.py`, `build_tree_variants.py`, `compare_outputs.py`, `render_forest_size_views.py`. Refactored `a_scenario_runscenario.py`, `a_scenario_manager.py`, `a_scenario_generateVTKs.py`, `a_scenario_get_baselines.py`, `a_scenario_urban_elements_count.py`. Built comparison pathways v2 builder with SS site variants. Added `run_all_simulations.py`
- **15:47** — Added decayed deadwood phase to v2 engine. New `structure_ids.py`, `a_scenario_baseline_variants.py`, `refresh_indicator_csvs_from_baseline.py`. Validation docs. Comparison pathway decayed variants
- **16:00** — Optimised v2 export path (voxeliser, VTK gen, resource distributor)

---

## Week 3: Apr 1

### Wed Apr 1 (~3 hrs)
- Finalised canonical v2 outputs and edge lab updates
- New `scenario_engine_v2_model.md` (526 lines — full model documentation)
- Added depth outliner, mist remapped, sizes renderers to edge lab
- Built Photoshop stack exporter for edge lab
- Updated resource distributor dataframes
- Finalised SS site comparison pathway indicators

---

## Summary by Day

| Date       | Day | Approx Hours | Focus |
|------------|-----|-------------|-------|
| Mar 20 Thu | 1   | ~1 hr       | Archiving legacy scripts |
| Mar 26 Thu | 2   | ~3 hrs      | Blender edge detection lab pipeline (v1–v4 render scripts) |
| Mar 28 Sat | 3   | ~2 hrs      | Baseline Blender workflows, Windows compat fix |
| Mar 29 Sun | 4   | ~8 hrs      | Major refactor: pathway tracking, project restructure, edge lab expansion |
| Mar 30 Mon | 5   | ~4 hrs      | AGENTS.md, pathway tracking graphs, edge lab renderers, rewilded envelopes |
| Mar 31 Tue | 6   | ~6 hrs      | v2 scenario engine (engine, validation, tree variants, comparison pathways, decayed deadwood) |
| Apr 1  Wed | 7   | ~3 hrs      | Finalise v2 outputs, model docs, edge lab Photoshop stack |

**Total: ~27 hours across 7 active days over 3 weeks**
