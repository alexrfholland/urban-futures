# bV2 Temporary Implementation Place

## bV2 conventions

### 1. Naming and versioning

- reusable Blender v2 scripts use the `bV2_` prefix
- `bV2_` filenames stay stable and role-based, not `v3`, `v4`, or ad hoc suffixes
- temporary or inspection scripts use the `tmp_` prefix and should be treated as disposable
- versioning lives at the system/folder level as `blenderv2`, not in the name of every reusable script
- use `uni` consistently, not the old `street` alias, except when documenting legacy v1.5 behavior

### 2. Locations

- reusable code lives under [\_code-refactored](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored)
- reusable docs live under [\_documentation-refactored/blenderv2](/d:/2026%20Arboreal%20Futures/urban-futures/_documentation-refactored/blenderv2)
- the current production EXR / AOV contract note is [bV2_exr_aov_contract.md](/d:/2026%20Arboreal%20Futures/urban-futures/_documentation-refactored/blenderv2/bV2_exr_aov_contract.md)
- reusable template/data assets live under [\_data-refactored](/d:/2026%20Arboreal%20Futures/urban-futures/_data-refactored)
- temporary scripts should live in a clearly named temporary area, not mixed with the reusable pipeline
- generated outputs can live on `E:` when that is safer for disk space

### 3. Blend and script conventions

- VS Code / git-hosted Python is the source of truth for pipeline logic
- the template `.blend` is the source of truth for persistent Blender assets only
- template = assets, script = structure
- keep in the template only the things Blender should own directly: geometry nodes, materials, cameras, source road/building objects, and core scene render/world defaults
- create in script all structural scene state: collections, view layers, compositor outputs, pass indices, generated objects, visibility rules, validation setup, and mode-specific assembly
- if something can be rebuilt reliably by script, do not store it as required scene state in the template
- run the real external scripts from git against the open blend; do not depend on many hidden in-blend text scripts
- if a generated `.blend` will not stay open in Blender GUI, make a GUI-safe inspection copy by opening it with `load_ui=False` and resaving it, instead of repeatedly retrying the raw file
- when opening GUI blends for inspection, verify that the Blender process is still alive after launch before treating the file as successfully opened

## bV2 plan

### Template blend

Keep one base template blend containing:

- one base scene only
- render settings, mist, and world defaults
- base road/building source objects for each site, present as source assets and excluded from production view layers until used
- geometry nodes
- materials
- cameras under a strict naming convention

### Script responsibilities

The `bV2_*` scripts should:

1. open the template and create a clean working version for the target site/mode
2. create the view layers in script
3. create the compositor EXR output setup in script
4. run instancer build, bioenvelope build, and world rebuild
5. assign object passes, collection visibility, and mode-specific scene state
6. save the result for validation and later render/output steps

### Guiding rule

- avoid split ownership between blend and script for structural scene setup
- for `bV2`, prefer structural setup in script and persistent assets in the blend
- timeline and single-state should use the same operator family, with data assembly as the main mode difference

## Current bV2 status

### Implemented now

- [bV2_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored/refactor_code/blenderv2/bV2_scene_contract.py)
  - canonical names, materials, node groups, view layers, AOVs, site contracts, and naming helpers
- [bV2_init_scene.py](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored/refactor_code/blenderv2/bV2_init_scene.py)
  - scene shell creation, source asset cloning, view-layer creation, AOV reset, scene metadata, and semantic collection exclusion
- [bV2_build_instancers.py](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored/refactor_code/blenderv2/bV2_build_instancers.py)
  - single-state and timeline instancer build
  - timeline dataframe clipping/splicing before translation
  - `source-year` stamping
  - ignored size filtering for trees
  - priority derivation from positive
  - shared hidden `model_cache`
  - live file logging via `BV2_LOG_PATH`
  - debug material mode via `BV2_INSTANCER_DISPLAY_MODE=source-year`

### Built outputs so far

- timeline instancers built for:
  - `city`
  - `trimmed-parade`
  - `uni`
- single-state instancers built for `yr180` for:
  - `city`
  - `trimmed-parade`
  - `uni`

Outputs currently live under:

- [E:\\2026 Arboreal Futures\\blenderv2\\blends](/e:/2026%20Arboreal%20Futures/blenderv2/blends)
- [E:\\2026 Arboreal Futures\\blenderv2\\logs](/e:/2026%20Arboreal%20Futures/blenderv2/logs)

### Important current implementation notes

- timeline strip spacing is controlled by `TIMELINE_OFFSET_STEP = 5.0` in [bV2_build_instancers.py](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored/refactor_code/blenderv2/bV2_build_instancers.py)
- timeline dataframe assembly is done in pandas before Blender object creation
- `city` and `trimmed-parade` currently offset on `Y`
- `uni` currently offsets on `X`
- collection visibility now uses explicit `bV2_role` tags, not truncated Blender collection names
- workspace-pruning / saved-UI cleanup is disabled by default because it stalled `init_scene`; keep it opt-in only

### Known issues / handoff notes

- some generated blends do not reliably stay open through the normal GUI launch path on this machine
- use a GUI-safe `load_ui=False` resave when needed for inspection
- verify the Blender process is still alive after launch before claiming a GUI open succeeded
- the model cache currently shares imported mesh datablocks across states, but state collections still hold separate object instances
- `trimmed-parade` timeline produced zero-alpha seam columns in some production EXRs when rendered with Cycles `OPTIX`
- a full rerender on `2026-04-06` using Cycles `CUDA` removed those seam columns
- treat that as a backend-specific render artifact; if `trimmed-parade` seams reappear, rerender that site on `CUDA`

## Current bV2 TODOs

- `source-year` should be the canonical provenance AOV/attribute for build outputs
- initialize `source-year` as `-1`
- during world rebuild, use `source-year` to do the job currently implied by `sim_Matched`: mark whether a Blender point successfully received transferred source data, and from which year that data came
- add `source-year` to all geometry families, not just world geometry:
  - world rebuild outputs
  - instancer point clouds
  - bioenvelopes
- this is especially important for timeline builds, so we can inspect which year each attribute payload actually came from

## bV2 Schema Recap

### Global pipeline rules

- positive and trending are the canonical scenario branches
- positive priority is derived from positive
- all renderable geometry carries `source-year`
- `source-year` is initialized as `-1` until real provenance is assigned
- world uses the canonical shared world path:
  - material: `v2WorldAOV`
  - Geometry Nodes: `v2WorldPoints`
- instancers use `MINIMAL_RESOURCES`
- bioenvelopes use `Envelope`

### Per-site asset contract

For each site, the contract defines:

- source world objects
- timeline camera
- optional hero or alternate cameras
- which instancer families are valid at that site

### Per-mode rules

Only define true mode differences here:

- year token:
  - single-state: `yr{year}`
  - timeline: `timeline`
- data preparation:
  - single-state uses one year
  - timeline merges multiple years
- camera rule:
  - timeline uses the timeline camera
  - single-state can use the timeline camera or an explicitly chosen alternate
- provenance behavior:
  - timeline may contain mixed `source-year` values
  - single-state usually has one consistent `source-year`

### Canonical view layers

- `existing_condition_positive`
- `existing_condition_trending`
- `positive_state`
- `positive_priority_state`
- `trending_state`
- `bioenvelope_positive`
- `bioenvelope_trending`

### View-layer semantics

- `existing_condition_positive`
  - positive world attributes
- `existing_condition_trending`
  - trending world attributes
- `positive_state`
  - positive instances
  - positive world attributes
  - positive bioenvelopes
- `positive_priority_state`
  - positive-priority instances
  - positive world attributes
- `trending_state`
  - trending instances
  - trending world attributes
  - trending bioenvelopes
- `bioenvelope_positive`
  - positive world attributes
  - positive bioenvelopes
- `bioenvelope_trending`
  - trending world attributes
  - trending bioenvelopes

### Current naming direction

- built scene names include site and mode
- internal collection structure stays flatter and more generic
- generated positions and model collections keep explicit site/year/state in the name
- for timeline mode, the year token is `timeline`

### Per-site asset contract

`city`

- world sources:
  - `city_buildings_source`
  - `city_roads_source`
- cameras:
  - timeline: `city - camera - time slice - zoom`
  - hero: `city-yr180-hero-image`
- instancer families:
  - `trees`
  - `logs`

`trimmed-parade`

- world sources:
  - `trimmed-parade_buildings_source`
  - `trimmed-parade_roads_source`
- cameras:
  - timeline: `parade - camera - time slice - zoom`
- instancer families:
  - `trees`
  - `logs`

`uni`

- world sources:
  - `uni_buildings_source`
  - `uni_roads_source`
- cameras:
  - timeline: `uni - camera - time slice - zoom`
- instancer families:
  - `trees`
  - `logs`
  - `poles`

### Generated naming contract

#### Scene names

- `bV2_{site}_timeline`
- `bV2_{site}_single_state_{year}`

Examples:

- `bV2_city_timeline`
- `bV2_city_single_state_yr180`
- `bV2_trimmed-parade_timeline`
- `bV2_uni_single_state_yr180`

#### Top-level collections

- `cameras`
- `world`
- `instancers`
- `bioenvelopes`
- `build`

#### Generated position objects

- `trees_positions_{site}_{year}_{state}`
- `logs_positions_{site}_{year}_{state}`
- `poles_positions_{site}_{year}_{state}`

#### Generated model collections

- `trees_models_{site}_{year}_{state}`
- `logs_models_{site}_{year}_{state}`
- `poles_models_{site}_{year}_{state}`

#### Generated world objects

- `buildings_{site}_{year}_{state}`
- `roads_{site}_{year}_{state}`

#### Generated bioenvelope objects

- `bioenvelope_{site}_{year}_positive`
- `bioenvelope_{site}_{year}_trending`

#### State tokens

- `positive`
- `positive_priority`
- `trending`

#### Year token rules

- single-state uses `yr{year}`
- timeline uses `timeline`

## Existing v1.5 refactor

The current v1.5 Blender refactor lives in [final/_code-refactored/blender/timeline](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline).

This is the stack we have been using to generate the current timeline and single-state scenes. It already has a public `unified` surface, but the architecture is still mixed:

- the operator-facing entrypoints are the `b2026_unified_*` files
- most of the heavy build work still happens in the older `b2026_timeline_*` files
- template prep, view-layer contract, camera import, and validation are partly centralized
- instancer, bioenvelope, and world generation are still implementation-heavy and not yet fully modularized

The important practical point for `blenderv2` is that v1.5 is not a clean from-scratch system. It is a partially unified pipeline sitting on top of older timeline internals.

## Current v1.5 code Flow

### Public entry flow

1. [b2026_unified_build_template.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_template.py)
   - opens the source template blend
   - chooses `timeline` or `single_state`
   - prepares the collection shell, view layers, materials, world, and approved camera
2. [b2026_unified_build_scene.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_scene.py)
   - optional template build when `B2026_UNIFIED_BUILD_TEMPLATE=1`
   - runs instancers
   - runs bioenvelopes
   - runs world rebuild
   - applies single-state post-build layer setup when needed
   - runs validation by default
3. [b2026_unified_setup_render.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_setup_render.py)
   - restores render materials, AOV/passes, camera, mist, and compositor setup
4. [b2026_unified_render_workbench_view_layers.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_render_workbench_view_layers.py)
   - fast workbench QA renders per view layer
5. [b2026_unified_render_exrs.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_render_exrs.py)
   - production multilayer EXR render path

### What `b2026_unified_build_scene.py` actually runs

1. [b2026_unified_build_template.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_template.py) if template prep is enabled
2. [b2026_unified_build_instancers.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_instancers.py)
   - delegates to [b2026_timeline_instancer.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_instancer.py)
3. [b2026_unified_build_bioenvelopes.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_bioenvelopes.py)
   - delegates to [b2026_timeline_bioenvelopes.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_bioenvelopes.py)
4. [b2026_unified_build_world.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_world.py)
   - delegates to [b2026_timeline_rebuild_world_year_attrs.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_rebuild_world_year_attrs.py)
5. [b2026_timeline_generate_single_state.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_generate_single_state.py)
   - only for the single-state post-build view-layer contract
6. [b2026_unified_validate_scene.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_validate_scene.py)

### Core support files

- [b2026_unified_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_scene_contract.py)
  - current public contract for naming, view layers, expected collections, expected instancer specs, and validation expectations
- [b2026_unified_runtime.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_runtime.py)
  - utility layer for env flags and local-script execution
- [b2026_timeline_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_scene_contract.py)
  - older contract/naming layer still used by implementation scripts
- [b2026_timeline_scene_setup.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_scene_setup.py)
  - scene shell and view-layer setup helper functions
- [b2026_timeline_layout.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_layout.py)
  - site specs, bundle resolution, timeline strip positions, and dataframe assembly

### Current mode split in v1.5

- `single_state`
  - one year only
  - still uses the same heavy instancer, bioenvelope, and world scripts
  - adds a post-build layer-contract pass through [b2026_timeline_generate_single_state.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_generate_single_state.py)
- `timeline`
  - five years merged into timeline strips
  - uses [b2026_timeline_layout.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_layout.py) to preassemble multi-year dataframes before the instancer/world build stages

### Current v1.5 strengths

- one public operator surface exists
- both modes now share the same seven view-layer names
- scene validation exists and checks actual collection visibility plus instancer wiring
- approved time-slice cameras can be imported during template prep
- the same public wrapper can build city timeline and single-state outputs end to end

### Current v1.5 limitations

- the public layer is still too wrapper-heavy
- instancer logic is still monolithic in [b2026_timeline_instancer.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_instancer.py)
- bioenvelope and world build logic are still legacy-heavy
- the `street` / `uni` alias history is still present in v1.5 and should not be carried forward into `blenderv2`
- source templates and generated outputs are not yet organized under the `_...-refactored` roots the way we want for `blenderv2`

## v1.5 files

### Public v1.5 entrypoints

- [b2026_unified_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_scene_contract.py)
- [b2026_unified_scene_setup.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_scene_setup.py)
- [b2026_unified_build_template.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_template.py)
- [b2026_unified_build_instancers.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_instancers.py)
- [b2026_unified_build_bioenvelopes.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_bioenvelopes.py)
- [b2026_unified_build_world.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_world.py)
- [b2026_unified_build_scene.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_build_scene.py)
- [b2026_unified_setup_render.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_setup_render.py)
- [b2026_unified_render_workbench_view_layers.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_render_workbench_view_layers.py)
- [b2026_unified_render_exrs.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_render_exrs.py)
- [b2026_unified_validate_scene.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_validate_scene.py)
- [b2026_unified_runtime.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_unified_runtime.py)

### Heavy implementation files still carrying most of the build logic

- [b2026_timeline_build_template_from_single_state.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_build_template_from_single_state.py)
- [b2026_timeline_instancer.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_instancer.py)
- [b2026_timeline_bioenvelopes.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_bioenvelopes.py)
- [b2026_timeline_rebuild_world_year_attrs.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_rebuild_world_year_attrs.py)
- [b2026_timeline_render_lightweight_isolated_exrs_generic.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_render_lightweight_isolated_exrs_generic.py)
- [b2026_timeline_render_workbench_view_layers.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_render_workbench_view_layers.py)
- [b2026_timeline_generate_single_state.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_generate_single_state.py)

### Shared helpers and contracts behind v1.5

- [b2026_timeline_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_scene_contract.py)
- [b2026_timeline_scene_setup.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_scene_setup.py)
- [b2026_timeline_layout.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_layout.py)
- [b2026_timeline_runtime_flags.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/b2026_timeline_runtime_flags.py)

### Current v1.5 documentation

- [AGENTS.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/AGENTS.md)
- [TIMELINE_RESTRUCTURE_PLAN.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/TIMELINE_RESTRUCTURE_PLAN.md)
- [TIMELINE_RUNBOOK.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/TIMELINE_RUNBOOK.md)
- [TIMELINE_SCENE_TEMPLATE.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/TIMELINE_SCENE_TEMPLATE.md)
- [VIEWS.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/VIEWS.md)

### Current v1.5 source blends and working assets

- city single-state template: [2026 futures city single-state template.blend](/d:/2026%20Arboreal%20Futures/data/2026%20futures%20city%20single-state%20template.blend)
- city cleaned scene: [2026 futures city lightweight cleaned.blend](/d:/2026%20Arboreal%20Futures/data/2026%20futures%20city%20lightweight%20cleaned.blend)
- parade cleaned scene: [2026 futures parade lightweight cleaned.blend](/d:/2026%20Arboreal%20Futures/data/2026%20futures%20parade%20lightweight%20cleaned.blend)
- street/uni cleaned scene: [2026 futures street lightweight cleaned.blend](/d:/2026%20Arboreal%20Futures/data/2026%20futures%20street%20lightweight%20cleaned.blend)
- approved debug camera source: [2026 futures timeslice debug camera framing v3.blend](/d:/2026%20Arboreal%20Futures/data/renders/timeslices/camera_tests/2026%20futures%20timeslice%20debug%20camera%20framing%20v3.blend)

### v1.5 data/output locations currently in use

- working repo code: [final/_code-refactored/blender/timeline](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline)
- staged sim tweak bundle: [simv3recruitanddecaytweaks](/e:/2026%20Arboreal%20Futures/blender/inputs/v3%20tests/simv3recruitanddecaytweaks)
- generated validation outputs: [urban-futures_validation](/e:/2026%20Arboreal%20Futures/urban-futures_validation)

### Current bV2 bundle preference

- `blenderv2` should read from the local temp bundle root, not directly from the network share
- shared settings now live in [bV2_paths.py](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored/refactor_code/blenderv2/bV2_paths.py):
  - `BLENDER_USE_REMOTE`
  - `BLENDER_TEMP_REPO`
  - `BLENDER_REPO_ROOT`
- sync helper now lives in [bV2_sync_inputs.py](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored/refactor_code/blenderv2/bV2_sync_inputs.py)
- current intended flow:
  - copy from `BLENDER_REPO_ROOT` (currently the mapped `Z:` drive location)
  - into `BLENDER_TEMP_REPO`
  - write `_bV2_source.txt` in the temp repo for provenance
  - builders read from `BLENDER_TEMP_REPO`

## Immediate note for bV2

When we rebuild this as `blenderv2`, the goal is not to rename the existing v1.5 wrappers. The goal is to move to a genuinely modular `bV2_*` pipeline inside the `_...-refactored` roots, with:

- `uni` used consistently instead of the `street` alias
- docs under [\_documentation-refactored/blenderv2](/d:/2026%20Arboreal%20Futures/urban-futures/_documentation-refactored/blenderv2)
- code under a new Blender v2 area in [\_code-refactored](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored)
- data/templates under a matching Blender v2 area in [\_data-refactored](/d:/2026%20Arboreal%20Futures/urban-futures/_data-refactored)
- outputs allowed on `E:` when storage pressure makes that necessary
