# Compositor Refactor Plan

This is the staged refactor plan for moving Edge Lab toward a cleaner
compositor-owned model.

## Phase 1. Freeze The Contract

Goal:

- define template ownership clearly before moving files or changing behavior

Deliverables:

- `COMPOSITOR_TEMPLATE_CONTRACT.md`
- script status / ownership map

Test gate:

- no behavior changes
- documentation reflects the current system honestly

## Phase 2. Identify Canonical Assets

Goal:

- decide which blends and scripts are current canonical assets

Deliverables:

- canonical template list
- helper blend list
- legacy / historical list

Test gate:

- every kept blend has a stated role
- every kept script has a stated role

## Phase 3. Stop Hidden Script Ownership

Goal:

- identify places where scripts secretly own graph logic instead of the template

Current known problem areas:

- `render_edge_lab_current_mist.py`
- `render_edge_lab_current_depth_outliner.py`
- overloaded shading path in `render_edge_lab_legacy_shading.py`

Deliverables:

- one note per script stating:
  - template-native
  - wrapper around legacy logic
  - template mutation script

Test gate:

- script classification is explicit
- no behavior change yet

## Phase 4. Simplify Runtime Layer

Goal:

- reduce the runtime layer to thin template drivers

Target shape:

- one suite runner
- thin per-family renderers
- helper renderers only where the helper blend is itself canonical

Test gate:

- family output counts still match
- output folders remain correct
- no canonical blend regression

## Phase 5. Migrate Mist To Template Ownership

Goal:

- make mist render from saved canonical template logic instead of legacy scratch-scene reconstruction

Steps:

- document current mist output contract
- match saved template branch to accepted output behavior
- replace legacy adapter path with template-native rendering

Test gate:

- render city timeline mist outputs
- compare expected filenames and visual result

## Phase 6. Migrate Depth To Template Ownership

Goal:

- make depth outliner render from saved canonical template logic instead of legacy scratch-scene reconstruction

Test gate:

- render city timeline depth outputs
- compare expected filenames and visual result

## Phase 7. Rename And Rehome

Goal:

- move canonical compositor assets into the new refactored home

Planned target shape:

- durable code under `_code-refactored/refactor_code/blender/compositor`
- durable data under `_data-refactored/compositor`
- temp blends under:
  - `template_development`
  - `template_instantiations`
  - `checkpoints`
  - `scratch`

Test gate:

- paths updated cleanly
- old locations still readable until cutover is complete

## First Implementation Order

Implement one change at a time:

1. classify current scripts
2. classify current blends
3. remove hidden script ownership from mist
4. remove hidden script ownership from depth
5. clean naming and runtime entrypoints
6. move files

After each change:

- render the smallest relevant subset
- confirm folder structure
- confirm expected PNG names
- stop if the output contract drifts
