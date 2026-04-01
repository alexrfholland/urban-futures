# Scenario Engine V3 Refactor Plan

This note is the working plan for the third scenario-engine refactor.

It is written against the current canonical v2 state described in:

- [scenario_engine_v2_model.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_model.md)
- [scenario_engine_v2_status.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_status.md)
- [validation.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/validation.md)

## Required Reading Order

Before touching code, read these in this order:

1. this plan: [scenario_engine_v3_plan.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v3_plan.md). Read carefully first to understand the branch split, output roots, execution rules, and verification process.
2. [scenario_engine_v2_model.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_model.md). This is the main implementation specification. It describes the proposal/intervention model changes V3 is meant to implement, and the implementation should follow it closely.
3. [scenario_engine_v2_status.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_status.md). This defines what is currently canonical in v2 and what must not be overwritten.
4. [scenario_engine_v2_canonical_checklist.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_canonical_checklist.md). Use this as the explicit checklist of ways canonical v2 diverges from the old v1 flow. Before trusting a v3 run, verify it is inheriting these v2 behaviors rather than silently falling back to old routes.
5. [validation.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/validation.md). Use this as the base pattern for quick verification and full verification outputs.

Important rule:

- when reading [scenario_engine_v2_model.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_model.md), read the `Conventions` section first and treat it as mandatory
- do not start implementation from the later proposal sections without first understanding:
  - legacy field names
  - renamed v2/v3 terms
  - `[add: ...]` versus `[rename: ...]`

Important implementation note:

- unresolved `TODO` items in [scenario_engine_v2_model.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_model.md) are not prerequisites for starting V3
- do not stop the refactor to resolve them up front
- only raise a `TODO` if it becomes a real implementation blocker

## Current Canonical V2

Current canonical v2 is:

- git branch: `master`
- scenario outputs: [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
- engine outputs: [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)
- statistics: [_statistics-refactored-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_statistics-refactored-v2)

Canonical v2 currently means:

- `NodeID = -1` fix
- accepted template settings:
  - `fallens_use = nonpre-direct`
  - `snags_use = elm-snags-old`
- `decayed` lifecycle phase
- regenerated baselines for that setup
- stable persistent `structureID`
- deterministic template/log fallback selection in export
- in-memory `urban_features` handoff

V3 work should not overwrite these roots until v3 is accepted.

## Canonical Template Root Requirement

V3 must explicitly inherit the accepted canonical v2 template library configuration.

Approved template root:

- [_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/tree_variants/template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen/trees)

Required settings:

- `fallens_use = nonpre-direct`
- `snags_use = elm-snags-old`
- include the `decayed-small-fallen` variant bundle
- `voxel_size = 1`

Do not rely on the loader default.

For v3 candidate runs:

- set `TREE_TEMPLATE_ROOT` explicitly
- persist that root into the run metadata
- treat a missing template-root record as a validation failure

Not acceptable for v3 candidate verification:

- `data/revised/trees`

Reason:

- that is the historical default loader path
- it does not guarantee the approved snag/fallen/decayed variant bundle
- a v3 run can look structurally correct while still exporting the wrong deadwood geometry

## Recommended V3 Working Split

Use a separate git branch and separate output roots.

Recommended git branch:

- `engine-v3`

Recommended candidate output roots:

- scenario outputs: `data/revised/final-v3`
- engine outputs: `_data-refactored/v3engine_outputs`
- statistics: `_statistics-refactored-v3`

Recommended rule:

- keep `final-v2` and `v2engine_outputs` read-only as the comparison target
- do all v3 generation into `final-v3` and `v3engine_outputs`
- only promote v3 after full verification

## Data Source Freeze

Do not touch the project data sources as part of the v3 refactor.

For v3, treat these as read-only inputs:

- canonical source datasets under [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)
- canonical generated engine outputs under [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)
- preflight site inputs and resistance-grid source data
- accepted tree template libraries already in use for canonical v2

V3 should change:

- engine/runtime code
- reporting/render logic
- candidate generated outputs under the dedicated v3 roots

If a datasource change becomes necessary, treat it as a separate explicit ticket with its own versioned output path. Do not edit canonical inputs in place.

## Scope Of V3

The v3 refactor is not just a rename pass. It rebases proposal and intervention outputs onto the actual updated simulation logic described in [scenario_engine_v2_model.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_model.md).

The main model changes to implement are:

- proposal-decay terminology and decision/intervention cleanup
- proposal-release-control as explicit accepted/rejected logic plus intervention assignment
- proposal-colonise as explicit accepted/rejected logic rebased on under-node treatment and bio-envelope states
- proposal-recruit as two-channel logic:
  - `buffer-feature`
  - `rewild-ground`
- proposal-deploy-structure as explicit accepted intervention assignment on logs and poles
- new V3 proposal arrays and intervention arrays on the xarray / VTK outputs
- eventual merge back to one source of truth after V3 has been verified

## Proposal / Intervention Schema Contract

The key V3 requirement is alignment with the project language of proposals and interventions.

That means the resulting state tables and VTK outputs should expose clearly defined proposal fields, rather than relying on mixed legacy naming.

Target dataframe contract:

- `proposal-{proposal name}_decision`
- `proposal-{proposal name}_intervention`

Apply that contract to the relevant state tables:

- `tree_df`
- `log_df`
- `pole_df`
- `node_df` where a node-level proposal/intervention state is actually stored

Proposal names to support:

- `proposal-decay`
- `proposal-release-control`
- `proposal-colonise`
- `proposal-recruit`
- `proposal-deploy-structure`

Target VTK/xarray contract during the V3 candidate phase:

- `proposal_decayV3`
- `proposal_release_controlV3`
- `proposal_coloniseV3`
- `proposal_recruitV3`
- `proposal_deploy_structureV3`
- `proposal_decayV3_intervention`
- `proposal_release_controlV3_intervention`
- `proposal_coloniseV3_intervention`
- `proposal_recruitV3_intervention`
- `proposal_deploy_structureV3_intervention`

Important rule:

- dataframe fields and VTK/xarray fields must describe the same proposal logic
- do not leave proposal decisions only in tables while keeping stale proposal arrays in the VTKs
- if a proposal/intervention is renamed in the model, the exported fields must make that rename legible

## Proposal Reference Documents

Use these as the semantic reference for proposal/intervention naming and meaning:

- [scenario_engine_v2_model.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_model.md)
- [2-3-INFO-2-proposal-and-intervention-descriptions.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/pathway_tracking/2-3-INFO-2-proposal-and-intervention-descriptions.md)
- [2-3-INFO-2-proposal-and-intervention-technical-specifications.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/pathway_tracking/2-3-INFO-2-proposal-and-intervention-technical-specifications.md)
- [project.sentience.proposal.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/ddl/project.sentience.proposal.md)

Use these as the current reporting/output references:

- [a_info_proposal_interventions.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_proposal_interventions.py)
- [a_info_pathway_tracking_graphs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_pathway_tracking_graphs.py)

## Non-Goals During Initial V3 Refactor

Do not combine these with the initial V3 proposal/intervention refactor:

- exporter/performance optimization beyond what is already canonical in v2
- datasource edits
- tree-template library redesign
- removal of the legacy `proposal_*` arrays before V3 is accepted
- cleanup of old v2 code paths before V3 is verified

Reason:

- V3 already changes model semantics
- mixing semantic changes with optimization or cleanup makes verification much harder

## Recommended Implementation Strategy

### 1. Create A Separate Runtime Path

Do this first.

Actions:

- add a new engine module:
  - [_code-refactored/refactor_code/scenario/engine_v3.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v3.py)
- keep [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py) untouched as the v2 reference
- update the v3 branch wrapper path so [final/a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py) delegates to `engine_v3.py`

Reason:

- this keeps v2 readable and diffable
- it avoids a mixed v2/v3 engine file during the transition
- it gives a clean rollback path if v3 proposal logic needs multiple iterations

### 2. Establish V3 Output Routing Before Logic Changes

Do this before running any v3 generation.

Actions:

- route all v3 writes to:
  - `data/revised/final-v3`
  - `_data-refactored/v3engine_outputs`
  - `_statistics-refactored-v3`
- do not reuse `final-v2` or `_data-refactored/v2engine_outputs`
- prefer explicit root overrides first; only generalize [_code-refactored/refactor_code/paths.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/paths.py) once the v3 root layout is stable
- set `TREE_TEMPLATE_ROOT` explicitly to the approved canonical variant root before any candidate export or verification run

Reason:

- path separation is the main protection against accidental overwrite
- it keeps v2 available as a live comparison target during the refactor
- template-root explicitness is required to preserve deadwood geometry parity with canonical v2

### 3. Add V3 State Fields In Parallel, Not By Immediate Replacement

Follow the model note and add the new V3 proposal fields without removing the old ones yet.

Actions:

- keep existing v2-compatible fields while adding the new V3 fields
- introduce the proposal decision/intervention fields described in the model note
- keep legacy export fields where downstream code still needs them

Important examples from the model note:

- `control_realized` -> `control_reached`
- `lifecycle_decision` -> `proposal-decay_decision`
- `pruning_target` -> `proposal-release-control_intervention`
- `release_control_support` -> `proposal-release-control_support`
- `colonise_support` -> `proposal-colonise-interventions`

Reason:

- this lets v3 outputs coexist with current downstream readers
- it makes verification possible before final cleanup

### 4. Implement Proposal Logic In Engine Order

Recommended order:

1. proposal-decay
2. proposal-release-control
3. proposal-colonise
4. proposal-recruit
5. proposal-deploy-structure

Reason:

- proposal-decay and release-control define the under-node treatment state
- colonise and recruit then read that state
- deploy-structure is comparatively self-contained

### 5. Introduce V3 Proposal Arrays On The Xarray / VTK Side

Do not replace the old proposal arrays immediately.

Add:

- `proposal_decayV3`
- `proposal_release_controlV3`
- `proposal_coloniseV3`
- `proposal_recruitV3`
- `proposal_deploy_structureV3`

Add matching intervention arrays:

- `proposal_decayV3_intervention`
- `proposal_release_controlV3_intervention`
- `proposal_coloniseV3_intervention`
- `proposal_recruitV3_intervention`
- `proposal_deploy_structureV3_intervention`

Primary files to update:

- [final/a_scenario_generateVTKs.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_generateVTKs.py)
- [final/a_info_gather_capabilities.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_info_gather_capabilities.py)

Important rule:

- keep the current `proposal_*` arrays alive until V3 has been validated
- use the V3 arrays as the explicit candidate truth during the refactor

### 6. Update Reporting And Render Logic After The V3 Arrays Exist

Actions:

- update the comparison/pathway builders so they can read the V3 proposal vocabulary
- update proposal render logic so it can display the V3 intervention labels cleanly
- prefer adding a V3-specific proposal view first rather than silently redefining the existing one

Recommended temporary render view name:

- `proposal-hybrid-v3`

Reason:

- this avoids confusion while V2 and V3 are both live
- it makes visual comparison easier during validation

### 7. Only After Verification, Collapse Back To One Proposal Source Of Truth

Do this last.

Actions:

- once the V3 arrays and pathway outputs are accepted, decide whether:
  - V3 arrays replace the old `proposal_*` arrays directly
  - or downstream readers are migrated to consume the V3 arrays explicitly

Do not do this earlier.

## Concrete Work Plan

### Phase A. Branch And Path Setup

1. Create branch `engine-v3` from current `master`.
2. Reserve new output roots:
   - `data/revised/final-v3`
   - `_data-refactored/v3engine_outputs`
   - `_statistics-refactored-v3`
3. Copy the current v2 validation instructions and adapt them into a v3-specific validation note once the first candidate run exists.

### Phase B. Engine And Schema Refactor

1. Create `engine_v3.py` from `engine_v2.py`.
2. Add the V3 proposal decision/intervention fields in the tree, log, and pole tables.
3. Implement the proposal-decay / release-control / colonise / recruit / deploy-structure behavior described in the model note.
4. Preserve compatibility fields until the V3 outputs are proven.

### Phase C. Xarray / VTK Refactor

1. Add the V3 proposal arrays and V3 intervention arrays to the xarray state.
2. Write them into the generated VTKs.
3. Mirror them into the augmented `state_with_indicators` VTKs.
4. Add a V3 render view rather than overloading the old one during development.

### Phase D. Reporting Refactor

1. Update pathway comparison code to understand the new proposal/intervention vocabulary.
2. Update validation/reporting docs so V3 comparisons are always made against current canonical v2.
3. Keep the old V2 tables as the comparison baseline throughout the refactor.

### Phase E. Verification And Promotion

1. Run quick verification on a small targeted subset.
2. Fix obvious logic/output issues.
3. Run full verification across all sites and years.
4. Only then decide whether `final-v3` becomes the new canonical root.

## Quick Verification

Quick verification should happen before any all-sites full run.

Use these targets:

- `trimmed-parade`
- `city`

Use these years:

- `0`
- `30`
- `180`

Use both:

- `positive`
- `trending`

Reason:

- `trimmed-parade` is the fastest sensitive site for pathway drift
- `city` catches log/pole behavior and broader urban envelope logic
- `0 / 30 / 180` covers seeded baseline behavior, early path dependence, and final divergence

### Quick Verification Checks

#### A. Engine Output Sanity

Confirm for the quick-run subset:

- scenario CSVs exist
- `urban_features` VTKs exist
- `state_with_indicators` VTKs exist
- V3 proposal arrays are present on the augmented VTKs
- the run metadata records the approved `TREE_TEMPLATE_ROOT`

Fail quick verification immediately if:

- no template-root metadata is recorded
- the recorded root is `data/revised/trees`
- the recorded root is not the approved canonical variant bundle

#### B. Vocabulary Sanity

Inspect one VTK per proposal type and confirm the new arrays contain only expected labels.

Examples:

- `proposal_decayV3`
- `proposal_release_controlV3`
- `proposal_coloniseV3`
- `proposal_recruitV3`
- `proposal_deploy_structureV3`

And their intervention companions.

#### C. Repeatability Sanity

Run the same saved-state export twice for:

- `trimmed-parade / positive / yr30`
- `city / positive / yr30`

Confirm:

- same point counts
- `0` mismatches in:
  - `scenario_outputs`
  - `search_bioavailable`
  - `search_design_action`
  - `search_urban_elements`
- if added, the V3 proposal arrays should also match exactly

#### D. Quick Pathway Sanity

Build a quick year-180 comparison for `trimmed-parade` and `city` only.

Check:

- `positive` still exceeds `trending` in the expected key cells
- no intervention family is empty when it should be active
- no obviously impossible values appear

Sensitive cells to inspect first:

- `Lizard / Acquire Resources`
- `Lizard / Communicate`
- `Lizard / Reproduce`
- `Tree / Communicate`
- `Tree / Reproduce`

#### E. Quick Visual Sanity

Render only the quick subset first.

Confirm:

- V3 proposal colors are legible
- the V3 proposal view is not dominated by `other`
- expected interventions appear where the engine says they should

## Full Verification

Only do this after quick verification is acceptable.

### 1. Full Run Coverage

Run:

- all sites:
  - `trimmed-parade`
  - `city`
  - `uni`
- all states:
  - `baseline`
  - `positive`
  - `trending`
- all years:
  - `0, 10, 30, 60, 90, 120, 150, 180`

### 2. File Presence Checks

Expected per site:

- `16` scenario `urban_features` VTKs
- `17` augmented `state_with_indicators` VTKs
- per-site indicator CSVs

Expected total renders:

- `51` PNGs per view

If a V3-specific proposal view is used during development, confirm:

- `classic`
- `merged`
- `proposal-hybrid-v3`

### 3. Rolled-Up Comparisons

Produce:

- one full V3 pathway table
- one direct delta against current canonical v2

Recommended names:

- `comparison_pathways_indicators_v3.csv`
- `comparison_pathways_indicators_v3.md`
- `comparison_pathways_v2_vs_v3.csv`
- `comparison_pathways_v2_vs_v3.md`

### 4. Required Divergence Reporting

For each year-180 cell, report:

- site
- persona
- capability
- what the indicator measures
- old divergence from canonical v2
- new divergence from V3
- reason for the difference

Use the existing sentence structure from [validation.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/validation.md).

### 5. Full Acceptance Criteria

Do not promote V3 unless all of these are true:

- the quick repeatability checks pass
- all expected VTKs and CSVs exist
- all render sequences exist
- the full V3 pathway table exists
- the full `v2 vs v3` delta exists
- no site loses expected proposal coverage because of labeling or assignment bugs
- no large divergence appears without a model explanation

## Verification Artifacts To Save

Do not leave verification only in chat.

Save these on disk for each serious V3 candidate:

- quick verification note
- full verification note
- verification counts CSV
- repeatability check note
- `v2 vs v3` comparison table

Recommended tracked doc paths:

- `_documentation-refactored/scenario_engine_v3_status.md`
- `_documentation-refactored/scenario_engine_v3_validation.md`

Recommended generated verification paths:

- `_data-refactored/v3engine_outputs/validation/verification_summary.md`
- `_data-refactored/v3engine_outputs/validation/verification_counts.csv`

At minimum, the tracked validation note should record:

- which candidate roots were checked
- which years/sites/pathways were in quick verification
- whether repeatability passed
- whether full file counts passed
- where the `v2 vs v3` deltas live

## Promotion Checklist

When V3 is accepted, do these in order:

1. freeze the accepted V3 candidate roots
2. write the final `v2 vs v3` comparison summary
3. update the canonical definition in the status note
4. point the validation note at the new canonical V3 roots
5. only then promote `final-v3` and `v3engine_outputs` as canonical
6. keep `final-v2` and `v2engine_outputs` intact until the first canonical V3 rerun is confirmed

## Promotion Rules

If V3 is accepted:

1. keep current v2 roots as historical reference until the v3 promotion is confirmed
2. promote:
   - `data/revised/final-v3`
   - `_data-refactored/v3engine_outputs`
   - `_statistics-refactored-v3`
3. update the status and validation notes to point to the new canonical roots
4. only then decide whether the old V2 proposal arrays can be removed

If V3 is not accepted:

- keep `final-v3` as a candidate root
- keep `final-v2` as canonical
- iterate on the `engine-v3` branch

## Recommended First Implementation Order

If the work starts now, the most pragmatic order is:

1. branch `engine-v3`
2. create `engine_v3.py`
3. route writes to `final-v3` / `v3engine_outputs` / `_statistics-refactored-v3`
4. add V3 decision/intervention fields in parallel
5. implement V3 proposal arrays on xarray / VTK
6. quick verification on `trimmed-parade` and `city`
7. update pathway builders and renderers
8. full verification across all sites
9. decide promotion

This keeps the refactor staged, reviewable, and reversible.

## Execution Recommendation

This refactor should be run as one main process, not as a fully split multi-subagent implementation.

Reason:

- the engine logic, xarray assignment, VTK proposal arrays, and pathway reporting are tightly coupled
- intermediate schema changes will ripple across multiple files
- parallel code edits are likely to create merge conflicts and verification confusion

Subagents are still useful for bounded sidecar work after the main implementation direction is set.

Good subagent tasks:

- verification counts and file-presence checks
- render-count checks
- pathway-delta writeups
- read-only codebase tracing for one specific proposal path

Bad subagent tasks:

- splitting proposal-decay / release-control / recruit / colonise implementation across workers at the same time
- parallel edits to shared files like the engine, VTK generator, and capability builder before the schema is stable
