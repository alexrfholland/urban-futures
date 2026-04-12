# Compositor Sync Contract

This note defines how the refactored compositor should relate to:

- simulation roots
- Blender v2 EXR families
- compositor runs

Use this alongside:

- [COMPOSITOR_TEMPLATE_CONTRACT.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/COMPOSITOR_TEMPLATE_CONTRACT.md)
- [_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/MEDIAFLUX_SYNC_CONTRACT.md)

## Lineage

The lineage is:

- `sim_root`
- `exr_family`
- `compositor_run`

Compositor outputs are children of one EXR family.

They are not peers of the EXR family and they are not standalone top-level run
roots.

## Inputs

The compositor input is a chosen EXR family.

That EXR family comes from Blender v2 output:

- local:
  - `_data-refactored/blenderv2/output/<sim_root>/<exr_family>/`
- remote:
  - `pipeline/<sim_root>/blender_exrs/<exr_family>/`

The compositor should read EXRs directly from that family.

Do not create a separate compositor-input cache layer as part of the canonical
contract.

## Outputs

Canonical local compositor outputs should live under:

- `_data-refactored/compositor/outputs/<sim_root>/<exr_family>/<compositor_run>/`

Canonical remote compositor outputs should live under:

- `pipeline/<sim_root>/compositor_pngs/<exr_family>/<compositor_run>/`

Example:

- local:
  - `_data-refactored/compositor/outputs/v4.9/city_timeline__hero-test/mist__20260411_1730/`
- remote:
  - `pipeline/v4.9/compositor_pngs/city_timeline__hero-test/mist__20260411_1730/`

## EXR Family

The EXR family should align with the bV2 `case` contract.

Base form:

- `<case>`

Optional extended form:

- `<case>__<note>`

Examples:

- `city_timeline`
- `city_timeline__hero-test`
- `city_single-state_yr180`

Important rule:

- the EXR family folder may include an optional `__<note>`
- the EXR filenames inside still use the canonical bV2 `case`

So if:

- `exr_family = city_timeline__hero-test`

the EXR files inside still look like:

- `city_timeline__positive_state__8k64s.exr`
- `city_timeline__trending_state__8k64s.exr`

## Input Wiring On Mismatch

When the available EXR set does not match a canonical blend's EXR input
hooks 1:1 (wrong count, missing a view-layer the blend expects, extra
EXRs, ambiguous naming), runners and agents must not auto-resolve the
wiring. Silent remapping is how the wrong view-layer ends up feeding
the wrong hook without anyone noticing.

Instead the runner or agent should:

1. Open the canonical blend headless and **enumerate its EXR input hooks**. For each hook, report:
   - an index number
   - the input node name
   - the view-layer / semantic label the hook expects (e.g. `positive_state`, `arboreal_positive_mask`, `existing_condition`)
2. **Enumerate the available EXRs** in the chosen EXR family. For each EXR, report:
   - an index number
   - the filename
   - the view-layer it was rendered from
3. **Ask the user** which EXR fills which hook, or which hooks to leave empty. Present it as a numbered wiring choice and wait for the user's decision before rendering.

When the EXR set matches the hooks unambiguously (same count, same
view-layer names), the runner may proceed without asking.

This rule exists because:

- the user will always be the one triggering the final render, so pausing to confirm has no speed cost
- baseline EXR families legitimately lack trending and bioenvelope inputs; a suite blend built around a 7-input contract cannot be silently fed a 3-input baseline
- the same canonical blend may be run against several EXR families (baseline, single-state, timeline) whose shape differs

It is the concrete runtime expression of the Hidden Fallback Rule from
[COMPOSITOR_TEMPLATE_CONTRACT.md](COMPOSITOR_TEMPLATE_CONTRACT.md).

## Compositor Run

The compositor run name should be:

- `<compositor_family>__<timestamp>`

Optional extended form:

- `<compositor_family>__<timestamp>__<note>`

Examples:

- `mist__20260411_1730`
- `mist__20260411_1730__mask-fix`
- `final-template__20260411_1805`

## Public Selectors

The intended compositor-facing selectors are:

- `COMPOSITOR_SIM_ROOT`
- `COMPOSITOR_EXR_FAMILY`
- `COMPOSITOR_FAMILY`
- optional `COMPOSITOR_RUN_TIMESTAMP`
- optional `COMPOSITOR_RUN_NOTE`

## Current Runner Behavior

The main refactored runners now derive default roots from those selectors.

Main runners:

- [run_edge_lab_final_template.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/scripts/run_edge_lab_final_template.py)
- [instantiate_template_and_render.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/blender/compositor/scripts/instantiate_template_and_render.py)

If:

- `COMPOSITOR_SIM_ROOT=v4.9`
- `COMPOSITOR_EXR_FAMILY=city_timeline__hero-test`
- `COMPOSITOR_FAMILY=mist`

then the default output root becomes:

- `_data-refactored/compositor/outputs/v4.9/city_timeline__hero-test/mist__<timestamp>/`

and the default EXR input root becomes:

- `_data-refactored/blenderv2/output/v4.9/city_timeline__hero-test/`

## Scope Rule

This note is only about sync and path lineage.

It does not change the compositor ownership contract:

- canonical `.blend` files still own graph logic
- scripts still act as thin runners around those templates
