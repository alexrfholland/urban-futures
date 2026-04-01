# Validation

This note is for subagents validating a scenario-engine run against the current canonical v2.

Use it when a run has already produced scenario outputs and you need to confirm:

- the augmented indicator VTKs exist
- the per-site indicator CSVs exist
- the render sequences exist
- the rolled-up pathway comparison exists
- the direct delta against canonical v2 exists

If the run root is ambiguous, stop and ask the supervising agent which run is authoritative before proceeding.

## Canonical References

Current canonical v2 comparison files:

- [comparison_pathways_indicators_v2.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_v2.csv)
- [comparison_pathways_indicators_v2.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_v2.md)
- [comparison_pathways_v2_deltas.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_v2_deltas.md)

Current canonical v2 engine-output root:

- [_data-refactored/v2engine_outputs](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs)

Current canonical v2 scenario-output root:

- [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)

## Preflight Checks

### 1. Confirm The Run Root

Write down the run you are validating.

You need these two roots:

- scenario-output root
- engine-output root

Example refreshed run:

- scenario-output root: [data/revised/final-v2-template-edits](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2-template-edits)
- engine-output root: [_data-refactored/v2engine_outputs-template-edits](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_data-refactored/v2engine_outputs-template-edits)

If those roots are not explicit, stop and ask the supervising agent.

### 2. Ensure `state_with_indicators` VTKs Exist

Check the augmented VTK directory:

- `{engine-output-root}/vtks/trimmed-parade`
- `{engine-output-root}/vtks/city`
- `{engine-output-root}/vtks/uni`

Expected count per site:

- `17` VTKs per site

Meaning:

- `1` baseline VTK
- `8` positive years: `0, 10, 30, 60, 90, 120, 150, 180`
- `8` trending years: `0, 10, 30, 60, 90, 120, 150, 180`

Expected naming pattern:

- `{site}_baseline_1_state_with_indicators.vtk`
- `{site}_positive_1_yr{year}_state_with_indicators.vtk`
- `{site}_trending_1_yr{year}_state_with_indicators.vtk`

If a site does not have `17` files, stop and report exactly which files are missing.

If you are unsure whether a partial set is acceptable, ask the supervising agent before continuing.

### 3. Ensure Per-Site Indicator CSVs Exist

Check:

- `{scenario-output-root}/output/csv/trimmed-parade_1_indicator_counts.csv`
- `{scenario-output-root}/output/csv/city_1_indicator_counts.csv`
- `{scenario-output-root}/output/csv/uni_1_indicator_counts.csv`

Optional companion files:

- `*_action_counts.csv`

Minimum requirement:

- all three `*_indicator_counts.csv` files exist

If they do not exist, stop and report that the capability pass is incomplete.

## Image Validation

If the render sequence does not already exist, generate it from the augmented VTKs.

Expected render root:

- `{engine-output-root}/validation/renders`

Expected folders:

- `classic`
- `merged`
- `proposal-hybrid`

Expected count per folder:

- `51` PNGs

Meaning:

- `17` states per site
- `3` sites

Expected naming pattern:

- `{site}_{scenario}_yr{year}_classic.png`
- `{site}_{scenario}_yr{year}_merged.png`
- `{site}_{scenario}_yr{year}_proposal-hybrid.png`
- baseline uses `scenario=baseline`, `year=0`

If images are missing:

1. confirm the `state_with_indicators` VTKs exist
2. rerun the render script for the target root
3. report final counts per folder

## Divergence Validation

You must produce two things:

### 1. Rolled-Up Pathway Comparison For The Run

This is the run’s own year-180 table.

Use:

- [build_comparison_pathways_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/build_comparison_pathways_v2.py)

Input:

- `{scenario-output-root}/output/csv`

Required outputs:

- one CSV
- one markdown table

Suggested naming:

- `comparison_pathways_indicators_<run-name>.csv`
- `comparison_pathways_indicators_<run-name>.md`

The markdown should preserve the existing style:

- simple
- concrete
- one line of explanation per cell

### 2. Direct Delta Against Current Canonical v2

This is the comparison between:

- canonical v2: `comparison_pathways_indicators_v2.csv`
- candidate run: `comparison_pathways_indicators_<run-name>.csv`

Required outputs:

- one CSV
- one markdown summary

Suggested naming:

- `comparison_pathways_v2_vs_<run-name>.csv`
- `comparison_pathways_v2_vs_<run-name>.md`

## Required Reporting Template

For each year-180 cell, report:

- site
- persona
- capability
- what the indicator measures
- old divergence
- new divergence
- reason for the difference

Use this exact structure:

- `{Site} / {Persona} / {Capability}: old {old_ratio}x ({old_positive_pct}% nonhuman-led baseline vs {old_trending_pct}% human-led baseline), measured {measure}; new {new_ratio}x ({new_positive_pct}% nonhuman-led baseline vs {new_trending_pct}% human-led baseline), measured {measure}; reason: {reason}.`

Example:

- `Street / Lizard / Communicate: old 3.64x (153.0% nonhuman-led baseline vs 42.1% human-led baseline), measured non-paved surface area; new 3.60x (150.5% nonhuman-led baseline vs 41.8% human-led baseline), measured non-paved surface area; reason: the node-id fix returned the human-led ground coverage close to the earlier split.`

Rules:

- `nonhuman-led` means `positive`
- `human-led` means `trending`
- percentages are always `% of the site baseline`
- use the canonical v2 table as the old reference when you are comparing a new run against current v2
- use the legacy v1 markdown only when the task explicitly asks for v1 versus current

## What To Flag

Flag these immediately:

- any site missing augmented VTKs
- any missing `*_indicator_counts.csv`
- any render folder missing expected images
- any cell where `positive` no longer exceeds `trending`
- any large ground-indicator shift in:
  - `Lizard / Acquire Resources`
  - `Lizard / Communicate`
  - `Tree / Communicate`
  - `Tree / Reproduce`

These are the sensitive cells that previously exposed the `NodeID = -1` bug.

## Minimum Deliverables

Do not report the validation as complete unless you have all of:

- confirmed augmented VTK counts
- confirmed per-site indicator CSVs
- generated or confirmed render sequences
- generated the rolled-up pathway comparison for the run
- generated the direct delta against canonical v2
- written the year-180 divergence summary in the required sentence format
