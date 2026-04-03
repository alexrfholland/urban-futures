# Scenario Engine Statistics Comparison

This note is the current source of truth for the pathway statistics comparison.

Scope:

- year-180 capability statistics only
- not VTK/file-count/render validation

## Current Comparison Basis

Current branch:

- `engine-v3`

Current candidate scenario-output root:

- [data/revised/final-v3](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v3)

Current canonical comparison target:

- [data/revised/final-v2](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final-v2)

Statistics source used for the rolled-up pathway tables:

- `{scenario-output-root}/output/csv/*_indicator_counts.csv`

## What `comparison_pathways` Means Now

The folder [final/assesment/comparison_pathways](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways) now contains both historical pathway notes and current generated comparison outputs.

Use these meanings:

Historical base only, not the live comparison target:

- [comparison_pathways_indicators.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators.md)

Current canonical v2 statistics table:

- [comparison_pathways_indicators_v2.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_v2.csv)
- [comparison_pathways_indicators_v2.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_v2.md)

Current v3 candidate statistics table:

- [comparison_pathways_indicators_v3.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_v3.csv)
- [comparison_pathways_indicators_v3.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_v3.md)

Current direct delta that should be cited for v3 discussion:

- [comparison_pathways_v2_vs_v3.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_v2_vs_v3.csv)
- [comparison_pathways_v2_vs_v3.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_v2_vs_v3.md)

Archive snapshot only:

- [v3_2026-04-02](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/v3_2026-04-02)

As checked on April 3, 2026, the dated `v3_2026-04-02` copies are byte-identical to the root-level `comparison_pathways_indicators_v3.*` and `comparison_pathways_v2_vs_v3.*` files. Treat the dated folder as an archive snapshot, not the default citation target.

## What We Say

- `comparison_pathways_indicators.md` was the original base narrative, but it is no longer the live statistics comparison document.
- For current reporting, compare [comparison_pathways_indicators_v3.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_v3.md) against [comparison_pathways_indicators_v2.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators_v2.md), and cite the delta in [comparison_pathways_v2_vs_v3.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_v2_vs_v3.md).
- The current statistics story is directional stability, not sign reversal: `positive` remains above `trending` in all `27` year-180 cells in the current v3 candidate table.
- The live question is magnitude drift between canonical v2 and current v3, especially in deadwood-, senescence-, and recruitment-sensitive cells.

## Sensitive Current Drift

The current v2 vs v3 delta file shows the main compression or semantic drift in these cells:

- `Parade / Tree / Reproduce`: `5095.65x` in v2 to `10.69x` in v3, because `positive` fell from `88.4%` to `80.5%` of baseline and `trending` rose from `0.0%` to `7.5%`.
- `Parade / Lizard / Acquire Resources`: `33.81x` in v2 to `13.51x` in v3.
- `Street / Tree / Acquire Resources`: `53.46x` in v2 to `13.70x` in v3.
- `City / Tree / Acquire Resources`: `69.31x` in v2 to `18.81x` in v3.

These are the cells to use when describing current statistical divergence between canonical v2 and the current v3 candidate.

## Working Rule

- Use the root-level `v2`, `v3`, and `v2_vs_v3` files in `comparison_pathways` for current reporting.
- Treat the original [comparison_pathways_indicators.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/assesment/comparison_pathways/comparison_pathways_indicators.md) as background only.
- Keep render/file-completeness validation separate from statistics comparison.
- If a new candidate run is generated, write `comparison_pathways_indicators_<run-name>.*` and `comparison_pathways_v2_vs_<run-name>.*` beside the existing files, then only promote them to the root-level `v3` names once that run is the accepted current comparison.
