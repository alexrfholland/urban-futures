# Nodeidfix Year-180 Divergences

The nodeidfix rerun keeps `positive` ahead of `trending` in all 27 year-180 cells. Relative to the older v1 engine, the ground-facing Street and City indicators move back toward the earlier human-led split, while Parade changes more modestly.

| Site / Persona | Capability | v1 | nodeidfix | Change | Note |
| --- | --- | --- | --- | --- | --- |
| Parade / Bird | Acquire Resources | 3.28x (65.2% vs 19.9%) | 5.88x (46.4% vs 7.9%) | +2.60x | Arboreal resource split stays positive-led with a small ratio increase. |
| Parade / Bird | Communicate | 1.48x (61.0% vs 41.1%) | 1.39x (27.2% vs 19.6%) | -0.09x | Perchable canopy stays similar; this is mostly unchanged. |
| Parade / Bird | Reproduce | 11.27x (92.5% vs 8.2%) | 104.44x (52.5% vs 0.5%) | +93.17x | Hollows remain strongly positive-led and drift further away from v1. |
| Parade / Lizard | Acquire Resources | 8.06x (72.9% vs 9.0%) | 18.81x (54.9% vs 2.9%) | +10.75x | Ground resource totals expand sharply in nodeidfix relative to v1 and v2. |
| Parade / Lizard | Communicate | 1.59x (81.7% vs 51.4%) | 1.43x (73.6% vs 51.4%) | -0.16x | Non-paved surface remains moderate and stable. |
| Parade / Lizard | Reproduce | 439.14x (27.5% vs 0.1%) | ∞x (15.1% vs 0.0%) | 439.14x -> ∞ | Fallen-tree habitat stays absent on trending and remains positive-led. |
| Parade / Tree | Acquire Resources | ∞x (55.5% vs 0.0%) | ∞x (94.9% vs 0.0%) | ∞ -> ∞ | Senescent biovolume remains exclusive to positive. |
| Parade / Tree | Communicate | 1.83x (81.3% vs 44.4%) | 1.67x (73.5% vs 43.9%) | -0.16x | Soil-near-canopy coverage stays positive-led with a slightly lower ratio. |
| Parade / Tree | Reproduce | 710.56x (59.6% vs 0.1%) | 2204.86x (50.3% vs 0.0%) | +1494.30x | Recruitment grassland remains strongly positive-led, but less extreme than v2. |
| Street / Bird | Acquire Resources | 5.22x (33.9% vs 6.5%) | 12.59x (34.7% vs 2.8%) | +7.37x | Bark-bearing canopy remains positive-led and slightly widens. |
| Street / Bird | Communicate | 2.39x (37.9% vs 15.8%) | 3.40x (35.6% vs 10.5%) | +1.01x | Perchable canopy stays positive-led with modest change. |
| Street / Bird | Reproduce | 5.84x (43.1% vs 7.4%) | 12.27x (45.3% vs 3.7%) | +6.43x | Hollows remain positive-led and strengthen relative to v1. |
| Street / Lizard | Acquire Resources | 14.28x (88.2% vs 6.2%) | 21.41x (86.5% vs 4.0%) | +7.13x | This is the main v2 inflation that the fix reverses toward v1. |
| Street / Lizard | Communicate | 3.64x (153.0% vs 42.1%) | 3.60x (150.5% vs 41.8%) | -0.04x | Non-paved surface now sits close to the old v1 split. |
| Street / Lizard | Reproduce | 15.00x (11.4% vs 0.8%) | 18.36x (8.3% vs 0.4%) | +3.36x | Nurse-log/fallen-tree split stays positive-led and changes little. |
| Street / Tree | Acquire Resources | 5.63x (12.5% vs 2.2%) | ∞x (25.8% vs 0.0%) | 5.63x -> ∞ | Senescent biovolume stays exclusive to positive. |
| Street / Tree | Communicate | 3.61x (62.5% vs 17.3%) | 3.65x (60.2% vs 16.5%) | +0.04x | Soil-near-canopy returns close to the v1 ratio. |
| Street / Tree | Reproduce | 12.60x (44.0% vs 3.5%) | 12.51x (41.5% vs 3.3%) | -0.09x | Grassland-for-recruitment returns close to the v1 ratio. |
| City / Bird | Acquire Resources | 2.11x (34.5% vs 16.4%) | 8.22x (22.7% vs 2.8%) | +6.11x | Bark-bearing canopy stays positive-led and grows relative to v1. |
| City / Bird | Communicate | 1.47x (38.7% vs 26.3%) | 3.73x (22.5% vs 6.0%) | +2.26x | Perchable canopy stays positive-led with a modest increase. |
| City / Bird | Reproduce | 2.71x (54.5% vs 20.1%) | 10.53x (30.0% vs 2.8%) | +7.82x | Hollows remain positive-led and strengthen relative to v1. |
| City / Lizard | Acquire Resources | 4.95x (97.9% vs 19.8%) | 8.16x (85.9% vs 10.5%) | +3.21x | Ground-resource split remains positive-led but is far closer to v1 than v2. |
| City / Lizard | Communicate | 6.72x (165.0% vs 24.5%) | 6.86x (159.5% vs 23.2%) | +0.14x | Non-paved surface remains close to parity and near v1. |
| City / Lizard | Reproduce | 5.01x (27.2% vs 5.4%) | 5.21x (19.3% vs 3.7%) | +0.20x | Fallen-tree habitat remains positive-led, slightly above v1. |
| City / Tree | Acquire Resources | 2.89x (21.9% vs 7.6%) | 54.08x (25.8% vs 0.5%) | +51.19x | Senescent biovolume stays strongly positive-led. |
| City / Tree | Communicate | 4.10x (45.1% vs 11.0%) | 4.08x (39.9% vs 9.8%) | -0.02x | Soil-near-canopy returns almost exactly to the v1 ratio. |
| City / Tree | Reproduce | 6.43x (51.7% vs 8.0%) | 7.13x (46.0% vs 6.5%) | +0.70x | Grassland-for-recruitment remains positive-led with a slightly smaller gap than v1. |

## Summary

- `positive` outranks `trending` in all 27 cells after the nodeidfix patch.
- The largest reversion toward v1 is in Street and City ground indicators, especially lizard/tree communicate and tree reproduce.
- Parade still shifts, but mostly on the arboreal side and less on the ground-envelope side.
