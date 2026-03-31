# Comparison Pathways v2 vs Template Edits

The template-edits run changes only the fallen/snag templates. The sim core, scenario CSVs, and year-180 pathway direction stay the same. Positive remains ahead of trending in all 27 cells; there are no sign flips relative to the current v2 run.

The main caution is denominator drift. Because the baseline is rebuilt from the edited templates, some percentage changes are driven as much by the new baseline as by the year-180 scenario counts.

Largest increases in positive-vs-trending separation:
- Parade / Tree / Reproduce: v2 4142.51x grassland-for-recruitment indicator (94.6% vs 0.0%) -> template-edits 7006.24x grassland-for-recruitment indicator (159.9% vs 0.0%)
- Parade / Bird / Reproduce: v2 104.44x hollow count (52.5% vs 0.5%) -> template-edits 135.11x hollow count (57.6% vs 0.4%)
- Parade / Lizard / Acquire Resources: v2 32.07x combined ground-cover/dead branch/epiphyte indicators (96.8% vs 3.0%) -> template-edits 48.07x combined ground-cover/dead branch/epiphyte indicators (145.5% vs 3.0%)
- Street / Bird / Reproduce: v2 11.84x hollow count (45.0% vs 3.8%) -> template-edits 15.58x hollow count (54.6% vs 3.5%)
- Parade / Tree / Communicate: v2 2.29x soil near canopy features (100.6% vs 43.9%) -> template-edits 3.68x soil near canopy features (162.1% vs 44.1%)

Largest decreases in positive-vs-trending separation:
- Street / Lizard / Reproduce: v2 16.03x combined nurse-log/fallen-tree indicators (8.0% vs 0.5%) -> template-edits 10.06x combined nurse-log/fallen-tree indicators (23.6% vs 2.3%)
- City / Bird / Reproduce: v2 10.45x hollow count (29.8% vs 2.8%) -> template-edits 9.05x hollow count (31.8% vs 3.5%)
- Street / Bird / Acquire Resources: v2 11.41x peeling bark volume (35.1% vs 3.1%) -> template-edits 10.09x peeling bark volume (31.1% vs 3.1%)
- City / Lizard / Reproduce: v2 5.05x combined nurse-log/fallen-tree indicators (19.1% vs 3.8%) -> template-edits 4.41x combined nurse-log/fallen-tree indicators (50.5% vs 11.5%)
- Street / Bird / Communicate: v2 3.35x perchable canopy volume (36.4% vs 10.8%) -> template-edits 2.91x perchable canopy volume (33.1% vs 11.4%)

## Parade

### Bird

- Acquire Resources: v2 5.53x peeling bark volume (45.9% vs 8.3%) -> template-edits 5.41x peeling bark volume (45.6% vs 8.4%). Delta: ratio -0.12, positive baseline share -0.3 points, trending baseline share +0.1 points. Direct arboreal-resource effect. Snag and fallen template edits change bark, hollow, or senescent voxels in both the scenario and baseline.
- Communicate: v2 1.40x perchable canopy volume (27.7% vs 19.8%) -> template-edits 1.23x perchable canopy volume (25.9% vs 21.0%). Delta: ratio -0.17, positive baseline share -1.8 points, trending baseline share +1.2 points. Mostly indirect. The sim path is unchanged; the canopy-support footprint shifts because the edited arboreal templates occupy different voxels.
- Reproduce: v2 104.44x hollow count (52.5% vs 0.5%) -> template-edits 135.11x hollow count (57.6% vs 0.4%). Delta: ratio +30.67, positive baseline share +5.1 points, trending baseline share -0.1 points. Direct arboreal-resource effect. Snag and fallen template edits change bark, hollow, or senescent voxels in both the scenario and baseline.

### Lizard

- Acquire Resources: v2 32.07x combined ground-cover/dead branch/epiphyte indicators (96.8% vs 3.0%) -> template-edits 48.07x combined ground-cover/dead branch/epiphyte indicators (145.5% vs 3.0%). Delta: ratio +16.00, positive baseline share +48.7 points, trending baseline share +0.0 points. Mostly a deadwood-template effect. The old-snag geometry and elm-fallen swap change resource voxels, then the baseline moves with them.
- Communicate: v2 2.90x non-paved surface area (148.9% vs 51.4%) -> template-edits 4.10x non-paved surface area (210.4% vs 51.4%). Delta: ratio +1.20, positive baseline share +61.5 points, trending baseline share +0.0 points. Mostly indirect. Ground and overlap indicators move because the edited tree footprints change where arboreal voxels intersect the search masks, and the baseline denominator moves too.
- Reproduce: v2 ∞x combined nurse-log/fallen-tree indicators (15.3% vs 0.0%) -> template-edits ∞x combined nurse-log/fallen-tree indicators (71.1% vs 0.0%). Delta: ratio +0.00, positive baseline share +55.8 points, trending baseline share +0.0 points. Direct fallen-template effect. The new fallen geometry changes fallen-tree coverage and also shifts the baseline denominator.

### Tree

- Acquire Resources: v2 ∞x senescent biovolume (97.9% vs 0.0%) -> template-edits ∞x senescent biovolume (97.1% vs 0.0%). Delta: ratio +0.00, positive baseline share -0.8 points, trending baseline share +0.0 points. Direct arboreal-resource effect. Snag and fallen template edits change bark, hollow, or senescent voxels in both the scenario and baseline.
- Communicate: v2 2.29x soil near canopy features (100.6% vs 43.9%) -> template-edits 3.68x soil near canopy features (162.1% vs 44.1%). Delta: ratio +1.39, positive baseline share +61.5 points, trending baseline share +0.2 points. Mostly indirect. Ground and overlap indicators move because the edited tree footprints change where arboreal voxels intersect the search masks, and the baseline denominator moves too.
- Reproduce: v2 4142.51x grassland-for-recruitment indicator (94.6% vs 0.0%) -> template-edits 7006.24x grassland-for-recruitment indicator (159.9% vs 0.0%). Delta: ratio +2863.73, positive baseline share +65.3 points, trending baseline share +0.0 points. Mostly indirect. Ground and overlap indicators move because the edited tree footprints change where arboreal voxels intersect the search masks, and the baseline denominator moves too.

## Street

### Bird

- Acquire Resources: v2 11.41x peeling bark volume (35.1% vs 3.1%) -> template-edits 10.09x peeling bark volume (31.1% vs 3.1%). Delta: ratio -1.32, positive baseline share -4.0 points, trending baseline share +0.0 points. Direct arboreal-resource effect. Snag and fallen template edits change bark, hollow, or senescent voxels in both the scenario and baseline.
- Communicate: v2 3.35x perchable canopy volume (36.4% vs 10.8%) -> template-edits 2.91x perchable canopy volume (33.1% vs 11.4%). Delta: ratio -0.44, positive baseline share -3.3 points, trending baseline share +0.6 points. Mostly indirect. The sim path is unchanged; the canopy-support footprint shifts because the edited arboreal templates occupy different voxels.
- Reproduce: v2 11.84x hollow count (45.0% vs 3.8%) -> template-edits 15.58x hollow count (54.6% vs 3.5%). Delta: ratio +3.74, positive baseline share +9.6 points, trending baseline share -0.3 points. Direct arboreal-resource effect. Snag and fallen template edits change bark, hollow, or senescent voxels in both the scenario and baseline.

### Lizard

- Acquire Resources: v2 1.26x combined ground-cover/dead branch/epiphyte indicators (155.1% vs 123.1%) -> template-edits 1.28x combined ground-cover/dead branch/epiphyte indicators (161.6% vs 126.2%). Delta: ratio +0.02, positive baseline share +6.5 points, trending baseline share +3.1 points. Mostly a deadwood-template effect. The old-snag geometry and elm-fallen swap change resource voxels, then the baseline moves with them.
- Communicate: v2 1.02x non-paved surface area (282.0% vs 275.7%) -> template-edits 1.07x non-paved surface area (297.9% vs 277.5%). Delta: ratio +0.05, positive baseline share +15.9 points, trending baseline share +1.8 points. Mostly indirect. Ground and overlap indicators move because the edited tree footprints change where arboreal voxels intersect the search masks, and the baseline denominator moves too.
- Reproduce: v2 16.03x combined nurse-log/fallen-tree indicators (8.0% vs 0.5%) -> template-edits 10.06x combined nurse-log/fallen-tree indicators (23.6% vs 2.3%). Delta: ratio -5.97, positive baseline share +15.6 points, trending baseline share +1.8 points. Direct fallen-template effect. The new fallen geometry changes fallen-tree coverage and also shifts the baseline denominator.

### Tree

- Acquire Resources: v2 ∞x senescent biovolume (26.6% vs 0.0%) -> template-edits ∞x senescent biovolume (25.4% vs 0.0%). Delta: ratio +0.00, positive baseline share -1.2 points, trending baseline share +0.0 points. Direct arboreal-resource effect. Snag and fallen template edits change bark, hollow, or senescent voxels in both the scenario and baseline.
- Communicate: v2 1.52x soil near canopy features (103.4% vs 67.9%) -> template-edits 1.71x soil near canopy features (119.3% vs 69.9%). Delta: ratio +0.19, positive baseline share +15.9 points, trending baseline share +2.0 points. Mostly indirect. Ground and overlap indicators move because the edited tree footprints change where arboreal voxels intersect the search masks, and the baseline denominator moves too.
- Reproduce: v2 1.99x grassland-for-recruitment indicator (104.0% vs 52.3%) -> template-edits 2.19x grassland-for-recruitment indicator (121.1% vs 55.3%). Delta: ratio +0.20, positive baseline share +17.1 points, trending baseline share +3.0 points. Mostly indirect. Ground and overlap indicators move because the edited tree footprints change where arboreal voxels intersect the search masks, and the baseline denominator moves too.

## City

### Bird

- Acquire Resources: v2 7.91x peeling bark volume (22.7% vs 2.9%) -> template-edits 8.57x peeling bark volume (22.1% vs 2.6%). Delta: ratio +0.66, positive baseline share -0.6 points, trending baseline share -0.3 points. Direct arboreal-resource effect. Snag and fallen template edits change bark, hollow, or senescent voxels in both the scenario and baseline.
- Communicate: v2 3.52x perchable canopy volume (22.3% vs 6.3%) -> template-edits 3.55x perchable canopy volume (21.6% vs 6.1%). Delta: ratio +0.03, positive baseline share -0.7 points, trending baseline share -0.2 points. Mostly indirect. The sim path is unchanged; the canopy-support footprint shifts because the edited arboreal templates occupy different voxels.
- Reproduce: v2 10.45x hollow count (29.8% vs 2.8%) -> template-edits 9.05x hollow count (31.8% vs 3.5%). Delta: ratio -1.40, positive baseline share +2.0 points, trending baseline share +0.7 points. Direct arboreal-resource effect. Snag and fallen template edits change bark, hollow, or senescent voxels in both the scenario and baseline.

### Lizard

- Acquire Resources: v2 1.11x combined ground-cover/dead branch/epiphyte indicators (209.2% vs 188.3%) -> template-edits 1.20x combined ground-cover/dead branch/epiphyte indicators (236.7% vs 196.6%). Delta: ratio +0.09, positive baseline share +27.5 points, trending baseline share +8.3 points. Mostly a deadwood-template effect. The old-snag geometry and elm-fallen swap change resource voxels, then the baseline moves with them.
- Communicate: v2 1.03x non-paved surface area (440.7% vs 426.8%) -> template-edits 1.09x non-paved surface area (471.4% vs 434.5%). Delta: ratio +0.06, positive baseline share +30.7 points, trending baseline share +7.7 points. Mostly indirect. Ground and overlap indicators move because the edited tree footprints change where arboreal voxels intersect the search masks, and the baseline denominator moves too.
- Reproduce: v2 5.05x combined nurse-log/fallen-tree indicators (19.1% vs 3.8%) -> template-edits 4.41x combined nurse-log/fallen-tree indicators (50.5% vs 11.5%). Delta: ratio -0.64, positive baseline share +31.4 points, trending baseline share +7.7 points. Direct fallen-template effect. The new fallen geometry changes fallen-tree coverage and also shifts the baseline denominator.

### Tree

- Acquire Resources: v2 58.77x senescent biovolume (26.3% vs 0.4%) -> template-edits 59.06x senescent biovolume (27.0% vs 0.5%). Delta: ratio +0.29, positive baseline share +0.7 points, trending baseline share +0.1 points. Direct arboreal-resource effect. Snag and fallen template edits change bark, hollow, or senescent voxels in both the scenario and baseline.
- Communicate: v2 1.15x soil near canopy features (113.1% vs 98.6%) -> template-edits 1.35x soil near canopy features (143.7% vs 106.3%). Delta: ratio +0.20, positive baseline share +30.6 points, trending baseline share +7.7 points. Mostly indirect. Ground and overlap indicators move because the edited tree footprints change where arboreal voxels intersect the search masks, and the baseline denominator moves too.
- Reproduce: v2 1.48x grassland-for-recruitment indicator (118.2% vs 80.1%) -> template-edits 1.64x grassland-for-recruitment indicator (152.2% vs 92.7%). Delta: ratio +0.16, positive baseline share +34.0 points, trending baseline share +12.6 points. Mostly indirect. Ground and overlap indicators move because the edited tree footprints change where arboreal voxels intersect the search masks, and the baseline denominator moves too.

