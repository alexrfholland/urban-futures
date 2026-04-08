# Growth Allometry Models

## Overview

The simulation engine supports multiple diameter-at-breast-height (DBH) growth models, selectable via the `growth_model` parameter. Each model computes new DBH values per simulation step using a virtual-age round-trip: the current DBH is mapped to an implied age, the age is advanced by the step duration, and the new age is mapped back to DBH.

The default (`constant`) preserves backward compatibility with the v3 engine. The allometric models replace it with empirically derived, size-dependent growth that accelerates early-life diameter accumulation and decelerates as trees mature.

A `split` mode applies different allometric curves to colonial and precolonial trees using the `precolonial` column in the tree dataframe, reflecting the different species composition of planted (predominantly elm) and remnant (predominantly eucalypt) urban trees.

## Model definitions

### `constant`

The original v3/v4 growth model. Applies a fixed annual DBH increment equal to the mean of `growth_factor_range` (default 0.44 cm/yr) regardless of tree size.

- **Source**: Not empirically derived. The value 0.44 cm/yr approximates observed growth rates for mature trees (DBH ~70--80 cm) but substantially underestimates growth for young trees.
- **Equation**: $\Delta\text{DBH} = \bar{g} \times \Delta t$, where $\bar{g} = \text{mean}(\texttt{growth\_factor\_range})$.

### `fischer`

Derived from Fischer et al. (2010) Supporting Information,[^fischer_si] which regressed tree age as a linear function of basal area using empirical data from Banks (1997).[^banks] The regression was calibrated across multiple species standardised to yellow-box-equivalent diameters. Fischer report $R^2 = 0.998$ ($P < 0.0001$).

This is the same equation used by Le Roux et al. (2014)[^leroux] to determine cohort residence times in their demographic model, from which our mortality rates are also derived. Using the Fischer growth curve therefore maintains internal consistency between the growth and mortality components of the simulation.

- **Equation**: $\text{Age} = 0.0197135 \times \pi \times (\text{DBH}/2)^2$
- **Inverse**: $\text{DBH} = \sqrt{\text{Age} / k}$, where $k = 0.0197135 \times \pi / 4 \approx 0.01548$
- **Implied growth rate**: $d\text{DBH}/dt = 32.26 / \text{DBH}$ cm/yr
- **Species**: *Eucalyptus melliodora* (yellow box) and other species standardised to yellow-box-equivalent diameters
- **Region**: Canberra, Australia

| DBH (cm) | Implied age (yr) | Growth rate (cm/yr) |
|----------:|------------------:|--------------------:|
| 2 | 0.1 | 16.13 |
| 10 | 1.5 | 3.23 |
| 30 | 13.9 | 1.08 |
| 50 | 38.7 | 0.65 |
| 80 | 99.1 | 0.40 |

Note: The growth rate at very small DBH (e.g. 16 cm/yr at DBH 2) is implausibly high. This is an artefact of the basal-area regression being calibrated primarily on mature cohorts. In practice, newly replaced trees at DBH 2 cm transit through the fastest-growth regime within a single simulation pulse, so the overshoot has limited effect on population-level outcomes.

### `sideroxylon`

*Eucalyptus sideroxylon* (red ironbark) growth equation from the US Forest Service Urban Tree Database (McPherson et al., 2016),[^mcpherson] Inland Empire region. A quadratic age-to-DBH relationship ($R^2 = 0.963$, $n = 61$).

Red ironbark is native to southeastern Australia and is a common urban planting in the same bioregion as the study sites. It grows faster through the mid-range than yellow box but converges at large diameters.

- **Equation**: $\text{DBH} = 4.850 + 1.821 \times \text{Age} - 0.011 \times \text{Age}^2$
- **Valid range**: DBH 4.85--80.6 cm (the quadratic reaches its maximum near 80 cm, aligning with the simulation's `large` threshold)
- **Species**: *Eucalyptus sideroxylon* (red ironbark)
- **Region**: Inland Empire, California (planted Australian species in urban setting)

| DBH (cm) | Implied age (yr) | Growth rate (cm/yr) |
|----------:|------------------:|--------------------:|
| 5 | 0.1 | 1.82 |
| 10 | 2.9 | 1.76 |
| 30 | 15.2 | 1.49 |
| 50 | 30.4 | 1.15 |
| 80 | 79.3 | 0.08 |

### `banks`

Power-law fit to four measured cohorts of *Eucalyptus melliodora* (yellow box) in Canberra, reported by Banks (1997).[^banks] This is the primary empirical source from which Fischer et al. derived their regression.

- **Equation**: $\text{DBH} = 4.50 \times \text{Age}^{0.6}$
- **Empirical data**:

| Cohort age (yr) | Observed DBH (cm) | Predicted DBH (cm) |
|-----------------:|-------------------:|--------------------:|
| 6 | 9.5 | 13.1 |
| 29 | 38.8 | 36.4 |
| 64 | 51.9 | 55.4 |
| 150 | 91.0 | 89.9 |

- **Implied growth rate**: $d\text{DBH}/dt = 2.70 \times \text{Age}^{-0.4}$
- **Species**: *Eucalyptus melliodora* (yellow box)
- **Region**: Canberra, Australia

| DBH (cm) | Implied age (yr) | Growth rate (cm/yr) |
|----------:|------------------:|--------------------:|
| 2 | 0.2 | 4.64 |
| 10 | 3.6 | 1.59 |
| 30 | 22.8 | 0.76 |
| 50 | 50.4 | 0.54 |
| 80 | 107.2 | 0.40 |

### `ulmus`

*Ulmus americana* (American elm) growth equation from the US Forest Service Urban Tree Database (McPherson et al., 2016),[^mcpherson] Pacific Northwest region. A quadratic age-to-DBH relationship ($R^2 = 0.982$, $n = 41$).

Elms (*Ulmus* spp.) are a dominant component of Melbourne's colonial street tree population. *Ulmus americana* in the Pacific Northwest serves as a proxy for the English and Dutch elms common in the study sites, following the approach of Croeser et al. (2025),[^croeser] who use this same equation set for urban canopy modelling.

- **Equation**: $\text{DBH} = -0.707 + 1.817 \times \text{Age} - 0.005 \times \text{Age}^2$
- **Valid range**: DBH 1.1--138.6 cm
- **Species**: *Ulmus americana* (American elm)
- **Region**: Pacific Northwest, USA

| DBH (cm) | Implied age (yr) | Growth rate (cm/yr) |
|----------:|------------------:|--------------------:|
| 2 | 1.5 | 1.80 |
| 10 | 5.9 | 1.76 |
| 30 | 17.6 | 1.64 |
| 50 | 30.2 | 1.51 |
| 80 | 51.7 | 1.30 |

### `split`

Not a growth curve itself but a routing mode that applies different allometric models to colonial and precolonial trees based on the `precolonial` boolean column in the tree dataframe.

- **Precolonial trees** (`precolonial == True`): uses the model specified by `growth_model_precolonial` (default: `fischer`)
- **Colonial trees** (`precolonial == False`): uses the model specified by `growth_model_colonial` (default: `ulmus`)

## Comparison of growth rates

All values in cm/yr of DBH increase.

| DBH (cm) | `constant` | `fischer` | `sideroxylon` | `banks` | `ulmus` |
|----------:|-----------:|----------:|--------------:|--------:|--------:|
| 2 | 0.44 | 16.13 | 1.82 | 4.64 | 1.80 |
| 10 | 0.44 | 3.23 | 1.76 | 1.59 | 1.76 |
| 30 | 0.44 | 1.08 | 1.49 | 0.76 | 1.64 |
| 50 | 0.44 | 0.65 | 1.15 | 0.54 | 1.51 |
| 80 | 0.44 | 0.40 | 0.08 | 0.40 | 1.30 |

## Parameter configuration

Set in the scenario parameter dictionary (see `params_v3.py`):

```python
"growth_model": "split",                    # or "constant", "fischer", "sideroxylon", "banks", "ulmus"
"growth_model_precolonial": "fischer",       # eucalypt curve for precolonial trees (split mode only)
"growth_model_colonial": "ulmus",            # elm curve for colonial trees (split mode only)
```

## Implementation

Growth models are implemented in `engine_v3.py` as stateless functions that accept a NumPy array of current DBH values, a step duration in years, and the parameter dictionary. Each returns an array of new DBH values. The virtual-age approach avoids the need to track tree age as a persistent column.

For quadratic models (`sideroxylon`, `ulmus`), a floor constraint ensures trees never shrink: if the quadratic form would produce a smaller DBH than the input (possible near the equation's maximum), the original DBH is retained.

## References

[^fischer_si]: Fischer, J., Stott, J., Zerger, A., Warren, G., Sherren, K., & Forrester, R. I. (2010). Reversing a tree regeneration crisis in an endangered ecoregion. *Proceedings of the National Academy of Sciences*, 107(25), 10386--10391. Supporting Information: Details on the Demographic Model and Its Calibration.

[^banks]: Banks, J. C. G. (1997). Tree ages and ageing in yellow box. In J. Dargavel (Ed.), *The Coming of Age---Forest Age and Heritage Values* (pp. 17--28). Environment Australia, Canberra.

[^leroux]: Le Roux, D. S., Ikin, K., Lindenmayer, D. B., Manning, A. D., & Gibbons, P. (2014). The future of large old trees in urban landscapes. *PLOS ONE*, 9(6), e99403.

[^mcpherson]: McPherson, E. G., van Doorn, N. S., & Peper, P. J. (2016). Urban tree database. In *Forest Service Research Data Archive*. doi: 10.2737/RDS-2016-0005.

[^croeser]: Croeser, T., Weisser, W. W., Hurley, J., Rotzer, T., Parhizgar, L., Sun, Q. C., & Bekessy, S. A. (2025). Defining 'adequate' tree protection: Meeting urban canopy targets requires careful retention of mature trees. *Landscape and Urban Planning*, 264, 105484.
