# Parameters Discussion

Notes on outcomes from setting specific parameters. Not a full reference — just observations worth keeping.

---

## Release-control tree-support thresholds

Two parameters in `params_v3.py` control which living trees accept release control (and which get rejected):

- `maximum-tree-support-threshold` — `CanopyResistance < this` → `eliminate-canopy-pruning` (autonomous)
- `moderate-tree-support-threshold` — CR between max and this → `reduce-canopy-pruning` (moderated). CR ≥ this → rejected.

A third, `minimal-tree-support-threshold`, is **not** used by release control — only the decay pathway uses it.

Current values across scenarios:

| Site / Scenario | `maximum-` | `moderate-` | `minimal-` (decay only) |
|---|---|---|---|
| Parade positive | 10 | 50 | 70 |
| Parade trending | 0 | 1 | 2 |
| City positive | 10 | 50 | 70 |
| City trending | 0 | 1 | 2 |
| Uni positive | 10 | 50 | 70 |
| Uni trending | 0 | 1 | 2 |

All three positive scenarios use identical thresholds.

### Observation: uni rejection rate is template-driven, not threshold-driven

At yr180 in v4.8, living-tree CR distributions per positive-scenario site:

| Site | CR<10 | CR 10–50 | CR≥50 | % rejected |
|---|---|---|---|---|
| Parade positive | 319 | 126 | 79 | 15.1% |
| City positive | 72 | 167 | 75 | 23.9% |
| Uni positive | 61 | 122 | 58 | 24.1% |

Uni and city have nearly identical rejection rates (~24%). Parade has by far the lowest (15%) and by far the most CR<10 trees (319) — a much less hostile template. Uni's rejection rate is high because its input template has more trees in high-CR (hostile urban) locations, not because it uses looser thresholds.

`CanopyResistance` comes from `analysis_combined_resistance` in the site's input xarray dataset, sampled per tree when the treeDF is initialised. Trees in roadway / busy roadway / parking inherit high CR.

### Levers to reduce uni rejection

1. Raise `moderate-tree-support-threshold` (e.g. 50 → 70) — converts the ~43 trees in the CR 50–70 bucket from rejected into `reduce-canopy-pruning`. Quickest lever for immediate impact.
2. Raise `maximum-tree-support-threshold` (e.g. 10 → 20) — shifts trees from moderated into autonomous. Does not reduce rejection.
3. Modify `CanopyResistance` at source — edit the template's resistance field, or propagate depaving further to soften the urban matrix around high-CR trees.
