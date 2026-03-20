# Definitions: Proposals and Interventions

## Proposal definitions

The manuscript defines five proposals:

- `Deploy Structure`
- `Decay`
- `Recruit`
- `Colonise`
- `Release Control`

## Intervention definitions

- `Buffer feature`: convert ground to a habitat island around a feature, allowing tree ageing, collapse, and pruning withdrawal.
- `Brace feature`: retain an ageing feature in place without converting surrounding ground to habitat island or allowing collapse, while pruning is moderated.
- `Rewild ground`: convert ground to a larger habitat patch by depaving and reducing management, allowing grasslands to form.
- `Adapt utility pole`: insert an artificial canopy feature at a utility pole.
- `Upgrade feature`: convert a canopy feature to a hybrid variant.
- `Enrich envelope`: convert rooftops to support plants, shrublands, and large translocated fallen trees.
- `Roughen envelope`: make rooftops or walls traversable with climbable elements, scattered rocks, and small logs.

## Proposal-intervention compatibility matrix

| Intervention | Decay | Release Control | Deploy Structure | Recruit | Colonise |
|---|---|---|---|---|---|
| `Buffer feature` | full | full |  | full |  |
| `Brace feature` | partial | partial |  |  |  |
| `Rewild ground` |  |  |  | full | full |
| `Adapt utility pole` |  |  | full |  |  |
| `Upgrade feature` |  |  | full |  |  |
| `Enrich envelope` |  |  |  |  | full |
| `Roughen envelope` |  |  |  |  | partial |

## Proposal-first view

`B = Bird`, `L = Lizard`, `T = Tree`

`Full` means the intervention fully supports the proposal.

`Partial` means the intervention accepts the proposal but constrains it.

### `Decay`

- `Buffer feature = full`
  - Converts ground to a habitat island around a feature, allowing tree ageing, collapse, and pruning withdrawal.
  - Indicators: `Acquire Resources (B, L, T)`, `Communicate (B, L, T)`, `Reproduce (B, L, T)`
- `Brace feature = partial`
  - Retains an ageing feature in place without converting surrounding ground to habitat island or allowing collapse, while pruning is moderated.
  - Indicators: `Acquire Resources (B, T)`, `Communicate (B)`, `Reproduce (B)`

### `Release Control`

- `Buffer feature = full`
  - Converts ground to a habitat island around a feature, allowing pruning withdrawal and lower-control canopy conditions.
  - Indicators: `Acquire Resources (B, L, T)`, `Communicate (B, L, T)`, `Reproduce (B, L, T)`
- `Brace feature = partial`
  - Retains the feature and moderates pruning without ground conversion or collapse.
  - Indicators: `Acquire Resources (B, T)`, `Communicate (B)`, `Reproduce (B)`

### `Deploy Structure`

- `Adapt utility pole = full`
  - Inserts an artificial canopy feature at a utility pole.
  - Indicators: `Acquire Resources (B)`, `Communicate (B)`, `Reproduce (B)`
- `Upgrade feature = full`
  - Converts a canopy feature to a hybrid variant, such as added bark, hollows, or epiphyte-supporting structure.
  - Indicators: `Acquire Resources (B)`, `Communicate (B)`, `Reproduce (B)`

### `Recruit`

- `Buffer feature = full`
  - Converts ground to a habitat island around a feature, creating local conditions for sapling recruitment near canopy.
  - Indicators: `Acquire Resources (B, L, T)`, `Communicate (B, L, T)`, `Reproduce (B, L, T)`
- `Rewild ground = full`
  - Converts ground to a larger habitat patch by depaving and reducing management, allowing grasslands to form.
  - Indicators: `Acquire Resources (L)`, `Communicate (L, T)`, `Reproduce (T)`

### `Colonise`

- `Rewild ground = full`
  - Converts ground to a larger habitat patch by depaving and reducing management, allowing grasslands to form.
  - Indicators: `Acquire Resources (L)`, `Communicate (L, T)`, `Reproduce (T)`
- `Enrich envelope = full`
  - Converts rooftops to support plants, shrublands, and large translocated fallen trees.
  - Indicators: `Acquire Resources (B, L)`, `Communicate (L)`, `Reproduce (L)`
- `Roughen envelope = partial`
  - Makes rooftops or walls traversable with climbable elements, scattered rocks, and small logs.
  - Indicators: `Communicate (L)`

## Intervention allocation and indicators

| Intervention | Decay | Release Control | Recruit | Deploy Structure | Colonise | Spatial compatibility | Resistance metric | Resulting action | Acquire Resources indicators | Communicate indicators | Reproduce indicators |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Buffer feature` | full | full | full |  |  | Under-canopy ground voxels around an ageing feature | Feature resistance | Convert ground to a habitat island around a feature; allow ageing, collapse, and pruning withdrawal | Peeling bark `(B)`; deadwood / epiphytes / ground cover `(L)`; senescent biovolume `(T)` | Perchable canopy `(B)`; non-paved ground `(L)`; soil near canopy `(T)` | Hollows `(B)`; nurse logs / fallen trees `(L)`; grassland for recruitment `(T)` |
| `Brace feature` | partial | partial |  |  |  | Large old or senescent feature where surrounding ground cannot be converted | Feature resistance | Retain ageing feature in place; no habitat island, no collapse; pruning moderated | Peeling bark `(B)`; senescent biovolume `(T)` | Perchable canopy `(B)` | Hollows `(B)` |
| `Rewild ground` |  |  | full |  | full | Ground voxels that can be depaved and managed as a larger patch | Connectivity resistance | Convert ground to a larger habitat patch by depaving and reducing management so grasslands can form | Ground cover / epiphytes `(L)` | Non-paved ground `(L)`; soil near canopy `(T)` | Grassland for recruitment `(T)` |
| `Adapt utility pole` |  |  |  | full |  | Utility pole voxels | Feature resistance | Insert an artificial canopy feature at a utility pole | Peeling bark / bark sleeves `(B)` | Perchable canopy `(B)` | Artificial hollows `(B)` |
| `Enrich envelope` |  |  |  |  | full | Rooftop voxels with structural capacity for vegetation and large timber loads | Connectivity resistance | Convert rooftops to support plants, shrublands, and large translocated fallen trees | Peeling bark on translocated deadwood `(B)`; ground cover / deadwood / epiphytes `(L)` | Non-paved traversable surface `(L)` | Nurse logs / fallen trees `(L)` |
| `Roughen envelope` |  |  |  |  | partial | Roof or wall voxels that can support traversable roughening but not full habitat loading | Connectivity resistance | Make rooftops or walls traversable with climbable elements, scattered rocks, and small logs |  | Non-paved traversable surface `(L)` |  |
