# Proposal Alignment Between Code and Manuscript

This note tracks what is currently implemented in `a_info_proposal_interventions.py` and where that does or does not align with the manuscript proposal language.

## Scope

The script computes three output tables from existing scenario CSVs and urban-feature VTKs:

- `*_proposal_opportunities.csv`
- `*_proposal_interventions.csv`
- `*_proposal_qc.csv`

Support in the computed intervention table is recorded as either `full`, `partial`, or `stub`.

## Proposal Vocabulary

### Manuscript proposal set

The manuscript defines five proposals:

- `Deploy Structure`
- `Decay`
- `Recruit`
- `Colonise`
- `Release Control`

### Current code proposal set

The current script defines six proposal ids:

- `deploy`
- `decay`
- `recruit`
- `colonise`
- `release_control`
- `translocate`

## Proposal Status in the Current Script

| Proposal in manuscript/code language | Current code id | Status in current script | Note |
|---|---|---|---|
| `Decay` | `decay` | Computed | Opportunity and intervention support are both computed. |
| `Release Control` | `release_control` | Computed | Opportunity and intervention support are both computed. |
| `Colonise` | `colonise` | Computed | Computed in code, but the intervention family is labelled as `Connect`. |
| `Deploy Structure` / `Deploy` | `deploy` | Stub only | Present in manuscript and code vocabulary, but not yet mapped to computed intervention logic in this pass. |
| `Recruit` | `recruit` | Stub only | Present in manuscript and code vocabulary, but not yet mapped to computed intervention logic in this pass. |
| `Translocate` | `translocate` | Stub only | Present in code as a placeholder, but not part of the five-proposal set named in Section `6.2.1`. |

## Measured Intervention Support Currently Emitted

These are the intervention labels that the current script actually writes to `*_proposal_interventions.csv`.

| Proposal | Opportunity definition | Full support in current code | Partial support in current code | Output labels written | Notes |
|---|---|---|---|---|---|
| `Decay` | Existing trees where `isNewTree == False` and `action` is `AGE-IN-PLACE` or `SENESCENT` | `Buffer`: `tree_df.rewilded` in `node-rewilded` or `footprint-depaved`, and `vtk.scenario_rewilded` in `node-rewilded`, `footprint-depaved`, or `rewilded` | `Brace`: `tree_df.rewilded == exoskeleton` and `vtk.scenario_rewilded == exoskeleton` | `Buffer`, `Brace` | This now matches the current manuscript wording. |
| `Release Control` | Arboreal voxels where `vtk.search_bioavailable == arboreal` | `Eliminate pruning`: `vtk.forest_control` in `reserve-tree` or `improved-tree` | `Reduce pruning`: `vtk.forest_control == park-tree` | `Eliminate pruning`, `Reduce pruning` | This proposal is measured at voxel level, not tree level. |
| `Colonise` | Voxels where `vtk.scenario_outputs` is `brownRoof`, `greenRoof`, `livingFacade`, `footprint-depaved`, `node-rewilded`, `otherGround`, or `rewilded` | `Connect (full)`: `vtk.scenario_outputs` in `greenRoof`, `node-rewilded`, or `rewilded` | `Connect (partial)`: `vtk.scenario_outputs` in `brownRoof`, `footprint-depaved`, or `livingFacade` | `Connect (full)`, `Connect (partial)` | In the script this proposal is stored under `proposal_id = colonise`, but the intervention family is named `Connect`. |
| `Recruit` | Stub row only | Stub | Stub | `Stub` | Not yet mapped to computed intervention logic in this pass. |
| `Deploy` | Stub row only | Stub | Stub | `Stub` | Not yet mapped to computed intervention logic in this pass. |
| `Translocate` | Stub row only | Stub | Stub | `Stub` | Not yet mapped to computed intervention logic in this pass. |

## Additional Code-side Labels and Aliases

These names appear in code or output handling, but they are not the current manuscript intervention families.

### `Eliminate pruning`

- Present as a computed support label for `Release Control`.
- In the current manuscript matrix, pruning withdrawal is treated as part of `Buffer feature` rather than as its own top-level intervention family.

### `Reduce pruning`

- Present as a computed support label for `Release Control`.
- In the current manuscript matrix, pruning moderation is treated as part of `Brace feature` rather than as its own top-level intervention family.

### `Connect (green envelopes)`

- Present in the code alias table.
- Not emitted as a standalone support label in the current output tables.
- Belongs to the same family as `Connect (full)` and points to higher-support roof or ground envelope conditions.

### `Connect (brown envelopes)`

- Present in the code alias table.
- Not emitted as a standalone support label in the current output tables.
- Belongs to the same family as `Connect (partial)` and points to lower-support roof or facade envelope conditions.

### `Depave`

- Present in the code alias table.
- Not emitted as a standalone support label in the current output tables.
- In current outputs, depaving effects appear through scenario states such as `footprint-depaved`, and these are currently counted inside `Buffer` or `Connect (partial)` rather than as a separate intervention row.

### `Stub`

- Present in the current script as an internal placeholder support label.
- This is not a real intervention family. It marks proposals whose intervention mapping has not yet been implemented.

## Manuscript-to-Code Alignment Notes

### `Decay`: manuscript and code now align

The current manuscript and the current script now use the same Decay support mapping:

- `Buffer = full support`
- `Brace = partial support`

### `Deploy Structure` and `Recruit` are manuscript terms before they are measured terms

The current pathway explanations already use `Deploy`, `Recruit`, and related intervention language. That is fine as manuscript interpretation, but these are not yet computed support categories in `a_info_proposal_interventions.py`.

For now:

- use `Deploy Structure`, `Recruit`, and `Adapt` as manuscript-level language
- do not imply that they are already emitted as measured support labels in the proposal/intervention CSVs

### `Release Control` differs most strongly between the manuscript matrix and the current code

In the current manuscript matrix:

- `Buffer feature` provides full support for `Release Control`
- `Brace feature` provides partial support for `Release Control`

In the current script:

- `Eliminate pruning` is the full support label for `Release Control`
- `Reduce pruning` is the partial support label for `Release Control`

This means the manuscript now treats pruning change as part of broader intervention families, while the current script still measures it as standalone support labels.

### `Colonise` and envelope interventions should be kept distinct

The manuscript frames `Colonise` as the proposal, while `Rewild ground`, `Enrich envelope`, and `Roughen envelope` describe the intervention responses.

The current script measures only part of this family directly, under the `Connect` labels.

- `Colonise` = what the persona proposes
- `Enrich envelope` and `Roughen envelope` = the current manuscript intervention family for building envelopes
- `Connect (full)` and `Connect (partial)` = the current measured support states in code

The pathway explanations should preserve that distinction.
