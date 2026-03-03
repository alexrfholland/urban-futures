# a_info_todo

## Proposal / Intervention Quantification Status

Implemented:
1. Decay quantities (opportunities, brace/buffer support) with site/scenario/year reporting.
2. Release Control quantities at VTK level:
- Proposal voxels: `search_bioavailable == 'arboreal'`
- Partial support: `forest_control == 'park-tree'`
- Full support: `forest_control in {'reserve-tree','improved-tree'}`
3. Connect quantities at VTK level from `scenario_outputs`:
- Proposal set: `brownRoof`, `greenRoof`, `livingFacade`, `footprint-depaved`, `node-rewilded`, `otherGround`, `rewilded`
- Full support: `greenRoof`, `node-rewilded`, `rewilded`
- Partial support: `brownRoof`, `footprint-depaved`, `livingFacade`
4. Totals outputs for capabilities by site/persona/pathway and pathway aggregates.

Remaining:
1. Recruit metric definition and implementation (currently stub).
2. Deploy metric definition and implementation (currently stub).
3. Translocate metric definition and implementation (currently stub).

## Engine / Simulation TODO

1. Convert the scenario engine to chain states between timesteps.
- Current runs are snapshot-style; each timestep should consume the previous timestep outputs as input state.

2. Generate states with improved snag models.
- Integrate improved snag/decay transition behavior into scenario generation and output states.

3. Decouple canopy control regime from tree lifecycle model.
- Separate management regime states from tree biological state.
- Support explicit control phases, e.g. aggressive pruning, moderate pruning, eliminate pruning.
