# Scenario Engine V2 Model

This note describes how the v2 scenario engine works as a model, rather than as a validation log or change record.

For current status, verification, and open issues, see [scenario_engine_v2_status.md](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_documentation-refactored/scenario_engine_v2_status.md).

## Runtime Position

The v2 engine lives in [_code-refactored/refactor_code/scenario/engine_v2.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/engine_v2.py).

[final/a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py) is now a compatibility wrapper around that engine.

This model works through two linked calculation layers.

path-dependent habitat-feature calculations
These happen in the pulse on `tree_df`, and sometimes `log_df` and `pole_df`. They change which habitat features exist, how they are modified by proposal and intervention fields, and how these changes carry through time.

prebaked possibility-space calculations
These happen later on the possibility-space xarray. They classify where proposals and interventions can spread, connect, or be read spatially across voxels in the habitat volume.

The possibility-space xarray can be used in a pulse or state as a temporary layer for spatial calculations. It is reset per pulse/state. All information carried to the next state or pulse is recorded in the dataframes.

TODO: check if all the prebaked calculations have a `sim_Turns` component. consider unifiyign the parameters that are setting the effort thrsholds for the prebaked possibility space vs the parameters for the habitat-feature calculations

## Conventions

`[rename: x]` marks the future renaming we want to apply to an existing term. After first mention, this note then refers to the term using the renamed term.

V2 introduced parallel fields for some legacy fields in order to preserve compatibility with downstream code and outputs.

Where a legacy field and a v2 field carry the same underlying state in different vocabularies, this note uses the convention:

- legacy: `field_name`
- v2 of legacy `field_name`: `field_name` [rename: `term_name`]

In these pairs, the rename applies to the v2 field, not the legacy field.

Example:

- legacy: `action`
- v2 of legacy `action`: `lifecycle_decision` [rename: `proposal-decay_decision`]

## Initialised state fields

`control_realized` [rename: `control_reached`] defines the `control` part of the model lookup a tree feature uses.

This is initialised from `control` at the beginning of the model:

- `control = street-tree` -> `control_reached = street-tree`
- `control = park-tree` -> `control_reached = park-tree`
- otherwise -> `control_reached = low-control`

## Under-Canopy Regions

| under-canopy region | what voxels | calculations done in preflight |
| --- | --- | --- |
| `exoskeleton` | under-canopy voxels linked to the tree/node, but no larger connected region is used | `node_CanopyID`: links canopy voxels to a tree/node. `CanopyArea`: area of the canopy-linked voxel region. `CanopyResistance`: resistance value used to classify what level of under-canopy treatment the tree can support. |
| `footprint-depaved` | under-canopy voxels linked to the tree/node | This is the same canopy-linked area as `exoskeleton`, so it uses the same preflight fields: `node_CanopyID`, `CanopyArea`, and `CanopyResistance`. |
| `node-rewilded` | under-canopy context plus the larger simulated growth region linked to the node | `sim_Nodes`: links voxels to the larger simulated growth region of a node. `sim_NodesArea`: area of that simulated node region. `sim_Turns`: number of growth turns needed to reach each voxel in that region. `sim_averageResistance`: average no-canopy resistance of the early reachable part of the simulated node region. `CanopyResistance`: still used at tree level to assign the initial treatment band. |

TODO: update proposal-decay logic so fallen trees do not happen under decay

## Global parameters
Ramp starts define the start of the probability ramp for lifecycle transitions driven by remaining useful life:

- `senescingThreshold` [rename: `lifecycle_senescing_ramp_start`]
- `snagThreshold` [rename: `lifecycle_snag_ramp_start`]
- `collapsedThreshold` [rename: `lifecycle_fallen_ramp_start`]

`decayed` is not controlled by a probability ramp. It is triggered after a seeded persistence period.

Effort-thresholds are different. These control whether proposals are accepted or constrained.

| effort-threshold | Proposal-Decay | Proposal-Release-Control | Proposal-Colonise | Proposal-Recruit | Proposal-Deploy-Structure |
| --- | --- | --- | --- | --- | --- |
| `ageInPlaceThreshold` [rename: `minimal-tree-support-threshold`] | minimum effort needed to support a proposal-decay, and assigns the lowest-effort brace intervention: `exoskeleton` | not used | not used | not used | not used |
| `plantThreshold` [rename: `moderate-tree-support-threshold`] | middle accepted effort, assigns the buffer intervention: `footprint-depaved` | lower-effort threshold used to assign `reduce-pruning` | not used directly in current engine logic | not used directly in current engine logic | not used |
| `rewildThreshold` [rename: `maximum-tree-support-threshold`] | highest effort threshold, assigns the larger connected patch: `node-rewilded` | highest effort threshold used to assign `eliminate-pruning` | not used directly in current engine logic | not used directly in current engine logic | not used |
| `sim_averageResistance` | not used | not used | used together with `sim_TurnsThreshold` to build `bioMask`, which enables `otherGround`, `livingFacade`, `greenRoof`, and `brownRoof`; also used together with `sim_TurnsThreshold` to enable logs | not used directly in current engine logic | used on poles as the resistance threshold for enabling artificial structures |
| `sim_TurnsThreshold` | not used | not used | used together with `sim_averageResistance` to build `bioMask`, which enables `otherGround`, `livingFacade`, `greenRoof`, and `brownRoof`; also used together with `sim_averageResistance` to enable logs | used to keep only the earlier-reached portion of the simulated growth region | not used |

For logs, `assign_logs(...)` enables a log only if it passes both:

- `log_df["sim_averageResistance"] <= sim_averageResistance`
- `log_df["sim_Turns"] <= sim_TurnsThreshold`

Note: translocated fallen logs in `log_df` may fit better under Proposal-Deploy-Structure than under the current colonise/log threshold grouping.

## Stored but not used in current engine logic

- `lifecycle_state`

`lifecycle_state` is still referenced in [_code-refactored/refactor_code/scenario/validation.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_code-refactored/refactor_code/scenario/validation.py), so that validation path should derive it from `size` rather than treat it as an independent state field.

## Preflight

Preflight prepares:

- the possibility-space voxel array per site, stored in `subset_ds` [rename: `possibility-space_ds`]
- the initial feature-state tables

`possibility-space_ds` is prebaked and then reused.

Path dependencies live in the feature-state tables:

- `tree_df`
- optionally `log_df`
- optionally `pole_df`

Preflight also prepares the node ID systems that link feature rows back into the possibility space.

- `NodeID` links feature dataframes with the possibility-space xarray. It is a unique ID shared across `node_df[NodeID]` and `possibility-space_ds[analysis_nodeID]`, so the voxel containing the position of each feature is recorded in the possibility-space model. It is created in [a_create_resistance_grid.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_create_resistance_grid.py#L227) and copied into the dataframes in [a_create_resistance_grid.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_create_resistance_grid.py#L329).
- `possibility-space_ds[node_CanopyID]` assigns under-canopy voxels a node: what tree does this canopy voxel belong to? Creation is in [a_create_resistance_grid.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_create_resistance_grid.py#L369).
- `possibility-space_ds[sim_Nodes]` is the output of the growth simulation in [a_rewilding.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_rewilding.py#L365). In that simulation, every voxel with `analysis_nodeID != -1` starts a growth. Every voxel reached by a growth is labelled with its originating node ID. This allows assessment of the potential connected growth region from a node.
- `possibility-space_ds[sim_Turns]` records how many turns the growth simulation took to reach each voxel, as a proxy for how difficult that voxel was to reach through resistance. `sim_TurnsThreshold` then lets the model keep only the earlier-reached portion of that simulated growth region. This is used mainly in proposal-recruit. It is written in [a_rewilding.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_rewilding.py#L403).

## Per State

A state takes:

- the previous state's feature-state tables
- the fixed possibility-space voxel array `possibility-space_ds`

Almost all state logic happens inside pulses.

Outside the pulse loop, the main work is:

- year `0`: write seeded baseline state
- nonzero year: set up the pulse sequence
- after pulses: evaluate final-year logs and poles, then write outputs

### Per Pulse

A pulse splits a state into internal substeps of max year [default 30 years.]

A pulse takes:

- the previous pulse's `tree_df`
- the shared `possibility-space_ds`

Before proposals:

- age and grow living trees

#### Proposal-Decay

1. decides proposal-decay acceptance or rejection
2. applies proposal-decay accepted lifecycle changes
3. applies proposal-decay rejected changes
4. assigns proposal-decay interventions

- legacy: `action`
- v2 of legacy `action`: `lifecycle_decision` [rename: `proposal-decay_decision`]
- `rewilded` [rename: `under-node_treatment`]
- `decay_support` [rename: `decay_intervention`]

1. `determine_lifecycle_decisions(...)` [rename: `determine_proposal-decay(...)`]

   Determines proposal-decay_decision.

   Calculates `senesceChance` [rename: `proposal-decay_chance`] from:

   - `useful_life_expectancy`
   - `lifecycle_senescing_ramp_start`

   As useful life gets closer to `0`, proposal-decay_chance gets higher.

   - `useful_life_expectancy = lifecycle_senescing_ramp_start` -> `0%` chance
   - `useful_life_expectancy = 0` -> `100%` chance

   `senesceRoll` [rename: `proposal-decay_roll`] is compared against proposal-decay_chance.

   If proposal-decay_roll < proposal-decay_chance

   then the tree enters the proposal-decay branch.

   The model then determines community support for proposal-decay.

   If `CanopyResistance < minimal-tree-support-threshold`

   then:

   - `legacy: action = AGE-IN-PLACE`
   - `proposal-decay_decision = age-in-place` [rename: `proposal-decay_accepted`]

   Otherwise:

   - `legacy: action = REPLACE`
   - `proposal-decay_decision = replace` [rename: `proposal-decay_rejected`]

2. `apply_senescence_states(...)` [rename: `apply_proposal-decay_accepted_lifecycle_changes(...)`]

   When `proposal-decay_decision = proposal-decay_accepted`

   the engine writes:

   - living trees with `size in {small, medium, large}` -> `size = senescing`
   - if `size = senescing` and `snagRoll < snagChance` -> `size = snag`
   - if `size in {senescing, snag}` and `collapseRoll < collapseChance` -> `size = fallen`
   - `lifecycle_state` = the current decay class as an explicit mirror

   The probability ramps used here are:

   - `snagThreshold` [rename: `lifecycle_snag_ramp_start`]
   - `collapsedThreshold` [rename: `lifecycle_fallen_ramp_start`]

3. `handle_replace_trees(...)` [rename: `apply_proposal-decay_rejected_changes(...)`]

   When `proposal-decay_decision = proposal-decay_rejected`

   the model simulates replacing the aging tree with a sapling by

   - estimating replacement age from how far `useful_life_expectancy` has dropped below `0`, limited to the current pulse length
   - converting this into a trunk width
   - defining a replacement tree with:

   - `diameter_breast_height = 2 + growth_factor * replacement_growth_years`
   - `precolonial = True`
   - `useful_life_expectancy = 120 - replacement_growth_years`
   - `size` recalculated from `diameter_breast_height`
   - `lifecycle_state = standing`
   - `structureID` = new replacement identity

4. `assign_decay_support(...)` [rename: `assign_decay_interventions(...)`]

   When `proposal-decay_decision = proposal-decay_accepted`

   the model assigns a decay intervention by placing `CanopyResistance` into ordered effort bands:

   - highest effort: `(-inf, maximum-tree-support-threshold]` -> `under-node_treatment = node-rewilded`
   - middle effort: `(maximum-tree-support-threshold, moderate-tree-support-threshold]` -> `under-node_treatment = footprint-depaved`
   - lowest accepted effort: `(moderate-tree-support-threshold, minimal-tree-support-threshold]` -> `under-node_treatment = exoskeleton`
   - above available effort: `(minimal-tree-support-threshold, inf)` -> `rewilded = None`

   This `None` band exists in the code structure, but should not occur for accepted proposal-decay rows.

   Then:

   - `under-node_treatment = node-rewilded` -> `decay_intervention = buffer-feature`
   - `under-node_treatment = footprint-depaved` -> `decay_intervention = buffer-feature`
   - `under-node_treatment = exoskeleton` -> `decay_intervention = brace-feature`
   - `under-node_treatment = None` -> `decay_intervention = none`

   When these are later linked to a polydata of the state:

   - `exoskeleton` and `footprint-depaved` use `node_CanopyID == NodeID`, so they assign under-canopy voxels for that tree
   - `node-rewilded` uses `sim_Nodes == NodeID`, so it assigns the larger simulated connected patch for that tree

   NOTE
   `node-rewilded` is a spatial state that gets counted under interventions from multiple proposals:

   - `proposal-decay_buffer`
   - `proposal-recruit_buffer`
   - `proposal-colonise_rewild-ground`

#### Proposal-Release-Control

1. decides proposal-release-control acceptance or rejection
2. assigns proposal-release-control interventions
3. calculates changes resulting from release control
4. applies release control path dependent changes

- legacy: `control`
- v2 of legacy `control`: `control_realized` [rename: `control_reached`]
- `pruning_target` [rename: `proposal-release-control_intervention`]
- `pruning_target_years` [rename: `proposal-release-control_target_years`]
- `autonomy_years` [rename: `proposal-release-control_years`]
- `release_control_support` [rename: `proposal-release-control_support`]

`pruning_target_years` is kept as stored bookkeeping for how long the current target has been active, but it is not used to derive `control_reached`.

Use `control_reached` for the v2 field and `control` for the exported legacy lookup.

1. decide proposal-release-control acceptance or rejection in `apply_release_control(...)

   Add field `proposal-release-control-decision`, default `not-assessed`.

   For trees with `size in {small, medium, large}`:

   - if `CanopyResistance <= moderate-tree-support-threshold`
     - `proposal-release-control-decision = proposal-release-control_accepted`
   - otherwise
     - `proposal-release-control-decision = proposal-release-control_rejected`

2. assign proposal-release-control interventions in `apply_release_control(...)`

   For accepted trees, this step assigns proposal-release-control interventions through `pruning_target` [rename: `proposal-release-control_intervention`].

   The thresholds assign proposal-release-control interventions directly from `CanopyResistance`:

   - `(-inf, maximum-tree-support-threshold]` -> `proposal-release-control_intervention = eliminate-pruning`
   - `(maximum-tree-support-threshold, moderate-tree-support-threshold]` -> `proposal-release-control_intervention = reduce-pruning`

   TODO

   - should we differentiate the ground treatments for younger trees? We will have more trees having ground treatment than just decay. This means that there are more depaved and rewilded nodes than before, as all trees are assessed for this, not just trees undergoing decay
   - what happens if a larger rewild happens and the node is in it? Does this update?

   The model then assigns the related under-node treatment:

   - `proposal-release-control_intervention = eliminate-pruning` -> `under-node_treatment = node-rewilded`
   - `proposal-release-control_intervention = reduce-pruning` -> `under-node_treatment = footprint-depaved`

   If a tree already has a higher intervention, the model keeps the higher intervention rather than lowering it.

3. calculate changes resulting from release control in `apply_release_control(...)`

   The model then advances proposal-release-control through time using a counter that increases by `step_years`.

   If a tree moves from `reduce-pruning` to `eliminate-pruning`, the counter keeps going. It does not start again from zero.


   - under `reduce-pruning` or `eliminate-pruning`, a tree becomes `moderate-control` after 20 years
   - under `eliminate-pruning`, a tree becomes `low-control` after 40 years

   Trees with `size in {senescing, snag, fallen, decayed}` always have `control_reached = low-control`.


4. applies release control path dependent changes in `apply_release_control(...)`


   The model then converts `control_reached` into the model lookup `control`.

   The model exports the legacy `control` label from `control_reached`:

   - `street-tree` stays `street-tree`
   - `park-tree` stays `park-tree`
   - `low-control` becomes `reserve-tree`
   - senescing states are exported as `improved-tree`



#### Proposal-Colonise

1. decides proposal-colonise acceptance or rejection

- `colonise_support` [rename: `proposal-colonise-interventions`]

1. decide proposal-colonise acceptance or rejection in `refresh_colonise_support(...)`

   NOTE: this step only classifies under-canopy changes as colonise support. Most of Proposal-Colonise is calculated later in the prebaked possibility-space calculations.

   Add field `proposal-colonise-decision`, default `not-assessed`.

   The model resets:

   - `proposal-colonise-interventions = none`

   Then for rows where `under-node_treatment in {node-rewilded, footprint-depaved}`:

   - `proposal-colonise-interventions = rewild-ground`
   - `proposal-colonise-decision = proposal-colonise_accepted`

   Otherwise:

   - `proposal-colonise-decision = proposal-colonise_rejected`

#### Proposal-Recruit

Proposal-Recruit is a path-dependent habitat-feature calculation.

1. decides proposal-recruit acceptance or rejection

   Add field `proposal-recruit-decision`, but unlike the other proposals do not assign it in one first step. Assign it throughout the proposal-recruit logic, because proposal-recruit is determined through two recruit channels:

   - `buffer-feature` > recruits within tree-linked under-canopy regions defined by trees that previously met the `moderate-tree-support-threshold` or `maximum-tree-support-threshold`
   - `rewild-ground` > recruits within additional planting regions in the possibility space

2a. assigns `buffer-feature` proposal-recruit interventions and calculates proposal-recruit quantity

   Proposal-Recruit operates in 30-year recruitment cycles.

   For `buffer-feature`, the model recruits first around trees that already have eligible under-canopy treatment:

   - `under-node_treatment = footprint-depaved`
   - `under-node_treatment = node-rewilded`

   Assign these as:

   - `proposal-recruit-decision = proposal-recruit_accepted`
   - `proposal-recruit_intervention = buffer-feature`

   It then checks the area attached to that treatment:

   - if `under-node_treatment = footprint-depaved`
     - use `CanopyArea`
   - if `under-node_treatment = node-rewilded`
     - use `sim_NodesArea`

   Find the number of saplings recruited per 30-year recruitment cycle:

   - `plantingDensity * recruitable area`

   Find the number of saplings recruited this pulse by finding how many recruitment cycles are in this pulse:

   - `step_years / 30`

   For example:

   - `step_years = 30` -> full recruit amount
   - `step_years = 15` -> half recruit amount
   - `step_years = 60` -> double recruit amount

   The model then:

   - counts recruits already assigned to that source tree
   - subtracts those from the current recruit amount

2b. assigns `rewild-ground` proposal-recruit interventions and calculates proposal-recruit quantity

   For `rewild-ground`, this is also done in the pulse, but it uses planting voxels prepared just before recruit in the prebaked possibility-space calculations.

   To do so, `calculate_rewilded_status(...)` makes a deep copy of `possibility-space_ds` as a temporary spatial layer and writes the temporary fields:

   - `scenario_rewildingEnabled`
   - `scenario_rewildingPlantings`

   It assigns these temporary fields from the possibility space by:

   - keeping voxels where `sim_Turns <= sim_TurnsThreshold`
   - excluding `facade` and `roof` voxels

   This writes:

   - `scenario_rewildingEnabled`

   It then defines `scenario_rewildingPlantings` by also excluding voxels within 5 m of existing tree positions.

   Group voxels under consideration for additional `proposal-recruit` into planting regions:

   - first group by `sim_Nodes`
   - if a voxel has no `sim_Nodes`, fall back to `analysis_nodeID`
   - if it has neither, fall back to `node_CanopyID`

   Find the recruitable area of each planting region:

   - `number of voxels in the planting region * voxel area`

   Find the number of saplings recruited per 30-year recruitment cycle:

   - `plantingDensity * recruitable area`

   Find the number of saplings recruited this pulse by finding how many recruitment cycles are in this pulse:

   - `step_years / 30`

   The model then:

   - counts recruits already assigned to that planting region
   - subtracts those from the current recruit amount

3. applies proposal-recruit changes

   For `buffer-feature`, the model recruits features:

   - adds new tree features to the table near the source tree
   - offsets their position from the source tree by a small random XY shift within 2.5 m
   - on new trees from `buffer-feature`, assigns `recruit_intervention_type = buffer-feature` and `proposal-recruit-decision = proposal-recruit_accepted`
   - records the parent source as `recruit_source_id`

   For `rewild-ground`, the model recruits features:

   - adds new tree features at eligible planting voxel positions
   - on new trees from `rewild-ground`, assigns `recruit_intervention_type = rewild-ground` and `proposal-recruit-decision = proposal-recruit_accepted`
   - records the parent source as `recruit_source_id`

`recruit_source_id` stores the source tree or planting region that recruited this sapling, so the model can keep track of how much recruit has already been assigned there.

TODO

- consider how planting regions can become less full again, for example when recruited trees later become snags

## VTK Generation and Assignment

After state calculations are complete, the model assigns dataframe states to the temporary xarray possibility space and generates the state VTK.

### Proposal-Colonise

### Proposal-Recruit

TODO

For `proposal-recruit`:

- `proposal-recruit-decision`
  - `proposal-recruit_accepted` for voxels in planting regions eligible for recruit in this pulse
  - `proposal-recruit_rejected` for voxels in the broader enabled rewilding space that are not eligible planting voxels
- `proposal-recruit_intervention`
  - `rewild-ground` for voxels in planting regions that actually did recruit
  - `none` otherwise
- We also need to assign `proposal-recruit-decision` and `proposal-recruit_intervention` in the xarray / VTK for the tree-linked under-canopy recruit logic.

#### Proposal-Deploy-Structure

1. assigns proposal-deploy-structure acceptance or rejection and assigns proposal-deploy-structure interventions through enabling logs and poles in this state
2. updates voxels in xarray with proposals [done outside the state loop]

   In this state, `assign_logs(...)` enables logs if both:

   - `sim_averageResistance <= sim_averageResistance`
   - `sim_Turns <= sim_TurnsThreshold`

   `assign_poles(...)` enables poles if:

   - `sim_averageResistance < sim_averageResistance`

   Add to the `log_df` and `pole_df` dataframes:

   - `proposal-deploy-structure_decision`
   - `proposal-deploy-structure_intervention`

   For all enabled poles:

   - `proposal-deploy-structure_decision = proposal-deploy-structure_accepted`
   - `proposal-deploy-structure_intervention = proposal-deploy-structure_artificial_canopy`

   For all enabled logs:

   - `proposal-deploy-structure_decision = proposal-deploy-structure_accepted`
   - `proposal-deploy-structure_intervention = proposal-deploy-structure_translocated_log`

   NOTE: this proposal is simple as there is no rejected proposals.

TODO

- consider having fallen logs as a colonise intervention
