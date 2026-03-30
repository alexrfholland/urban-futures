# AGENTS

## TODO

### SIMULATION

#### Ticket 1. Year 0 Scenario Behaviour

High amount of release control in trending because `reduce_control_of_trees(...)` in [a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py#L204) assigns trees in positive to `street-tree`.

Specifics:

Because year `0` is already a scenario run, not a shared untouched baseline.

Starting `trimmed-parade` tree controls in the base [trimmed-parade_1_treeDF.csv](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/trimmed-parade/trimmed-parade_1_treeDF.csv):

- `park-tree`: `315`
- `street-tree`: `141`

After the year-0 run:

- `positive`: `454 street-tree`, `399 reserve-tree`, `2 park-tree`
- `trending`: `298 park-tree`, `158 street-tree`

Why this happens:

- both scenarios start from the same base treeDF via [a_scenario_initialiseDS.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_initialiseDS.py#L325)
- then year `0` still runs `run_scenario(...)` via [a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py#L648)
- in `reduce_control_of_trees(...)` via [a_scenario_runscenario.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_runscenario.py#L204), any non-senescent tree that falls under the rewild/depave mask gets its control reassigned from `unmanagedCount`
- at year `0`, `unmanagedCount` is `0`, so that reassignment lands in `street-tree`

Why positive gets hit much harder:

- `positive` thresholds are broad in [a_scenario_params.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/a_scenario_params.py#L148):
- `rewildThreshold = 10`
- `plantThreshold = 50`
- `trending` thresholds are much tighter:
- `rewildThreshold = 0`
- `plantThreshold = 1`

So in `positive`, many more trees enter that mask and get reassigned to `street-tree` at year `0`.
In `trending`, far fewer do, so most original `park-tree` canopy stays `park-tree`.

That is why:

- `trending` has much more `Release-Control -> Brace-Feature`
- `positive` has much more `street-tree`
- year `0` is not a neutral shared baseline for this metric

#### Ticket 2. Rename Release-Control Buffer / Brace

Rename `Release-Control -> Buffer-Feature` and `Release-Control -> Brace-Feature` to:

- `Release-Control -> Eliminate-Pruning`
- `Release-Control -> Reduce-Pruning`
