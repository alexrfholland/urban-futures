# 2.3-INFO-5 Graph Outputs

These scripts sit downstream of the assessment tables.

## 2.3-GRAPH-1 Stream Graphs

Script:

- `final/a_info_graphs.py`

Inputs:

- `data/revised/final/output/csv/{site}_{voxel_size}_indicator_counts.csv`

Description of processing:

- Reads the indicator counts, smooths the scenario trajectories against baseline, and writes stacked stream graphs for the indicator outcomes over time.

Outputs:

- `data/revised/final/output/plots/stream_graph_*.html`
- `data/revised/final/output/plots/stream_graph_*.png`

## 2.3-GRAPH-2 Performance Bubbles

Script:

- `final/SS/a_info_performance_bubbles.py`

Inputs:

- preferred `data/revised/final/output/csv/all_sites_{voxel_size}_indicator_counts.csv`
- fallback `data/revised/final/output/csv/{site}_{voxel_size}_indicator_counts.csv`

Description of processing:

- Reads the indicator counts, compares positive against trending through time, and writes bubble charts sized by positive performance and positioned by relative performance.

Outputs:

- `data/revised/final/output/plots/performance_bubbles_*.html`
- `data/revised/final/output/plots/performance_bubbles_*.png`

## 2.3-GRAPH-3 Companion Pathway Tracking Mini Streams

Script:

- `final/a_info_pathway_tracking_graphs.py`

Inputs:

- `_statistics-refactored/raw/{site}/interventions.csv`

Description of processing:

- Reads the raw intervention tables, selects a single canonical measure per intervention stream, and writes centered mini stream graphs for each proposal.
- Writes both `absolute` and `relative` variants.
- `absolute` uses raw intervention values.
- `relative` scales each proposal to its own peak total within the current scope.
- Writes both `combined` graphs and `per-proposal` graphs.
- Writes both `full-height` and `half-height` variants.
- The x axis uses the assessment years `0, 10, 30, 60, 90, 120, 150, 180`.

Outputs:

- `_statistics-refactored/plots/pathway_tracking/mini_streams/absolute/combined/full-height/{scope}.png`
- `_statistics-refactored/plots/pathway_tracking/mini_streams/absolute/combined/half-height/{scope}.png`
- `_statistics-refactored/plots/pathway_tracking/mini_streams/absolute/per-proposal/full-height/{proposal}_{scope}.png`
- `_statistics-refactored/plots/pathway_tracking/mini_streams/absolute/per-proposal/half-height/{proposal}_{scope}.png`
- `_statistics-refactored/plots/pathway_tracking/mini_streams/relative/combined/full-height/{scope}.png`
- `_statistics-refactored/plots/pathway_tracking/mini_streams/relative/combined/half-height/{scope}.png`
- `_statistics-refactored/plots/pathway_tracking/mini_streams/relative/per-proposal/full-height/{proposal}_{scope}.png`
- `_statistics-refactored/plots/pathway_tracking/mini_streams/relative/per-proposal/half-height/{proposal}_{scope}.png`

## 2.3-GRAPH-4 Legacy Capability Plots

Script:

- `final/SS/a_info_capability_plots.R`

Inputs:

- `*_capabilities_by_timestep.csv`

Description of processing:

- Reads an older wide capability table format and writes simple persona and capability line plots.

Outputs:

- `plots/*_all_capabilities.png`
- `plots/*_{persona}_capabilities.png`
