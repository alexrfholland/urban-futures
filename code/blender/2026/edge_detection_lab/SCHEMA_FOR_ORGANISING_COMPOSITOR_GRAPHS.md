# Schema For Organising Compositor Graphs

This file defines the preferred layout and structuring rules for Edge Lab compositor graphs.

## Core Rules

- Organise the compositor into clear top-level workflows.
- Each workflow should be independently understandable when viewed on its own.
- Workflow bands should be laid out top-to-bottom, with node flow moving left-to-right.
- All workflow frames should align to the same left edge on the compositor canvas.
- Keep enough spacing to read the graph, but avoid large empty gaps.

## Workflow Structure

- Each workflow should have its own EXR input nodes.
- Do not make one workflow depend on another workflow's internal nodes.
- If multiple workflows need the same logic, convert that logic into a reusable node group instead of sharing direct connections.
- Each workflow should end in one workflow-level `File Output` node where practical.

## Shared Logic

- Repeatable units should be modularised as node groups.
- Shared reusable groups should use the naming convention:
  - `shared_[what-it-does]_group`
- Workflow-specific reusable groups should use the naming convention:
  - `[workflow-name]_[what-it-does]_group`
- Group names should describe behavior, not implementation detail.

Examples:

- `shared_visible-arboreal-mask_group`
- `shared_flat-colour-mask-rgba_group`
- `mist-outliner_kirsch-edge-alpha_thin_group`

## Naming Convention

- Do not rename generic processing nodes just to describe their meaning.
- Keep standard node types readable as standard node types where possible.
- Name key outputs, masks, and branch handoff points with labeled reroutes instead.
- Prefer a named reroute after an important node rather than renaming the important node itself.
- Use reroute labels to describe semantic meaning such as:
  - `mask_base-activated`
  - `bioenvelope_outlines-depth`
  - `mist_normalized_visible`

- Use node labels sparingly.
- Reserve explicit naming for:
  - workflow frames
  - workflow groups
  - shared groups
  - EXR input nodes
  - file output nodes
  - semantically important reroutes

## Groups Within Workflows

- Use groups within a workflow for repeated local operations.
- Prefer small, meaningful groups over long repeated node chains.
- A workflow may contain workflow-internal groups for masks, colour assignment, edge shaping, or output preparation.
- Workflow-internal groups should still keep left-to-right signal flow.

## Layout Rules

- Inputs on the left.
- Masking and preprocessing in the middle-left.
- Main transforms in the middle.
- Final colour assignment and output preparation in the middle-right.
- File outputs on the right.

- Avoid wires crossing over unrelated workflows.
- Avoid placing nodes on top of links.
- Avoid stacking unrelated chains directly on top of one another.
- Minimise oversized frames and empty frames.
- Shrink frames to contents where possible.

## Frames

- Use one top-level frame per workflow.
- Top-level workflow frames should be vertically stacked.
- Remove empty frames.
- Do not keep decorative or placeholder frames that no longer contain active nodes.
- Nested frames are acceptable only when they materially improve readability.

## EXR Inputs

- Repeated EXR image nodes are acceptable when they point to the same Blender image datablock.
- For readability, keep workflow-local EXR inputs inside the workflow that uses them.
- Do not wire one workflow's EXR node directly into another workflow.

## Output Safety

- Layout cleanup should not delete output nodes.
- Do not delete muted or seemingly dead nodes unless outputs have been audited before and after.
- After structural cleanup, confirm:
  - output node counts are still correct
  - no output node is left unlinked
  - workflow output paths remain intact

## Practical Goal

The compositor should read like a set of independent horizontal pipelines:

1. inputs
2. masks
3. processing
4. colour/output prep
5. file output

The graph should be easy to scan, easy to debug, and safe to modify without accidental cross-workflow breakage.
