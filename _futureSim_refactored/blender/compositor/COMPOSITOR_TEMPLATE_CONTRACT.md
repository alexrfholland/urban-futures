# Compositor Template Contract

## IMPORTANT PLEASE READ

- use a thin runner when the workflow already exists and only the EXR set or output path changes
- runners do not rebuild an existing canonical workflow
- runners open the canonical blend, repath inputs, set outputs, render, and exit
- if the workflow logic needs to change, edit the canonical blend instead

This document defines the contract between canonical compositor `.blend` files
and Python scripts in Edge Lab.

## Core Rule

- Canonical template blends own compositor graph logic.
- Scripts own execution around those templates.

In Edge Lab, the visual graph is not a secondary artifact. It is part of the
canonical authored system.

## What The Canonical Template Owns

Canonical template blends are the source of truth for:

- workflow structure
- node groups
- masks
- compositing logic
- output semantics
- file-output layout inside the compositor
- graph organisation and readability

If an output changes because the node graph changed, that change belongs in the
canonical template.

## What Scripts Own

Scripts should only own repeatable operational tasks around the canonical
template:

- pointing EXR input nodes at a chosen EXR set
- choosing which workflow family to render
- setting output root folders
- triggering renders
- normalising `_0001` filenames
- creating transient runtime-only compatibility nodes when Blender itself
  requires them for execution, as long as those nodes are not saved back into
  the canonical template
- orchestrating a multi-family run

Scripts should not become alternate owners of compositor logic.

For this project, "runner script" means:

- open an existing canonical blend
- repath EXR input nodes
- set output folders
- render
- optionally normalize `_0001` filenames
- exit without saving runtime changes back into the canonical blend

If a script is rebuilding a canonical graph from scratch during a normal render,
it is not acting as a runner and is violating the intended contract.

## Render Script Rule

Normal render scripts may:

- open a canonical template
- repath EXR nodes
- toggle file outputs or workflow selection at runtime
- create transient runtime-only nodes needed to execute the saved graph on the
  current Blender version
- render outputs

Normal render scripts must not:

- rewrite canonical graph structure
- add or remove nodes in the canonical template
- change node layout in the canonical template
- save graph changes back into the canonical template as part of normal render execution

Practical example:

- proposal colored depth outlines may rebuild a saved File Output node in
  memory and add a transient Composite sink during render execution because of
  current Blender 4.2 behavior
- those compatibility nodes are runtime scaffolding, not canonical graph edits

Short version:

- renderers use templates
- renderers do not redefine templates

Operational version:

- if the blend already exists canonically, use it
- do not recreate it from factory startup during routine rendering
- if a helper workflow is repeatable, give it a thin runner rather than a builder

## Template Edit Rule

Only explicit template-edit work may change a canonical template.

This includes:

- manual edits in Blender
- intentional migration scripts
- deliberate graph refactors

If a script exists to modify a canonical template, that must be its explicit
purpose.

If an agent is changing:

- masks
- node groups
- output sockets
- workflow layout
- semantic naming

that agent is doing template-edit work and should edit the canonical blend, not
hide the change in a runner.

## Working Copy Rule

If a task needs a saved modified blend for a specific dataset, run, or debug
session, create a derivative working copy.

Do not save those changes back into the canonical template unless the goal is to
promote them into the canonical graph.

## Request Decision Rule

When the request is:

- same workflow, different EXR set
  - rerender only
  - do not change the canonical template
  - prefer a thin runner or one-off render command

- new output, new mask logic, new workflow branch, or changed graph behavior
  - change the canonical template first
  - then rerender

- exploratory debugging
  - prefer a working copy or helper blend first
  - promote to canonical only if the workflow becomes standard

- one-off inspection
  - a direct headless Blender command is acceptable
  - do not turn every inspection into a durable repo script

## Mist And Depth Rule

Mist and depth outliner should follow the same contract as other workflows.

Long term:

- the canonical template should own the saved mist and depth graph logic
- runtime scripts should only repath EXRs and render

Legacy scratch-scene adapters may remain temporarily, but they should be treated
as transitional, not canonical.

## Practical Goal

The system should read like this:

- canonical blend = graph truth
- script = runtime operator
- working copy = safe derivative when needed

For agents, the default questions should be:

1. Does the canonical blend already contain the requested workflow?
2. If yes, can I just repath and render it?
3. If not, do I need to edit the canonical blend or prototype in a working copy first?

That is the ownership boundary to preserve during refactor.
