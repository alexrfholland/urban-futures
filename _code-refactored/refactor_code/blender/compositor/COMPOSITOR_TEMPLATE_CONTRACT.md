# Compositor Template Contract

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
- orchestrating a multi-family run

Scripts should not become alternate owners of compositor logic.

## Render Script Rule

Normal render scripts may:

- open a canonical template
- repath EXR nodes
- toggle file outputs or workflow selection at runtime
- render outputs

Normal render scripts must not:

- rewrite canonical graph structure
- add or remove nodes in the canonical template
- change node layout in the canonical template
- save graph changes back into the canonical template as part of normal render execution

Short version:

- renderers use templates
- renderers do not redefine templates

## Template Edit Rule

Only explicit template-edit work may change a canonical template.

This includes:

- manual edits in Blender
- intentional migration scripts
- deliberate graph refactors

If a script exists to modify a canonical template, that must be its explicit
purpose.

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

- new output, new mask logic, new workflow branch, or changed graph behavior
  - change the canonical template first
  - then rerender

- exploratory debugging
  - prefer a working copy or helper blend first
  - promote to canonical only if the workflow becomes standard

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

That is the ownership boundary to preserve during refactor.
