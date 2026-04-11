# _futureSim_refactored Agent Note

## Environment note

Most of the code in this area was written on a different machine.

That machine uses a repo-root Python virtual environment:

- `urban-futures/.venv`

So the important split is:

- the `.venv` controls the Python interpreter and installed packages
- the package/import root controls how repo modules are imported

These are not the same thing.

## How this works on the other machine

The expected setup on the other machine is:

1. activate the repo-root virtual environment:
   - `urban-futures/.venv`
2. run Python or Blender using that interpreter / environment
3. add [_futureSim_refactored](/d:/2026%20Arboreal%20Futures/urban-futures/_futureSim_refactored) to `sys.path`
4. import repo modules from `_futureSim_refactored.*`

So the repo-root `.venv` is compatible with the current code layout.

The `.venv` supplies:

- Python itself
- installed packages such as `numpy` and `pandas`

The current package layout supplies:

- repo module names such as `_futureSim_refactored.paths`
- repo module names such as `_futureSim_refactored.blenderv2.*`

So the root `.venv` does not conflict with the current import root; it just does a different job.

## Current import root

Right now the code under [_futureSim_refactored](/d:/2026%20Arboreal%20Futures/urban-futures/_futureSim_refactored) is still organized around:

- [_futureSim_refactored](/d:/2026%20Arboreal%20Futures/urban-futures/_futureSim_refactored)

That means the current practical import pattern is:

- add [_futureSim_refactored](/d:/2026%20Arboreal%20Futures/urban-futures/_futureSim_refactored) to `sys.path`
- then import modules as `_futureSim_refactored.*`

Examples:

- `_futureSim_refactored.paths`
- `_futureSim_refactored.scenario.engine_v2`
- `_futureSim_refactored.blenderv2.bV2_build_instancers`

## Important implication

Even if the active Python environment is the repo-root `.venv`, imports still currently assume:

- `_futureSim_refactored` is on `sys.path`
- `_futureSim_refactored` is the package root

So the `.venv` does **not** remove the need for the current `_futureSim_refactored` package layout.

## For agents

- do not assume the repo root itself is currently the package root for this code
- do not delete or flatten `_futureSim_refactored` casually; many imports and docs still assume it
- if you need these modules, prefer adding [_futureSim_refactored](/d:/2026%20Arboreal%20Futures/urban-futures/_futureSim_refactored) to `sys.path`
- if a future cleanup removes `_futureSim_refactored`, treat that as a deliberate repo-wide migration

## blenderv2

The current Blender v2 rewrite lives here:

- [blenderv2](/d:/2026%20Arboreal%20Futures/urban-futures/_futureSim_refactored/blenderv2)

Use its local note for current Blender v2 guidance:

- [AGENTS.md](/d:/2026%20Arboreal%20Futures/urban-futures/_futureSim_refactored/blenderv2/AGENTS.md)
