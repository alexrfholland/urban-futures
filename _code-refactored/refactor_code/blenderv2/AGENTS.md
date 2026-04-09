!# blenderv2 Agent Guide

## Scope

This folder is the active `blenderv2` rewrite area.

Use this as the current Blender v2 source of truth for code-side guidance.

Older references that may still be useful:

- older Blender `v1` note:
  - [_documentation-refactored/blender/AGENTS.md](/d:/2026%20Arboreal%20Futures/urban-futures/_documentation-refactored/blender/AGENTS.md)
- timeline `v1.5` note:
  - [final/_code-refactored/blender/timeline/AGENTS.md](/d:/2026%20Arboreal%20Futures/urban-futures/final/_code-refactored/blender/timeline/AGENTS.md)

## Main plan

Read this first:

- [bV2_temp_implementation-place.md](/d:/2026%20Arboreal%20Futures/urban-futures/_documentation-refactored/blenderv2/bV2_temp_implementation-place.md)

Schema rename note:

- [bV2_blender_schema_changes.md](/d:/2026%20Arboreal%20Futures/urban-futures/_documentation-refactored/blenderv2/bV2_blender_schema_changes.md)

## Current code

- [bV2_scene_contract.py](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored/refactor_code/blenderv2/bV2_scene_contract.py)
- [bV2_init_scene.py](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored/refactor_code/blenderv2/bV2_init_scene.py)
- [bV2_build_instancers.py](/d:/2026%20Arboreal%20Futures/urban-futures/_code-refactored/refactor_code/blenderv2/bV2_build_instancers.py)

Implemented now:

- scene contract
- scene initialization
- instancer build for `timeline` and `single_state`

Not yet implemented as `bV2_*`:

- bioenvelope build
- world rebuild
- render setup / EXR outputs
- validation

## Current template and outputs

Template assets:

- [bV2_template.blend](/d:/2026%20Arboreal%20Futures/urban-futures/_data-refactored/blenderv2/bV2_template.blend)
- [bV2_template-raw.blend](/d:/2026%20Arboreal%20Futures/urban-futures/_data-refactored/blenderv2/bV2_template-raw.blend)

Generated outputs and logs currently live on `E:`:

- blends:
  - [E:\\2026 Arboreal Futures\\blenderv2\\blends](/e:/2026%20Arboreal%20Futures/blenderv2/blends)
- logs:
  - [E:\\2026 Arboreal Futures\\blenderv2\\logs](/e:/2026%20Arboreal%20Futures/blenderv2/logs)

## Important implementation notes

- `uni` is the canonical site name; do not carry the old `street` alias forward in `blenderv2`
- `template = assets, script = structure`
- collection visibility uses explicit `bV2_role` tags, not truncated Blender collection names
- timeline dataframe assembly is done in pandas before Blender point-cloud creation
- `source-year` is the canonical provenance attribute / AOV
- instancer debug display can be forced with:
  - `BV2_INSTANCER_DISPLAY_MODE=source-year`
- live build logging can be enabled with:
  - `BV2_LOG_PATH=<path>`
- timeline strip spacing is controlled by:
  - `TIMELINE_OFFSET_STEP = 5.0`

## Known issues

- some generated blends do not reliably stay open through the normal GUI launch path on this machine
- when that happens, make a GUI-safe inspection copy by opening the blend with `load_ui=False` and resaving it
- verify that the Blender process is still alive after launch before claiming a GUI open succeeded
- workspace-pruning / saved-UI cleanup is disabled by default because it previously stalled `init_scene`

## Current build status

Built so far:

- timeline instancers for:
  - `city`
  - `trimmed-parade`
  - `uni`
- single-state instancers for `yr180` for:
  - `city`
  - `trimmed-parade`
  - `uni`

## Immediate next steps

1. implement `bV2_build_bioenvelopes.py`
2. implement `bV2_build_world_attributes.py`
3. implement render setup / EXR output scripts
4. implement `bV2_validate_scene.py`
