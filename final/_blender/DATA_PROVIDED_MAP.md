# Data Provided Map

This note records the data currently visible in `D:\2026 Arboreal Futures\data` and maps it to the locations the Blender project expects.

Date checked: `2026-03-29`

## What You Have Given Me

Visible top-level items in `D:\2026 Arboreal Futures\data`:

- `2026 futures heroes6.blend`
- `treeMeshes`
- `treeMeshesPly`
- `logMeshes`
- `logMeshesPLY`
- `tree_VTKpts`

Quick inventory:

- `2026 futures heroes6.blend`
  - size: about `10.36 GB`
- `treeMeshes`
  - `183` files
  - appears to be source tree VTK library
- `treeMeshesPly`
  - `184` files
  - appears to be exported tree/pole PLY library
  - includes at least one `Archive.zip`
- `logMeshes`
  - `18` files
  - appears to be source log VTK library
- `logMeshesPLY`
  - `36` files
  - appears to be exported log PLY library plus per-file resource-mapping JSON sidecars
- `tree_VTKpts`
  - `90` files
  - appears to be tree point-cloud VTK source data

## Expected Project Locations

Per [`AGENTS.md`](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/AGENTS.md), the Blender project currently expects these canonical locations:

- blend files:
  - `data/blender/2026/2026 futures heroes6.blend`
  - `data/blender/2026/2026 futures heroes6_baseline.blend`
- tree/pole PLY library:
  - `data/revised/final/treeMeshesPly`
- log PLY library:
  - `data/revised/final/logMeshesPly`

The repo at `D:\2026 Arboreal Futures\urban-futures` does not currently contain a `data/` directory, so these expected paths are not yet present inside the repo.

## Source To Target Mapping

Current source in `D:\2026 Arboreal Futures\data` mapped to expected project destination:

| Current source | What it appears to be | Expected destination |
| --- | --- | --- |
| `D:\2026 Arboreal Futures\data\2026 futures heroes6.blend` | main production heavy-scene blend | `D:\2026 Arboreal Futures\urban-futures\data\blender\2026\2026 futures heroes6.blend` |
| `D:\2026 Arboreal Futures\data\treeMeshesPly` | canonical tree and pole PLY instancer library | `D:\2026 Arboreal Futures\urban-futures\data\revised\final\treeMeshesPly` |
| `D:\2026 Arboreal Futures\data\logMeshesPLY` | canonical log PLY instancer library | `D:\2026 Arboreal Futures\urban-futures\data\revised\final\logMeshesPly` |
| `D:\2026 Arboreal Futures\data\treeMeshes` | likely source tree VTK mesh library used before PLY export | no canonical target documented yet in `AGENTS.md` |
| `D:\2026 Arboreal Futures\data\logMeshes` | likely source log VTK mesh library used before PLY export | no canonical target documented yet in `AGENTS.md` |
| `D:\2026 Arboreal Futures\data\tree_VTKpts` | likely source tree point-cloud VTK library | no canonical target documented yet in `AGENTS.md` |

## What Is Missing Relative To The Current Blender Contract

I can see some important inputs already, but not the full render contract yet. I do not currently see these items in the copied external data set:

- `2026 futures heroes6_baseline.blend`
- `data/revised/final/{site}/...` state folders
- per-state `nodeDF` CSVs
- per-state scenario VTKs
- baseline CSV / VTK products
- site buildings / road PLYs
- per-state envelope PLYs

## Practical Read

Right now, the assets you have definitely provided are enough to start wiring up:

- the main heavy city blend
- the tree/pole instancer PLY library
- the log instancer PLY library

The extra VTK folders look useful as source libraries, but they are not yet matched to an explicitly documented destination in the current Blender agent guide.
