# PSD template contract

How `base_template.psd`, the per-variant base PSBs, and the outer variant PSBs
fit together. Living doc — keep it in sync with the scripts in this folder.

See also:

- [COMPOSITOR_RUN.md](../blender/compositor/COMPOSITOR_RUN.md) — short reflex
  guide for running compositor families (which runner, env vars, where
  outputs land). Read this first before rendering source PNGs.
- [COMPOSITOR_RUN-INSTRUCTIONS.md](../blender/compositor/COMPOSITOR_RUN-INSTRUCTIONS.md)
  — long-form notes on the compositor pipeline.

---

## 1. Directory layout

```
_data-refactored/_psds/
├── assembled/                  # reference / archive PSDs & PSBs
│   ├── parade-base.psd         # gold reference for base_template structure
│   └── <site>_<variant>.psb    # historical assembled PSBs (before live pipeline)
│
├── psd-live/                   # the live pipeline — everything regenerates here
│   ├── base_template.psd       # canonical 5-layer base (see §2) — source of truth
│   ├── base_<variant>.psb      # per-variant base clones (SOs relinked to variant PNGs)
│   ├── template.psb            # outer variant template (links a base_<variant>.psb inside)
│   └── <site>_<variant>.psb    # per-variant outer PSBs cloned from template.psb
│
├── linked_pngs/<variant>/      # relative-path mirror of the PSB layer tree
│   └── <group>/<sub>/<layer>.png   # path relative to this root IS the layer path
│
└── sources/
    └── <site>.yaml             # per-site manifest consumed by copy_pngs.py
```

`linked_pngs/<variant>/` is the source of truth for the outer PSB's smart-object
relink pass: every `.png` under that root lines up with a PSB layer whose path is
the file's path minus the extension.

---

## 2. `base_template.psd` structure

Canvas: **7680 × 4320**, RGB 8-bit.

Layer stack (top → bottom):

| # | Name                                    | Kind              | Blend    | Visible | Source PNG                                |
|---|-----------------------------------------|-------------------|----------|---------|-------------------------------------------|
| 0 | `base_depth_windowed_balanced_dense`    | Smart object      | NORMAL   | ✓       | `base_depth_windowed_balanced_dense.png`  |
| 1 | `base_depth_windowed_internal_refined`  | Smart object      | NORMAL   | ✗       | `base_depth_windowed_internal_refined.png`|
| 2 | `Hue/Saturation 1`                      | Adjustment        | NORMAL   | ✓       | —                                         |
| 3 | `base_rgb`                              | Smart object      | COLOR    | ✓       | `base_rgb.png`                            |
| 4 | `existing_condition_ao_full`            | Smart object      | NORMAL   | ✓       | `base_white_render.png` (= masked AO)     |

Hue/Saturation settings (matches `assembled/parade-base.psd`):

- `colorize = false`
- master triplet: `hue=0, saturation=-73, lightness=0`

All four smart objects are linked (not embedded) so the tight opaque pixel
bbox isn't baked into the layer transform. The SO's file reference is the
bare PNG name; the source folder is wired at generation time.

---

## 3. Source PNGs

All four pixel layers come from the **`base`** compositor family — one
`compositor_base.blend` render produces the 10-PNG output set that includes
every file listed in §2. The `existing_condition_ao_full` layer sources from
`base_white_render.png` because the base blend's white-render reroute now
pipes the masked AO pass (see commit that fixed the base blend's AO wiring).

Contract: **do not mix families for the base**. Rerun `render_current_base.py`
for each variant against that variant's `existing_condition_positive_*.exr`.

Input EXR: `_data-refactored/blenderv2/output/<sim_root>/<variant>/<variant>__existing_condition_positive__8k.exr`
Output dir: `_data-refactored/compositor/outputs/<sim_root>/<variant>/base__<ts>/`

---

## 4. Tools

All scripts live in `_futureSim_refactored/photoshopparser/`.

### 4a. Generate `base_template.psd` from compositor PNGs

- **[generate_base_template.jsx](generate_base_template.jsx)** — ExtendScript.
  Builds the 5-layer base_template.psd from scratch: creates an 8K canvas,
  places each PNG as a linked smart object, adds the Hue/Sat adjustment, saves
  to `psd-live/base_template.psd`. Edit `SRC_DIR` in the script to point at
  the desired `base__<ts>/` output folder.

### 4b. Per-variant base PSB

- **[clone_psb.py](clone_psb.py)** — APFS copy-on-write clone of a template PSB
  under a new name. Used to spin up `base_<variant>.psb` from `base_template.psd`
  or a new variant outer PSB from `template.psb`. Pass `--source` to override
  the source file (default: `template.psb`).
- **[relink_base_psb.py](relink_base_psb.py)** — relinks the 4 SOs inside a
  base PSB to the latest `base__<ts>/` folder under
  `compositor/outputs/<sim_root>/<exr_family>/`. `--psb <path>` names the PSB
  to edit; `--exr-family <name>` names the compositor source folder. Kept
  independent so a test PSB can be pointed at another variant's base output
  (e.g. `base_uni_timeline_test.psb` edited against `uni_timeline/`'s base
  PNGs). `--sim-root` defaults to `4.12`.
- **[relink_psb.jsx](relink_psb.jsx)** — on the active outer PSB, runs three
  passes: (1) relinks the **three** `Base` smart objects (`BASE WORLD/Base`
  and the two under `Trending/.../proposal-release-control-*/Base`) from
  `base_template.psd` → `base_<variant>.psb` (path derived from the active
  doc name); (2) prunes timeline-only Hue/Sat adjustments for non-timeline
  variants; (3) relinks every other smart object to its matching PNG under
  `linked_pngs/<variant>/`. Pass 1 uses `placedLayerRelinkToFile`, which
  preserves clipping-mask flags — important because the two Trending Bases
  act as clip targets for the proposal SOs above them.

### 4c. Populate `linked_pngs/<variant>/`

- **[copy_pngs.py](copy_pngs.py)** — reads a site manifest + the PSB layer
  JSON and copies the right compositor PNG into the right spot under
  `linked_pngs/<site>/`. Supports `--only-group` and `--dry-run`.
  `--site <name>` and `--exr-family <name>` are **required** — the caller
  must state both explicitly. `--site` names the output subdir
  (`linked_pngs/<site>/`); `--exr-family` names the compositor input subdir
  (`compositor/outputs/<sim_root>/<exr_family>/`). For `parade_timeline` they
  differ (`parade` vs `parade_timeline`); for the other variants they match.
- **[read_psb.py](read_psb.py)** — dumps the PSB layer tree to JSON (used as
  input for copy_pngs.py and for diffing PSB structure).

### 4d. Propagate a smart object across variants

- **[propagate_so.jsx](propagate_so.jsx)** — add the same smart object + linked
  PNG to a set of already-open outer variant PSBs. Edit the `CONFIG` block at
  the top of the script: `so_name`, `group_name`, `anchor_name` (the sibling
  after which the new group should sit), `parents` (top-level parent groups
  to populate), and `targets` (list of `{doc, png_dir}`). Rerun is idempotent
  — existing linked SOs are skipped; stale plain layers or embedded SOs with
  the matching name are removed and re-placed as linked. See §9 for the
  process + placement rule.

### 4e. Inspect / export

- **[export_base_pixels.jsx](export_base_pixels.jsx)** — exports pixel layers
  from a base PSB as canvas-sized PNGs (for reusing legacy base pixels when
  regeneration isn't viable).
- **[export_layer_comps.jsx](export_layer_comps.jsx)** — on the active outer
  variant PSB, exports the `Positive` and `Trending` layer comps as flattened
  canvas-sized TIFFs to `_data-refactored/_psds/exports/<variant>_<comp>.tiff`.
  Duplicates the doc per comp, applies the comp, flattens, saves; source
  PSB is untouched.
- **[restructure_psb.jsx](restructure_psb.jsx)** — one-off structural edits
  (e.g. raster-mask → clipping-mask conversions). Idempotent, but hardcoded
  to specific target layers — read the script before running.

### 4g. Update the site manifest from a PSD

- **[update_yaml_from_psd.py](update_yaml_from_psd.py)** — add a routing rule to
  `sources/<site>.yaml` by pointing at a group or single SO inside a PSD. The
  script walks the panel-order PSD path, classifies the target (group →
  `key_folder` rule, smart object → `key_layer` rule) and appends a new entry
  to the matching section. **Additive only**: if the exact PSD path already
  exists under that section the tool logs the current rule and exits without
  rewriting — never overwrites. Preserves comments + unrelated formatting by
  editing the yaml text (not reserialising).

See §8 for the two yaml-update flows (chat-driven vs PSD-driven) and when to
use each.

### 4f. Runner

- **[run_jsx.py](run_jsx.py)** — wraps `osascript -l JavaScript` to run a
  .jsx file against Photoshop. `--psb` is optional (omit when the JSX creates
  its own document, e.g. `generate_base_template.jsx`).

```bash
# Generate a fresh base_template.psd from 4.12 parade_timeline base output:
uv run python -m _futureSim_refactored.photoshopparser.run_jsx \
  --jsx _futureSim_refactored/photoshopparser/generate_base_template.jsx

# Relink the active PSB's smart objects against linked_pngs/<variant>/:
uv run python -m _futureSim_refactored.photoshopparser.run_jsx \
  --psb _data-refactored/_psds/psd-live/parade_single-state_yr180.psb \
  --jsx _futureSim_refactored/photoshopparser/relink_psb.jsx
```

---

## 5. Variants

A **variant** is one rendered view of a site — e.g. a timeline, a single-state
year snapshot, or a baseline. The variant string is the unit that links
blenderv2 EXRs, compositor PNGs, `linked_pngs/`, and PSB filenames.

### Naming

```
<variant> = <site>_<mode>[_yr<N>][__<note>]
```

- `<site>` — one of: `city`, `parade`, `uni`
- `<mode>` — one of: `timeline`, `single-state`, `baseline`
- `yr<N>` — required when `<mode> = single-state`; absent otherwise
- `__<note>` — optional free-form suffix (double-underscore separator) used to
  tag a render with extra context like a camera or experiment name, e.g.
  `parade_baseline__hero-camera`. Treated as part of the variant string
  end-to-end (same folder names in blenderv2/compositor/linked_pngs/psd-live).

Examples: `parade_timeline`, `parade_single-state_yr180`, `city_baseline`,
`parade_baseline__hero-camera`.

### Where the variant shows up

The same string is used in every location — swapping variants is one global
find-and-replace:

| Location                                                                      | Example                                                            |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------|
| EXR inputs: `_data-refactored/blenderv2/output/<sim_root>/<variant>/`         | `blenderv2/output/4.12/parade_single-state_yr180/`                 |
| Compositor runs: `_data-refactored/compositor/outputs/<sim_root>/<variant>/`  | `compositor/outputs/4.12/parade_single-state_yr180/base__<ts>/`    |
| Linked PNG mirror: `_data-refactored/_psds/linked_pngs/<variant>/`            | `linked_pngs/parade_single-state_yr180/Trending/.../size_large.png`|
| Outer PSB: `_data-refactored/_psds/psd-live/<variant>.psb`                    | `psd-live/parade_single-state_yr180.psb`                           |
| Base PSB:  `_data-refactored/_psds/psd-live/base_<variant>.psb`               | `psd-live/base_parade_single-state_yr180.psb`                      |

### Site manifest vs variant

The site manifest (`sources/<site>.yaml`) encodes *site-level rules* — which
compositor family + branch feeds each PSB layer group. These rules don't
change between modes/years. The manifest intentionally does **not** hardcode
`site` or `exr_family` — the caller passes `--site` and `--exr-family` on the
copy_pngs.py CLI so intent is explicit at invocation time. To run the same
rules against a different variant, pass that variant's `--site` /
`--exr-family` pair (for `parade_timeline` these differ — `parade` vs
`parade_timeline` — for every other variant they're identical).

---

## 6. Regeneration flow (per variant)

Assumes the variant's EXRs exist at
`_data-refactored/blenderv2/output/<sim_root>/<variant>/`.

### Step 1 — compositor renders

Run each compositor family the PSB references against the variant's EXRs.
Outputs land at `compositor/outputs/<sim_root>/<variant>/<family>__<ts>[__<branch>]/`.
See [COMPOSITOR_RUN.md](../blender/compositor/COMPOSITOR_RUN.md) for per-family runners.

At minimum you need: `base`, `bioenvelope`, `shading`, `intervention_int`,
`mist_complex_outlines`, `sizes_single_input`, `size_outline`,
`proposal_and_interventions` (mirrors the `runs:` list in `parade.yaml`).

### Step 2 — refresh `base_template.psd` (only when the base graph or layer order changes)

Point `SRC_DIR` in [generate_base_template.jsx](generate_base_template.jsx) at a
`base__<ts>/` folder from any variant (it's just the reference structure),
then run it. This rewrites `psd-live/base_template.psd` with the canonical
5-layer stack (see §2). You don't need to do this per variant.

```bash
uv run python -m _futureSim_refactored.photoshopparser.run_jsx \
  --jsx _futureSim_refactored/photoshopparser/generate_base_template.jsx
```

### Step 3 — build `base_<variant>.psb`

```bash
# Clone base_template.psd -> base_<variant>.psb
uv run python -m _futureSim_refactored.photoshopparser.clone_psb \
  --name base_<variant> \
  --source _data-refactored/_psds/psd-live/base_template.psd

# Relink its 4 SOs to the compositor base__<ts>/ PNGs.
# --psb is the PSB to edit; --exr-family is the compositor source folder.
# For normal variants they're the same <variant>; for a test PSB pointed at
# another variant's output they differ.
uv run python -m _futureSim_refactored.photoshopparser.relink_base_psb \
  --psb         _data-refactored/_psds/psd-live/base_<variant>.psb \
  --exr-family  <variant>
```

### Step 4 — populate `linked_pngs/<variant>/`

```bash
# Dump the outer PSB's layer tree (needed by copy_pngs)
uv run python -m _futureSim_refactored.photoshopparser.read_psb \
  _data-refactored/_psds/psd-live/<variant>.psb \
  _data-refactored/_psds/psd-live/<variant>.layers.json

# Copy compositor PNGs into linked_pngs/<site>/ using parade.yaml's rules.
# For every variant other than parade_timeline, --site and --exr-family match
# (both are <variant>). parade_timeline is the exception: --site parade,
# --exr-family parade_timeline.
uv run python -m _futureSim_refactored.photoshopparser.copy_pngs \
  --manifest    _data-refactored/_psds/sources/parade.yaml \
  --layers      _data-refactored/_psds/psd-live/<variant>.layers.json \
  --out-root    _data-refactored/_psds/linked_pngs \
  --site        <variant> \
  --exr-family  <variant>
```

If the outer PSB doesn't exist yet, clone it from `template.psb` first
(see Step 5) so its layer tree matches the relink target.

### Step 5 — build / refresh the outer `<variant>.psb`

```bash
# New variant: clone template.psb -> <variant>.psb
uv run python -m _futureSim_refactored.photoshopparser.clone_psb \
  --name <variant>

# Relink: all 3 Base SOs -> base_<variant>.psb, all PNG SOs -> linked_pngs/<variant>/
uv run python -m _futureSim_refactored.photoshopparser.run_jsx \
  --psb _data-refactored/_psds/psd-live/<variant>.psb \
  --jsx _futureSim_refactored/photoshopparser/relink_psb.jsx
```

`relink_psb.jsx` is idempotent — rerun any time `linked_pngs/<variant>/`
refreshes. **Do not** re-clone the outer PSB if it already exists with
manual edits; just re-run the relink pass.

### Step 6 — export flattened layer comps

Flatten the `Positive` and `Trending` layer comps on the outer variant PSB
to canvas-sized (7680 × 4320) TIFFs:

```bash
uv run python -m _futureSim_refactored.photoshopparser.run_jsx \
  --psb _data-refactored/_psds/psd-live/<variant>.psb \
  --jsx _futureSim_refactored/photoshopparser/export_layer_comps.jsx
```

Outputs land at:

```text
_data-refactored/_psds/exports/
├── <variant>_Positive.tiff
└── <variant>_Trending.tiff
```

The script duplicates the doc per comp, applies the comp, flattens, saves;
the source PSB is left untouched. TIFFs use LZW compression, no layers, no
alpha. Reruns overwrite.

---

## 7. Outer variant PSB structure

The outer variant PSB (`template.psb` → cloned per variant) is the "full
scene" PSB. It's organised into top-level groups; each group wraps a state
(trending / positive / priority) or a structural role (BASE WORLD). Inside
each group, layers live in a mini-hierarchy that mirrors `linked_pngs/<variant>/`.

Current state (verified against `psd-live/template.psb` after the 2026-04
restructure):

```text
<variant>.psb  (outer)
├── BASE WORLD/
│   ├── base_depth_windowed_balanced_dense   ← linked SO (depth pass kept at outer level)
│   └── Base                                 ← linked SO → base_<variant>.psb
│
├── Trending/
│   └── arboreal/
│       └── trending/
│           ├── sizes/
│           │   └── Hue/Saturation - medium     ← timeline-only adjustment;
│           │                                      pruned for single-state variants
│           ├── proposal-release-control-rejected/
│           │   ├── Base                        ← linked SO → base_<variant>.psb
│           │   └── proposal-release-control-rejected   ← clip target (Base clips to it)
│           ├── proposal-release-control-reduce-canopy-pruning/
│           │   ├── Base                        ← linked SO → base_<variant>.psb
│           │   └── proposal-release-control-reduce-canopy-pruning  ← clip target
│           └── <per-layer SOs>                 ← each repoints to
│                                                  linked_pngs/<variant>/Trending/…/*.png
│
└── Positive/
    └── arboreal/
        └── positive/
            ├── sizes/
            │   └── Hue/Saturation - mediun     ← (typo preserved from source PSB)
            └── <per-layer SOs>
```

Rules:

1. **Every populated SO has a matching PNG under `linked_pngs/<variant>/`**.
   The layer path (groups joined by `/`) equals the PNG's path relative to
   `linked_pngs/<variant>/`, minus the `.png` extension. `relink_psb.jsx`
   walks the folder and relinks by that rule — embedded SOs are converted
   to linked in the process.
2. **The base appears as three linked SOs**, all named `Base`, all pointing
   at the same `psd-live/base_<variant>.psb`:
   - `BASE WORLD/Base` — the world-fill base.
   - `Trending/arboreal/trending/proposal-release-control-rejected/Base` —
     clipped to the proposal SO immediately below it (so base imagery only
     shows where the proposal exists).
   - `Trending/arboreal/trending/proposal-release-control-reduce-canopy-pruning/Base`
     — same clipping setup.
   `template.psb` ships these pointing at `base_template.psd`; `relink_psb.jsx`
   Pass 1 swaps them to `base_<variant>.psb` per variant. Use
   `placedLayerRelinkToFile` (not delete + re-place) so the clipping flag and
   stack position survive.
3. **Timeline-only Hue/Saturation adjustments** under
   `Trending/arboreal/trending/sizes/` and `Positive/arboreal/positive/sizes/`
   exist in the timeline templates; `relink_psb.jsx` prunes them when the doc
   name doesn't match `/timeline/i` (i.e. for single-state variants).
4. **`BASE WORLD/base_depth_windowed_balanced_dense`** is intentionally kept
   at the outer level (a linked SO whose source matches the PNG of the same
   name). The depth inside `base_<variant>.psb` is separate — both coexist.

---

## 8. Updating the site manifest

`sources/<site>.yaml` has two sections (see [copy_pngs.py](copy_pngs.py)):

- `key_folder` — group-path **prefix** rules. Every PNG whose PSD layer path
  starts with that prefix uses the rule (family + optional branch).
- `key_layer` — full-layer-path **exact** pins. Slot filename can be
  overridden. Exact match wins over longest-prefix walk.

Both sections are additive. Existing folder entries are never replaced — if
you need to *change* a rule, edit it by hand.

### 8a. `update_yaml_direct` — chat-driven

Use when you can describe the rule in prose. Tell Claude the PSD layer path,
what kind of target it is (group or single SO), and the family/branch/slot.
Claude edits the yaml in place with the same additive-merge rule the CLI
enforces: skip + report if the key already exists, insert otherwise. Preserve
section comments and ordering.

### 8b. `update_yaml_from_psd` — PSD-driven

Use when the PSD is the source of truth and you want the tool to classify the
target. The script opens the PSD, walks the panel-order path, classifies
(group → `key_folder`, SO → `key_layer`) and appends the rule.

```bash
# Group → key_folder (applies to every PNG under that prefix)
uv run python -m _futureSim_refactored.photoshopparser.update_yaml_from_psd \
  --psd       _data-refactored/_psds/psd-live/uni_timeline.psb \
  --psd-path  "Trending/arboreal/trending/sizes/new_subgroup" \
  --family    sizes_single_input \
  --branch    trending

# Single SO → key_layer (precise, overrides folder rule)
uv run python -m _futureSim_refactored.photoshopparser.update_yaml_from_psd \
  --psd       _data-refactored/_psds/psd-live/uni_timeline.psb \
  --psd-path  "Positive/outlines base/base_outlines" \
  --family    base \
  --slot      base_outlines.png
```

Flags:

- `--psd-path` — slash-joined panel-order path from the PSD root.
- `--family` — compositor family for the new rule (required).
- `--branch` — compositor branch (optional; folder rules typically need one).
- `--slot` — override slot filename; only valid for single-SO targets.
- `--yaml` — manifest to edit (default: `sources/parade.yaml`).
- `--dry-run` — print the would-be edit without writing.

Nested folders work — each segment in `--psd-path` is walked in turn. If the
exact PSD path already exists under the matching section the tool logs the
current rule and exits 0 without touching the file.

---

## 9. Propagating a smart object across variants

When a new SO needs to appear in every outer variant PSB (e.g. a new `base_*`
layer added to one site gets rolled out to all), follow the pattern below.
`propagate_so.jsx` (§4d) automates the PSB side; the PNG copy is a shell step.

### 9a. Process

1. **Trace the source SO**. In the source PSB (open in Photoshop), record:
   (a) the SO layer name, (b) its full group path, (c) which PNG filename it
   references, (d) the compositor family that produces that PNG.
2. **Per target variant**: resolve the variant's latest `<family>__<ts>/<png>`
   under `compositor/outputs/<sim_root>/<variant>/` and copy it to
   `linked_pngs/<site>/<group-path>/<png>`. Mirror the same group-path
   segments that will appear in the PSB.
3. **Per target PSB** (`propagate_so.jsx` does this): for each configured
   parent group that exists and contains the `anchor` sibling, ensure the
   subgroup exists at the right sibling position and Place Linked the PNG
   inside. Remove any stale plain layer / embedded SO with the same name
   first so the rerun upgrades it cleanly.
4. **Update toggle**. Overwriting the PNG on disk at its linked location is
   the "update" path — the SO reloads automatically. You only need to rerun
   the JSX when the PSB structure (group, SO, or order) changes, not when the
   underlying PNG content changes.

### 9b. Placement rule — do not repeat this mistake

**Sibling order is part of the contract.** When creating a new subgroup
inside an existing parent, mirror the source PSB's sibling position.
`outlines base` sits **after** `arboreal` (and before any `envelope*` group)
in every variant — *not* at the top of `Trending`/`Positive`. A first pass
placed the new group at the top of each parent because "top of parent" felt
like a safe default; it wasn't. The correct anchor is the existing sibling
it belongs next to in the source PSB. `propagate_so.jsx` takes `anchor_name`
for exactly this reason.

### 9c. Photoshop `PLACEAFTER` quirk

`layer.move(anchor, ElementPlacement.PLACEAFTER)` **pops the layer out of the
parent** when `anchor` is the last child of that parent. This is a
long-standing PS scripting quirk. Workaround used in `propagate_so.jsx`:

1. Create the new group at the document root (`doc.layerSets.add()`).
2. `newGroup.move(anchor, PLACEBEFORE)` — lands inside the parent just above
   `anchor`.
3. `anchor.move(newGroup, PLACEBEFORE)` — swaps the pair so the order
   becomes `[anchor, newGroup, <rest>]`.

Never use `PLACEAFTER` on a last-child anchor; always use the swap.

### 9d. Current propagations (2026-04)

- `base_outlines` (family: `base`) → `Trending/outlines base/base_outlines`
  and `Positive/outlines base/base_outlines`, linked, across all 6 outer
  variant PSBs (parade_timeline, uni_timeline, city_timeline,
  parade_single-state_yr180, city_single-state_yr180,
  parade_baseline__hero-camera — the last has no `Trending` group, so only
  the Positive slot is populated).

---

## 10. Open items

- Whether to keep `base_depth_windowed_internal_refined` hidden-by-default at
  the variant level or pull that visibility toggle up into the outer template.
- Embedded vs linked SO policy for `base_template.psd`: `assembled/parade-base.psd`
  had them embedded (baking trim bboxes); the current `generate_base_template.jsx`
  writes them linked to keep the 8K identity transform.
- The three Base SOs in `template.psb` were rebuilt 2026-04 with full-canvas
  identity transforms by deleting + re-Place-Linked against `base_template.psd`.
  If you ever need to do that again, **only use `placedLayerRelinkToFile`** to
  swap base sources after that point — delete-and-replace destroys the clipping
  mask on the two Trending Bases (have to manually Cmd+Opt+G to restore).
