# THIS IS AN ARCHIVE

**Do not use anything in this folder as a source of truth.**

Files here are historical — superseded canonicals, abandoned experiments, legacy monoliths. They are kept only so old branches and scripts that reference them still open, and so that if anything here turns out to still be load-bearing we can move it back without losing history.

## Rules

- **Runners never load from `_archive/`.** If you find a `render_current_*.py` pointing into this folder, that runner is broken — fix it to use a canonical.
- **Do not edit files here.** If you need one, move it back to `canonical_templates/`, wire it to a runner, and add a row in the §2 family table of `COMPOSITOR_RUN-INSTRUCTIONS.md`. Otherwise leave it alone.
- **New experimental work does not go here.** Experimental blends belong in `_data-refactored/compositor/temp_blends/template_development/`. See §6 of `COMPOSITOR_RUN-INSTRUCTIONS.md`.

## What's here (and why)

| File | Superseded by | Reason archived |
|------|---------------|-----------------|
| `compositor_normals__remap_working_20260413.blend` | `compositor_normals.blend` | Intermediate working copy from the Apr 13 normals remap experiment; never promoted. |
| `compositor_proposal_masks.blend` | `proposal_*_layers.blend` family | Old mask-centric proposal template; replaced by the per-variant layer blends. |
| `compositor_sizes.blend` | `compositor_sizes_single_input.blend` | Old multi-input sizes template; simplified to single-input pipeline. |
| `edge_lab_final_template_safe_rebuild_20260405.blend` | per-family `compositor_*.blend` set | Legacy monolithic edge-lab template; split into one blend per family. |
| `shared_node_groups.blend` | node groups now live in each family's canonical | Old library-link source for shared groups; kept in case any blend still links it. Verify with link-check before deleting. |

If you delete something from here, grep the repo for its filename first — at minimum check `.blend` files that might link to it.
