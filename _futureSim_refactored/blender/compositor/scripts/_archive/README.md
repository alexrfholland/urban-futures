# THIS IS AN ARCHIVE

**Do not use anything in this folder as a source of truth.**

These are superseded compositor runners and one-off scripts. The canonical runners are `render_current_<family>.py` in the parent `scripts/` directory — see the §2 family table in `COMPOSITOR_RUN-INSTRUCTIONS.md`.

## Rules

- **The batch driver never imports from `_archive/`.** If it does, that's a bug.
- **Do not edit files here.** If one is still useful, move it back to `scripts/`, update its docstring, and confirm it matches the current thin-runner contract (§7a of `COMPOSITOR_RUN-INSTRUCTIONS.md`).
- **New one-off scripts do not go here.** Put them in `scripts/` with a leading underscore and a date suffix (`_fix_<thing>_<YYYYMMDD>.py`); delete when the fix lands.

## What's here (and why)

`render_edge_lab_current_*.py` — the pre-split runners that rendered against the old monolithic `edge_lab_final_template_safe_rebuild` blend. Replaced by one thin `render_current_<family>.py` runner per canonical blend.
