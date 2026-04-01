# Timeline Blender Scripts

This folder is the separate 2026 timeline workflow. It is intended to keep the timeline-specific pipeline away from the older non-timeline instancer.

## Run Order

1. `b2026_timeline_import_paraview_cameras.py`
   - Run if the ParaView cameras need to be created or refreshed.
2. `b2026_timeline_instancer.py`
   - Rebuilds the timeline tree/log instancer from the merged `0, 10, 30, 60, 180` CSV slices.
3. `b2026_timeline_bioenvelopes.py`
   - Rebuilds the timeline bioenvelopes from the per-year PLY files.
4. `b2026_timeline_debug_time_slices_material.py`
   - Optional debug pass. Colors trees and bioenvelopes by year slice.
5. `b2026_timeline_render_previews.py`
   - Optional preview renders for the timeline view layers.

## Supporting Scripts

- `b2026_timeline_layout.py`
  - Source of truth for strip boxes, translations, and refactored-data file resolution.
- `b2026_timeline_scene_contract.py`
  - Shared scene, collection, and view-layer naming.
- `b2026_timeline_clipbox_setup.py`
  - Optional clip-box helper setup.
- `b2026_timeline_camera_clipboxes.py`
  - Optional per-camera clip-box syncing.
- `b2026_timeline_runtime_flags.py`
  - Runtime switches for clip-box helpers.

## Notes

- `b2026_timeline_instancer.py` is the timeline-only variant and defaults `TIMELINE_MODE = True`.
- The older non-timeline workflow remains in [b2026_instancer.py](/d:/2026%20Arboreal%20Futures/urban-futures/final/_blender/2026/b2026_instancer.py).
- Clip-box helpers are off by default unless re-enabled in `b2026_timeline_runtime_flags.py`.
