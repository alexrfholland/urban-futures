# PyVista Render Settings

Named PyVista render-setting schemas for refactored scenario renderers.

Shared settings:

- `window_width`
- `window_height`
- `background`
- `lighting`
- `eye_dome_lighting`

Current named schemas:

- `engine3-proposals`
  - `point_size = 4.0`
  - `render_points_as_spheres = False`
  - default outputs:
    - `engine3-proposals_interventions_with-legend`
    - `engine3-proposals`
  - opt-in extra variants:
    - `engine3-proposals_interventions`
    - `engine3-proposals_with-legend`

Registry:

- [__init__.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/scenario/pyvista_render_settings/__init__.py)

Shared defaults:

- [shared.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/scenario/pyvista_render_settings/shared.py)

Named schemas:

- [engine3_proposals.py](/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/_futureSim_refactored/scenario/pyvista_render_settings/engine3_proposals.py)
