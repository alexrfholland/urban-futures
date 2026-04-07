# City / Street Lightweight Workflow

This folder holds the lightweight inspectable builds for the `city` scene and the parade-side scene used as `street`.

## Run Order

1. `b2026_city_street_lightweight_build.py`
   - Opens `D:\2026 Arboreal Futures\data\2026 futures heroes6.blend`
   - Writes a minimal `city` blend
   - Writes a minimal `street` blend, using the hero file's `parade` scene as the street-side source
2. `b2026_city_street_lightweight_render_previews.py`
   - Renders per-view-layer PNG previews from the lightweight blends

## Notes

- Test PNGs use transparent backgrounds and PNG `RGBA` output.
- The street-side source scene is `parade` because the hero blend does not contain a true `street` scene.
- Keep the minimal blends inspectable and scene-focused; do not touch the parade timeline workflow from the timeline folder.
