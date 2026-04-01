# Timeline Views

This note records the lightweight timeline camera views currently in use.

## Height Compare

Standardized height variants saved into the lightweight blends:

- `height_parade`
  - `z = 565.629272`
- `height_city_street`
  - `z = 871.929596`

These variants keep the current camera `x/y`, lens, and sensor-fit settings, then re-aim toward the same site focal point.

## Parade

Source blend:
- `D:\2026 Arboreal Futures\data\2026 futures parade lightweight cleaned.blend`

Current saved production view (`v3` framing):
- camera: `paraview_camera_parade`
- location: `(-484.211029, 141.647156, 565.629272)`
- rotation_euler: `(0.813206, -0.022163, -1.608964)`
- `angle_y`: `0.523599`
- `angle_x`: `0.764327`
- lens: `44.784611`
- sensor fit: `VERTICAL`

Raw ParaView reference:
- position: `(-710.5998866124906, 155.04839940955577, 780.0399301944501)`
- focal point: `(52.83318934160216, 109.85652084191102, 56.99999999999977)`
- view up: `(0.687286571444865, -0.010133360187298351, 0.7263156915025839)`
- view angle: `30`

## City

Source blend:
- `D:\2026 Arboreal Futures\data\2026 futures city lightweight cleaned.blend`

Current saved production view:
- camera: `paraview_camera_city`
- location: `(827.566101, 49.132877, 880.705994)`
- rotation_euler: `(0.78089, 0.061557, 1.640538)`
- `angle_y`: `0.491691`

Raw ParaView reference:
- position: `(827.5660735984109, 49.13287564742965, 880.7059951185498)`
- focal point: `(288.24903018994945, -22.32975691174787, 333.8400148551478)`
- view up: `(-0.7115782188004148, -0.006298457386287757, 0.702578656068758)`
- view angle: `28.171793383633197`

## Street

Source blend:
- `D:\2026 Arboreal Futures\data\2026 futures street lightweight cleaned.blend`

Current saved production view:
- camera: `paraview_camera_street`
- location: `(-76.761856, -879.492554, 863.153198)`
- rotation_euler: `(0.85655, -0.062729, -0.014206)`
- `angle_y`: `0.523599`

Raw ParaView reference used for street source (`uni`):
- position: `(-76.76185820395062, -879.4925282617797, 863.1531793423247)`
- focal point: `(-13.385258127015193, 44.270460390127, 63.18321407013855)`
- view up: `(-0.0380557896659148, 0.6556550948655048, 0.7541008907631722)`
- view angle: `30`
